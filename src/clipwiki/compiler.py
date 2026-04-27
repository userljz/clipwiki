"""Deterministic compiler for the ClipWiki baseline."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from clipwiki.markdown_store import MarkdownPage, MarkdownStore
from clipwiki.schemas import PreparedExample, SessionTurn
from clipwiki.tokens import content_tokens, estimate_text_tokens

NAME_PATTERN = re.compile(r"\[([A-Z][A-Z\s'\-]+)\]")
BOLD_SPEAKER_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?\*\*([^*\n]+)\*\*:\s*", re.MULTILINE)
PLAIN_SPEAKER_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z][A-Za-z\s'\-]{0,40}):\s+", re.MULTILINE)
DATE_PATTERN = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b")
PREFERENCE_KEYWORDS = ("favorite", "prefer", "likes", "like", "love", "enjoy", "wants", "plan", "plans")
METADATA_PREFIXES = ("Question:", "Tags:", "Concept:", "Why saved:", "Summary:", "Source:", "Evidence:")
SPEAKER_METADATA_LABELS = {
    "Question",
    "Tags",
    "Concept",
    "Why Saved",
    "Summary",
    "Source",
    "Evidence",
    "Sources",
    "Mode",
    "Date",
    "Recorded On",
    "Session Id",
    "Clipwiki Index",
    "Clipwiki Log",
}
PAGE_TYPE_BIASES = {
    "evidence": 2.0,
    "source": 1.5,
    "preference": 1.25,
    "event": 0.15,
    "person": 0.1,
    "concept": -1.0,
    "index": -1.25,
    "log": -1.5,
}
ORACLE_MODES = {"oracle-curated"}
NON_ORACLE_MODES = {"full-wiki", "curated", "noisy-curated"}
GOLD_LABEL_FIELDS = ("gold_evidence",)


@dataclass(slots=True)
class CompiledWiki:
    """Compiled wiki pages and size metadata."""

    pages: list[MarkdownPage]
    wiki_size_pages: int
    wiki_size_tokens: int
    selected_session_ids: list[str]


def compile_clipwiki(
    example: PreparedExample,
    output_dir: Path,
    mode: str,
    curated_top_k: int = 3,
    noisy_extra_sessions: int = 2,
) -> CompiledWiki:
    """Compile a deterministic markdown wiki for a single example."""

    selected_indices = select_session_indices(
        example,
        mode=mode,
        curated_top_k=curated_top_k,
        noisy_extra_sessions=noisy_extra_sessions,
    )
    selected_session_ids = [example.haystack_session_ids[index] for index in selected_indices]

    store = MarkdownStore(output_dir)
    for directory_name in ("sources", "evidence", "concepts", "people", "events", "preferences"):
        (output_dir / directory_name).mkdir(parents=True, exist_ok=True)
    pages: list[MarkdownPage] = []
    source_page_ids: list[str] = []
    evidence_page_ids: list[str] = []
    speaker_sources: dict[str, list[str]] = {}
    speaker_source_ids: dict[str, list[str]] = {}
    preference_pages: list[MarkdownPage] = []

    for index in selected_indices:
        session_id = example.haystack_session_ids[index]
        session_datetime = example.haystack_session_datetimes[index]
        summary = example.haystack_session_summaries[index] if index < len(example.haystack_session_summaries) else ""
        session_turns = example.haystack_sessions[index]
        transcript = session_to_markdown(session_turns)
        source_page_id = f"sources/{session_id}"
        evidence_page_id = f"evidence/{session_id}"
        source_page_ids.append(source_page_id)
        evidence_page_ids.append(evidence_page_id)

        evidence_lines = _session_evidence_lines(
            example,
            session_id=session_id,
            session_turns=session_turns,
            summary=summary,
            mode=mode,
        )

        source_page = MarkdownPage(
            page_id=source_page_id,
            relative_path=f"sources/{session_id}.md",
            title=f"Source {session_id}",
            source_ids=[session_id],
            page_type="source",
            is_answerable=True,
            content="\n".join(
                [
                    f"# Source {session_id}",
                    "",
                    f"- Recorded on: `{session_datetime.isoformat()}`",
                    f"- Session ID: `{session_id}`",
                    "",
                    "## Evidence Snippets",
                    *([f"- {fact}" for fact in evidence_lines] if evidence_lines else ["- No deterministic evidence snippets extracted."]),
                    "",
                    "## Transcript",
                    transcript,
                    "",
                ]
            ),
            search_text=_page_search_text([summary, *evidence_lines]),
            timestamp=session_datetime,
        )
        pages.append(source_page)
        store.write_page(source_page)

        evidence_page = MarkdownPage(
            page_id=evidence_page_id,
            relative_path=f"evidence/{session_id}.md",
            title=f"Evidence from {session_id}",
            source_ids=[session_id],
            page_type="evidence",
            is_answerable=True,
            content="\n".join(
                [
                    f"# Evidence from {session_id}",
                    "",
                    *([f"- {fact}" for fact in evidence_lines] if evidence_lines else ["- No deterministic evidence snippets extracted."]),
                    "",
                    "## Supports",
                    f"- [[{source_page_id}]]",
                    "",
                ]
            ),
            search_text=_page_search_text(evidence_lines),
            timestamp=session_datetime,
        )
        pages.append(evidence_page)
        store.write_page(evidence_page)

        for speaker_name in extract_speaker_names(transcript):
            speaker_sources.setdefault(speaker_name, []).append(source_page_id)
            speaker_source_ids.setdefault(speaker_name, []).append(session_id)

        preference_lines = [line for line in evidence_lines if any(keyword in line.lower() for keyword in PREFERENCE_KEYWORDS)]
        if preference_lines:
            preference_page = MarkdownPage(
                page_id=f"preferences/{session_id}",
                relative_path=f"preferences/{session_id}.md",
                title=f"Preferences from {session_id}",
                source_ids=[session_id],
                page_type="preference",
                is_answerable=True,
                content="\n".join(
                    [
                        f"# Preferences from {session_id}",
                        "",
                        *[f"- {line}" for line in preference_lines],
                        "",
                        "## Supports",
                        f"- [[{source_page_id}]]",
                        f"- [[{evidence_page_id}]]",
                        "",
                    ]
                ),
                search_text=_page_search_text(preference_lines),
                timestamp=session_datetime,
            )
            preference_pages.append(preference_page)
            pages.append(preference_page)
            store.write_page(preference_page)

        event_page = MarkdownPage(
            page_id=f"events/{session_id}",
            relative_path=f"events/{session_id}.md",
            title=f"Event {session_id}",
            source_ids=[session_id],
            page_type="event",
            is_answerable=False,
            content="\n".join(
                [
                    f"# Event {session_id}",
                    "",
                    f"- Recorded on: `{session_datetime.date().isoformat()}`",
                    f"- Session summary: {summary or 'No summary available.'}",
                    "",
                    "## Related Pages",
                    f"- [[{source_page_id}]]",
                    f"- [[{evidence_page_id}]]",
                    "",
                ]
            ),
            search_text=_page_search_text([summary]),
            timestamp=session_datetime,
        )
        pages.append(event_page)
        store.write_page(event_page)

    for speaker_name, linked_sources in sorted(speaker_sources.items()):
        slug = slugify(speaker_name)
        people_page = MarkdownPage(
            page_id=f"people/{slug}",
            relative_path=f"people/{slug}.md",
            title=speaker_name,
            source_ids=sorted(set(speaker_source_ids.get(speaker_name, []))),
            page_type="person",
            is_answerable=False,
            content="\n".join(
                [
                    f"# {speaker_name}",
                    "",
                    "## Source Pages",
                    *[f"- [[{source_id}]]" for source_id in sorted(set(linked_sources))],
                    "",
                ]
            ),
            search_text=_page_search_text([speaker_name]),
        )
        pages.append(people_page)
        store.write_page(people_page)

    concept_page = MarkdownPage(
        page_id=f"concepts/{example.question_type}",
        relative_path=f"concepts/{slugify(example.question_type)}.md",
        title=f"Question type: {example.question_type}",
        source_ids=selected_session_ids,
        page_type="concept",
        is_answerable=False,
        content="\n".join(
            [
                f"# Concept: {example.question_type}",
                "",
                f"- Topic label: `{example.question_type}`",
                "- Use linked source and evidence pages for answers.",
                "",
                "## Relevant Evidence Pages",
                *[f"- [[{page_id}]]" for page_id in evidence_page_ids],
                "",
                "## Relevant Sources",
                *[f"- [[{source_id}]]" for source_id in source_page_ids],
                "",
            ]
        ),
        search_text=_page_search_text([example.question_type or "", *selected_session_ids]),
    )
    pages.append(concept_page)
    store.write_page(concept_page)

    index_page = MarkdownPage(
        page_id="index",
        relative_path="index.md",
        title="Index",
        source_ids=selected_session_ids,
        page_type="index",
        is_answerable=False,
        content="\n".join(
            [
                "# ClipWiki Index",
                "",
                f"- Mode: `{mode}`",
                f"- Selected sessions: {', '.join(selected_session_ids)}",
                "",
                "## Evidence Pages",
                *[f"- [[evidence/{example.haystack_session_ids[index]}]]" for index in selected_indices],
                "",
                "## Source Pages",
                *[
                    f"- [[sources/{example.haystack_session_ids[index]}]]: "
                    f"{example.haystack_session_summaries[index] if index < len(example.haystack_session_summaries) else 'No summary available.'}"
                    for index in selected_indices
                ],
                "",
            ]
        ),
        search_text=_page_search_text([mode, *selected_session_ids]),
    )
    pages.append(index_page)
    store.write_page(index_page)

    log_page = MarkdownPage(
        page_id="log",
        relative_path="log.md",
        title="Log",
        source_ids=selected_session_ids,
        page_type="log",
        is_answerable=False,
        content="\n".join(
            [
                "# ClipWiki Log",
                "",
                f"- Example: `{example.example_id}`",
                f"- Question type: `{example.question_type}`",
                f"- Mode: `{mode}`",
                f"- Selected sessions: `{', '.join(selected_session_ids)}`",
                "",
            ]
        ),
        search_text=_page_search_text([example.question_type or "", mode, *selected_session_ids]),
    )
    pages.append(log_page)
    store.write_page(log_page)

    wiki_size_tokens = sum(estimate_text_tokens(page.content) for page in pages)
    return CompiledWiki(
        pages=pages,
        wiki_size_pages=len(pages),
        wiki_size_tokens=wiki_size_tokens,
        selected_session_ids=selected_session_ids,
    )


def select_session_indices(
    example: PreparedExample,
    mode: str,
    curated_top_k: int = 3,
    noisy_extra_sessions: int = 2,
) -> list[int]:
    """Choose which sessions to compile for a given ClipWiki mode."""

    total_sessions = len(example.haystack_session_ids)
    all_indices = list(range(total_sessions))

    if mode == "full-wiki":
        return all_indices

    if mode == "curated":
        curated_from_clips = _curated_session_indices_from_metadata(example)
        if curated_from_clips:
            return curated_from_clips
        return _heuristic_session_indices(example, top_k=curated_top_k)

    if mode == "oracle-curated":
        gold_indices = _gold_session_indices(example, top_k=curated_top_k)
        return gold_indices or _heuristic_session_indices(example, top_k=curated_top_k)

    if mode == "noisy-curated":
        curated = _heuristic_session_indices(example, top_k=curated_top_k)
        remaining = [index for index in all_indices if index not in curated]
        rng = random.Random(example.question_id or example.example_id)
        noisy_count = min(noisy_extra_sessions, len(remaining))
        noisy_indices = sorted(rng.sample(remaining, noisy_count)) if noisy_count > 0 else []
        return sorted(set(curated + noisy_indices))

    raise ValueError(f"Unsupported ClipWiki mode: {mode}")


def retrieve_wiki_pages(question: str, wiki: CompiledWiki, top_k: int = 4) -> list[MarkdownPage]:
    """Retrieve wiki pages with BM25-style lexical scoring."""

    documents = wiki.pages
    if not documents:
        return []

    scores = _page_retrieval_scores(question, wiki)
    ranked = sorted(
        ((page, scores.get(page.page_id, 0.0)) for page in documents),
        key=lambda item: (-item[1], item[0].page_id),
    )
    return [page for page, _ in ranked[:top_k]]


def page_score_lookup(question: str, wiki: CompiledWiki) -> dict[str, float]:
    """Return BM25 scores keyed by page id for all compiled pages."""

    return _page_retrieval_scores(question, wiki)


def session_to_markdown(session_turns: list[SessionTurn]) -> str:
    """Convert session turns into a markdown transcript block."""

    return "\n".join(f"- **{turn.role}**: {turn.content}" for turn in session_turns)


def extract_speaker_names(text: str) -> list[str]:
    """Extract likely speaker names from transcript text."""

    names = set()
    for pattern in (NAME_PATTERN, BOLD_SPEAKER_PATTERN, PLAIN_SPEAKER_PATTERN):
        for match in pattern.finditer(text):
            name = normalize_name(match.group(1))
            if _is_valid_speaker_name(name):
                names.add(name)
    return sorted(name for name in names if name)


def extract_fact_candidates(text: str) -> list[str]:
    """Extract a few deterministic fact-like lines from text."""

    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_candidate_line(raw_line)
        if not line:
            continue
        if _is_metadata_line(line):
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in PREFERENCE_KEYWORDS) or DATE_PATTERN.search(line):
            candidates.append(line)
        elif any(token in lowered for token in (" is ", " was ", " are ", " moved ", " current ", " correction", "stale")):
            candidates.append(line)
        if len(candidates) >= 6:
            break
    return candidates


def slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "page"


def normalize_name(value: str) -> str:
    """Normalize speaker names extracted from brackets."""

    cleaned = value.strip().strip(":").strip("*").strip("[]")
    return " ".join(part.capitalize() for part in cleaned.split())


def _page_retrieval_scores(question: str, wiki: CompiledWiki) -> dict[str, float]:
    documents = wiki.pages
    query_tokens = content_tokens(question)
    tokenized_docs = [content_tokens(page.search_text or page.content) for page in documents]
    lexical_scores = _bm25_scores(query_tokens, tokenized_docs)

    dated_pages = [page.timestamp.timestamp() for page in documents if page.timestamp is not None]
    min_timestamp = min(dated_pages) if dated_pages else None
    max_timestamp = max(dated_pages) if dated_pages else None

    scores: dict[str, float] = {}
    for page, lexical_score in zip(documents, lexical_scores, strict=True):
        recency_bonus = 0.0
        if page.timestamp is not None and min_timestamp is not None and max_timestamp is not None and max_timestamp > min_timestamp:
            normalized_recency = (page.timestamp.timestamp() - min_timestamp) / (max_timestamp - min_timestamp)
            recency_bonus = normalized_recency * 0.35
        scores[page.page_id] = lexical_score + PAGE_TYPE_BIASES.get(page.page_type, 0.0) + recency_bonus
    return scores


def _page_search_text(parts: list[str]) -> str:
    search_lines = [_clean_candidate_line(part) for part in parts if _clean_candidate_line(part)]
    return "\n".join(_dedupe_preserve_order(search_lines))


def _session_evidence_lines(
    example: PreparedExample,
    *,
    session_id: str,
    session_turns: list[SessionTurn],
    summary: str,
    mode: str,
) -> list[str]:
    curated_lines = _curated_session_lines(example, session_id=session_id) if mode == "curated" else []
    transcript_lines = curated_lines or [_turn_to_evidence_line(turn) for turn in session_turns]
    summary_lines = [fact for fact in extract_fact_candidates(summary) if fact not in transcript_lines]
    return _dedupe_preserve_order([*summary_lines, *transcript_lines])[:8]


def _curated_session_lines(example: PreparedExample, *, session_id: str) -> list[str]:
    curated_clip_ids = set(str(value) for value in example.metadata.get("curated_clips", []))
    if not curated_clip_ids:
        return []
    return [
        f"{clip.speaker}: {clip.text}"
        for clip in example.history_clips
        if clip.session_id == session_id and clip.clip_id in curated_clip_ids
    ]


def _turn_to_evidence_line(turn: SessionTurn) -> str:
    return f"{turn.role}: {turn.content}".strip()


def _clean_candidate_line(value: str) -> str:
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    cleaned = cleaned.strip().lstrip("-").strip()
    return cleaned


def _is_metadata_line(line: str) -> bool:
    normalized = _clean_candidate_line(line)
    return any(normalized.startswith(prefix) for prefix in METADATA_PREFIXES)


def _is_valid_speaker_name(name: str) -> bool:
    if not name:
        return False
    if name in SPEAKER_METADATA_LABELS:
        return False
    return len(name.split()) <= 4


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _gold_session_indices(example: PreparedExample, top_k: int) -> list[int]:
    if example.gold_evidence:
        selected = [
            index
            for index, session_id in enumerate(example.haystack_session_ids)
            if session_id in set(example.gold_evidence)
        ]
        if selected:
            return selected[:top_k]
    return []


def _heuristic_session_indices(example: PreparedExample, top_k: int) -> list[int]:
    session_texts = []
    for index, session_id in enumerate(example.haystack_session_ids):
        summary = example.haystack_session_summaries[index] if index < len(example.haystack_session_summaries) else ""
        transcript = session_to_markdown(example.haystack_sessions[index])
        session_texts.append(f"{session_id}\n{summary}\n{transcript}")

    scores = _bm25_scores(content_tokens(example.question), [content_tokens(text) for text in session_texts])
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    return sorted(index for index, _ in ranked[: min(top_k, len(ranked))])


def _curated_session_indices_from_metadata(example: PreparedExample) -> list[int]:
    curated_clip_ids = set(str(value) for value in example.metadata.get("curated_clips", []))
    if not curated_clip_ids:
        return []

    selected: set[int] = set()
    for index, session_turns in enumerate(example.haystack_sessions):
        session_id = example.haystack_session_ids[index]
        for turn_index, _turn in enumerate(session_turns):
            clip_id = f"{example.question_id}:{session_id}:turn-{turn_index}"
            if clip_id in curated_clip_ids:
                selected.add(index)
                break

    if selected:
        return sorted(selected)

    history_clip_to_session = {clip.clip_id: clip.session_id for clip in example.history_clips}
    selected_session_ids = {
        history_clip_to_session[clip_id]
        for clip_id in curated_clip_ids
        if clip_id in history_clip_to_session
    }
    return sorted(
        index for index, session_id in enumerate(example.haystack_session_ids) if session_id in selected_session_ids
    )


def _bm25_scores(query_tokens: list[str], documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents:
        return []

    document_frequencies: Counter[str] = Counter()
    for document in documents:
        document_frequencies.update(set(document))

    average_length = sum(len(document) for document in documents) / len(documents)
    scores: list[float] = []

    for document in documents:
        term_frequencies = Counter(document)
        doc_length = max(1, len(document))
        score = 0.0
        for token in query_tokens:
            frequency = term_frequencies.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1.0 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * (doc_length / max(1.0, average_length)))
            score += inverse_document_frequency * ((frequency * (k1 + 1.0)) / denominator)
        scores.append(score)
    return scores
