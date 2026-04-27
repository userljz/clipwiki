"""Reusable ClipWiki APIs for personal Markdown memory workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from clipwiki.compiler import CompiledWiki, compile_clipwiki, page_score_lookup, retrieve_wiki_pages, slugify
from clipwiki.markdown_store import MarkdownPage
from clipwiki.schemas import EvalCase, HistoryClip, RetrievedItem, SessionTurn, TaskType
from clipwiki.answering import DeterministicOpenQAAnswerer
from clipwiki.tokens import estimate_text_tokens

MANIFEST_FILENAME = ".clipwiki-manifest.json"
DEFAULT_PERSONAL_QUESTION = "What should this wiki remember?"


@dataclass(slots=True)
class ClipWikiTurn:
    """A single personal-memory turn before benchmark conversion."""

    role: str
    content: str


@dataclass(slots=True)
class ClipWikiSession:
    """A source-backed unit of personal memory that can be compiled into pages."""

    session_id: str
    turns: list[ClipWikiTurn]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    source_path: Path | None = None


@dataclass(slots=True)
class ClipWikiSearchHit:
    """Search result returned by the reusable ClipWiki API."""

    page_id: str
    title: str
    page_type: str
    score: float
    source_ids: list[str]
    text: str
    path: Path | None = None


@dataclass(slots=True)
class ClipWikiAnswer:
    """Deterministic answer and the retrieved wiki pages that supported it."""

    answer_text: str
    citations: list[str]
    hits: list[ClipWikiSearchHit]


def build_wiki_from_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    question: str = DEFAULT_PERSONAL_QUESTION,
    question_type: str = "personal-memory",
    mode: str = "full-wiki",
    suffixes: Iterable[str] = (".md", ".txt"),
) -> CompiledWiki:
    """Compile Markdown/TXT files into a personal ClipWiki vault."""

    sessions = list(iter_sessions_from_directory(source_dir, suffixes=suffixes))
    return build_wiki(
        sessions,
        output_dir=output_dir,
        question=question,
        question_type=question_type,
        mode=mode,
    )


def build_wiki(
    sessions: list[ClipWikiSession],
    output_dir: Path,
    *,
    question: str = DEFAULT_PERSONAL_QUESTION,
    question_type: str = "personal-memory",
    mode: str = "full-wiki",
) -> CompiledWiki:
    """Compile reusable ClipWiki sessions without calling the benchmark runner."""

    example = _example_from_sessions(sessions, question=question, question_type=question_type)
    compiled = compile_clipwiki(example, output_dir=output_dir, mode=mode)
    _write_manifest(output_dir, compiled)
    return compiled


def iter_sessions_from_directory(
    source_dir: Path,
    *,
    suffixes: Iterable[str] = (".md", ".txt"),
) -> Iterable[ClipWikiSession]:
    """Yield one ClipWiki session for each Markdown/TXT file under a directory."""

    allowed_suffixes = {suffix.lower() for suffix in suffixes}
    if not source_dir.exists():
        raise FileNotFoundError(f"ClipWiki source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"ClipWiki source path is not a directory: {source_dir}")

    used_ids: set[str] = set()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        relative_stem = path.relative_to(source_dir).with_suffix("").as_posix()
        session_id = _unique_session_id(slugify(relative_stem), used_ids)
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        yield session_from_text(session_id=session_id, text=text, timestamp=timestamp, source_path=path)


def session_from_text(
    *,
    session_id: str,
    text: str,
    timestamp: datetime | None = None,
    source_path: Path | None = None,
) -> ClipWikiSession:
    """Create a ClipWiki session from a Markdown or plain-text document."""

    paragraphs = _paragraphs(text)
    return ClipWikiSession(
        session_id=session_id,
        turns=[ClipWikiTurn(role="note", content=paragraph) for paragraph in paragraphs],
        timestamp=timestamp or datetime.now(timezone.utc),
        summary=_summary_from_text(text),
        source_path=source_path,
    )


def load_wiki(wiki_dir: Path) -> CompiledWiki:
    """Load a compiled ClipWiki vault from disk for search or deterministic QA."""

    manifest_path = wiki_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages = [_page_from_manifest(wiki_dir, page_payload) for page_payload in payload.get("pages", [])]
        selected_session_ids = [str(value) for value in payload.get("selected_session_ids", [])]
    else:
        pages = list(_load_pages_without_manifest(wiki_dir))
        selected_session_ids = sorted({source_id for page in pages for source_id in page.source_ids})

    return CompiledWiki(
        pages=pages,
        wiki_size_pages=len(pages),
        wiki_size_tokens=sum(estimate_text_tokens(page.content) for page in pages),
        selected_session_ids=selected_session_ids,
    )


def search_wiki(question: str, wiki: CompiledWiki, *, top_k: int = 5, wiki_dir: Path | None = None) -> list[ClipWikiSearchHit]:
    """Search a compiled ClipWiki vault and return scored page hits."""

    scores = page_score_lookup(question, wiki)
    pages = retrieve_wiki_pages(question, wiki, top_k=top_k)
    return [_hit_from_page(page, scores.get(page.page_id, 0.0), wiki_dir=wiki_dir) for page in pages]


def ask_wiki(question: str, wiki: CompiledWiki, *, top_k: int = 5, wiki_dir: Path | None = None) -> ClipWikiAnswer:
    """Answer a question over a compiled ClipWiki vault using deterministic snippets."""

    scores = page_score_lookup(question, wiki)
    pages = retrieve_wiki_pages(question, wiki, top_k=top_k)
    answerable_pages = [page for page in pages if page.is_answerable]
    retrieved_items = [
        RetrievedItem(
            clip_id=page.page_id,
            rank=index + 1,
            score=float(scores.get(page.page_id, 0.0)),
            text=page.search_text or page.content,
            retrieved_tokens=estimate_text_tokens(page.search_text or page.content),
        )
        for index, page in enumerate(answerable_pages or pages)
    ]
    example = EvalCase(
        example_id="clipwiki-personal-question",
        dataset_name="clipwiki-personal",
        task_type=TaskType.OPEN_QA,
        question=question,
        answer="unknown",
        question_type="personal-memory",
    )
    selection = DeterministicOpenQAAnswerer().answer_question(example, retrieved_items)
    citations = selection.citation_ids or ([selection.supporting_item.clip_id] if selection.supporting_item is not None else [])
    return ClipWikiAnswer(
        answer_text=selection.answer_text,
        citations=citations,
        hits=[_hit_from_page(page, scores.get(page.page_id, 0.0), wiki_dir=wiki_dir) for page in pages],
    )


def _example_from_sessions(sessions: list[ClipWikiSession], *, question: str, question_type: str) -> EvalCase:
    history_clips: list[HistoryClip] = []
    haystack_sessions: list[list[SessionTurn]] = []
    haystack_session_ids: list[str] = []
    haystack_session_summaries: list[str] = []
    haystack_session_datetimes: list[datetime] = []

    for session in sessions:
        session_turns = [SessionTurn(role=turn.role, content=turn.content) for turn in session.turns]
        haystack_sessions.append(session_turns)
        haystack_session_ids.append(session.session_id)
        haystack_session_summaries.append(session.summary)
        haystack_session_datetimes.append(session.timestamp)
        for index, turn in enumerate(session.turns):
            history_clips.append(
                HistoryClip(
                    clip_id=f"{session.session_id}:turn-{index}",
                    conversation_id="clipwiki-personal",
                    session_id=session.session_id,
                    speaker=turn.role,
                    timestamp=session.timestamp,
                    text=turn.content,
                    turn_id=str(index),
                    source_ref=str(session.source_path) if session.source_path is not None else session.session_id,
                    metadata={"source_path": str(session.source_path) if session.source_path is not None else ""},
                )
            )

    return EvalCase(
        example_id="clipwiki-personal",
        dataset_name="clipwiki-personal",
        task_type=TaskType.OPEN_QA,
        question=question,
        answer="unknown",
        question_id="clipwiki-personal",
        question_type=question_type,
        history_clips=history_clips,
        haystack_sessions=haystack_sessions,
        haystack_session_ids=haystack_session_ids,
        haystack_session_summaries=haystack_session_summaries,
        haystack_session_datetimes=haystack_session_datetimes,
        metadata={"source": "clipwiki-personal"},
    )


def _write_manifest(output_dir: Path, compiled: CompiledWiki) -> None:
    payload = {
        "selected_session_ids": compiled.selected_session_ids,
        "pages": [
            {
                "page_id": page.page_id,
                "relative_path": page.relative_path,
                "title": page.title,
                "source_ids": page.source_ids,
                "page_type": page.page_type,
                "is_answerable": page.is_answerable,
                "search_text": page.search_text,
                "timestamp": page.timestamp.isoformat() if page.timestamp is not None else None,
            }
            for page in compiled.pages
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _page_from_manifest(wiki_dir: Path, payload: dict[str, object]) -> MarkdownPage:
    relative_path = str(payload["relative_path"])
    content = (wiki_dir / relative_path).read_text(encoding="utf-8")
    timestamp_value = payload.get("timestamp")
    timestamp = datetime.fromisoformat(str(timestamp_value)) if timestamp_value else None
    return MarkdownPage(
        page_id=str(payload["page_id"]),
        relative_path=relative_path,
        title=str(payload["title"]),
        content=content,
        source_ids=[str(value) for value in payload.get("source_ids", [])],
        page_type=str(payload.get("page_type", "source")),
        is_answerable=bool(payload.get("is_answerable", True)),
        search_text=str(payload["search_text"]) if payload.get("search_text") is not None else None,
        timestamp=timestamp,
    )


def _load_pages_without_manifest(wiki_dir: Path) -> Iterable[MarkdownPage]:
    for path in sorted(wiki_dir.rglob("*.md")):
        relative_path = path.relative_to(wiki_dir).as_posix()
        page_id = relative_path.removesuffix(".md")
        page_type = page_id.split("/", 1)[0] if "/" in page_id else page_id
        source_ids = [page_id.split("/", 1)[1]] if "/" in page_id and page_type in {"sources", "evidence", "preferences", "events"} else []
        yield MarkdownPage(
            page_id=page_id,
            relative_path=relative_path,
            title=path.stem,
            content=path.read_text(encoding="utf-8"),
            source_ids=source_ids,
            page_type=page_type.removesuffix("s"),
            is_answerable=page_type in {"sources", "evidence", "preferences"},
        )


def _hit_from_page(page: MarkdownPage, score: float, *, wiki_dir: Path | None) -> ClipWikiSearchHit:
    path = wiki_dir / page.relative_path if wiki_dir is not None else None
    return ClipWikiSearchHit(
        page_id=page.page_id,
        title=page.title,
        page_type=page.page_type,
        score=float(score),
        source_ids=page.source_ids,
        text=page.search_text or page.content,
        path=path,
    )


def _paragraphs(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in text.replace("\r\n", "\n").split("\n\n") if chunk.strip()]
    return chunks or [text.strip()]


def _summary_from_text(text: str, max_length: int = 240) -> str:
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:max_length]
    return text.strip()[:max_length]


def _unique_session_id(base_id: str, used_ids: set[str]) -> str:
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id
    suffix = 2
    while f"{base_id}-{suffix}" in used_ids:
        suffix += 1
    session_id = f"{base_id}-{suffix}"
    used_ids.add(session_id)
    return session_id
