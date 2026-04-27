"""Deterministic answer helpers for standalone ClipWiki."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Sequence

from clipwiki.schemas import RetrievedItem, TokenUsage
from clipwiki.tokens import content_tokens, normalize_text

ABSTENTION_ANSWER = "Not enough information in memory."
METADATA_LINE_PREFIXES = ("Question:", "Tags:", "Concept:", "Why saved:", "Summary:", "Source:", "Evidence:")
PRIMARY_EVIDENCE_PREFIXES = ("sources/", "evidence/", "preferences/")
NAVIGATION_PREFIXES = ("concepts/", "people/", "events/", "index", "log")


@dataclass(slots=True)
class OpenQASelection:
    answer_text: str
    supporting_item: RetrievedItem | None
    confidence: float
    rationale: str | None = None
    citation_ids: list[str] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class OpenQACandidate:
    snippet: str
    item: RetrievedItem
    score: float


class DeterministicOpenQAAnswerer:
    def set_artifact_dir(self, artifact_dir: Path) -> None:
        pass

    def answer_question(self, example: object, retrieved_items: Sequence[RetrievedItem]) -> OpenQASelection:
        if not retrieved_items:
            return OpenQASelection(answer_text=ABSTENTION_ANSWER, supporting_item=None, confidence=0.0)
        question = str(getattr(example, "question", ""))
        question_tokens = set(content_tokens(question))
        max_retrieval_score = max((item.score for item in retrieved_items), default=1.0) or 1.0
        candidates: list[OpenQACandidate] = []
        for item in retrieved_items:
            for snippet in _candidate_snippets(item.text):
                score = _score_open_qa_candidate(question_tokens=question_tokens, snippet=snippet, item=item, max_retrieval_score=max_retrieval_score)
                if score > 0.0:
                    candidates.append(OpenQACandidate(snippet=snippet, item=item, score=score))
        if not candidates:
            return OpenQASelection(answer_text=ABSTENTION_ANSWER, supporting_item=None, confidence=0.0)
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.item.rank, candidate.item.clip_id, candidate.snippet))
        best_candidate = candidates[0]
        if _should_abstain_from_candidate(question, best_candidate):
            return OpenQASelection(answer_text=ABSTENTION_ANSWER, supporting_item=best_candidate.item, confidence=best_candidate.score, citation_ids=[best_candidate.item.clip_id])
        if _is_multi_snippet_question(question):
            combined_answer, combined_items = _combine_open_qa_candidates(candidates)
            if combined_answer is not None and combined_items:
                return OpenQASelection(answer_text=combined_answer, supporting_item=combined_items[0], confidence=sum(candidate.score for candidate in candidates[: len(combined_items)]), citation_ids=[item.clip_id for item in combined_items])
        return OpenQASelection(answer_text=best_candidate.snippet, supporting_item=best_candidate.item, confidence=best_candidate.score, citation_ids=[best_candidate.item.clip_id])


def _candidate_snippets(text: str) -> list[str]:
    lines = [_normalize_candidate_line(line) for line in text.splitlines() if line.strip()]
    snippets: list[str] = []
    for line in lines:
        if not line:
            continue
        if _is_metadata_line(line) or line.startswith("#") or line.startswith("[["):
            continue
        snippets.append(line)
        snippets.extend([part.strip() for part in re.split(r"[.;]\s+", line) if part.strip() and part.strip() != line])
    return snippets or [text.strip()]


def _score_open_qa_candidate(*, question_tokens: set[str], snippet: str, item: RetrievedItem, max_retrieval_score: float) -> float:
    snippet_tokens = set(content_tokens(snippet))
    if not snippet_tokens:
        return 0.0
    overlap = max(len(question_tokens & snippet_tokens), len(_light_stems(question_tokens) & _light_stems(snippet_tokens)))
    if overlap == 0 and not _looks_answer_bearing(snippet):
        return 0.0
    normalized_retrieval = max(0.0, item.score) / max_retrieval_score
    return overlap * 2.0 + normalized_retrieval * 0.75 + _page_type_bonus(item.clip_id) + (0.75 if _looks_answer_bearing(snippet) else 0.0) + (0.5 if _looks_like_speaker_evidence(snippet) else 0.0)


def _light_stems(tokens: set[str]) -> set[str]:
    stems: set[str] = set()
    for token in tokens:
        if len(token) > 5 and token.endswith("ing"):
            stems.add(token[:-3])
        elif len(token) > 4 and token.endswith("ed"):
            stems.add(token[:-2])
        elif len(token) > 3 and token.endswith("s"):
            stems.add(token[:-1])
        else:
            stems.add(token)
    return stems


def _page_type_bonus(clip_id: str) -> float:
    if clip_id.startswith(PRIMARY_EVIDENCE_PREFIXES):
        return 1.5
    if clip_id.startswith(NAVIGATION_PREFIXES) or clip_id in NAVIGATION_PREFIXES:
        return -2.0
    return 0.0


def _looks_answer_bearing(snippet: str) -> bool:
    lowered = normalize_text(snippet)
    if re.search(r"\d{4}-\d{2}-\d{2}", snippet):
        return True
    return any(token in lowered for token in (" is ", " are ", " was ", " moved ", " update ", " current ", " favorite ", " prefer", ":"))


def _looks_like_speaker_evidence(snippet: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z\s'\-]{0,40}:\s+\S", snippet))


def _is_metadata_line(line: str) -> bool:
    normalized = _normalize_candidate_line(line)
    if any(normalized.startswith(prefix) for prefix in METADATA_LINE_PREFIXES):
        return True
    if normalized.startswith(("## ", "# ")):
        return True
    return normalized in {"Supports", "Source Pages", "Relevant Sources", "Relevant Evidence Pages"}


def _normalize_candidate_line(line: str) -> str:
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"", line)
    return cleaned.strip().lstrip("-").strip()


def _is_multi_snippet_question(question: str) -> bool:
    return any(phrase in question.lower() for phrase in (" and ", "both", "which two", "what are the"))


def _combine_open_qa_candidates(candidates: Sequence[OpenQACandidate], limit: int = 2) -> tuple[str | None, list[RetrievedItem]]:
    selected: list[OpenQACandidate] = []
    seen_text: set[str] = set()
    for candidate in candidates:
        normalized = normalize_text(candidate.snippet)
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < 2:
        return None, []
    return "; ".join(candidate.snippet for candidate in selected), [candidate.item for candidate in selected]


def _should_abstain_from_candidate(question: str, candidate: OpenQACandidate) -> bool:
    return "current" in question.lower() and "current" not in candidate.snippet.lower() and candidate.score < 4.0
