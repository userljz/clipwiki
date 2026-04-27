"""Internal note index for ClipWiki incremental updates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipwiki.markdown_sections import MarkdownSection, parse_markdown_sections, title_from_markdown
from clipwiki.tokens import content_tokens

MEMORY_DIR_NAME = ".clipwiki_note_memory"
INDEX_FILENAME = "index.json"
OUTLINE_FILENAME = "outline.md"
SECTION_SUMMARIES_FILENAME = "section_summaries.json"
KNOWLEDGE_HASHES_FILENAME = "knowledge_hashes.json"


@dataclass(slots=True)
class CandidateSection:
    """A section selected as likely relevant to new knowledge."""

    file_path: str
    heading_path: list[str]
    level: int
    summary: str
    keywords: list[str]
    score: float
    start_line: int
    end_line: int
    content_hash: str

    @property
    def section_id(self) -> str:
        return f"{self.file_path}#{'/'.join(self.heading_path)}"


def memory_dir(notes_root: Path) -> Path:
    return notes_root / MEMORY_DIR_NAME


def build_note_index(notes_root: Path) -> dict[str, Any]:
    """Build and persist the internal note index for all visible Markdown notes."""

    files: list[dict[str, Any]] = []
    note_root = notes_root.resolve()
    if notes_root.exists():
        for path in sorted(notes_root.rglob("*.md")):
            if MEMORY_DIR_NAME in path.parts:
                continue
            markdown = path.read_text(encoding="utf-8")
            relative_path = path.resolve().relative_to(note_root).as_posix()
            sections = [_section_record(relative_path, section) for section in parse_markdown_sections(markdown)]
            files.append(
                {
                    "path": relative_path,
                    "title": title_from_markdown(markdown, fallback=path.stem),
                    "summary": _summarize_text(markdown),
                    "sections": sections,
                }
            )

    payload = {"files": files}
    write_note_index(notes_root, payload)
    return payload


def write_note_index(notes_root: Path, payload: dict[str, Any]) -> None:
    """Persist index.json, outline.md, summaries and knowledge hashes."""

    index_dir = memory_dir(notes_root)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / INDEX_FILENAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (index_dir / OUTLINE_FILENAME).write_text(render_outline(payload), encoding="utf-8")
    summaries = {
        section["id"]: {
            "summary": section["summary"],
            "keywords": section["keywords"],
            "heading_path": section["heading_path"],
            "file_path": file_record["path"],
        }
        for file_record in payload.get("files", [])
        for section in file_record.get("sections", [])
    }
    (index_dir / SECTION_SUMMARIES_FILENAME).write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hashes = {
        section["content_hash"]: section["id"]
        for file_record in payload.get("files", [])
        for section in file_record.get("sections", [])
    }
    (index_dir / KNOWLEDGE_HASHES_FILENAME).write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")


def load_or_build_note_index(notes_root: Path) -> dict[str, Any]:
    """Load an index if it exists, otherwise build one."""

    index_path = memory_dir(notes_root) / INDEX_FILENAME
    if not index_path.exists():
        return build_note_index(notes_root)
    return json.loads(index_path.read_text(encoding="utf-8"))


def render_outline(payload: dict[str, Any]) -> str:
    """Render a compact outline for planner prompts."""

    lines = ["# ClipWiki Note Outline", ""]
    for file_record in payload.get("files", []):
        lines.append(f"- `{file_record['path']}` — {file_record.get('title', '')}")
        for section in file_record.get("sections", []):
            indent = "  " * max(1, int(section.get("level", 1)))
            heading = " / ".join(section.get("heading_path", []))
            lines.append(f"{indent}- {heading}: {section.get('summary', '')}")
    return "\n".join(lines).rstrip() + "\n"


def retrieve_candidate_sections(
    payload: dict[str, Any],
    knowledge_units: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[CandidateSection]:
    """Return top-k relevant sections using lightweight lexical scoring."""

    query_terms = _query_terms(knowledge_units)
    candidates: list[CandidateSection] = []
    for file_record in payload.get("files", []):
        for section in file_record.get("sections", []):
            haystack = " ".join(
                [
                    file_record.get("title", ""),
                    " ".join(section.get("heading_path", [])),
                    section.get("summary", ""),
                    " ".join(section.get("keywords", [])),
                ]
            )
            score = _keyword_score(query_terms, haystack)
            if score <= 0:
                continue
            candidates.append(
                CandidateSection(
                    file_path=str(file_record["path"]),
                    heading_path=[str(value) for value in section.get("heading_path", [])],
                    level=int(section.get("level", 1)),
                    summary=str(section.get("summary", "")),
                    keywords=[str(value) for value in section.get("keywords", [])],
                    score=score,
                    start_line=int(section.get("start_line", 0)),
                    end_line=int(section.get("end_line", 0)),
                    content_hash=str(section.get("content_hash", "")),
                )
            )
    return sorted(candidates, key=lambda item: (-item.score, item.file_path, item.heading_path))[:top_k]


def candidate_sections_prompt(candidates: list[CandidateSection]) -> str:
    """Render candidate section summaries for planner prompts."""

    if not candidates:
        return "(none)"
    return "\n\n".join(
        [
            "\n".join(
                [
                    f"- id: {candidate.section_id}",
                    f"  file_path: {candidate.file_path}",
                    f"  heading_path: {' / '.join(candidate.heading_path)}",
                    f"  summary: {candidate.summary}",
                    f"  keywords: {', '.join(candidate.keywords)}",
                    f"  score: {candidate.score:.3f}",
                ]
            )
            for candidate in candidates
        ]
    )


def _section_record(relative_path: str, section: MarkdownSection) -> dict[str, Any]:
    keywords = _keywords(section.heading + "\n" + section.content)
    section_id = f"{relative_path}#{'/'.join(section.heading_path)}"
    return {
        "id": section_id,
        "heading_path": section.heading_path,
        "level": section.level,
        "summary": _summarize_text(section.content),
        "keywords": keywords,
        "content_hash": _hash(section.content),
        "start_line": section.start_line,
        "end_line": section.end_line,
        "semantic_key": " ".join([*section.heading_path, *keywords[:8]]),
        "embedding_id": None,
    }


def _summarize_text(text: str, max_length: int = 600) -> str:
    cleaned = " ".join(_strip_markdown_noise(text).split())
    return cleaned[:max_length]


def _keywords(text: str, limit: int = 24) -> list[str]:
    tokens = [token for token in content_tokens(text) if len(token) > 2]
    counts = Counter(tokens)
    return [token for token, _count in counts.most_common(limit)]


def _query_terms(knowledge_units: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for unit in knowledge_units:
        parts.append(str(unit.get("claim", "")))
        parts.extend(str(value) for value in unit.get("keywords", []) if str(value).strip())
        parts.extend(str(value) for value in unit.get("possible_topics", []) if str(value).strip())
    return content_tokens(" ".join(parts))


def _keyword_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_tokens = set(content_tokens(text))
    query_counts = Counter(query_terms)
    return float(sum(count for token, count in query_counts.items() if token in text_tokens))


def _strip_markdown_noise(text: str) -> str:
    without_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    return re.sub(r"[#>*_\-\[\]()`]", " ", without_code)


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
