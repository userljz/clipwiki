"""Validation helpers for ClipWiki incremental note updates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clipwiki.markdown_normalization import markdown_format_issues
from clipwiki.markdown_sections import parse_markdown_sections

CHAT_TRACE_PATTERNS = (
    "用户问",
    "用户提问",
    "AI回答",
    "AI 答",
    "这段回答",
    "这段内容",
    "copied text",
    "the ai answer",
    "the user asked",
    "new content",
)

PLACEHOLDER_PATTERNS = (
    "此处保留",
    "待补充",
    "占位",
    "placeholder",
    "todo",
    "insert command here",
    "insert details here",
    "原文示例内容",
    "命令片段，确保笔记可替代原文",
)


@dataclass(slots=True)
class ValidationResult:
    """Validation outcome for a proposed section update."""

    valid: bool
    issues: list[str] = field(default_factory=list)


def validate_updated_section(
    original_section: str,
    updated_section: str,
    *,
    expected_heading_path: list[str] | None = None,
    check_append: bool = True,
) -> ValidationResult:
    """Run deterministic checks before accepting a note update."""

    issues: list[str] = []
    normalized = updated_section.lower()
    for pattern in CHAT_TRACE_PATTERNS:
        if pattern.lower() in normalized:
            issues.append(f"chat_trace:{pattern}")

    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.lower() in normalized:
            issues.append(f"placeholder_text:{pattern}")

    headings = [section.heading for section in parse_markdown_sections(updated_section)]
    duplicates = sorted({heading for heading in headings if headings.count(heading) > 1})
    if duplicates:
        issues.append(f"duplicate_headings:{', '.join(duplicates)}")

    repeated = _repeated_paragraphs(updated_section)
    if repeated:
        issues.append("duplicate_paragraphs")

    if updated_section.lstrip().startswith("---"):
        issues.append("visible_frontmatter")

    if "## 整理笔记" in updated_section or "## 原始摘录" in updated_section:
        issues.append("machine_wrapper_heading")

    if check_append and _looks_like_append(original_section, updated_section):
        issues.append("append_like_update")

    issues.extend(markdown_format_issues(updated_section))

    if expected_heading_path:
        parsed = parse_markdown_sections(updated_section)
        if parsed and parsed[0].heading != expected_heading_path[-1]:
            issues.append("heading_path_changed")

    return ValidationResult(valid=not issues, issues=issues)


def validate_note_file(markdown: str) -> ValidationResult:
    """Validate a full visible note for readability constraints."""

    return validate_updated_section("", markdown)


def _repeated_paragraphs(markdown: str) -> list[str]:
    paragraphs = [
        " ".join(part.split()).strip().lower()
        for part in re.split(r"\n\s*\n", markdown)
        if len(" ".join(part.split()).strip()) > 80
    ]
    return [paragraph for paragraph in set(paragraphs) if paragraphs.count(paragraph) > 1]


def _looks_like_append(original: str, updated: str) -> bool:
    if not original.strip():
        return False
    original_clean = original.strip()
    updated_clean = updated.strip()
    if not updated_clean.startswith(original_clean):
        return False
    added = updated_clean[len(original_clean) :].strip()
    return len(added) > 80 and any(marker in added for marker in ("##", "---", "更新", "New Notes"))
