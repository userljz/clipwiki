"""Markdown heading parsing and section-level patch helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(slots=True)
class MarkdownSection:
    """A Markdown section with one-based line ranges."""

    heading: str
    heading_path: list[str]
    level: int
    start_line: int
    end_line: int
    content: str

    @property
    def id_path(self) -> str:
        return "/".join(self.heading_path)


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """Parse Markdown headings into sections, ignoring headings inside code fences."""

    lines = markdown.splitlines()
    headings: list[tuple[int, int, str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    in_code = False
    frontmatter_end = _frontmatter_end(lines)

    for index, line in enumerate(lines, start=1):
        if index <= frontmatter_end:
            continue
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        headings.append((index, level, heading, [value for _level, value in stack]))

    sections: list[MarkdownSection] = []
    for heading_index, (start_line, level, heading, heading_path) in enumerate(headings):
        end_line = len(lines)
        for next_start, next_level, _next_heading, _next_path in headings[heading_index + 1 :]:
            if next_level <= level:
                end_line = next_start - 1
                break
        content = "\n".join(lines[start_line - 1 : end_line])
        sections.append(
            MarkdownSection(
                heading=heading,
                heading_path=heading_path,
                level=level,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )
    return sections


def find_section(markdown: str, heading_path: list[str]) -> MarkdownSection | None:
    """Find a section by exact heading path."""

    target = [part.strip() for part in heading_path if part.strip()]
    for section in parse_markdown_sections(markdown):
        if section.heading_path == target:
            return section
    return None


def replace_section(markdown: str, section: MarkdownSection, updated_section: str) -> str:
    """Replace exactly one section using its one-based line range."""

    lines = markdown.splitlines()
    replacement = updated_section.strip().splitlines()
    new_lines = lines[: section.start_line - 1] + replacement + lines[section.end_line :]
    return "\n".join(new_lines).rstrip() + "\n"


def insert_section_under_heading(markdown: str, parent: MarkdownSection | None, new_section: str) -> str:
    """Insert a new section below a parent section or at the end of a file."""

    lines = markdown.splitlines()
    insertion_lines = ["", *new_section.strip().splitlines()]
    if parent is None:
        return "\n".join([*lines, *insertion_lines]).rstrip() + "\n"
    new_lines = lines[: parent.end_line] + insertion_lines + lines[parent.end_line :]
    return "\n".join(new_lines).rstrip() + "\n"


def neighboring_context(markdown: str, section: MarkdownSection, *, context_lines: int = 6) -> str:
    """Return a small amount of context around a section."""

    lines = markdown.splitlines()
    before_start = max(0, section.start_line - 1 - context_lines)
    before = lines[before_start : section.start_line - 1]
    after = lines[section.end_line : section.end_line + context_lines]
    return "\n".join([*before, *after]).strip()


def title_from_markdown(markdown: str, fallback: str = "Untitled") -> str:
    """Return the first H1 from a Markdown document."""

    for section in parse_markdown_sections(markdown):
        if section.level == 1:
            return section.heading
    return fallback


def _frontmatter_end(lines: list[str]) -> int:
    if not lines or lines[0].strip() != "---":
        return 0
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return index
    return 0
