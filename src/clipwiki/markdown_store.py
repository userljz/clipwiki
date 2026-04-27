"""Simple markdown page writer for ClipWiki artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class MarkdownPage:
    """In-memory markdown page representation."""

    page_id: str
    relative_path: str
    title: str
    content: str
    source_ids: list[str]
    page_type: str = "source"
    is_answerable: bool = True
    search_text: str | None = None
    timestamp: datetime | None = None


class MarkdownStore:
    """Filesystem-backed markdown store used by ClipWiki."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_page(self, page: MarkdownPage) -> Path:
        """Write a markdown page and return its full path."""

        page_path = self.root / page.relative_path
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(page.content, encoding="utf-8")
        return page_path
