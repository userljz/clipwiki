"""Deterministic link graph helpers for visible ClipWiki notes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clipwiki.markdown_sections import title_from_markdown

WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+(?:#[^\]|]+)?)(?:\|[^\]]+)?\]\]")
INTERNAL_DIR_NAMES = {".clipwiki", ".clipwiki_note_memory"}
EXCLUDED_ORPHAN_BASENAMES = {"index.md", "log.md", "wiki_index.md", "change_log.md"}


@dataclass(slots=True)
class LinkPage:
    """Inbound and outbound links for one visible Markdown page."""

    path: str
    title: str
    outbound: list[str] = field(default_factory=list)
    inbound: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BrokenLink:
    """A wikilink target that could not be resolved to a visible note."""

    source: str
    target: str


@dataclass(slots=True)
class LinkGraph:
    """Serializable link graph for a ClipWiki notes directory."""

    pages: dict[str, LinkPage]
    broken_links: list[BrokenLink]
    orphans: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": {path: asdict(page) for path, page in self.pages.items()},
            "broken_links": [asdict(link) for link in self.broken_links],
            "orphans": self.orphans,
        }


def extract_wikilinks(markdown: str) -> list[str]:
    """Extract raw wikilink targets from Markdown, ignoring aliases."""

    return [match.group(1).strip() for match in WIKILINK_PATTERN.finditer(markdown) if match.group(1).strip()]


def visible_note_paths(notes_root: Path) -> list[Path]:
    """Return visible Markdown notes, excluding ClipWiki internal directories."""

    if not notes_root.exists():
        return []
    notes_root = notes_root.resolve()
    paths: list[Path] = []
    for path in sorted(notes_root.rglob("*.md")):
        relative_parts = path.resolve().relative_to(notes_root).parts
        if any(part in INTERNAL_DIR_NAMES or part.startswith(".") for part in relative_parts):
            continue
        paths.append(path)
    return paths


def build_link_graph(notes_root: Path) -> LinkGraph:
    """Build inbound, outbound, broken-link and orphan metadata for notes."""

    notes_root = notes_root.resolve()
    page_text: dict[str, str] = {}
    pages: dict[str, LinkPage] = {}
    for path in visible_note_paths(notes_root):
        relative_path = path.resolve().relative_to(notes_root).as_posix()
        markdown = path.read_text(encoding="utf-8")
        page_text[relative_path] = markdown
        pages[relative_path] = LinkPage(
            path=relative_path,
            title=title_from_markdown(markdown, fallback=path.stem),
        )

    resolver = _LinkResolver(pages)
    broken_links: list[BrokenLink] = []
    for source, markdown in page_text.items():
        page = pages[source]
        for target in extract_wikilinks(markdown):
            resolved = resolver.resolve(target)
            if resolved is None:
                broken_links.append(BrokenLink(source=source, target=target))
                continue
            if resolved not in page.outbound:
                page.outbound.append(resolved)
            inbound = pages[resolved].inbound
            if source not in inbound:
                inbound.append(source)

    orphans = sorted(path for path, page in pages.items() if not page.inbound and not _is_excluded_orphan(path))
    return LinkGraph(pages=pages, broken_links=broken_links, orphans=orphans)


def resolve_wikilink(notes_root: Path, target: str) -> str | None:
    """Resolve a single wikilink target against the current visible note set."""

    graph = build_link_graph(notes_root)
    return _LinkResolver(graph.pages).resolve(target)


class _LinkResolver:
    def __init__(self, pages: dict[str, LinkPage]) -> None:
        self.pages = pages
        self.by_title = {_normalize_title(page.title): path for path, page in pages.items()}
        self.by_basename = {Path(path).name.lower(): path for path in pages}

    def resolve(self, raw_target: str) -> str | None:
        target = _normalize_target(raw_target)
        if not target:
            return None

        for candidate in _path_candidates(target):
            if candidate in self.pages:
                return candidate

        basename = Path(target if target.endswith(".md") else f"{target}.md").name.lower()
        if basename in self.by_basename:
            return self.by_basename[basename]

        title_key = _normalize_title(target.removesuffix(".md"))
        return self.by_title.get(title_key)


def _normalize_target(target: str) -> str:
    target = target.split("#", 1)[0].strip().replace("\\", "/")
    while target.startswith("./"):
        target = target[2:]
    return target.lstrip("/")


def _path_candidates(target: str) -> list[str]:
    base = target if target.endswith(".md") else f"{target}.md"
    candidates = [base]
    if base.startswith("wiki/"):
        candidates.append(base.removeprefix("wiki/"))
    else:
        candidates.append(f"wiki/{base}")
    return list(dict.fromkeys(candidates))


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _is_excluded_orphan(path: str) -> bool:
    return Path(path).name in EXCLUDED_ORPHAN_BASENAMES
