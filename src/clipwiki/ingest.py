"""LLM-assisted ClipWiki ingestion for copied web AI answers."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import markdown as markdown_lib

from clipwiki.compiler import slugify
from clipwiki.incremental import run_incremental_ingest
from clipwiki.markdown_normalization import normalize_generated_markdown
from clipwiki.note_index import build_note_index
from clipwiki.wiki_maintenance import record_ingest_event

DEFAULT_NOTES_DIR = Path("clipwiki-notes")
DEFAULT_HTML_DIR = Path("clipwiki-html")
DEFAULT_INBOX_DIR = Path("clipwiki-inbox")


@dataclass(slots=True)
class ClipWikiIngestResult:
    """Files produced by an LLM-assisted ClipWiki ingest run."""

    note_path: Path | None
    html_path: Path | None
    action: str
    category: str
    title: str
    tags: list[str]
    artifact_path: str | None = None
    llm_backend_calls: int = 0
    llm_cached: bool = False
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    pruned_empty_notes: int = 0
    pruned_duplicate_notes: int = 0
    pruned_orphan_html_pages: int = 0
    status: str = "updated"
    reason: str = ""
    diff: str | None = None
    validation_issues: list[str] | None = None


def ingest_web_ai_result(
    input_path: Path,
    *,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    html_dir: Path = DEFAULT_HTML_DIR,
    category_hint: str | None = None,
    title_hint: str | None = None,
    user_question: str | None = None,
    top_k: int = 5,
    dry_run: bool = False,
    model: str | None = None,
    cheap_model: str | None = None,
    strong_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    artifact_dir: Path | None = None,
) -> ClipWikiIngestResult:
    """Organize copied AI output into a Markdown note and matching HTML page."""

    raw_text = input_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError(
            f"ClipWiki ingest input is empty: {input_path}. "
            "Paste the copied AI answer into this file before running ingest."
        )
    pruned_empty_notes = prune_empty_notes(notes_dir, html_dir=html_dir)
    pruned_duplicate_notes = prune_duplicate_notes(notes_dir, html_dir=html_dir)
    pruned_orphan_html_pages = prune_orphan_html_pages(notes_dir, html_dir=html_dir)
    if pruned_empty_notes or pruned_duplicate_notes or pruned_orphan_html_pages:
        build_note_index(notes_dir)
        if html_dir.exists():
            _write_html_index(html_dir, notes_dir)
    incremental = run_incremental_ingest(
        raw_content=raw_text,
        source_path=input_path,
        notes_root=notes_dir,
        user_question=user_question,
        category_hint=category_hint,
        title_hint=title_hint,
        top_k=top_k,
        dry_run=dry_run,
        model=model,
        cheap_model=cheap_model,
        strong_model=strong_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    incremental.pruned_empty_notes += pruned_empty_notes

    html_path: Path | None = None
    if incremental.status in {"created", "updated"} and incremental.note_path is not None and incremental.markdown is not None:
        html_path = _html_path_for_note(incremental.note_path, notes_dir=notes_dir, html_dir=html_dir)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            render_note_html(
                incremental.markdown,
                title=incremental.title,
                category=incremental.category,
                note_path=incremental.note_path,
            ),
            encoding="utf-8",
        )
        _write_html_index(html_dir, notes_dir)

    result = ClipWikiIngestResult(
        note_path=incremental.note_path,
        html_path=html_path,
        action=incremental.decision,
        category=incremental.category,
        title=incremental.title,
        tags=incremental.tags,
        artifact_path=incremental.llm_stats.artifact_paths[-1] if incremental.llm_stats.artifact_paths else None,
        llm_backend_calls=incremental.llm_stats.backend_calls,
        llm_cached=incremental.llm_stats.any_cached,
        llm_input_tokens=incremental.llm_stats.input_tokens,
        llm_output_tokens=incremental.llm_stats.output_tokens,
        llm_total_tokens=incremental.llm_stats.total_tokens,
        llm_estimated_cost_usd=incremental.llm_stats.estimated_cost_usd,
        pruned_empty_notes=incremental.pruned_empty_notes,
        pruned_duplicate_notes=pruned_duplicate_notes,
        pruned_orphan_html_pages=pruned_orphan_html_pages,
        status=incremental.status,
        reason=incremental.reason,
        diff=incremental.diff,
        validation_issues=incremental.validation_issues,
    )
    if result.status in {"created", "updated", "skipped", "failed"}:
        record_ingest_event(
            notes_dir,
            status=result.status,
            action=result.action,
            note_path=result.note_path,
            title=result.title,
            reason=result.reason,
            llm_backend_calls=result.llm_backend_calls,
            llm_total_tokens=result.llm_total_tokens,
            llm_estimated_cost_usd=result.llm_estimated_cost_usd,
        )
    return result


def prune_empty_notes(notes_dir: Path, *, html_dir: Path | None = None) -> int:
    """Remove known-empty/no-value notes produced by earlier failed ingests."""

    if not notes_dir.exists():
        return 0
    removed = 0
    for note_path in sorted(notes_dir.rglob("*.md")):
        markdown = note_path.read_text(encoding="utf-8")
        if not _is_disposable_empty_note(markdown):
            continue
        note_path.unlink()
        removed += 1
        if html_dir is not None:
            html_path = _html_path_for_note(note_path, notes_dir=notes_dir, html_dir=html_dir)
            if html_path.exists():
                html_path.unlink()
                _remove_empty_parents(html_path.parent, stop_at=html_dir)
        _remove_empty_parents(note_path.parent, stop_at=notes_dir)
    return removed


def prune_duplicate_notes(notes_dir: Path, *, html_dir: Path | None = None) -> int:
    """Prune obvious duplicate visible notes as part of normal wiki maintenance."""

    if not notes_dir.exists():
        return 0
    by_title: dict[str, list[Path]] = {}
    for note_path in sorted(notes_dir.rglob("*.md")):
        if ".clipwiki_note_memory" in note_path.parts:
            continue
        markdown = note_path.read_text(encoding="utf-8")
        title = _title_from_markdown(markdown)
        if not title:
            continue
        by_title.setdefault(_normalize_title(title), []).append(note_path)

    removed = 0
    for paths in by_title.values():
        if len(paths) < 2:
            continue
        keep = _preferred_note_path(paths, notes_dir=notes_dir)
        for duplicate in paths:
            if duplicate == keep:
                continue
            duplicate.unlink()
            removed += 1
            if html_dir is not None:
                html_path = _html_path_for_note(duplicate, notes_dir=notes_dir, html_dir=html_dir)
                if html_path.exists():
                    html_path.unlink()
                    _remove_empty_parents(html_path.parent, stop_at=html_dir)
            _remove_empty_parents(duplicate.parent, stop_at=notes_dir)
    return removed


def prune_orphan_html_pages(notes_dir: Path, *, html_dir: Path) -> int:
    """Remove rendered HTML pages whose source Markdown note no longer exists."""

    if not html_dir.exists():
        return 0
    removed = 0
    for html_path in sorted(html_dir.rglob("*.html")):
        if html_path.name == "index.html":
            continue
        relative = html_path.relative_to(html_dir)
        note_path = notes_dir / relative.with_suffix(".md")
        if note_path.exists():
            continue
        html_path.unlink()
        removed += 1
        _remove_empty_parents(html_path.parent, stop_at=html_dir)
    return removed


def render_note_html(markdown: str, *, title: str, category: str, note_path: Path) -> str:
    """Render a single Markdown note into a styled standalone HTML page."""

    body_markdown = normalize_generated_markdown(_markdown_without_top_title(markdown, title=title))
    body = _markdown_to_html(body_markdown)
    escaped_title = html.escape(title)
    escaped_category = html.escape(category)
    escaped_note_path = html.escape(str(note_path))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title} - ClipWiki</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7fb;
      --paper: #ffffff;
      --text: #18202f;
      --muted: #637083;
      --line: #dfe5ef;
      --accent: #4169e1;
      --accent-soft: #eef3ff;
      --code: #f1f4f9;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0e1117;
        --paper: #171b24;
        --text: #e8edf7;
        --muted: #9aa6bb;
        --line: #2a3140;
        --accent: #8aadff;
        --accent-soft: #1f2b46;
        --code: #202634;
      }}
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, var(--accent-soft), transparent 36rem), var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.72;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      max-width: 980px;
      margin: 0 auto;
      padding: 48px 22px;
    }}
    article {{
      background: color-mix(in srgb, var(--paper) 92%, transparent);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 60px rgba(15, 23, 42, 0.12);
      padding: 42px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 0.86rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .meta {{
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      padding-bottom: 22px;
      margin-bottom: 28px;
      font-size: 0.95rem;
    }}
    h1, h2, h3 {{
      line-height: 1.22;
      letter-spacing: -0.02em;
    }}
    h1 {{
      font-size: clamp(2rem, 5vw, 3.4rem);
      margin: 10px 0 14px;
    }}
    h2 {{
      border-top: 1px solid var(--line);
      padding-top: 26px;
      margin-top: 34px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    code {{
      background: var(--code);
      border-radius: 6px;
      padding: 0.12rem 0.35rem;
    }}
    pre {{
      background: var(--code);
      border-radius: 16px;
      overflow-x: auto;
      padding: 18px;
    }}
    pre code {{
      background: transparent;
      padding: 0;
      border-radius: 0;
      font-size: 0.92rem;
      line-height: 1.62;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1.2rem 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 0.65rem 0.8rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--accent-soft);
      font-weight: 700;
    }}
    blockquote {{
      border-left: 4px solid var(--accent);
      margin-left: 0;
      padding: 4px 0 4px 18px;
      color: var(--muted);
      background: var(--accent-soft);
      border-radius: 0 12px 12px 0;
    }}
    li {{
      margin: 0.35rem 0;
    }}
    .footer {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 36px;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 14px;
      margin: 24px 0;
      padding: 12px 16px;
      background: color-mix(in srgb, var(--paper) 84%, var(--accent-soft));
    }}
    summary {{
      cursor: pointer;
      color: var(--muted);
      font-weight: 650;
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <main class="layout">
    <article>
      <div class="eyebrow">ClipWiki / {escaped_category}</div>
      <h1>{escaped_title}</h1>
      {body}
      <details class="footer">
        <summary>页面信息</summary>
        <p>Markdown source: <code>{escaped_note_path}</code></p>
        <p>Generated: {generated_at}</p>
      </details>
    </article>
  </main>
</body>
</html>
"""


def _build_ingest_prompt(
    raw_text: str,
    *,
    input_path: Path,
    existing_notes: list[dict[str, str]],
    category_hint: str | None,
    title_hint: str | None,
) -> str:
    trimmed_text = raw_text[:24000]
    existing_note_lines = "\n".join(
        f"- path: {note['path']}\n  title: {note['title']}\n  excerpt: {note['excerpt']}" for note in existing_notes[:40]
    )
    existing_note_content = "\n\n".join(
        [
            f"EXISTING_NOTE_BEGIN path={note['path']}\n{note.get('content', '')[:3000]}\nEXISTING_NOTE_END"
            for note in existing_notes[:8]
            if note.get("content")
        ]
    )
    return "\n".join(
        [
            "You are ClipWiki, a local personal wiki curator and knowledge crystallizer.",
            "The user copied an AI answer from a web chat. The copied text may be noisy, missing context, or written as a chat explanation.",
            "Your job is NOT to merely summarize it. Your job is to extract durable knowledge that is worth saving in a personal wiki.",
            "Your larger job is to maintain a readable, logically organized personal notebook over time.",
            "Do not behave like a converter from txt to Markdown, and do not behave like an append-only log.",
            "Infer the topic from the content when context is missing. Preserve uncertainty explicitly instead of pretending unsupported facts are certain.",
            "Decide whether to merge into an existing note path or create a new category/note.",
            "Return ONLY a JSON object with this schema:",
            "{",
            '  "action": "create" | "merge",',
            '  "title": "clear note title",',
            '  "category": "short category folder name",',
            '  "target_note": "relative/path.md or empty string",',
            '  "confidence": "EXTRACTED" | "INFERRED" | "AMBIGUOUS" | "UNVERIFIED",',
            '  "summary": "2-4 sentence summary",',
            '  "key_points": ["durable knowledge point", "..."],',
            '  "details": ["useful implementation detail, caveat, command, or example", "..."],',
            '  "tags": ["tag", "..."],',
            '  "related_topics": ["topic", "..."],',
            '  "source_excerpt": "1-3 short original excerpts worth preserving",',
            '  "markdown_body": "complete Markdown body without YAML frontmatter"',
            "}",
            "",
            "Knowledge extraction rules:",
            "- Treat the copied text as source material, even if it lacks background.",
            "- Never answer that the input is empty unless the SOURCE_TEXT block below is actually empty.",
            "- Do not invent facts not supported by the copied content.",
            "- Use EXTRACTED when the note mostly preserves explicit claims from the text.",
            "- Use INFERRED when you infer structure/topic from fragmentary text.",
            "- Use AMBIGUOUS when the copied text is contradictory or missing too much context.",
            "- Use UNVERIFIED for claims that look useful but need later checking.",
            "- Prefer stable knowledge over chatty wording, but keep useful commands and concrete examples.",
            "- Preserve concrete commands, code snippets, URLs, API names, file paths, model names, and caveats when present.",
            "- If this is a how-to/explanation, convert it into a reusable note with steps, mental model, pitfalls, and examples.",
            "- If this is a design discussion, extract decisions, rationale, tradeoffs, and follow-up questions.",
            "- If this is a debugging transcript, extract symptoms, root cause, fix, and prevention.",
            "",
            "Markdown body requirements:",
            "- Write polished Chinese Markdown unless the copied text is mainly English technical text.",
            "- The result must read like a maintained wiki note: clear title, coherent sections, no duplicate headings, no chat filler.",
            "- Prefer reorganizing and consolidating ideas over appending a new dated chunk.",
            "- markdown_body must be the final student-quality note body. Do not include a top-level H1 title; the program adds the title.",
            "- Do not include sections named 摘要, 关键要点, 细节, 整理笔记, 原始摘录, or 值得保留的原文摘录 unless they are truly natural for the topic.",
            "- Avoid repeating the same idea in both bullet form and prose.",
            "- Inline math must use `$...$`, for example `$p$`, `$k$`, and `$\\alpha/\\beta$`.",
            "- Display math must use `$$...$$` on its own lines.",
            "- Do not write math as `(p)`, `(k)`, or `(\\alpha/\\beta)` when it is meant to render as a formula.",
            "- Parent bullets with child items must use nested Markdown lists with four-space indentation for renderer compatibility, e.g. `- **方法**：` followed by `    - 空间显著性：...`.",
            "- Use this structure when applicable:",
            "  ## 核心结论",
            "  ## 背景与问题",
            "  ## 操作步骤 / 方法",
            "  ## 注意事项",
            "  ## 相关概念",
            "  ## 待验证",
            "- Do not include YAML frontmatter; the program will add it.",
            "",
            "Merge/create rules:",
            "- Do not invent facts not supported by the copied content.",
            "- If an existing note is clearly about the same topic, set action to merge and target_note to that path.",
            "- When action is merge, markdown_body MUST be the full rewritten body for the whole note, integrating existing useful content and the new source.",
            "- When action is merge, do NOT output an incremental patch, changelog, or append-only section.",
            "- If creating a new note, use a concise category and title.",
            "",
            f"Input file: {input_path}",
            f"Source character count: {len(raw_text)}",
            f"Category hint: {category_hint or ''}",
            f"Title hint: {title_hint or ''}",
            "",
            "Existing notes:",
            existing_note_lines or "(none)",
            "",
            "Existing note contents for possible merge:",
            existing_note_content or "(none)",
            "",
            "SOURCE_TEXT_BEGIN",
            trimmed_text,
            "SOURCE_TEXT_END",
        ]
    )


def _normalize_ingest_payload(
    payload: dict[str, Any],
    *,
    raw_text: str,
    category_hint: str | None,
    title_hint: str | None,
) -> dict[str, Any]:
    title = str(payload.get("title") or title_hint or _first_nonempty_line(raw_text) or "Untitled ClipWiki Note").strip()
    category = str(payload.get("category") or category_hint or "general").strip()
    markdown_body = str(payload.get("markdown_body") or "").strip()
    if not markdown_body or _looks_like_empty_input_response(markdown_body, raw_text=raw_text):
        markdown_body = _fallback_markdown_body(payload, raw_text=raw_text)
    action = str(payload.get("action") or "create").strip().lower()
    if action not in {"create", "merge"}:
        action = "create"
    confidence = str(payload.get("confidence") or "INFERRED").strip().upper()
    if confidence not in {"EXTRACTED", "INFERRED", "AMBIGUOUS", "UNVERIFIED"}:
        confidence = "INFERRED"
    return {
        "action": action,
        "title": title,
        "category": category,
        "target_note": str(payload.get("target_note") or "").strip(),
        "confidence": confidence,
        "summary": str(payload.get("summary") or "").strip(),
        "key_points": _string_list(payload.get("key_points")),
        "details": _string_list(payload.get("details")),
        "tags": _string_list(payload.get("tags")),
        "related_topics": _string_list(payload.get("related_topics")),
        "source_excerpt": str(payload.get("source_excerpt") or "").strip(),
        "markdown_body": markdown_body,
    }


def _compose_markdown(
    payload: dict[str, Any],
    *,
    raw_text: str,
    source_path: Path,
    existing_markdown: str,
) -> str:
    title = payload["title"]
    body = _clean_note_body(payload["markdown_body"], title=title)
    lines = [f"# {title}", ""]
    if payload["summary"] and _summary_not_repeated(payload["summary"], body):
        lines.extend([f"> {payload['summary']}", ""])
    lines.extend([body, ""])
    if payload["related_topics"]:
        lines.extend(["## 相关主题", "", *[f"- [[{topic}]]" for topic in payload["related_topics"]], ""])
    lines.extend(
        [
            "<details>",
            "<summary>来源信息</summary>",
            "",
            f"- 来源文件：`{source_path}`",
            f"- 分类：{payload['category']}",
            f"- 置信度：{payload['confidence']}",
            f"- 标签：{', '.join(payload['tags']) if payload['tags'] else '无'}",
            "",
        ]
    )
    if payload["source_excerpt"]:
        lines.extend(["### 保留摘录", "", payload["source_excerpt"], ""])
    lines.extend(["</details>", ""])
    new_markdown = "\n".join(lines).strip() + "\n"
    return new_markdown


def _fallback_markdown_body(payload: dict[str, Any], *, raw_text: str) -> str:
    lines = []
    summary = str(payload.get("summary") or "").strip()
    if summary:
        lines.extend(["## Summary", "", summary, ""])
    key_points = _string_list(payload.get("key_points"))
    if key_points:
        lines.extend(["## Key Points", "", *[f"- {point}" for point in key_points], ""])
    lines.extend(
        [
            "## 核心结论",
            "",
            _first_nonempty_line(raw_text) or "这段复制内容需要进一步人工整理。",
            "",
            "## 背景与问题",
            "",
            "该条目由 ClipWiki 根据网页端复制内容自动兜底生成。模型返回的结构化整理结果不足，因此保留原文并生成一个可继续编辑的笔记骨架。",
            "",
            "## 待整理原文",
            "",
            raw_text.strip(),
        ]
    )
    return "\n".join(lines).strip()


def _clean_note_body(markdown_body: str, *, title: str) -> str:
    """Remove machine-looking wrapper headings from an LLM note body."""

    lines = markdown_body.strip().splitlines()
    cleaned: list[str] = []
    skip_next_blank = False
    forbidden_headings = {
        "整理笔记",
        "原始摘录",
        "值得保留的原文摘录",
    }
    for line in lines:
        stripped = line.strip()
        normalized_heading = stripped.lstrip("#").strip()
        if stripped.startswith("# "):
            if normalized_heading == title or normalized_heading in forbidden_headings:
                skip_next_blank = True
                continue
        if stripped.startswith("## ") and normalized_heading in forbidden_headings:
            skip_next_blank = True
            continue
        if skip_next_blank and not stripped:
            continue
        skip_next_blank = False
        cleaned.append(line.rstrip())

    body = "\n".join(cleaned).strip()
    return normalize_generated_markdown(body or "## 核心结论\n\n这条笔记需要继续整理。")


def _summary_not_repeated(summary: str, body: str) -> bool:
    normalized_summary = " ".join(summary.split()).lower()
    normalized_body = " ".join(body.split()).lower()
    if len(normalized_summary) < 24:
        return normalized_summary not in normalized_body
    return normalized_summary[:80] not in normalized_body


def _looks_like_empty_input_response(markdown_body: str, *, raw_text: str) -> bool:
    if not raw_text.strip():
        return False
    normalized = markdown_body.lower()
    empty_markers = (
        "内容为空",
        "无可用内容",
        "no content was present",
        "input was empty",
        "entirely empty",
    )
    return any(marker in normalized for marker in empty_markers)


def _resolve_note_path(notes_dir: Path, payload: dict[str, Any]) -> Path:
    if payload["action"] == "merge" and payload["target_note"]:
        candidate = _safe_relative_path(payload["target_note"])
        if candidate.suffix != ".md":
            candidate = candidate.with_suffix(".md")
        return notes_dir / candidate
    category = slugify(payload["category"])
    title = slugify(payload["title"])
    return notes_dir / category / f"{title}.md"


def _html_path_for_note(note_path: Path, *, notes_dir: Path, html_dir: Path) -> Path:
    relative = note_path.relative_to(notes_dir)
    return html_dir / relative.with_suffix(".html")


def _write_html_index(html_dir: Path, notes_dir: Path) -> None:
    links = []
    for html_path in sorted(html_dir.rglob("*.html")):
        if html_path.name == "index.html":
            continue
        relative = html_path.relative_to(html_dir)
        links.append(f'<li><a href="{html.escape(relative.as_posix())}">{html.escape(relative.with_suffix("").as_posix())}</a></li>')
    content = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>ClipWiki Index</title>",
            "  <style>body{font-family:system-ui,sans-serif;max-width:860px;margin:48px auto;line-height:1.7;padding:0 20px}a{color:#4169e1;text-decoration:none}li{margin:.4rem 0}</style>",
            "</head>",
            "<body>",
            "  <h1>ClipWiki Index</h1>",
            f"  <p>Markdown source: <code>{html.escape(str(notes_dir))}</code></p>",
            "  <ul>",
            *[f"    {link}" for link in links],
            "  </ul>",
            "</body>",
            "</html>",
        ]
    )
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "index.html").write_text(content + "\n", encoding="utf-8")


def _markdown_to_html(markdown: str) -> str:
    return markdown_lib.markdown(
        markdown,
        extensions=[
            "extra",
            "pymdownx.arithmatex",
            "pymdownx.superfences",
            "pymdownx.highlight",
            "tables",
            "sane_lists",
            "toc",
            "nl2br",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,
            },
            "pymdownx.highlight": {
                "use_pygments": False,
            }
        },
        output_format="html5",
    )


def _markdown_without_top_title(markdown: str, *, title: str) -> str:
    """Remove the first H1 when the HTML template already renders the page title."""

    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.strip() == f"# {title}":
            next_index = index + 1
            if next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            return "\n".join(lines[next_index:]).strip()
        return markdown
    return markdown


def _existing_note_summaries(notes_dir: Path) -> list[dict[str, str]]:
    if not notes_dir.exists():
        return []
    notes = []
    for path in sorted(notes_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        notes.append(
            {
                "path": path.relative_to(notes_dir).as_posix(),
                "title": _title_from_markdown(text) or path.stem,
                "excerpt": " ".join(text.split())[:360],
                "content": text,
            }
        )
    return notes


def _is_disposable_empty_note(markdown: str) -> bool:
    normalized = markdown.lower()
    if not any(
        marker in normalized
        for marker in (
            "空的复制ai输出",
            "copied ai output from the web chat was entirely empty",
            "no content was present",
            "unprocessed",
        )
    ):
        return False
    if "```text\n\n```" in markdown or "```text\r\n\r\n```" in markdown:
        return True
    return "提供的复制AI输出内容为空" in markdown or "无可用内容整理" in markdown


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    stop_at = stop_at.resolve()
    current = path
    while current.exists() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return None


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _preferred_note_path(paths: list[Path], *, notes_dir: Path) -> Path:
    """Prefer categorized notes over root-level legacy notes."""

    return sorted(paths, key=lambda path: (len(path.relative_to(notes_dir).parts) == 1, len(path.relative_to(notes_dir).parts), str(path)))[0]


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
    return Path(*clean_parts) if clean_parts else Path("general") / "untitled.md"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:80]
    return ""
