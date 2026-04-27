"""Command line interface for standalone ClipWiki."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clipwiki.core import ask_wiki, build_wiki_from_directory, load_wiki, search_wiki
from clipwiki.ingest import ingest_web_ai_result
from clipwiki.wiki_graph import build_link_graph, resolve_wikilink
from clipwiki.wiki_maintenance import build_lint_report, read_change_log, refresh_wiki_maintenance

app = typer.Typer(help="Maintain and query local ClipWiki notes.")


def _resolve_user_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _relative_note_arg(note: str, notes_dir: Path) -> str:
    note_path = Path(note).expanduser()
    if note_path.is_absolute():
        try:
            return note_path.resolve().relative_to(notes_dir.resolve()).as_posix()
        except ValueError:
            return note_path.as_posix()
    return note_path.as_posix()


@app.command("ingest")
def ingest_command(
    source: str,
    notes: str = typer.Option("clipwiki-notes", "--notes"),
    html: str = typer.Option("clipwiki-html", "--html"),
    question: str | None = typer.Option(None, "--question"),
    category: str | None = typer.Option(None, "--category"),
    title: str | None = typer.Option(None, "--title"),
    top_k: int = typer.Option(5, "--top-k", min=1),
    dry_run: bool = typer.Option(False, "--dry-run"),
    model: str | None = typer.Option(None, "--model"),
    cheap_model: str | None = typer.Option(None, "--cheap-model"),
    strong_model: str | None = typer.Option(None, "--strong-model"),
    api_key: str | None = typer.Option(None, "--api-key"),
    base_url: str | None = typer.Option(None, "--base-url"),
) -> None:
    notes_dir = _resolve_user_path(notes)
    html_dir = _resolve_user_path(html)
    result = ingest_web_ai_result(
        _resolve_user_path(source),
        notes_dir=notes_dir,
        html_dir=html_dir,
        user_question=question,
        category_hint=category,
        title_hint=title,
        top_k=top_k,
        dry_run=dry_run,
        model=model,
        cheap_model=cheap_model,
        strong_model=strong_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=notes_dir / ".clipwiki" / "llm-artifacts",
    )
    table_title = "ClipWiki Ingest Dry Run" if result.status == "dry_run" else "ClipWiki Ingest Complete"
    if result.status == "skipped":
        table_title = "ClipWiki Ingest Skipped"
    if result.status == "failed":
        table_title = "ClipWiki Ingest Failed"
    table = Table(title=table_title)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Status", result.status)
    table.add_row("Action", result.action)
    table.add_row("Reason", result.reason or "-")
    table.add_row("Title", result.title)
    table.add_row("Category", result.category)
    table.add_row("Markdown note", str(result.note_path) if result.note_path is not None else "not written")
    table.add_row("HTML page", str(result.html_path) if result.html_path is not None else "not written")
    table.add_row("HTML index", str(html_dir / "index.html") if result.html_path is not None else "not written")
    table.add_row("Tags", ", ".join(result.tags) if result.tags else "-")
    table.add_row("Pruned empty notes", str(result.pruned_empty_notes))
    table.add_row("Pruned duplicate notes", str(result.pruned_duplicate_notes))
    table.add_row("Pruned orphan HTML pages", str(result.pruned_orphan_html_pages))
    if result.validation_issues:
        table.add_row("Validation issues", "; ".join(result.validation_issues))
    base_model = model or os.getenv("LLM_MODEL") or "-"
    table.add_row("LLM model", base_model)
    table.add_row("Cheap model", cheap_model or base_model)
    table.add_row("Strong model", strong_model or base_model)
    table.add_row("LLM backend calls", str(result.llm_backend_calls))
    table.add_row("LLM cached", str(result.llm_cached))
    table.add_row("LLM input tokens", str(result.llm_input_tokens))
    table.add_row("LLM output tokens", str(result.llm_output_tokens))
    table.add_row("LLM total tokens", str(result.llm_total_tokens))
    table.add_row("LLM estimated cost", f"${result.llm_estimated_cost_usd:.6f}")
    table.add_row("LLM artifact", result.artifact_path or "-")
    Console().print(table)


@app.command("build")
def build_command(source_dir: str, wiki: str = typer.Option("clipwiki-vault", "--wiki"), question: str = typer.Option("What should this wiki remember?", "--question")) -> None:
    compiled = build_wiki_from_directory(_resolve_user_path(source_dir), _resolve_user_path(wiki), question=question)
    table = Table(title="ClipWiki Built")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Wiki dir", str(_resolve_user_path(wiki)))
    table.add_row("Pages", str(compiled.wiki_size_pages))
    table.add_row("Tokens", str(compiled.wiki_size_tokens))
    Console().print(table)


@app.command("search")
def search_command(query: str, wiki: str = typer.Option("clipwiki-vault", "--wiki"), top_k: int = typer.Option(5, "--top-k", min=1)) -> None:
    wiki_dir = _resolve_user_path(wiki)
    compiled = load_wiki(wiki_dir)
    hits = search_wiki(query, compiled, top_k=top_k, wiki_dir=wiki_dir)
    table = Table(title="ClipWiki Search")
    table.add_column("Rank")
    table.add_column("Page")
    table.add_column("Type")
    table.add_column("Score")
    table.add_column("Snippet")
    for index, hit in enumerate(hits, start=1):
        table.add_row(str(index), hit.page_id, hit.page_type, f"{hit.score:.3f}", " ".join(hit.text.split())[:120])
    Console().print(table)


@app.command("ask")
def ask_command(question: str, wiki: str = typer.Option("clipwiki-vault", "--wiki"), top_k: int = typer.Option(5, "--top-k", min=1)) -> None:
    wiki_dir = _resolve_user_path(wiki)
    compiled = load_wiki(wiki_dir)
    answer = ask_wiki(question, compiled, top_k=top_k, wiki_dir=wiki_dir)
    table = Table(title="ClipWiki Answer")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Question", question)
    table.add_row("Answer", answer.answer_text)
    table.add_row("Citations", ", ".join(answer.citations) if answer.citations else "-")
    Console().print(table)


@app.command("reindex")
def reindex_command(notes: str = typer.Option("clipwiki-notes", "--notes")) -> None:
    notes_dir = _resolve_user_path(notes)
    artifacts = refresh_wiki_maintenance(notes_dir)
    table = Table(title="ClipWiki Maintenance Rebuilt")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Notes", str(notes_dir))
    table.add_row("Wiki index", str(artifacts.wiki_index_path))
    table.add_row("Link graph", str(artifacts.link_graph_path))
    table.add_row("Lint report", str(artifacts.lint_report_path))
    table.add_row("Pages checked", str(artifacts.pages_checked))
    table.add_row("Issues", str(artifacts.issue_count))
    Console().print(table)


@app.command("lint")
def lint_command(
    notes: str = typer.Option("clipwiki-notes", "--notes"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    notes_dir = _resolve_user_path(notes)
    report = build_lint_report(notes_dir)
    if json_output:
        Console().print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    table = Table(title="ClipWiki Lint")
    table.add_column("Type")
    table.add_column("Path")
    table.add_column("Message")
    for issue in report.issues:
        table.add_row(issue.type, issue.path or "-", issue.message)
    if not report.issues:
        table.add_row("ok", "-", "No issues found")
    Console().print(table)


@app.command("links")
def links_command(note: str, notes: str = typer.Option("clipwiki-notes", "--notes")) -> None:
    notes_dir = _resolve_user_path(notes)
    graph = build_link_graph(notes_dir)
    relative_note = _relative_note_arg(note, notes_dir)
    resolved_note = relative_note if relative_note in graph.pages else resolve_wikilink(notes_dir, relative_note)
    if resolved_note is None or resolved_note not in graph.pages:
        Console().print(f"Page not found: {note}")
        raise typer.Exit(code=1)

    page = graph.pages[resolved_note]
    table = Table(title=f"ClipWiki Links: {resolved_note}")
    table.add_column("Direction")
    table.add_column("Target")
    if page.outbound:
        for target in page.outbound:
            table.add_row("outbound", target)
    else:
        table.add_row("outbound", "(none)")
    if page.inbound:
        for source in page.inbound:
            table.add_row("inbound", source)
    else:
        table.add_row("inbound", "(none)")
    Console().print(table)


@app.command("log")
def log_command(
    notes: str = typer.Option("clipwiki-notes", "--notes"),
    last: int = typer.Option(10, "--last", min=1),
) -> None:
    notes_dir = _resolve_user_path(notes)
    entries = read_change_log(notes_dir, last=last)
    if not entries:
        Console().print("No ClipWiki change log entries found.")
        return
    Console().print("\n\n".join(entries))
