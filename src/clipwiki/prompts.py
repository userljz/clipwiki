"""Prompt template loading for ClipWiki incremental ingest."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).with_name("prompts")


def render_prompt(template_name: str, **values: object) -> str:
    """Render a prompt template using simple {{ name }} placeholders."""

    template_path = PROMPT_DIR / template_name
    template = template_path.read_text(encoding="utf-8")
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
