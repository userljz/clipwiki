"""Incremental ClipWiki note maintenance pipeline."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipwiki.compiler import slugify
from clipwiki.markdown_sections import (
    MarkdownSection,
    find_section,
    insert_section_under_heading,
    neighboring_context,
    replace_section,
    title_from_markdown,
)
from clipwiki.note_index import (
    build_note_index,
    candidate_sections_prompt,
    load_or_build_note_index,
    render_outline,
    retrieve_candidate_sections,
)
from clipwiki.markdown_normalization import normalize_generated_markdown
from clipwiki.prompts import render_prompt
from clipwiki.validation import validate_updated_section
from clipwiki.schemas import TokenUsage
from clipwiki.llm import LiteLLMRuntime
from clipwiki.tokens import estimate_text_tokens


EXTRACT_CHUNK_TOKEN_BUDGET = 4000
PLANNING_CONTEXT_CHAR_BUDGET = 6000
MAX_KNOWLEDGE_UNITS = 120


@dataclass(slots=True)
class LLMCallStats:
    """Aggregate LLM usage across the incremental pipeline."""

    backend_calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    artifact_paths: list[str] = field(default_factory=list)

    @property
    def any_cached(self) -> bool:
        return self.cached_calls > 0

    def add(self, usage: TokenUsage | None, metadata: dict[str, Any]) -> None:
        cached = bool(metadata.get("cached", False))
        if cached:
            self.cached_calls += 1
        else:
            self.backend_calls += 1
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.total_tokens += usage.total_tokens
            self.estimated_cost_usd += usage.estimated_cost_usd
        artifact = metadata.get("artifact_path")
        if artifact:
            self.artifact_paths.append(str(artifact))


@dataclass(slots=True)
class IncrementalIngestResult:
    """Result from the section-level incremental ingest pipeline."""

    status: str
    decision: str
    reason: str
    note_path: Path | None = None
    title: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    markdown: str | None = None
    diff: str | None = None
    validation_issues: list[str] = field(default_factory=list)
    pruned_empty_notes: int = 0
    llm_stats: LLMCallStats = field(default_factory=LLMCallStats)
    knowledge_units: list[dict[str, Any]] = field(default_factory=list)


def run_incremental_ingest(
    *,
    raw_content: str,
    source_path: Path,
    notes_root: Path,
    user_question: str | None = None,
    category_hint: str | None = None,
    title_hint: str | None = None,
    model: str | None = None,
    cheap_model: str | None = None,
    strong_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    artifact_dir: Path | None = None,
    top_k: int = 5,
    dry_run: bool = False,
) -> IncrementalIngestResult:
    """Run the strict incremental note maintenance pipeline."""

    cleaned_content = clean_ingest_content(raw_content)
    if not cleaned_content.strip():
        return IncrementalIngestResult(status="skipped", decision="skip", reason="Input file is empty")

    cheap_stage_model = cheap_model or model
    strong_stage_model = strong_model or model
    llm_stats = LLMCallStats()
    language_instruction = _language_instruction(cleaned_content)
    note_profile = _classify_note_profile(
        cleaned_content=cleaned_content,
        user_question=user_question or "",
        language_instruction=language_instruction,
        stats=llm_stats,
        model=strong_stage_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    detail_instruction = _detail_instruction(cleaned_content, note_profile)
    extract_payload = _extract_knowledge_units(
        cleaned_content=cleaned_content,
        user_question=user_question or "",
        language_instruction=language_instruction,
        detail_instruction=detail_instruction,
        stats=llm_stats,
        model=cheap_stage_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    knowledge_units = _deduplicate_knowledge_units(
        [
            *[
                unit
                for unit in _list(extract_payload.get("knowledge_units"))
                if isinstance(unit, dict) and bool(unit.get("should_keep", True))
            ],
            *(_heuristic_research_question_units(cleaned_content) if _is_research_deep_dive(note_profile) else []),
            *(_heuristic_research_detail_units(cleaned_content) if _is_research_deep_dive(note_profile) else []),
        ]
    )
    if not knowledge_units:
        return IncrementalIngestResult(
            status="skipped",
            decision="skip",
            reason=str(extract_payload.get("content_summary") or "No durable knowledge units found"),
            llm_stats=llm_stats,
        )

    index_payload = load_or_build_note_index(notes_root)
    candidates = retrieve_candidate_sections(index_payload, knowledge_units, top_k=top_k)
    plan_payload = _complete_json(
        task_name="clipwiki-plan-note-update",
        prompt=render_prompt(
            "plan_note_update.md",
            note_outline=render_outline(index_payload),
            candidate_sections=candidate_sections_prompt(candidates),
            raw_content=_planning_source_context(cleaned_content, str(extract_payload.get("content_summary") or "")),
            user_question=user_question or "",
            knowledge_units_json=_json(knowledge_units),
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
        ),
        stats=llm_stats,
        model=strong_stage_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    plan = _normalize_plan(plan_payload, category_hint=category_hint, title_hint=title_hint, knowledge_units=knowledge_units)
    if plan["decision"] == "skip" or bool(plan.get("duplicate_check", {}).get("is_duplicate")):
        return IncrementalIngestResult(
            status="skipped",
            decision="skip",
            reason=str(plan.get("reason") or plan.get("duplicate_check", {}).get("explanation") or "Already covered"),
            llm_stats=llm_stats,
            knowledge_units=knowledge_units,
        )
    if plan.get("edit_level") == "level_3_global_restructure" or plan["decision"] == "split_or_reorganize":
        return IncrementalIngestResult(
            status="skipped",
            decision="split_or_reorganize",
            reason=str(plan.get("reason") or "Global restructure is not executed during normal ingest"),
            llm_stats=llm_stats,
            knowledge_units=knowledge_units,
        )

    target = _resolve_target(notes_root, plan, knowledge_units, title_hint=title_hint, category_hint=category_hint)
    original_markdown = target.file_path.read_text(encoding="utf-8") if target.file_path.exists() else ""
    original_section, neighbor_context = _target_section_context(original_markdown, target)
    editor_payload = _complete_json(
        task_name="clipwiki-edit-section",
        prompt=render_prompt(
            "edit_section.md",
            neighbor_context=neighbor_context,
            target_section=original_section.content,
            edit_plan_json=_json(plan),
            knowledge_units_json=_json(knowledge_units),
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
        ),
        stats=llm_stats,
        model=strong_stage_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    updated_section = _ensure_section_heading(
        str(editor_payload.get("updated_section_markdown") or original_section.content),
        original_section,
    )
    updated_section = normalize_generated_markdown(updated_section)
    if editor_payload.get("changed") is False or updated_section.strip() == original_section.content.strip():
        return IncrementalIngestResult(
            status="skipped",
            decision=plan["decision"],
            reason=str(editor_payload.get("reason") or "No section changes needed"),
            note_path=target.file_path,
            llm_stats=llm_stats,
            knowledge_units=knowledge_units,
        )

    local_validation = validate_updated_section(
        original_section.content,
        updated_section,
        expected_heading_path=original_section.heading_path,
        check_append=target.existed and plan["decision"] == "update_existing_section",
    )
    if not local_validation.valid:
        retry_payload = _retry_edit_section(
            original_section=original_section,
            neighbor_context=neighbor_context,
            plan=plan,
            knowledge_units=knowledge_units,
            issues=local_validation.issues,
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
            stats=llm_stats,
            model=strong_stage_model,
            api_key=api_key,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )
        updated_section = _ensure_section_heading(
            str(retry_payload.get("updated_section_markdown") or updated_section),
            original_section,
        )
        updated_section = normalize_generated_markdown(updated_section)
        local_validation = validate_updated_section(
            original_section.content,
            updated_section,
            expected_heading_path=original_section.heading_path,
            check_append=target.existed and plan["decision"] == "update_existing_section",
        )
        if not local_validation.valid:
            return IncrementalIngestResult(
                status="failed",
                decision=plan["decision"],
                reason="Local validation failed",
                note_path=target.file_path,
                validation_issues=local_validation.issues,
                llm_stats=llm_stats,
                knowledge_units=knowledge_units,
            )

    validator_payload = _complete_json(
        task_name="clipwiki-validate-note-update",
        prompt=render_prompt(
            "validate_note_update.md",
            original_section=original_section.content,
            updated_section=updated_section,
            edit_plan_json=_json(plan),
            knowledge_units_json=_json(knowledge_units),
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
        ),
        stats=llm_stats,
        model=cheap_stage_model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    if validator_payload.get("valid") is False:
        issues = [
            str(issue.get("description") or issue)
            for issue in _list(validator_payload.get("issues"))
        ]
        retry_payload = _retry_edit_section(
            original_section=original_section,
            neighbor_context=neighbor_context,
            plan=plan,
            knowledge_units=knowledge_units,
            issues=issues,
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
            stats=llm_stats,
            model=strong_stage_model,
            api_key=api_key,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )
        updated_section = _ensure_section_heading(
            str(retry_payload.get("updated_section_markdown") or updated_section),
            original_section,
        )
        updated_section = normalize_generated_markdown(updated_section)
        local_validation = validate_updated_section(
            original_section.content,
            updated_section,
            expected_heading_path=original_section.heading_path,
            check_append=target.existed and plan["decision"] == "update_existing_section",
        )
        if not local_validation.valid:
            return IncrementalIngestResult(
                status="failed",
                decision=plan["decision"],
                reason="Retry local validation failed",
                note_path=target.file_path,
                validation_issues=local_validation.issues,
                llm_stats=llm_stats,
                knowledge_units=knowledge_units,
            )
        validator_payload = _complete_json(
            task_name="clipwiki-validate-note-update-retry",
            prompt=render_prompt(
                "validate_note_update.md",
                original_section=original_section.content,
                updated_section=updated_section,
                edit_plan_json=_json(plan),
                knowledge_units_json=_json(knowledge_units),
                language_instruction=language_instruction,
                detail_instruction=detail_instruction,
            ),
            stats=llm_stats,
            model=cheap_stage_model,
            api_key=api_key,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )
        if validator_payload.get("valid") is False:
            retry_issues = [
                str(issue.get("description") or issue)
                for issue in _list(validator_payload.get("issues"))
            ]
            return IncrementalIngestResult(
                status="failed",
                decision=plan["decision"],
                reason=str(validator_payload.get("reason") or "LLM validation failed after retry"),
                note_path=target.file_path,
                validation_issues=retry_issues,
                llm_stats=llm_stats,
                knowledge_units=knowledge_units,
            )

    new_markdown = _apply_target_update(original_markdown, original_section, updated_section, target)
    diff = _diff(original_markdown, new_markdown, fromfile=f"{target.file_path} (before)", tofile=f"{target.file_path} (after)")
    title = title_from_markdown(new_markdown, fallback=target.file_path.stem)
    if dry_run:
        return IncrementalIngestResult(
            status="dry_run",
            decision=plan["decision"],
            reason=str(plan.get("reason", "")),
            note_path=target.file_path,
            title=title,
            category=target.file_path.parent.name,
            tags=_tags_from_units(knowledge_units),
            markdown=new_markdown,
            diff=diff,
            llm_stats=llm_stats,
            knowledge_units=knowledge_units,
        )

    target.file_path.parent.mkdir(parents=True, exist_ok=True)
    target.file_path.write_text(new_markdown, encoding="utf-8")
    build_note_index(notes_root)
    return IncrementalIngestResult(
        status="updated" if target.existed else "created",
        decision=plan["decision"],
        reason=str(plan.get("reason", "")),
        note_path=target.file_path,
        title=title,
        category=target.file_path.parent.name,
        tags=_tags_from_units(knowledge_units),
        markdown=new_markdown,
        diff=diff,
        llm_stats=llm_stats,
        knowledge_units=knowledge_units,
    )


def clean_ingest_content(raw_content: str) -> str:
    """Clean copied AI content without losing code blocks or useful lists."""

    lines = raw_content.replace("\r\n", "\n").splitlines()
    cleaned: list[str] = []
    blank_seen = False
    noise_prefixes = ("share", "copy", "regenerate", "thumbs up", "thumbs down")
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            cleaned.append(line.rstrip())
            blank_seen = False
            continue
        if not in_code and stripped.lower() in noise_prefixes:
            continue
        if not stripped:
            if not blank_seen:
                cleaned.append("")
            blank_seen = True
            continue
        cleaned.append(line.rstrip())
        blank_seen = False
    return "\n".join(cleaned).strip()


def _extract_knowledge_units(
    *,
    cleaned_content: str,
    user_question: str,
    language_instruction: str,
    detail_instruction: str,
    stats: LLMCallStats,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    chunks = _content_chunks(cleaned_content)
    payloads: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            chunk_header = f"[Chunk {index} of {len(chunks)}]\n\n{chunk}"
            task_name = f"clipwiki-extract-knowledge-units-{index:02d}"
        else:
            chunk_header = chunk
            task_name = "clipwiki-extract-knowledge-units"
        payloads.append(
            _complete_json(
                task_name=task_name,
                prompt=render_prompt(
                    "extract_knowledge_units.md",
                    user_question=user_question,
                    raw_content=chunk_header,
                    language_instruction=language_instruction,
                    detail_instruction=detail_instruction,
                ),
                stats=stats,
                model=model,
                api_key=api_key,
                base_url=base_url,
                artifact_dir=artifact_dir,
            )
        )

    summaries = [str(payload.get("content_summary", "")).strip() for payload in payloads if str(payload.get("content_summary", "")).strip()]
    units = _deduplicate_knowledge_units(
        [
            unit
            for payload in payloads
            for unit in _list(payload.get("knowledge_units"))
            if isinstance(unit, dict)
        ]
    )
    discarded = [
        item
        for payload in payloads
        for item in _list(payload.get("discarded_content"))
    ]
    return {
        "content_summary": " / ".join(summaries) if summaries else "Extracted durable knowledge from the source content.",
        "knowledge_units": units,
        "discarded_content": discarded,
    }


def _classify_note_profile(
    *,
    cleaned_content: str,
    user_question: str,
    language_instruction: str,
    stats: LLMCallStats,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    payload = _complete_json(
        task_name="clipwiki-classify-note-mode",
        prompt=render_prompt(
            "classify_note_mode.md",
            user_question=user_question,
            raw_content=_classification_source_context(cleaned_content),
            language_instruction=language_instruction,
        ),
        stats=stats,
        model=model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    mode = str(payload.get("note_mode") or "").strip()
    if mode not in {"research_deep_dive", "troubleshooting_runbook", "howto_reference", "lightweight_memory"}:
        mode = _heuristic_note_mode(cleaned_content)
    return {
        "note_mode": mode,
        "should_preserve_source_details": bool(payload.get("should_preserve_source_details", mode == "research_deep_dive")),
        "detail_policy": str(payload.get("detail_policy") or ""),
        "reason": str(payload.get("reason") or ""),
    }


def _classification_source_context(content: str, *, char_budget: int = 12000) -> str:
    if len(content) <= char_budget:
        return content
    head_budget = char_budget // 2
    tail_budget = char_budget - head_budget
    return "\n\n".join(
        [
            "<SOURCE_HEAD>",
            content[:head_budget].strip(),
            "<SOURCE_TAIL>",
            content[-tail_budget:].strip(),
        ]
    )


def _heuristic_note_mode(content: str) -> str:
    lowered = content.lower()
    troubleshooting_markers = ("traceback", "modulenotfounderror", "ssl", "pip install", "apt-get", "command not found", "报错", "怎么解决")
    research_markers = ("proposal", "ablation", "baseline", "reviewer", "主管", "评审", "消融", "实验协议", "研究")
    if any(marker in lowered for marker in troubleshooting_markers):
        return "troubleshooting_runbook"
    if any(marker in lowered for marker in research_markers):
        return "research_deep_dive"
    return "howto_reference"


def _is_research_deep_dive(note_profile: dict[str, Any]) -> bool:
    return str(note_profile.get("note_mode")) == "research_deep_dive" and bool(note_profile.get("should_preserve_source_details", True))


def _content_chunks(text: str, *, token_budget: int = EXTRACT_CHUNK_TOKEN_BUDGET) -> list[str]:
    if estimate_text_tokens(text) <= token_budget:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for segment in _iter_segments(text, token_budget=token_budget):
        segment_tokens = estimate_text_tokens(segment)
        if current and current_tokens + segment_tokens > token_budget:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_tokens = 0
        current.append(segment)
        current_tokens += segment_tokens
    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk.strip()] or [text]


def _iter_segments(text: str, *, token_budget: int) -> list[str]:
    segments: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if estimate_text_tokens(paragraph) <= token_budget:
            segments.append(paragraph)
            continue
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1:
            for line in lines:
                segments.extend(_split_long_segment(line, token_budget=token_budget))
        else:
            segments.extend(_split_long_segment(paragraph, token_budget=token_budget))
    return [segment for segment in segments if segment.strip()]


def _split_long_segment(segment: str, *, token_budget: int) -> list[str]:
    if estimate_text_tokens(segment) <= token_budget:
        return [segment]
    char_budget = max(1200, token_budget)
    return [segment[index : index + char_budget].strip() for index in range(0, len(segment), char_budget)]


def _deduplicate_knowledge_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        claim = " ".join(str(unit.get("claim", "")).split())
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(unit)
        normalized["claim"] = claim
        normalized["id"] = f"ku_{len(deduped) + 1}"
        deduped.append(normalized)
        if len(deduped) >= MAX_KNOWLEDGE_UNITS:
            break
    return deduped


def _heuristic_research_question_units(content: str, *, limit: int = 40) -> list[dict[str, Any]]:
    """Preserve explicit research/review questions even if the extractor summarizes them away."""

    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in content.splitlines():
        question = _clean_question_line(line)
        if question is None:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        units.append(
            {
                "id": f"heuristic_question_{len(units) + 1}",
                "claim": question,
                "type": "checklist",
                "keywords": ["question", "review", "checklist"],
                "possible_topics": ["研究评审问题清单"],
                "detail_role": "question_checklist",
                "preserve_as": "checklist",
                "should_keep": True,
                "novelty_hint": "new",
                "reason": "原文中的显式研究/评审问题，应作为可复用检查清单保留。",
            }
        )
        if len(units) >= limit:
            break
    return units


def _heuristic_research_detail_units(content: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Preserve original examples and technical detail snippets verbatim."""

    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for detail_role, preserve_as, snippet in _iter_research_detail_snippets(content):
        normalized = " ".join(snippet.split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        units.append(
            {
                "id": f"heuristic_detail_{len(units) + 1}",
                "claim": f"原文细节（必须按原文保留，不要替换或自造）：\n{snippet.strip()}",
                "type": "example" if detail_role == "example" else "method",
                "keywords": ["original detail", detail_role, "verbatim"],
                "possible_topics": ["原文可复用细节"],
                "detail_role": detail_role,
                "preserve_as": preserve_as,
                "should_keep": True,
                "novelty_hint": "new",
                "reason": "这是原文中的具体例子、数据结构、模型结构或流程细节，笔记要能替代原文，必须保留。",
            }
        )
        if len(units) >= limit:
            break
    return units


def _iter_research_detail_snippets(content: str) -> list[tuple[str, str, str]]:
    lines = content.splitlines()
    snippets: list[tuple[str, str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in ("比如", "例如", "示例")) and len(stripped) <= 280:
            snippets.append(("example", "bullet", stripped))

    block_markers = [
        ("数据准备", "data_schema", "code_block"),
        ("原始数据格式", "data_schema", "code_block"),
        ("预处理步骤", "algorithm_step", "numbered_steps"),
        ("预处理结束后", "data_schema", "code_block"),
        ("模型结构", "architecture", "code_block"),
        ("推理流程", "inference_flow", "numbered_steps"),
        ("在线自进化", "algorithm_step", "bullet"),
    ]
    for marker, role, preserve_as in block_markers:
        block = _collect_block_after_marker(lines, marker)
        if block:
            snippets.append((role, preserve_as, block))

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if any(token in lowered for token in ("shape", "hidden states", "logprob", "teacher forcing", "input:", "output:", "mlp:", "transformer:", "pooling:", "d_z")):
            snippets.append(("implementation_constraint", "bullet", stripped))
    return snippets


def _collect_block_after_marker(lines: list[str], marker: str, *, max_lines: int = 28) -> str:
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        collected: list[str] = [line.strip()]
        for next_line in lines[index + 1 : index + 1 + max_lines]:
            stripped = next_line.rstrip()
            if not stripped and len(collected) > 1:
                break
            if _looks_like_top_level_cn_heading(stripped) and len(collected) > 1:
                break
            if stripped:
                collected.append(stripped)
        return "\n".join(collected).strip()
    return ""


def _looks_like_top_level_cn_heading(line: str) -> bool:
    return bool(line and any(line.startswith(prefix) for prefix in ("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、")))


def _clean_question_line(line: str) -> str | None:
    cleaned = line.strip()
    cleaned = cleaned.lstrip("-*•●○0123456789. ）)、\t ").strip()
    if not cleaned:
        return None
    if not (cleaned.endswith("?") or cleaned.endswith("？")):
        return None
    if len(cleaned) < 6 or len(cleaned) > 220:
        return None
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ("http://", "https://", "```")):
        return None
    return cleaned


def _planning_source_context(content: str, summary: str, *, char_budget: int = PLANNING_CONTEXT_CHAR_BUDGET) -> str:
    if len(content) <= char_budget:
        return content
    head_budget = char_budget // 3
    tail_budget = char_budget // 3
    middle_budget = char_budget - head_budget - tail_budget
    middle_start = max(0, (len(content) - middle_budget) // 2)
    parts = [
        "The full source was long and was extracted in chunks. Use KNOWLEDGE_UNITS as the source of truth.",
        f"Content summary: {summary}" if summary else "",
        "<SOURCE_EXCERPT_START>",
        content[:head_budget].strip(),
        "<SOURCE_EXCERPT_MIDDLE>",
        content[middle_start : middle_start + middle_budget].strip(),
        "<SOURCE_EXCERPT_END>",
        content[-tail_budget:].strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _language_instruction(content: str) -> str:
    cjk_count = sum(1 for character in content if "\u4e00" <= character <= "\u9fff")
    latin_count = sum(1 for character in content if character.isascii() and character.isalpha())
    if cjk_count >= 80 and cjk_count >= latin_count * 0.2:
        return (
            "The source is primarily Chinese or Chinese-led mixed technical discussion. "
            "Write all visible note prose in Simplified Chinese. Keep technical terms, paper names, model names, commands, "
            "metric names, and acronyms in their original English when appropriate."
        )
    return (
        "Write visible note prose in the source's primary language. Keep technical terms, paper names, model names, "
        "commands, metric names, and acronyms in their original language when appropriate."
    )


def _detail_instruction(content: str, note_profile: dict[str, Any]) -> str:
    mode = str(note_profile.get("note_mode") or _heuristic_note_mode(content))
    if mode == "research_deep_dive":
        return (
            "Research/detail preservation mode is ON. The note is a reusable research/design asset, not an executive summary. "
            "Preserve concrete examples, data schemas, tensor shapes, formulas, architecture components, hyperparameters, "
            "algorithm steps, evaluation protocols, ablation plans, baseline comparisons, reviewer/supervisor questions, "
            "risk checklists, rejected alternatives, final decisions, and the reasons behind those decisions. "
            "Question lists must remain as explicit checklists. Do not collapse them into generic risk bullets."
        )
    if mode == "troubleshooting_runbook":
        return (
            "Troubleshooting/runbook mode is ON. The note should be a concise reusable runbook, not a source transcript. "
            "Preserve error symptoms, root causes, final working commands, diagnostic commands, environment assumptions, "
            "paths only when they are needed as examples, and decision rules for what to try next. "
            "Do not preserve full tracebacks, repeated failed attempts, chat flow, or every original log line. "
            "Never write placeholders such as '此处保留...', '待补充', 'TODO', or 'placeholder'. If an exact command is missing, omit that bullet instead of inventing a placeholder."
        )
    if mode == "howto_reference":
        return (
            "How-to/reference mode is ON. Preserve final recommended steps, commands, parameter meanings, verification commands, "
            "applicability conditions, and common pitfalls. Do not keep redundant dialogue or one-off failed attempts. "
            "Never write placeholders such as '此处保留...', '待补充', 'TODO', or 'placeholder'."
        )
    return (
        "Lightweight memory mode is ON. Keep only durable reusable facts, decisions, caveats, and short procedures. "
        "Avoid source-transcript detail and never write placeholders."
    )


@dataclass(slots=True)
class _Target:
    file_path: Path
    heading_path: list[str]
    insert_position: str
    decision: str
    existed: bool


def _complete_json(
    *,
    task_name: str,
    prompt: str,
    stats: LLMCallStats,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    runtime = LiteLLMRuntime(
        task_name=task_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )
    parsed, usage, metadata = runtime.complete_json(prompt)
    stats.add(usage, metadata)
    return parsed


def _retry_edit_section(
    *,
    original_section: MarkdownSection,
    neighbor_context: str,
    plan: dict[str, Any],
    knowledge_units: list[dict[str, Any]],
    issues: list[str],
    language_instruction: str,
    detail_instruction: str,
    stats: LLMCallStats,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    retry_plan = {
        **plan,
        "validation_feedback": issues,
        "retry_instruction": (
            "Revise the target section to fix these validation issues. "
            "Return only the corrected full target section. Do not add raw chat, duplicate headings, duplicate paragraphs, or append-only chunks."
        ),
    }
    return _complete_json(
        task_name="clipwiki-edit-section-retry",
        prompt=render_prompt(
            "edit_section.md",
            neighbor_context=neighbor_context,
            target_section=original_section.content,
            edit_plan_json=_json(retry_plan),
            knowledge_units_json=_json(knowledge_units),
            language_instruction=language_instruction,
            detail_instruction=detail_instruction,
        ),
        stats=stats,
        model=model,
        api_key=api_key,
        base_url=base_url,
        artifact_dir=artifact_dir,
    )


def _normalize_plan(
    payload: dict[str, Any],
    *,
    category_hint: str | None,
    title_hint: str | None,
    knowledge_units: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = str(payload.get("decision") or "create_new_file")
    if decision not in {"skip", "update_existing_section", "create_new_section", "create_new_file", "split_or_reorganize"}:
        decision = "create_new_file"
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    heading_path = _string_list(target.get("heading_path"))
    if not heading_path:
        title = title_hint or _topic_from_units(knowledge_units) or "Untitled Note"
        heading_path = [title]
    file_path = str(target.get("file_path") or "")
    if not file_path:
        category = slugify(category_hint or _topic_from_units(knowledge_units) or "general")
        file_path = f"{category}/{slugify(heading_path[0])}.md"
    deduplication = payload.get("deduplication") if isinstance(payload.get("deduplication"), dict) else {}
    duplicate_check = payload.get("duplicate_check") if isinstance(payload.get("duplicate_check"), dict) else {}
    is_duplicate = bool(duplicate_check.get("is_duplicate", deduplication.get("is_fully_duplicate", False)))
    covered_by = duplicate_check.get("covered_by", deduplication.get("overlapping_sections", []))
    edit_plan = payload.get("edit_plan") if isinstance(payload.get("edit_plan"), dict) else {}
    edit_intent = payload.get("edit_intent") if isinstance(payload.get("edit_intent"), dict) else {}
    required_context = payload.get("required_context") if isinstance(payload.get("required_context"), dict) else {}
    context_needed = payload.get("context_needed_for_editing") if isinstance(payload.get("context_needed_for_editing"), dict) else {}
    operation = str(target.get("operation") or target.get("insert_position") or "inside_section")
    edit_level = str(payload.get("edit_level") or "level_1_target_section_only")
    if edit_level not in {"level_1_target_section_only", "level_2_nearby_sections_refactor", "level_3_global_restructure"}:
        edit_level = "level_1_target_section_only"
    insert_position = {
        "no_change": "inside_section",
        "replace_section": "inside_section",
        "insert_into_section": "inside_section",
        "add_subsection": "new_subsection_under_heading",
    }.get(operation, operation)
    return {
        **payload,
        "decision": decision,
        "edit_level": edit_level,
        "target": {
            "file_path": file_path,
            "heading_path": heading_path,
            "insert_position": insert_position,
            "operation": operation,
        },
        "duplicate_check": {
            "is_duplicate": is_duplicate,
            "covered_by": covered_by if isinstance(covered_by, list) else [],
            "explanation": str(duplicate_check.get("explanation", deduplication.get("explanation", ""))),
        },
        "edit_intent": {
            "summary": str(edit_intent.get("summary", edit_plan.get("main_goal", ""))),
            "knowledge_units_to_add": _string_list(edit_intent.get("knowledge_units_to_add")) or _string_list(edit_plan.get("points_to_add")),
            "knowledge_units_to_merge": _string_list(edit_intent.get("knowledge_units_to_merge")) or _string_list(edit_plan.get("points_to_merge")),
            "knowledge_units_to_ignore": _string_list(edit_intent.get("knowledge_units_to_ignore")) or _string_list(edit_plan.get("points_to_remove_or_avoid")),
            "expected_heading_changes": edit_intent.get("expected_heading_changes", edit_plan.get("suggested_structure", [])),
        },
        "required_context": {
            "need_full_file": bool(required_context.get("need_full_file", context_needed.get("need_full_file", False))),
            "need_sections": required_context.get("need_sections", context_needed.get("sections_to_load", [])),
            "max_sections_to_edit": int(context_needed.get("max_sections_to_edit", required_context.get("max_sections_to_edit", 1)) or 1),
        },
    }


def _resolve_target(
    notes_root: Path,
    plan: dict[str, Any],
    knowledge_units: list[dict[str, Any]],
    *,
    title_hint: str | None,
    category_hint: str | None,
) -> _Target:
    target = plan["target"]
    relative = _safe_relative_note_path(str(target.get("file_path") or ""))
    if not relative.parts:
        category = slugify(category_hint or _topic_from_units(knowledge_units) or "general")
        title = slugify(title_hint or _topic_from_units(knowledge_units) or "untitled")
        relative = Path(category) / f"{title}.md"
    return _Target(
        file_path=notes_root / relative,
        heading_path=_string_list(target.get("heading_path")) or [title_hint or _topic_from_units(knowledge_units) or "Untitled"],
        insert_position=str(target.get("insert_position") or "inside_section"),
        decision=str(plan.get("decision") or "create_new_file"),
        existed=(notes_root / relative).exists(),
    )


def _target_section_context(markdown: str, target: _Target) -> tuple[MarkdownSection, str]:
    if not markdown.strip():
        heading = target.heading_path[-1]
        level = max(1, min(len(target.heading_path), 6))
        prefix = "#" * level
        content = f"{prefix} {heading}\n"
        section = MarkdownSection(heading=heading, heading_path=target.heading_path, level=level, start_line=1, end_line=1, content=content)
        return section, ""

    section = find_section(markdown, target.heading_path)
    if section is not None:
        return section, neighboring_context(markdown, section)

    heading = target.heading_path[-1]
    level = max(1, min(len(target.heading_path), 6))
    prefix = "#" * level
    content = f"{prefix} {heading}\n"
    section = MarkdownSection(heading=heading, heading_path=target.heading_path, level=level, start_line=1, end_line=1, content=content)
    return section, markdown[:1200]


def _ensure_section_heading(updated: str, section: MarkdownSection) -> str:
    stripped = updated.strip()
    expected_prefix = "#" * section.level + " "
    if stripped.startswith(expected_prefix):
        return stripped + "\n"
    return f"{expected_prefix}{section.heading}\n\n{stripped}\n"


def _apply_target_update(markdown: str, original_section: MarkdownSection, updated_section: str, target: _Target) -> str:
    if not markdown.strip():
        return updated_section.strip() + "\n"
    existing = find_section(markdown, original_section.heading_path)
    if existing is not None and target.decision == "update_existing_section":
        return replace_section(markdown, existing, updated_section)
    parent_path = original_section.heading_path[:-1]
    parent = find_section(markdown, parent_path) if parent_path else None
    return insert_section_under_heading(markdown, parent, updated_section)


def _diff(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _safe_relative_note_path(value: str) -> Path:
    cleaned = value.removeprefix("notes/").lstrip("/")
    path = Path(cleaned)
    parts = [part for part in path.parts if part not in {"", ".", ".."}]
    if not parts:
        return Path()
    relative = Path(*parts)
    return relative if relative.suffix == ".md" else relative.with_suffix(".md")


def _topic_from_units(knowledge_units: list[dict[str, Any]]) -> str:
    for unit in knowledge_units:
        topics = _string_list(unit.get("possible_topics"))
        if topics:
            return topics[0]
    for unit in knowledge_units:
        claim = str(unit.get("claim", "")).strip()
        if claim:
            return claim[:60]
    return "general"


def _tags_from_units(knowledge_units: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for unit in knowledge_units:
        for keyword in _string_list(unit.get("keywords")):
            if keyword not in tags:
                tags.append(keyword)
    return tags[:12]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
