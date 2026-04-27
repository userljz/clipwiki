"""Markdown normalization helpers for generated ClipWiki notes."""

from __future__ import annotations

import re

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

GREEK_OR_TEX_COMMAND_RE = re.compile(
    r"(?<!\\)\((\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|omega|phi|varphi|psi|chi|frac|sqrt|sum|prod|int|log|exp|cdot|times|to|rightarrow|leftarrow|leq|geq|neq|approx)[^()]*)\)"
)
PAREN_VARIABLE_RE = re.compile(r"(?<!\\)\(([A-Za-z])\)")
PROTECTED_INLINE_RE = re.compile(r"(`[^`]*`|\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\))")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
UNORDERED_BULLET_RE = re.compile(r"^(\s*)([-*+]\s+)(.*)$")

MATH_VARIABLES = frozenset("pkmnqxyzabcdLNK")
MATH_CONTEXT_KEYWORDS = (
    "概率",
    "变量",
    "参数",
    "候选",
    "长度",
    "维度",
    "阈值",
    "权重",
    "分数",
    "评分",
    "得分",
    "公式",
    "记为",
    "表示",
    "其中",
    "令",
    "取值",
    "采样",
    "样本",
    "token",
    "top-",
    "Top-",
    "probability",
    "parameter",
    "variable",
    "score",
    "rank",
)


def normalize_generated_markdown(markdown: str) -> str:
    """Fix common generated Markdown patterns before validation/rendering."""

    normalized = _normalize_inline_math(markdown)
    return _normalize_parent_colon_lists(normalized)


def markdown_format_issues(markdown: str) -> list[str]:
    """Return deterministic Markdown-format issues that LLM validation may miss."""

    issues: list[str] = []
    issues.extend(_nonstandard_math_issues(markdown))
    issues.extend(_pymarkdown_lint_issues(markdown))
    issues.extend(_flat_parent_list_issues(markdown))
    return issues


def _normalize_inline_math(markdown: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in markdown.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue
        lines.append(_normalize_inline_math_line(line))
    return "".join(lines)


def _normalize_inline_math_line(line: str) -> str:
    parts = PROTECTED_INLINE_RE.split(line)
    for index, part in enumerate(parts):
        if not part or PROTECTED_INLINE_RE.fullmatch(part):
            continue
        part = GREEK_OR_TEX_COMMAND_RE.sub(r"$\1$", part)
        part = PAREN_VARIABLE_RE.sub(lambda match: _replace_parenthesized_variable(match, part), part)
        parts[index] = part
    return "".join(parts)


def _replace_parenthesized_variable(match: re.Match[str], text: str) -> str:
    variable = match.group(1)
    if variable not in MATH_VARIABLES:
        return match.group(0)
    window_start = max(0, match.start() - 24)
    window_end = min(len(text), match.end() + 24)
    window = text[window_start:window_end]
    if any(keyword in window for keyword in MATH_CONTEXT_KEYWORDS):
        return f"${variable}$"
    return match.group(0)


def _normalize_parent_colon_lists(markdown: str) -> str:
    lines = markdown.splitlines(keepends=True)
    normalized: list[str] = []
    in_fence = False
    active_parent_indent: str | None = None

    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            active_parent_indent = None
            normalized.append(line)
            continue

        if in_fence:
            normalized.append(line)
            continue

        if _is_parent_label_bullet(line):
            active_parent_indent = _bullet_indent(line)
            normalized.append(line)
            continue

        if active_parent_indent is not None:
            if not line.strip():
                active_parent_indent = None
            elif _is_same_indent_ordinary_bullet(line, active_parent_indent):
                # Python-Markdown renders nested list items reliably with four spaces.
                line = f"    {line}"
            elif _is_two_space_child_bullet(line, active_parent_indent):
                line = f"  {line}"
            elif not line.startswith(active_parent_indent + " "):
                active_parent_indent = None

        normalized.append(line)

    return "".join(normalized)


def _nonstandard_math_issues(markdown: str) -> list[str]:
    issues: list[str] = []
    in_fence = False
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for part in PROTECTED_INLINE_RE.split(line):
            if not part or PROTECTED_INLINE_RE.fullmatch(part):
                continue
            if GREEK_OR_TEX_COMMAND_RE.search(part):
                issues.append(f"nonstandard_math:line {line_number}:use $...$ for TeX math")
                break
            if _contains_contextual_parenthesized_variable(part):
                issues.append(f"nonstandard_math:line {line_number}:use $...$ for inline variables")
                break
    return issues


def _contains_contextual_parenthesized_variable(text: str) -> bool:
    return any(_replace_parenthesized_variable(match, text) != match.group(0) for match in PAREN_VARIABLE_RE.finditer(text))


def _pymarkdown_lint_issues(markdown: str) -> list[str]:
    try:
        scan_result = (
            PyMarkdownApi()
            .log_error_and_above()
            .disable_rule_by_identifier("all")
            .enable_rule_by_identifier("pml101")
            .scan_string(markdown if markdown.endswith("\n") else f"{markdown}\n")
        )
    except PyMarkdownApiException as exc:
        return [f"markdown_lint_error:{exc}"]

    return [
        (
            f"markdown_lint:{failure.rule_id}:line {failure.line_number}:"
            f"{failure.rule_description}{failure.extra_error_information}"
        )
        for failure in scan_result.scan_failures
    ]


def _flat_parent_list_issues(markdown: str) -> list[str]:
    issues: list[str] = []
    lines = markdown.splitlines()
    in_fence = False
    parent_indent: str | None = None
    parent_line_number: int | None = None

    for line_number, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            parent_indent = None
            parent_line_number = None
            continue
        if in_fence:
            continue
        if _is_parent_label_bullet(line):
            parent_indent = _bullet_indent(line)
            parent_line_number = line_number
            continue
        if parent_indent is None:
            continue
        if not line.strip():
            parent_indent = None
            parent_line_number = None
            continue
        if _is_same_indent_ordinary_bullet(line, parent_indent):
            issues.append(
                f"nested_list_indentation:line {line_number}:indent child bullet after parent bullet on line {parent_line_number}"
            )
            parent_indent = None
            parent_line_number = None
            continue
        if not line.startswith(parent_indent + " "):
            parent_indent = None
            parent_line_number = None

    return issues


def _is_parent_label_bullet(line: str) -> bool:
    match = UNORDERED_BULLET_RE.match(line.rstrip("\r\n"))
    if match is None:
        return False
    content = match.group(3).strip()
    return bool(re.fullmatch(r"(?:\*\*[^*]+\*\*|[^:：\n]{1,40})[:：]", content))


def _is_same_indent_ordinary_bullet(line: str, indent: str) -> bool:
    match = UNORDERED_BULLET_RE.match(line.rstrip("\r\n"))
    if match is None or match.group(1) != indent:
        return False
    return not _is_parent_label_bullet(line)


def _is_two_space_child_bullet(line: str, parent_indent: str) -> bool:
    match = UNORDERED_BULLET_RE.match(line.rstrip("\r\n"))
    if match is None:
        return False
    return match.group(1) == parent_indent + "  "


def _bullet_indent(line: str) -> str:
    match = UNORDERED_BULLET_RE.match(line)
    return match.group(1) if match else ""
