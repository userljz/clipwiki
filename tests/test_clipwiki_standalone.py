from pathlib import Path

from typer.testing import CliRunner

from clipwiki.cli import app
from clipwiki.ingest import ingest_web_ai_result, prune_orphan_html_pages, render_note_html
from clipwiki.incremental import clean_ingest_content, run_incremental_ingest
from clipwiki.llm import _actual_completion_cost_usd
from clipwiki.markdown_normalization import normalize_generated_markdown
from clipwiki.note_index import load_or_build_note_index, memory_dir, retrieve_candidate_sections
from clipwiki.schemas import TokenUsage
from clipwiki.validation import validate_updated_section
from clipwiki.wiki_graph import build_link_graph, extract_wikilinks
from clipwiki.wiki_maintenance import read_change_log, refresh_wiki_maintenance


def test_prune_orphan_html_pages(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    html = tmp_path / "html"
    orphan = html / "old" / "gone.html"
    valid_note = notes / "ok" / "page.md"
    valid_html = html / "ok" / "page.html"
    orphan.parent.mkdir(parents=True)
    valid_note.parent.mkdir(parents=True)
    valid_html.parent.mkdir(parents=True)
    orphan.write_text("old", encoding="utf-8")
    valid_note.write_text("# Page\n", encoding="utf-8")
    valid_html.write_text("ok", encoding="utf-8")

    assert prune_orphan_html_pages(notes, html_dir=html) == 1
    assert not orphan.exists()
    assert valid_html.exists()


def test_chinese_retrieval(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    note = notes / "agent" / "gui.md"
    note.parent.mkdir(parents=True)
    note.write_text("# GUI 代理\n\n## 技能化\n\nGUI Agent 可以学习技能形式的多动作输出，并用 guard 控制风险。", encoding="utf-8")
    payload = load_or_build_note_index(notes)
    hits = retrieve_candidate_sections(payload, [{"claim": "GUI Agent 使用技能进行多动作预测", "keywords": ["技能", "多动作"], "possible_topics": ["GUI 代理"]}])
    assert hits


def test_actual_cost_prefers_provider_usage_cost() -> None:
    class Usage:
        prompt_tokens = 10
        completion_tokens = 5
        cost = 0.123

    class Response:
        usage = Usage()

    assert _actual_completion_cost_usd(Response()) == 0.123


def test_actual_cost_uses_hidden_response_cost() -> None:
    class Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class Response:
        usage = Usage()
        _hidden_params = {"response_cost": 0.456}

    assert _actual_completion_cost_usd(Response()) == 0.456


def test_clean_ingest_content_removes_chat_export_wrappers() -> None:
    raw = """You said: 你给我分析一下这个方法？
16:36
Claude responded: 结论是这个方法可以作为研究假设。
搜寻相关研究并评估创新性。
搜寻相关研究并评估创新性。
Show more
Claude is AI and can make mistakes. Please double-check responses.
"""

    cleaned = clean_ingest_content(raw)

    assert "You said:" not in cleaned
    assert "Claude responded:" not in cleaned
    assert "16:36" not in cleaned
    assert "Show more" not in cleaned
    assert "搜寻相关研究并评估创新性" not in cleaned
    assert "你给我分析一下这个方法？" in cleaned
    assert "结论是这个方法可以作为研究假设。" in cleaned


def test_render_note_html_uses_markdown_library_for_code_tables_and_math(tmp_path: Path) -> None:
    markdown = """# Demo

## Commands

```bash
python -m pip install python-dateutil
echo "$REQUESTS_CA_BUNDLE"
```

| Name | Value |
| --- | --- |
| alpha | $a+b$ |
"""

    rendered = render_note_html(markdown, title="Demo", category="test", note_path=tmp_path / "demo.md")

    assert "```bash" not in rendered
    assert "python -m pip install python-dateutil" in rendered
    assert "language-bash" in rendered
    assert "<table>" in rendered
    assert '<span class="arithmatex">\\(a+b\\)</span>' in rendered
    assert "tex-svg.js" in rendered


def test_normalize_generated_markdown_fixes_common_inline_math_patterns() -> None:
    markdown = "其中 (p) 为猜对概率，(\\alpha/\\beta) 分别为延迟。增大候选数 (k) 通常更快。\n- **方法**：\n  - 空间显著性：..."

    normalized = normalize_generated_markdown(markdown)

    assert "$p$ 为猜对概率" in normalized
    assert "$\\alpha/\\beta$" in normalized
    assert "候选数 $k$" in normalized
    assert "- **方法**：\n    - 空间显著性：..." in normalized


def test_normalize_generated_markdown_skips_code_when_fixing_math() -> None:
    markdown = """# Demo

概率 (p) 需要渲染。

`概率 (p)` should stay literal.

```text
概率 (p)
(\\alpha)
```
"""

    normalized = normalize_generated_markdown(markdown)

    assert "概率 $p$" in normalized
    assert "`概率 (p)`" in normalized
    assert "```text\n概率 (p)\n(\\alpha)\n```" in normalized


def test_render_note_html_normalizes_legacy_math(tmp_path: Path) -> None:
    markdown = """# Demo

概率 (p) 和比例 (\\alpha/\\beta) 应该作为公式渲染。

- **方法**：
- 空间显著性：先过滤 token。
- 时间冗余评分：再合并帧。
"""

    rendered = render_note_html(markdown, title="Demo", category="test", note_path=tmp_path / "demo.md")

    assert '<span class="arithmatex">\\(p\\)</span>' in rendered
    assert '<span class="arithmatex">\\(\\alpha/\\beta\\)</span>' in rendered
    assert "<li><strong>方法</strong>：<ul>" in rendered


def test_validation_rejects_markdown_math_and_list_format_issues() -> None:
    result = validate_updated_section(
        "",
        "# Demo\n\n概率 (p) 与比例 (\\alpha/\\beta)。\n\n- **方法**：\n- 空间显著性：先过滤 token。\n",
    )

    assert not result.valid
    assert any(issue.startswith("nonstandard_math") for issue in result.issues)
    assert any(issue.startswith("nested_list_indentation") for issue in result.issues)


def test_validation_uses_pymarkdown_for_python_markdown_list_indent() -> None:
    result = validate_updated_section(
        "",
        "# Demo\n\n- **方法**：\n  - 空间显著性：两空格在 Python-Markdown 里不会渲染为子列表。\n",
    )

    assert not result.valid
    assert any(issue.startswith("markdown_lint:PML101") for issue in result.issues)


def test_validation_does_not_block_on_nonsemantic_markdown_lint_rules() -> None:
    result = validate_updated_section(
        "",
        "# Demo\n## Heading Without Blank Line\n" + ("这是一行很长但语义正确的中文技术说明。" * 8) + "\n",
    )

    assert not any("MD013" in issue or "MD022" in issue or "MD032" in issue for issue in result.issues)


def test_incremental_ingest_routes_cheap_and_strong_models(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    prompts: list[str] = []
    progress: list[str] = []
    responses = [
        {
            "note_mode": "research_deep_dive",
            "should_preserve_source_details": True,
            "detail_policy": "Preserve research details.",
            "reason": "Research proposal.",
        },
        {
            "content_summary": "ClipWiki model routing.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ClipWiki can route cheap and strong models by pipeline stage.",
                    "type": "method",
                    "keywords": ["clipwiki", "model routing"],
                    "possible_topics": ["ClipWiki Model Routing"],
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Useful configuration behavior.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New model routing note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/model-routing.md", "heading_path": ["ClipWiki Model Routing"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# ClipWiki Model Routing\n\nCheap stages and strong stages are separated."},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        prompts.append(prompt)
        calls.append((self.task_name, self.model))
        return responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content=(
            "这是一个 research proposal，需要保留主管问题清单、ablation、baseline、"
            "tensor shape 和 implementation constraints。\n"
            "比如同样是\"怀孕初期能不能吃螃蟹\"，对专业医生应该给客观简洁的医学结论。\n"
            "原始数据格式：\n"
            "{\n"
            '  "query": "怀孕初期可以吃螃蟹吗？",\n'
            '  "responses": {"objective": "...", "empathetic": "..."}\n'
            "}\n"
            "模型结构:\n"
            "Style Encoder:\n"
            "  Input: (L, 4096)\n"
            "  MLP: (L, 4096) → (L, d_z)\n"
            "你怎么证明学到的是风格，而不是内容/长度/病种特征？\n"
            "为什么 surprise-weighted aggregation 合理？"
        ),
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
        progress_callback=progress.append,
    )

    assert result.status == "created"
    assert calls == [
        ("clipwiki-classify-note-mode", "cheap-model"),
        ("clipwiki-extract-knowledge-units", "cheap-model"),
        ("clipwiki-plan-note-update", "cheap-model"),
        ("clipwiki-edit-section", "strong-model"),
        ("clipwiki-validate-note-update", "cheap-model"),
    ]
    assert all("Detail preservation policy" in prompt for prompt in prompts[1:])
    assert "怀孕初期能不能吃螃蟹" in prompts[2]
    assert "怀孕初期可以吃螃蟹吗？" in prompts[3]
    assert "Input: (L, 4096)" in prompts[3]
    assert "你怎么证明学到的是风格，而不是内容/长度/病种特征？" in prompts[2]
    assert "为什么 surprise-weighted aggregation 合理？" in prompts[3]
    assert "question_checklist" in prompts[1]
    assert "detail_sections_to_preserve" in prompts[2]
    assert "Do not compress 10 specific questions into 2 generic risk bullets" in prompts[3]
    assert "must not collapse concrete reusable details" in prompts[4]
    assert result.llm_stats.model_backend_calls == {"cheap-model": 4, "strong-model": 1}
    assert result.llm_stats.model_total_tokens == {"cheap-model": 8, "strong-model": 2}
    assert "判断笔记类型" in progress
    assert "抽取知识单元" in progress
    assert "规划写入位置和目标章节" in progress


def test_incremental_ingest_uses_cheap_model_for_llm_cleaning(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    prompts: list[str] = []
    responses = [
        {
            "cleaned_content": "这是一个需要保留的技术结论。",
            "discarded_content": [{"text_or_summary": "chat wrappers", "reason": "UI noise"}],
            "risk_level": "low",
            "risk_reason": "Only wrappers removed.",
        },
        {
            "note_mode": "howto_reference",
            "should_preserve_source_details": False,
            "detail_policy": "Keep practical details.",
            "reason": "Short technical note.",
        },
        {
            "content_summary": "Cheap cleaning.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ClipWiki can clean copied chat wrappers with the cheap model.",
                    "type": "method",
                    "keywords": ["cleaning", "cheap model"],
                    "possible_topics": ["Cheap Model Cleaning"],
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Useful pipeline behavior.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New cleaning note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/cheap-cleaning.md", "heading_path": ["Cheap Model Cleaning"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# Cheap Model Cleaning\n\nCheap cleaning removes wrappers before extraction."},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        prompts.append(prompt)
        calls.append((self.task_name, self.model))
        return responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content="You said: 这是一个需要保留的技术结论。\n12:30\nClaude responded: 好的。",
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
    )

    assert result.status == "created"
    assert calls == [
        ("clipwiki-clean-ingest-content", "cheap-model"),
        ("clipwiki-classify-note-mode", "cheap-model"),
        ("clipwiki-extract-knowledge-units", "cheap-model"),
        ("clipwiki-plan-note-update", "cheap-model"),
        ("clipwiki-edit-section", "strong-model"),
        ("clipwiki-validate-note-update", "cheap-model"),
    ]
    assert "这是一个需要保留的技术结论" in prompts[1]
    assert "You said:" not in prompts[1]


def test_incremental_ingest_falls_back_when_synthesis_times_out(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    progress: list[str] = []
    units = [
        {
            "id": f"ku_{index}",
            "claim": f"Research detail {index} should remain usable after synthesis fallback.",
            "type": "method",
            "keywords": ["research", "fallback"],
            "possible_topics": ["Synthesis Fallback"],
            "detail_role": "algorithm_step",
            "preserve_as": "bullet",
            "should_keep": True,
            "novelty_hint": "new",
            "reason": "Useful detail.",
        }
        for index in range(46)
    ]
    responses = {
        "clipwiki-classify-note-mode": {
            "note_mode": "research_deep_dive",
            "should_preserve_source_details": True,
            "detail_policy": "Preserve research details.",
            "reason": "Research proposal.",
        },
        "clipwiki-extract-knowledge-units": {
            "content_summary": "Many research units.",
            "knowledge_units": units,
            "discarded_content": [],
        },
        "clipwiki-plan-note-update": {
            "decision": "create_new_file",
            "reason": "New fallback note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/synthesis-fallback.md", "heading_path": ["Synthesis Fallback"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        "clipwiki-edit-section": {
            "changed": True,
            "updated_section_markdown": "# Synthesis Fallback\n\nResearch details continue without synthesized units.",
        },
        "clipwiki-validate-note-update": {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    }

    def fake_complete_json(self, prompt: str):
        calls.append(self.task_name)
        if self.task_name == "clipwiki-synthesize-knowledge-units":
            raise RuntimeError("timeout")
        return responses[self.task_name], TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content="这是一个 research proposal，需要保留很多 algorithm steps 和 ablation details。",
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
        progress_callback=progress.append,
    )

    assert result.status == "created"
    assert "clipwiki-synthesize-knowledge-units" in calls
    assert "综合长对话知识单元失败，使用抽取结果继续" in progress
    assert result.knowledge_units[0]["claim"].startswith("Research detail")


def test_troubleshooting_mode_does_not_inject_research_verbatim_details(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []
    responses = [
        {
            "note_mode": "troubleshooting_runbook",
            "should_preserve_source_details": False,
            "detail_policy": "Keep symptoms, root causes, diagnostic commands, and final fixes.",
            "reason": "Python dependency troubleshooting.",
        },
        {
            "content_summary": "Python dependency troubleshooting.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ModuleNotFoundError: No module named 'dateutil' is fixed by installing python-dateutil.",
                    "type": "command",
                    "keywords": ["python-dateutil", "ModuleNotFoundError"],
                    "possible_topics": ["Python dependency troubleshooting"],
                    "detail_role": "core_claim",
                    "preserve_as": "bullet",
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Reusable troubleshooting fix.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New troubleshooting runbook.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "python/dependency-troubleshooting.md", "heading_path": ["Python 依赖排障"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# Python 依赖排障\n\n## 症状\n\nModuleNotFoundError: No module named 'dateutil'\n\n## 修复\n\n`python -m pip install python-dateutil`"},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        prompts.append(prompt)
        return responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content=(
            "Traceback...\nModuleNotFoundError: No module named 'dateutil'\n"
            "比如很多场景下可以先凑合替代：\ncp -a source_dir target_dir\n"
            "python -m pip install python-dateutil"
        ),
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
    )

    assert result.status == "created"
    assert "Troubleshooting/runbook mode is ON" in prompts[1]
    assert "heuristic_detail" not in prompts[2]
    assert '"possible_topics": ["原文可复用细节"]' not in prompts[2]
    assert "此处保留" not in (result.markdown or "")


def test_incremental_ingest_repairs_local_validation_failures_until_valid(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    responses = [
        {
            "note_mode": "howto_reference",
            "should_preserve_source_details": False,
            "detail_policy": "Keep the fix.",
            "reason": "How-to note.",
        },
        {
            "content_summary": "Repair invalid Markdown.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ClipWiki should repair invalid generated Markdown before giving up.",
                    "type": "method",
                    "keywords": ["clipwiki", "repair"],
                    "possible_topics": ["ClipWiki Repair"],
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Prevents empty failed ingests.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New repair note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/repair.md", "heading_path": ["ClipWiki Repair"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# ClipWiki Repair\n\n- 此处保留原文示例内容"},
        {"changed": True, "updated_section_markdown": "# ClipWiki Repair\n\n- TODO"},
        {"changed": True, "updated_section_markdown": "# ClipWiki Repair\n\nClipWiki repairs invalid generated Markdown before writing notes."},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        calls.append(self.task_name)
        return responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content="ClipWiki 需要在本地校验失败后继续把错误交给 LLM 修复。",
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
    )

    assert result.status == "created"
    assert "clipwiki-edit-section-repair-01" in calls
    assert "clipwiki-edit-section-repair-02" in calls
    assert "TODO" not in (result.markdown or "")
    assert "此处保留" not in (result.markdown or "")


def test_incremental_ingest_writes_best_effort_after_repair_attempts(tmp_path: Path, monkeypatch) -> None:
    responses = [
        {
            "note_mode": "howto_reference",
            "should_preserve_source_details": False,
            "detail_policy": "Keep the note.",
            "reason": "How-to note.",
        },
        {
            "content_summary": "Best-effort write.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ClipWiki writes the best available note after bounded repair attempts.",
                    "type": "method",
                    "keywords": ["clipwiki", "best-effort"],
                    "possible_topics": ["ClipWiki Best Effort"],
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Avoids empty outputs.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New best-effort note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/best-effort.md", "heading_path": ["ClipWiki Best Effort"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# ClipWiki Best Effort\n\n- TODO"},
        {"changed": True, "updated_section_markdown": "# ClipWiki Best Effort\n\n- TODO"},
        {"changed": True, "updated_section_markdown": "# ClipWiki Best Effort\n\n- TODO"},
        {"changed": True, "updated_section_markdown": "# ClipWiki Best Effort\n\n- TODO"},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        return responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = run_incremental_ingest(
        raw_content="ClipWiki 需要在多次修复失败后也写出最佳版本。",
        source_path=tmp_path / "source.txt",
        notes_root=tmp_path / "notes",
        cheap_model="cheap-model",
        strong_model="strong-model",
    )

    assert result.status == "created"
    assert result.note_path is not None and result.note_path.exists()
    assert result.validation_issues
    assert "best-effort" in result.reason


def test_validation_rejects_placeholder_text() -> None:
    result = validate_updated_section(
        "",
        "# Python 排障\n\n## 原文可复用细节示例\n\n- （此处保留原文示例内容与命令片段，确保笔记可替代原文）",
    )

    assert not result.valid
    assert any(issue.startswith("placeholder_text") for issue in result.issues)


def test_validation_rejects_chat_export_residue() -> None:
    result = validate_updated_section(
        "",
        "# Latent Policy Memory\n\nYou said: 我有个问题。\n\nClaude responded: 下面分析。",
    )

    assert not result.valid
    assert any(issue.startswith("chat_trace") for issue in result.issues)


def test_wikilink_graph_resolves_titles_paths_and_broken_links(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    agent = notes / "concepts" / "agent-loop.md"
    runtime = notes / "tools" / "runtime.md"
    agent.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    agent.write_text("# Agent Loop\n\nSee [[tools/runtime.md|runtime]] and [[Missing Topic]].", encoding="utf-8")
    runtime.write_text("# Runtime\n\nBack to [[Agent Loop]].", encoding="utf-8")

    assert extract_wikilinks(agent.read_text(encoding="utf-8")) == ["tools/runtime.md", "Missing Topic"]

    graph = build_link_graph(notes)

    assert graph.pages["concepts/agent-loop.md"].outbound == ["tools/runtime.md"]
    assert graph.pages["tools/runtime.md"].outbound == ["concepts/agent-loop.md"]
    assert graph.pages["concepts/agent-loop.md"].inbound == ["tools/runtime.md"]
    assert [(link.source, link.target) for link in graph.broken_links] == [("concepts/agent-loop.md", "Missing Topic")]
    assert graph.orphans == []


def test_refresh_maintenance_writes_internal_artifacts_without_polluting_note_index(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    note = notes / "agent" / "gui.md"
    note.parent.mkdir(parents=True)
    note.write_text("# GUI 代理\n\n## 技能化\n\n使用 [[工具调用]] 组合动作。", encoding="utf-8")

    artifacts = refresh_wiki_maintenance(notes)
    payload = load_or_build_note_index(notes)
    indexed_paths = [file_record["path"] for file_record in payload["files"]]

    assert artifacts.wiki_index_path.exists()
    assert artifacts.link_graph_path.exists()
    assert artifacts.lint_report_path.exists()
    assert indexed_paths == ["agent/gui.md"]
    assert not any(".clipwiki_note_memory" in path for path in indexed_paths)


def test_ingest_records_change_log_and_maintenance_artifacts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.txt"
    notes = tmp_path / "notes"
    html = tmp_path / "html"
    source.write_text("ClipWiki 应该记录确定性的 wiki 维护产物。", encoding="utf-8")
    responses = [
        {
            "note_mode": "howto_reference",
            "should_preserve_source_details": False,
            "detail_policy": "Keep reusable implementation detail.",
            "reason": "How-to note.",
        },
        {
            "content_summary": "ClipWiki maintenance artifacts.",
            "knowledge_units": [
                {
                    "id": "ku_1",
                    "claim": "ClipWiki writes deterministic maintenance artifacts after ingest.",
                    "type": "method",
                    "keywords": ["clipwiki", "maintenance"],
                    "possible_topics": ["ClipWiki Maintenance"],
                    "should_keep": True,
                    "novelty_hint": "new",
                    "reason": "Useful engineering behavior.",
                }
            ],
            "discarded_content": [],
        },
        {
            "decision": "create_new_file",
            "reason": "New maintenance note.",
            "duplicate_check": {"is_duplicate": False},
            "target": {"file_path": "clipwiki/maintenance.md", "heading_path": ["ClipWiki Maintenance"], "operation": "replace_section"},
            "edit_intent": {},
            "required_context": {"need_full_file": False, "need_sections": []},
        },
        {"changed": True, "updated_section_markdown": "# ClipWiki Maintenance\n\nClipWiki writes deterministic maintenance artifacts."},
        {"valid": True, "reason": "ok", "issues": [], "requires_retry": False},
    ]

    def fake_complete_json(self, prompt: str):
        return responses.pop(0), TokenUsage(input_tokens=2, output_tokens=3), {"artifact_path": None}

    monkeypatch.setattr("clipwiki.incremental.LiteLLMRuntime.complete_json", fake_complete_json)

    result = ingest_web_ai_result(source, notes_dir=notes, html_dir=html, cheap_model="cheap-model", strong_model="strong-model")
    entries = read_change_log(notes, last=1)
    internal = memory_dir(notes)

    assert result.status == "created"
    assert entries and "created | create_new_file" in entries[0]
    assert (internal / "wiki_index.md").exists()
    assert (internal / "link_graph.json").exists()
    assert (internal / "lint_report.json").exists()


def test_cli_reindex_reports_maintenance_artifacts(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    note = notes / "python" / "deps.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Python 依赖\n\n安装依赖时记录命令。", encoding="utf-8")

    result = CliRunner().invoke(app, ["reindex", "--notes", str(notes)])

    assert result.exit_code == 0
    assert "ClipWiki Maintenance Rebuilt" in result.output
    assert (memory_dir(notes) / "wiki_index.md").exists()
