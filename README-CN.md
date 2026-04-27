# ClipWiki 使用说明

ClipWiki 是一个独立的本地 Markdown 知识库工具，用来把你从 AI 对话中复制出来的长文本整理成长期可维护的笔记，并同步生成 HTML 页面。

## 安装/运行

```bash
cd clipwiki
uv sync --extra llm
```

本地运行目录都放在项目根目录下：

- `clipwiki-inbox/`：放从网页 AI 对话复制出来的原始文本。
- `clipwiki-notes/`：保存生成和维护后的 Markdown 笔记。
- `clipwiki-html/`：保存同步生成的 HTML 页面。

## 生成笔记

```bash
export LLM_API_KEY=你的key
export LLM_MODEL=openrouter/inclusionai/ling-2.6-flash:free

uv run clipwiki ingest clipwiki-inbox/test3.txt --notes clipwiki-notes --html clipwiki-html --top-k 8
```

也可以使用双层模型：`cheap-model` 负责分块抽取和校验，`strong-model` 负责规划和正式编辑。

```bash
uv run clipwiki ingest clipwiki-inbox/test3.txt \
  --notes clipwiki-notes \
  --html clipwiki-html \
  --cheap-model openrouter/inclusionai/ling-2.6-flash:free \
  --strong-model openrouter/inclusionai/ling-2.6-1t:free \
  --top-k 8
```

输出只显示摘要表格，不会打印完整 diff。

运行时会显示当前阶段，例如读取输入、抽取知识单元、规划写入位置、编辑章节、校验和渲染 HTML。长文本可能需要多轮 LLM 调用，看到阶段提示说明命令仍在正常推进。

`--top-k` 表示在更新笔记前，ClipWiki 会从现有笔记库里召回多少个最相关的候选章节给规划模型参考。数值越大，模型看到的候选上下文越多，更容易合并到已有笔记，但 prompt 也会更长；数值越小，运行更省，但可能更倾向于新建笔记。它不是“生成几篇笔记”的意思，通常保持 `5` 到 `8` 即可。

## 自动同步 HTML

如果你手动删除了某个 Markdown 笔记，下次执行 `clipwiki ingest` 时，ClipWiki 会自动删除没有对应 Markdown 的旧 HTML 页面，并重建 HTML index。

## 构建、搜索和问答

```bash
uv run clipwiki build clipwiki-notes --wiki clipwiki-vault
uv run clipwiki search "latent policy memory" --wiki clipwiki-vault --top-k 5
uv run clipwiki ask "什么是 latent policy memory?" --wiki clipwiki-vault
```

## Python API

```python
from pathlib import Path
from clipwiki.ingest import ingest_web_ai_result

result = ingest_web_ai_result(
    Path("clipwiki-inbox/test3.txt"),
    notes_dir=Path("clipwiki-notes"),
    html_dir=Path("clipwiki-html"),
    model="openrouter/inclusionai/ling-2.6-flash:free",
)
print(result.note_path)
```

## 与 wiki-memory-bench 的关系

`wiki-memory-bench` 现在只负责测评。它不再内置 ClipWiki 源码，而是通过外部 `clipwiki` 包调用相关功能。
