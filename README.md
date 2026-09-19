# xxCode

自研 Code Agent Runtime。

参考 Claude Code 的核心设计思路（文件型 Memory、Context Isolation、SubAgent、后台记忆整理），
但**不依赖 LangGraph 等 Agent Framework** —— 目的是理解 Agent Runtime 本身的运行机制，
而不是快速堆叠框架。

## 技术栈

| 项 | 选择 |
|---|---|
| 语言 | Python 3.13 |
| 数据校验 | Pydantic |
| 并发 | asyncio |
| LLM | DeepSeek（OpenAI 兼容接口） |
| Persistence | 文件系统（V1 不引入数据库） |
| 短期记忆 | JSONL（Session 事件流水） |
| 长期记忆 | Markdown（`MEMORY.md` 作为索引） |

## 目录结构

```
app/
  agent/      main_agent.py  react_loop.py  subagent.py
  context/    context.py  context_builder.py
  tools/      base.py  registry.py  file_tool.py  bash_tool.py
              search_tool.py  subagent_tool.py
  memory/     memory_manager.py  session_store.py  auto_dream.py
  scheduler/  scheduler.py
  llm/        client.py
.agent/
  memory/     MEMORY.md  project.md  preferences.md
              architecture.md  lessons.md
  sessions/   YYYY-MM-DD.jsonl
tests/
config/
main.py
```

`.agent/sessions/` 是运行时数据，不入库；`.agent/memory/` 是长期知识，入库。

## 核心理念

- **Context 与 State 分离**：Context 是给模型看的消息工作区（会膨胀、会被压缩）；
  State 是结构化任务状态（`session_id` / `plan` / `files_changed` / `status`）。
- **Context Isolation**：Main Agent、SubAgent、AutoDream 各自持有独立 Context。
  SubAgent 拿不到 Main 的完整历史，只能拿到 task、项目文件和 Memory；
  执行结束即销毁，只有最终 Result 和显式写入的 Memory 会留下。
- **AutoDream 不是 Tool**：它由 Scheduler 在后台触发，生命周期独立于 Main Agent 的循环，
  因此不注册进 ToolRegistry。
- **文件型 Persistence**：Store 实现可替换，未来换数据库不触碰 Runtime 核心逻辑。

## 开发进度

- [x] Phase 0 — 仓库脚手架
- [ ] Phase 1 — LLM Client → Agent → ReAct Loop
- [ ] Phase 2 — Tool 抽象 → ToolRegistry → FileTool / BashTool / SearchTool
- [ ] Phase 3 — SubAgent Runtime → Explore / Plan / General-Purpose → SubAgentTool
- [ ] Phase 4 — State → JSONL Session Store
- [ ] Phase 5 — MemoryManager → MEMORY.md → Markdown Memory
- [ ] Phase 6 — Scheduler → AutoDream → Memory Consolidation
- [ ] Phase 7 — Retry / Logging / Async / Parallel SubAgent / Permission / Streaming …

## 环境准备

本项目使用 conda 环境 `langgraph`（Python 3.13）：

```bash
conda activate langgraph
pip install -r requirements-dev.txt
```

配置密钥：

```bash
cp .env.example .env   # 然后填入真实的 LLM_API_KEY
```

`.env` 已被 `.gitignore` 排除，不会入库。

## 开发原则

1. 严格按 Phase 顺序推进，不一次性实现全部模块。
2. 每个 Phase 完成并跑通测试后，再进入下一阶段。
3. 核心 Runtime 自己实现，不借助框架隐藏关键机制。
4. 长期 Memory 存的是提炼后的信息，不是整个 Session 的拷贝。
