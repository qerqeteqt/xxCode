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
  agent/      main_agent.py  react_loop.py
  llm/        client.py
  tools/      base.py  registry.py  sandbox.py  text.py
              file_tool.py  bash_tool.py  search_tool.py
.agent/
  memory/     MEMORY.md  project.md  preferences.md
              architecture.md  lessons.md
  sessions/   YYYY-MM-DD.jsonl
tests/
config/       settings.py
main.py
pytest.ini
```

标为「规划中」的模块（`context/`、`memory/`、`scheduler/`）随对应 Phase 落地。
`.agent/sessions/` 是运行时数据，不入库；`.agent/memory/` 是长期知识，入库。

## 核心理念

- **Context 与 State 分离**：Context 是给模型看的消息工作区（会膨胀、会被压缩）；
  State 是结构化任务状态（`session_id` / `plan` / `files_changed` / `status`）。
- **Context Isolation**：Main Agent、SubAgent、AutoDream 各自持有独立 Context。
  SubAgent 拿不到 Main 的完整历史，只能拿到 task、项目文件和 Memory；
  执行结束即销毁，只有最终 Result 和显式写入的 Memory 会留下。
- **AutoDream 不是 Tool**：它由 Scheduler 在后台触发，生命周期独立于 Main Agent 的循环，
  因此不注册进 ToolRegistry。
- **ReAct Loop 不认识具体 Tool**：循环通过注入的 `execute_tool` 调用工具，
  Phase 2 接上 ToolRegistry 时循环代码一行未改。
- **路径沙箱**：所有文件类工具的路径统一经 `Sandbox.resolve()` 解析 ——
  先 resolve 成真实绝对路径（展开 `..` 和符号链接），再判断是否仍在 root 内。
  它既是安全边界，也是「这次让 agent 看哪个项目」的开关。
- **文件型 Persistence**：Store 实现可替换，未来换数据库不触碰 Runtime 核心逻辑。

## 开发进度

- [x] Phase 0 — 仓库脚手架
- [x] Phase 1 — LLM Client → Agent → ReAct Loop
- [x] Phase 2 — Tool 抽象 → ToolRegistry → FileTool / BashTool / SearchTool
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

## 已注册的工具

由 `build_default_registry(root)` 集中装配，全部共用同一个 `Sandbox(root)`：

| 工具 | 用途 |
|---|---|
| `Read` | 读文件，返回带行号文本，用 `offset` / `limit` 分段读大文件 |
| `Write` | 整体写入（新建或覆盖），父目录自动创建 |
| `Edit` | 精确字符串替换，`old_string` 必须在文件中唯一出现 |
| `List` | 递归列目录，跳过 `.git` / `.venv` / `__pycache__` 等 |
| `Glob` | 按文件名 glob 模式查找文件 |
| `Grep` | 按内容正则搜索，返回 `文件:行号: 那一行` |
| `Bash` | 在项目根目录执行命令，返回 exit code / stdout / stderr |

`ToolRegistry.execute()` 的契约是**永不抛异常** —— 工具名不存在、`arguments` 不是合法
JSON、参数不符合 schema、工具自身执行失败，四种情况都会转成一条模型看得懂的消息回灌，
让它自己纠正。

Bash 的安全档位目前是「危险命令黑名单」，挡的是**误伤而非攻击者**；
完整的权限系统（可配置策略 / 人工确认 / 容器隔离）属于 Phase 7。

## 运行

```bash
python main.py "app/tools 下注册了哪些工具？"
python main.py --root D:/pycharm/其他项目 "这个项目的入口在哪"
```

`--root` 同时是路径沙箱边界，默认取当前工作目录，启动时会打印出来。

## 开发原则

1. 严格按 Phase 顺序推进，不一次性实现全部模块。
2. 每个 Phase 完成并跑通测试后，再进入下一阶段。
3. 核心 Runtime 自己实现，不借助框架隐藏关键机制。
4. 长期 Memory 存的是提炼后的信息，不是整个 Session 的拷贝。
