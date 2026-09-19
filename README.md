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
  llm/        client.py
  context/    compactor.py
  memory/     session_store.py  memory_manager.py
              memory_tools.py  auto_dream.py
  scheduler/  scheduler.py
  tools/      base.py  registry.py  sandbox.py  text.py
              file_tool.py  bash_tool.py  search_tool.py  subagent_tool.py
.agent/
  memory/          MEMORY.md  <key>.md …
  memory-backups/  <时间戳>/            AutoDream 跑之前的记忆快照
  sessions/        YYYY-MM-DD/<session_id>.jsonl
  consolidation.json                    上次整理的时间
tests/
config/       settings.py
main.py
pytest.ini
```

规划中的模块随对应 Phase 落地（`context/` 在 Phase 7b 到了，只剩 Web UI）。
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
  Phase 2 接上 ToolRegistry 时循环代码一行未改；Phase 3 的 SubAgent 复用的也是同一个循环。
- **路径沙箱**：所有文件类工具的路径统一经 `Sandbox.resolve()` 解析 ——
  先 resolve 成真实绝对路径（展开 `..` 和符号链接），再判断是否仍在 root 内。
  它既是安全边界，也是「这次让 agent 看哪个项目」的开关。
- **权限靠「能力不存在」实现，不靠 prompt 自律**：只读的 SubAgent 拿不到写类工具，
  子 Agent 的工具集里没有 `SubAgent`。这是事实约束，不是请求模型配合。
- **工具返回值分两层**：`Tool.execute()` 返回结构化的 `ToolResult(ok, text, changed_path)`，
  给 Runtime 自己用；`ToolRegistry.execute()` 只把 `text` 交给模型。
  这样「这个文件到底改成了没有」是读一个字段，而不是嗅探返回文本的前缀。
- **派生数据不手工维护**：`MEMORY.md` 是从记忆目录里的文件重新算出来的，
  不是谁记得去更新它。手工维护索引有个很隐蔽的失败模式 —— **孤儿记忆**：
  文件存在、内容很好，但索引里没有链接，于是没有任何人会知道它。
- **文件型 Persistence**：Store 实现可替换，未来换数据库不触碰 Runtime 核心逻辑。

## 会话

会话记录是 append-only 的 JSONL，落在 `<root>/.agent/sessions/<日期>/<会话id>.jsonl`：

```jsonl
{"v":1,"ts":"...","session_id":"20260919-191419-e649","type":"session_start","root":"..."}
{"v":1,"ts":"...","type":"message","message":{"role":"user","content":"给 calc.py 加个函数"}}
{"v":1,"ts":"...","type":"message","message":{"role":"assistant","tool_calls":[...]}}
{"v":1,"ts":"...","type":"state","state":{"status":"running","files_changed":["src/calc.py"]}}
{"v":1,"ts":"...","type":"session_end","status":"finished"}
```

三个关键点：

- **`message` 里存的是原样的 API 消息**，恢复时过滤出 `type=="message"` 取出 `.message`
  就直接得到能喂给 LLM 的 messages —— 零转换。
- **system prompt 不落盘**：它是「配置」不是「历史」，而且带着长期记忆的索引，
  而记忆是会变的。冻结进 JSONL 的话，续会话时 Agent 看到的就是过时的索引。
  所以每次运行现拼一份插在最前面。
- **append-only**：每条消息产生时立刻写一行。进程崩了，已经发生的事还在。
  所以 State 也是「每次变化追加一行」，不覆盖写 —— 覆盖写遇到写一半崩溃会留下坏文件。

State 只保留真正有人读的字段（`session_id` / `status` / `files_changed` / 时间戳）。
文档第八节列的 `task` / `plan` / `findings` 在 V1 没有消费者 —— 模型本来就用自然语言
在 messages 里表达了它们，再抽一遍只是空转。

## 长期记忆

`.agent/memory/` 下平铺若干 Markdown 文件，`MEMORY.md` 是它们的索引：

```markdown
---
name: calc 模块编码约定
description: 给 calc.py 加函数时必须遵守的规则，代码里看不出来
category: Project
---

1. 所有函数必须是纯函数，不许读写模块级变量或全局状态
...
```

元信息放 frontmatter，正文是纯 Markdown —— 文件本身就是一篇能读的文章，
元信息只服务于索引生成。

**`MEMORY.md` 是派生数据**：由 `MemoryManager` 扫描目录重新算出来，启动时和每次增删改后
自动重建。所以手工丢一个 `.md` 进去，下次启动它就会被收录 —— 没人需要记得更新索引。

**记忆是给 Agent 读的**，所以沙箱放行 `.agent/memory`（但仍然挡住 `.agent/sessions`）。
Agent 用普通的 `Read` / `Grep` 就能读记忆，不需要专用工具。

索引会拼进 Main Agent 的 system prompt（每行一条链接加一句描述），
这样 Agent 知道有哪些记忆存在；需要细节时再 `Read` 打开具体文件。
SubAgent 不注入 —— 它是执行者，任务已经很具体，一份索引只会分散注意力。

> `MemoryManager` 刻意保持"哑"：只有 `list` / `read` / `write` / `update` /
> `delete` / `sync_index`，不含任何判断 —— 文档第十三节把「什么信息值得长期保存」
> 划给了 AutoDream。

## 记忆整理（AutoDream）

`Scheduler` 判断该不该整理，该就启动 `AutoDream` 把近期会话提炼成记忆。

触发条件是**并且**关系，都满足才跑：距上次 ≥ `CONSOLIDATE_MIN_HOURS`（默认 24）
**且** 新增会话 ≥ `CONSOLIDATE_MIN_SESSIONS`（默认 5）。想立刻跑一次用 `--consolidate`。

从没整理过的项目**同样要等够阈值** —— 基线取「最早那个会话的时间」，也就是这个项目
第一次被使用的时刻。不让新项目一上来就跑一次，是因为阈值设成「并且」的意图就是尽量
少跑；绕过它等于把那个意图作废。

AutoDream 是第三个拥有独立 Context 的 Agent，它看到的东西**完全由我们喂**：

```
Main Agent    看得见：整个对话历史 + 项目文件
SubAgent      看得见：一条 task + 项目文件
AutoDream     看得见：只有会话摘要 + 现有记忆索引，别的没有任何通道
```

两处刻意的收缩：

1. **读的是会话「摘要」而不是原始 JSONL**。只取三样：用户问了什么、最终答了什么、
   改了哪些文件；中间的 Read/Grep 过程全部丢掉。既省 token，也直接服务于文档那句
   「不应该把整个 Session 原样复制到长期 Memory」—— 如果它看到的本来就是原样 Session，
   它很可能就照着抄了。
2. **工具集里只有 `ReadMemory` / `WriteMemory` / `UpdateMemory` / `DeleteMemory`**，
   没有 Read / Glob / Grep / Bash。它是全系统唯一能写记忆的角色，而它想跑去读项目代码
   都没有工具可用。

关于「后台」：CLI 跑完就退出，后台 asyncio 任务会跟着进程死。所以这一版取的「后台」
是文档第十四节要求的那层——**独立于 Main Agent 的循环、不注册进 ToolRegistry**——
而不是「另一个进程」。`Scheduler.check()` 和 `consolidate()` 是分开的两个方法，
将来上 Web 把 `check()` 挂到事件循环上定期调用就变成真后台了，这个文件不用改。

**安全网**：跑之前把 `.agent/memory/` 整份快照到 `.agent/memory-backups/<时间戳>/`
（保留最近 5 份）。AutoDream 手里有 `DeleteMemory`，而它是个 LLM —— 要防它的判断失误、
防它跑到一半被 Ctrl+C、也方便你事后对比它到底改了什么。整理没跑完就不更新状态文件，
下次会重试。

## SubAgent

Main Agent 通过 `SubAgent` 工具派发子 Agent。三种规格只是三份配置
（`app/agent/subagent.py`）：

| 规格 | 工具集 | 产出 |
|---|---|---|
| `Explore` | `Read` `List` `Glob` `Grep` | 事实：代码在哪、怎么组织的 |
| `Plan` | 同上（差别只在 system prompt） | 方案：分几步改、风险在哪 |
| `General-Purpose` | 全部（含 `Write` `Edit` `Bash`） | 结果：改了什么、验证过没有 |

`Bash` 不在只读工具集里 —— 它「常用作只读」不等于「只能只读」，权限按最大能力算。

三条结构性保证：

1. **工具集裁剪**：子 Agent 的能力边界在注册表这一层被切干净，它连 `Write` 存在都不知道。
2. **Context 隔离**：子 Agent 的 messages 从零构造，只有 `system` + `user` 两条，
   不含 Main 的历史。代价是 Main 必须把关键信息显式写进 `context` 参数。
3. **递归禁止**：子 Agent 的工具集里没有 `SubAgentTool`，靠能力缺失而不是深度计数。

返回值带元信息，让 Main 不必猜：

```
[General-Purpose 完成 | 7 步 | 修改: app/auth/login.py, tests/test_login.py]
```

改动清单来自**工具调用记录**（`Write`/`Edit` 实际成功了几次），不是让模型自己总结 ——
模型会漏、会把「打算改」说成「已经改」。

## 开发进度

- [x] Phase 0 — 仓库脚手架
- [x] Phase 1 — LLM Client → Agent → ReAct Loop
- [x] Phase 2 — Tool 抽象 → ToolRegistry → FileTool / BashTool / SearchTool
- [x] Phase 3 — SubAgent Runtime → Explore / Plan / General-Purpose → SubAgentTool
- [x] Phase 4 — State → JSONL Session Store
- [x] Phase 5 — MemoryManager → MEMORY.md → Markdown Memory
- [x] Phase 6 — Scheduler → AutoDream → Memory Consolidation
- [ ] Phase 7 — 完善（文档把它写成一个大清单，但一次做完就违反「不一次性实现全部模块」的原则，所以拆开推）
  - [x] 7a — Retry + Token Management
  - [x] 7b — Context Compaction
  - [ ] 7c — Permission
  - [ ] 7d — Parallel SubAgent / Streaming / Web UI

Logging 与 Async 在前六个阶段已经顺手做掉了：每个模块一个 logger；从 Phase 1 起就是
async 的，遗留只是搜索时的同步文件 IO 会短暂阻塞事件循环（当前规模无感）。

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

由 `build_default_registry(root, llm=None)` 集中装配，全部共用同一个 `Sandbox(root)`：

| 工具 | 用途 |
|---|---|
| `Read` | 读文件，返回带行号文本，用 `offset` / `limit` 分段读大文件 |
| `Write` | 整体写入（新建或覆盖），父目录自动创建 |
| `Edit` | 精确字符串替换，`old_string` 必须在文件中唯一出现 |
| `List` | 递归列目录，跳过 `.git` / `.venv` / `__pycache__` 等 |
| `Glob` | 按文件名 glob 模式查找文件 |
| `Grep` | 按内容正则搜索，返回 `文件:行号: 那一行` |
| `Bash` | 在项目根目录执行命令，返回 exit code / stdout / stderr |
| `SubAgent` | 派一个子 Agent 独立完成任务，只带回结论（需要 `llm` 才会注册） |

`ToolRegistry.execute()` 的契约是**永不抛异常** —— 工具名不存在、`arguments` 不是合法
JSON、参数不符合 schema、工具自身执行失败，四种情况都会转成一条模型看得懂的消息回灌，
让它自己纠正。

Bash 的安全档位目前是「危险命令黑名单」，挡的是**误伤而非攻击者**；
完整的权限系统（可配置策略 / 人工确认 / 容器隔离）属于 Phase 7。

## 上下文压缩

`messages` 只增不减，跑几十轮之后每轮都要把完整历史重发一遍 —— token 往平方上走，
迟早撞上模型上限直接报错。所以循环在**每次 LLM 调用之前**检查一次，超了就压：

```
[system] + [要压掉的一大段] + [最近 ≥8 条原样保留]
                ↓ 一次独立的 LLM 摘要调用
[system] + [摘要] + [最近 ≥8 条]
```

三个决定：

- **何时压**：用 `llm.last_prompt_tokens`，也就是上一次调用**真实的** prompt 大小
  （`COMPACT_THRESHOLD_TOKENS`，默认 40000）。比拿字符数估算准得多，而且免费。
- **切在哪**：优先 `user` 消息（轮次的自然边界）。**硬约束是切点不能落在 `tool` 消息上** ——
  tool 必须紧跟它归属的 assistant，配对断了 API 直接 400。找不到合适的 user 就退一步切在
  任何非 tool 的位置，长时间单轮任务（一路 Read/Grep 中间没有新 user 发言）靠的就是这条退路。
- **system 必须留住**：它是整个会话的角色和规则，丢了 Agent 会在压缩之后「忘记自己是谁」。

**JSONL 怎么办**：压缩会把内存里的 messages 整个换掉，但 JSONL 是 append-only 的，
原始消息早写下去了、不删。所以压缩时追加一条声明：

```jsonl
{"type":"compaction","summary":"...","keep_count":8}
```

意思是「从这一刻起，我前面的消息等价于这段摘要 + 最后 8 条」。恢复时按顺序重放，
碰到 compaction 记录就做一次替换 —— **反复压缩也能正确还原**，因为重放等于把当时的
压缩过程又走了一遍。

## 韧性与成本

**重试**：`LLMClient` 对「等一会儿再来就好」的失败自动重试 —— 网络超时、429、5xx。
4xx（429 除外）**不重试**，因为它说明请求本身有问题（key 不对、模型名写错、参数非法），
重试只会把一个明确的配置错误变成一个要等半分钟的谜题。

最多 3 次，指数退避 + 抖动（1s → 2s → 4s），并尊重服务端的 `Retry-After`（封顶 30 秒）。
抖动不是装饰：同时跑多个 SubAgent 时它们会同时失败、同时重试，没有随机量就是一批一批
地一起撞上去。

**Token 用量**：每次调用的 `usage` 会累加到 `LLMClient.usage`，会话结束时写进
`SessionState`（`--list-sessions` 能看到）。失败重试的那些不计入 —— 没拿到 usage。

SubAgent 的返回值会带上自己的开销：

```
[Explore 完成 | 4 步 | 6.4k tokens]
```

这一项不是锦上添花。实测一次会话总量 11.2k tokens，其中 **6.4k 花在那个子 Agent 身上**，
而 Main Agent 的视角里它只占 1 步 —— 没有这个数字，你完全看不出钱花在哪了。

## 运行

```bash
python main.py "app/tools 下注册了哪些工具？"          # 新会话
python main.py --continue "接着上一个问题"             # 续最近活跃的会话
python main.py --session e649 "接着问"                 # 续指定会话（前后缀都认）
python main.py --list-sessions                         # 列出最近的会话
python main.py --consolidate                           # 强制整理一次记忆后退出
python main.py --no-consolidate "随便问问"              # 这次跑完不检查整理
python main.py --root D:/pycharm/其他项目 "看看入口在哪"
```

`--root` 同时是路径沙箱边界，默认取当前工作目录，启动时会打印出来。
恢复会话时会校验记录的 root 与当前是否一致 —— 项目被改名或搬走后，
旧会话会明确报错而不是在错位的上下文里继续跑。

**执行顺序是先给答案、再跑整理**（整理可能几十秒，不该挡在答案前面）。
整理失败只记一条 warning，不影响已经拿到的答案。达到步数上限时也不再甩 traceback，
而是告诉你会话 id —— 过程已经逐条落盘，`--continue` 就能接着聊。

## 开发原则

1. 严格按 Phase 顺序推进，不一次性实现全部模块。
2. 每个 Phase 完成并跑通测试后，再进入下一阶段。
3. 核心 Runtime 自己实现，不借助框架隐藏关键机制。
4. 长期 Memory 存的是提炼后的信息，不是整个 Session 的拷贝。
