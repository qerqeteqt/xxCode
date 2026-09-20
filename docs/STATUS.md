# xxCode 现状

> 本文是**当前状态的快照**，不是设计文档。设计动机写在代码注释和 README 里。
> 快照会过期 —— 和代码对不上时以代码为准。

## 一句话

一个自己写的 Code Agent Runtime。不依赖 LangGraph 等框架，目的是把
Agent 的运行机制（循环、上下文、工具、记忆、子 Agent）全部显式地握在手里。

## 规模

| | 数字 |
|---|---|
| 源码 | 5718 行（`app/` 5214 + `config/` 73 + `main.py` 431） |
| 测试 | 4538 行，**349 个用例全过** |
| 提交 | 30 个 |
| 前端 | 2 个文件，无构建（`index.html` 986 行 + `markdown.js` 132 行） |

按模块：

```
app/agent      476 行   ReAct Loop · MainAgent · SubAgent 规格
app/context    213 行   上下文压缩
app/llm        412 行   LLM 客户端（重试 / 流式 / token 统计）
app/memory    1313 行   Session · MemoryManager · 提取器 · AutoDream
app/scheduler  248 行   触发判断 · 备份
app/tools     1973 行   9 个工具 · 注册表 · 沙箱 · 权限闸门
app/web        579 行   FastAPI + SSE
```

## 架构总览

```
User
 │
 ├─ CLI     main.py
 └─ Web     app/web/server.py（FastAPI + SSE）
        │
        ▼
   MainAgent ────────────────────────────────────┐
        │                                         │
        ├─ System Prompt ← 记忆索引（每轮现拼）    │
        │                                         │
        ▼                                         │
   ReAct Loop  ← 注入 execute_tool / llm          │
        │       ← 每轮检查上下文压缩              │
        │                                         │
        ▼                                         │
   ToolRegistry ── PermissionGate ── Sandbox      │
        │                                         │
        ├─ Read / Write / Edit / List              │
        ├─ Glob / Grep                             │
        ├─ Bash（Git Bash）                        │
        ├─ WebSearch（Tavily）                     │
        └─ SubAgent ─→ Explore / Plan /              │
                       General-Purpose               │
                                                     │
   旁路（不进 Main 的 Context）────────────────────┘
        ├─ SessionStore      会话落 JSONL
        ├─ MemoryExtractor   每轮提取长期记忆
        └─ Scheduler ─→ AutoDream   攒够条件批量整理
```

## 一次提问的完整链路

```
1. main.py / web           建会话（新建或 --continue 恢复）
2. MemoryManager.sync_index()   重建索引（派生数据，保证不过期）
3. MainAgent 拼 system prompt   = 基础提示 + 记忆索引
4. ReAct Loop 开始
   ├─ 检查上下文压缩（上次 prompt > 40000 tokens 就压）
   ├─ llm.chat(messages, tools)   ← 可能流式
   ├─ 有 tool_calls？
   │    ├─ 没有 → 它就是最终答案，结束
   │    └─ 有   → 逐个交给 ToolRegistry.execute
   │              └─ 四道关卡 → 权限闸门 → 沙箱 → 执行
   └─ 回到上一步，直到 max_steps(60) 用完
5. SessionStore 写 JSONL（每条消息产生时就写，append-only）
6. 答案打印/推给浏览器
7. MemoryExtractor 提取本轮值得长期保存的信息
8. Scheduler 判断该不该跑 AutoDream 整理
```

**关键：7、8 都排在答案之后。** 它们加起来可能几十秒，挡在答案前面没人受得了。

## 工具（9 个）

| 名字 | 风险 | 干什么 |
|---|---|---|
| `Read` | read | 读文件，带行号，`offset`/`limit` 分段 |
| `Write` | write | 整体写入，父目录自动创建 |
| `Edit` | write | 精确字符串替换，`old_string` 必须唯一 |
| `List` | read | 递归列目录，跳过 `.git`/`.venv`/`__pycache__` |
| `Glob` | read | 按文件名模式查找 |
| `Grep` | read | 按内容正则搜索 |
| `Bash` | execute | 跑命令（Windows 上用 Git Bash），带超时杀进程树 |
| `WebSearch` | read | 联网搜索（Tavily）。**配了 key 才注册** |
| `SubAgent` | read | 派子 Agent，只带回结论 |

另有 4 个记忆工具（`ReadMemory`/`WriteMemory`/`UpdateMemory`/`DeleteMemory`），
**只给 AutoDream 和 MemoryExtractor**，Main Agent 一个都没有。

## 上下文

**Context 与 State 分离：**

- Context = 给模型看的消息数组，会膨胀、会被压缩
- State = 给程序看的结构化字段（`session_id` / `status` / `files_changed` /
  `prompt_tokens` / `completion_tokens` / `llm_calls` / 时间戳）

**压缩（ContextCompactor）：**

- 触发用 `llm.last_prompt_tokens` —— **上一次调用真实的 prompt 大小**，
  比拿字符数估算准，而且免费
- 切点优先落在 `user` 消息上（轮次边界）。**硬约束：不能切在 `tool` 消息上** ——
  tool 必须紧跟它归属的 assistant，配对断了 API 直接 400
- 找不到合适的 user 就退到任何非 tool 的位置 —— 长时间单轮任务（一路 Read/Grep）
  靠的就是这条退路
- **`system` 必须留住**：丢了它 Agent 会"忘记自己是谁"
- 压缩时往 JSONL 追加一条 `{"type":"compaction","summary":...,"keep_count":...}`，
  恢复时按顺序重放 —— 反复压缩也能正确还原

## 记忆系统

**三个角色，分工不同：**

| | 触发 | 职责 | 工具 |
|---|---|---|---|
| **Main Agent** | 实时 | 读索引（注入 system prompt）+ `Read` 具体文件 | —— |
| **MemoryExtractor** | **每轮对话后** | **提取** —— 把刚出现的事实写下来 | 读/写/改，**不含删除** |
| **AutoDream** | 攒够 24h + 5 会话 | **整理** —— 合并、去重、更新过时 | 四个（含删除） |

缺了提取器，「记住 X」就是句空话：模型只能口头答应，然后等 AutoDream 哪天
攒够条件去流水里翻。缺了 AutoDream，记忆会越攒越碎。

**文件格式**：`.agent/memory/` 下平铺 Markdown + frontmatter，`MEMORY.md` 是索引。

```markdown
---
name: 项目规范
description: pytest 验证、中文提交信息
category: Project
---

正文……
```

**`MEMORY.md` 是派生数据** —— 从目录内容重新算出来的，不是手工维护的。
手工维护有个很隐蔽的失败模式：**孤儿记忆**（文件在、索引里没有链接，
没有任何人会知道它存在）。

**当前记忆（4 条）：**

```
[User]    代码注释语言   — 用户要求写代码时注释用中文
[Project] 项目规范       — pytest 验证、中文提交信息、8000 端口
[User]    用户姓名       — 用户叫肖鑫
[User]    用户喜欢的角色 — 用户喜欢《命运石之门》的牧濑红莉栖
```

**安全网**：AutoDream 跑之前把 `.agent/memory/` 整份快照到
`.agent/memory-backups/<时间戳>/`（保留最近 5 份）。它手里有 `DeleteMemory`，
而它是个 LLM —— 要防判断失误、防跑到一半被 Ctrl+C、也方便事后对比。
整理没跑完就不更新状态文件，下次重试。

## SubAgent

三种规格只是三份配置，差别只有**工具白名单**和 **system prompt**：

| 规格 | 工具集 | 产出 |
|---|---|---|
| `Explore` | `Read` `List` `Glob` `Grep` | 事实：代码在哪、怎么组织的 |
| `Plan` | 同上（差别只在 system prompt） | 方案：分几步改、风险在哪 |
| `General-Purpose` | 全部（含 `Write` `Edit` `Bash`） | 结果：改了什么、验证过没有 |

三条结构性保证：

1. **工具集裁剪** —— 只读规格连 `Write` 存在都不知道
2. **Context 隔离** —— 子 Agent 的 messages 从零构造，只有 `system` + `user`，
   不含 Main 的历史
3. **递归禁止** —— 子 Agent 的工具集里没有 `SubAgentTool`，**不靠深度计数，
   靠能力缺失**

返回值带元信息：`[General-Purpose 完成 | 7 步 | 6.4k tokens | 修改: a.py]`。
改动清单来自工具调用的**实际成功记录**，不是让模型自己总结。

**一个改不动的事实**：`deepseek-chat` **不会自发派 SubAgent**。给过
"需要打开 3 个以上文件就派 Explore" 的硬规则，它仍然自己读了 15+ 个文件。
明确要求时链路完全正常 —— 这是模型倾向，不是 Runtime 问题。

## 权限

三层判定，**顺序不能反**：

```
① allow 规则命中   → 放行      （显式放开，优先级最高）
② deny 规则命中    → 拒绝      （敏感路径，硬拒，不问）
③ 按工具的风险等级 → read 放行 / write·execute 问人
```

- **风险等级由工具自己声明**（`Tool.risk`），默认值是最严格的 `execute` ——
  忘了声明会被「问」，而不是被静默放行
- **敏感文件默认拒绝**（`.env` / `*.pem` / `id_rsa*` …），要放开改
  `.agent/permissions.json`
- **敏感文件在遍历层就被摘掉** —— `Grep` 搜不到、`Glob` 列不出来。
  这条不能靠规则做：规则没法表达「这次搜索会扫到哪些文件」
- **只读 shell 命令不弹窗**（`ls`/`cat`/`git log`/`pytest`…）——
  每条命令都问的结果是用户闭着眼睛按 y，那比不问更危险
- **会话放行只活在内存里** —— 一次手滑选了「永久允许」就跟着项目走了，
  而你多半不记得自己什么时候做的决定
- 子 Agent 内部每次写/执行都过**同一份**闸门（共享放行记录，只换来源标签）

## 会话与持久化

`.agent/sessions/<日期>/<会话id>.jsonl`，**append-only**。

```jsonl
{"v":1,"ts":"...","session_id":"...","type":"session_start","root":"..."}
{"v":1,"ts":"...","type":"message","message":{"role":"user","content":"..."}}
{"v":1,"ts":"...","type":"state","state":{"status":"running","files_changed":[...]}}
{"v":1,"ts":"...","type":"compaction","summary":"...","keep_count":8}
{"v":1,"ts":"...","type":"session_end","status":"finished"}
```

- **message 存的是原样的 API 消息**，恢复时零转换
- **system prompt 不落盘** —— 它是「配置」不是「历史」，而且带记忆索引（会变）
- **append-only**：每条消息产生时就写。进程崩了，已经发生的还在

## Web 界面

```bash
双击 web.cmd          # 或 python main.py --web
```

- 左侧会话列表（标题取自第一条用户提问，可删除），右侧对话
- 模型输出**逐字流式**、按 Markdown 渲染（标题/代码块/列表，代码块带复制按钮）
- 工具调用显示成可折叠卡片，带来源标签（`SubAgent: Explore`）
- 授权请求变成按钮，不用回终端
- 深/浅主题，⚙ 里能改配置（写回 `.env`，保留注释，验证不过就回滚）

**只有一个进程**：uvicorn 同时负责 API、SSE 和发 `index.html`。没有前端构建。

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` | `deepseek-chat` | 模型 |
| `MAX_STEPS` | 60 | **防打转的兜底，不是预算** |
| `EXTRACT_MEMORY` | `true` | 每轮提取长期记忆 |
| `COMPACT_THRESHOLD_TOKENS` | 40000 | 上下文压缩阈值 |
| `CONSOLIDATE_MIN_HOURS` | 24 | 记忆整理：最小间隔 |
| `CONSOLIDATE_MIN_SESSIONS` | 5 | 记忆整理：最小新会话数 |
| `TAVILY_API_KEY` | 空 | 留空则不注册 `WebSearch` |

后 6 项可以在网页设置面板改。

**⚠️ `MAX_STEPS` 只管 Main Agent。** 其他角色各有独立预算，按角色定，
本来就不该跟 Main 一样：

| 角色 | 步数 | 在哪 |
|---|---|---|
| Main Agent | `MAX_STEPS`（60） | `config/settings.py` |
| Explore | 8 | `app/agent/subagent.py` |
| Plan | 8 | `app/agent/subagent.py` |
| General-Purpose | 12 | `app/agent/subagent.py` |
| AutoDream | 12 | `app/memory/auto_dream.py` |
| 记忆提取器 | 6 | `app/memory/extractor.py` |`LLM_API_KEY` 和 `TAVILY_API_KEY` **不在面板里** ——
`.env` 里的密钥不该被网页随便读写。

## Phase 进度

```
[x] Phase 0  仓库脚手架
[x] Phase 1  LLM Client → Agent → ReAct Loop
[x] Phase 2  Tool 抽象 → ToolRegistry → FileTool / BashTool / SearchTool
[x] Phase 3  SubAgent Runtime → Explore / Plan / General-Purpose
[x] Phase 4  State → JSONL Session Store
[x] Phase 5  MemoryManager → MEMORY.md → Markdown Memory
[x] Phase 6  Scheduler → AutoDream → Memory Consolidation
[x] Phase 7a Retry + Token Management
[x] Phase 7b Context Compaction
[x] Phase 7c Permission
[x] Phase 7d Streaming + Web UI（Parallel SubAgent 未做）
```

## 已知局限

这些是**明确记录在案的「知情不补」**，不是遗漏：

| | 说明 |
|---|---|
| **Bash 不受沙箱管辖** | 它交给操作系统，`cwd` 只是起点不是边界。`cd / && cat /etc/passwd` 黑名单拦不住。真隔离要容器/seccomp/Job Object |
| **闸门拦调用不拦意图** | 模型可以把 `Write` 绕成 `Bash("echo >")`。有测试专门记录这一点 —— 它断言的**不是「安全」，是「这里挡不住」** |
| **WebSearch 会把关键词发出去** | 靠 prompt 提醒别放代码和密钥，不做技术强制（本地脱敏误伤率太高） |
| **`deepseek-chat` 不自发派 SubAgent** | 模型倾向问题，明确要求时正常 |
| **Grep/List 是同步文件 IO** | 会短暂阻塞事件循环，当前规模无感 |
| **JSONL 只增不减** | 压缩只压内存里的 messages，文件本身一直追加 |
| **单进程单会话** | 没有并发，没有多用户，没有鉴权 |
| **没开热重载** | 改完代码要重启服务。被这个坑过两次：浏览器连的其实是旧进程 |

## 离「上线」还差什么

- **服务化**：HTTP API + 会话路由（现在 API 有了，但只服务单机单用户）
- **多用户与隔离**：文档明确 V1 不做，上线就得做
- **容器隔离**：补 Bash 那个洞
- **部署运维**：日志收集、监控、限流、成本控制、鉴权
