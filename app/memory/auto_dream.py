"""AutoDream —— 后台记忆整理 Agent。

它和 Main / SubAgent 是同一个模式的第三个实例：一个独立的 Context + 同一个
`run_react_loop`。区别在于它看到的东西**完全由我们喂** —— 它没有别的信息通道。

    Main Agent    看得见：整个对话历史 + 项目文件
    SubAgent      看得见：一条 task + 项目文件
    AutoDream     看得见：只有我们喂的材料（会话摘要 + 现有记忆索引）

## 两个刻意的收缩

**一、它读的是「会话摘要」，不是原始 JSONL。**

一个会话几十行 JSONL，二十个会话就是几百行，其中大部分是「Read 了哪个文件、
Grep 搜了什么」的过程噪声 —— 对「什么值得长期记住」毫无价值。所以这里先把每个
会话压成三样：用户问了什么、最终答了什么、改了哪些文件。

这既省 token，也直接服务于文档第十七节那句「不应该把整个 Session 原样复制到
长期 Memory」：如果它看到的本来就是原样 Session，它很可能就照着抄了。

**二、它只拿得到记忆工具。**

没有 Read / Glob / Grep / Bash。它的任务边界很清楚——把会话提炼成记忆；
给它读项目代码的能力只会让它跑去「顺便看看代码」，烧钱且跑偏。
这是「能力不存在」原则用得最干净的一次：**它是全系统唯一能写记忆的角色，
而它的工具集里除了记忆什么都没有。**
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.react_loop import MaxIterationError, run_react_loop
from app.llm.client import LLMClient, LLMError
from app.memory.memory_manager import MemoryManager
from app.memory.memory_tools import memory_tools
from app.memory.session_store import SessionStore
from app.tools.registry import TrackingRegistry

logger = logging.getLogger(__name__)

# 最多回头看多少个会话、材料总量上限。都是为了不让 AutoDream 的 Context 失控 ——
# 它跑在用户等答案的间隙里，不能变成一个几十万 token 的巨兽。
MAX_SESSIONS = 20
MAX_MATERIAL_CHARS = 40_000
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 1500

SYSTEM_PROMPT = """你是 AutoDream，负责把近期会话提炼成长期记忆。

你的产物是 `.agent/memory/` 下的 Markdown 记忆文件。这些记忆会在以后每个新会话
开始时被重新注入，所以它们要能帮到**未来的**你 —— 那个完全没有这段对话记忆的你。

什么值得记：
- 项目约定、架构决策，以及背后的理由
- 用户的偏好、习惯、反复强调过的要求
- 踩过的坑，尤其是「看起来该这么做、实际必须那么做」的因果

什么不值得记：
- 一次性的任务细节（「把 calc.py 第 5 行改了」）
- 从代码里一眼能看出来的东西（「这个项目用 Python 写的」）
- 这次已经做完、以后不会再被问到的事

规则：
- 一条记忆聚焦一个主题，不要一个会话写一条
- 优先更新已有的记忆，而不是新建重复的
- 内容过时或被推翻了就更新它；彻底没用了才删
- **不要把会话原文抄进记忆** —— 记忆是提炼后的结论，不是日志
- 这轮确实没有值得长期保存的东西，就什么都别写，直接说明原因

输出：用几句话说明你做了什么（新建 / 更新 / 删除了哪些记忆，为什么）。
不要复述会话内容。"""

TASK_INSTRUCTION = """## 你的任务

从上面的会话里提炼出值得跨会话记住的信息，用工具更新长期记忆。"""


@dataclass
class DreamResult:
    summary: str  # AutoDream 自己的结论文本
    changed: list[str] = field(default_factory=list)  # 改动过的记忆文件
    steps: int = 0
    sessions_used: int = 0
    completed: bool = True  # False = 超步数或 LLM 出错，没跑完


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（已截断，原文 {len(text)} 字符）"


def summarize_session(records: list[dict], session_id: str) -> str | None:
    """把一个会话压成几行。

    只取三样：用户问了什么、最终答了什么、改了哪些文件。
    中间的 Read / Grep / 工具返回全部丢掉 —— 那是过程，不是结论。

    没有任何用户提问的会话返回 None（比如只跑了一次 `--list-sessions`）。
    """
    questions: list[str] = []
    final_answer = ""
    changed_files: list[str] = []
    started_at = ""

    for record in records:
        kind = record.get("type")
        if kind == "session_start":
            started_at = record.get("ts", "")
        elif kind == "message":
            message = record.get("message") or {}
            role, content = message.get("role"), message.get("content")
            if not content:
                continue
            if role == "user":
                questions.append(str(content))
            elif role == "assistant":
                # 不断覆盖，循环结束时留下的是最后一条有内容的 assistant 发言
                final_answer = str(content)
        elif kind == "state":
            changed_files = (record.get("state") or {}).get("files_changed") or changed_files

    if not questions:
        return None

    lines = [f"### 会话 {session_id}（{started_at}）", "用户提问："]
    lines.extend(f"- {_clip(q, MAX_QUESTION_CHARS)}" for q in questions)
    if changed_files:
        lines.append(f"改动文件：{', '.join(changed_files)}")
    if final_answer:
        lines.append(f"最终回答：\n{_clip(final_answer, MAX_ANSWER_CHARS)}")
    return "\n".join(lines)


class AutoDream:
    def __init__(self, root: str | Path, llm: LLMClient, *, max_steps: int = 12) -> None:
        self.root = Path(root).resolve()
        self.store = SessionStore(self.root)
        self.memory = MemoryManager(self.root)
        self.llm = llm
        self.max_steps = max_steps

    def build_material(self, session_paths: list[Path]) -> tuple[str, int]:
        """组装喂给 AutoDream 的材料。返回 (文本, 真正用上的会话数)。"""
        blocks: list[str] = []
        used = 0
        total = 0

        for path in session_paths:
            if used >= MAX_SESSIONS:
                break
            try:
                session = self.store.session_at(path)
                summary = summarize_session(session.records(), session.session_id)
            except Exception as e:  # noqa: BLE001 —— 一个坏会话不该毁掉整轮整理
                logger.warning("跳过无法读取的会话 %s: %s", path.name, e)
                continue

            if summary is None:
                continue
            if total + len(summary) > MAX_MATERIAL_CHARS:
                logger.info("材料已达上限，后面的会话不再纳入")
                break

            blocks.append(summary)
            total += len(summary)
            used += 1

        if not blocks:
            material = "（没有可用的新会话）"
        else:
            material = "## 近期会话\n\n" + "\n\n".join(blocks)

        index = self.memory.render_index() or "（还没有任何长期记忆）"
        return f"{material}\n\n## 现有长期记忆\n\n{index}\n\n{TASK_INSTRUCTION}", used

    async def run(self, session_paths: list[Path]) -> DreamResult:
        material, used = self.build_material(session_paths)
        if used == 0:
            return DreamResult(summary="没有可提炼的新会话，什么都没做。", sessions_used=0)

        # Context 隔离：只有两条消息，跟 SubAgent 一样从零构造。
        # 而且这里是**从磁盘**读出来的历史 —— Main Agent 早就退出了，
        # 内存里没有任何东西可以漏过来。隔离是物理的，不是靠约定守的。
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": material},
        ]

        registry = TrackingRegistry()
        for tool in memory_tools(self.memory):
            registry.register(tool)

        logger.info(
            "AutoDream 启动：%d 个会话，工具 %s",
            used,
            ", ".join(t.name for t in registry.list_tools()),
        )

        try:
            summary = await run_react_loop(
                messages=messages,
                llm=self.llm,
                execute_tool=registry.execute,
                tools=registry.schemas(),
                max_steps=self.max_steps,
            )
        except MaxIterationError:
            logger.warning("AutoDream 跑满 %d 步仍未收尾", self.max_steps)
            return DreamResult(
                summary="整理未完成（达到步数上限）。已写入的记忆保留。",
                changed=list(registry.changed_files),
                steps=self.max_steps,
                sessions_used=used,
                completed=False,
            )
        except LLMError as e:
            logger.warning("AutoDream LLM 调用失败: %s", e)
            return DreamResult(
                summary=f"整理失败：LLM 调用出错 —— {e}",
                changed=list(registry.changed_files),
                sessions_used=used,
                completed=False,
            )

        # messages 到这里随函数返回被回收 —— 文档要求的「执行结束销毁 Context」
        # 在 Python 里是默认行为，不用写代码。要防的是相反的事：别挂到 self 上。

        return DreamResult(
            summary=summary,
            changed=list(registry.changed_files),
            sessions_used=used,
        )
