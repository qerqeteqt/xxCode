"""每轮对话后提取长期记忆。

## 它和 AutoDream 的分工

    MemoryExtractor   每轮跑       负责「**提取**」—— 把刚出现的事实写进记忆
    AutoDream         攒够条件才跑  负责「**整理**」—— 合并、去重、更新过时

缺了前者，「记住 X」就是句空话：模型只能口头答应「好的我记住了」，
然后等 AutoDream 哪天攒够条件去会话流水里翻。缺了后者，记忆会越攒越碎。

## 它只能碰记忆

工具集里只有 ReadMemory / WriteMemory / UpdateMemory —— **没有 Delete，
也没有任何能碰业务代码的东西**。这是「能力不存在」那条原则的又一次应用：
让它在结构上就没法改代码，比在 prompt 里叮嘱「不要修改业务代码」可靠。

不给 Delete 是因为「这条记忆是不是真的没用了」需要全局视野，那是 AutoDream 的活。

## 成本

每轮一次 LLM 调用。所以喂给它的**不是整个 transcript**，而是本轮的三样东西：
用户说了什么、最后答了什么、动了哪些文件。中间的 Read/Grep 过程对
「有什么值得长期记住的」毫无价值，喂进去只是白烧 token —— 和 AutoDream
的会话摘要是同一个思路。
"""

import logging
from dataclasses import dataclass, field

from app.agent.react_loop import MaxIterationError, run_react_loop
from app.llm.client import LLMClient, LLMError
from app.memory.memory_manager import MemoryManager
from app.memory.memory_tools import ReadMemoryTool, UpdateMemoryTool, WriteMemoryTool
from app.tools.registry import TrackingRegistry

logger = logging.getLogger(__name__)

MAX_STEPS = 6
MAX_ANSWER_CHARS = 1500

SYSTEM_PROMPT = """你负责从刚刚结束的一轮对话里提取值得长期记住的信息。

你的产物是 `.agent/memory/` 下的 Markdown 记忆文件，它们会在以后**每个新会话开始时
被注入**。判断标准只有一条：

    这条信息对「下一个会话的我」有用吗？

值得记：
- 用户明确要求记住的（「记住…」「以后都…」「这个项目的规范是…」）
- 用户的偏好、习惯、约定
- 项目里非显然的事实，尤其是因果（「看起来该这样、实际必须那样」）
- 刚踩过的坑

不值得记：
- 一次性的任务细节（「把 calc.py 第 5 行改了」）
- 从代码里一眼能看出的东西（「项目用 Python」）
- 已经有记忆覆盖的内容 —— 那种情况用 UpdateMemory 改，**不要新建重复的**

**大多数轮次都没有值得记的东西。** 那就什么都别写，直接说「没有可提取的」。
宁可漏掉，也不要写一堆以后没人看的记忆 —— 记忆目录一旦变成垃圾场，
下次注入给模型的索引就全是噪声。
"""


@dataclass
class ExtractResult:
    summary: str
    changed: list[str] = field(default_factory=list)
    completed: bool = True


def summarize_turn(messages: list[dict]) -> str | None:
    """把**最近一轮**压成几行。

    只取三样：用户说了什么、最后答了什么、以及最后一条 assistant 提到的改动。
    中间的 Read / Grep 全部丢掉 —— 那是过程，对「什么值得长期记住」没有价值。

    返回 None 表示这一轮没什么可看的（比如还没答完）。
    """
    start = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            start = index
            break
    if start is None:
        return None

    questions: list[str] = []
    answer = ""
    for message in messages[start:]:
        role, content = message.get("role"), message.get("content")
        if not content:
            continue
        if role == "user":
            questions.append(str(content))
        elif role == "assistant":
            answer = str(content)  # 不断覆盖，留下最后一条

    if not questions or not answer:
        return None  # 还没答完，下一轮再说

    lines = ["## 刚刚这一轮", "用户说："]
    lines.extend(f"- {' '.join(q.split())[:400]}" for q in questions)
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS] + "…（已截断）"
    lines.append(f"最终回答：\n{answer}")
    return "\n".join(lines)


class MemoryExtractor:
    def __init__(self, root: str, llm: LLMClient, *, max_steps: int = MAX_STEPS) -> None:
        self.memory = MemoryManager(root)
        self.llm = llm
        self.max_steps = max_steps

    def build_material(self, messages: list[dict]) -> str | None:
        turn = summarize_turn(messages)
        if turn is None:
            return None

        index = self.memory.render_index() or "（还没有任何长期记忆）"
        return (
            f"{turn}\n\n"
            f"## 现有长期记忆\n\n{index}\n\n"
            f"## 你的任务\n\n"
            f"如果这一轮里有值得长期记住的信息，用工具写进记忆；"
            f"没有就什么都不做，直接说明原因。"
        )

    async def run(self, messages: list[dict]) -> ExtractResult | None:
        material = self.build_material(messages)
        if material is None:
            return None

        # 只给三个工具：能读、能写、能改，**不能删**。
        # 更不能碰业务代码 —— 这个注册表里压根没有文件工具
        registry = TrackingRegistry()
        registry.register(ReadMemoryTool(self.memory))
        registry.register(WriteMemoryTool(self.memory))
        registry.register(UpdateMemoryTool(self.memory))

        logger.info("[memory] 提取长期记忆 | 工具: %s",
                    ", ".join(t.name for t in registry.list_tools()))

        try:
            summary = await run_react_loop(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": material},
                ],
                llm=self.llm,
                execute_tool=registry.execute,
                tools=registry.schemas(),
                max_steps=self.max_steps,
            )
        except MaxIterationError:
            logger.warning("[memory] 提取未完成（达到步数上限）")
            return ExtractResult(
                summary="提取未完成（达到步数上限）",
                changed=list(registry.changed_files),
                completed=False,
            )
        except LLMError as e:
            logger.warning("[memory] 提取失败: %s", e)
            return ExtractResult(summary=f"提取失败：{e}", completed=False)

        return ExtractResult(summary=summary, changed=list(registry.changed_files))
