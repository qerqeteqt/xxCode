"""ReAct Loop —— 不认识任何具体 Tool 的循环。

循环本身只有四步：

    1. 把当前 messages 交给 LLM
    2. 返回里没有 tool_calls  -> 它就是最终答案，结束
    3. 返回里有 tool_calls    -> 逐个交给 execute_tool 执行，
                                 把结果作为 role="tool" 的消息追加进 messages
    4. 回到第 1 步

它不知道 Calculator / FileTool / BashTool 的存在，只知道「有一只能执行 tool_call
的异步函数」，而这个函数是调用方注入进来的：

    Phase 1  -> 一个占位实现（还没注册任何工具）
    Phase 2  -> ToolRegistry.execute

所以 Phase 2 落地 Tool 时，这个文件一行都不用改。这就是文档第四节说的
「ReAct Loop 独立于具体 Tool」的落地方式：依赖方向是 Loop <- 调用方，
而不是 Loop -> Registry。
"""

import asyncio
import logging
from collections import Counter
from typing import TYPE_CHECKING, Awaitable, Callable

from app.llm.client import LLMClient

from app.events import EventHook, emit

if TYPE_CHECKING:
    from app.context.compactor import ContextCompactor
    from app.llm.client import DeltaHook

logger = logging.getLogger(__name__)

# 执行一次 tool_call，返回要回灌给模型的字符串结果。
# 参数是 API 原样的 tool_call 对象：{"id":..., "type":"function",
# "function": {"name":..., "arguments": "<JSON 字符串>"}}
# 注意 arguments 是字符串不是 dict —— 解析它是 ToolRegistry 的职责（Phase 2），
# 不在这里，因为 Loop 不该知道任何工具的参数长什么样。
ExecuteTool = Callable[[dict], Awaitable[str]]

# 每往 messages 里追加一条消息就回调一次。Session 靠它做 append-only 落盘。
MessageHook = Callable[[dict], None]

# 被用户中止、没能跑完的工具，回灌给模型的占位结果。
# 文字要写清楚「结果未知」—— 模型下一轮会看到它，不能让它以为工具成功返回了空
ABORTED_TOOL_RESULT = "（用户中止了这一轮：这个工具没有执行完，结果未知）"


def last_progress(messages: list[dict]) -> str:
    """给「跑到上限」用的收尾报告。

    **只报最后一句助手发言是不够的。** 实测里模型经常十几步都只调工具、
    不说话（一路 pytest → 改 → 再 pytest），那时最后一条有文字的消息
    可能是**第 1 步**的开场白 —— 报出来等于没说：
    「它中断前的最后进展：我先看一下项目结构和 tests 目录的现状。」

    所以主体是**工具活动**（那才是「做了多少」的事实），
    模型最后说的话作为补充（有才加）。
    """
    lines: list[str] = []

    counts: Counter[str] = Counter()
    for message in messages:
        for call in message.get("tool_calls") or []:
            counts[call.get("function", {}).get("name", "?")] += 1
    if counts:
        detail = "、".join(f"{name}×{n}" for name, n in counts.most_common())
        lines.append(f"调用了 {sum(counts.values())} 次工具：{detail}")

    action = _last_action(messages)
    if action:
        name, args = action
        lines.append(f"最后一步：{name} {args[:160]}")

    text = _last_assistant_text(messages)
    if text:
        lines.append(f"它最后说的话：\n{text}")

    return "\n".join(lines)


def _last_action(messages: list[dict]) -> tuple[str, str] | None:
    """最后一个工具调用的（名字，参数原文）。"""
    for message in reversed(messages):
        calls = message.get("tool_calls")
        if calls:
            function = calls[-1].get("function", {})
            return function.get("name", "?"), str(function.get("arguments") or "")
    return None


def _last_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"]).strip()
    return ""


class MaxIterationError(RuntimeError):
    """循环次数用完仍未得出最终答案。"""

    def __init__(self, max_steps: int, note: str = "", partial: str = "") -> None:
        # note 是给调用方补充说明用的（比如「过程已保存，可以 --continue」）。
        # 它是**参数**而不是让调用方再包一层 —— 直接构造一个新的
        # MaxIterationError("一大段文字") 会把那段文字塞进 max_steps 的位置，
        # 消息就被套成「达到最大循环次数 达到最大循环次数 10，…」
        super().__init__(f"达到最大循环次数 {max_steps}，仍未得出最终答案{note}")
        self.max_steps = max_steps
        self.partial = partial


async def run_react_loop(
    messages: list[dict],
    llm: LLMClient,
    execute_tool: ExecuteTool,
    max_steps: int,
    tools: list[dict] | None = None,
    on_message: MessageHook | None = None,
    compactor: "ContextCompactor | None" = None,
    on_delta: "DeltaHook | None" = None,
    on_event: EventHook | None = None,
) -> str:
    """驱动 ReAct 循环，返回模型的最终回答。

    **`max_steps` 是必填的，没有默认值。** 一个通用循环不该替调用方决定跑几步 ——
    之前这里有个默认的 10，和 `settings.max_steps`（60）并存，读代码的人会
    以为上限是 10。每个调用方的合理预算本来就不同（Main 要 60，Explore 8 步够了，
    提取记忆 6 步），所以"必须显式给"才是对的。


    返回后 messages 里包含完整的一轮交互（含最初的 system / user），
    直到最后那条 assistant 最终回答 —— 它是**就地修改**的，不是复制。

    这一点是刻意的：这个 list 就是 Context 本身。调用方拿到的就是跑完之后的完整历史，
    Phase 4 的 Session 持久化、以及多轮对话的「接着上次聊」，要的正是它。

    on_message 每追加一条消息就回调一次。为什么需要这个口子：循环是**就地修改**
    messages 的，调用方在循环外面看不到中间追加了什么。没有它就只能等整个任务
    跑完再一次性落盘 —— 那就丢掉了 JSONL 唯一的优势：进程崩了，已经发生的还在。

    compactor 是可选的上文压缩器。放在循环里而不是调用方，是因为它必须在
    **每一次 LLM 调用之前**检查 —— 上下文是在循环内部一步步长起来的，
    等到循环结束再压就已经晚了（那一次调用可能已经直接撞上上限报错）。
    """
    def record(message: dict) -> None:
        messages.append(message)
        if on_message is not None:
            on_message(message)

    for step in range(1, max_steps + 1):
        # 步数也发成事件 —— CLI 靠它打日志，Web 靠它画「第 N 步」的分隔线
        emit(on_event, "step", step=step, max_steps=max_steps)

        if compactor is not None:
            # 压缩是就地改 messages 的，所以 record 的调用方（Session）不会
            # 收到「消息被删了」的通知 —— 它靠 compactor 自己的回调写 compaction 记录
            await compactor.maybe_compact(messages)

        msg = await llm.chat(messages, tools=tools, on_delta=on_delta)

        # 流式输出没有结尾换行。不补的话，紧接着的日志会粘在回答末尾
        # （实测：「…说完。21:12:28 INFO app.agent.react_loop | step 1/20: …」）。
        # 必须放在分支**之前** —— 放在某条分支里，另一条路就走不到了
        if on_delta is not None and msg.get("content"):
            on_delta("\n")

        tool_calls = msg.get("tool_calls")

        # 唯一的终止条件。
        # 这里是 API 给出的结构化事实，不存在「模型格式写歪了」这件事。
        if not tool_calls:
            logger.info("step %d/%d: 无 tool_calls，输出最终答案", step, max_steps)
            # 最终回答也要进历史。否则「messages 就是完整会话」这个不变量就破了，
            # 多轮对话时下一轮会看不到上一轮回答了什么。
            record(msg)
            return msg.get("content") or ""

        logger.info(
            "step %d/%d: %d 个 tool_call -> %s",
            step,
            max_steps,
            len(tool_calls),
            ", ".join(
                tc.get("function", {}).get("name", "?") for tc in tool_calls
            ),
        )

        # 模型每步说的话，就是它的判断和计划。**不打出来用户只能看到一串工具调用**，
        # 完全不知道它在想什么 —— 跑偏了看不出来，也没法判断该不该叫停。
        # 这是「20 步跑完却看不懂发生了什么」的直接原因。
        #
        # 流式的时候这段话已经一个字一个字显示过了，再打一遍就是重复。
        if msg.get("content") and on_delta is None:
            logger.info("[模型] %s", str(msg["content"]).strip())

        # assistant 这条必须原样回灌，否则下一步的 tool 消息没有归属的 tool_call_id，
        # API 会直接报 400。
        record(msg)

        # 已经拿到结果、回灌过的 tool_call_id。取消时靠它找出「哪几个还没应答」
        done_ids: set[str] = set()
        try:
            for tool_call in tool_calls:
                name = tool_call.get("function", {}).get("name")

                # 工具崩了不能让整个 Runtime 崩。异常转成字符串回灌给模型，
                # 模型看到报错通常能自己换个思路重试 —— 这是 ReAct 自愈能力的一部分。
                try:
                    result = await execute_tool(tool_call)
                except Exception as e:  # noqa: BLE001 —— 故意兜住所有工具异常
                    result = f"工具执行出错: {type(e).__name__}: {e}"

                record(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    }
                )
                done_ids.add(tool_call["id"])
        except asyncio.CancelledError:
            # 用户按了停止，取消正落在这几个工具中间。**必须把每一个还没有结果的
            # tool_call 都补上一条**：上面那条 assistant 已经带着 tool_calls 落盘了，
            # 少了应答它的 tool 消息，历史就是坏的。而 Session 是 append-only 的 ——
            # 坏历史已经写进 JSONL，删不掉，下一次调用（或 --continue）拿到它
            # API 会直接 400。
            #
            # 用 except 而不是 finally：只有取消才该补，正常路径和工具自己报错
            # 都不该被伪造出结果。
            #
            # 补完仍然要原样抛出。CancelledError 继承 BaseException，上面的
            # except Exception 兜不住它，这里吞掉则会破坏 Task 的取消语义。
            for tool_call in tool_calls:
                if tool_call["id"] in done_ids:
                    continue
                record(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": ABORTED_TOOL_RESULT,
                    }
                )
            raise

    # 撞上限时把最后的进展带上 —— 那通常不是一无所获，
    # 而是「查了一大堆但没来得及收尾」
    raise MaxIterationError(max_steps, partial=last_progress(messages))
