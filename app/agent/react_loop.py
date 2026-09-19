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

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from app.llm.client import LLMClient

if TYPE_CHECKING:
    from app.context.compactor import ContextCompactor

logger = logging.getLogger(__name__)

# 执行一次 tool_call，返回要回灌给模型的字符串结果。
# 参数是 API 原样的 tool_call 对象：{"id":..., "type":"function",
# "function": {"name":..., "arguments": "<JSON 字符串>"}}
# 注意 arguments 是字符串不是 dict —— 解析它是 ToolRegistry 的职责（Phase 2），
# 不在这里，因为 Loop 不该知道任何工具的参数长什么样。
ExecuteTool = Callable[[dict], Awaitable[str]]

# 每往 messages 里追加一条消息就回调一次。Session 靠它做 append-only 落盘。
MessageHook = Callable[[dict], None]


class MaxIterationError(RuntimeError):
    """循环次数用完仍未得出最终答案。"""

    def __init__(self, max_steps: int) -> None:
        super().__init__(f"达到最大循环次数 {max_steps}，仍未得出最终答案")
        self.max_steps = max_steps


async def run_react_loop(
    messages: list[dict],
    llm: LLMClient,
    execute_tool: ExecuteTool,
    tools: list[dict] | None = None,
    max_steps: int = 10,
    on_message: MessageHook | None = None,
    compactor: "ContextCompactor | None" = None,
) -> str:
    """驱动 ReAct 循环，返回模型的最终回答。

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
        if compactor is not None:
            # 压缩是就地改 messages 的，所以 record 的调用方（Session）不会
            # 收到「消息被删了」的通知 —— 它靠 compactor 自己的回调写 compaction 记录
            await compactor.maybe_compact(messages)

        msg = await llm.chat(messages, tools=tools)

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

        # assistant 这条必须原样回灌，否则下一步的 tool 消息没有归属的 tool_call_id，
        # API 会直接报 400。
        record(msg)

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

    raise MaxIterationError(max_steps)
