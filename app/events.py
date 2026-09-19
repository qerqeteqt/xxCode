"""结构化事件 —— Runtime 给外部看的通道。

## 它和 Context 是两条不同的通道

    Context（给模型看）   有隔离：SubAgent 的中间过程不进 Main 的上下文
    事件流（给人看）       不隔离：全都推给浏览器

这两件事不是一回事，也不该是一回事。做 Web 之前，"给人看"只有日志这一条通道，
但日志是纯文本、结构化不了 —— 前端没法把一行 `[tool] Read({"path": ...})`
渲染成一张可折叠的卡片。所以这里补一个显式的事件类型。

## 为什么用回调而不是队列

事件在哪产生（工具执行、循环）和它去哪（SSE、日志、什么都不做）是两回事。
回调让产生方完全不知道消费方的存在 —— 和 `execute_tool`、`llm` 的注入是同一个套路。
CLI 不传就什么都不发，一行都不用改。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


EventHook = Callable[[Event], None]

# 工具返回可能很长（一整个文件），推给浏览器之前截一下。
# 8000 和 BashTool 的 MAX_OUTPUT_CHARS 保持一致 —— 反正模型那边也只看到这么多
TEXT_LIMIT = 8000


def emit(hook: EventHook | None, type_: str, **data: Any) -> None:
    """有 hook 就发，没有就算了。

    调用点不用到处写 `if hook is not None` —— 那种噪音会让人懒得加事件。
    """
    if hook is not None:
        hook(Event(type_, data))


def tag_events(hook: EventHook | None, source: str) -> EventHook | None:
    """给事件盖一个来源戳，让前端知道这条是谁发的。

    **事件流和 Context 是两条通道**：Context 有隔离（子 Agent 的中间过程
    不进 Main 的上下文），但事件流是给人看的，**不隔离** —— 你在网页上
    应该看得到子 Agent 在翻什么文件，否则它就成了一个黑盒。
    这是很早以前就说过的那件事，现在在这里落地。
    """
    if hook is None:
        return None

    def tagged(event: Event) -> None:
        hook(Event(event.type, {**event.data, "source": source}))

    return tagged
