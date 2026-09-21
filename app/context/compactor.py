"""上下文压缩。

## 要解决什么

`messages` 只增不减。跑几十轮之后，每一轮都要把**完整历史**重新发一遍 ——
token 随轮数往平方上走，而且迟早撞上模型的上下文上限直接报错。

## 什么时候压

用 `llm.last_prompt_tokens`，也就是**上一次调用真实的 prompt 大小**。
不用字符数估算 token：7a 之后我们已经拿得到准确数字了，免费的准确值没理由不用。

## 压什么

    [system] + [要压掉的一大段] + [最近 min_keep 条原样保留]
                    ↓ LLM 摘要
    [system] + [摘要] + [最近 min_keep 条]

## 一条硬约束：切点不能落在 tool 消息上

tool 消息必须紧跟它归属的那条 assistant（靠 tool_call_id 配对）。
把 assistant 压掉而 tool 还留着，API 直接 400。

所以优先切在 `user` 消息上 —— 那是对话轮次的自然边界。找不到合适的 user 就退一步，
切在任何非 tool 的位置。**长时间单轮任务**（一路 Read/Grep 几十次、中间没有新的
user 发言）靠的就是这条退路，而那恰好是最需要压缩的场景。

## 摘要进了谁的 Context

摘要那次调用是**独立**的：给它的是「要压掉的那段」，它返回一段文字，没有工具、
没有循环。它不共享 Agent 的 Context，也不往里面写 —— 和 SubAgent / AutoDream
是同一个思路。
"""

import logging
from dataclasses import dataclass
from typing import Callable

from app.llm.client import LLMClient, LLMError
from app.llm.content import content_to_text

logger = logging.getLogger(__name__)

DEFAULT_MIN_KEEP = 8
DEFAULT_THRESHOLD_TOKENS = 40_000

# 摘要消息的标记。恢复会话时靠它把这条件认出来 ——
# API 只认 system/user/assistant/tool 四种角色，没有「摘要」这个角色可用，
# 所以只能借 user 并用文字标记出来
SUMMARY_TAG = "[此前对话的摘要]"

# 摘要回调：压缩发生时被调用一次，参数是 (摘要正文, 保留了多少条)
CompactionHook = Callable[[str, int], None]

SUMMARIZE_PROMPT = """你负责压缩一段 Agent 的对话历史。

下面是这段历史。请把它压缩成一段紧凑的说明，供 Agent 后续继续工作时参考。

必须保留：
- 用户的目标，以及明确提出的要求和约束
- 已经做过的决定，以及**为什么**这么做（理由往往比结论重要）
- 改过哪些文件、改成了什么
- 已知的事实：文件位置、函数名、接口、发现的问题
- 还没解决的问题，以及踩过的坑

可以丢弃：
- 工具调用的原始输出（读到的文件全文、搜索结果列表）
- 已经放弃的中间思路

要求：
- 用陈述句直接写事实，不要「用户说」「助手认为」这种转述
- **保留具体的文件名、函数名、路径、数字** —— 它们是后续工作的锚点，
  笼统地说「改了几个文件」等于没记
- 不要编造历史里没有的内容。不确定的地方宁可不写
- 直接输出摘要正文，不要任何前言后语"""


@dataclass(frozen=True)
class Compaction:
    summary: str
    keep_count: int  # 摘要之后保留了多少条原始消息


def split_point(messages: list[dict], min_keep: int) -> int | None:
    """找出「从哪一条开始保留」。返回 None 表示这次没法压。

    从后往前找，尽量多压 —— 但要保证保留的部分不少于 min_keep 条，
    且切点上的那条**不能是 tool 消息**。
    """
    fallback: int | None = None

    # 从 len-min_keep 往前退，下标至少到 2（给 system 和至少一条要压的消息留位置）
    for index in range(len(messages) - min_keep, 1, -1):
        role = messages[index].get("role")
        if role == "tool":
            continue  # 硬约束：切在这里会让 tool 消息失去归属的 assistant
        if role == "user":
            return index  # 轮次的自然边界，优先
        if fallback is None:
            fallback = index  # 退路：非 tool 就行（长单轮任务靠这条）

    return fallback


class ContextCompactor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS,
        min_keep: int = DEFAULT_MIN_KEEP,
        on_compaction: CompactionHook | None = None,
    ) -> None:
        self.llm = llm
        self.threshold_tokens = threshold_tokens
        self.min_keep = min_keep
        self._on_compaction = on_compaction
        self.count = 0  # 压过几次，供日志和测试观察

    async def maybe_compact(self, messages: list[dict]) -> Compaction | None:
        """该压就压，就地改 messages；返回本次压缩的结果，没压则返回 None。

        **就地修改**和 ReAct Loop 保持一致：调用方持有的那个 list 就是 Context 本身。
        """
        # 先存下来：下面那次摘要调用会把 last_prompt_tokens 覆盖成**摘要请求**
        # 的大小，之后再读它就成了「摘要用了多少 token」，而不是「被压缩的上下文多大」
        prompt_tokens = getattr(self.llm, "last_prompt_tokens", 0)
        if prompt_tokens < self.threshold_tokens:
            return None

        cut = split_point(messages, self.min_keep)
        if cut is None:
            logger.info(
                "上下文已达 %d tokens，但找不到安全的切分点，这次不压", prompt_tokens
            )
            return None

        # 开头的 system 要留着，其余交给摘要。之前压出来的摘要也在这段里面，
        # 所以会被折进新的摘要 —— 反复压缩是滚雪球，不会各压各的
        offset = 1 if messages and messages[0].get("role") == "system" else 0
        to_summarize = messages[offset:cut]
        total_before = len(messages)

        try:
            summary = await self._summarize(to_summarize)
        except LLMError as e:
            # 摘要失败不该毁掉整个任务 —— 不压就是了，下一轮还会再试
            logger.warning("上下文摘要失败，本轮不压缩: %s", e)
            return None

        keep_count = len(messages) - cut
        # prefix 就是开头的 system，必须留住 —— 它是整个会话的角色和规则，
        # 丢了它 Agent 会在压缩之后「忘记自己是谁」。切点之前的部分（要压掉的那段）
        # 换成了摘要，prefix 不在其中，所以得单独拼回去
        messages[:] = (
            messages[:offset]
            + [{"role": "user", "content": f"{SUMMARY_TAG}\n{summary}"}]
            + messages[cut:]
        )
        self.count += 1

        logger.info(
            "已压缩上下文：%d 条消息 -> 1 条摘要 + 保留 %d 条（压缩前 prompt %d tokens）",
            total_before,
            keep_count,
            prompt_tokens,
        )

        result = Compaction(summary=summary, keep_count=keep_count)
        if self._on_compaction is not None:
            self._on_compaction(result.summary, result.keep_count)
        return result

    async def _summarize(self, messages: list[dict]) -> str:
        """把一段消息交给 LLM 压成文字。

        不带 tools —— 这是纯文本任务，给它工具只会引入跑偏的可能。
        """
        transcript = _render_transcript(messages)
        reply = await self.llm.chat(
            [
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": transcript},
            ]
        )
        return (reply.get("content") or "").strip()


def _render_transcript(messages: list[dict]) -> str:
    """把消息渲染成给人读的文本。

    为什么要自己渲染而不是直接 json.dumps：工具调用和返回的原始结构对摘要模型
    是噪声，转成「谁说了什么」这种朴素格式，它更容易抓住重点。
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "?")
        # content 可能是个块列表（带图片的 user 消息）。必须渲染成纯文本，
        # 否则这里会把这个列表的 repr 整个拼进摘要 prompt —— 带上图片块就是
        # 几百 KB。图片只留一个 [图片] 占位
        content = content_to_text(message.get("content"))
        if content:
            lines.append(f"【{role}】{content}")
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            lines.append(f"【{role} 调用工具】{function.get('name')}({function.get('arguments')})")
    return "\n\n".join(lines)
