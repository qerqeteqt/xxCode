"""Phase 7b 上下文压缩测试。

两块最要紧的：
    1. 切点绝不能落在 tool 消息上 —— 那会让 tool 失去归属的 assistant，API 直接 400
    2. 压缩记录能被 JSONL 忠实重放 —— 反复压缩也要对得上
"""

import asyncio
import json

from app.agent.react_loop import MaxIterationError, run_react_loop
from app.context.compactor import (
    SUMMARY_TAG,
    Compaction,
    ContextCompactor,
    split_point,
)
from app.llm.client import LLMError
from app.memory.session_store import SessionStore


def _run(coro):
    return asyncio.run(coro)


def _conversation(turns: int = 3, tools_per_turn: int = 2) -> list[dict]:
    """造一段有 system、user、assistant(带 tool_calls) 和 tool 的对话。"""
    messages: list[dict] = [{"role": "system", "content": "系统提示"}]
    for t in range(turns):
        messages.append({"role": "user", "content": f"问题{t}"})
        for i in range(tools_per_turn):
            call_id = f"c{t}{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "function": {"name": "Read", "arguments": "{}"}}
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": "结果"})
        messages.append({"role": "assistant", "content": f"答案{t}"})
    return messages


class SummarizerLLM:
    """假的 LLM：只用来产出摘要，同时汇报自己的 prompt 有多大。"""

    def __init__(self, summary: str = "压缩后的摘要", *, prompt_tokens: int = 0, fail: bool = False):
        self.last_prompt_tokens = prompt_tokens
        self._summary = summary
        self._fail = fail
        self.calls: list[list[dict]] = []
        self.tools_seen: list = []

    async def chat(self, messages: list[dict], tools: list | None = None, on_delta=None) -> dict:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if self._fail:
            raise LLMError("摘要调用挂了")
        return {"role": "assistant", "content": self._summary}


# ================================================================ 切点


def test_优先切在_user_消息上():
    messages = _conversation(turns=3)

    cut = split_point(messages, min_keep=8)

    assert messages[cut]["role"] == "user"


def test_切点之后不会出现孤立的_tool_消息():
    """这是硬约束：tool 消息必须紧跟它归属的 assistant。"""
    for min_keep in range(1, 15):
        messages = _conversation(turns=4)
        cut = split_point(messages, min_keep)
        if cut is None:
            continue
        assert messages[cut]["role"] != "tool", f"min_keep={min_keep} 切到了 tool 上"


def test_没有_user_可用时退到任何非_tool_位置():
    """长时间单轮任务（一路 Read/Grep、中间没有新的 user 发言）靠的就是这条退路。"""
    messages = [{"role": "system", "content": "系统提示"}]
    messages.append({"role": "user", "content": "帮我查一堆东西"})
    for i in range(6):
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]}
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "结果"})

    cut = split_point(messages, min_keep=2)

    assert messages[cut]["role"] == "assistant"


def test_消息太少时不切():
    assert split_point([{"role": "system"}, {"role": "user"}], min_keep=8) is None


def test_min_keep_越小压得越狠():
    """切点是「保留从哪条开始」—— 越靠后保留得越少，压掉的越多。"""
    messages = _conversation(turns=4)

    cut_keep_little = split_point(messages, min_keep=2)
    cut_keep_much = split_point(messages, min_keep=12)

    assert cut_keep_little is not None and cut_keep_much is not None
    assert cut_keep_little > cut_keep_much
    # 保留的条数确实不少于 min_keep
    assert len(messages) - cut_keep_much >= 12


# ================================================================ 触发与压缩


def test_没到阈值不压():
    llm = SummarizerLLM(prompt_tokens=100)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)
    messages = _conversation()

    assert _run(compactor.maybe_compact(messages)) is None
    assert not llm.calls  # 一次 LLM 都没调
    assert messages[1]["role"] == "user"


def test_到阈值就压():
    llm = SummarizerLLM("摘要正文", prompt_tokens=50_000)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)
    messages = _conversation(turns=3)
    before = len(messages)

    result = _run(compactor.maybe_compact(messages))

    assert isinstance(result, Compaction)
    assert len(messages) < before
    # system 必须留住 —— 它是整个会话的角色和规则，丢了 Agent 会「忘记自己是谁」
    assert messages[0]["role"] == "system"
    assert "系统提示" in messages[0]["content"]
    assert SUMMARY_TAG in messages[1]["content"]
    assert "摘要正文" in messages[1]["content"]
    assert len(messages) == 2 + result.keep_count  # system + 摘要 + 保留的
    assert compactor.count == 1


def test_摘要调用不带工具():
    """这是纯文本任务，给它工具只会引入跑偏的可能。"""
    llm = SummarizerLLM(prompt_tokens=50_000)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)

    _run(compactor.maybe_compact(_conversation()))

    assert llm.tools_seen == [None]


def test_摘要看不到_system_提示():
    """system 是保留的，不该被折进摘要里重复一遍。"""
    llm = SummarizerLLM(prompt_tokens=50_000)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)

    _run(compactor.maybe_compact(_conversation(turns=3)))

    assert "系统提示" not in llm.calls[0][1]["content"]


def test_摘要失败时不动_messages():
    """摘要挂了不该毁掉整个任务 —— 不压就是了，下一轮还会再试。"""
    llm = SummarizerLLM(prompt_tokens=50_000, fail=True)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)
    messages = _conversation()
    snapshot = [dict(m) for m in messages]

    result = _run(compactor.maybe_compact(messages))

    assert result is None
    assert messages == snapshot
    assert compactor.count == 0


def test_找不到切点时不压():
    llm = SummarizerLLM(prompt_tokens=50_000)
    compactor = ContextCompactor(llm, threshold_tokens=40_000)
    messages = [{"role": "system", "content": "系统"}]  # 太短

    assert _run(compactor.maybe_compact(messages)) is None
    assert not llm.calls


def test_压缩回调被调用():
    seen: list[tuple[str, int]] = []
    llm = SummarizerLLM("摘要", prompt_tokens=50_000)
    compactor = ContextCompactor(
        llm, threshold_tokens=40_000, on_compaction=lambda s, k: seen.append((s, k))
    )

    result = _run(compactor.maybe_compact(_conversation()))

    assert seen == [("摘要", result.keep_count)]


def test_反复压缩时旧摘要被折进新摘要():
    llm = SummarizerLLM("第二份摘要", prompt_tokens=50_000)
    compactor = ContextCompactor(llm, threshold_tokens=40_000, min_keep=4)
    messages = _conversation(turns=3)

    _run(compactor.maybe_compact(messages))
    first_summary = messages[1]["content"]
    _run(compactor.maybe_compact(messages))

    assert compactor.count == 2
    # 第一份摘要进了第二次摘要的输入
    assert first_summary in llm.calls[1][1]["content"]
    assert "第二份摘要" in messages[1]["content"]


# ================================================================ 循环集成


def test_循环里会在每次调用前检查压缩():
    class ReplyLLM(SummarizerLLM):
        """既当摘要器又当正式回复 —— 真环境里本来就是同一个 client。"""

        async def chat(self, messages, tools=None, on_delta=None):
            self.calls.append([dict(m) for m in messages])
            self.tools_seen.append(tools)
            return {"role": "assistant", "content": "完成"}

    reply_llm = ReplyLLM("摘要", prompt_tokens=50_000)
    compactor = ContextCompactor(reply_llm, threshold_tokens=40_000)
    messages = _conversation(turns=3)
    messages.append({"role": "user", "content": "接着干"})

    answer = _run(
        run_react_loop(
            messages, reply_llm, _never_called, max_steps=10, compactor=compactor
        )
    )

    assert answer == "完成"
    assert compactor.count == 1
    # 摘要和正式回复共用同一个 client，所以 calls[0] 是摘要调用、calls[1] 才是回复
    assert SUMMARY_TAG not in reply_llm.calls[0][1]["content"]
    reply_prompt = reply_llm.calls[1]
    assert SUMMARY_TAG in reply_prompt[1]["content"]
    assert reply_prompt[0]["role"] == "system"


async def _never_called(tool_call: dict) -> str:
    raise AssertionError("这个用例不该调用工具")


# ================================================================ JSONL 重放


def test_压缩记录能被重放(tmp_path):
    session = SessionStore(tmp_path).create()
    for i in range(6):
        session.append_message({"role": "user", "content": f"问{i}"})
    session.record_compaction("这是摘要", keep_count=2)
    session.append_message({"role": "assistant", "content": "之后"})

    messages = session.load_messages()

    assert len(messages) == 4
    assert SUMMARY_TAG in messages[0]["content"]
    assert "这是摘要" in messages[0]["content"]
    assert messages[1]["content"] == "问4"
    assert messages[2]["content"] == "问5"
    assert messages[3]["content"] == "之后"


def test_压缩记录有版本号和类型(tmp_path):
    session = SessionStore(tmp_path).create()
    session.record_compaction("摘要", keep_count=3)

    record = [
        json.loads(line)
        for line in session.path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "compaction"
    ][0]

    assert record["v"] == 1
    assert record["keep_count"] == 3
    assert record["summary"] == "摘要"


def test_反复压缩的重放也对得上(tmp_path):
    """重放是按发生顺序走一遍，所以压几次都应该还原出当时的状态。"""
    session = SessionStore(tmp_path).create()
    for i in range(6):
        session.append_message({"role": "user", "content": f"问{i}"})
    session.record_compaction("第一份", keep_count=2)
    session.append_message({"role": "assistant", "content": "中间"})
    session.record_compaction("第二份", keep_count=2)

    messages = session.load_messages()

    assert "第二份" in messages[0]["content"]
    assert len(messages) == 3  # 摘要 + 最后两条（中间、问5）


def test_keep_count_为零时只留摘要(tmp_path):
    """`messages[-0:]` 会返回整个列表 —— 这个坑要挡住。"""
    session = SessionStore(tmp_path).create()
    for i in range(3):
        session.append_message({"role": "user", "content": f"问{i}"})
    session.record_compaction("全压掉了", keep_count=0)

    messages = session.load_messages()

    assert len(messages) == 1
    assert "全压掉了" in messages[0]["content"]


def test_恢复出来的_messages_能直接喂给_LLM(tmp_path):
    """压缩之后仍然要满足 API 的形状约束：没有孤立的 tool 消息。"""
    session = SessionStore(tmp_path).create()
    for i in range(4):
        session.append_message({"role": "user", "content": f"问题{i}"})
        session.append_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "function": {"name": "Read", "arguments": "{}"}}],
            }
        )
        session.append_message({"role": "tool", "tool_call_id": f"c{i}", "content": "结果"})
    session.record_compaction("摘要", keep_count=3)

    messages = session.load_messages()

    # 第一条是摘要，后面每条 tool 都能在自己的前面找到归属的 assistant
    assert messages[0]["role"] == "user"
    seen_ids: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                seen_ids.add(call["id"])
        elif message["role"] == "tool":
            assert message["tool_call_id"] in seen_ids, "tool 消息失去了归属"


def test_超步数之类的旧会话不受影响(tmp_path):
    """没压过的会话照旧是完整流水。"""
    session = SessionStore(tmp_path).create()
    session.append_message({"role": "user", "content": "问"})
    session.append_message({"role": "assistant", "content": "答"})

    assert session.load_messages() == [
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "答"},
    ]
