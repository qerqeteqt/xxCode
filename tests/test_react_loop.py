"""ReAct Loop 的单元测试 —— 全程不碰真 LLM。

Loop 依赖的两个外部东西（LLM、execute_tool）都是注入进来的，所以这里塞两个假的，
就能把循环的行为单独测干净：终止判断、一轮多个 tool_call、工具异常回灌、超步数。

如果 Loop 当初写死了 httpx.post 和 TOOL_FUNCS[name]，这些测试就只能靠真网络和真工具，
既慢又不稳定 —— 这就是「注入」这个设计换来的直接好处。

异步测试统一用 asyncio.run() 在同步函数里跑，不引入 pytest-asyncio 插件。
"""

import asyncio
import json

import pytest

from app.agent.react_loop import MaxIterationError, run_react_loop

SYSTEM = {"role": "system", "content": "你是助手"}
USER = {"role": "user", "content": "问题"}


def _tool_call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


def _assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class FakeLLM:
    """按脚本依次吐出预置回复，并记录每轮收到的 messages 快照。"""

    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []
        self.tools_seen: list = []

    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        # 存快照，否则后续原地追加会污染已记录的调用现场
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self._replies:
            raise AssertionError("回复脚本用完了 —— 循环比预期多跑了一轮")
        return self._replies.pop(0)


class FakeTools:
    """假的工具执行器：记录被调用过什么，按预设返回结果或抛异常。"""

    def __init__(self, results: dict | None = None, raises: dict | None = None) -> None:
        self._results = results or {}
        self._raises = raises or {}
        self.received: list[dict] = []

    async def __call__(self, tool_call: dict) -> str:
        name = tool_call["function"]["name"]
        self.received.append(tool_call)
        if name in self._raises:
            raise self._raises[name]
        return self._results.get(name, f"{name} 的结果")


def _run(coro):
    return asyncio.run(coro)


def test_returns_final_answer_when_no_tool_call():
    """没有 tool_calls 就是最终答案，循环只跑一轮。"""
    llm = FakeLLM([_assistant("这是答案")])
    messages = [SYSTEM, USER]

    answer = _run(run_react_loop(messages, llm, FakeTools()))

    assert answer == "这是答案"
    assert len(llm.calls) == 1
    # 最终回答也要留在历史里，否则多轮对话会断片
    assert messages[-1] == {"role": "assistant", "content": "这是答案"}


def test_tool_call_then_final_answer():
    """一轮工具调用后拿到答案，历史形状完整且 tool_call_id 对得上。"""
    llm = FakeLLM(
        [
            _assistant(tool_calls=[_tool_call("Read", {"path": "a.py"})]),
            _assistant("a.py 里是一个空文件"),
        ]
    )
    tools = FakeTools({"Read": "文件内容"})
    messages = [SYSTEM, USER]

    answer = _run(run_react_loop(messages, llm, tools))

    assert answer == "a.py 里是一个空文件"
    assert len(tools.received) == 1
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    # tool 消息必须挂回 assistant 给出的那个 id，否则下一轮 API 报 400
    assert messages[3]["tool_call_id"] == "call_1"
    assert messages[3]["content"] == "文件内容"


def test_multiple_tool_calls_in_one_turn_all_executed():
    """一轮返回多个 tool_call 时全部执行，各自带回自己的 id。"""
    llm = FakeLLM(
        [
            _assistant(
                tool_calls=[
                    _tool_call("Read", {"path": "a.py"}, call_id="call_a"),
                    _tool_call("Grep", {"pattern": "def"}, call_id="call_b"),
                ]
            ),
            _assistant("两个都查完了"),
        ]
    )
    tools = FakeTools()
    messages = [SYSTEM, USER]

    _run(run_react_loop(messages, llm, tools))

    assert [tc["id"] for tc in tools.received] == ["call_a", "call_b"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]


def test_tool_exception_is_fed_back_not_raised():
    """工具抛异常时转成字符串回灌，循环继续而不是崩掉。"""
    llm = FakeLLM(
        [
            _assistant(tool_calls=[_tool_call("Bash", {"cmd": "boom"})]),
            _assistant("Bash 不可用，我改用别的方式"),
        ]
    )
    tools = FakeTools(raises={"Bash": RuntimeError("命令超时")})
    messages = [SYSTEM, USER]

    answer = _run(run_react_loop(messages, llm, tools))

    assert answer == "Bash 不可用，我改用别的方式"
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "工具执行出错" in tool_msg["content"]
    assert "命令超时" in tool_msg["content"]


def test_raises_max_iteration_error():
    """永远返回 tool_calls 时，跑满 max_steps 后抛 MaxIterationError。"""
    llm = FakeLLM([_assistant(tool_calls=[_tool_call("Read")]) for _ in range(3)])
    messages = [SYSTEM, USER]

    with pytest.raises(MaxIterationError) as exc_info:
        _run(run_react_loop(messages, llm, FakeTools(), max_steps=3))

    assert exc_info.value.max_steps == 3
    assert len(llm.calls) == 3


def test_撞上限时把最后的进展带出来():
    """跑到上限通常不是一无所获，而是「查了一大堆没来得及收尾」。
    白扔掉等于让调用方白付一次 token。"""
    llm = FakeLLM(
        [
            _assistant("我先看看这个文件怎么被用的", tool_calls=[_tool_call("Read")]),
            _assistant(tool_calls=[_tool_call("Read")]),
        ]
    )

    with pytest.raises(MaxIterationError) as exc_info:
        _run(run_react_loop([SYSTEM, USER], llm, FakeTools(), max_steps=2))

    assert exc_info.value.partial == "我先看看这个文件怎么被用的"


def test_没有中间发言时_partial_为空():
    llm = FakeLLM([_assistant(tool_calls=[_tool_call("Read")])])

    with pytest.raises(MaxIterationError) as exc_info:
        _run(run_react_loop([SYSTEM, USER], llm, FakeTools(), max_steps=1))

    assert exc_info.value.partial == ""


def test_last_progress_取最后一条有内容的发言():
    from app.agent.react_loop import last_progress

    history = [
        {"role": "assistant", "content": "早先说的"},
        {"role": "tool", "content": "工具结果不算"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
        {"role": "assistant", "content": " 最后说的 "},
        {"role": "tool", "content": "结果",
         "tool_call_id": "c"},
    ]

    assert last_progress(history) == "最后说的"


def test_max_iteration_消息不会被套两层():
    """构造时传字符串会被塞进 max_steps 的位置，消息就变成
    「达到最大循环次数 达到最大循环次数 10，…」—— 这个 bug 真出现过。"""
    error = MaxIterationError(10, note="。可以 --continue")

    assert str(error) == "达到最大循环次数 10，仍未得出最终答案。可以 --continue"
    assert str(error).count("达到最大循环次数") == 1


def test_on_message_每条新增消息都回调一次():
    """Session 靠这个钩子做 append-only 落盘 —— 回调的内容必须和 messages
    实际追加的部分完全一致，否则恢复出来的历史会和真实 Context 对不上。"""
    llm = FakeLLM(
        [
            _assistant(tool_calls=[_tool_call("Read")]),
            _assistant("答案"),
        ]
    )
    seen: list[dict] = []
    messages = [SYSTEM, USER]

    _run(run_react_loop(messages, llm, FakeTools(), on_message=seen.append))

    assert [m["role"] for m in seen] == ["assistant", "tool", "assistant"]
    assert seen == messages[2:]


def test_tools_schema_is_passed_through():
    """tools schema 原样透传给 LLM —— Phase 2 接入 ToolRegistry 靠的就是这个口子。"""
    schema = [{"type": "function", "function": {"name": "Read"}}]
    llm = FakeLLM([_assistant("好")])

    _run(run_react_loop([SYSTEM, USER], llm, FakeTools(), tools=schema))

    assert llm.tools_seen == [schema]
