"""Phase 7a LLMClient 测试：重试策略与 token 统计。

重试用 httpx.MockTransport 塞一个假网络层，不碰真网络也不碰真等待
（base_delay=0，需要观察等待时长时把 asyncio.sleep 换成记录器）。
"""

import asyncio
import json

import httpx
import pytest

from app.llm.client import (
    LLMClient,
    LLMError,
    TokenUsage,
    human_tokens,
)
from app.llm.content import REF_PREFIX, image_block


def _run(coro):
    return asyncio.run(coro)


def _ok(content: str = "hi", prompt: int = 10, completion: int = 5) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def _client(handler, **kwargs) -> LLMClient:
    # 默认不等待 —— 除非用例自己要看退避时长
    kwargs.setdefault("base_delay", 0.0)
    return LLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


class _Sequence:
    """按顺序吐响应，并记下被调了几次。"""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if not self._responses:
            raise AssertionError("响应脚本用完了 —— 重试次数比预期多")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ================================================================ 重试


def test_正常调用不重试():
    handler = _Sequence(_ok("答案"))

    result = _run(_client(handler).chat([{"role": "user", "content": "问"}]))

    assert result["content"] == "答案"
    assert handler.calls == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_可重试的状态码会重试(status):
    handler = _Sequence(httpx.Response(status, text="抖了一下"), _ok("终于好了"))

    result = _run(_client(handler).chat([{"role": "user", "content": "问"}]))

    assert result["content"] == "终于好了"
    assert handler.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_四开头的错误不重试(status):
    """盲目重试 4xx 会把一个明确的配置错误变成一个要等半分钟的谜题。"""
    handler = _Sequence(httpx.Response(status, text="请求本身有问题"))

    with pytest.raises(LLMError) as exc_info:
        _run(_client(handler).chat([{"role": "user", "content": "问"}]))

    assert handler.calls == 1  # 一次都没重试
    assert str(status) in str(exc_info.value)


def test_网络超时会重试():
    handler = _Sequence(httpx.ReadTimeout("超时了"), _ok("好了"))

    result = _run(_client(handler).chat([{"role": "user", "content": "问"}]))

    assert result["content"] == "好了"
    assert handler.calls == 2


def test_重试次数用完就抛错():
    handler = _Sequence(*[httpx.Response(503, text="一直挂")] * 4)

    with pytest.raises(LLMError, match="重试 3 次后仍然失败"):
        _run(_client(handler, max_retries=3).chat([{"role": "user", "content": "问"}]))

    assert handler.calls == 4  # 首次 + 3 次重试


def test_退避是指数增长的():
    """1s → 2s → 4s。抖动是 [0, delay*0.3]，所以下界就是这几个数。"""
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    handler = _Sequence(*[httpx.Response(503, text="挂")] * 4)

    async def _go():
        client = _client(handler, max_retries=3, base_delay=1.0)
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "问"}])

    original = asyncio.sleep
    asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        _run(_go())
    finally:
        asyncio.sleep = original  # type: ignore[assignment]

    assert len(slept) == 3
    assert slept[0] < slept[1] < slept[2]
    assert 1.0 <= slept[0] <= 1.3
    assert 2.0 <= slept[1] <= 2.6
    assert 4.0 <= slept[2] <= 5.2


def test_尊重服务端给的_Retry_After():
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    handler = _Sequence(
        httpx.Response(429, text="慢点", headers={"Retry-After": "7"}), _ok("好了")
    )

    async def _go():
        client = _client(handler)
        return await client.chat([{"role": "user", "content": "问"}])

    original = asyncio.sleep
    asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        _run(_go())
    finally:
        asyncio.sleep = original  # type: ignore[assignment]

    # 服务端说等 7 秒就等 7 秒，不用自己那套退避
    assert slept == [7.0]


def test_Retry_After_封顶():
    """服务端要是说等一小时，我们不能真等一小时。"""
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    handler = _Sequence(
        httpx.Response(429, text="慢点", headers={"Retry-After": "7200"}), _ok("好了")
    )

    async def _go():
        client = _client(handler)
        return await client.chat([{"role": "user", "content": "问"}])

    original = asyncio.sleep
    asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        _run(_go())
    finally:
        asyncio.sleep = original  # type: ignore[assignment]

    assert slept == [30.0]


def test_Retry_After_不是数字就退回自己的退避():
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    handler = _Sequence(
        httpx.Response(429, text="慢点", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        _ok("好了"),
    )

    async def _go():
        client = _client(handler)
        return await client.chat([{"role": "user", "content": "问"}])

    original = asyncio.sleep
    asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        _run(_go())
    finally:
        asyncio.sleep = original  # type: ignore[assignment]

    # HTTP 日期格式不认，退回 base_delay=0
    assert slept == [0.0]


# ================================================================ Token 用量


def test_每次调用累加用量():
    handler = _Sequence(_ok(prompt=100, completion=20), _ok(prompt=300, completion=40))
    client = _client(handler)

    _run(client.chat([{"role": "user", "content": "一"}]))
    _run(client.chat([{"role": "user", "content": "二"}]))

    assert client.usage.prompt_tokens == 400
    assert client.usage.completion_tokens == 60
    assert client.usage.total_tokens == 460
    assert client.usage.calls == 2


def test_失败的调用不计入用量():
    """只有真正返回内容的调用才算数 —— 重试失败的那些没拿到 usage。"""
    handler = _Sequence(httpx.Response(503, text="挂"), _ok(prompt=50, completion=10))
    client = _client(handler)

    _run(client.chat([{"role": "user", "content": "问"}]))

    assert client.usage.calls == 1
    assert client.usage.total_tokens == 60


def test_没有_usage_字段也不炸():
    handler = _Sequence(
        httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    )
    client = _client(handler)

    result = _run(client.chat([{"role": "user", "content": "问"}]))

    assert result["content"] == "hi"
    assert client.usage.total_tokens == 0
    assert client.usage.calls == 1


def test_TokenUsage_加减法():
    a = TokenUsage(100, 20, 2)
    b = TokenUsage(30, 5, 1)

    assert (a + b) == TokenUsage(130, 25, 3)
    assert (a - b) == TokenUsage(70, 15, 1)
    assert a.total_tokens == 120


def test_TokenUsage_差值不为负():
    """SubAgent 取差值时，万一基准比当前还大也不该出现负数。"""
    assert (TokenUsage(10, 1, 1) - TokenUsage(99, 99, 9)) == TokenUsage(0, 0, 0)


def test_human_tokens_可读化():
    assert human_tokens(0) == "0"
    assert human_tokens(999) == "999"
    assert human_tokens(18_234) == "18.2k"


def test_空用量为假():
    """SubAgent 的头部靠这个判断该不该显示 tokens。"""
    assert not TokenUsage()
    assert TokenUsage(1, 0, 1)


# ================================================================ 流式

def _sse(*chunks: dict) -> httpx.Response:
    body = "".join(
        f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks
    ) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content=body.encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


def _delta(content=None, tool_calls=None, usage=None) -> dict:
    chunk: dict = {"choices": [{"index": 0, "delta": {}}]}
    if content is not None:
        chunk["choices"][0]["delta"]["content"] = content
    if tool_calls is not None:
        chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_流式把文字一段段交给回调():
    handler = _Sequence(_sse(
        _delta(content="ReAct "), _delta(content="是"), _delta(content="一种范式"),
    ))
    got: list[str] = []

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=got.append
    ))

    assert got == ["ReAct ", "是", "一种范式"]   # 回调拿到的是碎片
    assert result["content"] == "ReAct 是一种范式"  # 返回的是拼好的完整消息


def test_流式请求要带上_stream_参数():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _sse(_delta(content="好"))

    _run(_client(handler).chat([{"role": "user", "content": "问"}], on_delta=lambda _: None))

    assert seen["stream"] is True
    # usage 默认不在流式响应里，得显式要，否则 token 统计全变 0
    assert seen["stream_options"] == {"include_usage": True}


def test_流式的_tool_call_碎片要按_index_拼起来():
    """**流式最容易写错的地方。** id 和 name 只在第一片出现，arguments 一片片来。
    拼错的表现是「工具名变成空字符串」或「arguments 少了前半截」，
    而两者都不会立刻报错，只会让工具调用莫名其妙地失败。"""
    handler = _Sequence(_sse(
        _delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "Read", "arguments": ""}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '{"pa'}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": '}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '"a.py"}'}}]),
    ))

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=lambda _: None
    ))

    calls = result["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "Read"
    assert calls[0]["function"]["arguments"] == '{"path": "a.py"}'


def test_流式一轮里的多个_tool_call_按_index_分开():
    handler = _Sequence(_sse(
        _delta(tool_calls=[
            {"index": 0, "id": "c0", "function": {"name": "Read", "arguments": '{"a"'}},
            {"index": 1, "id": "c1", "function": {"name": "Grep", "arguments": '{"b"'}},
        ]),
        _delta(tool_calls=[
            {"index": 0, "function": {"arguments": ": 1}"}},
            {"index": 1, "function": {"arguments": ": 2}"}},
        ]),
    ))

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=lambda _: None
    ))

    assert [c["function"]["name"] for c in result["tool_calls"]] == ["Read", "Grep"]
    assert result["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert result["tool_calls"][1]["function"]["arguments"] == '{"b": 2}'


def test_流式文字和工具调用可以同时出现():
    """模型常常先说一句再调工具 —— 两种情况要在同一个响应里都处理好。"""
    handler = _Sequence(_sse(
        _delta(content="我先看看"),
        _delta(tool_calls=[{"index": 0, "id": "c", "function": {"name": "Read", "arguments": "{}"}}]),
    ))

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=lambda _: None
    ))

    assert result["content"] == "我先看看"
    assert result["tool_calls"][0]["function"]["name"] == "Read"


def test_流式的_usage_从最后一个分片取():
    handler = _Sequence(_sse(
        _delta(content="答"),
        _delta(usage={"prompt_tokens": 120, "completion_tokens": 30}),
    ))
    client = _client(handler)

    _run(client.chat([{"role": "user", "content": "问"}], on_delta=lambda _: None))

    assert client.usage.prompt_tokens == 120
    assert client.usage.completion_tokens == 30
    assert client.last_prompt_tokens == 120


def test_流式也会重试可重试的状态码():
    """429/5xx 发生在响应头阶段 —— 那时用户还没看到任何内容，重试是安全的。"""
    handler = _Sequence(httpx.Response(429, text="慢点"), _sse(_delta(content="好了")))
    got: list[str] = []

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=got.append
    ))

    assert result["content"] == "好了"
    assert handler.calls == 2
    assert got == ["好了"]  # 失败那次没吐出任何东西


def test_流式读到一半断了不重试():
    """用户已经看到一部分内容了，重来一遍只会看到重复的一坨。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("连接断了", request=request)

    with pytest.raises(LLMError, match="流式响应中断|重试"):
        _run(_client(handler).chat(
            [{"role": "user", "content": "问"}], on_delta=lambda _: None
        ))


def test_流式忽略空行和非_data_行():
    body = (
        ": 这是注释\n\n"
        "event: message\n"
        'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
        "\n"
        'data: {"choices":[{"delta":{"content":"B"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    handler = _Sequence(httpx.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    ))

    result = _run(_client(handler).chat(
        [{"role": "user", "content": "问"}], on_delta=lambda _: None
    ))

    assert result["content"] == "AB"


# ================================================================ 图片引用展开


NAME = "3f9a1c2b4d5e6f70.png"
DATA_URL = "data:image/png;base64,AAAA"


def _image_message(name: str = NAME) -> dict:
    return {"role": "user", "content": [image_block(name)]}


def _capture(seen: list, response: httpx.Response | None = None):
    """记下每次请求体，返回固定响应。"""
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return response if response is not None else _ok()
    return handler


def test_引用被展开成_data_url_发出去():
    seen: list = []
    llm = _client(_capture(seen), resolve_image=lambda name: DATA_URL)

    _run(llm.chat([_image_message()]))

    url = seen[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == DATA_URL
    assert not url.startswith(REF_PREFIX)


def test_展开不会污染原来的_messages():
    """**这条是整个设计的哨兵。**

    `payload["messages"] = messages` 是引用传递，不是拷贝。就地展开的话，
    200 KB 的 base64 会永久留在活的 Context 里 —— 污染压缩器的摘要渲染、
    last_progress、以及之后任何一次 re-append。所以展开必须发生在拷贝上。
    """
    seen: list = []
    llm = _client(_capture(seen), resolve_image=lambda name: DATA_URL)
    messages = [_image_message()]

    _run(llm.chat(messages))

    # 发出去的确实展开了
    assert seen[0]["messages"][0]["content"][0]["image_url"]["url"] == DATA_URL
    # 但传进来的那个对象一个字节都没变
    assert messages[0]["content"][0]["image_url"]["url"] == f"{REF_PREFIX}{NAME}"


def test_展开后发第二次仍然是引用():
    """同样的 list 连发两次，第二次不能因为「上次展开过」就带着 base64。"""
    seen: list = []
    llm = _client(_capture(seen), resolve_image=lambda name: DATA_URL)
    messages = [_image_message()]

    _run(llm.chat(messages))
    _run(llm.chat(messages))

    for body in seen:
        assert body["messages"][0]["content"][0]["image_url"]["url"] == DATA_URL
    assert messages[0]["content"][0]["image_url"]["url"] == f"{REF_PREFIX}{NAME}"


def test_没有_resolver_时引用原样发出去():
    """不传 resolver 就是「这个 Runtime 不认引用」，不该自作聪明。"""
    seen: list = []
    llm = _client(_capture(seen))

    _run(llm.chat([_image_message()]))

    url = seen[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == f"{REF_PREFIX}{NAME}"


def test_图片读不到时降级成文字而不是抛错():
    """图片文件被手工删掉，不该让整个会话从此再也发不出去。"""
    seen: list = []

    def boom(name):
        raise RuntimeError("文件没了")

    llm = _client(_capture(seen), resolve_image=boom)

    _run(llm.chat([_image_message()]))

    blocks = seen[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "text"
    assert "无法读取" in blocks[0]["text"]


def test_非法引用名不会被拿去解析():
    """`ref:` 后面跟着不是合法名字的东西时，原样留着 —— 交给服务端报错。"""
    seen: list = []

    def fail(name):
        raise AssertionError(f"不该解析这个名字: {name}")

    llm = _client(_capture(seen), resolve_image=fail)

    _run(llm.chat([_image_message("../../.env")]))

    url = seen[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == f"{REF_PREFIX}../../.env"


def test_不含图片的消息不会被复制():
    """纯文本消息保持原对象 —— 没改动就不该白建一遍 dict。"""
    seen: list = []
    llm = _client(_capture(seen), resolve_image=lambda name: DATA_URL)

    _run(llm.chat([{"role": "user", "content": "你好"}]))

    assert seen[0]["messages"][0] == {"role": "user", "content": "你好"}


def test_流式路径也会展开引用():
    """流式是另一个 payload 消费者，别假设它跟着整包一起改好了。"""
    seen: list = []
    llm = _client(_capture(seen, _sse(_delta(content="好"))), resolve_image=lambda n: DATA_URL)

    _run(llm.chat([_image_message()], on_delta=lambda _: None))

    url = seen[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == DATA_URL


def test_文字和图片混在一起时块顺序不变():
    """内容里混着文字和图片时，块顺序不能乱。"""
    seen: list = []
    llm = _client(_capture(seen), resolve_image=lambda name: DATA_URL)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "看看"}, image_block(NAME)],
        }
    ]

    _run(llm.chat(messages))

    blocks = seen[0]["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "看看"}
    assert blocks[1]["image_url"]["url"] == DATA_URL
