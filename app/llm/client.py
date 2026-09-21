"""LLM 客户端。

对上层只暴露一件事：`await llm.chat(messages, tools) -> assistant message`。
Agent Runtime 不需要知道底下是 DeepSeek、OpenAI 还是别的什么 —— 这就是文档里
「LLM Provider 通过接口抽象」的最小形态：一个类 + 一个契约清晰的方法，
而不是先搭一堆 Protocol / 工厂 / 注册表（第一版只用 DeepSeek，不需要那些）。

## 重试策略

「哪些错该重试」是这里唯一有含量的判断：

    网络超时 / 连接重置    重试  —— 多半是抖动
    429                    重试  —— 限流，等一会儿再来
    5xx                    重试  —— 服务端抖动
    4xx（429 除外）        不重试 —— 请求本身有问题，重试一百次也一样

盲目重试 4xx 是最常见的错误做法：它把一个明确的配置错误（key 不对、模型名写错）
变成一个要等半分钟的谜题。
"""

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from app.llm.content import IMAGE_UNREADABLE, REF_PREFIX, is_image_name

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 每收到一小段模型输出就回调一次。给它文字，怎么显示是调用方的事
DeltaHook = Callable[[str], None]


class _Retryable(Exception):
    """「等一会儿再来就好」的失败。只有它和网络异常会触发重试。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

# 这些状态码表示「等一会儿再来就好」。不在这个集合里的 4xx 说明请求本身有问题。
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BASE_DELAY = 1.0
# 服务端给的 Retry-After 也封个顶，否则它说等一小时就真等一小时
MAX_RETRY_AFTER = 30.0


class LLMError(RuntimeError):
    """LLM 调用失败：网络错误、超时、HTTP 4xx/5xx、响应结构不对、重试耗尽。"""


@dataclass(frozen=True)
class TokenUsage:
    """token 用量。

    calls 单独记，是因为「花了多少 token」和「调了多少次」是两个不同的问题：
    前者是钱，后者是步数。SubAgent 的开销之所以隐形，就是因为只看步数看不出钱。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.calls + other.calls,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        """算差值。SubAgent 用它算「这一趟花了多少」，不包含 Main 之前的开销。"""
        return TokenUsage(
            max(0, self.prompt_tokens - other.prompt_tokens),
            max(0, self.completion_tokens - other.completion_tokens),
            max(0, self.calls - other.calls),
        )

    def __bool__(self) -> bool:
        return self.calls > 0

    def __str__(self) -> str:
        return (
            f"{human_tokens(self.total_tokens)} tokens"
            f"（prompt {human_tokens(self.prompt_tokens)}"
            f" / completion {human_tokens(self.completion_tokens)}，{self.calls} 次调用）"
        )


def human_tokens(count: int) -> str:
    """1000 以下原样，以上用 k —— 让「18.2k」一眼能读，而不是数位数。"""
    if count < 1000:
        return str(count)
    return f"{count / 1000:.1f}k"


def _parse_retry_after(raw: str | None) -> float | None:
    """Retry-After 可能是秒数也可能是 HTTP 日期。只认秒数，认不出就算了。"""
    if not raw:
        return None
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(raw)))
    except ValueError:
        return None


def _normalize_message(msg: dict) -> dict:
    """只保留 role / content / tool_calls 三个字段。

    为什么必须做这一步：服务端返回的 message 可能带额外字段
    （比如 deepseek-reasoner 的 reasoning_content）。这些字段如果原样写回
    messages 再发出去，轻则被忽略，重则直接 400。
    这里白名单过滤，保证「回灌进历史的对象」和「API 能接受的对象」是同一个形状。
    """
    out: dict[str, Any] = {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content"),
    }
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _merge_tool_call(accumulator: dict[int, dict], fragment: dict) -> None:
    """把流式到达的 tool_call 碎片按 index 拼起来。

    **这是流式最容易写错的地方。** 非流式时 arguments 是一个完整字符串；
    流式时它一片一片地来，而 id 和 name 只在第一片里出现：

        {"index":0,"id":"call_1","function":{"name":"Read","arguments":""}}
        {"index":0,"function":{"arguments":"{\\"pa"}}
        {"index":0,"function":{"arguments":"th\\":\\"a.py\\"}"}}

    拼错的表现是「工具名变成了空字符串」或者「arguments 少了前半截」——
    而两者都不会立刻报错，只会让工具调用莫名其妙地失败。
    """
    index = int(fragment.get("index", 0))
    slot = accumulator.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if fragment.get("id"):
        slot["id"] = fragment["id"]

    function = fragment.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] += function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def _usage_of(data: dict) -> TokenUsage:
    usage = data.get("usage") or {}
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        calls=1,
    )


def _expand_images(messages: list[dict], resolve: Callable[[str], str]) -> list[dict]:
    """把 messages 里的 `ref:<name>` 展开成真正的 data URL。

    **三件事都是刻意的：**

    1. **返回新对象，绝不就地改。** 传进来的 `messages` 就是活的 Context，
       而 `payload["messages"] = messages` 是引用传递。就地展开的话，200 KB 的
       base64 会永久留在上下文里 —— 污染压缩器的摘要渲染、`last_progress`、
       以及之后任何一次 re-append。

    2. **没改动就返回原来那个 list**，不白分配。

    3. **解析失败降级成文字，不往上抛。** 图片文件被手工删掉不该让整个会话
       永久不可用（每次重试都失败）。和 `react_loop` 里「工具崩了不能毁掉整个
       Runtime」是同一条原则。
    """
    out: list[dict] = []
    changed_any = False

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue

        blocks: list[dict] = []
        changed = False
        for block in content:
            url = None
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url")

            if not isinstance(url, str) or not url.startswith(REF_PREFIX):
                blocks.append(block)
                continue

            name = url[len(REF_PREFIX) :]
            if not is_image_name(name):
                blocks.append(block)
                continue

            try:
                blocks.append(
                    {"type": "image_url", "image_url": {"url": resolve(name)}}
                )
            except Exception as e:  # noqa: BLE001 —— 见上面第 3 条
                logger.warning("图片 %s 读不出来，降级成文字: %s", name, e)
                blocks.append({"type": "text", "text": IMAGE_UNREADABLE})
            changed = True

        if changed:
            out.append({**message, "content": blocks})
            changed_any = True
        else:
            out.append(message)

    return out if changed_any else messages


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        resolve_image: Callable[[str], str] | None = None,
    ) -> None:
        # httpx 拼接 base_url 时是把路径直接接上去的（/v1 + chat/completions），
        # 不补斜杠就会拼成 /v1chat/completions。末尾这个 "/" 不能省。
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            # 只给测试用：httpx.MockTransport 可以在这里塞一个假的网络层
            transport=transport,
        )
        self._model = model
        self._temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        # 把 ref:<name> 变成 data:image/...;base64,... 的回调。注入而不是 import：
        # app/llm/client.py 是个零依赖的叶子节点，而 app/memory/__init__ →
        # session_store → app/context/compactor → app.llm.client 已经是一条环，
        # 反向 import 会直接在 ImportError 上炸掉。
        # 不传（None）就是「这个 Runtime 不认引用」—— 摘要 / 记忆提取这类纯文本
        # 调用方本来就不需要它
        self._resolve_image = resolve_image
        # 累计用量。SubAgent / AutoDream 走的是同一个 client，所以这里统计的是
        # 「这个 client 一共花了多少」—— 对账单来说正是想要的数
        self.usage = TokenUsage()
        # 上一次调用的 prompt 有多大。上下文压缩靠它判断该不该压 ——
        # 这是**真实数字**，比拿字符数估算 token 准得多，而且不额外花钱
        self.last_prompt_tokens = 0

    # ------------------------------------------------------------ 网络

    def _delay_for(self, attempt: int) -> float:
        """第 attempt 次失败后该等多久（attempt 从 0 起）。

        抖动不是装饰：同时跑多个 SubAgent 时，它们会同时失败、同时重试；
        没有随机量就是一批一批地一起撞上去，等于没退避。
        """
        delay = self.base_delay * (2**attempt)
        return delay + random.uniform(0, delay * 0.3)

    async def _attempt(self, operation: Callable[[], Awaitable[T]]) -> T:
        """跑一次 operation，对「等一会儿再来就好」的失败自动重试。

        重试的粒度是**「发请求 + 判状态码」这一整段**，而不是整个调用。
        这个边界对流式很关键：响应头到达之前的失败（429 / 5xx / 连不上）重试是安全的，
        **但 body 一旦开始往用户那边吐，就不能重试了** —— 用户已经看到内容，
        重来一遍只会看到重复的一坨。所以流式解析放在 operation **之外**。
        """
        error: Exception = LLMError("未执行任何尝试")
        delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            delay = self._delay_for(attempt)
            try:
                return await operation()
            except _Retryable as e:
                error = e
                if e.retry_after is not None:
                    delay = e.retry_after
            except (httpx.TimeoutException, httpx.TransportError) as e:
                error = e

            if attempt >= self.max_retries:
                break

            logger.warning(
                "LLM 调用失败，%.1fs 后重试（第 %d/%d 次）：%s",
                delay,
                attempt + 1,
                self.max_retries,
                error,
            )
            await asyncio.sleep(delay)

        raise LLMError(f"重试 {self.max_retries} 次后仍然失败：{error}") from error

    async def _post_full(self, payload: dict) -> httpx.Response:
        """非流式：一次拿到完整响应。"""

        async def once() -> httpx.Response:
            response = await self._client.post("/chat/completions", json=payload)
            if response.status_code in RETRYABLE_STATUS:
                raise _Retryable(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    _parse_retry_after(response.headers.get("retry-after")),
                )
            # 成功，或者「重试也没用」。两种情况都原样交给上层判断
            return response

        return await self._attempt(once)

    async def _post_stream(
        self, payload: dict, on_delta: "DeltaHook"
    ) -> tuple[dict, TokenUsage]:
        """流式：边收边把文字交给 on_delta，最后拼出完整消息。

        解析写在 operation **外面**（见 _attempt 的注释）——
        一旦开始吐内容就不能重试了。
        """

        async def once() -> tuple[dict, TokenUsage]:
            # stream() 进入时就把请求发出去、拿到响应头 —— 所以 429/5xx
            # 在这一层就能判，此时**一个字节都还没交给用户**，重试是安全的
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.status_code in RETRYABLE_STATUS:
                    await response.aread()
                    raise _Retryable(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        _parse_retry_after(response.headers.get("retry-after")),
                    )
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()

                # 到这里才开始读 body。之后再出错就不能重试了 ——
                # 用户已经看到一部分内容，重来一遍只会看到重复的一坨
                try:
                    return await self._read_sse(response, on_delta)
                except httpx.HTTPError as e:
                    raise LLMError(f"流式响应中断: {e}") from e

        return await self._attempt(once)

    async def _read_sse(
        self, response: httpx.Response, on_delta: "DeltaHook"
    ) -> tuple[dict, TokenUsage]:
        """解析 SSE 流，拼出 assistant 消息。

        usage 只在**最后一个** chunk 里，而且要先在请求里带
        `stream_options.include_usage`（见 chat）。
        """
        parts: list[str] = []
        fragments: dict[int, dict] = {}
        usage = TokenUsage()

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue  # 空行、event: 行、注释行都跳过
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("跳过无法解析的流式分片: %s", data[:120])
                continue

            if chunk.get("usage"):
                usage = _usage_of(chunk)

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            text = delta.get("content")
            if text:
                parts.append(text)
                on_delta(text)

            for fragment in delta.get("tool_calls") or []:
                _merge_tool_call(fragments, fragment)

        return (
            _normalize_message(
                {
                    "role": "assistant",
                    "content": "".join(parts) or None,
                    "tool_calls": [
                        fragments[i] for i in sorted(fragments)
                    ] or None,
                }
            ),
            usage,
        )

    # ------------------------------------------------------------ 调用

    def _record(self, usage: TokenUsage) -> None:
        self.usage = self.usage + usage
        self.last_prompt_tokens = usage.prompt_tokens

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaHook | None = None,
    ) -> dict:
        """发起一次 chat completion，返回 assistant message。

        有 tool_calls 时，返回的 message 里会带 tool_calls 字段；
        没有时，content 就是模型的最终回答。

        on_delta 给了就走流式：模型每吐出一小段文字就回调一次。
        不给就走整包（SubAgent / AutoDream 不需要流式，也少一层解析）。
        """
        # 展开图片引用。放在这里是因为这是**唯一**组装 payload 的地方 ——
        # 流式和整包两条路都从这一个 dict 走，所以只需要接一次。
        # 展开的产物只挂在 payload 上，messages 本身摸都不摸（见 _expand_images）
        outgoing = (
            _expand_images(messages, self._resolve_image)
            if self._resolve_image is not None
            else messages
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": outgoing,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools

        if on_delta is not None:
            payload["stream"] = True
            # usage 默认不在流式响应里，得显式要 —— 不要的话 token 统计全变 0
            payload["stream_options"] = {"include_usage": True}
            message, usage = await self._post_stream(payload, on_delta)
            self._record(usage)
            return message

        resp = await self._post_full(payload)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"LLM 返回 {e.response.status_code}: {e.response.text[:500]}"
            ) from e

        try:
            data = resp.json()
            self._record(_usage_of(data))
            return _normalize_message(data["choices"][0]["message"])
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM 响应结构异常: {resp.text[:500]}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
