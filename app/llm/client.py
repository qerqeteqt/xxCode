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
import logging
import random
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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


def _usage_of(data: dict) -> TokenUsage:
    usage = data.get("usage") or {}
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        calls=1,
    )


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

    async def _post_with_retry(self, payload: dict) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            error: Exception
            delay = self._delay_for(attempt)

            try:
                response = await self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                error = e
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    # 成功，或者「重试也没用」。两种情况都原样交给上层判断
                    return response
                error = LLMError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                if retry_after is not None:
                    delay = retry_after

            if attempt >= self.max_retries:
                raise LLMError(
                    f"重试 {self.max_retries} 次后仍然失败：{error}"
                ) from error

            logger.warning(
                "LLM 调用失败，%.1fs 后重试（第 %d/%d 次）：%s",
                delay,
                attempt + 1,
                self.max_retries,
                error,
            )
            await asyncio.sleep(delay)

        raise AssertionError("unreachable")  # 循环里要么 return 要么 raise

    # ------------------------------------------------------------ 调用

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """发起一次 chat completion，返回 assistant message。

        有 tool_calls 时，返回的 message 里会带 tool_calls 字段；
        没有时，content 就是模型的最终回答。
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools

        resp = await self._post_with_retry(payload)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"LLM 返回 {e.response.status_code}: {e.response.text[:500]}"
            ) from e

        try:
            data = resp.json()
            usage = _usage_of(data)
            self.usage = self.usage + usage
            self.last_prompt_tokens = usage.prompt_tokens
            return _normalize_message(data["choices"][0]["message"])
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM 响应结构异常: {resp.text[:500]}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
