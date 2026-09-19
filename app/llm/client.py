"""LLM 客户端。

对上层只暴露一件事：`await llm.chat(messages, tools) -> assistant message`。
Agent Runtime 不需要知道底下是 DeepSeek、OpenAI 还是别的什么 —— 这就是文档里
「LLM Provider 通过接口抽象」的最小形态：一个类 + 一个契约清晰的方法，
而不是先搭一堆 Protocol / 工厂 / 注册表（第一版只用 DeepSeek，不需要那些）。
"""

from typing import Any

import httpx


class LLMError(RuntimeError):
    """LLM 调用失败：网络错误、超时、HTTP 4xx/5xx、响应结构不对。"""


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


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        # httpx 拼接 base_url 时是把路径直接接上去的（/v1 + chat/completions），
        # 不补斜杠就会拼成 /v1chat/completions。末尾这个 "/" 不能省。
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model
        self._temperature = temperature

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

        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"LLM 返回 {e.response.status_code}: {e.response.text[:500]}"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM 请求失败: {e}") from e

        try:
            return _normalize_message(resp.json()["choices"][0]["message"])
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM 响应结构异常: {resp.text[:500]}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
