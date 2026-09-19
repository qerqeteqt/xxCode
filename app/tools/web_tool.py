"""WebSearchTool —— 联网搜索（Tavily）。

## 和 Glob / Grep 的分工

    Glob / Grep   搜**这个项目**里的东西
    WebSearch     搜**互联网**上的东西

这两件事容易混。所以工具描述里要写死一句「搜本地代码请用 Glob / Grep」——
不写的话模型会拿它去搜项目里的函数名，白花一次网络请求，还不知道自己在干什么。

## 两个要提前说清的代价

**一、关键词会发到外部服务。** 所以不能把代码或密钥塞进 query。这条靠 prompt
提醒，不靠技术强制 —— 真要强制就得在本地做关键词脱敏，那个误伤率高得离谱，
拦掉的东西比放过的更有价值。

**二、它给的是摘要，不是原文。** Tavily 返回的 content 是搜索服务截出来的一段，
可能不完整、也可能过时。所以返回值末尾会提醒模型：需要细节就用更具体的关键词
再搜，或者把链接给用户自己看 —— 而不是拿着半截摘要下结论。
"""

import logging

import httpx
from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolError, ToolResult
from app.tools.text import truncate

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_CAP = 10
TIMEOUT = 30.0

# 每条摘要和整体输出都封顶：搜索结果直接进 Context，不封顶就是拿钱买噪声
MAX_SNIPPET_CHARS = 600
MAX_TOTAL_CHARS = 6000


class WebSearchParams(BaseModel):
    query: str = Field(
        description="搜索关键词。用具体的词，例如 'httpx AsyncClient timeout 参数'，"
        "而不是一整句话"
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_RESULTS_CAP,
        description=f"返回几条结果，上限 {MAX_RESULTS_CAP}",
    )


class WebSearchTool(Tool):
    name = "WebSearch"
    risk = "read"
    description = (
        "联网搜索，返回网页的标题、链接和摘要。"
        "适合查库/框架的用法、报错信息、最新文档 —— 也就是**本地代码库里没有的知识**。"
        "**搜项目内的代码请用 Glob / Grep**，不要拿它搜本地文件。"
        "注意关键词会发到外部服务，不要往 query 里放代码或密钥。"
    )
    params_model = WebSearchParams

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._api_key = api_key
        self._transport = transport  # 只给测试用：塞一个假的网络层

    async def execute(self, query: str, max_results: int) -> ToolResult:
        # 每次调用起一个客户端、用完就关。搜索是低频操作，
        # 为此在 Tool 上加一套生命周期钩子不划算
        async with httpx.AsyncClient(
            timeout=TIMEOUT, transport=self._transport
        ) as client:
            try:
                response = await client.post(
                    ENDPOINT,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"query": query, "max_results": max_results},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ToolError(
                    f"搜索失败，Tavily 返回 {e.response.status_code}: "
                    f"{e.response.text[:200]}"
                ) from e
            except httpx.HTTPError as e:
                raise ToolError(f"搜索请求失败: {e}") from e

        try:
            results = response.json().get("results") or []
        except ValueError as e:
            raise ToolError(f"搜索响应不是合法 JSON: {response.text[:200]}") from e

        if not results:
            # 空结果是正常现象，不是错误 —— 让模型自己换关键词，别让它以为工具坏了
            return ToolResult(f"没有找到和「{query}」相关的结果。换个关键词再试试。")

        return ToolResult(truncate(_render(query, results), MAX_TOTAL_CHARS))


def _render(query: str, results: list[dict]) -> str:
    lines = [f"关于「{query}」的搜索结果：", ""]
    for index, item in enumerate(results, 1):
        title = str(item.get("title") or "（无标题）")
        # 摘要里的换行会把「一条结果」摊成好几行，压平了更好扫
        snippet = " ".join(str(item.get("content") or "").split())
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "…"

        lines.append(f"{index}. {title}")
        lines.append(f"   {item.get('url', '')}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    lines.append(
        "（以上摘要是搜索服务截的，未必完整。需要细节就用更具体的关键词再搜，"
        "或者把链接给用户让他自己看。）"
    )
    return "\n".join(lines)
