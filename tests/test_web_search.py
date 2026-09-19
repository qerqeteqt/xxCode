"""WebSearchTool 的测试。

网络层用 httpx.MockTransport 顶掉，不碰真 Tavily —— 这样测试既快又不会
把关键词发出去。
"""

import asyncio
import json

import httpx
import pytest

from app.tools.base import ToolError
from app.tools.web_tool import WebSearchTool


def _run(coro):
    return asyncio.run(coro)


def _call(tool, **kwargs):
    """直接调 execute —— 这里测的是工具本身，不是 registry 那层。"""
    return _run(tool.execute(**kwargs))


def _tool(handler, key="tvly-test") -> WebSearchTool:
    return WebSearchTool(key, transport=httpx.MockTransport(handler))


def _ok(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"query": "q", "results": results})


def _hit(title="标题", url="https://example.com/a", content="摘要内容"):
    return {"title": title, "url": url, "content": content, "score": 0.9}


def test_正常搜索返回标题链接摘要():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _ok([_hit(title="ReAct 是什么", url="https://x.com/1", content="一种范式")])

    result = _tool(handler).execute(query="ReAct agent", max_results=3)
    text = _run(result).text

    assert "1. ReAct 是什么" in text
    assert "https://x.com/1" in text
    assert "一种范式" in text
    # 请求本身要对：端点、鉴权头、参数名
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"] == {"query": "ReAct agent", "max_results": 3}


def test_把摘要里的换行压平():
    """搜索服务返回的 content 里常有换行，照原样输出会把一条结果摊成好几行。"""
    handler = lambda r: _ok([_hit(content="第一行\n第二行\n\n第三行")])

    text = _run(_tool(handler).execute(query="q", max_results=1)).text

    assert "第一行 第二行  第三行" in text or "第一行 第二行" in text
    assert "第一行\n第二行" not in text


def test_过长的摘要被截断():
    handler = lambda r: _ok([_hit(content="很长的摘要" * 500)])

    text = _run(_tool(handler).execute(query="q", max_results=1)).text

    assert "…" in text
    assert len(text) < 7000


def test_空结果是提示而不是错误():
    """搜不到是正常现象。报成错误的话模型会以为工具坏了，然后就不敢再搜了。"""
    handler = lambda r: _ok([])

    text = _run(_tool(handler).execute(query="乱码关键词", max_results=5)).text

    assert "没有找到" in text
    assert "换个关键词" in text


def test_接口报错转成_ToolError():
    handler = lambda r: httpx.Response(401, text="invalid api key")

    with pytest.raises(ToolError, match="401"):
        _call(_tool(handler), query="q", max_results=5)


def test_服务端_5xx_也是_ToolError():
    handler = lambda r: httpx.Response(503, text="overloaded")

    with pytest.raises(ToolError, match="503"):
        _call(_tool(handler), query="q", max_results=5)


def test_网络异常转成_ToolError():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连不上", request=request)

    with pytest.raises(ToolError, match="搜索请求失败"):
        _call(_tool(handler), query="q", max_results=5)


def test_响应不是_JSON_也转成_ToolError():
    handler = lambda r: httpx.Response(200, text="<html>502 Bad Gateway</html>")

    with pytest.raises(ToolError, match="不是合法 JSON"):
        _call(_tool(handler), query="q", max_results=5)


def test_没有结果时也不报错_只提示():
    handler = lambda r: httpx.Response(200, json={"query": "q"})

    text = _run(_tool(handler).execute(query="q", max_results=5)).text

    assert "没有找到" in text


def test_返回值末尾提醒摘要是截的():
    """防的是模型拿着半截摘要当结论。"""
    handler = lambda r: _ok([_hit()])

    text = _run(_tool(handler).execute(query="q", max_results=1)).text

    assert "未必完整" in text


# ================================================================ 装配层


def test_没有_key_就不注册这个工具(tmp_path):
    """「能力由装配决定」：没配 key 就是没有这个能力，
    而不是注册上去、调用时才报错。"""
    from app.tools import build_default_registry

    without = {t.name for t in build_default_registry(tmp_path).list_tools()}
    with_key = {
        t.name for t in build_default_registry(tmp_path, tavily_api_key="tvly-x").list_tools()
    }

    assert "WebSearch" not in without
    assert "WebSearch" in with_key


def test_风险等级是_read():
    """它不改本地任何东西。代价是会把关键词发出去 —— 那条靠 prompt 提醒，
    不靠权限拦（拦的话误伤太高）。"""
    assert WebSearchTool.risk == "read"
