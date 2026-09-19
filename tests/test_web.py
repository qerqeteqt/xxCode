"""Web 层测试。

重点是两块容易出事的地方：
    1. 写 .env 必须保留注释，且**写坏了要回滚**（否则下次启动服务起不来）
    2. 事件要真的从 Runtime 里发出来，而不是「前端等着一个永远不来的事件」
"""

import asyncio
import json

import pytest
from fastapi import HTTPException

from app.agent.react_loop import run_react_loop
from app.events import Event, emit, tag_events  # noqa: F401  (tag_events 由子 Agent 用)
from app.memory.session_store import SessionStore
from app.tools import build_default_registry
from app.web.server import apply_settings, update_env


def _run(coro):
    return asyncio.run(coro)


ENV_TEMPLATE = """# 这是注释，不能被冲掉
LLM_API_KEY=sk-test

# 模型名
LLM_MODEL=deepseek-chat
MAX_STEPS=20
"""


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(ENV_TEMPLATE, encoding="utf-8")
    return path


# ================================================================ .env 读写


def test_写回保留注释和顺序(env_file):
    update_env(env_file, {"LLM_MODEL": "deepseek-reasoner", "MAX_STEPS": "30"})

    text = env_file.read_text(encoding="utf-8")
    assert "# 这是注释，不能被冲掉" in text
    assert "# 模型名" in text
    assert "LLM_MODEL=deepseek-reasoner" in text
    assert "MAX_STEPS=30" in text
    # 没被改的项原样在
    assert "LLM_API_KEY=sk-test" in text
    # 顺序：注释在前，LLM_API_KEY 在它后面
    assert text.index("这是注释") < text.index("LLM_API_KEY")


def test_新键追加到末尾(env_file):
    update_env(env_file, {"CONSOLIDATE_MIN_SESSIONS": "5"})

    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "CONSOLIDATE_MIN_SESSIONS=5"


def test_空文件也能写(tmp_path):
    path = tmp_path / ".env"
    update_env(path, {"MAX_STEPS": "30"})

    assert path.read_text(encoding="utf-8") == "MAX_STEPS=30\n"


def test_写坏了要回滚(env_file):
    """写坏的后果很具体：下次启动 Settings() 直接抛错，服务起不来，
    而用户改的只是「max_steps 写成 abc」这种小事。"""
    before = env_file.read_text(encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        apply_settings(env_file, {"MAX_STEPS": "不是数字"})

    assert exc.value.status_code == 400
    assert env_file.read_text(encoding="utf-8") == before  # 一字未变


def test_合法的改动会落盘(env_file):
    apply_settings(env_file, {"MAX_STEPS": "42"})

    assert "MAX_STEPS=42" in env_file.read_text(encoding="utf-8")


def test_空的关键配置也回滚(env_file):
    """空字符串能过 str 的类型检查，但一个空 key 会在请求时变成 401 ——
    那比「配置不合法」难查得多。所以 settings 里给它加了 min_length=1。"""
    before = env_file.read_text(encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        apply_settings(env_file, {"LLM_API_KEY": ""})

    assert exc.value.status_code == 400
    assert env_file.read_text(encoding="utf-8") == before


# ================================================================ 事件


class _ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    async def chat(self, messages, tools=None, on_delta=None):
        return self._replies.pop(0)


def test_循环发出_step_事件(tmp_path):
    llm = _ScriptedLLM([{"role": "assistant", "content": "答"}])
    events: list[Event] = []

    _run(
        run_react_loop(
            [{"role": "user", "content": "问"}],
            llm,
            _never,
            on_event=events.append,
        )
    )

    assert [e.type for e in events] == ["step"]
    assert events[0].data == {"step": 1, "max_steps": 10}


def test_工具调用和结果各发一条事件(tmp_path):
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    registry = build_default_registry(tmp_path)
    events: list[Event] = []

    async def _go():
        registry._on_event = events.append
        return await registry.execute(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path": "a.txt"}'},
            }
        )

    _run(_go())

    assert [e.type for e in events] == ["tool_call", "tool_result"]
    assert events[0].data["name"] == "Read"
    assert events[0].data["call_id"] == "c1"
    assert events[1].data["ok"] is True
    assert "内容" in events[1].data["text"]


def test_工具的失败也发事件而不是只抛异常(tmp_path):
    registry = build_default_registry(tmp_path)
    events: list[Event] = []

    async def _go():
        registry._on_event = events.append
        await registry.execute(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path": "不存在.txt"}'},
            }
        )

    _run(_go())

    assert [e.type for e in events] == ["tool_call", "tool_result"]
    assert events[1].data["ok"] is False
    assert "文件不存在" in events[1].data["text"]


def test_超长结果被截断后再推给前端(tmp_path):
    """一整个文件推过去会把 SSE 流撑爆，而且浏览器也显示不下。"""
    (tmp_path / "big.txt").write_text("x" * 20000, encoding="utf-8")
    registry = build_default_registry(tmp_path)
    events: list[Event] = []

    async def _go():
        registry._on_event = events.append
        await registry.execute(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path": "big.txt"}'},
            }
        )

    _run(_go())

    assert events[1].data["truncated"] is True
    assert len(events[1].data["text"]) == 8000


def test_没传事件钩子时什么都不发(tmp_path):
    """CLI 一行都不用改 —— 事件是纯增量。"""
    registry = build_default_registry(tmp_path)

    result = _run(
        registry.execute(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path": "a.txt"}'},
            }
        )
    )

    assert "不存在" in result  # 照常返回错误字符串，没崩


def test_来源戳能盖上去():
    """子 Agent 的事件要标出来源，否则前端没法知道「这是谁在翻文件」。"""
    got: list[Event] = []
    tagged = tag_events(got.append, "SubAgent: Explore")

    tagged(Event("tool_call", {"name": "Read"}))

    assert got[0].data == {"name": "Read", "source": "SubAgent: Explore"}


def test_没有钩子时盖戳不报错():
    assert tag_events(None, "SubAgent: Explore") is None


async def _never(tool_call: dict) -> str:
    raise AssertionError("这个用例不该调用工具")


# ================================================================ 会话接口


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app.web.server import create_app

    session = SessionStore(tmp_path).create()
    session.append_message({"role": "user", "content": "第一条提问"})
    session.append_message({"role": "assistant", "content": "回答"})
    session.finish("finished")
    return TestClient(create_app(tmp_path)), session


def test_列表带出标题(client):
    c, session = client
    rows = c.get("/api/sessions").json()

    assert len(rows) == 1
    assert rows[0]["title"] == "第一条提问"
    assert rows[0]["session_id"] == session.session_id


def test_空会话不列出来(client, tmp_path):
    """点了「新会话」但没提问会留下一条空记录 —— 那是占位，不是历史。"""
    c, _ = client
    SessionStore(tmp_path).create()  # 空的

    assert len(c.get("/api/sessions").json()) == 1  # 还是只有那条说过话的


def test_删除会话(client):
    c, session = client

    assert c.delete(f"/api/sessions/{session.session_id}").status_code == 200
    assert c.get("/api/sessions").json() == []
    assert not session.path.exists()


def test_删除不存在的会话返回_404(client):
    c, _ = client

    assert c.delete("/api/sessions/不存在的东西").status_code == 404


def test_会话消息能读回来(client):
    c, session = client
    msgs = c.get(f"/api/sessions/{session.session_id}/messages").json()

    assert msgs == [
        {"role": "user", "text": "第一条提问"},
        {"role": "assistant", "text": "回答"},
    ]


def test_设置接口拒绝白名单之外的键(client):
    """`.env` 里还有 API Key，绝不能让网页随便写。"""
    c, _ = client

    resp = c.post("/api/config", json={"values": {"LLM_API_KEY": "偷改"}})

    assert resp.status_code == 400
    assert "不可修改" in resp.json()["detail"]
