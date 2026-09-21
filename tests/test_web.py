"""Web 层测试。

重点是两块容易出事的地方：
    1. 写 .env 必须保留注释，且**写坏了要回滚**（否则下次启动服务起不来）
    2. 事件要真的从 Runtime 里发出来，而不是「前端等着一个永远不来的事件」
"""

import asyncio
import base64
import json
import re
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.agent.react_loop import run_react_loop
from app.events import Event, emit, tag_events  # noqa: F401  (tag_events 由子 Agent 用)
from app.llm.client import TokenUsage
from app.memory.memory_manager import MemoryManager
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
            max_steps=10,
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


# ================================================================ 对话端点


class _FakeLLM:
    """顶替真的 LLMClient，按脚本回复。

    为什么这个测试值得存在：`/api/chat` 是全项目唯一「只靠手工点过」的路径。
    实测里就是它漏了一个函数定义（局部名只在调用时才解析，静态检查看不出来），
    而这个 bug 只在浏览器里点才暴露 —— 单元测试和 CLI 冒烟都是绿的。
    """

    def __init__(self, *args, **kwargs):
        self.usage = TokenUsage()
        self.last_prompt_tokens = 0
        self._replies = [
            {"role": "assistant", "content": "好的记住了"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "WriteMemory",
                            "arguments": json.dumps(
                                {
                                    "key": "pref", "name": "偏好",
                                    "description": "测试", "category": "User",
                                    "content": "用户喜欢简洁。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "写入 1 条"},
        ]

    async def chat(self, messages, tools=None, on_delta=None):
        # 有 on_delta 说明走流式：把内容切片吐出去，行为要和真 client 一致
        reply = self._replies.pop(0)
        if on_delta is not None and reply.get("content"):
            for ch in reply["content"]:
                on_delta(ch)
        return reply

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


def test_网页发一条消息会跑完整条链路(tmp_path, monkeypatch):
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        resp = client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "记住我喜欢简洁"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

    kinds = [e["type"] for e in events]
    assert "done" in kinds
    # 关键：记忆提取这一步真的跑了，而且带了 phase 标记
    assert "memory" in kinds
    extract = next(e for e in events if e["type"] == "memory")
    assert extract["data"]["phase"] == "extract"
    assert extract["data"]["changed"] == [".agent/memory/pref.md"]

    # 记忆真的写进去了
    assert "简洁" in MemoryManager(tmp_path).read_memory("pref").content


def _parse_sse(text: str) -> list[dict]:
    """把 SSE 文本拆成 [{type, data}, ...]。"""
    out = []
    for block in text.split("\n\n"):
        kind = next((l[7:] for l in block.splitlines() if l.startswith("event: ")), None)
        payload = next((l[6:] for l in block.splitlines() if l.startswith("data: ")), None)
        if kind and payload:
            out.append({"type": kind, "data": json.loads(payload)})
    return out


class _NeverFinishesLLM(_FakeLLM):
    """永远返回 tool_calls —— 用来把循环逼到步数上限。"""

    async def chat(self, messages, tools=None, on_delta=None):
        self.usage = TokenUsage(100, 10, self.usage.calls + 1)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "List", "arguments": '{"path": "."}'},
                }
            ],
        }


def test_撞上限的会话标为_failed_并记下用量(tmp_path, monkeypatch):
    """两个只有在真实使用里才暴露的 bug：

    1. 原先 finally 里不管成败都 finish("finished") —— 撞上限中断的会话
       在列表里显示成正常结束，查问题时会误导
    2. web 端从来不调 session.record_usage —— 会话列表里 token 永远是 0
    """
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _NeverFinishesLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        resp = client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "一直干下去"},
        )
        events = _parse_sse(resp.text)

    error = next(e for e in events if e["type"] == "error")
    assert "最大循环次数" in error["data"]["message"]
    # 报告里要有「做了什么」，而不是一句开场白
    assert "调用了" in error["data"]["partial"]
    assert error["data"]["can_continue"] is True

    reloaded = SessionStore(tmp_path).load(session.session_id)
    assert reloaded.state.status == "failed"  # 不是 finished
    assert reloaded.state.total_tokens > 0  # 用量记下来了


# ================================================================ 终止


class _StallsLLM(_FakeLLM):
    """第 1 步照常调一个只读工具，第 2 步真的挂住，直到被取消。

    为什么先走一步：只有走过一步，会话里才有「中断前的进展」可报。
    一上来就挂住的话 partial 本来就该是空的 —— 那等于没测到 last_progress
    从会话文件里取数这条路。

    **不能复用 _NeverFinishesLLM**：它是靠一直返回 tool_calls 来「不结束」的，
    协程从不 yield —— 而 CancelledError 只在 await 点投递，忙等的协程压根收不到取消。
    所以要真 `await` 住。

    用 threading.Event 而不是 asyncio.Event，是因为测试线程在事件循环外面，
    没法 await。
    """

    started = threading.Event()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).started.clear()
        self._step = 0

    async def chat(self, messages, tools=None, on_delta=None):
        self._step += 1
        self.usage = TokenUsage(100, 10, self.usage.calls + 1)
        if self._step == 1:
            # List 是只读工具，不弹权限窗，正好用来留下一步「进展」
            return {
                "role": "assistant",
                "content": "先看一眼目录",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "List", "arguments": '{"path": "."}'},
                    }
                ],
            }
        type(self).started.set()
        await asyncio.Event().wait()   # 一直等，等外面来取消


def test_停止会中止本轮并推_stopped_事件(tmp_path, monkeypatch):
    """用户按停止：这一轮要停下来，历史仍然合法，状态如实记成 stopped。

    必须用 `with TestClient(...)`：裸 fixture 每次请求新建一个事件循环，
    跨两个请求去取消一个任务会炸「Future attached to a different loop」。
    """
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _StallsLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        box: dict = {}
        # 流式响应会把整个 body 读完之后才返回，所以 /api/chat 得丢到别的线程去，
        # 主线程才有机会发 /api/stop
        thread = threading.Thread(
            target=lambda: box.update(
                resp=client.post(
                    "/api/chat",
                    json={"session_id": session.session_id, "question": "慢慢想"},
                )
            )
        )
        thread.start()
        assert _StallsLLM.started.wait(5), "模型一直没被调用，测不了停止"

        resp = client.post("/api/stop", json={"session_id": session.session_id})
        assert resp.status_code == 200
        # 这条要**先**断言：回归时它会干脆地失败，而不是挂住整个测试套件
        thread.join(5)
        assert not thread.is_alive(), "/api/chat 没有随着停止一起结束"

    events = _parse_sse(box["resp"].text)
    kinds = [e["type"] for e in events]
    assert "stopped" in kinds
    stopped = next(e for e in events if e["type"] == "stopped")
    assert stopped["data"]["can_continue"] is True
    assert "调用了" in stopped["data"]["partial"]   # 中断前的进展有报出来

    reloaded = SessionStore(tmp_path).load(session.session_id)
    assert reloaded.state.status == "stopped"       # 不是 failed，也不是 finished


def test_答案已经给出后再停止会话仍算完成(tmp_path, monkeypatch):
    """答案发出去之后才停（停在记忆提取那一步），这一轮其实是**完成**的。

    报成 stopped 会让会话列表说谎，也会给前端一个没东西可继续的「继续」按钮。
    """
    import app.web.server as server

    class _AnswersThenStalls(_FakeLLM):
        started = threading.Event()
        calls = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).started.clear()
            type(self).calls = 0

        async def chat(self, messages, tools=None, on_delta=None):
            type(self).calls += 1
            if type(self).calls > 1:
                # 第 2 次调用是记忆提取那一步
                type(self).started.set()
                await asyncio.Event().wait()
            reply = self._replies.pop(0)
            if on_delta is not None and reply.get("content"):
                for ch in reply["content"]:
                    on_delta(ch)
            return reply

    monkeypatch.setattr(server, "LLMClient", _AnswersThenStalls)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        box: dict = {}
        thread = threading.Thread(
            target=lambda: box.update(
                resp=client.post(
                    "/api/chat",
                    json={"session_id": session.session_id, "question": "问一句"},
                )
            )
        )
        thread.start()
        assert _AnswersThenStalls.started.wait(5), "记忆提取那一步一直没跑起来"
        client.post("/api/stop", json={"session_id": session.session_id})
        thread.join(5)
        assert not thread.is_alive()

    kinds = [e["type"] for e in _parse_sse(box["resp"].text)]
    assert "done" in kinds
    assert "stopped" not in kinds

    assert SessionStore(tmp_path).load(session.session_id).state.status == "finished"


def test_跑完之后再停止返回_409(tmp_path, monkeypatch):
    """用户慢了半拍：点停止时这一轮刚好已经跑完。

    前端把非 2xx 当静默无操作处理 —— 以 SSE 流为准，这里只是尽力而为，
    不该弹一个「停止失败」的错。
    """
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "记住我喜欢简洁"},
        )
        resp = client.post("/api/stop", json={"session_id": session.session_id})

    assert resp.status_code == 409


def test_停止没跑过的会话返回_404(client):
    """没有 LiveSession = 这个会话根本没在跑。"""
    c, session = client

    assert c.post("/api/stop", json={"session_id": session.session_id}).status_code == 404
    assert c.post("/api/stop", json={"session_id": "根本不存在"}).status_code == 404


def test_收尾会落状态并推结束哨兵(tmp_path, monkeypatch):
    """finalize 是同步的，而且必须在 detach 之前推哨兵 —— 顺序错了流就永远不结束。"""
    import app.web.server as server
    from config.settings import get_settings

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    async def scenario() -> asyncio.Queue:
        live = server.LiveSession(tmp_path, session, get_settings())
        queue: asyncio.Queue = asyncio.Queue()
        live.busy = True
        live.attach(queue)
        live.finalize("stopped")
        await live.close()
        return queue

    queue = _run(scenario())

    assert queue.get_nowait() is None                       # 哨兵推出来了
    assert SessionStore(tmp_path).load(session.session_id).state.status == "stopped"


# ================================================================ 图片


# 一个**真的** 1×1 红色 PNG（69 字节）。不用大 fixture 也不用手搓假字节
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


def _upload(c, data: str = PNG_DATA_URL):
    return c.post("/api/images", json={"data": data})


def test_上传图片返回引用名(client, tmp_path):
    c, _ = client

    resp = _upload(c)

    assert resp.status_code == 200
    ref = resp.json()["ref"]
    assert re.fullmatch(r"[0-9a-f]{16}\.png", ref), ref
    assert resp.json()["bytes"] == len(PNG_BYTES)
    assert (tmp_path / ".agent" / "images" / ref).exists()


def test_上传的图片真的落盘并且能取回(client, tmp_path):
    c, _ = client

    ref = _upload(c).json()["ref"]

    assert (tmp_path / ".agent" / "images" / ref).read_bytes() == PNG_BYTES
    got = c.get(f"/api/images/{ref}")
    assert got.status_code == 200
    assert got.content == PNG_BYTES
    assert got.headers["content-type"] == "image/png"
    # 内容寻址的名字不可变，可以放心永久缓存
    assert "immutable" in got.headers["cache-control"]


def test_同一张图上传两次得到同一个引用(client):
    c, _ = client

    assert _upload(c).json()["ref"] == _upload(c).json()["ref"]


def test_上传非图片被拒绝(client):
    c, _ = client

    text_bytes = "我是文本，不是图片".encode("utf-8")
    resp = _upload(c, "data:image/png;base64," + base64.b64encode(text_bytes).decode())

    assert resp.status_code == 400
    assert "图片" in resp.json()["detail"]


def test_上传的数据不是合法_base64_被拒绝(client):
    c, _ = client

    assert _upload(c, "这不是 base64!!!").status_code == 400


@pytest.mark.parametrize(
    "name",
    ["../../.env", "..%2F..%2F.env", "C:/Windows/win.ini", "abc.png", "0123456789abcdef.exe"],
)
def test_读取非法名字返回_404(client, name):
    """引用名是安全边界：它拼进文件路径，所以校验必须在拼之前。"""
    c, _ = client

    assert c.get(f"/api/images/{name}").status_code == 404


def test_读取不存在的图片返回_404(client):
    c, _ = client

    assert c.get("/api/images/0123456789abcdef.png").status_code == 404


def test_发消息带图片时_jsonl_里只存引用(tmp_path, monkeypatch):
    """**这条证明整个设计的前提成立。**

    图片是 160 KB 级别的二进制，如果 base64 直接落进 append-only 的 JSONL，
    文件就会猛猛膨胀且永不删除。所以盘上必须只有 ref，
    base64 只在发请求前那一刻存在于一份拷贝上。
    """
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        ref = _upload(client).json()["ref"]
        resp = client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "看看这张图", "images": [ref]},
        )
        assert resp.status_code == 200

    raw = session.path.read_text(encoding="utf-8")
    assert f"ref:{ref}" in raw
    # 关键：整个文件里不该有任何 base64
    assert "data:image" not in raw
    assert base64.b64encode(PNG_BYTES).decode()[:40] not in raw


def test_历史消息接口把图片引用带回前端(tmp_path, monkeypatch):
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        ref = _upload(client).json()["ref"]
        client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "看看", "images": [ref]},
        )
        msgs = client.get(f"/api/sessions/{session.session_id}/messages").json()

    first = msgs[0]
    assert first["role"] == "user"
    assert first["text"] == "看看"
    assert first["images"] == [ref]


def test_只贴图不打字也能发出去(tmp_path, monkeypatch):
    """空 question + 有图片是正常用法，不该被拒。"""
    import app.web.server as server

    monkeypatch.setattr(server, "LLMClient", _FakeLLM)
    session = SessionStore(tmp_path).create()

    with TestClient(server.create_app(tmp_path)) as client:
        ref = _upload(client).json()["ref"]
        resp = client.post(
            "/api/chat",
            json={"session_id": session.session_id, "question": "", "images": [ref]},
        )
        msgs = client.get(f"/api/sessions/{session.session_id}/messages").json()

    assert resp.status_code == 200
    assert msgs[0]["text"] == ""
    assert msgs[0]["images"] == [ref]


@pytest.mark.parametrize("bad", ["../../.env", "abc.png", "0123456789abcdef"])
def test_引用格式非法时_chat_返回_400(client, bad):
    c, session = client

    resp = c.post(
        "/api/chat",
        json={"session_id": session.session_id, "question": "看看", "images": [bad]},
    )

    assert resp.status_code == 400


def test_引用合法但图片不存在时_chat_返回_400(client):
    """宁可在这里拒掉，也别让它落进 JSONL 变成一个永久解析不出来的死引用。"""
    c, session = client

    resp = c.post(
        "/api/chat",
        json={
            "session_id": session.session_id,
            "question": "看看",
            "images": ["0123456789abcdef.png"],
        },
    )

    assert resp.status_code == 400
    assert "不存在" in resp.json()["detail"]
