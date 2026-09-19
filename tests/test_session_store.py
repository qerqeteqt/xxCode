"""Phase 4 Session 持久化测试。

除了存储层本身，重点测两件容易写错、错了又很难发现的事：

    1. 恢复出来的 messages 必须**能直接喂给 LLM**（含 system、顺序正确）
    2. root 不一致时必须拒绝 —— 否则会在 A 项目的上下文里用 B 项目的沙箱
"""

import asyncio
import json
import os
import time
from datetime import datetime

import pytest

from app.agent.main_agent import MainAgent
from app.memory.session_store import (
    SESSION_DIR,
    SessionError,
    SessionState,
    SessionStore,
)
from app.tools import build_default_registry


def _run(coro):
    return asyncio.run(coro)


def _assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class ScriptedLLM:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise AssertionError("回复脚本用完了 —— 循环比预期多跑了一轮")
        return self._replies.pop(0)


# ================================================================ 存储层


def test_会话文件落在日期目录下(tmp_path):
    session = SessionStore(tmp_path).create()

    # 目录名必须是合法日期（解析失败会抛 ValueError，测试即失败）
    datetime.strptime(session.path.parent.name, "%Y-%m-%d")
    # 文件名就是会话 id，而会话 id 本身以同样的日期开头 —— 所以目录名和 id 对得上
    assert session.session_id.startswith(session.path.parent.name.replace("-", ""))
    assert session.path.parent.parent == tmp_path / SESSION_DIR
    assert session.path.name == f"{session.session_id}.jsonl"
    assert session.path.exists()


def test_session_start_记下_root(tmp_path):
    session = SessionStore(tmp_path).create()
    first = json.loads(session.path.read_text(encoding="utf-8").splitlines()[0])

    assert first["type"] == "session_start"
    assert first["root"] == str(tmp_path.resolve())
    assert first["v"] == 1


def test_写入的消息能原样读回(tmp_path):
    session = SessionStore(tmp_path).create()
    session.append_message({"role": "user", "content": "你好"})
    session.append_message({"role": "assistant", "content": "你好呀"})

    assert session.load_messages() == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


def test_中文不会被转义(tmp_path):
    """ensure_ascii=False —— JSONL 是给人看的，不该满屏 \\uXXXX。"""
    session = SessionStore(tmp_path).create()
    session.append_message({"role": "user", "content": "帮我看看登录模块"})

    assert "帮我看看登录模块" in session.path.read_text(encoding="utf-8")


def test_损坏的行被跳过而不是让整个会话读不出来(tmp_path):
    """append-only 的代价：崩溃时最后一行可能是写了一半的。"""
    session = SessionStore(tmp_path).create()
    session.append_message({"role": "user", "content": "第一条"})

    with session.path.open("a", encoding="utf-8") as f:
        f.write('{"type":"message","message":{"role":"user","conte')  # 写了一半

    assert session.load_messages() == [{"role": "user", "content": "第一条"}]


# ================================================================ State


def test_add_changed_file_写_state_行且去重(tmp_path):
    session = SessionStore(tmp_path).create()
    session.add_changed_file("a.py")
    session.add_changed_file("a.py")
    session.add_changed_file("b.py")

    assert session.state.files_changed == ["a.py", "b.py"]

    state_lines = [
        json.loads(line)
        for line in session.path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "state"
    ]
    # 一次 start + 两次真正变化
    assert len(state_lines) == 3
    assert state_lines[-1]["state"]["files_changed"] == ["a.py", "b.py"]


def test_finish_记下最终状态(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    session.finish("finished")

    assert store.load(session.session_id).state.status == "finished"


def test_未结束的会话状态是_running(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()

    assert store.load(session.session_id).state.status == "running"


def test_SessionState_可以往返序列化():
    state = SessionState(
        session_id="s1",
        status="finished",
        files_changed=["a.py"],
        prompt_tokens=1200,
        completion_tokens=340,
        llm_calls=7,
    )
    assert SessionState.from_dict(state.to_dict()) == state


def test_记录_token_用量(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    session.record_usage(prompt_tokens=1200, completion_tokens=340, calls=7)
    session.finish("finished")

    resumed = store.load(session.session_id)
    assert resumed.state.prompt_tokens == 1200
    assert resumed.state.completion_tokens == 340
    assert resumed.state.total_tokens == 1540
    assert resumed.state.llm_calls == 7


def test_旧会话没有_token_字段也不炸(tmp_path):
    """Phase 7 之前写的会话文件里没有这几个字段，读回来该当 0 而不是 KeyError。"""
    session = SessionStore(tmp_path).create()
    session.finish("finished")

    text = session.path.read_text(encoding="utf-8")
    stripped = "\n".join(
        line
        for line in text.splitlines()
        if '"prompt_tokens"' not in line
    )
    session.path.write_text(stripped + "\n", encoding="utf-8")

    assert SessionStore(tmp_path).load(session.session_id).state.total_tokens == 0


def test_list_sessions_带出_token_用量(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    session.record_usage(1000, 200, 3)
    session.finish("finished")

    assert store.list_sessions()[0].total_tokens == 1200


# ================================================================ 查找与恢复


def test_按唯一前缀查找(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    # 用后半段（含随机后缀）当前缀，保证唯一
    prefix = session.session_id[-4:]

    assert store.find(prefix).stem == session.session_id


def test_前缀不唯一时报错并列出候选(tmp_path):
    store = SessionStore(tmp_path)
    store.create()
    store.create()

    with pytest.raises(SessionError, match="匹配到 2 个会话"):
        store.find("2")  # 时间戳都以 2 开头


def test_找不到会话时报错(tmp_path):
    with pytest.raises(SessionError, match="找不到会话"):
        SessionStore(tmp_path).find("不存在的东西")


def test_项目被改名或搬走后拒绝恢复(tmp_path):
    """会话记录的 root 就是当时的沙箱边界。把项目目录改名/搬走之后，
    .agent 跟着一起过去了，但记录的绝对路径还停在旧位置 —— 这时必须拒绝，
    否则会在旧路径的上下文里用新路径的沙箱。

    这正是 PythonProject 改名成 xxCode 会发生的事。
    """
    old_dir = tmp_path / "旧名字"
    old_dir.mkdir()
    session = SessionStore(old_dir).create()

    new_dir = tmp_path / "新名字"
    old_dir.rename(new_dir)

    with pytest.raises(SessionError, match="请用 --root"):
        SessionStore(new_dir).load(session.session_id)


def test_latest_返回最近活跃的那个(tmp_path):
    """排序依据是 mtime（最近活跃），不是文件名里的创建时间 ——
    同一秒建的两个会话，文件名分不出先后。"""
    store = SessionStore(tmp_path)
    first = store.create()
    second = store.create()

    # 把 first 的 mtime 拨回 10 秒前，模拟"它更早就不再活动了"
    stale = time.time() - 10
    os.utime(first.path, (stale, stale))

    assert store.load_latest().session_id == second.session_id


def test_没有会话时_load_latest_报错(tmp_path):
    with pytest.raises(SessionError, match="无法 --continue"):
        SessionStore(tmp_path).load_latest()


def test_list_sessions_返回摘要(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    session.append_message({"role": "user", "content": "你好"})
    session.add_changed_file("a.py")
    session.finish("finished")

    infos = store.list_sessions()

    assert len(infos) == 1
    assert infos[0].session_id == session.session_id
    assert infos[0].status == "finished"
    assert infos[0].message_count == 1
    assert infos[0].files_changed == ["a.py"]


# ================================================================ 与 Agent 集成


def test_跑一轮之后_jsonl_里有完整会话(tmp_path):
    """system prompt 不落盘（它是每次现拼的配置，见 MainAgent.run 的注释），
    所以 JSONL 里只有对话本身。"""
    session = SessionStore(tmp_path).create()
    llm = ScriptedLLM([_assistant("我是答案")])
    registry = build_default_registry(tmp_path, on_file_changed=session.add_changed_file)
    agent = MainAgent(llm=llm, registry=registry, session=session)

    answer = _run(agent.run("你好"))
    session.finish("finished")

    assert answer == "我是答案"
    messages = session.load_messages()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "你好"
    assert messages[1]["content"] == "我是答案"


def test_恢复会话后接着聊能看到之前的历史(tmp_path):
    """这是 Phase 4 存在的全部理由：关掉进程再打开，上下文还在。"""
    store = SessionStore(tmp_path)

    session = store.create()
    llm = ScriptedLLM([_assistant("第一次回答")])
    registry = build_default_registry(tmp_path, on_file_changed=session.add_changed_file)
    _run(MainAgent(llm=llm, registry=registry, session=session).run("第一个问题"))
    session.finish("finished")

    # 模拟"下一次启动"：只从磁盘拿会话，内存里的东西一概没有
    resumed = store.load(session.session_id)
    llm2 = ScriptedLLM([_assistant("第二次回答")])
    registry2 = build_default_registry(tmp_path, on_file_changed=resumed.add_changed_file)
    _run(MainAgent(llm=llm2, registry=registry2, session=resumed).run("第二个问题"))

    first_call = llm2.calls[0]
    assert [m["role"] for m in first_call] == ["system", "user", "assistant", "user"]
    assert first_call[1]["content"] == "第一个问题"
    assert first_call[2]["content"] == "第一次回答"
    assert first_call[-1]["content"] == "第二个问题"
    # system 只应该有一条 —— 恢复时不能再插一条
    assert sum(1 for m in first_call if m["role"] == "system") == 1


def test_工具改动文件会实时进_state(tmp_path):
    def _tool_call(name, args, call_id="call_1"):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    session = SessionStore(tmp_path).create()
    llm = ScriptedLLM(
        [
            _assistant(tool_calls=[_tool_call("Write", {"path": "新文件.py", "content": "X = 1\n"})]),
            _assistant("建好了"),
        ]
    )
    registry = build_default_registry(tmp_path, on_file_changed=session.add_changed_file)
    _run(MainAgent(llm=llm, registry=registry, session=session).run("建个文件"))
    session.finish("finished")

    # 落盘之后再读回来，改动清单还在
    assert SessionStore(tmp_path).load(session.session_id).state.files_changed == ["新文件.py"]
