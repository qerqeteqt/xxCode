"""Phase 6 AutoDream 测试。

重点在两处收缩上：
    1. 它看到的是会话**摘要**，不是原始 JSONL
    2. 它的工具集里**只有记忆工具**，没有读代码的手段
"""

import asyncio
import json

from app.memory.auto_dream import (
    MAX_SESSIONS,
    AutoDream,
    summarize_session,
)
from app.memory.memory_manager import MemoryManager
from app.memory.memory_tools import memory_tools
from app.memory.session_store import SessionStore


def _run(coro):
    return asyncio.run(coro)


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class ScriptedLLM:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []
        self.tools_seen: list = []

    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self._replies:
            raise AssertionError("回复脚本用完了 —— 循环比预期多跑了一轮")
        return self._replies.pop(0)


def _make_session(root, *, question="问题", answer="答案", changed=()):
    session = SessionStore(root).create()
    session.append_message({"role": "user", "content": question})
    session.append_message(
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("Read", {"path": "x.py"})]}
    )
    session.append_message({"role": "tool", "tool_call_id": "call_1", "content": "文件内容"})
    session.append_message({"role": "assistant", "content": answer})
    for path in changed:
        session.add_changed_file(path)
    session.finish("finished")
    return session


# ================================================================ 会话摘要


def test_摘要只取提问_最终回答_改动文件():
    records = [
        {"type": "session_start", "ts": "2026-09-19T10:00:00"},
        {"type": "message", "message": {"role": "user", "content": "加个函数"}},
        {
            "type": "message",
            "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        },
        {"type": "message", "message": {"role": "tool", "content": "一大段文件内容"}},
        {"type": "message", "message": {"role": "assistant", "content": "加好了"}},
        {"type": "state", "state": {"files_changed": ["src/calc.py"]}},
    ]

    summary = summarize_session(records, "s1")

    assert "加个函数" in summary
    assert "加好了" in summary
    assert "src/calc.py" in summary
    # 过程噪声必须被丢掉 —— 这是「不原样复制 Session」的落地点
    assert "一大段文件内容" not in summary


def test_最终回答取最后一条而不是第一条():
    records = [
        {"type": "message", "message": {"role": "user", "content": "问"}},
        {"type": "message", "message": {"role": "assistant", "content": "中间想法"}},
        {"type": "message", "message": {"role": "tool", "content": "结果"}},
        {"type": "message", "message": {"role": "assistant", "content": "最终结论"}},
    ]

    summary = summarize_session(records, "s1")

    assert "最终结论" in summary
    assert "中间想法" not in summary


def test_没有用户提问的会话不产出摘要():
    records = [{"type": "session_start", "ts": "..."}]
    assert summarize_session(records, "s1") is None


def test_过长的回答被截断():
    records = [
        {"type": "message", "message": {"role": "user", "content": "问"}},
        {"type": "message", "message": {"role": "assistant", "content": "很长的回答" * 2000}},
    ]

    summary = summarize_session(records, "s1")

    assert "已截断" in summary
    assert len(summary) < 3000


# ================================================================ 材料组装


def test_材料包含会话_现有记忆索引和任务说明(tmp_path):
    _make_session(tmp_path, question="第一个问题", answer="第一个答案")
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "内容", name="项目架构")

    dream = AutoDream(tmp_path, ScriptedLLM([]))
    material, used = dream.build_material(dream.store.session_files())

    assert used == 1
    assert "第一个问题" in material
    assert "## 现有长期记忆" in material
    assert "项目架构" in material
    assert "## 你的任务" in material


def test_没有记忆时材料里如实说明(tmp_path):
    _make_session(tmp_path)

    dream = AutoDream(tmp_path, ScriptedLLM([]))
    material, _ = dream.build_material(dream.store.session_files())

    assert "还没有任何长期记忆" in material


def test_会话数量有上限(tmp_path):
    for i in range(MAX_SESSIONS + 3):
        _make_session(tmp_path, question=f"问题{i}")

    dream = AutoDream(tmp_path, ScriptedLLM([]))
    _, used = dream.build_material(dream.store.session_files())

    assert used == MAX_SESSIONS


def test_坏掉的会话被跳过而不是毁掉整轮(tmp_path):
    _make_session(tmp_path, question="好会话")
    broken = tmp_path / ".agent" / "sessions" / "2026-01-01" / "坏的.jsonl"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("这不是 JSONL\n也不是\n", encoding="utf-8")

    dream = AutoDream(tmp_path, ScriptedLLM([]))
    material, used = dream.build_material(dream.store.session_files())

    # 坏文件没有 session_start，会被 session_at 拒绝；好的那个照常纳入
    assert used == 1
    assert "好会话" in material


# ================================================================ Context 隔离


def test_AutoDream_只有_system_和_user_两条消息(tmp_path):
    _make_session(tmp_path)
    llm = ScriptedLLM([_assistant("没记什么")])
    dream = AutoDream(tmp_path, llm)

    _run(dream.run(dream.store.session_files()))

    assert [m["role"] for m in llm.calls[0]] == ["system", "user"]


def test_AutoDream_的工具集里只有记忆工具(tmp_path):
    """它想跑去读项目代码都没有工具可用。"""
    _make_session(tmp_path)
    llm = ScriptedLLM([_assistant("没记什么")])
    dream = AutoDream(tmp_path, llm)

    _run(dream.run(dream.store.session_files()))

    exposed = {s["function"]["name"] for s in llm.tools_seen[0]}
    assert exposed == {"ReadMemory", "WriteMemory", "UpdateMemory", "DeleteMemory"}
    for forbidden in ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "SubAgent"):
        assert forbidden not in exposed


def test_没有会话时不启动_LLM(tmp_path):
    llm = ScriptedLLM([])
    dream = AutoDream(tmp_path, llm)

    result = _run(dream.run([]))

    assert result.sessions_used == 0
    assert not llm.calls  # 一次 LLM 都没调


# ================================================================ 执行


def test_AutoDream_能写出记忆(tmp_path):
    _make_session(tmp_path, question="项目约定是什么", answer="要写测试")
    llm = ScriptedLLM(
        [
            _assistant(
                tool_calls=[
                    _tool_call(
                        "WriteMemory",
                        {
                            "key": "conventions",
                            "content": "- 每加一个函数都要补测试",
                            "name": "项目约定",
                            "description": "加函数时要做什么",
                            "category": "Project",
                        },
                    )
                ]
            ),
            _assistant("新建了 1 条记忆：项目约定。"),
        ]
    )
    dream = AutoDream(tmp_path, llm)

    result = _run(dream.run(dream.store.session_files()))

    assert result.completed
    assert result.changed == [".agent/memory/conventions.md"]
    # 记忆真的落盘了，而且索引同步了
    written = MemoryManager(tmp_path).read_memory("conventions")
    assert "每加一个函数都要补测试" in written.content
    assert "项目约定" in (tmp_path / ".agent" / "memory" / "MEMORY.md").read_text(
        encoding="utf-8"
    )


def test_更新已有记忆(tmp_path):
    _make_session(tmp_path)
    MemoryManager(tmp_path).write_memory("arch", "旧内容", name="架构")
    llm = ScriptedLLM(
        [
            _assistant(
                tool_calls=[_tool_call("UpdateMemory", {"key": "arch", "content": "新内容"})]
            ),
            _assistant("更新了架构记忆。"),
        ]
    )

    dream = AutoDream(tmp_path, llm)
    _run(dream.run(dream.store.session_files()))

    updated = MemoryManager(tmp_path).read_memory("arch")
    assert updated.content == "新内容"
    assert updated.name == "架构"  # 元信息保留


def test_写非法_key_被回灌而不是崩溃(tmp_path):
    _make_session(tmp_path)
    llm = ScriptedLLM(
        [
            _assistant(
                tool_calls=[
                    _tool_call(
                        "WriteMemory",
                        {"key": "../../逃逸", "content": "x", "name": "坏"},
                    )
                ]
            ),
            _assistant("那个 key 不合法，我换成合法的名字重写。"),
        ]
    )
    dream = AutoDream(tmp_path, llm)

    result = _run(dream.run(dream.store.session_files()))

    assert result.completed
    assert result.changed == []
    assert not (tmp_path / "逃逸.md").exists()


def test_超步数时标记为未完成(tmp_path):
    _make_session(tmp_path)
    llm = ScriptedLLM(
        [
            _assistant(
                tool_calls=[_tool_call("ReadMemory", {"key": "不存在"}, call_id=f"c{i}")]
            )
            for i in range(3)
        ]
    )
    dream = AutoDream(tmp_path, llm, max_steps=3)

    result = _run(dream.run(dream.store.session_files()))

    assert result.completed is False
    assert "未完成" in result.summary


# ================================================================ 记忆工具本身


def test_读记忆带上元信息(tmp_path):
    """带上 key/name/category，模型才能正确地更新它。"""
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "正文", name="架构", description="说明", category="Project")
    tools = {t.name: t for t in memory_tools(memory)}

    result = _run(tools["ReadMemory"].execute(key="arch"))

    assert result.ok
    assert "key: arch" in result.text
    assert "name: 架构" in result.text
    assert "category: Project" in result.text
    assert "正文" in result.text


def test_删除记忆填了_changed_path(tmp_path):
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "正文")
    tools = {t.name: t for t in memory_tools(memory)}

    result = _run(tools["DeleteMemory"].execute(key="arch"))

    assert result.ok
    assert result.changed_path == ".agent/memory/arch.md"
    assert not (tmp_path / ".agent" / "memory" / "arch.md").exists()
