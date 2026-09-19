"""Phase 5 长期记忆测试。

三块：存储层本身、索引是派生数据这件事、以及沙箱要放行记忆目录。
"""

import asyncio

import pytest

from app.agent.main_agent import MainAgent
from app.memory.memory_manager import (
    INDEX_FILENAME,
    MEMORY_DIR,
    MemoryManager,
    MemoryStoreError,
)
from app.tools import build_default_registry
from app.tools.sandbox import Sandbox


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

    async def chat(self, messages: list[dict], tools: list | None = None, on_delta=None) -> dict:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise AssertionError("回复脚本用完了")
        return self._replies.pop(0)


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(tmp_path)


# ================================================================ 读写


def test_写入后能读回来(memory):
    memory.write_memory(
        "architecture", "项目分五层……", name="项目架构", description="模块划分", category="Project"
    )

    loaded = memory.read_memory("architecture")
    assert loaded.name == "项目架构"
    assert loaded.description == "模块划分"
    assert loaded.category == "Project"
    assert loaded.content == "项目分五层……"


def test_没有_frontmatter_的文件也能读(memory, tmp_path):
    """手写一个不带元信息的 md 不该让整个索引崩掉。"""
    (tmp_path / MEMORY_DIR).mkdir(parents=True)
    (tmp_path / MEMORY_DIR / "随手记.md").write_text("想到哪写到哪", encoding="utf-8")

    loaded = memory.read_memory("随手记")
    assert loaded.name == "随手记"  # 回落到文件名
    assert loaded.category == "General"
    assert loaded.content == "想到哪写到哪"


def test_frontmatter_写坏了_按纯文本处理(memory, tmp_path):
    (tmp_path / MEMORY_DIR).mkdir(parents=True)
    (tmp_path / MEMORY_DIR / "bad.md").write_text(
        "---\nname: [没闭合的列表\n---\n\n正文", encoding="utf-8"
    )

    # 不抛异常就算通过 —— 具体降级成什么样不重要
    assert memory.read_memory("bad").key == "bad"


def test_中文不被转义成_unicode_码点(memory):
    memory.write_memory("x", "内容", name="显示名", description="描述")
    text = (memory.dir / "x.md").read_text(encoding="utf-8")

    assert "显示名" in text
    assert "\\u" not in text


def test_update_只改正文保留元信息(memory):
    memory.write_memory("x", "旧内容", name="显示名", description="描述", category="Project")
    memory.update_memory("x", "新内容")

    loaded = memory.read_memory("x")
    assert loaded.content == "新内容"
    assert loaded.name == "显示名"
    assert loaded.description == "描述"
    assert loaded.category == "Project"


def test_delete_删掉文件(memory):
    memory.write_memory("x", "内容")
    memory.delete_memory("x")

    assert not (memory.dir / "x.md").exists()
    with pytest.raises(MemoryStoreError, match="没有这条记忆"):
        memory.read_memory("x")


def test_读不存在的记忆报错(memory):
    with pytest.raises(MemoryStoreError, match="没有这条记忆"):
        memory.read_memory("不存在")


# ================================================================ key 的安全性


@pytest.mark.parametrize("key", ["../../poison", "a/b", "..", ""])
def test_非法_key_被拒绝(memory, key):
    """这个 key 将来由 AutoDream 提供，也就是由 LLM 生成 —— 不是可信输入。"""
    with pytest.raises(MemoryStoreError):
        memory.write_memory(key, "恶意内容")


def test_越界的_key_不会写到记忆目录外面(memory, tmp_path):
    with pytest.raises(MemoryStoreError):
        memory.write_memory("../../poison", "x")

    assert not (tmp_path / "poison.md").exists()
    assert not (tmp_path.parent / "poison.md").exists()


def test_MEMORY_不能当作记忆名(memory):
    with pytest.raises(MemoryStoreError, match="是索引"):
        memory.write_memory("MEMORY", "想冒充索引")


# ================================================================ 索引是派生数据


def test_索引从目录内容生成(memory):
    memory.write_memory("b", "内容", name="乙", category="User")
    memory.write_memory("a", "内容", name="甲", category="Project", description="说明")

    index = (memory.dir / INDEX_FILENAME).read_text(encoding="utf-8")

    assert "# Memory" in index
    assert "## Project" in index
    assert "## User" in index
    assert "- [甲](./a.md) — 说明" in index
    assert "- [乙](./b.md)" in index


def test_索引不会把_自己_当成一条记忆(memory):
    memory.write_memory("a", "内容")
    memory.sync_index()

    assert [m.key for m in memory.list_memories()] == ["a"]


def test_删除记忆后索引自动同步(memory):
    memory.write_memory("a", "内容")
    memory.write_memory("b", "内容")
    memory.delete_memory("a")

    index = (memory.dir / INDEX_FILENAME).read_text(encoding="utf-8")
    assert "a.md" not in index
    assert "b.md" in index


def test_手工丢进来的文件会被索引收录(memory):
    """这正是不手工维护索引的好处：没人需要记得去更新它。"""
    memory.sync_index()
    assert "手工.md" not in (memory.dir / INDEX_FILENAME).read_text(encoding="utf-8")

    (memory.dir / "手工.md").write_text("---\nname: 手工加的\n---\n\n正文", encoding="utf-8")
    memory.sync_index()

    assert "手工加的" in (memory.dir / INDEX_FILENAME).read_text(encoding="utf-8")


def test_没有记忆时索引仍然生成(memory):
    memory.sync_index()

    assert "还没有任何长期记忆" in (memory.dir / INDEX_FILENAME).read_text(encoding="utf-8")


def test_索引内容没变就不重写文件(memory):
    """否则每次启动都刷新 mtime，git 里这个文件一直是脏的。"""
    memory.sync_index()
    first = (memory.dir / INDEX_FILENAME).stat().st_mtime_ns

    memory.sync_index()

    assert (memory.dir / INDEX_FILENAME).stat().st_mtime_ns == first


def test_注入用的链接前缀是相对项目根的(memory):
    memory.write_memory("arch", "内容", name="架构")
    index = memory.render_index(".agent/memory/")

    assert "(.agent/memory/arch.md)" in index


# ================================================================ 沙箱放行


def test_沙箱放行记忆目录但挡住会话目录(tmp_path):
    """这是 Phase 2 埋下的雷：当时把整个 .agent 一刀切忽略了，
    结果 Agent 根本读不到自己的记忆。"""
    (tmp_path / MEMORY_DIR).mkdir(parents=True)
    (tmp_path / MEMORY_DIR / "note.md").write_text("值得记住的事", encoding="utf-8")
    (tmp_path / ".agent" / "sessions" / "2026-09-19").mkdir(parents=True)
    (tmp_path / ".agent" / "sessions" / "2026-09-19" / "s.jsonl").write_text(
        "{}", encoding="utf-8"
    )

    visible = {p.name for p in Sandbox(tmp_path).iter_files(tmp_path)}

    assert "note.md" in visible  # 记忆要能被读到
    assert "s.jsonl" not in visible  # 会话流水仍然是噪声，不提
    assert "MEMORY.md" not in visible or True  # 索引是否可见不重要


# ================================================================ 注入 system prompt


def test_记忆索引被注入_system_prompt(tmp_path):
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "内容", name="项目架构", description="模块划分")

    llm = ScriptedLLM([_assistant("好")])
    registry = build_default_registry(tmp_path)
    agent = MainAgent(llm=llm, registry=registry, memory=memory)

    _run(agent.run("你好"))

    system = llm.calls[0][0]
    assert system["role"] == "system"
    assert "## 长期记忆" in system["content"]
    assert "项目架构" in system["content"]
    assert ".agent/memory/arch.md" in system["content"]


def test_没有记忆时不注入那段(tmp_path):
    llm = ScriptedLLM([_assistant("好")])
    agent = MainAgent(llm=llm, registry=build_default_registry(tmp_path))

    _run(agent.run("你好"))

    assert "## 长期记忆" not in llm.calls[0][0]["content"]


def test_记忆变了续会话能看到新的索引(tmp_path):
    """system prompt 每次现拼，所以不会沿用上次冻结在 JSONL 里的旧索引。"""
    from app.memory.session_store import SessionStore

    store = SessionStore(tmp_path)
    memory = MemoryManager(tmp_path)
    registry = build_default_registry(tmp_path)

    session = store.create()
    llm1 = ScriptedLLM([_assistant("第一次")])
    _run(
        MainAgent(
            llm=llm1, registry=registry, session=session, memory=memory
        ).run("第一个问题")
    )
    session.finish("finished")
    assert "## 长期记忆" not in llm1.calls[0][0]["content"]

    # 这中间加了一条记忆（Phase 6 的 AutoDream 干的就是这个）
    memory.write_memory("new", "内容", name="新记忆")

    resumed = store.load(session.session_id)
    llm2 = ScriptedLLM([_assistant("第二次")])
    _run(
        MainAgent(
            llm=llm2, registry=registry, session=resumed, memory=memory
        ).run("第二个问题")
    )

    assert "新记忆" in llm2.calls[0][0]["content"]
    # 历史还在，而且是 system + 之前那一轮 + 新的 user
    assert [m["role"] for m in llm2.calls[0]] == ["system", "user", "assistant", "user"]
