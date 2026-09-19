"""Phase 6 Scheduler 测试。

核心是触发条件：**两个条件是并且关系**，只满足一个不跑。
以及安全网：整理前必须备份、整理没跑完不能更新状态。
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta

from app.memory.auto_dream import DreamResult
from app.memory.memory_manager import MemoryManager
from app.memory.session_store import SessionStore
from app.scheduler.scheduler import KEEP_BACKUPS, Scheduler


def _run(coro):
    return asyncio.run(coro)


def _make_session(root, question="问题", age_hours: float = 0.0):
    """建一个会话，可选把 mtime 拨到过去。"""
    session = SessionStore(root).create()
    session.append_message({"role": "user", "content": question})
    session.append_message({"role": "assistant", "content": "答案"})
    session.finish("finished")
    if age_hours:
        stamp = time.time() - age_hours * 3600
        os.utime(session.path, (stamp, stamp))
    return session


def _set_last_run(root, hours_ago: float) -> None:
    path = root / ".agent" / "consolidation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    when = datetime.now() - timedelta(hours=hours_ago)
    path.write_text(
        json.dumps({"last_run": when.isoformat(timespec="seconds")}), encoding="utf-8"
    )


class FakeLLM:
    """整理阶段不需要真 LLM —— 这些测试只关心「跑不跑」和「备份了没」。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        return {"role": "assistant", "content": "什么都没记"}


# ================================================================ 触发条件


def test_没有会话时不整理(tmp_path):
    decision = Scheduler(tmp_path).check()

    assert decision.should_run is False
    assert "还没有任何会话" in decision.reason


def test_从未整理过但有会话时要整理(tmp_path):
    _make_session(tmp_path)

    decision = Scheduler(tmp_path).check()

    assert decision.should_run is True
    assert "从未整理过" in decision.reason


def test_两个条件都满足才跑(tmp_path):
    for i in range(5):
        _make_session(tmp_path, question=f"问题{i}")
    _set_last_run(tmp_path, hours_ago=30)

    decision = Scheduler(tmp_path).check()

    assert decision.should_run is True


def test_时间到了但会话不够不跑(tmp_path):
    """这是「并且」语义的关键用例。"""
    for i in range(3):
        _make_session(tmp_path, question=f"问题{i}")
    _set_last_run(tmp_path, hours_ago=48)

    decision = Scheduler(tmp_path).check()

    assert decision.should_run is False
    assert "新增 3 个会话" in decision.reason


def test_会话够了但时间没到不跑(tmp_path):
    for i in range(9):
        _make_session(tmp_path, question=f"问题{i}")
    _set_last_run(tmp_path, hours_ago=2)

    decision = Scheduler(tmp_path).check()

    assert decision.should_run is False
    assert "2.0 小时" in decision.reason


def test_只统计上次整理之后的新会话(tmp_path):
    _make_session(tmp_path, question="老的", age_hours=50)
    for i in range(4):
        _make_session(tmp_path, question=f"新的{i}")
    _set_last_run(tmp_path, hours_ago=24)

    assert len(Scheduler(tmp_path).new_sessions()) == 4


def test_阈值可调(tmp_path):
    _make_session(tmp_path)
    _set_last_run(tmp_path, hours_ago=1)

    assert Scheduler(tmp_path).check().should_run is False
    assert Scheduler(tmp_path, min_hours=0, min_sessions=1).check().should_run is True


def test_状态文件坏掉时当作从未整理过(tmp_path):
    _make_session(tmp_path)
    path = tmp_path / ".agent" / "consolidation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{不是 json", encoding="utf-8")

    decision = Scheduler(tmp_path).check()

    # 大不了多整理一次，总比因为一个坏文件再也整理不了强
    assert decision.should_run is True


# ================================================================ 执行与安全网


def test_不满足条件时不启动_AutoDream(tmp_path):
    _make_session(tmp_path)
    _set_last_run(tmp_path, hours_ago=1)
    llm = FakeLLM()

    result = _run(Scheduler(tmp_path).consolidate(llm))

    assert llm.calls == 0
    assert "跳过整理" in result.summary


def test_强制整理跳过条件判断(tmp_path):
    _make_session(tmp_path)
    _set_last_run(tmp_path, hours_ago=0.1)
    llm = FakeLLM()

    result = _run(Scheduler(tmp_path, min_hours=999).consolidate(llm, force=True))

    assert llm.calls > 0
    assert result.sessions_used == 1


def test_整理成功后更新状态文件(tmp_path):
    _make_session(tmp_path)
    _run(Scheduler(tmp_path).consolidate(FakeLLM()))

    state = json.loads((tmp_path / ".agent" / "consolidation.json").read_text("utf-8"))
    assert "last_run" in state
    # 状态一写，立刻就不该再触发了
    assert Scheduler(tmp_path).check().should_run is False


def test_整理前自动备份记忆(tmp_path):
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "重要内容", name="架构")
    _make_session(tmp_path)

    _run(Scheduler(tmp_path).consolidate(FakeLLM()))

    backups = list((tmp_path / ".agent" / "memory-backups").glob("*"))
    assert len(backups) == 1
    assert (backups[0] / "arch.md").read_text(encoding="utf-8").count("重要内容") == 1


def test_同秒内的多次备份不会互相覆盖(tmp_path):
    """覆盖掉的那一份很可能正是「还没被搞坏」的干净备份。"""
    MemoryManager(tmp_path).write_memory("arch", "内容")
    scheduler = Scheduler(tmp_path)

    first = scheduler.backup_memory()
    second = scheduler.backup_memory()

    assert first != second
    assert first.exists() and second.exists()


def test_备份数量有上限(tmp_path):
    memory = MemoryManager(tmp_path)
    memory.write_memory("arch", "内容")
    _make_session(tmp_path)
    scheduler = Scheduler(tmp_path)

    # 直接调 backup 五次以上，绕过「整理成功才更新状态」的限制。
    # 不用错开时间 —— 同秒内连续备份也必须各自留下目录，不能互相覆盖
    for _ in range(KEEP_BACKUPS + 3):
        scheduler.backup_memory()

    backups = list((tmp_path / ".agent" / "memory-backups").glob("*"))
    assert len(backups) == KEEP_BACKUPS


def test_没有记忆目录时不备份(tmp_path):
    _make_session(tmp_path)

    assert Scheduler(tmp_path).backup_memory() is None


def test_整理没跑完不更新状态(tmp_path, monkeypatch):
    """半途而废的整理不该被当成「已经整理过」，否则下次就跳过了。"""
    _make_session(tmp_path)
    scheduler = Scheduler(tmp_path)

    async def _failed(self, session_paths):
        return DreamResult(summary="没跑完", sessions_used=1, completed=False)

    monkeypatch.setattr("app.memory.auto_dream.AutoDream.run", _failed)

    result = _run(scheduler.consolidate(FakeLLM()))

    assert result.completed is False
    assert not (tmp_path / ".agent" / "consolidation.json").exists()


def test_没有新会话时不启动(tmp_path):
    _set_last_run(tmp_path, hours_ago=30)
    llm = FakeLLM()

    result = _run(Scheduler(tmp_path).consolidate(llm, force=True))

    assert llm.calls == 0
    assert "没有新会话" in result.summary
