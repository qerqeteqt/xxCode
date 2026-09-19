"""Session 持久化 —— JSONL 事件流。

目录结构：

    .agent/sessions/<YYYY-MM-DD>/<session_id>.jsonl

一天一个文件夹，一个会话一个文件。会话 id 形如 `20260919-182031-a3f2`，本身按
时间可排序，所以「今天有哪些会话」「最近一次是哪个」都是一次目录名排序的事。

每行一条记录，四种类型：

    session_start   会话开始，带 root（也就是当时的沙箱边界）
    message         一条 API 消息，原样存
    state           State 快照（status / files_changed）
    session_end     会话结束，带最终状态

## 两个关键决定

**message 里存的是原样的 API 消息。** 恢复会话时只要过滤出 type=="message"、取出
.message，就得到一份能直接喂给 LLM 的 messages —— 零转换。如果改成记「事件」
（tool_called / tool_result 分开记），恢复时还得写一层拼装逻辑，还要处理
tool_call_id 的配对，而那层逻辑本身就是新的 bug 来源。

**append-only。** 这是 JSONL 相对 JSON 的全部意义：每条消息产生时就写一行，
进程崩了，已经发生的事还在。所以 State 也是「每次变化追加一行」，而不是覆盖写
一个小文件 —— 覆盖写遇到写一半崩溃会留下坏文件，追加不会。

## 一个已知取舍

文件 IO 是同步的，会短暂阻塞事件循环。一次会话几十上百次追加，每次就一行，
当前规模下无感。真要优化应该攒批 + 后台线程，那是后面的事。
"""

import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_DIR = ".agent/sessions"
SCHEMA_VERSION = 1


class SessionError(RuntimeError):
    """会话读写失败。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class SessionState:
    """结构化任务状态。

    文档第八节把 State 列得很全（task / plan / findings / subagent_results），
    但那些在 V1 里没有消费者 —— 模型本来就用自然语言在 messages 里表达了它们。
    这里只留真正有人读的字段：

        session_id      定位会话
        status          会话是否正常结束（恢复、展示都要）
        files_changed   最终回答「改了哪些文件」；Phase 5 的 Memory 也会读
        created_at      列表、排序
        updated_at      同上

    Context（messages）与 State 分离是文档第八节的原则：Context 是给模型看的工作区，
    State 是给程序看的结构化状态。这里只放后者。
    """

    session_id: str
    status: str = "running"  # running | finished | failed
    files_changed: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "files_changed": list(self.files_changed),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        return cls(
            session_id=data.get("session_id", ""),
            status=data.get("status", "running"),
            files_changed=list(data.get("files_changed") or []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class SessionInfo:
    """会话摘要，用于列表展示。"""

    session_id: str
    path: Path
    started_at: str
    status: str
    message_count: int
    files_changed: list[str]


def _read_records(path: Path) -> list[dict]:
    """读一个 JSONL 文件，跳过坏行。

    为什么要容忍坏行：append-only 的代价是「最后一行可能是写了一半的」。
    如果崩溃恰好发生在写入中途，最后一行就是残缺 JSON。跳过它、保留前面
    所有完整的记录，比让整个会话读不出来强得多。
    """
    if not path.exists():
        raise SessionError(f"会话文件不存在: {path}")

    records: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("跳过损坏的记录 %s:%d", path.name, lineno)
    return records


class Session:
    """一次会话。负责把消息和状态增量写进 JSONL，以及把消息读回来。"""

    def __init__(
        self,
        store: "SessionStore",
        session_id: str,
        path: Path,
        state: SessionState,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.path = path
        self.state = state

    # ---------------------------------------------------------- 写入

    def _append(self, record: dict) -> None:
        record = {"v": SCHEMA_VERSION, "ts": _now(), "session_id": self.session_id, **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start(self, root: Path) -> None:
        """写会话头。root 要记下来 —— 恢复会话时靠它校验沙箱边界是否一致。"""
        self.state.created_at = self.state.updated_at = _now()
        self._append({"type": "session_start", "root": str(root)})
        self._append({"type": "state", "state": self.state.to_dict()})

    def append_message(self, message: dict) -> None:
        """记一条消息。直接挂在 ReAct Loop 的 on_message 钩子上。"""
        self._append({"type": "message", "message": message})

    def add_changed_file(self, path: str) -> None:
        """记一个被真正改动的文件。挂在 TrackingRegistry 的 on_change 上。"""
        if path in self.state.files_changed:
            return
        self.state.files_changed.append(path)
        self.state.updated_at = _now()
        self._append({"type": "state", "state": self.state.to_dict()})

    def finish(self, status: str) -> None:
        self.state.status = status
        self.state.updated_at = _now()
        self._append({"type": "session_end", "status": status, "state": self.state.to_dict()})

    # ---------------------------------------------------------- 读取

    def load_messages(self) -> list[dict]:
        """读出可以直接喂给 LLM 的 messages。

        零转换：JSONL 里存的就是原样的 API 消息，这里只是把它们摘出来。
        system 消息也在里面，所以恢复会话时不需要另外拼 system prompt。
        """
        return [
            record["message"]
            for record in _read_records(self.path)
            if record.get("type") == "message" and isinstance(record.get("message"), dict)
        ]

    def summary(self) -> SessionInfo:
        records = _read_records(self.path)
        started_at, status = "", "running"
        for record in records:
            if record.get("type") == "session_start":
                started_at = record.get("ts", "")
            elif record.get("type") == "state":
                status = (record.get("state") or {}).get("status", status)
            elif record.get("type") == "session_end":
                status = record.get("status", status)

        return SessionInfo(
            session_id=self.session_id,
            path=self.path,
            started_at=started_at,
            status=status,
            message_count=sum(1 for r in records if r.get("type") == "message"),
            files_changed=list(self.state.files_changed),
        )


class SessionStore:
    """定位、创建、恢复会话。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.base_dir = self.root / SESSION_DIR

    # ---------------------------------------------------------- 创建与查找

    def create(self) -> Session:
        now = datetime.now()
        # 4 位随机后缀：同一秒内起两个会话也不会撞名
        session_id = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        path = self.base_dir / f"{now:%Y-%m-%d}" / f"{session_id}.jsonl"
        session = Session(self, session_id, path, SessionState(session_id=session_id))
        session.start(self.root)
        return session

    def _iter_session_files(self) -> list[Path]:
        """所有会话文件，**最近活跃的在前**。

        排序用 mtime 而不是文件名：文件名里带的是**创建**时间，而 --continue 想要的
        是「最后一次聊的那个会话」。同一秒内建的两个会话靠文件名根本分不出先后 ——
        毫秒级的创建顺序没被记录，末尾那 4 位随机码也不带顺序信息。
        mtime 是最后一次追加写入的时间，正好就是「最近活跃」。

        stem 作为次级排序键只是为了结果确定 —— 同一 mtime 时至少每次顺序一致。
        """
        if not self.base_dir.exists():
            return []
        files = list(self.base_dir.glob("*/*.jsonl"))
        return sorted(
            files, key=lambda p: (p.stat().st_mtime, p.stem), reverse=True
        )

    def find(self, session_id_or_prefix: str) -> Path:
        """按完整 id 或任意一段来查找。

        会话 id 形如 `20260919-182031-a3f2`：**前半段是时间，后半段是随机码**。
        时间那半段同一天的会话全都一样，压根没法用来区分；真正好用的是末尾那 4 位。
        所以这里前后缀都认 —— `--session a3f2` 比 `--session 20260919-182031-a3f2`
        顺手得多，也比 `--session 2026` 有用得多。

        撞多个时报错并列出候选，而不是猜一个。
        """
        needle = session_id_or_prefix.strip()
        if not needle:
            raise SessionError("会话 id 不能为空")

        matches = [
            path
            for path in self._iter_session_files()
            if path.stem == needle
            or path.stem.startswith(needle)
            or path.stem.endswith(needle)
        ]

        if not matches:
            raise SessionError(f"找不到会话: {needle}")
        if len(matches) > 1:
            candidates = "\n  ".join(p.stem for p in matches[:10])
            raise SessionError(
                f"{needle!r} 匹配到 {len(matches)} 个会话，请再写具体一点：\n  {candidates}"
            )
        return matches[0]

    def latest(self) -> Path | None:
        files = self._iter_session_files()
        return files[0] if files else None

    def list_sessions(self, limit: int = 10) -> list[SessionInfo]:
        return [self._open(path).summary() for path in self._iter_session_files()[:limit]]

    # ---------------------------------------------------------- 打开

    def _open(self, path: Path) -> Session:
        """从文件构造 Session 对象（含 root 校验），不做别的副作用。"""
        records = _read_records(path)
        start = next((r for r in records if r.get("type") == "session_start"), None)
        if start is None:
            raise SessionError(f"{path.name} 不是合法的会话文件：没有 session_start 记录")

        recorded_root = start.get("root")
        if recorded_root and Path(recorded_root).resolve() != self.root:
            # 在另一个项目根目录里续会话，会带着 A 项目的上下文却用 B 项目的沙箱，
            # 上下文和文件都对不上，所以宁可直接拒绝
            raise SessionError(
                f"这个会话是在 {recorded_root} 下建立的，当前 root 是 {self.root}。\n"
                f"请用 --root {recorded_root} 续它。"
            )

        state = SessionState(session_id=path.stem)
        for record in records:
            if record.get("type") == "state":
                state = SessionState.from_dict(record.get("state") or {})
            elif record.get("type") == "session_end":
                state.status = record.get("status", state.status)

        return Session(self, path.stem, path, state)

    def load(self, session_id_or_prefix: str) -> Session:
        return self._open(self.find(session_id_or_prefix))

    def load_latest(self) -> Session:
        path = self.latest()
        if path is None:
            raise SessionError("还没有任何会话记录，无法 --continue")
        return self._open(path)
