"""Scheduler —— 判断该不该整理记忆，该就启动 AutoDream。

## 关于「后台」的实话

文档写的是「后台 Scheduler，异步启动 AutoDream」。但在 CLI 里，`python main.py "问题"`
跑完就退出，**后台 asyncio 任务会跟着进程一起死**。所以这一版取的「后台」是它的
另一层意思，也是文档第十四节明确要求的那层：

    AutoDream 不作为 Main Agent 的普通 Tool

也就是说它**独立于 Main Agent 的循环、不注册进 ToolRegistry**、有自己的 Context
和自己的生命周期。至于「另一个进程」——将来上 Web 之后服务进程不会退出，
把 `check()` 挂到事件循环上定期调用就变成真后台了，**这个文件不用改**。
所以「判断该不该跑」和「跑」是分开的两个方法，不是一回事。

## 状态记在哪

`.agent/consolidation.json`。不能放 `.agent/memory/` —— 那目录是给人和 Agent 看的
记忆，塞个状态文件进去会被索引扫到，变成一条莫名其妙的「记忆」。

「上次之后新增了几个会话」不用额外记账：数 `.agent/sessions/**/*.jsonl` 里
mtime 晚于上次运行时间的有几个就行。**文件系统本身就是账本。**
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from app.llm.client import LLMClient
from app.memory.auto_dream import AutoDream, DreamResult
from app.memory.session_store import SessionStore

logger = logging.getLogger(__name__)

STATE_PATH = ".agent/consolidation.json"
BACKUP_DIR = ".agent/memory-backups"
KEEP_BACKUPS = 5


@dataclass(frozen=True)
class ConsolidationDecision:
    should_run: bool
    reason: str  # 给日志用：为什么跑 / 为什么不跑


class Scheduler:
    def __init__(
        self,
        root: str | Path,
        *,
        min_hours: float = 24.0,
        min_sessions: int = 5,
    ) -> None:
        self.root = Path(root).resolve()
        self.min_hours = min_hours
        self.min_sessions = min_sessions
        self.store = SessionStore(self.root)
        self.state_path = self.root / STATE_PATH

    # ------------------------------------------------------------ 状态

    def _last_run(self) -> datetime | None:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return datetime.fromisoformat(data["last_run"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            # 状态文件坏了就当作「从没整理过」—— 大不了多整理一次，
            # 总比因为一个坏文件再也整理不了强
            logger.warning("整理状态文件读取失败，当作从未整理过: %s", e)
            return None

    def _save_last_run(self, when: datetime) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"last_run": when.isoformat(timespec="seconds")}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------ 判断

    def new_sessions(self) -> list[Path]:
        """上次整理之后新产生的会话，新的在前。

        一个会话的「产生时间」用文件 mtime —— Session 是 append-only 的，
        所以 mtime 就是它最后一次活动的时间，也正是我们想比的。
        """
        last_run = self._last_run()
        cutoff = last_run.timestamp() if last_run else 0.0
        return [
            path
            for path in self.store.session_files()
            if path.stat().st_mtime > cutoff
        ]

    def check(self) -> ConsolidationDecision:
        """判断该不该整理。两个条件是**并且**关系 —— 都满足才跑。"""
        last_run = self._last_run()
        pending = self.new_sessions()

        if last_run is None:
            if not pending:
                return ConsolidationDecision(False, "还没有任何会话记录")
            return ConsolidationDecision(
                True, f"从未整理过，现有 {len(pending)} 个会话"
            )

        elapsed = datetime.now() - last_run
        enough_time = elapsed >= timedelta(hours=self.min_hours)
        enough_sessions = len(pending) >= self.min_sessions

        detail = (
            f"距上次 {elapsed.total_seconds() / 3600:.1f} 小时"
            f"（需 ≥{self.min_hours:g}），新增 {len(pending)} 个会话"
            f"（需 ≥{self.min_sessions}）"
        )

        # 两个条件是并且：时间到了但会话不够，或会话够了但时间没到，都不跑。
        # 刻意保守 —— 整理要花真金白银，宁可少跑几次。
        if enough_time and enough_sessions:
            return ConsolidationDecision(True, detail)
        return ConsolidationDecision(False, detail)

    # ------------------------------------------------------------ 备份

    def backup_memory(self) -> Path | None:
        """把记忆目录整份复制一份，返回备份路径。

        为什么不能省：AutoDream 手里有 DeleteMemory 和 WriteMemory，而它是个 LLM。
        要防的三件事——它的判断失误、它跑到一半被 Ctrl+C 打断、以及你事后想对比
        「它到底改了什么」。备份是唯一能同时兜住这三件事的低成本手段
        （文档第十七节：AutoDream 出错不应破坏已有 Memory）。
        """
        memory_dir = self.root / ".agent" / "memory"
        if not memory_dir.exists():
            return None

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self.root / BACKUP_DIR
        base.mkdir(parents=True, exist_ok=True)

        # 同名绝不覆盖。时间戳精确到秒，同一秒内跑两次就会撞名 ——
        # 而覆盖掉的那一份，很可能正是「AutoDream 还没搞坏之前」的干净备份，
        # 也就是你唯一想留着的那一份。
        target = base / stamp
        suffix = 2
        while target.exists():
            target = base / f"{stamp}-{suffix}"
            suffix += 1

        shutil.copytree(memory_dir, target)

        self._prune_backups()
        return target

    def _prune_backups(self) -> None:
        backups = sorted(
            (self.root / BACKUP_DIR).glob("*"), key=lambda p: p.name, reverse=True
        )
        for stale in backups[KEEP_BACKUPS:]:
            shutil.rmtree(stale, ignore_errors=True)

    # ------------------------------------------------------------ 执行

    async def consolidate(self, llm: LLMClient, *, force: bool = False) -> DreamResult:
        """跑一次整理。

        force=True 跳过条件判断（`--consolidate` 走的这条路），
        但仍然只在「确实有新会话」时才真的启动 AutoDream。
        """
        decision = self.check()
        if not force and not decision.should_run:
            logger.info("不整理记忆：%s", decision.reason)
            return DreamResult(summary=f"跳过整理：{decision.reason}", sessions_used=0)

        pending = self.new_sessions()
        if not pending:
            # 首次运行（从没整理过）时状态文件不存在，所有会话都算「新增」，
            # 所以这里通常不会命中；但 --consolidate 连跑两次就会
            return DreamResult(summary="没有新会话，跳过整理。", sessions_used=0)

        logger.info(
            "%s，共 %d 个新会话，开始整理",
            "强制整理" if force else decision.reason,
            len(pending),
        )

        backup = self.backup_memory()
        if backup is not None:
            logger.info("整理前已备份记忆到 %s", backup.relative_to(self.root).as_posix())

        dream = AutoDream(self.root, llm)
        try:
            result = await dream.run(pending)
        except BaseException:
            # 包括 Ctrl+C。状态文件不更新，这样下次还会再试一遍 ——
            # 半途而废的整理不该被当成「已经整理过」
            logger.warning("整理中断，记忆已备份在 %s", backup)
            raise

        if result.completed:
            self._save_last_run(datetime.now())
        else:
            logger.warning("整理未正常完成，不更新状态文件，下次会重试")

        logger.info(
            "整理完成：用了 %d 个会话，改动 %d 条记忆",
            result.sessions_used,
            len(result.changed),
        )
        if result.changed:
            logger.info("改动的记忆: %s", ", ".join(result.changed))
        return result
