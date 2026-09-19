"""记忆层。

短期记忆 = Session JSONL（Phase 4）；长期记忆 = Markdown + 索引（Phase 5）。
AutoDream（Phase 6）会在这两者之间建立联系：读 Session，提炼成 Markdown。
"""

from app.memory.memory_manager import (
    INDEX_LINK_PREFIX,
    Memory,
    MemoryManager,
    MemoryStoreError,
)
from app.memory.session_store import (
    Session,
    SessionError,
    SessionInfo,
    SessionState,
    SessionStore,
)

__all__ = [
    "INDEX_LINK_PREFIX",
    "Memory",
    "MemoryManager",
    "MemoryStoreError",
    "Session",
    "SessionError",
    "SessionInfo",
    "SessionState",
    "SessionStore",
]
