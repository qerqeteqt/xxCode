"""FastAPI 服务 —— 把 Runtime 包成网页。

## 这一层只做三件事

    1. 把 Runtime 的事件翻译成 SSE
    2. 把权限确认变成「等浏览器回一个 HTTP 请求」
    3. 把 `.env` 暴露成可读写的设置

**Runtime 本身一行没改。** 事件、流式、权限确认这三个口子都是早就留好的
回调/注入点 —— 这也是为什么加一个 Web 前端不需要动任何核心代码。
如果有人问「依赖注入有什么用」，这个文件就是答案。

## 一个刻意的取舍

每个浏览器会话一套 `LLMClient + registry + gate`，不共用。因为：

    LLMClient.usage   是按会话统计的，共用会把账单混在一起
    PermissionGate    的「本会话放行」必须只影响这个会话

代价是每个会话一个连接池 —— 对本地单用户工具完全无所谓。

## 也是刻意的：没有鉴权

这是「你自己在本机开一个窗口用」的形态。要给别人用得先做容器隔离、鉴权、限流，
那是另一件事，现在做等于猜。
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from app.agent.main_agent import MainAgent
from app.agent.react_loop import MaxIterationError
from app.context.compactor import ContextCompactor
from app.events import Event
from app.llm.client import LLMClient
from app.memory import MemoryManager
from app.memory.extractor import MemoryExtractor
from app.memory.session_store import Session, SessionError, SessionStore
from app.scheduler import Scheduler
from app.tools import (
    Decision,
    PermissionGate,
    PermissionRequest,
    build_default_registry,
)
from config.settings import PROJECT_ROOT, Settings, get_settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# 设置面板里允许改的项。**白名单**，不是黑名单 —— 加一项要显式来这里加。
# .env 里还有 API Key 之类的东西，绝不能让它被网页随便写
EDITABLE_KEYS = (
    "EXTRACT_MEMORY",
    "LLM_MODEL",
    "MAX_STEPS",
    "COMPACT_THRESHOLD_TOKENS",
    "CONSOLIDATE_MIN_HOURS",
    "CONSOLIDATE_MIN_SESSIONS",
)

# 待确认的权限请求：request_id -> 等在那儿的 future。
# 放模块级而不是挂在会话上，是因为浏览器回请求时只带得回 request_id
PENDING: dict[str, asyncio.Future] = {}


# ================================================================ .env 读写


def update_env(env_path: Path, values: dict[str, str]) -> None:
    """把值写回 `.env`，**保留原有注释和顺序**。

    整体重写会丢掉所有注释 —— 而注释正是「每个配置项是干什么的」的唯一说明。
    只替换同名的那一行，找不到才追加。
    """
    lines = (
        env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    )
    remaining = dict(values)
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)

    out.extend(f"{key}={value}" for key, value in remaining.items())
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def apply_settings(env_path: Path, values: dict[str, str]) -> None:
    """写回设置并验证。验证不过就**原样回滚**。

    为什么要回滚：.env 写坏了，下次启动 `Settings()` 会直接抛错 ——
    服务起不来，而用户改的只是「max_steps」这种小事。宁可拒绝这次修改。
    """
    backup = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    update_env(env_path, values)
    try:
        # 验证的必须是**刚写进去的那个文件**（_env_file 指过去），
        # 否则它一直在验证生产用的 .env，测也测不了，改错了也拦不住
        Settings(_env_file=env_path)
    except ValidationError as e:
        env_path.write_text(backup, encoding="utf-8")
        raise HTTPException(400, f"配置不合法，已回滚：{e}") from e

    get_settings.cache_clear()  # 让新会话读到新值（进行中的会话不受影响）


# ================================================================ 会话运行时


class LiveSession:
    """一次浏览器对话在服务端的全部状态。"""

    def __init__(self, root: Path, session: Session, settings: Settings) -> None:
        self.root = root
        self.session = session
        self.busy = False
        self._queue: asyncio.Queue | None = None

        self.llm = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )
        self.gate = PermissionGate.for_project(root, confirmer=self._confirm)
        self.registry = build_default_registry(
            root,
            llm=self.llm,
            on_file_changed=session.add_changed_file,
            gate=self.gate,
            on_event=self._on_event,
        )
        self.memory = MemoryManager(root)
        self.memory.sync_index()
        self.compactor = ContextCompactor(
            self.llm,
            threshold_tokens=settings.compact_threshold_tokens,
            on_compaction=session.record_compaction,
        )
        self.agent = MainAgent(
            llm=self.llm,
            registry=self.registry,
            session=session,
            memory=self.memory,
            compactor=self.compactor,
            on_delta=self._on_delta,
            on_event=self._on_event,
            max_steps=settings.max_steps,
        )

    # ---------------------------------------------------------- 事件

    def attach(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def detach(self) -> None:
        self._queue = None

    def _push(self, event: Event) -> None:
        if self._queue is not None:
            self._queue.put_nowait(event)

    def _on_event(self, event: Event) -> None:
        self._push(event)

    def _on_delta(self, text: str) -> None:
        self._push(Event("delta", {"text": text}))

    # ---------------------------------------------------------- 权限

    async def _confirm(self, request: PermissionRequest) -> Decision:
        """权限闸门问到这里时，把问题推给浏览器，然后**等**。

        `confirmer` 从一开始就是个 async 回调（Phase 7c 的设计），
        这里正好派上用场：await 一个 future，等浏览器 POST 回决策。
        """
        request_id = uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        PENDING[request_id] = future
        self._push(
            Event(
                "permission",
                {
                    "request_id": request_id,
                    "summary": request.summary,
                    "risk": request.risk,
                    "tool": request.tool_name,
                    "subject": request.subject,
                },
            )
        )
        try:
            return await future
        finally:
            PENDING.pop(request_id, None)

    async def close(self) -> None:
        await self.llm.aclose()


# ================================================================ 应用


class ChatRequest(BaseModel):
    session_id: str
    question: str


class DecisionRequest(BaseModel):
    decision: str  # allow_once | allow_session | deny


class ConfigRequest(BaseModel):
    values: dict[str, Any]


def create_app(root: str | Path) -> FastAPI:
    project_root = Path(root).resolve()
    store = SessionStore(project_root)
    live_sessions: dict[str, LiveSession] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("root: %s", project_root)
        yield
        for live in live_sessions.values():
            await live.close()

    app = FastAPI(title="xxCode", lifespan=lifespan)

    def _live(session_id: str) -> LiveSession:
        """拿到（或按需创建）一个会话的运行时。

        恢复一个已有会话时走的是 `store.load` —— 和 CLI 的 `--continue` 同一条路，
        所以那边成立的事情（root 校验、压缩记录重放、token 统计）这边自动成立。
        """
        if session_id in live_sessions:
            return live_sessions[session_id]
        try:
            session = store.load(session_id)
        except SessionError as e:
            raise HTTPException(404, str(e)) from e
        live = LiveSession(project_root, session, get_settings())
        live_sessions[session_id] = live
        return live

    # ---------------------------------------------------------- 页面

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ---------------------------------------------------------- 会话

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        """只列**说过话**的会话。

        点了「新会话」但没提问的会留下一条空记录 —— 那是占位，不是历史。
        全列出来的话列表很快就被这些占位淹没了。
        """
        return [
            {
                "session_id": info.session_id,
                "started_at": info.started_at,
                "status": info.status,
                "message_count": info.message_count,
                "total_tokens": info.total_tokens,
                "files_changed": info.files_changed,
                "title": info.title,
            }
            for info in store.list_sessions(60)
            if info.message_count > 0
        ]

    @app.post("/api/sessions")
    async def new_session() -> dict:
        session = store.create()
        logger.info("新会话 %s", session.session_id)
        return {"session_id": session.session_id}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        live = live_sessions.get(session_id)
        if live is not None and live.busy:
            # 正在跑的时候删掉文件，等于把 append-only 的日志从底下抽走 ——
            # 后面每次写入都会失败，模型也拿不到结果。让它跑完再删
            raise HTTPException(409, "这个会话正在跑，等它结束再删")

        if live is not None:
            await live.close()
            live_sessions.pop(session_id, None)

        try:
            store.delete(session_id)
        except SessionError as e:
            raise HTTPException(404, str(e)) from e

        logger.info("已删除会话 %s", session_id)
        return {"ok": True}

    @app.get("/api/sessions/{session_id}/messages")
    async def session_messages(session_id: str) -> list[dict]:
        """恢复会话时把历史消息还给前端。

        system 不在里面（它每次现拼），前端也只显示 user / assistant / tool。
        """
        try:
            session = store.load(session_id)
        except SessionError as e:
            raise HTTPException(404, str(e)) from e

        out: list[dict] = []
        for message in session.load_messages():
            role = message.get("role")
            if role == "user":
                out.append({"role": "user", "text": message.get("content") or ""})
            elif role == "assistant" and message.get("content"):
                out.append({"role": "assistant", "text": message["content"]})
        return out

    # live 在 chat() 里才存在，必须当参数传进来 —— 嵌套函数看不到调用方的局部变量
    async def _extract_memory(queue: asyncio.Queue, live: LiveSession) -> None:
        """每轮对话后跑一次：看看这轮有没有值得长期记住的东西。

        独立成一步而不是塞进 Main Agent 的工具里，好处是主对话的 Context
        完全不受影响 —— 提取失败的、判断「不值得记」的，统统不会污染
        用户正在看的那段对话。
        """
        if not get_settings().extract_memory:
            return

        try:
            # 用会话自己的 client：提取的开销也算进这次的账单，不然它会凭空消失
            result = await MemoryExtractor(project_root, live.llm).run(
                live.session.load_messages()
            )
        except Exception as e:  # noqa: BLE001 —— 提取是「顺带做的事」
            logger.exception("记忆提取出错")
            queue.put_nowait(
                Event(
                    "memory",
                    {"phase": "extract", "status": "failed", "message": str(e)},
                )
            )
            return

        if result is None:
            return  # 这一轮没什么可看的
        queue.put_nowait(
            Event(
                "memory",
                {
                    "phase": "extract",
                    "status": "done" if result.completed else "failed",
                    "summary": result.summary,
                    "changed": result.changed,
                },
            )
        )

    async def _consolidate_if_due(
        queue: asyncio.Queue, live: LiveSession
    ) -> None:
        """该整理就整理，把过程也推给浏览器。

        Scheduler 判断该不该跑，AutoDream 干活 —— 和 CLI 走的是同一套，
        只是把结果从「打日志」换成了「发事件」。整理失败绝不能影响
        已经发出的答案，所以整段包在 try 里。
        """
        settings = get_settings()
        scheduler = Scheduler(
            project_root,
            min_hours=settings.consolidate_min_hours,
            min_sessions=settings.consolidate_min_sessions,
        )
        decision = scheduler.check()
        if not decision.should_run:
            logger.info("不整理记忆：%s", decision.reason)
            return

        queue.put_nowait(
            Event("memory", {"phase": "consolidate", "status": "running", "reason": decision.reason})
        )
        try:
            # 用会话自己的 client：AutoDream 的开销也算进这次的账单里，
            # 不然它的花费会凭空消失
            result = await scheduler.consolidate(live.llm, force=False)
        except Exception as e:  # noqa: BLE001 —— 整理是「顺带做的事」
            logger.exception("记忆整理出错")
            queue.put_nowait(
                Event("memory", {"phase": "consolidate", "status": "failed", "message": str(e)})
            )
            return

        queue.put_nowait(
            Event(
                "memory",
                {
                    "phase": "consolidate",
                    "status": "done",
                    "summary": result.summary,
                    "changed": result.changed,
                    "sessions_used": result.sessions_used,
                },
            )
        )

    # ---------------------------------------------------------- 对话

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> StreamingResponse:
        live = _live(payload.session_id)
        if live.busy:
            raise HTTPException(409, "这个会话正在跑，等它结束或开个新会话")

        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        live.busy = True
        live.attach(queue)

        async def run() -> None:
            try:
                answer = await live.agent.run(payload.question)
                queue.put_nowait(
                    Event(
                        "done",
                        {
                            "answer": answer,
                            "total_tokens": live.llm.usage.total_tokens,
                            "calls": live.llm.usage.calls,
                        },
                    )
                )
                # 提取和整理都放在 done 之后：**先让用户拿到答案**。
                # 两个加起来可能几十秒，挡在答案前面没人受得了
                await _extract_memory(queue, live)
                await _consolidate_if_due(queue, live)
            except MaxIterationError as e:
                queue.put_nowait(
                    Event("error", {"message": str(e), "partial": e.partial})
                )
            except Exception as e:  # noqa: BLE001 —— 出错要送回浏览器，不能让流干挂着
                logger.exception("会话执行出错")
                queue.put_nowait(
                    Event("error", {"message": f"{type(e).__name__}: {e}"})
                )
            finally:
                # 会话状态落盘：和 CLI 一样，失败也要留下痕迹
                live.session.finish("finished")
                live.busy = False
                live.detach()
                queue.put_nowait(None)  # 结束哨兵

        task = asyncio.create_task(run())

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield (
                        f"event: {event.type}\n"
                        f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                    )
            finally:
                # 浏览器关掉页面时也会走到这儿。任务让它自己跑完 ——
                # 半途掐掉会留下一个「正在改文件却没人知道」的烂摊子
                if not task.done():
                    logger.info("客户端断开，任务继续在后台跑完")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/permission/{request_id}")
    async def decide(request_id: str, payload: DecisionRequest) -> dict:
        future = PENDING.get(request_id)
        if future is None or future.done():
            raise HTTPException(404, "这个授权请求已经失效了")
        try:
            future.set_result(Decision(payload.decision))
        except ValueError as e:
            raise HTTPException(400, f"未知的决定: {payload.decision}") from e
        return {"ok": True}

    # ---------------------------------------------------------- 设置

    @app.get("/api/config")
    async def read_config() -> dict:
        settings = get_settings()
        env_path = PROJECT_ROOT / ".env"
        return {
            "editable": {
                "EXTRACT_MEMORY": settings.extract_memory,
                "LLM_MODEL": settings.llm_model,
                "MAX_STEPS": settings.max_steps,
                "COMPACT_THRESHOLD_TOKENS": settings.compact_threshold_tokens,
                "CONSOLIDATE_MIN_HOURS": settings.consolidate_min_hours,
                "CONSOLIDATE_MIN_SESSIONS": settings.consolidate_min_sessions,
            },
            "readonly": {
                "root": str(project_root),
                "llm_base_url": settings.llm_base_url,
                "env_file": str(env_path),
                "memory_count": len(MemoryManager(project_root).list_memories()),
            },
        }

    @app.post("/api/config")
    async def write_config(payload: ConfigRequest) -> dict:
        unknown = set(payload.values) - set(EDITABLE_KEYS)
        if unknown:
            # 白名单之外的键一律拒绝 —— .env 里还有 API Key
            raise HTTPException(400, f"不可修改的配置项: {', '.join(sorted(unknown))}")

        env_path = PROJECT_ROOT / ".env"
        apply_settings(env_path, {k: str(v) for k, v in payload.values.items()})
        return {"ok": True, "note": "已写入 .env。新会话生效，进行中的会话不受影响。"}

    return app
