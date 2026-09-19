"""入口。

    python main.py "你的问题"                     新会话
    python main.py --continue "接着问"            续最近一次会话
    python main.py --session a3f2 "接着问"        续指定会话（支持唯一前缀）
    python main.py --list-sessions                列出最近的会话

--root 决定 agent 能看哪个项目，同时也就是路径沙箱的边界。默认取当前工作目录；
因为「沙箱根 = 你此刻所在目录」这件事不够显然，启动时会把它打出来。

会话记录落在 <root>/.agent/sessions/<日期>/<会话id>.jsonl（append-only）。
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.agent.main_agent import MainAgent
from app.llm.client import LLMClient
from app.memory import MemoryManager
from app.memory.session_store import SessionError, SessionStore
from app.tools import build_default_registry
from config.settings import get_settings

logger = logging.getLogger("xxcode")


async def _run(question: str, root: Path, session_ref: str | None, resume: bool) -> str:
    settings = get_settings()
    store = SessionStore(root)

    # LLMClient 实现了 async 上下文管理器，退出时自动关掉连接池
    async with LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    ) as llm:
        if resume:
            session = store.load_latest()
        elif session_ref:
            session = store.load(session_ref)
        else:
            session = store.create()

        # registry 要在 llm 之后建：SubAgentTool 需要 llm 才能驱动子循环。
        # on_file_changed 让文件真被改动时立刻落进 State，而不是等任务结束再补。
        registry = build_default_registry(
            root, llm=llm, on_file_changed=session.add_changed_file
        )

        # 启动时重建一次索引：索引是派生数据，这样永远和目录里的文件一致，
        # 也顺带覆盖了「你手工丢了个新 md 进去」这种情况
        memory = MemoryManager(root)
        memory.sync_index()

        logger.info("项目 root: %s", root)
        logger.info("会话: %s", session.session_id)
        logger.info("已注册工具: %s", ", ".join(t.name for t in registry.list_tools()))
        logger.info("长期记忆: %d 条", len(memory.list_memories()))
        if session.state.files_changed:
            logger.info("本会话此前改动: %s", ", ".join(session.state.files_changed))

        agent = MainAgent(
            llm=llm,
            registry=registry,
            session=session,
            memory=memory,
            max_steps=settings.max_steps,
        )

        try:
            answer = await agent.run(question)
        except BaseException:
            # 包括 Ctrl+C 和 LLM 报错。会话文件里要留下「这次没跑完」的痕迹，
            # 否则下次 --continue 会以为上次是正常结束的
            session.finish("failed")
            raise

        session.finish("finished")
        return answer


def _print_sessions(root: Path, limit: int) -> None:
    infos = SessionStore(root).list_sessions(limit)
    if not infos:
        print(f"{root} 下还没有任何会话记录。")
        return
    for info in infos:
        changed = f"  改动 {len(info.files_changed)} 个文件" if info.files_changed else ""
        print(
            f"{info.started_at}  {info.session_id}  "
            f"[{info.status}]  {info.message_count} 条消息{changed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="xxCode —— 自研 Code Agent Runtime")
    parser.add_argument("question", nargs="?", help="要交给 Agent 的问题")
    parser.add_argument(
        "--root",
        default=".",
        help="agent 可访问的项目根目录，同时是路径沙箱边界（默认当前目录）",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="续指定会话，支持唯一前缀，例如 --session a3f2",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="续最近一次会话",
    )
    parser.add_argument(
        "--list-sessions",
        dest="list_sessions",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="N",
        help="列出最近 N 个会话后退出（默认 10）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    root = Path(args.root).resolve()

    if args.list_sessions is not None:
        _print_sessions(root, args.list_sessions)
        return

    question = args.question or "用一句话解释 ReAct 是什么"

    try:
        answer = asyncio.run(_run(question, root, args.session, args.resume))
    except SessionError as e:
        # 会话相关的失败（找不到、前缀不唯一、root 对不上）是用户能自己修的问题，
        # 不该甩一段 traceback 出来
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"\n{answer}")


if __name__ == "__main__":
    main()
