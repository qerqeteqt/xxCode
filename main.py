"""入口。

    python main.py "你的问题"                     新会话
    python main.py --continue "接着问"            续最近活跃的会话
    python main.py --session a3f2 "接着问"        续指定会话（前后缀都认）
    python main.py --list-sessions                列出最近的会话
    python main.py --consolidate                  强制整理一次长期记忆后退出

--root 决定 agent 能看哪个项目，同时也就是路径沙箱的边界。默认取当前工作目录；
因为「沙箱根 = 你此刻所在目录」这件事不够显然，启动时会把它打出来。

会话记录落在 <root>/.agent/sessions/<日期>/<会话id>.jsonl（append-only）。
长期记忆落在 <root>/.agent/memory/，由 AutoDream 整理。

**顺序是先给你答案，再跑记忆整理。** 整理可能要几十秒，不该挡在答案前面。
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.agent.main_agent import MainAgent
from app.agent.react_loop import MaxIterationError
from app.context.compactor import ContextCompactor
from app.llm.client import LLMClient, human_tokens
from app.memory import MemoryManager
from app.memory.session_store import SessionError, SessionStore
from app.scheduler import Scheduler
from app.tools import (
    Decision,
    PermissionGate,
    PermissionRequest,
    build_default_registry,
)
from config.settings import get_settings

logger = logging.getLogger("xxcode")


async def _ask_permission(request: PermissionRequest) -> Decision:
    """终端上问一句。

    `input()` 是阻塞的，直接调会把事件循环连同日志一起卡住，所以丢进线程跑。
    """
    options = (
        "  (y) 允许这一次\n"
        f"  (a) 本会话内允许所有「{request.risk}」类操作\n"
        "  (n) 拒绝 —— 会把原因回灌给模型，让它换个做法\n"
    )
    prompt = f"\n[权限] Agent 想{request.summary}\n{options}选择 [n]: "
    try:
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
    except EOFError:
        # 标准输入不是终端（管道、重定向）时 input() 抛这个。
        # 没人可问 = 拒绝，和闸门自己的默认保持一致
        print("\n[权限] 读不到输入，按拒绝处理。要用脚本跑请加 --yes", file=sys.stderr)
        return Decision.DENY

    if answer.startswith("y"):
        return Decision.ALLOW_ONCE
    if answer.startswith("a"):
        return Decision.ALLOW_SESSION
    return Decision.DENY


async def _always_allow(request: PermissionRequest) -> Decision:  # noqa: ARG001
    """`--yes` 用的确认器：一律放行。"""
    return Decision.ALLOW_SESSION


async def _run_session(
    question: str,
    root: Path,
    session_ref: str | None,
    resume: bool,
    assume_yes: bool = False,
) -> str:
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

        # 权限闸门。会话内的放行记录只活在内存里 —— 一次手滑选了「永久允许」
        # 就跟着项目走了，而你多半不记得自己什么时候做的决定
        gate = PermissionGate.for_project(
            root, confirmer=_always_allow if assume_yes else _ask_permission
        )

        # registry 要在 llm 之后建：SubAgentTool 需要 llm 才能驱动子循环。
        # on_file_changed 让文件真被改动时立刻落进 State，而不是等任务结束再补。
        registry = build_default_registry(
            root, llm=llm, on_file_changed=session.add_changed_file, gate=gate
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

        # 压缩记录要写进会话文件（on_compaction），恢复时才能把内存和磁盘对上。
        # 只给 Main Agent 配：SubAgent 和 AutoDream 的步数是有上界的，Context 长不到哪去
        compactor = ContextCompactor(
            llm,
            threshold_tokens=settings.compact_threshold_tokens,
            on_compaction=session.record_compaction,
        )

        agent = MainAgent(
            llm=llm,
            registry=registry,
            session=session,
            memory=memory,
            compactor=compactor,
            max_steps=settings.max_steps,
        )

        def _record_usage() -> None:
            # 不管成没成，token 都已经花出去了，如实记下 —— 失败的那次尤其要看
            session.record_usage(
                llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.calls
            )

        try:
            answer = await agent.run(question)
        except MaxIterationError as e:
            # 达到步数上限：过程已经逐条落盘了（on_message 钩子），
            # 所以这里能告诉用户「去哪接着聊」，而不是让他从头再来一遍
            _record_usage()
            session.finish("failed")
            raise MaxIterationError(
                e.max_steps,
                note=(
                    f"。本次过程已保存在会话 {session.session_id}，"
                    f'用 python main.py --continue "接着上次" 可以继续'
                ),
                partial=e.partial,
            ) from None
        except BaseException:
            # 包括 Ctrl+C 和 LLM 报错。会话文件里要留下「这次没跑完」的痕迹，
            # 否则下次 --continue 会以为上次是正常结束的
            _record_usage()
            session.finish("failed")
            raise

        _record_usage()
        logger.info("本次会话用量: %s", llm.usage)
        session.finish("finished")
        return answer


async def _run_consolidation(root: Path, *, force: bool) -> None:
    """跑一次（或检查一次）记忆整理。和会话完全独立的两件事。"""
    settings = get_settings()
    scheduler = Scheduler(
        root,
        min_hours=settings.consolidate_min_hours,
        min_sessions=settings.consolidate_min_sessions,
    )

    async with LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    ) as llm:
        result = await scheduler.consolidate(llm, force=force)

    print(f"\n[记忆整理] {result.summary}")
    if result.changed:
        print(f"[记忆整理] 改动: {', '.join(result.changed)}")


def _print_sessions(root: Path, limit: int) -> None:
    infos = SessionStore(root).list_sessions(limit)
    if not infos:
        print(f"{root} 下还没有任何会话记录。")
        return
    for info in infos:
        tokens = f"  {human_tokens(info.total_tokens)} tokens" if info.total_tokens else ""
        changed = f"  改动 {len(info.files_changed)} 个文件" if info.files_changed else ""
        print(
            f"{info.started_at}  {info.session_id}  "
            f"[{info.status}]  {info.message_count} 条消息{tokens}{changed}"
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
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="强制整理一次长期记忆后退出（跳过触发条件判断）",
    )
    parser.add_argument(
        "--no-consolidate",
        dest="no_consolidate",
        action="store_true",
        help="本次会话结束后不检查记忆整理",
    )
    parser.add_argument(
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="跳过权限确认，全部放行（给脚本用。交互使用时别加，那就等于没有权限系统）",
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

    if args.consolidate:
        try:
            asyncio.run(_run_consolidation(root, force=True))
        except SessionError as e:
            print(f"错误: {e}", file=sys.stderr)
            raise SystemExit(1) from None
        return

    question = args.question or "用一句话解释 ReAct 是什么"

    try:
        answer = asyncio.run(
            _run_session(question, root, args.session, args.resume, args.assume_yes)
        )
    except SessionError as e:
        # 会话相关的失败（找不到、前缀不唯一、root 对不上）是用户能自己修的问题，
        # 不该甩一段 traceback 出来
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    except MaxIterationError as e:
        # 同理：跑不完是个正常结果，不是程序崩了
        print(f"\n未能完成：{e}", file=sys.stderr)
        if e.partial:
            # 跑到上限通常不是一无所获，而是「查了一大堆没来得及收尾」。
            # 把它说出来，用户才知道它卡在哪
            print(f"\n它中断前的最后进展：\n{e.partial}", file=sys.stderr)
        raise SystemExit(1) from None

    # 先把答案给你看，再跑整理 —— 整理可能几十秒，不该挡在答案前面
    print(f"\n{answer}")

    if not args.no_consolidate:
        try:
            asyncio.run(_run_consolidation(root, force=False))
        except Exception as e:  # noqa: BLE001 —— 见下面的注释
            # 这里刻意宽兜：整理是「顺带做的事」，失败了绝不能影响用户已经拿到的答案。
            # 不捕获 BaseException，所以 Ctrl+C 仍然能正常打断。
            logger.warning("记忆整理出错（不影响本次会话）: %s", e)


if __name__ == "__main__":
    main()
