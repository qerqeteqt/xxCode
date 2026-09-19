"""入口：python main.py "你的问题"

--root 决定 agent 能看哪个项目，同时也就是路径沙箱的边界。
默认取当前工作目录；因为「沙箱根 = 你此刻所在目录」这件事不够显然，
启动时会把它打出来。
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.agent.main_agent import MainAgent
from app.llm.client import LLMClient
from app.tools import build_default_registry
from config.settings import get_settings

logger = logging.getLogger("xxcode")


async def _run(question: str, root: Path) -> str:
    settings = get_settings()

    # LLMClient 实现了 async 上下文管理器，退出时自动关掉连接池
    async with LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    ) as llm:
        # registry 要在 llm 之后建：SubAgentTool 需要 llm 才能驱动子循环
        registry = build_default_registry(root, llm=llm)

        logger.info("项目 root: %s", root)
        logger.info("已注册工具: %s", ", ".join(t.name for t in registry.list_tools()))

        agent = MainAgent(llm=llm, registry=registry, max_steps=settings.max_steps)
        return await agent.run(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="xxCode —— 自研 Code Agent Runtime")
    parser.add_argument("question", nargs="?", help="要交给 Agent 的问题")
    parser.add_argument(
        "--root",
        default=".",
        help="agent 可访问的项目根目录，同时是路径沙箱边界（默认当前目录）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    question = args.question or "用一句话解释 ReAct 是什么"
    answer = asyncio.run(_run(question, Path(args.root).resolve()))
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
