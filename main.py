"""入口：python main.py "你的问题"

Phase 1 只做一件事：把问题交给 Main Agent，打印回答。
没有工具、没有 Session 持久化、没有 Memory —— 那些分别属于 Phase 2 / 4 / 5。
"""

import argparse
import asyncio
import logging

from app.agent.main_agent import MainAgent
from app.llm.client import LLMClient
from config.settings import get_settings


async def _run(question: str) -> str:
    settings = get_settings()

    # LLMClient 实现了 async 上下文管理器，退出时自动关掉连接池。
    async with LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    ) as llm:
        agent = MainAgent(llm=llm, max_steps=settings.max_steps)
        return await agent.run(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="xxCode —— 自研 Code Agent Runtime")
    parser.add_argument("question", nargs="?", help="要交给 Agent 的问题")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    question = args.question or "用一句话解释 ReAct 是什么"
    answer = asyncio.run(_run(question))
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
