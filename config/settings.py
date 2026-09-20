"""集中配置。

为什么用 pydantic-settings 而不是直接 os.getenv：
    1. .env 自动加载，不用手写 load_dotenv
    2. 字段有类型，LLM_TIMEOUT=abc 这类错误在启动时就报出来，而不是等发起请求才崩
    3. 缺 LLM_API_KEY 时直接抛错——宁可启动失败，也不要带着空 key 跑到线上

get_settings() 用 lru_cache 包了一层：配置只在第一次调用时读取并校验，
后续复用同一个实例。这样 import 本模块不会因为环境变量缺失而失败，
单元测试也就不需要准备 .env。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 定位到仓库根目录，而不是「当前工作目录」。
# 否则在 tests/ 里跑 pytest 时 cwd 变了，就找不到 .env 了。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有多余的键也不报错
    )

    # 字段名 llm_api_key 会自动匹配环境变量 LLM_API_KEY（大小写不敏感）。
    # min_length=1 不能省：空字符串是合法的 str，能过类型检查，
    # 但它会在发请求时变成一个 401 —— 那比「配置不合法」难查得多。
    # 挡在门口，错在启动时，一行就说清楚
    llm_api_key: str = Field(min_length=1)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout: float = 60.0

    # Tavily 联网搜索的 key。**留空就不注册 WebSearch 工具** ——
    # 又是「能力由装配决定」：没有 key 就是没有这个能力，
    # 而不是注册上去、调用时才报错
    tavily_api_key: str = ""

    # **Main Agent** 的最大循环次数，超了就抛 MaxIterationError。
    #
    # 定位是「防打转的兜底」，不是「预算」。别拿它当成本控制 ——
    # 写一个 200 行文件是 1 步，`git status` 也是 1 步，两者成本差两三个量级，
    # 共用一个计数器必然一头不够用、一头管不住。成本看 token 统计。
    #
    # 60 而不是 20：实测里「写个程序 + 跑测试 + 调试」这种正常任务要 30-40 步
    # （撞 20 的那次，光 pytest → 改 → 再 pytest 就占了 17 步）。
    #
    # ⚠️ **这一项只管 Main Agent。** 其他角色各有自己的步数预算，都是独立的
    # （按角色定，本来就不该跟 Main 一样）：
    #
    #     Explore              8    app/agent/subagent.py
    #     Plan                 8    app/agent/subagent.py
    #     General-Purpose     12    app/agent/subagent.py
    #     AutoDream           12    app/memory/auto_dream.py
    #     记忆提取器            6    app/memory/extractor.py
    #
    # 改这里不会影响它们。真要让它们也可配，再往外提成配置项。
    max_steps: int = 60

    # 记忆整理（AutoDream）的触发条件。两个是**并且**关系 —— 都满足才跑。
    # 刻意保守：整理要花真金白银，宁可少跑几次。想立刻跑一次用 --consolidate。
    consolidate_min_hours: float = 24.0
    consolidate_min_sessions: int = 5

    # 每轮对话后自动提取长期记忆。默认开 —— 关了的话「记住 X」就只能等
    # AutoDream 哪天攒够条件去翻会话流水。代价是每轮多一次 LLM 调用
    extract_memory: bool = True

    # 上下文压缩的触发阈值（上一次调用真实的 prompt token 数）。
    # deepseek-chat 的上限是 64k，40k 留出余量给回复和后续增长。
    compact_threshold_tokens: int = 40_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
