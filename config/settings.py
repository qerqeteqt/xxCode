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

    # 字段名 llm_api_key 会自动匹配环境变量 LLM_API_KEY（大小写不敏感）
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout: float = 60.0

    # ReAct Loop 的最大循环次数，超过就抛 MaxIterationError
    max_steps: int = 10

    # 记忆整理（AutoDream）的触发条件。两个是**并且**关系 —— 都满足才跑。
    # 刻意保守：整理要花真金白银，宁可少跑几次。想立刻跑一次用 --consolidate。
    consolidate_min_hours: float = 24.0
    consolidate_min_sessions: int = 5

    # 上下文压缩的触发阈值（上一次调用真实的 prompt token 数）。
    # deepseek-chat 的上限是 64k，40k 留出余量给回复和后续增长。
    compact_threshold_tokens: int = 40_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
