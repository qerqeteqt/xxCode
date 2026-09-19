"""Main Agent —— 系统核心控制器。

它是「装配者」而不是「思考者」：把 system prompt、用户问题、工具 schema 组装成
一次调用的输入，然后交给 ReAct Loop 去跑。

Phase 2 起它持有 ToolRegistry：schema 从 registry 取，执行也从 registry 走。
注意它**仍然不认识任何具体工具** —— Read 还是 Bash 对它来说都是「registry 里的
一个名字」，和 ReAct Loop 保持同一个抽象层级。
"""

from app.agent.react_loop import ExecuteTool, run_react_loop
from app.llm.client import LLMClient
from app.tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """你是一个 Code Agent，可以读写代码文件、执行命令、搜索代码库。

工作方式：
- 先用 Glob / Grep 定位，再用 Read 确认原文，最后才动手改
- 改局部内容用 Edit，不要用 Write 整体覆盖
- 所有路径都相对项目根目录，不要试图访问项目外的文件
- 能直接回答的问题就直接回答，不要为了用工具而用工具"""


async def _no_tools_available(tool_call: dict) -> str:
    """没注册任何工具时的兜底执行器（Phase 1 的行为）。"""
    name = tool_call.get("function", {}).get("name")
    return f"当前没有注册任何工具（被请求的工具: {name}）"


class MainAgent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_steps = max_steps

        # registry 是可选依赖：不传就是 Phase 1 那种「没有工具」的状态。
        # 两种能力（schema 给模型看、execute 真执行）都从同一个对象取，
        # 不会出现「告诉模型有 Read 工具，但执行时找不到」这种不一致。
        self._tools = registry.schemas() if registry is not None else None
        self._execute_tool: ExecuteTool = (
            registry.execute if registry is not None else _no_tools_available
        )

    async def run(self, question: str, messages: list[dict] | None = None) -> str:
        """回答一个问题。

        messages 传 None 就是开一段新对话（自动加 system prompt）；
        传进来一个已有的 messages 就是接着上一轮聊 —— Phase 4 的 Session
        恢复会话走的就是这条路径。
        """
        if messages is None:
            messages = [{"role": "system", "content": self._system_prompt}]

        messages.append({"role": "user", "content": question})

        return await run_react_loop(
            messages=messages,
            llm=self._llm,
            execute_tool=self._execute_tool,
            tools=self._tools,
            max_steps=self._max_steps,
        )
