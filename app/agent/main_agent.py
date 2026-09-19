"""Main Agent —— 系统核心控制器。

它是「装配者」而不是「思考者」：把 system prompt、用户问题、工具 schema 组装成
一次调用的输入，然后交给 ReAct Loop 去跑。

Phase 1 它还很简单：准备 prompt + 驱动循环。
Phase 2 起它多两件事：把 ToolRegistry 的 schema 传给 loop，把 registry.execute
作为 execute_tool 注入进去（就是下面那个占位函数的位置）。

注意 MainAgent 自己不持有 tools 的具体实现，只持有「能执行 tool_call 的函数」，
和 Loop 保持同一个抽象层级。
"""

from app.agent.react_loop import ExecuteTool, run_react_loop
from app.llm.client import LLMClient

DEFAULT_SYSTEM_PROMPT = (
    "你是一个 Code Agent，可以读写代码文件、执行命令、搜索代码库。"
    "能直接回答的问题就直接回答；需要查看或修改代码时，调用相应的工具。"
)


async def _no_tools_available(tool_call: dict) -> str:
    """Phase 1 的占位实现：还没有任何工具被注册。

    它存在的意义不是「能用」，而是把注入点显式暴露出来 ——
    Phase 2 换成 ToolRegistry.execute 时，改的就是这一行的位置。
    """
    name = tool_call.get("function", {}).get("name")
    return f"当前没有注册任何工具（被请求的工具: {name}）"


class MainAgent:
    def __init__(
        self,
        llm: LLMClient,
        execute_tool: ExecuteTool | None = None,
        tools: list[dict] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> None:
        self._llm = llm
        self._execute_tool = execute_tool or _no_tools_available
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_steps = max_steps

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
