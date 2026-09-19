"""Main Agent —— 系统核心控制器。

它是「装配者」而不是「思考者」：把 system prompt、用户问题、工具 schema 组装成
一次调用的输入，然后交给 ReAct Loop 去跑。

Phase 2 起它持有 ToolRegistry：schema 从 registry 取，执行也从 registry 走。
注意它**仍然不认识任何具体工具** —— Read 还是 Bash 对它来说都是「registry 里的
一个名字」，和 ReAct Loop 保持同一个抽象层级。
"""

from typing import TYPE_CHECKING

from app.events import EventHook

from app.agent.react_loop import ExecuteTool, MessageHook, run_react_loop
from app.llm.client import LLMClient
from app.memory.memory_manager import INDEX_LINK_PREFIX, MemoryManager
from app.memory.session_store import Session
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.context.compactor import ContextCompactor
    from app.llm.client import DeltaHook

DEFAULT_SYSTEM_PROMPT = """你是一个 Code Agent，可以读写代码文件、执行命令、搜索代码库。

你是**调度者**，不是唯一的干活的人。怎么选，看一条标准：

    要回答这个问题，需要打开 3 个以上的文件吗？

    需要   → 派 SubAgent。查代码在哪、模块怎么组织的 → Explore；
             要改代码并反复验证 → General-Purpose
    不需要 → 自己来。先 Glob / Grep 定位，再 Read 确认原文，最后动手改

**为什么按这个标准分**：你读过的每个文件都会一直留在你的上下文里，直到被压缩。
派出去的 SubAgent 读多少文件，都只还给你一段结论。
实测同一个问题，自己依次读 6 个文件花了 10 万 token，而先派 Explore 能省掉大部分。

派 SubAgent 的代价是你只拿到结论、看不到它的推理过程，
所以 task 要写清楚产出什么，context 里要把已知信息给全（它看不到你的对话历史）。

其它：
- 改局部内容用 Edit，不要用 Write 整体覆盖
- 所有路径都相对项目根目录，不要试图访问项目外的文件
- **动手之前先判断值不值得改**。没有明确的改进点就如实说，不要为了显得在干活而改
- 能直接回答的问题就直接回答，不要为了用工具而用工具"""


def _compose_system_prompt(base: str, memory: MemoryManager | None) -> str:
    """把长期记忆的索引拼进 system prompt。

    为什么必须注入：记忆要有用，Agent 得**先知道它存在**。不告诉它，
    它根本不会想到去翻 .agent/memory/ —— 再好的记忆也等于没有。

    代价很小：索引是一行一条链接加一句描述，相对它带来的价值可以忽略。
    这是渐进式披露 —— 索引告诉它「有什么」，需要细节时再用 Read 打开具体文件。
    """
    if memory is None:
        return base

    index = memory.render_index(INDEX_LINK_PREFIX)
    if not index:
        return base

    return (
        f"{base}\n\n"
        f"## 长期记忆\n\n"
        f"你有一份跨会话的长期记忆，索引如下。需要细节时用 Read 打开对应文件"
        f"（路径相对项目根目录）。\n\n"
        f"{index}"
    )


async def _no_tools_available(tool_call: dict) -> str:
    """没注册任何工具时的兜底执行器（Phase 1 的行为）。"""
    name = tool_call.get("function", {}).get("name")
    return f"当前没有注册任何工具（被请求的工具: {name}）"


class MainAgent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        session: Session | None = None,
        memory: MemoryManager | None = None,
        compactor: "ContextCompactor | None" = None,
        on_delta: "DeltaHook | None" = None,
        on_event: EventHook | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> None:
        self._llm = llm
        self._system_prompt = _compose_system_prompt(system_prompt, memory)
        self._max_steps = max_steps
        self._session = session
        self._compactor = compactor
        self._on_delta = on_delta
        self._on_event = on_event

        # registry 是可选依赖：不传就是 Phase 1 那种「没有工具」的状态。
        # 两种能力（schema 给模型看、execute 真执行）都从同一个对象取，
        # 不会出现「告诉模型有 Read 工具，但执行时找不到」这种不一致。
        self._tools = registry.schemas() if registry is not None else None
        self._execute_tool: ExecuteTool = (
            registry.execute if registry is not None else _no_tools_available
        )

    def _record(self, messages: list[dict], message: dict) -> None:
        """追加一条消息，并同步落盘。

        循环内部追加的消息走的是 on_message 钩子；这里处理的是循环之外的两条
        （system 和本次的 user）—— 它们同样是会话的一部分，漏记会让恢复出来的
        Context 缺头。
        """
        messages.append(message)
        if self._session is not None:
            self._session.append_message(message)

    @property
    def _message_hook(self) -> MessageHook | None:
        return self._session.append_message if self._session is not None else None

    async def run(self, question: str, messages: list[dict] | None = None) -> str:
        """回答一个问题。

        messages 传 None 时：有 session 就从它恢复历史，没有就是全新对话。

        **system prompt 不落盘、每次现拼**，理由是它是「配置」而不是「历史」：
        它带着长期记忆的索引，而记忆是会变的。要是把它冻结进 JSONL，续会话时
        Agent 看到的就是上次那份过时的索引 —— 记忆更新了却读不到，这毛病很难查。
        另外它也省掉了在 JSONL 里重复存一大段固定文本。
        """
        if messages is None:
            messages = self._session.load_messages() if self._session is not None else []

        system_message = {"role": "system", "content": self._system_prompt}
        if messages and messages[0].get("role") == "system":
            # 调用方自己塞了 system（同进程内多轮），用当前的覆盖掉
            messages[0] = system_message
        else:
            messages.insert(0, system_message)

        self._record(messages, {"role": "user", "content": question})

        return await run_react_loop(
            messages=messages,
            llm=self._llm,
            execute_tool=self._execute_tool,
            tools=self._tools,
            max_steps=self._max_steps,
            on_message=self._message_hook,
            compactor=self._compactor,
            on_delta=self._on_delta,
            on_event=self._on_event,
        )
