"""SubAgentTool —— Main Agent 派发 SubAgent 的唯一入口。

它**本身就是一个普通 Tool**。Main 的 ReAct Loop 完全不知道背后起了一个新 Agent，
只知道「调用了一个工具，拿到一段字符串」。这是 Phase 1 那个注入设计的又一次回报：
整个 SubAgent Runtime 没有新增任何循环代码，复用的就是同一个 run_react_loop。

三个关键设计都在下面能看到：

    1. 工具集裁剪   —— 子 Agent 的能力边界在注册表这一层被切干净
    2. Context 隔离 —— 子 Agent 的 messages 从零构造，不含 Main 的历史
    3. 递归禁止     —— 子 Agent 的工具集里没有 SubAgentTool
"""

import json
import logging

from pydantic import BaseModel, Field

from app.agent.react_loop import MaxIterationError, run_react_loop
from app.agent.subagent import AGENT_SPECS, AgentSpec, AgentType
from app.llm.client import LLMClient, LLMError
from app.tools.base import SandboxedTool, ToolError
from app.tools.bash_tool import BashTool
from app.tools.file_tool import (
    WRITE_SUCCESS_PREFIXES,
    EditTool,
    ListTool,
    ReadTool,
    WriteTool,
)
from app.tools.registry import ToolRegistry
from app.tools.sandbox import Sandbox
from app.tools.search_tool import GlobTool, GrepTool

logger = logging.getLogger(__name__)

# 子 Agent 能拿到的工具**全集**。
#
# 刻意不包含 SubAgentTool —— 子 Agent 无法再派子 Agent。Main → General-Purpose →
# General-Purpose → …… 的递归没有任何上界，一个失控任务就能把 API 余额烧干。
# 我们不靠深度计数去拦它（那要写状态、要处理边界），而是让这个能力
# **压根不存在于子 Agent 的工具集里**。
_SUB_TOOL_FACTORIES = (
    ReadTool,
    WriteTool,
    EditTool,
    ListTool,
    GlobTool,
    GrepTool,
    BashTool,
)

_WRITE_TOOLS = frozenset({"Write", "Edit"})


def _allowed_tools(spec: AgentSpec, sandbox: Sandbox) -> list:
    """按规格过滤出子 Agent 该拿到的工具实例。

    这是权限的落地点：spec.allowed_tools 里没有的名字，这里就不会产出，
    注册表里也就不会有 —— 模型连这个工具存在都不知道。
    """
    return [
        tool
        for tool in (factory(sandbox) for factory in _SUB_TOOL_FACTORIES)
        if tool.name in spec.allowed_tools
    ]


def build_sub_registry(spec: AgentSpec, sandbox: Sandbox) -> ToolRegistry:
    """装配一个子 Agent 的工具集（测试和外部调用用这个入口）。"""
    registry = ToolRegistry()
    for tool in _allowed_tools(spec, sandbox):
        registry.register(tool)
    return registry


class _TrackingRegistry(ToolRegistry):
    """在 ToolRegistry 上加一层记录：子 Agent 实际成功改了哪些文件。

    为什么不让模型自己总结「我改了哪些文件」：模型会漏、会忘、会把「打算改」
    说成「已经改」。工具调用记录是事实，模型的自述是转述。

    只记成功的：Write/Edit 会因为「old_string 不唯一」这类原因失败，
    那种情况下文件没动，不该出现在改动清单里。
    """

    def __init__(self) -> None:
        super().__init__()
        self.changed_files: list[str] = []

    async def execute(self, tool_call: dict) -> str:
        name = tool_call.get("function", {}).get("name", "")
        result = await super().execute(tool_call)

        if name in _WRITE_TOOLS and result.startswith(WRITE_SUCCESS_PREFIXES):
            path = _extract_path(tool_call)
            if path and path not in self.changed_files:
                self.changed_files.append(path)

        return result


def _extract_path(tool_call: dict) -> str | None:
    """从 tool_call 里尽力取出 path 参数。

    取不到就算了 —— 这只是为了生成清单，不值得为它抛异常。
    """
    try:
        args = json.loads(tool_call.get("function", {}).get("arguments") or "{}")
        path = args.get("path")
        return str(path) if path else None
    except (json.JSONDecodeError, AttributeError):
        return None


class _CountingLLM:
    """给 LLM 客户端套一层，数一数子 Agent 跑了几步。

    为什么不改成让 run_react_loop 返回步数：循环的签名是稳定的公共契约，
    为了一个统计数字去改它不划算。而 llm 本来就是注入进来的 ——
    包装它是最小侵入的做法，正是「依赖注入」这个设计的又一次顺手收益。
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls += 1
        return await self._inner.chat(messages, tools=tools)


def _compose_task(task: str, context: str | None) -> str:
    """把任务和已知信息拼成子 Agent 的第一条 user 消息。

    context 这一块不是装饰 —— 子 Agent 看不到 Main 的对话历史，
    Main 不主动交代的信息它就永远不知道。这是 Context 隔离的代价，
    也是这个方法存在的理由。
    """
    if not context:
        return task
    return f"{task}\n\n--- 已知信息（来自调用方，可直接采信） ---\n{context}"


def _last_progress(messages: list[dict]) -> str:
    """从已污染的历史里捞出最后一段有内容的 assistant 发言。

    run_react_loop 是就地修改 messages 的，所以即使它抛了 MaxIterationError，
    已经查到的东西也还在 —— 白扔掉等于让调用方白付一次 token。
    """
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


class _SubAgentParams(BaseModel):
    agent_type: AgentType = Field(
        description=(
            "Explore: 只读探索，回答「代码在哪、怎么组织的」；"
            "Plan: 只读研究，产出「该怎么改」的方案；"
            "General-Purpose: 可以改代码、跑命令，执行复杂任务"
        )
    )
    task: str = Field(
        description="交给 SubAgent 的任务。写清楚目标，以及你期望它产出什么"
    )
    context: str | None = Field(
        default=None,
        description=(
            "已知信息：相关文件路径、你已经排除的方向、遇到的报错等。"
            "**SubAgent 看不到你的对话历史**，不写在这里的信息它一概不知道"
        ),
    )


class SubAgentTool(SandboxedTool):
    name = "SubAgent"
    description = (
        "派一个子 Agent 去独立完成一项任务，只把最终结论带回来。"
        "适合「需要翻很多文件才能回答」或「改动较多需要反复验证」的任务 —— "
        "子 Agent 的中间过程不会占用你的上下文。"
        "缺点是你只拿到结论，看不到它的推理过程，所以任务描述要写清楚。"
    )
    params_model = _SubAgentParams

    def __init__(self, sandbox: Sandbox, llm: LLMClient) -> None:
        super().__init__(sandbox)
        self._llm = llm

    async def execute(self, agent_type: AgentType, task: str, context: str | None) -> str:
        spec = AGENT_SPECS.get(agent_type)
        if spec is None:
            # agent_type 已被 schema 约束成枚举，走到这里只可能是 AGENT_SPECS
            # 和 AgentType 漂移了 —— 宁可报清楚，也不要一个裸 KeyError
            raise ToolError(
                f"未知的 SubAgent 类型: {agent_type!r}。可用: {', '.join(AGENT_SPECS)}"
            )

        # Context 隔离的落地点：子 Agent 的上下文从零构造，只有两条消息。
        # Main 前面跑了多少步、看过哪些文件，这里一概没有。
        messages: list[dict] = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": _compose_task(task, context)},
        ]

        registry = _TrackingRegistry()
        for tool in _allowed_tools(spec, self.sandbox):
            registry.register(tool)

        counting_llm = _CountingLLM(self._llm)

        logger.info("[subagent] %s 启动 | 工具: %s", agent_type, ", ".join(t.name for t in registry.list_tools()))

        try:
            answer = await run_react_loop(
                messages=messages,
                llm=counting_llm,
                execute_tool=registry.execute,
                tools=registry.schemas(),
                max_steps=spec.max_steps,
            )
        except MaxIterationError as e:
            partial = _last_progress(messages)
            logger.warning("[subagent] %s 未完成: %s", agent_type, e)
            if partial:
                return (
                    f"[{agent_type} 未完成：达到 {spec.max_steps} 步上限]\n"
                    f"以下是它中断前的最后输出：\n{partial}"
                )
            return f"[{agent_type} 未完成：达到 {spec.max_steps} 步上限] 没有任何中间结论。"
        except LLMError as e:
            logger.warning("[subagent] %s LLM 调用失败: %s", agent_type, e)
            return f"[{agent_type} 未能执行：LLM 调用失败 —— {e}]"

        # messages 在这里随函数返回被回收 —— 文档要求的「执行结束销毁 Context」
        # 在 Python 里是默认行为，不需要额外代码。真正要防的是相反的事：
        # 别把它挂到 self 上做缓存，那就再也释放不掉了。

        return self._format_result(agent_type, counting_llm.calls, registry, answer)

    @staticmethod
    def _format_result(
        agent_type: str, steps: int, registry: _TrackingRegistry, answer: str
    ) -> str:
        header = [f"{agent_type} 完成", f"{steps} 步"]
        if registry.changed_files:
            header.append("修改: " + ", ".join(registry.changed_files))
        elif agent_type == "General-Purpose":
            header.append("未修改任何文件")

        return f"[{' | '.join(header)}]\n{answer}"
