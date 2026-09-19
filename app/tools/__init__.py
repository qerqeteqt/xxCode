"""工具包。

build_default_registry 是 Runtime 装配工具的唯一入口 —— 分散在各个模块里
自己 new 一堆工具的话，很容易出现「沙箱没传」「工具漏注册」这类问题，
集中在一处装配就没有这个空间。
"""

from pathlib import Path

from app.llm.client import LLMClient
from app.tools.base import SandboxedTool, Tool, ToolError
from app.tools.bash_tool import BashTool
from app.tools.file_tool import EditTool, ListTool, ReadTool, WriteTool
from app.tools.registry import ToolRegistry
from app.tools.sandbox import PathOutOfSandboxError, Sandbox
from app.tools.search_tool import GlobTool, GrepTool
from app.tools.subagent_tool import SubAgentTool, build_sub_registry

__all__ = [
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ListTool",
    "PathOutOfSandboxError",
    "ReadTool",
    "Sandbox",
    "SandboxedTool",
    "SubAgentTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "WriteTool",
    "build_default_registry",
    "build_sub_registry",
]


def build_default_registry(root: str | Path, llm: LLMClient | None = None) -> ToolRegistry:
    """按项目根目录装配一整套工具。

    root 同时承担两个角色：安全边界（路径不许出圈）和「这次让 agent 看哪个项目」。
    所以它是参数而不是常量 —— 想分析别的项目，换个 root 就行。

    llm 是可选的，因为只有 SubAgent 需要它（要驱动子循环）。不传它就是
    「这个 Runtime 没有子 Agent 能力」的形态 —— 和 Phase 1 里 registry 可选
    是同一个套路：能力由装配决定，而不是靠运行时开关。
    """
    sandbox = Sandbox(root)
    registry = ToolRegistry()
    for tool in (
        ReadTool(sandbox),
        WriteTool(sandbox),
        EditTool(sandbox),
        ListTool(sandbox),
        GlobTool(sandbox),
        GrepTool(sandbox),
        BashTool(sandbox),
    ):
        registry.register(tool)

    if llm is not None:
        registry.register(SubAgentTool(sandbox, llm))

    return registry
