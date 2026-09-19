"""工具包。

build_default_registry 是 Runtime 装配工具的唯一入口 —— 分散在各个模块里
自己 new 一堆工具的话，很容易出现「沙箱没传」「工具漏注册」这类问题，
集中在一处装配就没有这个空间。
"""

from pathlib import Path

from app.tools.base import SandboxedTool, Tool, ToolError
from app.tools.bash_tool import BashTool
from app.tools.file_tool import EditTool, ListTool, ReadTool, WriteTool
from app.tools.registry import ToolRegistry
from app.tools.sandbox import PathOutOfSandboxError, Sandbox
from app.tools.search_tool import GlobTool, GrepTool

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
    "Tool",
    "ToolError",
    "ToolRegistry",
    "WriteTool",
    "build_default_registry",
]


def build_default_registry(root: str | Path) -> ToolRegistry:
    """按项目根目录装配一整套工具。

    root 同时承担两个角色：安全边界（路径不许出圈）和「这次让 agent 看哪个项目」。
    所以它是参数而不是常量 —— 想分析别的项目，换个 root 就行。
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
    return registry
