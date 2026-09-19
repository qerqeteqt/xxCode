"""ToolRegistry —— 工具注册表，也是 Phase 1 欠下的那笔账的还款处。

Phase 1 说「Invalid Tool Call 的处理留到 Phase 2，因为那时才有 parse 的动作」，
就是这里：`arguments` 是模型给的 **JSON 字符串**，它可能是坏的
（截断了、不是 JSON、字段名写错了）。这些都必须变成一条模型看得懂的 Observation，
而不是异常穿透上去把整个 Runtime 干掉。

所以本文件的核心契约只有一条：
    execute() 永不抛异常，永远返回可以回灌给模型的字符串。

这条契约有个代价：成败状态在字符串化的时候丢了。所以内部还有一层
execute_detailed() 返回 ToolResult，给 Runtime 自己用（累计 files_changed 等）。
外部（ReAct Loop、模型）只看到 execute() 的字符串那一面。
"""

import json
import logging
from collections.abc import Callable

from pydantic import ValidationError

from app.tools.base import Tool, ToolError, ToolResult

# 是拿到一个 logger 实例 ,打印日志的
logger = logging.getLogger(__name__)

# 回显给模型时的截断长度：出错时把原始参数贴回去帮它自查，但不能贴太长
_RAW_ARGS_PREVIEW = 200


class ToolRegistry:
    def __init__(self) -> None:
        # 也就是工具的名称 + 工具本身
        self._tools: dict[str, Tool] = {}

    # ---------- 增删查 ----------

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        """给 LLM 的 tools 参数。MainAgent 直接把这个传给 ReAct Loop。"""
        return [tool.schema() for tool in self._tools.values()]

    # ---------- 执行 ----------

    async def execute(self, tool_call: dict) -> str:
        """执行一次 tool_call，返回回灌给模型的字符串。

        这是**给模型看的**那一面 —— 调用方拿不到成败、改动文件之类的元信息。
        Runtime 内部需要元信息的消费者走 execute_detailed()。
        """
        return (await self.execute_detailed(tool_call)).text

    async def execute_detailed(self, tool_call: dict) -> ToolResult:
        """执行一次 tool_call，返回结构化结果。

        四道关卡，每一道对应模型的一种典型错误：
            1. 工具名不存在      -> 模型记错了工具名
            2. arguments 不是 JSON -> 模型把参数写坏了
            3. 参数不符合 schema   -> 模型字段名/类型写错了
            4. 工具自己执行失败    -> 路径不存在、命令超时等
        """
        function = tool_call.get("function", {})
        name = function.get("name", "")

        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "（无）"
            return ToolResult(f"没有名为 {name!r} 的工具。可用工具: {available}", ok=False)

        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return ToolResult(
                f"参数不是合法 JSON（{e.msg}）。收到的原始参数: "
                f"{raw_args[:_RAW_ARGS_PREVIEW]}",
                ok=False,
            )

        if not isinstance(args, dict):
            return ToolResult(
                f"参数必须是 JSON 对象，实际收到 {type(args).__name__}: "
                f"{raw_args[:_RAW_ARGS_PREVIEW]}",
                ok=False,
            )

        try:
            params = tool.params_model.model_validate(args)
        except ValidationError as e:
            # 默认的 str(e) 是一大坨，这里压成「字段: 原因」的形式，模型更好消化
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '(根)'}: {err['msg']}"
                for err in e.errors()
            )
            return ToolResult(f"参数不合法 —— {detail}", ok=False)

        logger.info("[tool] %s(%s)", name, raw_args[:_RAW_ARGS_PREVIEW])
        try:
            result = await tool.execute(**params.model_dump())
        except ToolError as e:
            # 预期内的失败（路径不存在、命令被拒……），消息本来就是写给模型看的
            logger.info("[tool] %s 未完成: %s", name, e)
            return ToolResult(f"{name}: {e}", ok=False)
        except Exception as e:  # noqa: BLE001 —— 见文件头的契约，什么都不许漏出去
            # 预期外的失败：这是 Runtime 自己的 bug，日志要留清楚，
            # 回给模型的措辞也要和上面那种区分开，否则没法从对话里看出哪里出了问题
            logger.exception("[tool] %s 内部异常", name)
            return ToolResult(f"{name} 内部异常: {type(e).__name__}: {e}", ok=False)

        logger.info("[tool] %s -> %s", name, result.text[:_RAW_ARGS_PREVIEW])
        return result


class TrackingRegistry(ToolRegistry):
    """额外记下「哪些文件被真的改动了」。

    Main 的 Session State 和 SubAgent 的改动清单都要这份数据，所以放在注册表层
    共用一份实现。

    在改成 ToolResult 之前，这个判断是「嗅探返回文本的前缀」—— 因为成败状态
    被字符串化丢掉了。现在它只是读 result.changed_path 一个字段。
    """

    def __init__(self, on_change: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.changed_files: list[str] = []
        self._on_change = on_change

    async def execute_detailed(self, tool_call: dict) -> ToolResult:
        result = await super().execute_detailed(tool_call)

        # result.ok 这一步不能省：Edit 会因为「old_string 不唯一」失败，
        # 那时文件根本没动，不该出现在改动清单里
        changed = result.changed_path
        if result.ok and changed and changed not in self.changed_files:
            self.changed_files.append(changed)
            if self._on_change is not None:
                self._on_change(changed)

        return result
