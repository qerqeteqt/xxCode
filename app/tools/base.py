"""Tool 抽象。

文档要求统一抽象：name、description、input_schema、execute()。

这里有个值得注意的取舍：input_schema 不是手写的字典，而是由 Pydantic 的
params_model **生成**出来的。好处是一份定义同时产出两样东西：

    参数校验（registry 用）  ← params_model.model_validate(args)
    给模型的说明书（LLM 用） ← params_model.model_json_schema()

手写 schema 的话这两份东西会各写一遍，改一处忘一处就会出现
「schema 说有这个参数，校验时却不认」这种极难查的 bug。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.tools.sandbox import Sandbox


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结果。

    为什么不直接返回字符串：Runtime 内部有两类消费者，需求不一样 ——

        · 模型：只要 text，一句能读懂的话
        · Runtime：想知道 ok 和 changed_path（累计 files_changed、展示成败）

    把状态编进字符串（比如嗅探返回文本的前缀）能让这两类人都勉强工作，
    但状态一旦字符串化，类型信息就丢了，调用方只能做文本判断 ——
    改一句文案就可能悄悄改掉别人的判断结果。

    changed_path 只在真正改动文件时才填（目前是 Write / Edit）。
    所以「哪些工具算写操作」不再需要一个工具名清单 —— 谁填了就算谁。
    """

    text: str
    ok: bool = True
    changed_path: str | None = None


class ToolError(Exception):
    """工具执行失败。

    消息会被回灌给模型，所以措辞要让模型能看懂、能据此纠正 ——
    比如「文件不存在: app/x.py」比「[Errno 2] No such file」有用得多。
    """


class Tool(ABC):
    """所有工具的基类。

    name / description / params_model 是**类属性**而不是实例属性：它们是
    「这个工具是什么」的静态事实，同一类工具的各个实例之间不会变。
    会变的是它操作的对象（比如 FileTool 系列各自持有的 sandbox）。
    """

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    def schema(self) -> dict:
        """转成 OpenAI function calling 的 tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_model.model_json_schema(),
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolResult:
        """执行工具。

        kwargs 已经过 params_model 校验，类型可信，可以直接用。

        出错请**抛异常**，不要自己拼 ToolResult(ok=False) —— 统一由 ToolRegistry
        兜住并转换。这样每个工具只需要关心「正常路径怎么做」和「我为什么失败」，
        不用到处铺 if/else 拼错误消息。
        """
        raise NotImplementedError


class SandboxedTool(Tool):
    """持有沙箱的工具。

    文件、搜索、Bash 三类工具都要过路径关，共用这一个基类比每个工具各写一遍
    `__init__(self, sandbox)` 更不容易漏 —— 漏掉的那个工具就成了沙箱上的洞。
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self.sandbox = sandbox
