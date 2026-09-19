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
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.tools.sandbox import Sandbox


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
    async def execute(self, **kwargs: object) -> str:
        """执行工具。

        kwargs 已经过 params_model 校验，类型可信，可以直接用。

        出错请**抛异常**，不要返回错误字符串 —— 统一由 ToolRegistry 兜住转成文本。
        这样每个工具只需要关心「正常路径怎么做」和「我为什么失败」，不用到处铺
        if/else 拼错误消息。
        """
        raise NotImplementedError


class SandboxedTool(Tool):
    """持有沙箱的工具。

    文件、搜索、Bash 三类工具都要过路径关，共用这一个基类比每个工具各写一遍
    `__init__(self, sandbox)` 更不容易漏 —— 漏掉的那个工具就成了沙箱上的洞。
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self.sandbox = sandbox
