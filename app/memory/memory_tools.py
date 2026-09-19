"""给 AutoDream 用的记忆工具。

这四个工具**只在 AutoDream 的注册表里**。Main Agent 的 registry 一个都没有 ——
这正是文档第十四节要求的「AutoDream 不作为 Main Agent 的普通 Tool」。

## 为什么需要它们（Main Agent 明明有 Write/Edit）

Main Agent 用普通 `Write` 工具确实能往 `.agent/memory/` 里写文件（沙箱放行了），
但那样**不会触发索引同步** —— `MEMORY.md` 要等下次启动 `sync_index()` 才更新。
中间这段时间里，那条记忆在索引里是缺失的，也就是个「孤儿记忆」。

走 `WriteMemory` 则会立刻同步。所以这组工具的价值不只是「能写」，
而是**「写完索引一定是对的」**。

## 为什么 changed_path 填的是记忆文件路径

`ToolResult.changed_path` 的语义是「这次调用真的改了哪个文件」。记忆工具改的
就是记忆文件，填上去名正言顺。好处是 TrackingRegistry 能原样复用 ——
AutoDream 结束时就能报出「这次改动了哪几条记忆」，跟 SubAgent 报改动清单是同一套机制。
"""

from pydantic import BaseModel, Field

from app.memory.memory_manager import (
    DEFAULT_CATEGORY,
    MEMORY_DIR,
    Memory,
    MemoryManager,
    MemoryStoreError,
)
from app.tools.base import Tool, ToolError, ToolResult
from app.tools.registry import ToolRegistry


def _render(memory: Memory) -> str:
    """把一条记忆渲染成给模型看的文本（含元信息）。

    元信息要带上：模型的下一步可能是「更新它」，那时它需要知道这条记忆的
    显示名和分类，否则 update 会把元信息丢掉。
    """
    lines = [f"key: {memory.key}", f"name: {memory.name}", f"category: {memory.category}"]
    if memory.description:
        lines.append(f"description: {memory.description}")
    return "\n".join(lines) + f"\n\n{memory.content}"


class _KeyParams(BaseModel):
    key: str = Field(description="记忆的文件名，不含 .md，例如 code-conventions")


class _MemoryTool(Tool):
    """持有 MemoryManager 的工具基类。

    不需要沙箱 —— 路径安全由 MemoryManager 自己负责（它的 key 也要过沙箱，
    因为 key 将来是 LLM 给的）。
    """

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory

    def subject(self, params: _KeyParams) -> str:
        """报出记忆文件的相对路径，好让权限规则能用路径模式表达。

        AutoDream 的注册表不挂闸门（见 memory_registry 的注释），所以这里
        平时不会被用到 —— 但工具该声明的东西还是要声明齐，
        万一将来给别的场景复用了这份注册表，不至于漏掉。
        """
        return f"{MEMORY_DIR}/{params.key}.md"


# ---------------------------------------------------------------- Read


class ReadMemoryTool(_MemoryTool):
    name = "ReadMemory"
    risk = "read"
    description = "读取一条长期记忆的完整内容（含元信息）。key 就是它在索引里的文件名。"
    params_model = _KeyParams

    async def execute(self, key: str) -> ToolResult:
        try:
            memory = self.memory.read_memory(key)
        except MemoryStoreError as e:
            # 转成 ToolError：那是「预期内的失败」，回给模型的措辞会保持原样；
            # 不转的话 registry 会当成 Runtime 内部 bug，附上一堆异常类名
            raise ToolError(str(e)) from e
        return ToolResult(_render(memory))


# ---------------------------------------------------------------- Write


class _WriteParams(BaseModel):
    key: str = Field(
        description="记忆的文件名，不含 .md。用小写加短横线，例如 code-conventions"
    )
    content: str = Field(description="正文，Markdown 格式。一条记忆聚焦一个主题")
    name: str = Field(description="显示名，会出现在索引里")
    description: str = Field(default="", description="一句话说明这条记忆讲什么")
    category: str = Field(
        default=DEFAULT_CATEGORY, description="分组，例如 Project / User / Lessons"
    )


class WriteMemoryTool(_MemoryTool):
    name = "WriteMemory"
    risk = "write"
    description = (
        "新建一条记忆，或整体覆盖已有的。索引会自动更新。"
        "只想改正文、保留元信息的话用 UpdateMemory。"
    )
    params_model = _WriteParams

    async def execute(
        self, key: str, content: str, name: str, description: str, category: str
    ) -> ToolResult:
        try:
            memory = self.memory.write_memory(
                key, content, name=name, description=description, category=category
            )
        except MemoryStoreError as e:
            raise ToolError(str(e)) from e
        return ToolResult(
            f"已写入记忆「{memory.name}」({memory.rel_path})", changed_path=memory.rel_path
        )


# ---------------------------------------------------------------- Update


class _UpdateParams(_KeyParams):
    # 继承 _KeyParams 而不是各写一遍 key：这样 _MemoryTool.subject 的签名
    # 对四个工具都是准确的，不会出现「注解说是 A、实际传的是 B」
    content: str = Field(description="新的正文，会整体替换旧正文")


class UpdateMemoryTool(_MemoryTool):
    name = "UpdateMemory"
    risk = "write"
    description = "更新一条已有记忆的正文，保留它的显示名、描述和分类。"
    params_model = _UpdateParams

    async def execute(self, key: str, content: str) -> ToolResult:
        try:
            memory = self.memory.update_memory(key, content)
        except MemoryStoreError as e:
            raise ToolError(str(e)) from e
        return ToolResult(
            f"已更新记忆「{memory.name}」({memory.rel_path})", changed_path=memory.rel_path
        )


# ---------------------------------------------------------------- Delete


class DeleteMemoryTool(_MemoryTool):
    name = "DeleteMemory"
    risk = "write"
    description = (
        "删除一条记忆。**只在它确实过时或被推翻时用** —— "
        "内容只是需要修正的话，用 UpdateMemory 更安全。"
    )
    params_model = _KeyParams

    async def execute(self, key: str) -> ToolResult:
        try:
            memory = self.memory.read_memory(key)
            self.memory.delete_memory(key)
        except MemoryStoreError as e:
            raise ToolError(str(e)) from e
        return ToolResult(
            f"已删除记忆「{memory.name}」({memory.rel_path})", changed_path=memory.rel_path
        )


def memory_tools(memory: MemoryManager) -> list[Tool]:
    """AutoDream 的全部工具。**只有记忆工具，没有别的。**

    刻意不给 Read / Glob / Grep / Bash。AutoDream 的任务边界很清楚 ——
    把会话提炼成记忆；给它读项目代码的能力只会让它跑去「顺便看看代码」，
    烧钱且跑偏。这是「能力不存在」原则用得最干净的一次。

    `list_memories` 不做成工具：索引会随材料一起给它，重复暴露只是浪费 schema。
    """
    return [
        ReadMemoryTool(memory),
        WriteMemoryTool(memory),
        UpdateMemoryTool(memory),
        DeleteMemoryTool(memory),
    ]


def build_memory_registry(memory: MemoryManager) -> ToolRegistry:
    """装配一个记忆工具注册表（测试和外部调用用这个入口）。"""
    registry = ToolRegistry()
    for tool in memory_tools(memory):
        registry.register(tool)
    return registry
