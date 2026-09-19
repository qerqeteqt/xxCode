"""FileTool 系列 —— 对应文档「FileTool：read、write、edit、list」。

**为什么拆成 4 个独立工具，而不是一个带 action 参数的 File 工具**

function calling 里「一个工具名 = 一套参数 schema」。而 read 要 offset/limit、
write 要 content、edit 要 old_string/new_string —— 塞进同一个 schema 就只能把所有
字段设成可选，模型每次都得先猜 action 再配对参数。拆开之后每个 schema 都是紧的，
模型选错的空间小得多。

它们仍然属于同一个 FileTool 家族：共用同一个沙箱、同一个解码器、同一套错误措辞。
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import SandboxedTool, ToolError, ToolResult
from app.tools.sandbox import Sandbox
from app.tools.text import decode_bytes

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 单文件读取上限，防 OOM
DEFAULT_READ_LINES = 2000
MAX_READ_LINES = 5000
MAX_LIST_ENTRIES = 500


def _read_text(sandbox: Sandbox, target: Path) -> str:
    """读文件内容，三种常见失败给出模型能自救的提示。"""
    display = sandbox.rel(target)

    if not target.exists():
        raise ToolError(f"文件不存在: {display}")
    if target.is_dir():
        raise ToolError(f"{display} 是目录不是文件，要看目录内容请用 List 工具")

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        limit_mb = MAX_FILE_BYTES // 1024 // 1024
        raise ToolError(
            f"{display} 太大（{size / 1024 / 1024:.1f} MB，上限 {limit_mb} MB），拒绝读取"
        )

    return decode_bytes(target.read_bytes())


# ---------------------------------------------------------------- Read


class ReadParams(BaseModel):
    path: str = Field(description="文件路径，相对项目根目录，例如 app/llm/client.py")
    offset: int = Field(default=1, ge=1, description="从第几行开始读，行号从 1 开始")
    limit: int = Field(
        default=DEFAULT_READ_LINES,
        ge=1,
        le=MAX_READ_LINES,
        description=f"最多读多少行，上限 {MAX_READ_LINES}",
    )


class ReadTool(SandboxedTool):
    name = "Read"
    description = (
        "读取项目内某个文件的内容，返回带行号的文本。"
        "文件很大时用 offset/limit 分段读取，不要试图一次读完。"
    )
    params_model = ReadParams

    async def execute(self, path: str, offset: int, limit: int) -> ToolResult:
        target = self.sandbox.resolve(path)
        display = self.sandbox.rel(target)

        lines = _read_text(self.sandbox, target).splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult(f"{display} 是空文件")
        if offset > total:
            raise ToolError(f"offset={offset} 超出范围：{display} 一共只有 {total} 行")

        chunk = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{offset + i}\t{line}" for i, line in enumerate(chunk))

        end = offset - 1 + len(chunk)
        if end < total:
            # 必须显式告知「还有多少没读」，否则模型会把这一段当成全文
            body += f"\n…（{display} 共 {total} 行，已显示 {offset}-{end} 行，继续读请用 offset={end + 1}）"
        return ToolResult(body)


# ---------------------------------------------------------------- Write


class WriteParams(BaseModel):
    path: str = Field(description="文件路径，相对项目根目录")
    content: str = Field(description="要写入的完整文件内容")


class WriteTool(SandboxedTool):
    name = "Write"
    description = (
        "把内容整体写入文件，文件已存在则覆盖，父目录不存在会自动创建。"
        "只改局部内容请用 Edit，避免覆盖掉你没读到的部分。"
    )
    params_model = WriteParams

    async def execute(self, path: str, content: str) -> ToolResult:
        target = self.sandbox.resolve(path)
        display = self.sandbox.rel(target)

        if target.is_dir():
            raise ToolError(f"{display} 是目录，不能写入")

        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return ToolResult(
            f"{'已覆盖' if existed else '已新建'} {display}（{len(content)} 字符）",
            changed_path=display,
        )


# ---------------------------------------------------------------- Edit


class EditParams(BaseModel):
    path: str = Field(description="文件路径，相对项目根目录")
    old_string: str = Field(
        description="要被替换的原文，必须与文件内容精确匹配（含缩进和空行），且在文件中唯一出现"
    )
    new_string: str = Field(description="替换后的新文本")


class EditTool(SandboxedTool):
    name = "Edit"
    description = (
        "在文件里做精确字符串替换。old_string 必须在文件中恰好出现一次，"
        "否则拒绝执行 —— 多带几行上下文可以让它唯一。"
    )
    params_model = EditParams

    async def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        target = self.sandbox.resolve(path)
        display = self.sandbox.rel(target)

        if not old_string:
            raise ToolError("old_string 不能为空")
        if old_string == new_string:
            raise ToolError("old_string 与 new_string 完全相同，没有任何改动")

        text = _read_text(self.sandbox, target)

        count = text.count(old_string)
        if count == 0:
            raise ToolError(
                f"在 {display} 里找不到 old_string。它必须与文件内容**精确**匹配"
                f"（含缩进、空行、标点），建议先用 Read 确认原文"
            )
        if count > 1:
            raise ToolError(
                f"old_string 在 {display} 里出现了 {count} 次，不唯一。"
                f"请把上下相邻的几行也带上，让它唯一"
            )

        line_no = text[: text.index(old_string)].count("\n") + 1
        target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return ToolResult(
            f"已修改 {display}（原第 {line_no} 行处）", changed_path=display
        )


# ---------------------------------------------------------------- List


class ListParams(BaseModel):
    path: str = Field(default=".", description="目录路径，相对项目根目录，默认项目根")


class ListTool(SandboxedTool):
    name = "List"
    description = (
        "递归列出目录下的所有文件和子目录，自动跳过 .git / .venv / __pycache__ 等。"
        "用来快速了解项目结构，比一个个 Read 高效得多。"
    )
    params_model = ListParams

    async def execute(self, path: str) -> ToolResult:
        base = self.sandbox.resolve(path)
        display = self.sandbox.rel(base)

        if not base.exists():
            raise ToolError(f"目录不存在: {display}")
        if base.is_file():
            raise ToolError(f"{display} 是文件不是目录，要读内容请用 Read")

        files = sorted(self.sandbox.iter_files(base))
        if not files:
            return ToolResult(f"{display} 下没有文件")

        # 目录从文件路径反推出来 —— 空目录列不出来（os.walk 本来也不产出空目录），
        # 对「了解项目结构」这个用途来说，没有文件的目录也不重要
        dirs: set[Path] = set()
        for f in files:
            for parent in f.parents:
                if parent == base:
                    break
                dirs.add(parent)

        entries: list[Path] = sorted(dirs | set(files))
        truncated = len(entries) > MAX_LIST_ENTRIES
        shown = entries[:MAX_LIST_ENTRIES]

        lines = [
            f"{self.sandbox.rel(e)}/" if e.is_dir() else self.sandbox.rel(e)
            for e in shown
        ]
        if truncated:
            lines.append(f"…（共 {len(entries)} 项，只显示前 {MAX_LIST_ENTRIES} 项）")
        return ToolResult("\n".join(lines))
