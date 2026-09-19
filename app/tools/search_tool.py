"""SearchTool —— 文件名搜索（Glob）与内容搜索（Grep）。

**为什么搜索是 code agent 的命根子**

agent 面对陌生代码库，工作流永远是：不知道代码在哪 → 搜关键词 → 找到文件 →
读文件 → 改文件。第一步只能靠搜索 —— 它不知道「登录模块」在哪个文件里。

而之所以不能「把整个仓库塞进 prompt」：真实项目几万行，塞不进 Context 也付不起钱。
**搜索的本质是把大仓库压缩成「值得看的几行」的过滤器**，它不给答案，只告诉你该去哪找。

**为什么坚持纯 Python 而不是调 rg**

搜出来的条数和每条的截断长度，直接决定 Context 会不会爆 —— 这两个数字必须由我们
说了算。交给外部命令反而不好收口，何况 Windows 上还不一定有 rg。

一个已知的取舍：这里的文件读写是同步的，会短暂阻塞事件循环。
在当前规模（几百个文件）下无感；真要扫几万个文件，应该用 run_in_executor 丢到
线程池 —— 那是 Phase 7 的活。
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import SandboxedTool, ToolError

MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 200
MAX_SCAN_BYTES = 2 * 1024 * 1024  # 单个文件超过这个大小就不搜了
MAX_LINE_CHARS = 200


def _match(rel_path: Path, pattern: str) -> bool:
    """按 glob 模式匹配相对路径。

    补一次 pattern[3:] 是因为 pathlib 的 `**` 要求至少一层目录：
    `Path("main.py").match("**/*.py")` 是 False。这和大多数人的直觉
    （以及 `**/*.py` 应该匹配根目录下所有 py 文件）不一致，所以这里手动兜一下。
    """
    if rel_path.match(pattern):
        return True
    if pattern.startswith("**/"):
        return rel_path.match(pattern[3:])
    return False


class GlobParams(BaseModel):
    pattern: str = Field(
        description="glob 模式，例如 **/*.py 匹配所有 py 文件，app/**/*.py 限定在 app/ 下"
    )
    path: str = Field(default=".", description="搜索起始目录，相对项目根目录")


class GlobTool(SandboxedTool):
    name = "Glob"
    description = (
        "按文件名模式查找文件，返回路径列表。"
        "适合回答「项目里有哪些测试文件」「入口文件在哪」这类问题。"
    )
    params_model = GlobParams

    async def execute(self, pattern: str, path: str) -> str:
        base = self.sandbox.resolve(path)
        if not base.exists():
            raise ToolError(f"目录不存在: {self.sandbox.rel(base)}")
        if base.is_file():
            raise ToolError(
                f"{self.sandbox.rel(base)} 是文件，Glob 只能搜目录；要搜内容请用 Grep"
            )

        hits: list[str] = []
        for file_path in self.sandbox.iter_files(base):
            if _match(file_path.relative_to(base), pattern):
                hits.append(self.sandbox.rel(file_path))

        if not hits:
            return f"没有文件匹配 {pattern!r}"

        hits.sort()
        truncated = len(hits) > MAX_GLOB_RESULTS
        shown = hits[:MAX_GLOB_RESULTS]
        body = "\n".join(shown)
        if truncated:
            body += f"\n…（共 {len(hits)} 个匹配，只显示前 {MAX_GLOB_RESULTS} 个，请用更精确的 pattern 收窄）"
        return body


class GrepParams(BaseModel):
    pattern: str = Field(
        description="正则表达式，例如 def login；忽略大小写可以用 (?i) 前缀"
    )
    path: str = Field(default=".", description="搜索起始目录，相对项目根目录")
    include: str | None = Field(
        default=None, description="只搜索文件名匹配该 glob 的文件，例如 *.py"
    )


class GrepTool(SandboxedTool):
    name = "Grep"
    description = (
        "按内容正则搜索，返回「文件:行号: 那一行」。"
        "这是定位代码最有效的工具 —— 想改什么，先用它找到在哪个文件。"
    )
    params_model = GrepParams

    async def execute(self, pattern: str, path: str, include: str | None) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"正则表达式不合法: {e}") from None

        base = self.sandbox.resolve(path)
        if not base.exists():
            raise ToolError(f"路径不存在: {self.sandbox.rel(base)}")

        files = [base] if base.is_file() else list(self.sandbox.iter_files(base))

        matches: list[str] = []
        binary_skipped = 0
        truncated = False

        for file_path in files:
            rel = file_path.relative_to(base) if base.is_dir() else file_path
            if include and not _match(rel, include):
                continue

            try:
                if file_path.stat().st_size > MAX_SCAN_BYTES:
                    continue
                # 这里刻意用严格 utf-8 而不是 decode_bytes：二进制文件会因为解码失败
                # 被跳过，正是我们想要的。用带 fallback 的解码器会把二进制变成乱码文本，
                # 反而污染搜索结果。
                text = file_path.read_bytes().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                binary_skipped += 1
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(
                        f"{self.sandbox.rel(file_path)}:{line_no}: "
                        f"{line.strip()[:MAX_LINE_CHARS]}"
                    )
                    if len(matches) > MAX_GREP_MATCHES:
                        truncated = True
                        break
            if truncated:
                break

        if not matches:
            note = f"（跳过了 {binary_skipped} 个二进制/非 UTF-8 文件）" if binary_skipped else ""
            return f"没有匹配 {pattern!r} 的内容{note}"

        body = "\n".join(matches)
        if truncated:
            body += f"\n…（匹配过多，只显示前 {MAX_GREP_MATCHES} 条，请用更精确的 pattern）"
        return body
