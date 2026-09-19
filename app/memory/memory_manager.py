"""长期记忆 —— Markdown 文件 + MEMORY.md 索引。

## 这个类负责什么，不负责什么

文档第十三节说得很明确：

    MemoryManager 不负责决定什么信息值得长期保存；这一决策由 Memory Agent /
    AutoDream 完成。

所以这里是一个**哑的读写层**：怎么存、怎么读、索引怎么维护。至于「这段对话里
什么值得记下来」，是 Phase 6 的 AutoDream 的活。别在这个文件里加"智能"。

## MEMORY.md 是派生数据

索引不是手工维护的，而是**从目录里的文件重新算出来**的。手工维护有个很隐蔽的
失败模式：**孤儿记忆** —— 文件存在、内容很好，但索引里没有链接，于是没有任何人
会知道它存在。存了等于没存，而且你根本不知道自己漏了什么。

派生数据不该手工维护。所以 write / update / delete 都会顺手重建索引。

## 为什么不暴露成 Tool

文档第十二节要求长期记忆「保证人和 Agent 都可以直接读取和修改」。那就该用普通的
Read / Write / Edit —— 沙箱已经放行了 `.agent/memory`（见 sandbox.py 里
IGNORED_PATHS 的注释）。为记忆发明一套专用工具，只会把它变成谁都碰不得的特殊数据。

## 文件格式

    ---
    name: 项目架构
    description: 模块划分与依赖方向，改代码前先看
    category: Project
    ---

    正文……

元信息放 frontmatter，正文是纯 Markdown —— 文件本身就是一篇能读的文章，
元信息只服务于索引生成。
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.tools.sandbox import Sandbox

logger = logging.getLogger(__name__)

MEMORY_DIR = ".agent/memory"
INDEX_FILENAME = "MEMORY.md"
DEFAULT_CATEGORY = "General"

# 注入 system prompt 时链接要写成相对项目根的路径，Agent 的 Read 才能直接用
INDEX_LINK_PREFIX = f"{MEMORY_DIR}/"

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.S)


class MemoryStoreError(RuntimeError):
    """记忆读写失败。

    注意别把这个类叫 MemoryError —— 那是 Python 内置异常（内存耗尽），
    覆盖掉它会让真正的 OOM 报错变成一个莫名其妙的记忆错误。
    """


@dataclass(frozen=True)
class Memory:
    key: str  # 文件名（不含 .md），也是读写时用的标识
    name: str  # frontmatter 里的显示名
    description: str
    category: str
    content: str  # 正文，不含 frontmatter
    path: Path

    @property
    def rel_path(self) -> str:
        """相对项目根的路径。给改动追踪和展示用（和 Sandbox.rel 是同一套写法）。"""
        return f"{MEMORY_DIR}/{self.key}.md"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """把文件拆成 (元信息, 正文)。

    解析失败时按「没有 frontmatter」处理，而不是抛异常 —— 你手写一个记忆文件
    忘了闭合 `---`，不该让整个索引崩掉。
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        logger.warning("frontmatter 解析失败，按纯文本处理: %s", e)
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[match.end() :]


def _render_document(name: str, description: str, category: str, content: str) -> str:
    meta = {"name": name, "description": description, "category": category}
    # allow_unicode=True：中文直接写出来，别变成 \u9879\u76ee 这种
    front = yaml.safe_dump(
        meta, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).strip()
    return f"---\n{front}\n---\n\n{content.strip()}\n"


class MemoryManager:
    def __init__(self, root: str | Path) -> None:
        self.sandbox = Sandbox(root)
        # 经沙箱解析一次，后面所有路径都以它为准
        self.dir = self.sandbox.resolve(MEMORY_DIR)

    # ------------------------------------------------------------ 路径

    def _path_for(self, key: str) -> Path:
        """把 key 变成记忆文件路径，并保证它真的落在记忆目录里。

        为什么用沙箱解析而不是字符串拼接：这个 key 将来由 AutoDream 提供，
        也就是**由 LLM 生成**，不是可信输入。`"../../poison"` 这种必须被挡住。
        沙箱会把 `..` 展开成真实路径，越界自然露馅。

        顺带，这个检查同时禁掉了子目录（`parent != self.dir`）——
        记忆文件平铺在记忆目录下，不分子目录。
        """
        key = key.strip()
        if not key:
            raise MemoryStoreError("记忆的 key 不能为空")
        # "." / ".." 要单独挡：因为后面会拼上 ".md"，`".."` 会变成合法的文件名
        # `"...md"`，落在目录里、也越不了界，但显然不是谁想要的名字
        if key in {".", ".."}:
            raise MemoryStoreError(f"非法的记忆 key: {key!r}")

        target = self.sandbox.resolve(f"{MEMORY_DIR}/{key}.md")
        if target.parent != self.dir:
            raise MemoryStoreError(
                f"非法的记忆 key: {key!r} —— key 里不能有路径分隔符或 .."
            )
        if target.name == INDEX_FILENAME:
            raise MemoryStoreError(f"{INDEX_FILENAME} 是索引，不能当作记忆名")
        return target

    def _read_file(self, path: Path) -> Memory:
        meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        return Memory(
            key=path.stem,
            name=str(meta.get("name") or path.stem),
            description=str(meta.get("description") or ""),
            category=str(meta.get("category") or DEFAULT_CATEGORY),
            content=body.strip(),
            path=path,
        )

    # ------------------------------------------------------------ 读

    def list_memories(self) -> list[Memory]:
        """列出所有记忆。索引文件本身不算记忆，要排除掉。"""
        if not self.dir.exists():
            return []
        return [
            self._read_file(path)
            for path in sorted(self.dir.glob("*.md"))
            if path.name != INDEX_FILENAME
        ]

    def read_memory(self, key: str) -> Memory:
        path = self._path_for(key)
        if not path.exists():
            raise MemoryStoreError(f"没有这条记忆: {key}")
        return self._read_file(path)

    # ------------------------------------------------------------ 写

    def write_memory(
        self,
        key: str,
        content: str,
        *,
        name: str | None = None,
        description: str = "",
        category: str = DEFAULT_CATEGORY,
    ) -> Memory:
        """新建或整体覆盖一条记忆。"""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _render_document(name or key, description, category, content),
            encoding="utf-8",
        )
        self.sync_index()
        return self._read_file(path)

    def update_memory(self, key: str, content: str) -> Memory:
        """只改正文，保留原有的元信息（显示名、描述、分类）。"""
        existing = self.read_memory(key)
        return self.write_memory(
            key,
            content,
            name=existing.name,
            description=existing.description,
            category=existing.category,
        )

    def delete_memory(self, key: str) -> None:
        path = self._path_for(key)
        if not path.exists():
            raise MemoryStoreError(f"没有这条记忆: {key}")
        path.unlink()
        self.sync_index()

    # ------------------------------------------------------------ 索引

    def render_index(self, link_prefix: str = "./") -> str:
        """渲染索引**正文**（不含一级标题）。

        link_prefix 决定链接怎么写，因为索引有两类读者：

            MEMORY.md 文件里    './'                相对自身，人和编辑器能点开
            注入 system prompt  '.agent/memory/'    相对项目根，Agent 的 Read 直接用

        **一条记忆都没有时返回空字符串**，而不是「还没有任何长期记忆」这种话 ——
        调用方（MainAgent）靠空串判断该不该往 system prompt 里塞这一段。
        给模型白送一句「你没有任何记忆」纯属浪费 token。
        """
        memories = self.list_memories()
        if not memories:
            return ""

        groups: dict[str, list[Memory]] = {}
        for memory in memories:
            groups.setdefault(memory.category, []).append(memory)

        lines: list[str] = []
        # 排序固定，否则每次重建出来的索引顺序可能不同，git diff 全是噪声
        for category in sorted(groups):
            lines.append(f"## {category}")
            for memory in sorted(groups[category], key=lambda m: m.key):
                description = f" — {memory.description}" if memory.description else ""
                lines.append(
                    f"- [{memory.name}]({link_prefix}{memory.key}.md){description}"
                )
            lines.append("")
        return "\n".join(lines).rstrip()

    def sync_index(self) -> Path:
        """重建 MEMORY.md。

        启动时和每次增删改之后都会调用 —— 索引永远和目录里的文件一致，
        这是结构性保证，不是靠人记得更新。
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / INDEX_FILENAME
        # 文件里要有个占位，否则打开 MEMORY.md 是一片空白，看不出这文件是干嘛的
        body = self.render_index() or "（还没有任何长期记忆）"
        rendered = f"# Memory\n\n{body}\n"

        # 内容没变就不写。否则每次启动都会刷新一次 mtime，git 里一直显示这个文件脏。
        if not path.exists() or path.read_text(encoding="utf-8") != rendered:
            path.write_text(rendered, encoding="utf-8")
            logger.info("已重建记忆索引: %s", path)
        return path
