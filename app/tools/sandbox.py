"""路径沙箱 —— 所有文件类工具的唯一入口。

问题：工具收到的路径是**模型给的字符串**，可能是
    "app/llm/client.py"             正常
    "D:/pycharm/xxCode/main.py"     绝对路径，也正常
    "../../../Users/x_x/.ssh/id_rsa"   往上跑
    "D:/pycharm/Mutil-Agent/src/x.py"  跑到别的项目

后两种的后果不是「看到了不该看的」，而是**任务直接失败**：一堆无关内容塞进
Context，把真正有用的信息淹没，token 费用照付，而模型不知道自己错了。

做法：先 resolve 成真实绝对路径（把 `..` 和符号链接全部展开），再判断它是否
仍然落在 root 之内。**顺序不能反** —— 直接检查字符串里有没有 ".." 是错的，
因为 "a/../b.py" 完全合法，而 Windows 上 "C:\\a\\..\\..\\b" 这种字符串很难用正则判对。

顺带一提：这个 root 不只是安全边界，它同时是「这次让 agent 看哪个项目」的开关。
"""

import os
from collections.abc import Iterator
from pathlib import Path

# 遍历时跳过的目录。
# 为什么这件事非做不可：Grep 会走遍大量文件，如果连 .git（几千个二进制对象）
# 和 .venv（几万个文件）都扫，一次搜索能跑几分钟，而且返回的全是乱码噪声。
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".agent",  # Runtime 自己的数据目录，对代码任务没意义
    }
)


class PathOutOfSandboxError(PermissionError):
    """路径越界。消息会给模型看，所以要说清楚「规则是什么」，而不只是「不行」。"""


class Sandbox:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, user_path: str) -> Path:
        """把模型给的路径字符串解析成 root 内的绝对路径，越界则抛错。

        两个 pathlib 行为在这里帮了大忙，不需要额外分支：

        1. `root / "D:/x/y"` —— 右边是绝对路径时，pathlib 直接丢掉左边。
           所以模型给绝对路径也能正常解析，然后照样被下面的检查拦住。
        2. `Path.resolve()` 会把 `..`、`.` 和符号链接全部展开成真实路径。
           `root/../../../Windows` 到这里就变成 `D:\\Windows`，原形毕露。
        """
        target = (self.root / user_path).resolve()
        if not target.is_relative_to(self.root):
            raise PathOutOfSandboxError(
                f"路径越界: {user_path!r} —— 只能访问项目目录内的文件，"
                f"不能使用 .. 或绝对路径跳出项目 root"
            )
        return target

    def rel(self, path: Path) -> str:
        """转成相对 root 的路径，用于回显给模型。

        回显相对路径有两个好处：省 token，以及让模型看到的路径和它传进来的
        形式一致，它下次改路径时不容易犯迷糊。
        """
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def iter_files(self, base: Path) -> Iterator[Path]:
        """遍历 base 下的所有文件，跳过 IGNORED_DIRS。

        遍历逻辑放在 Sandbox 里而不是各个工具里，是因为「哪些目录算项目的一部分」
        本身就是沙箱该回答的问题 —— 它和「路径能不能出圈」是同一个边界的两面。
        三个工具（List / Glob / Grep）共用这一份规则，就不会出现
        「List 看不到但 Grep 能搜到」这种不一致。

        用 os.walk 而不是 rglob，是因为 rglob 没法在遍历中途剪枝 ——
        它照样会走进 .venv 再逐个过滤，该慢还是慢。
        """
        for dirpath, dirnames, filenames in os.walk(base):
            # 原地修改 dirnames 是 os.walk 的剪枝约定：
            # 被移除的目录这一轮就不会再往下走
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for filename in filenames:
                yield Path(dirpath) / filename
