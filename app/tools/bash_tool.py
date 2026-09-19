"""BashTool —— 在项目目录内执行 shell 命令。

**关于安全档位，先把话说清楚，免得日后误以为这条路是安全的**

这里实现的是「档 1：危险命令黑名单」。它挡的是**误伤**，不是攻击者 ——
`rm -rf /` 会被拦住，但变量拼接、`r""m`、先写个脚本再执行，全都绕得过去。
真正的权限系统（可配置策略 / 人工确认 / 容器隔离）属于 Phase 7。

那为什么还要先做这一档：Phase 3-6 的调试过程中 agent 会反复执行命令，
没有这层，一个幻觉就足以删掉你整个工作目录。风险是不对称的 ——
几十行代码换掉一整类不可逆事故，划算。
"""

import asyncio
import contextlib
import os
import re
import signal
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import SandboxedTool, ToolError, ToolResult
from app.tools.text import decode_bytes, truncate

MAX_OUTPUT_CHARS = 8000
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0

# ---------------------------------------------------------------- 危险命令规则

# 与 rm 无关的关键词规则
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmkfs(\.[a-z0-9]+)?\b", re.I), "格式化文件系统"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/", re.I), "直接写入块设备"),
    (re.compile(r"\bformat\s+[a-z]:", re.I), "格式化磁盘分区"),
    (re.compile(r"\bdiskpart\b", re.I), "磁盘分区操作"),
    (re.compile(r">\s*/dev/(sd|hd|nvme|disk)", re.I), "覆写块设备"),
    (re.compile(r":\s*\(\s*\)\s*\{[^}]*\|[^}]*\}", re.I), "fork 炸弹"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I), "关机 / 重启系统"),
    (re.compile(r"\brd\s+/s\b|\brmdir\s+/s\b", re.I), "递归删除目录树"),
    (re.compile(r"\bdel\s+/[a-z]*[sf][a-z]*\b", re.I), "强制删除文件"),
    (
        re.compile(r"\bgit\s+push\b[^|;&]*(--force\b|(?<!\w)-f(?!\w))", re.I),
        "git 强制推送会覆盖远端历史",
    ),
    (
        re.compile(r"\bgit\s+clean\b[^|;&]*-\w*[xfd]", re.I),
        "git clean 会删除未跟踪的文件",
    ),
]

# rm 是唯一需要「看目标」才有意义的规则：`rm -rf build` 完全合法，
# `rm -rf /` 是灾难。所以不能只看有没有 -rf，还得看它删的是哪。
_FLAG_TOKEN = re.compile(r"-[A-Za-z]+")
_SPLIT_SEGMENTS = re.compile(r"[;&|]{1,2}")


def _is_catastrophic_target(token: str) -> bool:
    """rm 的目标是不是「往项目外面乱删」。

    判断依据和路径沙箱是同一套思路：项目内的相对路径安全，绝对路径、
    家目录、往外跳的 ../ 一律危险。

    注意 `./build` 是放行的 —— 它在项目内。会被拦的是单独一个 `.`
    （等于删光当前目录）和任何 `../` 开头的东西。
    """
    if token in {".", "..", "*", ".*"}:
        return True
    if token.startswith(("/", "~", "$HOME")):
        return True
    return token.startswith("../")


def _check_rm(command: str) -> str | None:
    """把命令按 ; & | 切段，逐段看是不是「rm 带 r+f 且目标危险」。"""
    for segment in _SPLIT_SEGMENTS.split(command):
        tokens = segment.split()
        if not tokens or os.path.basename(tokens[0]).lower() not in {"rm", "rm.exe"}:
            continue

        flags = [t for t in tokens[1:] if _FLAG_TOKEN.fullmatch(t)]
        targets = [t for t in tokens[1:] if not t.startswith("-")]

        # -r 和 -f 可以合成一个 -rf，也可以是分开的 -r -f，所以拼起来一起看
        letters = "".join(f.lstrip("-").lower() for f in flags)
        if "r" in letters and "f" in letters:
            if any(_is_catastrophic_target(t) for t in targets):
                return "递归强制删除项目外的路径 / 当前目录"
    return None


def check_dangerous(command: str) -> str | None:
    """命中危险规则则返回原因，否则返回 None。"""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return reason
    return _check_rm(command)


# ---------------------------------------------------------------- 工具


def find_bash() -> str | None:
    """找一个能用的 bash（Windows 上才需要）。

    为什么非找不可：Windows 上 `create_subprocess_shell` 默认走 **cmd.exe**，
    而模型（以及几乎所有 agent 的训练数据）写的是 Unix 命令 —— `grep`、`tail`、
    `head`、`cat` 在 cmd 里全都不存在。实测一条 `grep ... | head -30` 直接
    退出码 255，白烧一步，然后模型还得再花一步去绕。

    刻意**不**优先用 `shutil.which("bash")`：它可能返回
    `C:\\Windows\\System32\\bash.exe` —— 那是 WSL 的入口，跑在另一套文件系统
    视图里，拿它当 shell 只会错的更离谱。所以先认 Git 的安装路径。
    """
    if sys.platform != "win32":
        return None  # POSIX 上 /bin/sh 就是对的

    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


_BASH = find_bash()


async def _kill_tree(process: asyncio.subprocess.Process) -> None:
    """终止命令进程**及其所有子进程**，然后把管道收干净。

    为什么不能只调 process.kill()：
    `create_subprocess_shell` 起的是一个 shell，真正干活的命令是它的子进程。
    Windows 上 kill() 只杀得掉 shell 本身，子进程活着继续占着 stdout 管道 ——
    于是下一次读取会一直阻塞到它自然结束，「超时」形同虚设。
    实测：`python -c "time.sleep(10)"` 配 timeout=1，会卡满 10 秒。

    收尾的 communicate() 也不能省：少了它，管道传输没关闭，
    事件循环结束时 GC 会抛 "unclosed transport" 警告。
    """
    if process.returncode is None:
        if sys.platform == "win32":
            # taskkill 加 /T 才会沿着父子关系把整棵树杀掉
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    with contextlib.suppress(Exception):
        await process.communicate()


class BashParams(BaseModel):
    command: str = Field(description="要执行的 shell 命令")
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        ge=1,
        le=MAX_TIMEOUT,
        description=f"超时秒数，超时后进程会被强制终止，上限 {MAX_TIMEOUT}",
    )


class BashTool(SandboxedTool):
    name = "Bash"
    risk = "execute"
    description = (
        "在项目根目录下执行 shell 命令（Windows 上用 Git Bash），"
        "返回 exit code、stdout 和 stderr。"
        "适合运行测试、构建、git 等操作。不要用它做能用 Read/Glob/Grep 完成的事。"
    )
    params_model = BashParams

    def subject(self, params: BashParams) -> str:
        # 命令字符串不同于路径，命中敏感规则的可靠性有限（`cat .env` 能被
        # 通配符捞到，但换个写法就捞不到）。这里的定位是「让人看一眼」，
        # 不是「挡住」—— 挡 Bash 得靠操作系统级隔离，不是规则
        return params.command

    @staticmethod
    def _env() -> dict[str, str]:
        """子进程的环境变量。

        把 Agent 自己那个 Python 的目录放到 PATH 最前面。否则 Bash 里的 `python`
        取决于**调用者当时有没有激活 conda 环境**：从终端跑（激活过）能找到 pytest，
        从 PyCharm 跑（没激活）就 `No module named pytest`。同一句命令时好时坏，
        这种不确定性比报错本身更难查。
        """
        env = dict(os.environ)
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        return env

    async def _spawn(self, command: str) -> asyncio.subprocess.Process:
        """起一个进程跑命令。

        有 bash 就用 `bash -c`，没有才退回默认的 `create_subprocess_shell`
        （Windows 上是 cmd.exe）。
        """
        common = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": self.sandbox.root,
            "env": self._env(),
        }
        if _BASH is not None:
            return await asyncio.create_subprocess_exec(_BASH, "-c", command, **common)

        return await asyncio.create_subprocess_shell(
            command,
            # POSIX 上单开一个进程组，才能用 killpg 一次干掉整棵树
            start_new_session=sys.platform != "win32",
            **common,
        )

    async def execute(self, command: str, timeout: float) -> ToolResult:
        reason = check_dangerous(command)
        if reason:
            raise ToolError(
                f"命令被拒绝（命中危险操作：{reason}）。"
                f"如果确实是必要操作，请让用户手动执行。"
            )

        process = await self._spawn(command)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            await _kill_tree(process)
            raise ToolError(
                f"命令执行超时（超过 {timeout:g} 秒），已被强制终止。"
                f"如果是长时间任务，请拆小或提高 timeout 参数。"
            ) from None

        stdout = decode_bytes(stdout_b, prefer_utf8=False)
        stderr = decode_bytes(stderr_b, prefer_utf8=False)

        parts = [f"exit code: {process.returncode}"]
        if stdout.strip():
            parts.append(f"stdout:\n{truncate(stdout.rstrip(), MAX_OUTPUT_CHARS)}")
        if stderr.strip():
            parts.append(f"stderr:\n{truncate(stderr.rstrip(), MAX_OUTPUT_CHARS)}")
        if len(parts) == 1:
            parts.append("（无输出）")

        return ToolResult("\n".join(parts))
