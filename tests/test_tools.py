"""Phase 2 工具层测试。

覆盖四块：路径沙箱、ToolRegistry 的四道关卡、各工具的正常/异常路径、
危险命令识别。除了 Bash 的执行用例，其余全部不碰网络、不碰外部依赖。
"""

import asyncio
import json
import sys

import pytest
from pydantic import BaseModel

from app.tools import build_default_registry
from app.tools.bash_tool import check_dangerous
from app.tools.base import Tool, ToolError, ToolResult
from app.tools.file_tool import EditTool, ReadTool, WriteTool
from app.tools.registry import TrackingRegistry
from app.tools.sandbox import PathOutOfSandboxError, Sandbox


def _run(coro):
    return asyncio.run(coro)


def _call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    """造一个 API 形状的 tool_call。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


# ================================================================ fixtures


@pytest.fixture
def project(tmp_path):
    """一个假项目，包含会被跳过的目录，用来验证遍历规则。"""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "def login(user):\n    return user\n", encoding="utf-8"
    )
    (tmp_path / "app" / "utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import app\n", encoding="utf-8")

    # 这些目录必须被遍历跳过
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("login = 1\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("login in git\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sandbox(project):
    return Sandbox(project)


@pytest.fixture
def registry(project):
    return build_default_registry(project)


# ================================================================ 路径沙箱


def test_sandbox_允许项目内的相对路径(sandbox, project):
    assert sandbox.resolve("app/main.py") == (project / "app" / "main.py").resolve()


def test_sandbox_允许项目内的绝对路径(sandbox, project):
    target = str(project / "app" / "main.py")
    assert sandbox.resolve(target) == (project / "app" / "main.py").resolve()


def test_sandbox_拒绝上跳越界(sandbox):
    with pytest.raises(PathOutOfSandboxError):
        sandbox.resolve("../../../Windows/System32/drivers/etc/hosts")


def test_sandbox_拒绝项目外的绝对路径(sandbox):
    with pytest.raises(PathOutOfSandboxError):
        sandbox.resolve("D:/pycharm/Mutil-Agent/src/main.py")


def test_sandbox_允许原地绕一圈的路径(sandbox, project):
    """`app/../app/main.py` 里有 .. 但没出圈，必须放行 ——
    这正是「不能只检查字符串里有没有 ..」的原因。"""
    assert sandbox.resolve("app/../app/main.py") == (project / "app" / "main.py").resolve()


def test_sandbox_rel_返回相对路径(sandbox, project):
    assert sandbox.rel(project / "app" / "main.py") == "app/main.py"


# ================================================================ Registry


class _EchoParams(BaseModel):
    text: str


class _EchoTool(Tool):
    name = "Echo"
    description = "回显"
    params_model = _EchoParams

    async def execute(self, text: str) -> ToolResult:
        return ToolResult(f"echo: {text}")


class _FailTool(Tool):
    name = "Fail"
    description = "预期内失败"
    params_model = _EchoParams

    async def execute(self, text: str) -> ToolResult:
        raise ToolError("路径不存在: 没有这个文件")


class _CrashTool(Tool):
    name = "Crash"
    description = "预期外崩溃"
    params_model = _EchoParams

    async def execute(self, text: str) -> ToolResult:
        raise ValueError("我自己的 bug")


@pytest.fixture
def echo_registry():
    reg = build_default_registry(".")
    reg.unregister("Bash")
    reg.register(_EchoTool())
    reg.register(_FailTool())
    reg.register(_CrashTool())
    return reg


def test_registry_未知工具名给出可用列表(echo_registry):
    result = _run(echo_registry.execute(_call("Nope")))
    assert "没有名为 'Nope' 的工具" in result
    assert "Echo" in result  # 列出来了，模型能自己改对


def test_registry_arguments_不是合法_JSON(echo_registry):
    bad = {"id": "1", "function": {"name": "Echo", "arguments": "{不是 json"}}
    result = _run(echo_registry.execute(bad))
    assert "不是合法 JSON" in result


def test_registry_arguments_不是对象(echo_registry):
    bad = {"id": "1", "function": {"name": "Echo", "arguments": '"just a string"'}}
    result = _run(echo_registry.execute(bad))
    assert "必须是 JSON 对象" in result


def test_registry_参数校验失败指出字段(echo_registry):
    result = _run(echo_registry.execute(_call("Echo", {"wrong_field": 1})))
    assert "参数不合法" in result
    assert "text" in result  # 缺哪个字段要说清楚


def test_registry_正常执行(echo_registry):
    assert _run(echo_registry.execute(_call("Echo", {"text": "hi"}))) == "echo: hi"


def test_registry_ToolError_不带内部异常前缀(echo_registry):
    """预期内失败要说人话，不要暴露 Python 异常类型。"""
    result = _run(echo_registry.execute(_call("Fail", {"text": "x"})))
    assert result == "Fail: 路径不存在: 没有这个文件"


def test_registry_预期外异常标注为内部异常(echo_registry):
    """预期外失败是 Runtime 自己的 bug，措辞要和上面区分开。"""
    result = _run(echo_registry.execute(_call("Crash", {"text": "x"})))
    assert "内部异常" in result
    assert "ValueError" in result


def test_registry_重名注册报错(echo_registry):
    with pytest.raises(ValueError, match="工具名重复"):
        echo_registry.register(_EchoTool())


def test_registry_schemas_可直接给_LLM(echo_registry):
    schemas = echo_registry.schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "Echo" in names
    echo_schema = next(s for s in schemas if s["function"]["name"] == "Echo")
    assert echo_schema["type"] == "function"
    # schema 由 Pydantic 生成，参数名必须和 execute 的形参对得上
    assert "text" in echo_schema["function"]["parameters"]["properties"]


# ================================================================ Read


def test_read_返回带行号的内容(registry):
    result = _run(registry.execute(_call("Read", {"path": "app/main.py"})))
    assert "1\tdef login(user):" in result
    assert "2\t    return user" in result


def test_read_分段读取时提示还有多少没读(registry):
    result = _run(registry.execute(_call("Read", {"path": "app/main.py", "limit": 1})))
    assert "1\tdef login(user):" in result
    assert "共 2 行" in result
    assert "offset=2" in result


def test_read_offset_越界报错(registry):
    result = _run(registry.execute(_call("Read", {"path": "app/main.py", "offset": 99})))
    assert "超出范围" in result


def test_read_文件不存在(registry):
    result = _run(registry.execute(_call("Read", {"path": "app/nope.py"})))
    assert "文件不存在" in result


def test_read_目标是目录(registry):
    result = _run(registry.execute(_call("Read", {"path": "app"})))
    assert "是目录不是文件" in result


def test_read_越界被拒(registry):
    result = _run(registry.execute(_call("Read", {"path": "../../etc/hosts"})))
    assert "路径越界" in result


# ================================================================ Write


def test_write_新建文件并自动建父目录(registry, project):
    result = _run(
        registry.execute(_call("Write", {"path": "new/dir/a.py", "content": "X = 1\n"}))
    )
    assert "已新建" in result
    assert (project / "new" / "dir" / "a.py").read_text(encoding="utf-8") == "X = 1\n"


def test_write_覆盖已有文件(registry, project):
    result = _run(
        registry.execute(_call("Write", {"path": "README.md", "content": "# 新的\n"}))
    )
    assert "已覆盖" in result
    assert (project / "README.md").read_text(encoding="utf-8") == "# 新的\n"


# ================================================================ Edit


def test_edit_精确替换(registry, project):
    result = _run(
        registry.execute(
            _call(
                "Edit",
                {
                    "path": "app/main.py",
                    "old_string": "def login(user):",
                    "new_string": "def login(username):",
                },
            )
        )
    )
    assert "已修改" in result
    assert (project / "app" / "main.py").read_text(encoding="utf-8").startswith(
        "def login(username):"
    )


def test_edit_找不到原文时提示要精确匹配(registry):
    result = _run(
        registry.execute(
            _call("Edit", {"path": "app/main.py", "old_string": "def nope", "new_string": "x"})
        )
    )
    assert "找不到 old_string" in result


def test_edit_出现多次时拒绝执行(registry, project):
    """不唯一时必须拒绝，否则会把不该改的地方也改了。"""
    (project / "dup.py").write_text("X = 1\nX = 1\n", encoding="utf-8")
    result = _run(
        registry.execute(
            _call("Edit", {"path": "dup.py", "old_string": "X = 1", "new_string": "X = 2"})
        )
    )
    assert "出现了 2 次" in result
    # 关键：文件必须原样不动
    assert (project / "dup.py").read_text(encoding="utf-8") == "X = 1\nX = 1\n"


def test_edit_新旧相同时拒绝(registry):
    result = _run(
        registry.execute(
            _call("Edit", {"path": "app/main.py", "old_string": "X", "new_string": "X"})
        )
    )
    assert "没有任何改动" in result


# ================================================================ List


def test_list_递归列出文件和目录(registry):
    result = _run(registry.execute(_call("List", {"path": "."})))
    assert "app/" in result
    assert "app/main.py" in result
    assert "README.md" in result


def test_list_跳过_pycache_和_git(registry):
    result = _run(registry.execute(_call("List", {"path": "."})))
    assert "__pycache__" not in result
    assert ".git" not in result


# ================================================================ Glob


def test_glob_双星号能匹配到根目录下的文件(registry):
    """`**/*.py` 应该同时匹配 main.py 和 app/main.py ——
    pathlib 原生行为匹配不到根目录那个，这里补了一次。"""
    result = _run(registry.execute(_call("Glob", {"pattern": "**/*.py"})))
    assert "main.py" in result
    assert "app/main.py" in result


def test_glob_限定目录(registry):
    result = _run(registry.execute(_call("Glob", {"pattern": "app/*.py"})))
    assert "app/main.py" in result
    assert "app/utils.py" in result
    assert "README.md" not in result


def test_glob_跳过被忽略的目录(registry):
    result = _run(registry.execute(_call("Glob", {"pattern": "**/*.py"})))
    assert "__pycache__" not in result
    assert ".git" not in result


# ================================================================ Grep


def test_grep_返回文件行号和内容(registry):
    result = _run(registry.execute(_call("Grep", {"pattern": "def login"})))
    assert "app/main.py:1:" in result
    assert "def login(user)" in result


def test_grep_跳过被忽略的目录(registry):
    result = _run(registry.execute(_call("Grep", {"pattern": "login"})))
    assert "__pycache__" not in result
    assert ".git" not in result


def test_grep_include_限定文件类型(registry):
    result = _run(registry.execute(_call("Grep", {"pattern": "demo", "include": "*.md"})))
    assert "README.md" in result
    result_py = _run(
        registry.execute(_call("Grep", {"pattern": "demo", "include": "*.py"}))
    )
    assert "没有匹配" in result_py


def test_grep_非法正则(registry):
    result = _run(registry.execute(_call("Grep", {"pattern": "([unclosed"})))
    assert "正则表达式不合法" in result


def test_grep_跳过二进制文件(registry, project):
    (project / "blob.bin").write_bytes(b"\x00\x01\x02\xff\xfe login \x80\x81")
    result = _run(registry.execute(_call("Grep", {"pattern": "login"})))
    assert "blob.bin" not in result


# ================================================================ Bash 危险命令识别


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf $HOME",
        "rm -rf ..",
        "rm -rf .",
        "rm -r -f /",
        "rm -rf /usr/local",
        "rm -rf ../other-project",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
        "shutdown -h now",
        "reboot",
        "git push --force origin main",
        "git push -f",
        "git clean -xfd",
    ],
)
def test_危险命令被识别(command):
    assert check_dangerous(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -rf ./build",
        "rm -rf node_modules",
        "rm -f a.txt",
        "ls -la",
        "python -m pytest",
        "git push origin main",
        "git status",
        "git clean -n",  # 只预览不删，放行
        "git log --format=oneline",
    ],
)
def test_安全命令不该被误伤(command):
    assert check_dangerous(command) is None


# ================================================================ Bash 执行


def test_bash_正常执行返回退出码和输出(registry):
    result = _run(registry.execute(_call("Bash", {"command": "echo hello"})))
    assert "exit code: 0" in result
    assert "hello" in result


def test_bash_非零退出码原样返回(registry):
    command = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    result = _run(registry.execute(_call("Bash", {"command": command})))
    assert "exit code: 3" in result


def test_bash_超时被强制终止(registry):
    command = f'"{sys.executable}" -c "import time; time.sleep(10)"'
    result = _run(registry.execute(_call("Bash", {"command": command, "timeout": 1})))
    assert "超时" in result


def test_bash_危险命令不执行(registry):
    result = _run(registry.execute(_call("Bash", {"command": "rm -rf /"})))
    assert "被拒绝" in result


# ================================================================ ToolResult 与 TrackingRegistry


def test_execute_detailed_返回结构化结果(project):
    registry = TrackingRegistry()
    registry.register(ReadTool(Sandbox(project)))

    result = _run(registry.execute_detailed(_call("Read", {"path": "app/main.py"})))

    assert result.ok is True
    assert "def login" in result.text
    assert result.changed_path is None


def test_execute_detailed_失败时_ok_为_False(project):
    """成败不再靠嗅探文本 —— 四道关卡失败和工具失败统一走 ok=False。"""
    registry = TrackingRegistry()
    registry.register(ReadTool(Sandbox(project)))

    result = _run(registry.execute_detailed(_call("Read", {"path": "不存在.py"})))

    assert result.ok is False
    assert "文件不存在" in result.text


def test_execute_仍然只返回字符串(project):
    """给模型看的那一面没变：registry.execute 还是返回 str。"""
    registry = TrackingRegistry()
    registry.register(ReadTool(Sandbox(project)))

    result = _run(registry.execute(_call("Read", {"path": "app/main.py"})))

    assert isinstance(result, str)


def test_TrackingRegistry_记录成功的写入(project):
    registry = TrackingRegistry()
    registry.register(WriteTool(Sandbox(project)))

    _run(registry.execute(_call("Write", {"path": "a.py", "content": "X = 1\n"})))

    assert registry.changed_files == ["a.py"]


def test_TrackingRegistry_失败的写入不记录(project):
    """Edit 找不到原文时文件没动，不该进清单。"""
    registry = TrackingRegistry()
    registry.register(EditTool(Sandbox(project)))

    _run(
        registry.execute(
            _call(
                "Edit",
                {"path": "app/main.py", "old_string": "不存在", "new_string": "x"},
            )
        )
    )

    assert registry.changed_files == []


def test_TrackingRegistry_on_change_每个文件只回调一次(project):
    seen: list[str] = []
    registry = TrackingRegistry(on_change=seen.append)
    registry.register(WriteTool(Sandbox(project)))

    _run(registry.execute(_call("Write", {"path": "a.py", "content": "1\n"})))
    _run(registry.execute(_call("Write", {"path": "a.py", "content": "2\n"})))

    assert seen == ["a.py"]
