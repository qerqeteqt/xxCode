"""Phase 7c 权限闸门测试。

三层判定各测一遍，重点是顺序：allow 必须盖过 deny，deny 必须硬拒（不问人），
write/execute 才落到「问」。
"""

import asyncio
import json

import pytest

from app.tools import build_default_registry
from app.tools.permission import (
    PERMISSIONS_FILE,
    SENSITIVE_PATTERNS,
    Decision,
    PermissionGate,
    load_rules,
    matches_any,
)
from app.tools.sandbox import Sandbox


def _run(coro):
    return asyncio.run(coro)


def _call(name: str, args: dict | None = None, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


class FakeConfirmer:
    """按预置答案回复，并记下被问了什么。"""

    def __init__(self, *decisions: Decision) -> None:
        self._decisions = list(decisions) or [Decision.DENY]
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        if len(self._decisions) > 1:
            return self._decisions.pop(0)
        return self._decisions[0]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("LLM_API_KEY=sk-secret\n", encoding="utf-8")
    (tmp_path / "src" / ".env").write_text("NESTED=1\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("LLM_API_KEY=sk-your-key\n", encoding="utf-8")
    return tmp_path


def _registry(project, confirmer=None, **gate_kwargs):
    gate = PermissionGate.for_project(project, confirmer=confirmer, **gate_kwargs)
    return build_default_registry(project, gate=gate)


# ================================================================ 工具声明的风险等级


def test_每个工具都声明了风险等级(project):
    """默认值是 execute（最严），所以忘了声明会被「问」而不是被静默放行。
    但仍然要显式核对一遍，避免有人靠默认值蒙混过去。"""
    registry = _registry(project)  # llm=None，SubAgent 不注册

    declared = {t.name: t.risk for t in registry.list_tools()}

    assert declared == {
        "Read": "read",
        "Write": "write",
        "Edit": "write",
        "List": "read",
        "Glob": "read",
        "Grep": "read",
        "Bash": "execute",
    }


def test_记忆工具的风险等级(project):
    from app.memory.memory_manager import MemoryManager
    from app.memory.memory_tools import memory_tools

    declared = {t.name: t.risk for t in memory_tools(MemoryManager(project))}

    assert declared == {
        "ReadMemory": "read",
        "WriteMemory": "write",
        "UpdateMemory": "write",
        "DeleteMemory": "write",
    }


# ================================================================ 一层：read 不问


def test_读操作直接放行不问(project):
    confirmer = FakeConfirmer(Decision.DENY)
    registry = _registry(project, confirmer)

    result = _run(registry.execute(_call("Read", {"path": "src/calc.py"})))

    assert result == "1\tX = 1"
    assert confirmer.requests == []  # 一次都没问


# ================================================================ 二层：敏感路径硬拒


@pytest.mark.parametrize(
    "path", [".env", "src/.env", "id_rsa", "cert.pem", "server.key"]
)
def test_敏感文件被拒绝且不问人(project, path):
    """规则的意义就是「不用问」。命中规则还弹窗的话，规则和默认策略没区别。"""
    for name in ("id_rsa", "cert.pem", "server.key"):
        (project / name).write_text("secret", encoding="utf-8")
    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)
    registry = _registry(project, confirmer)

    result = _run(registry.execute(_call("Read", {"path": path})))

    assert "拒绝访问" in result
    assert "allow 列表" in result  # 消息里要说明怎么放开
    assert confirmer.requests == []  # 没问


def test_写敏感文件同样被拒(project):
    registry = _registry(project, FakeConfirmer(Decision.ALLOW_ONCE))

    result = _run(
        registry.execute(_call("Write", {"path": ".env", "content": "偷改"}))
    )

    assert "拒绝访问" in result
    assert (project / ".env").read_text(encoding="utf-8") == "LLM_API_KEY=sk-secret\n"


def test_不在清单里的_env_变体不受影响(project):
    registry = _registry(project, FakeConfirmer(Decision.DENY))

    result = _run(registry.execute(_call("Read", {"path": ".env.example"})))

    assert "LLM_API_KEY=sk-your-key" in result


def test_搜索看不到敏感文件(project):
    """敏感文件在**遍历层**就被摘掉 —— 规则没法表达「这次 Grep 会扫到哪些文件」。"""
    registry = _registry(project, FakeConfirmer(Decision.DENY))

    found = _run(registry.execute(_call("Grep", {"pattern": "LLM_API_KEY"})))
    listed = _run(registry.execute(_call("List", {"path": "."})))

    # 按完整路径判断 —— 用 `".env" not in found` 会被 .env.example 的子串骗到
    searched = {line.split(":", 1)[0] for line in found.splitlines()}
    assert ".env" not in searched
    assert "src/.env" not in searched
    assert ".env.example" in searched  # 模板不算敏感，照常能搜

    listed_paths = set(listed.splitlines())
    assert ".env" not in listed_paths
    assert ".env.example" in listed_paths


# ================================================================ 三层：write/execute 问人


def test_写操作会问人(project):
    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)
    registry = _registry(project, confirmer)

    result = _run(
        registry.execute(_call("Write", {"path": "new.py", "content": "Y = 2\n"}))
    )

    assert len(confirmer.requests) == 1
    assert confirmer.requests[0].risk == "write"
    assert confirmer.requests[0].subject == "new.py"
    assert "已新建" in result


def test_执行命令会问人(project):
    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)
    registry = _registry(project, confirmer)

    result = _run(registry.execute(_call("Bash", {"command": "echo hi"})))

    assert len(confirmer.requests) == 1
    assert confirmer.requests[0].risk == "execute"
    assert "hi" in result


def test_拒绝后把原因回灌给模型(project):
    registry = _registry(project, FakeConfirmer(Decision.DENY))

    result = _run(
        registry.execute(_call("Write", {"path": "new.py", "content": "Y = 2\n"}))
    )

    assert "用户拒绝" in result
    assert "换个做法" in result
    assert not (project / "new.py").exists()


def test_允许一次只对这一次有效(project):
    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)
    registry = _registry(project, confirmer)

    _run(registry.execute(_call("Write", {"path": "a.py", "content": "1\n"})))
    _run(registry.execute(_call("Write", {"path": "b.py", "content": "2\n"})))

    assert len(confirmer.requests) == 2  # 两次都问了


def test_本会话允许之后同类不再问(project):
    confirmer = FakeConfirmer(Decision.ALLOW_SESSION)
    registry = _registry(project, confirmer)

    _run(registry.execute(_call("Write", {"path": "a.py", "content": "1\n"})))
    _run(registry.execute(_call("Write", {"path": "b.py", "content": "2\n"})))

    assert len(confirmer.requests) == 1  # 只问了第一次


def test_会话放行不跨风险等级(project):
    """放行了「写文件」不等于也放行「执行命令」。"""
    confirmer = FakeConfirmer(Decision.ALLOW_SESSION)
    registry = _registry(project, confirmer)

    _run(registry.execute(_call("Write", {"path": "a.py", "content": "1\n"})))
    _run(registry.execute(_call("Bash", {"command": "echo hi"})))

    assert [r.risk for r in confirmer.requests] == ["write", "execute"]


def test_没有人可问时拒绝(project):
    """宁可不做，也不要静默地做。"""
    registry = _registry(project, confirmer=None)

    result = _run(registry.execute(_call("Bash", {"command": "echo hi"})))

    assert "没有人工确认通道" in result


# ================================================================ allow 覆盖 deny


def test_allow_规则能盖过内置敏感清单(project):
    """「显式放开」的唯一入口 —— 否则想让 Agent 读一次 .env 就只能改源码。"""
    (project / ".agent").mkdir()
    (project / PERMISSIONS_FILE).write_text(
        json.dumps({"allow": [".env"]}), encoding="utf-8"
    )
    registry = _registry(project, FakeConfirmer(Decision.DENY))

    result = _run(registry.execute(_call("Read", {"path": ".env"})))

    assert "sk-secret" in result


def test_配置文件能追加_deny(project):
    (project / ".agent").mkdir()
    (project / PERMISSIONS_FILE).write_text(
        json.dumps({"deny": ["src/**"]}), encoding="utf-8"
    )
    registry = _registry(project, FakeConfirmer(Decision.DENY))

    result = _run(registry.execute(_call("Read", {"path": "src/calc.py"})))

    assert "拒绝访问" in result


def test_配置文件坏掉时退回内置规则(project):
    """坏掉的配置不该让人完全失去保护。"""
    (project / ".agent").mkdir()
    (project / PERMISSIONS_FILE).write_text("{不是 json", encoding="utf-8")

    deny, allow = load_rules(project)

    assert list(deny) == list(SENSITIVE_PATTERNS)
    assert allow == []


def test_没有配置文件时用内置规则(project):
    deny, allow = load_rules(project)

    assert ".env" in deny
    assert allow == []


# ================================================================ 匹配规则


@pytest.mark.parametrize(
    "subject,expected",
    [
        (".env", True),
        ("subdir/.env", True),  # 路径在哪一层都算
        (".env.example", False),
        ("src/calc.py", False),
        ("keys/id_rsa", True),
        ("private.pem", True),
    ],
)
def test_匹配同时看整条路径和文件名(subject, expected):
    assert matches_any(subject, SENSITIVE_PATTERNS) is expected


def test_Bash_命令命中不了路径规则(project):
    """这是明知不补的洞：命令字符串不是路径，规则命中不可靠。
    这条测试记录的**不是**「安全」，而是「这里挡不住」。"""
    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)
    registry = _registry(project, confirmer)

    _run(registry.execute(_call("Bash", {"command": "cat .env"})))

    # 没被规则拦，走的是「问人」这条路 —— 而这正是 7c 选择档 A 的含义
    assert len(confirmer.requests) == 1
    assert confirmer.requests[0].risk == "execute"


# ================================================================ SubAgent 子闸门


def test_subagent_共用同一份会话放行记录(project):
    """子 Agent 内部每次写都要过闸门，但不该重复建一份记录。"""
    parent = PermissionGate.for_project(project, confirmer=FakeConfirmer())

    child = parent.with_source("SubAgent: Explore")
    child._grants.add("write")

    assert "write" in parent._grants  # 共享的是同一个 set

    both = PermissionGate.for_project(project, confirmer=FakeConfirmer())
    assert "write" not in both._grants  # 另一份闸门不受影响


def test_子闸门的来源标签会带到请求里(project):
    """用户看到一条光秃秃的「想写入 x.py」，不会知道自己刚才在问哪件事。"""
    from app.tools.file_tool import WriteTool

    confirmer = FakeConfirmer(Decision.ALLOW_ONCE)

    async def _go():
        parent = PermissionGate.for_project(project, confirmer=confirmer)
        child = parent.with_source("SubAgent: General-Purpose")
        return await child.check(WriteTool(Sandbox(project)), "a.py")

    outcome = _run(_go())

    assert outcome.allowed
    assert confirmer.requests[0].source == "SubAgent: General-Purpose"
    assert "来自 SubAgent: General-Purpose" in confirmer.requests[0].summary


def test_subagent_工具本身标为_read(project):
    """它内部每次写都会过子闸门，在外层再问一遍等于问两次。"""
    from app.agent.subagent import AGENT_SPECS
    from app.tools.subagent_tool import SubAgentTool

    assert SubAgentTool.risk == "read"
    assert "Explore" in AGENT_SPECS  # 顺带确认 import 没坏
