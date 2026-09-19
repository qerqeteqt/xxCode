"""Phase 3 SubAgent 测试。

重点不是「功能能不能跑」，而是三条**结构性保证**测出没被测出来：

    1. 工具集裁剪   —— 只读规格真的拿不到写类工具
    2. 递归禁止     —— 任何规格的工具集里都没有 SubAgent
    3. Context 隔离 —— 子 Agent 的第一条消息就是 system+user，没有 Main 的历史

这三条都是「能力不存在」，不是「prompt 请求它别用」。测试要锁的正是这一点。
"""

import asyncio
import json

from app.agent.subagent import AGENT_SPECS, AgentSpec
from app.tools.sandbox import Sandbox
from app.tools.subagent_tool import SubAgentTool, build_sub_registry


def _run(coro):
    return asyncio.run(coro)


def _tool_call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


def _assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class ScriptedLLM:
    """按脚本回复，并记录每次调用看到的 messages 和 tools。"""

    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []
        self.tools_seen: list = []

    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self._replies:
            raise AssertionError("回复脚本用完了 —— 循环比预期多跑了一轮")
        return self._replies.pop(0)


def _make_project(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "def login(user):\n    return user\n", encoding="utf-8"
    )
    return tmp_path


# ================================================================ 工具集裁剪


def test_explore_只拿得到只读工具(tmp_path):
    sandbox = Sandbox(_make_project(tmp_path))
    names = {t.name for t in build_sub_registry(AGENT_SPECS["Explore"], sandbox).list_tools()}

    assert names == {"Read", "List", "Glob", "Grep"}
    # 关键：不是「有但被禁用」，而是压根没有
    assert "Write" not in names
    assert "Edit" not in names
    assert "Bash" not in names


def test_plan_和_explore_工具集相同(tmp_path):
    sandbox = Sandbox(_make_project(tmp_path))
    explore = {t.name for t in build_sub_registry(AGENT_SPECS["Explore"], sandbox).list_tools()}
    plan = {t.name for t in build_sub_registry(AGENT_SPECS["Plan"], sandbox).list_tools()}

    assert explore == plan


def test_general_purpose_能改代码也能跑命令(tmp_path):
    sandbox = Sandbox(_make_project(tmp_path))
    names = {
        t.name
        for t in build_sub_registry(AGENT_SPECS["General-Purpose"], sandbox).list_tools()
    }

    assert {"Read", "Write", "Edit", "Bash"} <= names


def test_任何规格的子agent都不能再派子agent(tmp_path):
    """递归禁止靠工具集里根本没有 SubAgent 实现，不靠深度计数。"""
    sandbox = Sandbox(_make_project(tmp_path))

    for name, spec in AGENT_SPECS.items():
        names = {t.name for t in build_sub_registry(spec, sandbox).list_tools()}
        assert "SubAgent" not in names, f"{name} 的工具集里不该有 SubAgent"


# ================================================================ Context 隔离


def test_子agent看不到_main_的历史(tmp_path):
    llm = ScriptedLLM([_assistant("登录逻辑在 app/main.py:1")])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    _run(tool.execute(agent_type="Explore", task="找出登录相关代码", context=None))

    first_call = llm.calls[0]
    # 只有两条：system + user。Main 前面聊了什么，这里一概没有
    assert [m["role"] for m in first_call] == ["system", "user"]
    assert first_call[1]["content"] == "找出登录相关代码"
    assert "Explore" in first_call[0]["content"]


def test_context_被拼进任务描述(tmp_path):
    llm = ScriptedLLM([_assistant("好")])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    _run(
        tool.execute(
            agent_type="Explore",
            task="找出登录相关代码",
            context="入口确认在 app/api.py，tests/ 已经排查过了",
        )
    )

    user_message = llm.calls[0][1]["content"]
    assert "找出登录相关代码" in user_message
    assert "入口确认在 app/api.py" in user_message


def test_子agent只看到自己规格内的工具_schema(tmp_path):
    llm = ScriptedLLM([_assistant("好")])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    _run(tool.execute(agent_type="Explore", task="看看", context=None))

    # 模型连 Write 存在都不知道 —— 这比在 prompt 里写「不许写文件」可靠得多
    exposed = {s["function"]["name"] for s in llm.tools_seen[0]}
    assert exposed == {"Read", "List", "Glob", "Grep"}


# ================================================================ 返回格式


def test_返回格式包含类型和步数(tmp_path):
    llm = ScriptedLLM([_assistant("找到三处登录相关代码")])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    result = _run(tool.execute(agent_type="Explore", task="找登录", context=None)).text

    assert result.startswith("[Explore 完成 | 1 步]")
    assert "找到三处登录相关代码" in result


def test_成功的写入进入改动清单(tmp_path):
    project = _make_project(tmp_path)
    llm = ScriptedLLM(
        [
            _assistant(tool_calls=[_tool_call("Write", {"path": "new.py", "content": "X = 1\n"})]),
            _assistant("已创建 new.py"),
        ]
    )
    tool = SubAgentTool(Sandbox(project), llm)

    result = _run(
        tool.execute(agent_type="General-Purpose", task="建个文件", context=None)
    ).text

    assert "修改: new.py" in result
    assert (project / "new.py").exists()


def test_失败的写入不进改动清单(tmp_path):
    """Edit 因为找不到原文而失败时，文件没动，清单里就不该有它。"""
    project = _make_project(tmp_path)
    llm = ScriptedLLM(
        [
            _assistant(
                tool_calls=[
                    _tool_call(
                        "Edit",
                        {
                            "path": "app/main.py",
                            "old_string": "根本不存在的字符串",
                            "new_string": "x",
                        },
                    )
                ]
            ),
            _assistant("改动没能完成"),
        ]
    )
    tool = SubAgentTool(Sandbox(project), llm)

    result = _run(
        tool.execute(agent_type="General-Purpose", task="改点东西", context=None)
    ).text

    assert "未修改任何文件" in result
    assert (project / "app" / "main.py").read_text(encoding="utf-8") == (
        "def login(user):\n    return user\n"
    )


def test_没改文件的_general_purpose_明确标注(tmp_path):
    llm = ScriptedLLM([_assistant("看了一圈，不用改")])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    result = _run(
        tool.execute(agent_type="General-Purpose", task="看看", context=None)
    ).text

    assert "未修改任何文件" in result


# ================================================================ 超步数


def test_超步数时把中间结论带回来(tmp_path, monkeypatch):
    """白扔掉已查到的东西，等于让调用方白付一次 token。"""
    monkeypatch.setitem(
        AGENT_SPECS,
        "Explore",
        AgentSpec(
            name="Explore",
            system_prompt="只读探索",
            allowed_tools=frozenset({"List"}),
            max_steps=2,
        ),
    )
    llm = ScriptedLLM(
        [
            _assistant("我先看一下目录结构", tool_calls=[_tool_call("List", {"path": "."})]),
            _assistant(tool_calls=[_tool_call("List", {"path": "app"})]),
        ]
    )
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    result = _run(tool.execute(agent_type="Explore", task="看看项目", context=None)).text

    assert "未完成" in result
    assert "2 步上限" in result
    assert "我先看一下目录结构" in result


def test_超步数且无中间结论时如实说明(tmp_path, monkeypatch):
    monkeypatch.setitem(
        AGENT_SPECS,
        "Explore",
        AgentSpec(
            name="Explore",
            system_prompt="只读探索",
            allowed_tools=frozenset({"List"}),
            max_steps=1,
        ),
    )
    llm = ScriptedLLM([_assistant(tool_calls=[_tool_call("List", {"path": "."})])])
    tool = SubAgentTool(Sandbox(_make_project(tmp_path)), llm)

    result = _run(tool.execute(agent_type="Explore", task="看看", context=None)).text

    assert "没有任何中间结论" in result
