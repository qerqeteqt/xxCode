"""每轮记忆提取的测试。

两块重点：
    1. 工具集**只有记忆相关的三个** —— 它在结构上就碰不到业务代码
    2. 材料只喂本轮的那点东西，不喂整个 transcript（那是烧钱）
"""

import asyncio
import json

from app.llm.content import REF_PREFIX, image_block
from app.memory.extractor import MemoryExtractor, summarize_turn
from app.memory.memory_manager import MemoryManager

NAME = "3f9a1c2b4d5e6f70.png"


def _run(coro):
    return asyncio.run(coro)


def _assistant(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(name, args, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.tools_seen = []

    async def chat(self, messages, tools=None, on_delta=None):
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self._replies:
            raise AssertionError("回复脚本用完了")
        return self._replies.pop(0)


# ================================================================ 本轮摘要


def test_只取最后一条用户发言之后的内容():
    """前面几轮的东西不该重复喂 —— 每一轮都重读全部是 O(n²) 的烧钱。"""
    history = [
        {"role": "user", "content": "上一轮的问题"},
        {"role": "assistant", "content": "上一轮的答案"},
        {"role": "user", "content": "这一轮的问题"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "content": "一大段文件内容", "tool_call_id": "c"},
        {"role": "assistant", "content": "这一轮的答案"},
    ]

    turn = summarize_turn(history)

    assert "这一轮的问题" in turn
    assert "这一轮的答案" in turn
    # 上一轮和过程噪声都不该出现
    assert "上一轮的问题" not in turn
    assert "一大段文件内容" not in turn


def test_还没答完时返回_None():
    history = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
    ]

    assert summarize_turn(history) is None


def test_带图片的一轮不会把块列表写进材料(tmp_path):
    """**这条最要紧。**

    提取出来的材料会被写进 `.agent/memory/*.md`，而那个文件之后会被注入
    **每一轮**的 system prompt。一个块列表的 repr 落进去就是长期污染，
    而且再也清不掉。图片本身对「有什么值得长期记住」也没有价值。
    """
    history = [
        {"role": "user", "content": [{"type": "text", "text": "这个报错怎么修"},
                                     image_block(NAME)]},
        {"role": "assistant", "content": "改一下配置就行"},
    ]
    extractor = MemoryExtractor(str(tmp_path), object())

    material = extractor.build_material(history)

    assert "这个报错怎么修" in material
    assert "image_url" not in material
    assert REF_PREFIX not in material


def test_只贴图不说话的一轮在材料里只留下占位符(tmp_path):
    """纯图片消息如实变成「[图片]」，而不是被丢掉。

    这里是**刻意保留占位符**的：提取器该知道「这轮有张截图」，
    只是看不到内容。要守住的是别把块列表的 repr 写进去。
    """
    history = [
        {"role": "user", "content": [image_block(NAME)]},
        {"role": "assistant", "content": "看到了"},
    ]
    extractor = MemoryExtractor(str(tmp_path), object())

    material = extractor.build_material(history)

    assert "[图片]" in material
    assert "image_url" not in material
    assert REF_PREFIX not in material


def test_没有用户发言时返回_None():
    assert summarize_turn([{"role": "assistant", "content": "自言自语"}]) is None


def test_过长的回答被截断():
    history = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "很长的回答" * 1000},
    ]

    assert "已截断" in summarize_turn(history)


# ================================================================ 材料组装


def test_材料里带上现有记忆索引(tmp_path):
    MemoryManager(tmp_path).write_memory("arch", "内容", name="项目架构")

    messages = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    material = MemoryExtractor(str(tmp_path), object()).build_material(messages)

    assert "项目架构" in material
    assert "## 现有长期记忆" in material


def test_没有可看的轮次时材料是_None(tmp_path):
    extractor = MemoryExtractor(str(tmp_path), object())

    assert extractor.build_material([{"role": "user", "content": "问题"}]) is None


# ================================================================ 工具集


def test_工具集里只有记忆相关的三个(tmp_path):
    """**结构性保证**：它在结构上就碰不到业务代码。

    比在 prompt 里叮嘱「不要修改业务代码」可靠 —— 那条是请求配合，这条是能力不存在。
    """
    llm = ScriptedLLM([_assistant("没有可提取的")])
    _run(MemoryExtractor(str(tmp_path), llm).run(
        [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    ))

    exposed = {s["function"]["name"] for s in llm.tools_seen[0]}

    assert exposed == {"ReadMemory", "WriteMemory", "UpdateMemory"}
    for forbidden in ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "SubAgent", "DeleteMemory"):
        assert forbidden not in exposed


def test_上下文只有两条消息(tmp_path):
    """和 SubAgent / AutoDream 一样：独立 Context，从零构造。"""
    llm = ScriptedLLM([_assistant("没有可提取的")])
    _run(MemoryExtractor(str(tmp_path), llm).run(
        [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    ))

    assert [m["role"] for m in llm.calls[0]] == ["system", "user"]


# ================================================================ 执行


def test_提取到东西会写入记忆(tmp_path):
    llm = ScriptedLLM([
        _assistant(tool_calls=[_tool_call("WriteMemory", {
            "key": "env-pref", "name": "环境偏好",
            "description": "用 conda 不用 venv", "category": "User",
            "content": "用户习惯用 conda 环境 langgraph。",
        })]),
        _assistant("写入了 1 条：环境偏好。"),
    ])

    result = _run(MemoryExtractor(str(tmp_path), llm).run(
        [{"role": "user", "content": "记住我用 conda"},
         {"role": "assistant", "content": "好的记住了"}]
    ))

    assert result.completed
    assert result.changed == [".agent/memory/env-pref.md"]
    # 记忆真的落盘，索引也同步了
    assert "conda" in MemoryManager(tmp_path).read_memory("env-pref").content
    assert "环境偏好" in (tmp_path / ".agent" / "memory" / "MEMORY.md").read_text(
        encoding="utf-8"
    )


def test_没有值得记的时候什么都不做(tmp_path):
    """**大多数轮次都会走到这里。** 它必须能安静地什么都不干 ——
    不然每轮往记忆目录塞一条，很快就成垃圾场。"""
    llm = ScriptedLLM([_assistant("这一轮没有值得长期保存的信息。")])

    result = _run(MemoryExtractor(str(tmp_path), llm).run(
        [{"role": "user", "content": "1+1 等于几"},
         {"role": "assistant", "content": "2"}]
    ))

    assert result.completed
    assert result.changed == []
    assert MemoryManager(tmp_path).list_memories() == []


def test_超步数标为未完成(tmp_path):
    llm = ScriptedLLM([_assistant(tool_calls=[_tool_call("ReadMemory", {"key": "不存在"})])
                       for _ in range(3)])

    result = _run(MemoryExtractor(str(tmp_path), llm, max_steps=3).run(
        [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    ))

    assert result.completed is False


def test_非法记忆_key_被回灌而不是崩溃(tmp_path):
    llm = ScriptedLLM([
        _assistant(tool_calls=[_tool_call("WriteMemory", {
            "key": "../../逃逸", "name": "坏", "content": "x",
        })]),
        _assistant("那个 key 不合法，我不写它了。"),
    ])

    result = _run(MemoryExtractor(str(tmp_path), llm).run(
        [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    ))

    assert result.completed
    assert result.changed == []
    assert not (tmp_path / "逃逸.md").exists()
