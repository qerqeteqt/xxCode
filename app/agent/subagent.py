"""SubAgent 规格定义。

Explore / Plan / General-Purpose 听起来是三个东西，实现上其实只是**三份配置** ——
它们的差别只有两样：**能用哪些工具** 和 **system prompt**。

所以这里没有三个类，只有一个 dataclass 加一张表。新增一种 SubAgent 是加一条配置，
不是加一个类。

**贯穿全项目的权限原则**：能力限制靠「不注册这个工具」实现，而不是在 prompt 里
请求模型自律。prompt 是建议，工具集是事实 —— 前者可能被绕开，后者绕不开。
"""

from dataclasses import dataclass
from typing import Literal

# 只读工具集。
# 注意 Bash **不在**里面 —— 虽然 `ls`、`git log` 都是只读用法，但 Bash 能跑任意
# 命令，也就意味着它能写文件、能删文件。「常用作只读」不等于「只能只读」，
# 权限要按最大能力算，不能按典型用法算。
READ_ONLY_TOOLS = frozenset({"Read", "List", "Glob", "Grep"})

# 全量工具集。
# 注意这里也没有 SubAgent —— 子 Agent 不能再派子 Agent。
# 这不是靠深度计数拦住的，而是工具集里压根没有这个能力（见 subagent_tool.py）。
FULL_TOOLS = frozenset({"Read", "Write", "Edit", "List", "Glob", "Grep", "Bash"})

# 和下面的 AGENT_SPECS 必须保持一致 —— 两者刻意写在同一个文件里，改的时候一眼能看到
AgentType = Literal["Explore", "Plan", "General-Purpose"]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    system_prompt: str
    allowed_tools: frozenset[str]
    max_steps: int


_EXPLORE_PROMPT = """你是 Explore SubAgent，负责只读的分析与审阅。

你的职责：
- 找出相关代码分布在哪些文件、哪些函数，说明它们在项目里怎么被组织和调用
- 审阅一批已知路径的文件，逐个核对并汇总：哪些符合、哪些不符合、为什么

约束：
- 你没有任何修改文件的工具，不要尝试写代码或执行命令
- 只报告你**实际读到**的内容。没找到就说没找到，不要用一般经验补全
- 结论要具体到 文件:行号，让调用方能直接跳过去核对

输出要求：直接给结论。不要复述你搜索和阅读的过程 —— 那个过程对你没用，
对调用方也只是噪声。"""

_PLAN_PROMPT = """你是 Plan SubAgent，负责只读的研究与方案规划。

你的职责：
- 基于实际读到的代码，给出可执行的修改方案
- 说清楚改哪些文件、每个文件改什么、为什么这么改
- 主动指出风险点和你不确定的地方

约束：
- 你没有任何修改文件的工具，不要尝试写代码或执行命令
- 方案必须建立在你**实际读到**的代码上，不要凭一般经验臆测项目结构
- 如果信息不足以给出方案，直接说明还缺什么，不要硬编一个方案出来

输出要求：分步骤的方案 + 风险点。不要复述搜索过程。"""

_GENERAL_PROMPT = """你是 General-Purpose SubAgent，负责执行复杂的多步骤任务。

和 Explore / Plan 不同，你可以修改代码、执行命令。

你的职责：
- 完成交给你的任务，必要时修改代码并验证结果

约束：
- 改动前先用 Read 确认原文，不要凭印象改
- 改动后尽量运行测试或命令验证，不要改完就宣布成功
- 所有路径相对项目根目录，不要访问项目外的文件
- 完成后如实报告：改了什么、验证结果如何、有什么没做成

输出要求：先给结论，再列改动和验证结果。不要复述中间的搜索过程。"""

AGENT_SPECS: dict[AgentType, AgentSpec] = {
    "Explore": AgentSpec(
        name="Explore",
        system_prompt=_EXPLORE_PROMPT,
        allowed_tools=READ_ONLY_TOOLS,
        max_steps=8,
    ),
    "Plan": AgentSpec(
        name="Plan",
        system_prompt=_PLAN_PROMPT,
        allowed_tools=READ_ONLY_TOOLS,
        max_steps=8,
    ),
    "General-Purpose": AgentSpec(
        name="General-Purpose",
        system_prompt=_GENERAL_PROMPT,
        allowed_tools=FULL_TOOLS,
        # 比只读的多给几步：它要改代码 + 验证，来回次数天然更多
        max_steps=12,
    ),
}
