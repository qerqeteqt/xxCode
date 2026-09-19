"""权限闸门。

## 它和沙箱是什么关系

    沙箱    「到不了」    —— 结构约束，确定性，挡不住的东西根本碰不着
    闸门    「到了也要过规则/问人」 —— 策略，可配置，可被点错

闸门是**代码**，所以比「在 prompt 里请求模型别乱来」可靠。但它不是万能：
规则可能写错，人可能按疲劳了乱点「允许」。**纵深防御的每一层都不完美**，
闸门的价值在于降低误伤、降低对注意力的依赖，而不是「可以放心让它去跑别人的代码」。

## 三层判定，顺序不能反

    ① allow 规则命中   → 放行     （显式放开，优先级最高）
    ② deny 规则命中    → 拒绝     （敏感路径，不问）
    ③ 按风险等级默认   → read 放行 / write·execute 问人

①在②前面是刻意的：`allow` 就是那条「我知道我在干什么」的逃生通道，
要能盖过内置的敏感清单。否则你想让 Agent 读一次 .env 就只能改源码。

## 为什么 deny 是硬拒绝、不问人

规则的意义就是「**不用问**」。命中规则还要问一遍的话，规则和默认策略就没区别了，
只是多弹一次窗。要放开就改 `.agent/permissions.json` —— 一次编辑换永久生效，
比每次点「允许」强，也不会让你养成闭眼点 y 的习惯。

## 一个必须承认的局限

闸门拦的是**工具调用**，不是**意图**。模型发现 `Write a.py` 被拒，理论上可以改写成
`Bash("echo ... > a.py")` 绕过去 —— 而 Bash 的命令字符串不像路径那样容易被规则命中。
**Bash 不受沙箱管辖**（它被交给操作系统，`cwd` 只是起点不是边界），这是 V1 明知
而不补的洞：真正的进程级隔离要靠容器 / seccomp / Windows Job Object，不是这一版的事。
"""

import fnmatch
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from app.tools.base import Tool

logger = logging.getLogger(__name__)

# 默认拒绝的敏感文件。**默认拒绝、显式放开** ——
# 这些不是「可能有问题」，是「正常任务压根不需要碰」。
SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.*.local",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".netrc",
    ".npmrc",
)

PERMISSIONS_FILE = ".agent/permissions.json"

_RISK_VERB = {"read": "读取", "write": "写入", "execute": "执行"}


class Decision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    """一次待确认的调用。够渲染出一句人话就行。"""

    tool_name: str
    risk: str
    subject: str | None
    source: str
    summary: str


@dataclass(frozen=True)
class PermissionOutcome:
    allowed: bool
    message: str | None = None  # 被拒时回灌给模型的话，要写成它能据此改主意的样子


Confirmer = Callable[[PermissionRequest], Awaitable[Decision]]


def matches_any(subject: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """按 fnmatch 匹配。

    整个相对路径和文件名各试一次，所以 `.env` 这种写法既能命中根目录的，
    也能命中 `subdir/.env` —— 写规则的时候不用去想它在哪一层。
    """
    name = PurePosixPath(subject).name
    return any(
        fnmatch.fnmatch(subject, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def load_rules(root: str | Path) -> tuple[list[str], list[str]]:
    """读项目里的权限规则，返回 (deny, allow)。

    内置敏感清单**永远生效**，文件里的 deny 是追加在上面的；
    allow 能盖过 deny，那是「显式放开」的唯一入口。

    文件不存在、读不动、格式不对，都退回只用内置规则 ——
    坏掉的配置不该让人完全失去保护。
    """
    deny = list(SENSITIVE_PATTERNS)
    allow: list[str] = []

    path = Path(root) / PERMISSIONS_FILE
    if not path.exists():
        return deny, allow

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        deny.extend(str(p) for p in (data.get("deny") or []))
        allow.extend(str(p) for p in (data.get("allow") or []))
    except (json.JSONDecodeError, OSError, AttributeError) as e:
        logger.warning("权限规则文件读不了，只用内置规则: %s", e)

    return deny, allow


class PermissionGate:
    def __init__(
        self,
        *,
        confirmer: Confirmer | None = None,
        deny: tuple[str, ...] | list[str] = SENSITIVE_PATTERNS,
        allow: tuple[str, ...] | list[str] = (),
        source: str = "Main",
        grants: set[str] | None = None,
    ) -> None:
        self._confirmer = confirmer
        self._deny = tuple(deny)
        self._allow = tuple(allow)
        self._source = source
        # 本会话放行的风险等级。**只活在内存里** —— 一次手滑选了「永久允许」
        # 就跟着项目走了，而你多半不记得自己什么时候做的决定
        self._grants: set[str] = grants if grants is not None else set()

    @classmethod
    def for_project(
        cls,
        root: str | Path,
        *,
        confirmer: Confirmer | None = None,
        source: str = "Main",
        grants: set[str] | None = None,
    ) -> "PermissionGate":
        deny, allow = load_rules(root)
        return cls(
            confirmer=confirmer, deny=deny, allow=allow, source=source, grants=grants
        )

    def with_source(self, source: str) -> "PermissionGate":
        """换个来源标签，但**共享**同一份会话放行记录。

        SubAgent 内部的确认必须标出来 —— 否则用户看到一条光秃秃的「想写入 x.py」，
        完全不记得自己刚才在问哪件事。
        """
        return PermissionGate(
            confirmer=self._confirmer,
            deny=self._deny,
            allow=self._allow,
            source=source,
            grants=self._grants,
        )

    async def check(self, tool: Tool, subject: str | None) -> PermissionOutcome:
        risk = getattr(tool, "risk", "execute")

        if subject and matches_any(subject, self._allow):
            return PermissionOutcome(True)

        if subject and matches_any(subject, self._deny):
            return PermissionOutcome(
                False,
                f"{tool.name}: 拒绝访问 {subject!r} —— 它命中敏感路径规则。"
                f"这是安全默认值，不是错误。确实需要的话，"
                f"把该路径加进 {PERMISSIONS_FILE} 的 allow 列表再重试。",
            )

        if risk == "read":
            return PermissionOutcome(True)

        if risk in self._grants:
            return PermissionOutcome(True)

        return await self._ask(tool, risk, subject)

    async def _ask(self, tool: Tool, risk: str, subject: str | None) -> PermissionOutcome:
        if self._confirmer is None:
            # 没人可问。安全默认是**拒绝** —— 宁可不做，也不要静默地做。
            # 注意正常路径下不会走到这：AutoDream 的注册表压根不挂闸门，
            # SubAgent 拿到的是带 confirmer 的子闸门
            return PermissionOutcome(
                False, f"{tool.name}: 当前没有人工确认通道，已拒绝「{risk}」类操作"
            )

        verb = _RISK_VERB.get(risk, risk)
        target = f" {subject}" if subject else ""
        summary = f"{verb}{target}"
        if self._source != "Main":
            summary += f"（来自 {self._source}）"

        request = PermissionRequest(
            tool_name=tool.name, risk=risk, subject=subject,
            source=self._source, summary=summary,
        )
        decision = await self._confirmer(request)

        if decision is Decision.ALLOW_SESSION:
            self._grants.add(risk)
            return PermissionOutcome(True)
        if decision is Decision.ALLOW_ONCE:
            return PermissionOutcome(True)
        return PermissionOutcome(
            False,
            f"{tool.name}: 用户拒绝了这个操作。换个做法，或者向用户说明为什么需要它。",
        )
