"""多模态消息的线格式 —— `ref:` 引用方案的唯一定义处。

## 为什么单独一个模块

图片不落进 JSONL，那里只存一个引用：

    {"type": "image_url", "image_url": {"url": "ref:3f9a1c2b4d5e6f70.png"}}

也就是**和 API 完全同形的块，只有 URL 的前缀不同**。于是「展开」退化成一次字符串替换，
而发出请求的那一刻才把它变成 `data:image/png;base64,...`。

这个格式有好几个消费者：存储层要能认出它、压缩器要能在摘要里跳过它、记忆提取要能忽略它、
HTTP 层要能把它拆出来、而 `LLMClient` 要能展开它。所以它必须只有一处定义。

## 为什么放在 app/llm/ 而不是 app/memory/

`app/llm/client.py` 目前是个**零依赖的叶子节点**，而它必须在构造 payload 时展开引用。
如果格式定义在 `app/memory/`，client 就得反向 import —— 而
`app/memory/__init__` → `session_store` → `app.context.compactor` → `app.llm.client`
已经是一条环，那会在 `client.py` 还没定义完 `LLMClient` 的时候就回头 import 它，
直接 ImportError。

所以分工是：**格式（本文件，纯函数、不 import 任何 app 模块）** 与
**存取（`app/memory/image_store.py`）** 分离，两边靠同一个名字正则对齐。
"""

import re

# 引用前缀。存进 JSONL 的 URL 长这样：ref:3f9a1c2b4d5e6f70.png
REF_PREFIX = "ref:"

# 图片名的唯一合法形状：16 位 hex + 白名单扩展名。
# 这是**安全边界**，不是格式洁癖 —— 这个名字会从 HTTP 路径参数和 JSONL 里来，
# 拼进文件路径之前必须挡住 ../、盘符、反斜杠这些东西。
IMAGE_NAME_RE = re.compile(r"^[0-9a-f]{16}\.(?:png|jpg|jpeg|gif|webp)$")

# 渲染成纯文本时图片变成什么。摘要、标题、记忆文件里都只留这个。
IMAGE_PLACEHOLDER = "[图片]"

# 引用解析不出来时的替换文本。宁可让模型知道「这儿本来有张图但读不到」，
# 也不要让整个请求失败。
IMAGE_UNREADABLE = "[图片无法读取]"


def is_image_name(name: object) -> bool:
    """这个名字是不是合法的图片引用名。"""
    return isinstance(name, str) and IMAGE_NAME_RE.fullmatch(name) is not None


def image_block(name: str) -> dict:
    """造一个指向图片的 content 块（存进 JSONL 的就是它）。"""
    return {"type": "image_url", "image_url": {"url": f"{REF_PREFIX}{name}"}}


def image_refs_of(content: object) -> list[str]:
    """从 content 里把图片引用名抽出来。`image_block` 的逆。

    str 形状（纯文本消息）返回空列表 —— 调用方靠这个判断「这条消息有没有图」。
    """
    if not isinstance(content, list):
        return []

    refs: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_url":
            continue
        url = (block.get("image_url") or {}).get("url")
        if isinstance(url, str) and url.startswith(REF_PREFIX):
            name = url[len(REF_PREFIX) :]
            if is_image_name(name):
                refs.append(name)
    return refs


def content_to_text(content: object) -> str:
    """把任何形状的 content 渲染成纯文本，图片用 `[图片]` 标出来。

    存在的理由：好几处地方（摘要提示、记忆提取、AutoDream 材料）
    都是把 content 直接当字符串用的。图片进来之后 content 会变成块列表，
    不换这个函数的话，那些地方会把一个 Python 块列表的 repr 写进摘要 prompt
    甚至 `.agent/memory/*.md` —— 而后者**会被注入以后每一轮的 system prompt**。

    这里的取舍是「丢掉图片内容、只留占位符」：摘要和记忆要的是事实，
    不是 200 KB 的 base64。而占位符本身是有信息量的 ——「这儿本来有张图」。
    """
    return _render_blocks(content, with_images=True)


def content_text_only(content: object) -> str:
    """只取文字，图片直接丢掉，不留占位符。

    和 `content_to_text` 的差别是**问题不同**：

        content_to_text    「这条消息里有什么」 -> 图片要标成 [图片]
        content_text_only  「用户到底说了什么」 -> 图片要丢掉

    会话标题要的是后者。贴一张图再打字提问，标题该是那句提问；
    用 content_to_text 的话会变成「[图片] 这个报错怎么修」，
    甚至纯贴图的会话标题直接是「[图片]」—— 列表里等于没有标题。
    """
    return _render_blocks(content, with_images=False)


def _render_blocks(content: object, *, with_images: bool) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        # 不是我们认识的形状。这些调用点都在收尾路径上，宁可渲染个四不像也不要抛
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text") or ""
            if text:
                parts.append(str(text))
        elif kind == "image_url" and with_images:
            parts.append(IMAGE_PLACEHOLDER)
        # 其它块型跳过 —— 现在只有这两种，将来多了也不该在这里炸
    return " ".join(parts)


def build_user_content(text: str, refs: list[str] | None = None) -> str | list[dict]:
    """构造一条 user 消息的 content。

    **没有图片时必须原样返回那个字符串**，不能统一包成块列表：
    包了的话，所有既有会话、所有既有测试、以及每一个不含图片的请求，形状都会变，
    而「纯文本就是 str」是这个项目一路走来的不变量。

    有图片时 text 为空就**不产生空的文本块** —— 只贴一张图、不打字，是完全正常的用法，
    而有些 API 不喜欢 `{"type":"text","text":""}`。
    """
    if not refs:
        return text

    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(image_block(ref) for ref in refs)
    return blocks
