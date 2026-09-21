"""图片仓库 —— 用户贴进来的截图存这儿，JSONL 里只留一个引用名。

## 为什么图片不直接进 JSONL

一张真实截图（1454×912）是 160 KB，base64 之后 **213 KB**。而会话文件是 append-only 的：

    压缩只压内存里的 messages，文件本身一直追加（docs/STATUS.md 的「已知局限」）

纯文本时那句「留着没影响」是成立的 —— 最长的会话也才 869 KB。图片会把它变成
213 KB/张、永不删除。所以图片走文件，JSONL 里只存 `ref:<name>`。

顺带一个白捡的好处：base64 永远不进入上下文，压缩时也就不会出现
「把 200 KB 的 base64 拼进摘要 prompt」这种硬故障。

## 内容寻址

文件名 = `sha256(内容)[:16]` + 扩展名。于是同一张图贴两次只占一份。
这是**去重**，不是权限控制 —— 但正好顺带保证了名字不可变，
所以 `data_url` 的缓存无条件安全。

## 这个仓库是只写的

**没有 delete 方法，也不打算加。** 内容寻址 + append-only 的 JSONL 意味着
任何删除都可能删掉另一个会话还在引用的文件，而那个引用是永久且不可恢复的
（历史写进 JSONL 就不会被改写）。做引用计数又是另一套要维护的不变量。

真需要清理时用**离线 sweep**，而不是在热路径上加删除：
扫 `.agent/sessions/**/*.jsonl` 里的 `ref:` 建存活集，删掉不在集合里的。
约 20 行、不参与任何请求、不用维护任何不变量。按 ~150 KB/张 × ~5 张/会话
× ~150 会话/年 算，大约是 110 MB/年 —— 现在还不到要上机制的程度。

## 消费者分工

    格式定义（ref: 长什么样）  -> app/llm/content.py
    存取（本文件）             -> 读写的唯一入口，也是路径校验的落点
"""

import base64
import hashlib
from pathlib import Path

from app.llm.content import is_image_name

# 存放位置，和 sessions / memory 同级，都在 <root>/.agent/ 下面
IMAGE_DIR = ".agent/images"

# 单张图上限。真实截图 160 KB 级别，给到 8 MB 足够宽松，
# 同时挡住「手滑贴了一个几百 MB 的东西」把内存和磁盘都拖垮。
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_EXT_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


class ImageError(RuntimeError):
    """图片不可接受（太大 / 不是图片 / 名字非法 / 读不到）。消息会给用户看。"""


def _sniff(data: bytes) -> str | None:
    """按魔数判断图片格式，返回扩展名；不是认识的图片就返回 None。

    **刻意不看扩展名、也不看调用方给的 Content-Type** —— 两者都是外部可控的，
    而我们接下来要把这坨字节写进磁盘、再原样回吐给浏览器。
    只看内容，就没有「改个后缀骗过去」这回事。
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    # WebP 是 RIFF 容器：前 4 字节 RIFF，第 8-11 字节 WEBP
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def mime_of(name: str) -> str:
    """按扩展名给出 MIME。扩展名是我们自己按魔数定下来的，可采信。"""
    return _EXT_MIME.get(name.rsplit(".", 1)[-1], "application/octet-stream")


class ImageStore:
    def __init__(self, root: str | Path) -> None:
        self.base = Path(root).resolve() / IMAGE_DIR
        # 名字不可变（内容寻址），所以缓存无条件安全。
        # 不缓存的话，一张图在一轮里会被重读重编码最多 60 次（每次 LLM 调用都要展开）
        self._data_urls: dict[str, str] = {}

    # ------------------------------------------------------------ 写

    def save(self, data: bytes) -> str:
        """存一张图，返回引用名。同样的内容永远得到同样的名字。"""
        if not data:
            raise ImageError("图片内容为空")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageError(
                f"图片太大（{len(data) / 1024 / 1024:.1f} MB，"
                f"上限 {MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB）"
            )

        ext = _sniff(data)
        if ext is None:
            raise ImageError("这不是一张图片（只认 PNG / JPEG / GIF / WebP）")

        name = f"{hashlib.sha256(data).hexdigest()[:16]}.{ext}"
        target = self.base / name
        if target.exists():
            # 去重：同一张图贴第二次不重写。这也让「先存后引用」的时序更简单
            return name

        self.base.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return name

    # ------------------------------------------------------------ 读

    def path_for(self, name: str) -> Path:
        """把引用名解析成路径。**校验在前，拼路径在后。**

        名字会从两个地方来：HTTP 路径参数，和 JSONL 文件。两者都不是完全可信的
        （前者可以被构造，后者可能被手工改过），所以这里必须是唯一一道闸门。
        校验不过就抛，绝不返回一个「看起来还行」的路径。
        """
        if not is_image_name(name):
            raise ImageError(f"非法的图片引用名: {name!r}")
        return self.base / name

    def exists(self, name: str) -> bool:
        try:
            return self.path_for(name).exists()
        except ImageError:
            return False

    def data_url(self, name: str) -> str:
        """读出来拼成 `data:image/png;base64,...`。

        这是 `LLMClient` 构造 payload 时调的东西 —— 整个系统里 base64
        只在那一刻、那一份拷贝上存在。
        """
        cached = self._data_urls.get(name)
        if cached is not None:
            return cached

        path = self.path_for(name)
        if not path.exists():
            raise ImageError(f"图片不存在: {name}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{mime_of(name)};base64,{encoded}"
        self._data_urls[name] = url
        return url
