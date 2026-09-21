"""图片仓库测试。

两块重点：

1. **校验只看内容。** 文件名和 Content-Type 都是外部可控的，而我们会把
   这坨字节写进磁盘、再原样回吐给浏览器。所以魔数之外的一切都不采信。
2. **引用名是安全边界。** 它从 HTTP 路径和 JSONL 两处来，拼进文件路径之前
   必须挡住 ../ 和盘符。
"""

import base64

import pytest

from app.llm.content import REF_PREFIX
from app.memory.image_store import MAX_IMAGE_BYTES, ImageError, ImageStore

# 一个**真的** 1×1 红色 PNG（69 字节，PIL 能打开）。
# 不用「PNG 魔数 + 垃圾」是为了让「它确实是一张图」这件事没有歧义
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"pretend jpeg body"


@pytest.fixture
def store(tmp_path):
    return ImageStore(tmp_path)


def test_存进去能读回来(store, tmp_path):
    name = store.save(PNG_BYTES)

    assert name.endswith(".png")
    assert (tmp_path / ".agent" / "images" / name).read_bytes() == PNG_BYTES
    url = store.data_url(name)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES


def test_同样内容只存一份(store, tmp_path):
    first = store.save(PNG_BYTES)
    second = store.save(PNG_BYTES)

    assert first == second
    assert len(list((tmp_path / ".agent" / "images").iterdir())) == 1


def test_不同内容得到不同名字(store):
    assert store.save(PNG_BYTES) != store.save(JPEG_BYTES)


def test_不同格式按内容推扩展名(store):
    assert store.save(JPEG_BYTES).endswith(".jpg")


def test_引用名以_content_里的前缀规范为准(store):
    """名字形状是 app/llm/content.py 定的 —— 那边改了这边必须跟着。"""
    from app.llm.content import is_image_name

    assert is_image_name(store.save(PNG_BYTES))


# ---------------------------------------------------------------- 校验


def test_文本内容被拒绝(store):
    """校验看的是魔数，不是调用方说什么。"""
    with pytest.raises(ImageError, match="图片"):
        store.save("# 这是一段 Markdown，不是图片".encode("utf-8"))


def test_空内容被拒绝(store):
    with pytest.raises(ImageError):
        store.save(b"")


def test_超过大小上限被拒绝(store):
    huge = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_IMAGE_BYTES + 1)

    with pytest.raises(ImageError, match="太大"):
        store.save(huge)


def test_刚好等于上限是放行的(store):
    """边界要明确：上限是「不得超过」，不是「必须小于」。"""
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_IMAGE_BYTES - 8)

    assert store.save(data).endswith(".png")


# ---------------------------------------------------------------- 路径穿越


@pytest.mark.parametrize(
    "name",
    [
        "../../.env",
        "..\\..\\.env",
        "C:/Windows/win.ini",
        "3f9a1c2b4d5e6f70/../../.env",
        "abc.png/../../x",
        "",
        "3f9a1c2b4d5e6f70",
        "3f9a1c2b4d5e6f7.png",
        "3F9A1C2B4D5E6F70.png",
    ],
)
def test_路径穿越的引用名被拒绝(store, name):
    with pytest.raises(ImageError):
        store.path_for(name)


def test_路径穿越的名字_exists_返回_False_而不是抛错(store):
    assert store.exists("../../.env") is False


def test_文件不存在时_data_url_抛错(store):
    with pytest.raises(ImageError, match="不存在"):
        store.data_url("0123456789abcdef.png")


# ---------------------------------------------------------------- 缓存


def test_第二次读走缓存不再碰磁盘(store, tmp_path):
    """一轮对话里同一张图会被展开最多 60 次（每次 LLM 调用一次）。

    不缓存的话就是每轮重读重编码同一个文件几十遍。名字是内容哈希、
    内容永不变，所以缓存无条件安全 —— 这里用「把文件删了还能读出来」来证明。
    """
    name = store.save(PNG_BYTES)
    first = store.data_url(name)

    store.path_for(name).unlink()

    assert store.data_url(name) == first
