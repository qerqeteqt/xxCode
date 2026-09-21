"""多模态 content 的线格式测试。

为什么值得单独测：`build_user_content` 有一条**必须守住**的向后兼容契约 ——
没有图片时它必须原样返回那个字符串，不能统一包成块列表。包了的话，
所有既有会话、所有既有测试、以及每一个不含图片的请求，形状都会变。

而 `content_to_text` 挡的是另一类事故：会话标题、摘要 prompt、以及
`.agent/memory/*.md` 里都不该出现块列表的 repr。
"""

import pytest

from app.llm.content import (
    IMAGE_PLACEHOLDER,
    REF_PREFIX,
    build_user_content,
    content_text_only,
    content_to_text,
    image_block,
    image_refs_of,
    is_image_name,
)

NAME = "3f9a1c2b4d5e6f70.png"


# ---------------------------------------------------------------- 名字校验


@pytest.mark.parametrize(
    "name",
    [
        "../../.env",
        "..\\..\\.env",
        "C:/Windows/win.ini",
        "3f9a1c2b4d5e6f70/../../x.png",
        "abc.png/../../x",
        "",
        "3f9a1c2b4d5e6f70",
        "3f9a1c2b4d5e6f7.png",  # 15 位，少一个
        "3F9A1C2B4D5E6F70.png",  # 大写
        "3f9a1c2b4d5e6f70.exe",
        "3f9a1c2b4d5e6f70.png.exe",
        None,
        123,
    ],
)
def test_非法的图片引用名被拒绝(name):
    assert is_image_name(name) is False


def test_合法的图片引用名通过():
    assert is_image_name(NAME) is True
    assert is_image_name("3f9a1c2b4d5e6f70.JPG") is False  # 扩展名也大小写敏感


# ---------------------------------------------------------------- content_to_text


def test_纯文本原样通过():
    assert content_to_text("看看这个") == "看看这个"


@pytest.mark.parametrize("value", [None, ""])
def test_空值和_None_返回空串(value):
    assert content_to_text(value) == ""


def test_图片块渲染成占位符():
    out = content_to_text([image_block(NAME)])

    assert IMAGE_PLACEHOLDER in out
    # 不能把块本身或 base64 漏出来
    assert "image_url" not in out
    assert REF_PREFIX not in out


def test_文本加图片时两者都在():
    out = content_to_text([{"type": "text", "text": "这个报错"}, image_block(NAME)])

    assert "这个报错" in out
    assert IMAGE_PLACEHOLDER in out


def test_未知块型被跳过而不是抛错():
    out = content_to_text([{"type": "audio", "data": "..."}, {"type": "text", "text": "好"}])

    assert out == "好"


# ---------------------------------------------------------------- content_text_only


def test_只取文字时图片被丢掉不留占位符():
    """这是「用户说了什么」的口径，和 content_to_text 的「有什么」不同。

    会话标题用这个：贴张图再打字提问，标题该是那句提问，
    而不是「[图片] 这个报错怎么修」。
    """
    content = [{"type": "text", "text": "这个报错怎么修"}, image_block(NAME)]

    assert content_text_only(content) == "这个报错怎么修"
    # 对照：另一个函数会留下占位符
    assert IMAGE_PLACEHOLDER in content_to_text(content)


def test_只有图片时只取文字返回空串():
    """空串是「跳过这条」的信号 —— 会话标题靠它落到下一条真有文字的消息上。"""
    assert content_text_only([image_block(NAME)]) == ""


def test_只取文字时纯文本原样通过():
    assert content_text_only("你好") == "你好"
    assert content_text_only(None) == ""


# ---------------------------------------------------------------- build_user_content


def test_build_user_content_没有图片时返回字符串():
    """**整个图片功能里最重要的一条向后兼容断言。**

    统一包成块列表的话，无图会话的请求形状、会话文件形状、以及一堆既有测试
    会全部跟着变 —— 而它们本来和图片毫无关系。
    """
    result = build_user_content("你好", [])

    assert isinstance(result, str)
    assert result == "你好"


def test_没有_refs_参数时同样返回字符串():
    assert isinstance(build_user_content("你好"), str)


def test_build_user_content_有图片时返回块列表():
    result = build_user_content("看看", [NAME])

    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "看看"}
    assert result[1] == image_block(NAME)


def test_空文本时不产生空文本块():
    """只贴图不打字是正常用法。"""

    result = build_user_content("", [NAME])

    assert result == [image_block(NAME)]
    assert all(block["type"] == "image_url" for block in result)


# ---------------------------------------------------------------- image_refs_of


def test_image_refs_of_只取图片引用名():
    content = [{"type": "text", "text": "看看"}, image_block(NAME)]

    assert image_refs_of(content) == [NAME]


def test_image_refs_of_对纯文本返回空():
    assert image_refs_of("看看") == []
    assert image_refs_of(None) == []


def test_image_refs_of_忽略不是_ref_前缀的_url():
    """data URL（已经展开过的）和 http 链接都不该被当成引用。"""
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]

    assert image_refs_of(content) == []


def test_往返一致():
    """build_user_content 造出来的东西，image_refs_of 必须能原样读回来。"""
    content = build_user_content("看看", [NAME, "0123456789abcdef.jpg"])

    assert image_refs_of(content) == [NAME, "0123456789abcdef.jpg"]
