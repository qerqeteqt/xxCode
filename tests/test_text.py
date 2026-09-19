"""app/tools/text.py 的单元测试。

纯函数，不碰网络、不碰文件系统。
"""

import pytest

from app.tools.text import human_size


# ================================================================ human_size


@pytest.mark.parametrize(
    "n, expected",
    [
        # 边界：1024 是换单位的临界点，两边都要钉住
        (0, "0 B"),
        (1, "1 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1025, "1.0 KB"),
        # 各单位的典型值
        (1536, "1.5 KB"),
        (10 * 1024, "10.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**2 * 3 // 2, "1.5 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (1024**5, "1.0 PB"),
    ],
)
def test_human_size(n, expected):
    assert human_size(n) == expected


def test_human_size_bytes_have_no_decimal():
    """B 不带小数 —— '512 B' 而不是 '512.0 B'。"""
    assert human_size(512) == "512 B"
    assert "." not in human_size(512)


def test_human_size_uses_1024_not_1000():
    """1024 进制：1000 字节还是 B，1024 才进 KB。"""
    assert human_size(1000) == "1000 B"
    assert human_size(1024) == "1.0 KB"


def test_human_size_negative_treated_as_zero():
    """负数按 0 处理，不抛异常。"""
    assert human_size(-1) == "0 B"
    assert human_size(-1024**3) == "0 B"


def test_human_size_beyond_pb_does_not_overflow():
    """超过 PB 就停在 PB，不越界也不抛异常。"""
    assert human_size(1024**6) == "1024.0 PB"


def test_human_size_accepts_bool_and_int_like():
    """bool 是 int 的子类，顺手确认不会炸。"""
    assert human_size(True) == "1 B"
