"""Markdown 渲染器的测试。

断言写在 `markdown_check.js` 里（用 node 跑），这里只是把它接进 pytest，
这样一条 `pytest` 就能覆盖全部。

为什么非要测这个：**它是整个项目里唯一一处我们自己拼 HTML 的地方**，
而模型输出是不可信内容。手工在浏览器里点几下，测不出 `<script>` 有没有被转义 ——
而那正是唯一要紧的用例。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name("markdown_check.js")
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="没装 node，跳过 Markdown 渲染器的断言")
def test_markdown_渲染器全部断言通过():
    proc = subprocess.run(
        [NODE, str(CHECKER)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0, f"渲染器有断言失败:\n{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout
