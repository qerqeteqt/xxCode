"""Web 层 —— 把 Runtime 包成一个网页。

Runtime 一行没改：事件、流式、权限确认三个口子都是早就留好的
回调/注入点。这一层只负责翻译成 HTTP/SSE。
"""

from app.web.server import create_app, update_env

__all__ = ["create_app", "update_env"]
