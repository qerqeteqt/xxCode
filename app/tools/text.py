"""字节解码与文本截断 —— 工具返回值的两个基本功。"""

import locale


def decode_bytes(data: bytes) -> str:
    """把文件/子进程产出的 bytes 解码成 str。

    顺序：utf-8 -> 系统本地编码（简体中文 Windows 是 cp936）。
    两个都不行就 utf-8 + replace：宁可带几个乱码字符，也不要让整个工具调用失败 ——
    模型看到乱码还能自己判断，看到一个异常就只能盲目重试。
    """
    for encoding in dict.fromkeys(["utf-8", locale.getpreferredencoding(False)]):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def truncate(text: str, limit: int) -> str:
    """超长就截断并注明原始长度。

    模型看不到「这里被截断了」的话，会把半个文件当成全部，然后基于残缺信息下结论。
    所以截断提示必须写清楚，让它知道可以缩小范围再来一次。
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（输出已截断，原始长度 {len(text)} 字符）"
