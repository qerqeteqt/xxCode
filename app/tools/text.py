"""字节解码与文本截断 —— 工具返回值的两个基本功。"""

import locale


def decode_bytes(data: bytes, *, prefer_utf8: bool = True) -> str:
    """把字节解码成 str。

    **两种来源要用不同的优先顺序**，这是踩过坑的：

        读文件（prefer_utf8=True）    源码基本都是 UTF-8，先试 UTF-8
        子进程输出（prefer_utf8=False）Windows 上是本地代码页（中文是 cp936），
                                      而 **cp936 的字节序列可能是合法 UTF-8** ——
                                      先试 UTF-8 会「成功」解出一串乱码，而不是
                                      抛错回退。所以必须本地编码优先。

    当初就是因为没区分这两者，cmd.exe 的报错被解成了
    「'grep' �����ڲ����ⲿ���」。

    两种都失败就 utf-8 + replace：宁可带几个乱码字符，也不要让整个工具调用失败 ——
    模型看到乱码还能自己判断，看到一个异常就只能盲目重试。
    """
    encodings = ["utf-8", locale.getpreferredencoding(False)]
    if not prefer_utf8:
        encodings.reverse()

    for encoding in dict.fromkeys(encodings):
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
