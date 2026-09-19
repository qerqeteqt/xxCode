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


_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_size(n: int) -> str:
    """把字节数变成 '1.2 KB' 这样的可读形式，1024 进制。

    给模型看的数字要能一眼比大小：1048576 和 2097152 得数零，
    '1.0 MB' 和 '2.0 MB' 不用。所以超过 1 KB 就换单位，保留一位小数。

    两个刻意的选择：

        B 不带小数         '512 B' 比 '512.0 B' 干净，字节本来就是整数
        只保留一位小数      '1.2 KB' 够用了，再精确反而更难扫读

    负数按 0 处理 —— 调用方多半是 stat().st_size 或 len()，真出现负数说明上游
    有问题，但这里没必要抛异常把整个工具调用带崩。
    """
    if n < 0:
        n = 0

    size = float(n)
    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} {_SIZE_UNITS[-1]}"  # 到不了，兜底


def truncate(text: str, limit: int) -> str:
    """超长就截断并注明原始长度。

    模型看不到「这里被截断了」的话，会把半个文件当成全部，然后基于残缺信息下结论。
    所以截断提示必须写清楚，让它知道可以缩小范围再来一次。
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（输出已截断，原始长度 {len(text)} 字符）"
