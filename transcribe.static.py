# transcribe.static.py — StaticPy 版 transcribe.py：视频 → ASS 字幕（CT2 C++ Whisper）
# 用法:
#   transcribe                        扫描当前目录的视频，逐个生成字幕
#   transcribe 视频1.mp4 [视频2.mp4 ...]
#   transcribe --offset=0.3 视频.mp4
# 输出: <视频名>_<语言>.ass（仓耳今楷02 字体）
# 环境: WHISPER_MODEL_DIR / WHISPER_DEVICE(cpu)
extern fn wh_lang(path: str) -> str from "whisper"
extern fn wh_to_ass(path: str, out: str, offset: double) -> int from "whisper"


def is_video(name: str) -> int:
    exts = str_split(".mp4,.mkv,.avi,.mov,.flv,.wmv,.m4v", ",")
    j: int = 0
    m: int = list_length(exts)
    hit: int = 0
    while j < m:
        ext: str = list_ref(exts, j)
        if str_ends_with(name, ext):
            hit = 1
        if str_ends_with(name, str_upper(ext)):
            hit = 1
        j = j + 1
    return hit


def make_out(inp: str, lang: str) -> str:
    n: int = str_length(inp)
    base: str = str_slice(inp, 0, n - 4)
    return base + "_" + lang + ".ass"


def do_file(inp: str, offset: float) -> int:
    lang: str = wh_lang(inp)
    if str_length(lang) == 0:
        lang = "unknown"
    out: str = make_out(inp, lang)
    rc: int = wh_to_ass(inp, out, offset)
    if rc < 0:
        print(inp + "  [FAIL]")
    else:
        print("  " + inp + "  [" + lang + "] -> " + out + " (" + string_of_int(rc) + " 条)")
    return rc


def main():
    args = list_to_py_list(argv())
    n: int = py_list_length(args)
    offset: float = 0.0
    files = py_list()
    i: int = 1
    while i < n:
        a: str = py_list_ref(args, i)
        if str_starts_with(a, "--offset="):
            offset = string_to_float(str_slice(a, 9, str_length(a)))
        else:
            if os_file_exists(a):
                files = py_list_append(files, a)
            else:
                print("文件不存在: " + a)
        i = i + 1

    if py_list_length(files) == 0:
        entries = os_list_dir(".")
        k: int = 0
        m: int = list_length(entries)
        while k < m:
            name: str = list_ref(entries, k)
            if is_video(name) == 1:
                files = py_list_append(files, name)
            k = k + 1
        print("扫描当前目录，找到 " + string_of_int(py_list_length(files)) + " 个视频文件")

    total: int = py_list_length(files)
    if total == 0:
        print("没有有效视频文件")
        return

    t0: int = clock()
    ok: int = 0
    j: int = 0
    while j < total:
        f: str = py_list_ref(files, j)
        rc: int = do_file(f, offset)
        if rc >= 0:
            ok = ok + 1
        j = j + 1

    print("完成: " + string_of_int(ok) + "/" + string_of_int(total) +
          "  用时 " + string_of_int(clock() - t0) + "s")
