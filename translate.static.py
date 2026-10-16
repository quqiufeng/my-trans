# translate.static.py — StaticPy 版字幕翻译：ASS → 双语 ASS（Hunyuan-MT / llama.cpp 服务）
# 用法:
#   translate                     扫描当前目录的 ASS，逐个翻译
#   translate 字幕.ass [字幕2.ass]
#   translate --lang=zh 字幕.ass  目标语言 zh/en/ja/ko/fr/de/es/...
# 输出: <名字去掉 _en>.ass（双语，仓耳今楷02）
# 环境: LLM_HOST / LLM_MODEL（默认 127.0.0.1:8080/v1, hunyuan-mt）
extern fn llm_translate_ass(in_ass: str, out_ass: str, lang: str) -> int from "llm"


def is_ass(name: str) -> int:
    hit: int = 0
    if str_ends_with(name, ".ass"):
        hit = 1
    if str_ends_with(name, ".ASS"):
        hit = 1
    return hit


def make_out(inp: str) -> str:
    n: int = str_length(inp)
    base: str = str_slice(inp, 0, n - 4)
    if str_ends_with(base, "_en"):
        base = str_slice(base, 0, str_length(base) - 3)
    return base + ".ass"


def main():
    args = list_to_py_list(argv())
    n: int = py_list_length(args)
    lang: str = "zh"
    files = py_list()
    i: int = 1
    while i < n:
        a: str = py_list_ref(args, i)
        if str_starts_with(a, "--lang="):
            lang = str_slice(a, 7, str_length(a))
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
            if is_ass(name) == 1:
                files = py_list_append(files, name)
            k = k + 1
        print("扫描当前目录，找到 " + string_of_int(py_list_length(files)) + " 个 ASS 文件")

    total: int = py_list_length(files)
    if total == 0:
        print("没有字幕文件")
        return

    t0: int = clock()
    j: int = 0
    while j < total:
        f: str = py_list_ref(files, j)
        out: str = make_out(f)
        rc: int = llm_translate_ass(f, out, lang)
        if rc < 0:
            print("  " + f + "  [FAIL]")
        else:
            print("  " + f + " -> " + out + " (" + string_of_int(rc) + " 条)")
        j = j + 1
    print("完成 " + string_of_int(total) + " 个  用时 " + string_of_int(clock() - t0) + "s")
