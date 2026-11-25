# ass_retime.static.py — StaticPy 版 ass_retime.py
# 用音频能量把 ASS 每段的起止重新对齐到真实发声点（Whisper 段边界有 ±0.5s 抖动）。
# 用法: ass_retime <字幕.ass> <音轨.wav> [--dry-run] [--stats]
# WAV 采样访问走 C ABI（libasswav.so），算法全在这里。
extern fn aw_open(path: str) -> ptr from "asswav"
extern fn aw_len(h: ptr) -> int from "asswav"
extern fn aw_rate(h: ptr) -> int from "asswav"
extern fn aw_rms(h: ptr, start: int, n: int) -> double from "asswav"
extern fn aw_close(h: ptr) -> int from "asswav"
extern fn aw_bom(path: str) -> int from "asswav"

HOP: float = 0.01      # 10ms/帧
MIN_GAP: float = 0.30  # 静音超过 300ms 算换句
LEAD: float = 0.06     # 字幕比开口早一点点
TAIL: float = 0.12     # 说完留一点尾巴
SNAP: float = 0.40     # 吸附上限
REACH: float = 2.5     # 边界落在静音里时，向前/向后找最近发声点的最大距离
MINRUN: int = 8        # 短于 8 帧(0.08s)的发声段不参与吸附
BINS: int = 4096


# ---------- 时间戳 ----------
def parse_ts(s: str) -> float:
    p = list_to_py_list(str_split(s, ":"))
    if py_list_length(p) < 3:
        return 0.0
    h: float = string_to_float(py_list_ref(p, 0))
    m: float = string_to_float(py_list_ref(p, 1))
    r: float = string_to_float(py_list_ref(p, 2))
    return h * 3600.0 + m * 60.0 + r


def pad2(v: int) -> str:
    if v < 10:
        return "0" + string_of_int(v)
    return string_of_int(v)


def fmt_ts(t: float) -> str:
    x: float = t
    if x < 0.0:
        x = 0.0
    h: int = int(x / 3600.0)
    m: int = int((x - float(h) * 3600.0) / 60.0)
    s: float = x - float(h) * 3600.0 - float(m) * 60.0
    ss: str = float_fixed(s, 2)
    n: int = str_length(ss)
    if n < 5:
        ss = str_slice("00000", n, 5) + ss
    return string_of_int(h) + ":" + pad2(m) + ":" + ss


# ---------- 能量 / 分辨率无关的分位数 ----------
def order_stat(e2: list[float], nfr: int, k: int) -> float:
    "第 k 小的每帧能量：6 轮桶细分，精度足够复现 np.percentile"
    lo: float = 0.0
    hi: float = 1.0
    it: int = 0
    while it < 6:
        step: float = (hi - lo) / float(BINS)
        if step <= 0.0:
            return lo
        cnts: list[int] = make_int_array(BINS)
        below: int = 0
        i: int = 0
        while i < nfr:
            v: float = float_array_ref(e2, i)
            if v < lo:
                below = below + 1
            else:
                if v < hi:
                    b: int = int((v - lo) / step)
                    if b < 0:
                        b = 0
                    if b >= BINS:
                        b = BINS - 1
                    int_array_set(cnts, b, int_array_ref(cnts, b) + 1)
            i = i + 1
        want: int = k - below
        acc: int = 0
        hit: int = BINS - 1
        b2: int = 0
        while b2 < BINS:
            c: int = int_array_ref(cnts, b2)
            if acc + c > want:
                hit = b2
                b2 = BINS
            else:
                acc = acc + c
                b2 = b2 + 1
        lo = lo + float(hit) * step
        hi = lo + step
        it = it + 1
    return lo


def percentile30(e2: list[float], nfr: int) -> float:
    tgt: float = 0.30 * float(nfr - 1)
    r1: int = int(tgt)
    fr: float = tgt - float(r1)
    a: float = order_stat(e2, nfr, r1)
    if fr <= 0.000001 or r1 + 1 >= nfr:
        return a
    b: float = order_stat(e2, nfr, r1 + 1)
    return a + fr * (b - a)


def frames_rms(h: ptr, nsm: int, hh: int) -> list[float]:
    "逐帧均方能量"
    nfr: int = int(nsm / hh)
    e2: list[float] = make_float_array(nfr)
    i: int = 0
    while i < nfr:
        r: float = aw_rms(h, i * hh, hh)
        float_array_set(e2, i, r * r)
        i = i + 1
    return e2


def build_mask(e2: list[float], nfr: int, thr2: float) -> list[int]:
    m: list[int] = make_int_array(nfr)
    i: int = 0
    while i < nfr:
        if float_array_ref(e2, i) > thr2:
            int_array_set(m, i, 1)
        i = i + 1
    return m


def merge_runs(m: list[int], nfr: int) -> list[int]:
    "原始发声帧 -> 合并短静音后的段；段数*2 存在末尾槽 nfr+1"
    gaps: int = int(MIN_GAP / HOP)
    o: list[int] = make_int_array(nfr + 2)
    k: int = 0
    i: int = 0
    cur_a: int = -1
    cur_b: int = -1
    while i < nfr:
        if int_array_ref(m, i) == 1:
            if cur_a < 0:
                cur_a = i
                cur_b = i
            else:
                if i - cur_b <= gaps:
                    cur_b = i
                else:
                    int_array_set(o, k, cur_a)
                    int_array_set(o, k + 1, cur_b)
                    k = k + 2
                    cur_a = i
                    cur_b = i
        i = i + 1
    if cur_a >= 0:
        int_array_set(o, k, cur_a)
        int_array_set(o, k + 1, cur_b)
        k = k + 2
    int_array_set(o, nfr + 1, k)
    return o


def mask_from_runs(o: list[int], nfr: int) -> list[int]:
    "合并后的发声掩码（短静音也算发声）"
    m2: list[int] = make_int_array(nfr)
    k: int = int_array_ref(o, nfr + 1)
    i: int = 0
    while i < k:
        a: int = int_array_ref(o, i)
        b: int = int_array_ref(o, i + 1)
        f: int = a
        while f <= b:
            if f >= 0 and f < nfr:
                int_array_set(m2, f, 1)
            f = f + 1
        i = i + 2
    return m2


def voice_runs(o: list[int], cnt: int) -> int:
    "参与吸附的发声段数（滤掉 <MINRUN 帧的碎段）"
    n: int = 0
    i: int = 0
    while i < cnt:
        if int_array_ref(o, i * 2 + 1) - int_array_ref(o, i * 2) >= MINRUN:
            n = n + 1
        i = i + 1
    return n


def nearest_voice(m: list[int], nfr: int, t: float, reach: float) -> float:
    "t 处无声时，前后 reach 秒内找最近发声帧；找不到返回 -1"
    i: int = int(round(t / HOP))
    if i < 0 or i >= nfr:
        return -1.0
    if int_array_ref(m, i) == 1:
        return t
    kmax: int = int(reach / HOP)
    best: float = -1.0
    k: int = 1
    while k < kmax:
        if i + k < nfr and int_array_ref(m, i + k) == 1:
            best = float(i + k) * HOP
            k = kmax
        else:
            k = k + 1
    k = 1
    while k < kmax:
        if i - k >= 0 and int_array_ref(m, i - k) == 1:
            cand: float = float(i - k) * HOP
            if best < 0.0 or abs(cand - t) < abs(best - t):
                best = cand
            k = kmax
        else:
            k = k + 1
    return best


def snap_run(o: list[int], cnt: int, t: float, side: int, limit: float) -> float:
    "side=1 找最近的段起点；side=0 找最近的段终点。找不到返回 -1"
    best: float = -1.0
    bd: float = limit
    i: int = 0
    while i < cnt:
        a: int = int_array_ref(o, i * 2)
        b: int = int_array_ref(o, i * 2 + 1)
        if b - a >= MINRUN:
            v: float = 0.0
            if side == 1:
                v = float(a) * HOP
            else:
                v = float(b + 1) * HOP
            d: float = abs(v - t)
            if d < bd:
                best = v
                bd = d
        i = i + 1
    return best


def next_onset(o: list[int], cnt: int, t: float, limit: float) -> float:
    "t 之后（-0.05 容差）最近的一个发声段起点；超出 limit 返回 -1"
    best: float = -1.0
    i: int = 0
    while i < cnt:
        a: int = int_array_ref(o, i * 2)
        b: int = int_array_ref(o, i * 2 + 1)
        if b - a >= MINRUN:
            v: float = float(a) * HOP
            if v >= t - 0.05 and v - t <= limit:
                if best < 0.0 or v < best:
                    best = v
        i = i + 1
    return best


def prev_offset(o: list[int], cnt: int, t: float, limit: float) -> float:
    "t 之前（+0.05 容差）最近的一个发声段终点；超出 limit 返回 -1"
    best: float = -1.0
    i: int = 0
    while i < cnt:
        a: int = int_array_ref(o, i * 2)
        b: int = int_array_ref(o, i * 2 + 1)
        if b - a >= MINRUN:
            v: float = float(b + 1) * HOP
            if v <= t + 0.05 and t - v <= limit:
                if v > best:
                    best = v
        i = i + 1
    return best


def median_of(v: list[float], n: int) -> float:
    "近似中位数（桶细分够用）"
    mn: float = float_array_ref(v, 0)
    mx: float = mn
    i: int = 0
    while i < n:
        x: float = float_array_ref(v, i)
        if x < mn:
            mn = x
        if x > mx:
            mx = x
        i = i + 1
    # 直接在原区间上做 6 轮桶选择
    lo: float = mn
    hi: float = mx + 0.000001
    k: int = int((n - 1) / 2)
    it: int = 0
    while it < 6:
        step: float = (hi - lo) / float(BINS)
        if step <= 0.0:
            return lo
        cnts: list[int] = make_int_array(BINS)
        below: int = 0
        i = 0
        while i < n:
            x = float_array_ref(v, i)
            if x < lo:
                below = below + 1
            else:
                if x < hi:
                    b: int = int((x - lo) / step)
                    if b < 0:
                        b = 0
                    if b >= BINS:
                        b = BINS - 1
                    int_array_set(cnts, b, int_array_ref(cnts, b) + 1)
            i = i + 1
        want: int = k - below
        acc: int = 0
        hit: int = BINS - 1
        b2: int = 0
        while b2 < BINS:
            c: int = int_array_ref(cnts, b2)
            if acc + c > want:
                hit = b2
                b2 = BINS
            else:
                acc = acc + c
                b2 = b2 + 1
        lo = lo + float(hit) * step
        hi = lo + step
        it = it + 1
    return lo


# ---------- 主流程 ----------
def retime(ass: str, wav: str, dry: int) -> int:
    h: ptr = aw_open(wav)
    nsm: int = aw_len(h)
    if nsm <= 0:
        print("读不了音轨: " + wav)
        return 1
    sr: int = aw_rate(h)
    hh: int = int(HOP * float(sr))
    if hh < 1:
        hh = 1
    nfr: int = int(nsm / hh)

    e2: list[float] = frames_rms(h, nsm, hh)
    aw_close(h)
    fl: float = percentile30(e2, nfr)
    thr2: float = fl * 9.0
    if thr2 < 0.000036:
        thr2 = 0.000036
    o: list[int] = merge_runs(build_mask(e2, nfr, thr2), nfr)
    cnt: int = int(int_array_ref(o, nfr + 1) / 2)
    m: list[int] = mask_from_runs(o, nfr)

    thr: float = sqrt(thr2)
    print("音轨 " + float_fixed(float(nsm) / float(sr), 1) + "s  发声段 " +
          string_of_int(voice_runs(o, cnt)) + " 段  阈值 " + float_fixed(thr, 4))

    text: str = read_text_file(ass)
    lines = list_to_py_list(str_split(text, "\n"))
    nline: int = list_length(lines)

    # 第一遍：抽原始起止
    raw_s: list[float] = make_float_array(nline)
    raw_e: list[float] = make_float_array(nline)
    fs: list[float] = make_float_array(nline)
    fe: list[float] = make_float_array(nline)
    nraw: int = 0
    i: int = 0
    while i < nline:
        l: str = list_ref(lines, i)
        if str_starts_with(l, "Dialogue:"):
            p = list_to_py_list(str_split(l, ","))
            if py_list_length(p) >= 3:
                float_array_set(raw_s, nraw, parse_ts(py_list_ref(p, 1)))
                float_array_set(raw_e, nraw, parse_ts(py_list_ref(p, 2)))
                nraw = nraw + 1
        i = i + 1

    na: int = 0
    nb: int = 0
    j: int = 0
    while j < nraw:
        st: float = float_array_ref(raw_s, j)
        en: float = float_array_ref(raw_e, j)
        ns: float = snap_run(o, cnt, st, 1, SNAP)
        if ns < 0.0:
            ns = st
        else:
            ns = ns - LEAD
        prev: float = -1.0
        if j > 0:
            prev = float_array_ref(fe, j - 1)
        if ns < prev + 0.02:
            ns = prev + 0.02
        ne: float = snap_run(o, cnt, en, 0, SNAP)
        if ne < 0.0:
            ne = en
        else:
            ne = ne + TAIL
        if j + 1 < nraw:
            lim: float = float_array_ref(raw_s, j + 1) - 0.02
            if ne > lim:
                ne = lim
        if ne < ns + 0.5:
            ne = ns + 0.5
        # 兜底：起点不能落在静音里 —— 优先向前找真正的开口点（Whisper 常把句首
        # 静音一并算进段时间，导致字幕提前几百毫秒蹦出来），找不到再用就近发声帧
        ii: int = int(round(ns / HOP))
        if ii >= 0 and ii < nfr and int_array_ref(m, ii) == 0:
            lim_s: float = 0.0
            if j + 1 < nraw:
                lim_s = float_array_ref(raw_s, j + 1) - 0.05
            vn: float = next_onset(o, cnt, ns, REACH)
            if vn >= 0.0 and (lim_s <= 0.0 or vn <= lim_s):
                ns = vn - LEAD
            else:
                v: float = nearest_voice(m, nfr, ns, 0.8)
                if v >= 0.0 and abs(v - ns) <= 0.6:
                    ns = v - LEAD
            if ns < prev + 0.02:
                ns = prev + 0.02
        ie: int = int(round(ne / HOP))
        if ie > nfr - 1:
            ie = nfr - 1
        if ie >= 0 and int_array_ref(m, ie) == 0:
            vp: float = prev_offset(o, cnt, ne, REACH)
            if vp >= 0.0 and vp > ns + 0.2:
                ne = vp + TAIL
            else:
                v2: float = nearest_voice(m, nfr, ne, 0.8)
                if v2 >= 0.0 and abs(v2 - ne) <= 0.6:
                    ne = v2 + TAIL
        if ne < ns + 0.5:
            ne = ns + 0.5
        if ns < st - 0.65:
            ns = st - 0.65
        if ns > st + REACH + 0.2:
            ns = st + REACH + 0.2
        if ne < en - REACH - 0.2:
            ne = en - REACH - 0.2
        if ne > en + 0.65:
            ne = en + 0.65
        if ne < ns + 0.4:
            ne = ns + 0.4
        float_array_set(fs, j, ns)
        float_array_set(fe, j, ne)
        if abs(ns - st) > 0.1:
            na = na + 1
        if abs(ne - en) > 0.1:
            nb = nb + 1
        j = j + 1

    # clamp 可能把相邻段挤重叠，统一压一遍
    j = 0
    while j + 1 < nraw:
        a0: float = float_array_ref(fs, j)
        b0: float = float_array_ref(fe, j)
        a1: float = float_array_ref(fs, j + 1)
        b1: float = float_array_ref(fe, j + 1)
        lim2: float = a0 + 0.3
        if a1 < lim2 + 0.02:
            a1 = lim2 + 0.02
            if b1 < lim2 + 0.42:
                b1 = lim2 + 0.42
            float_array_set(fs, j + 1, a1)
            float_array_set(fe, j + 1, b1)
        if b0 > a1 - 0.02:
            b0 = a1 - 0.02
            float_array_set(fe, j, b0)
        j = j + 1

    # 统计
    ds: list[float] = make_float_array(nraw)
    sm: float = 0.0
    se: float = 0.0
    dmin: float = 999.0
    dmax: float = -999.0
    ov: int = 0
    j = 0
    while j < nraw:
        dsv: float = float_array_ref(fs, j) - float_array_ref(raw_s, j)
        dev: float = float_array_ref(fe, j) - float_array_ref(raw_e, j)
        float_array_set(ds, j, dsv)
        sm = sm + dsv
        se = se + dev
        if dsv < dmin:
            dmin = dsv
        if dsv > dmax:
            dmax = dsv
        if j + 1 < nraw and float_array_ref(fe, j) > float_array_ref(fs, j + 1) + 0.001:
            ov = ov + 1
        j = j + 1
    if nraw > 0:
        print("段数 " + string_of_int(nraw) + "  起点位移 " + float_fixed(sm / float(nraw), 3) +
              " 中位 " + float_fixed(median_of(ds, nraw), 3) +
              " [" + float_fixed(dmin, 2) + "," + float_fixed(dmax, 2) + "]  终点位移 " +
              float_fixed(se / float(nraw), 3))
        print("  起点挪动 >0.1s: " + string_of_int(na) + " 段   终点 >0.1s: " +
              string_of_int(nb) + " 段   段间重叠 " + string_of_int(ov) + " 处")

    if dry == 1:
        print("dry-run：未写回")
        return 0

    # 第二遍：改写 Dialogue 行
    out = py_list()
    k: int = 0
    i = 0
    while i < nline:
        l: str = list_ref(lines, i)
        if not str_starts_with(l, "Dialogue:"):
            out = py_list_append(out, l)
        else:
            p = list_to_py_list(str_split(l, ","))
            np_: int = py_list_length(p)
            if np_ < 10 or k >= nraw:
                out = py_list_append(out, l)
            else:
                head: str = py_list_ref(p, 0) + "," + fmt_ts(float_array_ref(fs, k)) + "," + fmt_ts(float_array_ref(fe, k))
                q: int = 3
                while q < 9:
                    head = head + "," + py_list_ref(p, q)
                    q = q + 1
                tail: str = py_list_ref(p, 9)
                q = 10
                while q < np_:
                    tail = tail + "," + py_list_ref(p, q)
                    q = q + 1
                out = py_list_append(out, head + "," + tail)
                k = k + 1
        i = i + 1

    res: str = str_join(out, "\n")
    if not str_ends_with(res, "\n"):
        res = res + "\n"
    if aw_bom(ass) == 1:        # read_text_file 吃掉 BOM，写回时补上
        res = "\ufeff" + res
    fp = file_open(ass, "w")
    file_write(fp, res)
    file_close(fp)
    print("已写回 " + ass + "（" + string_of_int(k) + " 段）")
    return 0


def main():
    args = list_to_py_list(argv())
    n: int = py_list_length(args)
    if n < 3:
        print("用法: ass_retime <字幕.ass> <音轨.wav> [--dry-run|--stats]")
        return
    ass: str = py_list_ref(args, 1)
    wav: str = py_list_ref(args, 2)
    dry: int = 0
    i: int = 3
    while i < n:
        a: str = py_list_ref(args, i)
        if a == "--dry-run" or a == "--stats":
            dry = 1
        i = i + 1
    if not os_file_exists(ass):
        print("字幕不存在: " + ass)
        return
    if not os_file_exists(wav):
        print("音轨不存在: " + wav)
        return
    retime(ass, wav, dry)
