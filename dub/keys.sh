#!/usr/bin/env bash
# keys.sh — 抽关键帧拼图，给 AI 直接读（不依赖任何模型/服务）
#
#   dub/keys.sh <视频> [--step 3] [--cols 3] [--rows 3] [--width 400] [--out DIR]
#
# 产出（DIR 默认 <视频目录>/keys）：
#   <base>_keys_01.jpg …   每张 cols×rows 格，格内有时间戳
#   <base>_keys.txt        每张图对应的 格号 → 时间 对照表
#   <base>.shotlist.json   待 AI 填写的空模板（phases 数组）
#
# 只做「切帧 + 缩小 + 拼图」，不判断内容；判断由 AI 看图后写进 shotlist。
set -uo pipefail

step=3; cols=3; rows=3; width=400; OUT=""
SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --step)  step="$2"; shift 2 ;;
        --cols)  cols="$2"; shift 2 ;;
        --rows)  rows="$2"; shift 2 ;;
        --width) width="$2"; shift 2 ;;
        --out)   OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) SRC="$1"; shift ;;
    esac
done
[ -n "$SRC" ] || { echo "用法: $0 <视频> [--step 3] [--cols 3] [--rows 3] [--width 400] [--out DIR]" >&2; exit 1; }
[ -f "$SRC" ] || { echo "找不到素材: $SRC" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "缺 ffmpeg" >&2; exit 1; }

dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
base=$(basename "$SRC"); base="${base%.*}"
[ -n "$OUT" ] || OUT="$(dirname "$SRC")/keys"
mkdir -p "$OUT" || exit 1

per=$(( cols * rows ))
cell=$(( per * step ))
n=$(( (${dur%.*} + cell - 1) / cell ))   # 需要几张图
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
filt="fps=1/${step},scale=${width}:-2"
[ -f "$FONT" ] && filt="$filt,drawtext=fontfile=${FONT}:text='%{pts\\:hms}':x=6:y=6:fontsize=18:fontcolor=yellow:box=1:boxcolor=black@0.55:boxborderw=4"
filt="$filt,tile=${cols}x${rows}"

echo "素材 $base  时长 ${dur}s  每 ${step}s 一帧  每张 ${per} 格（${cols}x${rows}）"
echo "预计 ${n} 张图 -> $OUT"

: > "$OUT/${base}_keys.txt"
for i in $(seq 1 "$n"); do
    ss=$(( (i - 1) * cell ))
    key="${base}_keys_$(printf '%02d' "$i")"
    # setpts 把时间戳抬回绝对时间，否则每张图的 drawtext 都从 0 开始，会误导
    ffmpeg -v error -ss "$ss" -t "$cell" -i "$SRC" \
        -vf "setpts=PTS+${ss}/TB,$filt" -frames:v 1 -q:v 4 -y "$OUT/${key}.jpg" \
        || { echo "第 $i 张失败" >&2; continue; }
    {
        echo "# $key.jpg  从 ${ss}s 起（格内时间戳为准）"
        k=1
        for r in $(seq 1 "$cols"); do
            for c in $(seq 1 "$rows"); do
                t=$(( ss + (k - 1) * step ))
                printf '  [%d] %02d:%02d  (%ds)\n' "$k" $((t / 60)) $((t % 60)) "$t"
                k=$(( k + 1 ))
            done
        done
    } >> "$OUT/${base}_keys.txt"
done

# 给 AI 填的空模板
cat > "$OUT/${base}.shotlist.json" <<JSON
{"src": "$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")",
 "duration": $dur,
 "step": $step,
 "phases": [
   {"in": 0.0, "out": 0.0, "tag": "填一个短标签", "subject": "画面里有什么", "quality": 3, "note": ""}
 ]}
JSON

ls -l "$OUT"/*.jpg 2>/dev/null | awk '{printf "  %s  %sKB\n", $9, int($5/1024)}'
echo "对照表: $OUT/${base}_keys.txt"
echo "模板  : $OUT/${base}.shotlist.json"
