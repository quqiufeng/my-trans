#!/usr/bin/env bash
# dub/render.sh — EDL → 成片（逐段切 → 拼接 → 烧字幕）
#
#   dub/render.sh edl.json out.mp4 [--keep-audio] [--crf 20] [--preset medium]
#   dub/render.sh edl.json out.mp4 --fit --shot=shot/IMG_0195.json [--rate=3.6]
#
# --fit：文案驱动。每句时长 = 字数/rate（夹在 --min-dur..--max-dur），画面从窗口池
# 取（同 tag 优先），窗口不够长就循环、池子用完就轮转 —— 话术随便加长都不会没画面。
#
# 为什么分两步：13 路 1080p hevc 同时在 filter_complex 里 trim 会撑爆内存被杀；
# 逐段用前置 -ss 快 seek 转码（一次只解一路），再 concat + 烧字幕。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDL="${1:?用法: dub/render.sh edl.json out.mp4 [--keep-audio] [--crf N]}"
OUT="${2:?缺输出路径}"
shift 2
KEEP=0; CRF=20; PRESET=medium; OW=1080; OH=1920
FIT=0; SHOT=""; RATE=3.6; MIND=1.0; MAXD=6.0
for a in "$@"; do
  case "$a" in
    --keep-audio) KEEP=1 ;;
    --crf=*)      CRF="${a#*=}" ;;
    --preset=*)   PRESET="${a#*=}" ;;
    --w=*)        OW="${a#*=}" ;;
    --h=*)        OH="${a#*=}" ;;
    --fit)        FIT=1 ;;
    --shot=*)     SHOT="${a#*=}" ;;
    --rate=*)     RATE="${a#*=}" ;;
    --min-dur=*)  MIND="${a#*=}" ;;
    --max-dur=*)  MAXD="${a#*=}" ;;
  esac
done

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/seg"

KC=""; [ "$KEEP" = 1 ] && KC="-c:a aac -b:a 160k -ac 1" || KC="-an"

FITARGS=()
if [ "$FIT" = 1 ]; then
  [ -n "$SHOT" ] || { echo "✗ --fit 需要 --shot=<窗口表.json>" >&2; exit 1; }
  FITARGS=(--fit --shot "$SHOT" --rate "$RATE" --min-dur "$MIND" --max-dur "$MAXD")
fi
luajit "$HERE/render.lua" "$EDL" --ass "$TMP/subs.ass" --plan "$TMP/plan.tsv" --segdir "$TMP/seg" \
  --w "$OW" --h "$OH" ${FITARGS[@]+"${FITARGS[@]}"} >/dev/null
SRC="$(sed -n 's/.*"src"[^"]*"\([^"]*\)".*/\1/p' "$EDL" | head -1)"

n=0
while IFS=$'\t' read -r T_IN T_OUT SEG S0 S1; do
  n=$((n+1))
  printf '  [%2d] %6.2f-%6.2f  %ss\n' "$n" "$T_IN" "$T_OUT" "$(awk -v a="$T_IN" -v b="$T_OUT" 'BEGIN{printf "%.2f", b-a}')"
  ffmpeg -y -nostdin -v error -ss "$T_IN" -to "$T_OUT" -i "$SRC" \
    -vf "scale=${OW}:${OH}:force_original_aspect_ratio=decrease,pad=${OW}:${OH}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30000/1001" \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p $KC "$SEG"
  printf "file '%s'\n" "$SEG" >> "$TMP/list.txt"
done < "$TMP/plan.tsv"

echo "▸ 拼接 + 烧字幕 → $OUT"
ffmpeg -y -nostdin -v error -f concat -safe 0 -i "$TMP/list.txt" \
  -vf "ass=$TMP/subs.ass" -c:v libx264 -preset "$PRESET" -crf "$CRF" \
  -pix_fmt yuv420p -movflags +faststart $KC "$OUT"
ls -lh "$OUT" | awk '{print "▸ 输出 " $5 "  " $9}'
