#!/usr/bin/env bash
# dub/batch.sh — 一条素材 × N 份文案 → N 条成片
#
#   ./batch.sh 素材.mp4 文案/               # 目录里每个 *.txt 出一条
#   ./batch.sh 素材.mp4 a.txt b.txt c.txt   # 点名几份
#   ./batch.sh 素材.mp4 文案/ --plan        # 只体检每份文案，不出片
#   ./batch.sh 素材.mp4 文案/ --probe       # 试合成报落点，不出片
#
# 其余参数（--voice/--rate/--max-gap/--keep-audio/--font/--clone-speed/--allow-drop）
# 原样透传给 dub.sh。输出：<素材名>_<文案名>.mp4，落在 --out 目录（默认素材同目录）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIDEO=""
OUTDIR=""
PLAN=0
PROBE=0
TEXTS=()
PASS=()

for a in "$@"; do
  case "$a" in
    --plan)   PLAN=1 ;;
    --probe)  PROBE=1 ;;
    --out)    OUTDIR="__NEXT__" ;;
    *) if [[ "${OUTDIR:-}" == "__NEXT__" ]]; then OUTDIR="$a"
       elif [[ -z "$VIDEO" ]]; then VIDEO="$a"
       elif [[ -f "$a" || -d "$a" ]]; then TEXTS+=("$a")
       else PASS+=("$a"); fi ;;
  esac
done
# --out 后跟的值已被上面吞掉，从 PASS 里剔除
[[ "${OUTDIR:-}" == "__NEXT__" ]] && OUTDIR=""

if [[ -z "$VIDEO" || ${#TEXTS[@]} -eq 0 ]]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

# 展开目录
FILES=()
for t in "${TEXTS[@]}"; do
  if [[ -d "$t" ]]; then
    while IFS= read -r f; do FILES+=("$f"); done < <(find "$t" -maxdepth 1 -type f -name '*.txt' | sort)
  else
    FILES+=("$t")
  fi
done
[[ ${#FILES[@]} -eq 0 ]] && { echo "没找到文案（*.txt）" >&2; exit 2; }

STEM="$(basename "${VIDEO%.*}")"
TAGDIR="${OUTDIR:-$(dirname "$VIDEO")}"
mkdir -p "$TAGDIR"
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null || echo '?')"

echo "素材: $VIDEO (${DUR}s)"
echo "文案: ${#FILES[@]} 份  →  目标目录: $TAGDIR"
echo "------------------------------------------------------------"

ok=0; fail=0
for f in "${FILES[@]}"; do
  NAME="$(basename "${f%.txt}")"
  DEST="$TAGDIR/${STEM}_${NAME}.mp4"
  echo "▸ [$((ok+fail+1))/${#FILES[@]}] $NAME"
  if (( PLAN )); then
    out="$("$HERE/dub.sh" "$VIDEO" "$f" --plan "${PASS[@]}" 2>&1)"
    echo "$out" | sed 's/^/    /'
    # 落盘的 _copy.* 按文案名区分，避免互相覆盖
    for ext in txt srt; do
      src="$TAGDIR/${STEM}_copy.$ext"
      [[ -f "$src" ]] && mv -f "$src" "$TAGDIR/${STEM}_${NAME}_copy.$ext"
    done
    if grep -q '✓ 长度贴合' <<<"$out"; then ok=$((ok+1)); else fail=$((fail+1)); fi
  elif (( PROBE )); then
    "$HERE/dub.sh" "$VIDEO" "$f" --probe "${PASS[@]}" 2>&1 | sed 's/^/    /' && ok=$((ok+1)) || fail=$((fail+1))
  else
    if "$HERE/dub.sh" "$VIDEO" "$f" --out "$DEST" "${PASS[@]}" 2>&1 | sed 's/^/    /'; then
      ok=$((ok+1)); echo "    ✓ $DEST"
    else
      fail=$((fail+1)); echo "    ✗ 失败（文案超时？先跑 --plan 看建议字数）"
    fi
  fi
done

echo "------------------------------------------------------------"
(( PLAN || PROBE )) && { echo "体检完成：可 $ok 份 / 需改 $fail 份"; exit 0; }
echo "完成：成功 $ok 条 / 失败 $fail 条"
(( fail > 0 )) && exit 1 || exit 0
