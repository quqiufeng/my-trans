#!/bin/bash
# build.sh — 编译 libasswav.so（ass_retime 的 WAV 读取 C ABI）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(dirname "$DIR")/../libasswav.so}"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
echo ">>> compiling $OUT"
cc -O2 -fPIC -shared -o "$OUT" "$DIR/asswav.c" -lm
ls -l "$OUT"
