#!/bin/bash
# build.sh — 用独立 StaticPy 项目（默认 /opt/StaticPy）编译 transcribe / translate
#
# 产物: ./transcribe、./translate（+ 各自 .so、build-info.json），只落本项目目录。
# 构建逻辑一律交给工具链自带入口，本脚本不自己实现任何一步：
#   C ABI 库   $ROOT/cpp/{whisper,llm}/build.sh
#   AOT 编译   $ROOT/staticpy/static_build.sh   （工具链唯一构建入口）
# STATICPY_ROOT 可覆盖工具链位置。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${STATICPY_ROOT:-/opt/StaticPy}"

# 1) C ABI 库（.so 就地留在 $ROOT/cpp/*/，运行时由 *_ffi.scm + LD_LIBRARY_PATH 加载）
bash "$ROOT/cpp/whisper/build.sh"
bash "$ROOT/cpp/llm/build.sh"

# 2) StaticPy AOT 编译：裸名产物目录由 STATICPY_OUTDIR 指定（工具链自带机制）
export STATICPY_OUTDIR="$HERE"
for app in transcribe translate; do
    bash "$ROOT/staticpy/static_build.sh" \
        "$HERE/$app.static.py" "$app" "$HERE/${app}_ffi.scm"
done

echo "=== 完成 ==="
ls -l "$HERE/transcribe" "$HERE/transcribe.so" "$HERE/translate" "$HERE/translate.so"
