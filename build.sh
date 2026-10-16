#!/bin/bash
# build.sh — 用 StaticPy 工具链把 transcribe.static.py 编译成独立二进制
#
# 产物: ./transcribe (ELF) + ./transcribe.so (Chez AOT)
# 依赖 StaticPy 的部分一律用绝对路径 (/opt/StaticPy)。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

STATICPY_REPO="/opt/StaticPy"
WHISPER_CPP="/opt/StaticPy/cpp/whisper"
STATICPY_BUILD="/opt/StaticPy/staticpy/static_build_comfycli.sh"

# 1) 编译 C ABI 库 libwhisper.so（ctranslate2 Whisper + 自写 log-mel 前端）
bash "$WHISPER_CPP/build.sh"

# 2) StaticPy AOT 编译（Python 子集 → Scheme → ELF）
#    工具链固定输出到 /opt/StaticPy/，随后拷回本项目目录
( cd /opt/StaticPy && bash "$STATICPY_BUILD" \
    "$HERE/transcribe.static.py" transcribe "$HERE/transcribe_ffi.scm" )

# 3) 取回产物
cp -f /opt/StaticPy/transcribe /opt/StaticPy/transcribe.so "$HERE/"

echo "OK: $HERE/transcribe"
