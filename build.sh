#!/bin/bash
# build.sh — 用 StaticPy 工具链把 transcribe/translate 编译成独立二进制
#
# 产物: ./transcribe, ./translate (+ 各自 .so)
# 依赖 StaticPy 的部分一律用绝对路径 (/opt/StaticPy)。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

STATICPY_REPO="/opt/StaticPy"
WHISPER_CPP="/opt/StaticPy/cpp/whisper"
LLM_CPP="/opt/StaticPy/cpp/llm"
STATICPY_BUILD="/opt/StaticPy/staticpy/static_build_comfycli.sh"

# 1) 编译 C ABI 库
bash "$WHISPER_CPP/build.sh"   # libwhisper.so：CT2 Whisper + 自写 log-mel
bash "$LLM_CPP/build.sh"       # libllm.so：Hunyuan-MT (llama.cpp) HTTP 封装

# 2) StaticPy AOT 编译（Python 子集 → Scheme → ELF），产物取回本项目目录
for app in transcribe translate; do
  ( cd /opt/StaticPy && bash "$STATICPY_BUILD" \
      "$HERE/$app.static.py" "$app" "$HERE/${app}_ffi.scm" )
  cp -f "/opt/StaticPy/$app" "/opt/StaticPy/$app.so" "$HERE/"
  echo "OK: $HERE/$app"
done
