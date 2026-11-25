#!/bin/bash
# build.sh — 编译 libllm.so（Hunyuan-MT / llama.cpp 服务的 C ABI 封装，本项目副本）
# 与工具链同一份源码；差异仅在 SUB_MARK（组内 [n] 行号标记，逐行对齐）。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(cd "$DIR/../.." && pwd)/libllm.so}"

echo ">>> compiling libllm.so -> $OUT"
g++ -O2 -std=c++17 -fPIC -shared "$DIR/llm_capi.cpp" \
  -I"$DIR/thirdparty" -lpthread -o "$OUT"
echo "OK: $OUT"
