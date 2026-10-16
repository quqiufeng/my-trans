#!/bin/bash
# translate.sh — 运行 translate 二进制（依赖 StaticPy 的部分用绝对路径）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="/opt/StaticPy/cpp/llm${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/translate" "$@"
