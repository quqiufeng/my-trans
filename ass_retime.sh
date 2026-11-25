#!/bin/bash
# ass_retime.sh — 运行 ass_retime 二进制（libasswav.so 在本目录）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="$HERE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/ass_retime" "$@"
