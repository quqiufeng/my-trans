#!/usr/bin/env bash
# dub/dub.sh — 视频配音 + 烧字幕（自包含副本，独立于 my-agent）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${COSYVOICE_LIB_DIR:-/opt/cosyvoice.cpp/build/lib}:${COSYVOICE_BIN_DIR:-/opt/cosyvoice.cpp/build/bin}:${LD_LIBRARY_PATH:-}"
exec luajit "$HERE/dub.lua" "$@"
