#!/usr/bin/env bash
# 启动本地 llama.cpp 服务（Hunyuan-MT 翻译模型）
set -e

LLAMA_SERVER="${LLAMA_SERVER:-/opt/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/data/models/Hunyuan-MT-7B-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-hunyuan-mt}"
NGL="${NGL:-99}"
CTX="${CTX:-8192}"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX" \
  -a "$ALIAS" \
  --no-webui
