#!/bin/bash
# subtitle.sh — 一步生成双语字幕：视频 → transcribe → translate → 双语 ASS
#
# 用法:
#   bash subtitle.sh                    扫描当前目录的视频，逐个处理
#   bash subtitle.sh 视频1.mp4 [视频2.mp4 ...]
#   bash subtitle.sh --offset=0.3 视频.mp4
#   bash subtitle.sh --lang=ja 视频.mp4  # 指定目标语言（默认 zh）
#
# 结果: <视频名>_<识别语言>.ass（原文）+ <视频名>.ass（双语）
# 依赖 StaticPy 的部分用绝对路径（/opt/StaticPy）；翻译需 llama.cpp 服务（未启动则自动拉起）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLM_HOST_URL="${LLM_HOST:-http://127.0.0.1:8080/v1}"

offset=""
lang="zh"
files=()
for a in "$@"; do
  case "$a" in
    --offset=*) offset="${a#--offset=}" ;;
    --lang=*)   lang="${a#--lang=}" ;;
    *)          files+=("$a") ;;
  esac
done

# 无参数则扫描当前目录的视频
if [ ${#files[@]} -eq 0 ]; then
  shopt -s nullglob nocaseglob
  files=( *.mp4 *.mkv *.avi *.mov *.flv *.wmv *.m4v )
  shopt -u nullglob nocaseglob
  echo "扫描当前目录，找到 ${#files[@]} 个视频文件"
fi
if [ ${#files[@]} -eq 0 ]; then
  echo "没有视频文件"
  exit 1
fi

# 确保翻译服务可用（未启动则后台拉起）
if ! curl -s -m 2 "$LLM_HOST_URL/models" >/dev/null 2>&1; then
  echo ">>> 翻译服务未运行，启动 start_server.sh ..."
  setsid bash "$HERE/start_server.sh" > "$HERE/llama_server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    curl -s -m 2 "$LLM_HOST_URL/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s -m 2 "$LLM_HOST_URL/models" >/dev/null 2>&1 || { echo "翻译服务启动失败"; exit 1; }
  echo ">>> 翻译服务就绪"
fi

t0=$(date +%s)
for v in "${files[@]}"; do
  echo ">>> [1/2] transcribe: $v"
  targs=()
  [ -n "$offset" ] && targs+=("--offset=$offset")
  bash "$HERE/transcribe.sh" "${targs[@]}" "$v"

  base="${v%????}"                        # 去掉扩展名（4 字符）
  shopt -s nullglob
  produced=( "${base}"_*.ass )
  shopt -u nullglob
  if [ ${#produced[@]} -eq 0 ]; then
    echo "  ! 未找到转写产物 ${base}_*.ass，跳过翻译"
    continue
  fi
  src="$(ls -t "${produced[@]}" | head -1)"   # 取最新生成的

  echo ">>> [2/2] translate: $src"
  bash "$HERE/translate.sh" --lang="$lang" "$src"
done
echo "完成，用时 $(( $(date +%s) - t0 ))s"
