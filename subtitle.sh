#!/bin/bash
# subtitle.sh — 一步生成双语字幕：视频 → 抽音频 → transcribe → translate → 双语 ASS
#
# 用法:
#   bash subtitle.sh                    扫描当前目录的视频，逐个处理
#   bash subtitle.sh 视频1.mp4 [视频2.mp4 ...]
#   bash subtitle.sh --offset=0.3 视频.mp4
#   bash subtitle.sh --lang=ja 视频.mp4  # 指定目标语言（默认 zh）
#
# 步骤: ① ffmpeg 抽取音频(<名字>.wav，16k 单声道无损，体积小、便于复用)
#       ② transcribe 生成 <名字>_<语言>.ass   ③ translate 生成双语 <名字>.ass
# 结果: <名字>_<语言>.ass（原文）+ <名字>.ass（双语，与视频同名 → VLC 打开视频自动加载）
# 依赖 StaticPy 的部分用绝对路径（/opt/StaticPy）；翻译需 llama.cpp 服务（未启动则自动拉起）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLM_HOST_URL="${LLM_HOST:-http://127.0.0.1:8080/v1}"

offset=""
lang="zh"
src_lang=""
files=()
for a in "$@"; do
  case "$a" in
    --offset=*)   offset="${a#--offset=}" ;;
    --lang=*)     lang="${a#--lang=}" ;;
    --src-lang=*) src_lang="${a#--src-lang=}" ;;
    *)            files+=("$a") ;;
  esac
done

# 强制源语言（跳过 Whisper 自动检测），避免误检（如日语被认成英语）
if [ -n "$src_lang" ]; then export WHISPER_LANG="$src_lang"; fi

# 无参数则扫描当前目录的视频
if [ ${#files[@]} -eq 0 ]; then
  shopt -s nullglob nocaseglob
  files=( *.mp4 *.mkv *.avi *.mov *.flv *.wmv *.m4v *.webm *.mpeg *.mpg *.ts *.m2ts )
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

is_video() {
  case "${1,,}" in
    *.mp4|*.mkv|*.avi|*.mov|*.flv|*.wmv|*.m4v|*.webm|*.mpeg|*.mpg|*.ts|*.m2ts) return 0 ;;
    *) return 1 ;;
  esac
}

t0=$(date +%s)
for v in "${files[@]}"; do
  base="${v%.*}"                          # 去掉扩展名（末个 . 之后）
  [ "$base" = "$v" ] && base="$v"         # 无扩展名

  # ① 抽取音频（16k 单声道 WAV，无损、解码快；体积远小于视频）
  if is_video "$v"; then
    audio="${base}.wav"
    if [ -f "$audio" ]; then
      echo ">>> [1/3] 已存在音频: $audio"
    else
      echo ">>> [1/3] 抽取音频: $v -> $audio"
      ffmpeg -nostdin -v error -y -i "$v" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$audio"
      echo "        音频大小: $(du -h "$audio" | cut -f1)"
    fi
  else
    audio="$v"                            # 已是音频文件
  fi

  # ② 语音识别（输出 <base>_<语言>.ass）
  echo ">>> [2/3] transcribe: $audio"
  targs=()
  [ -n "$offset" ] && targs+=("--offset=$offset")
  bash "$HERE/transcribe.sh" "${targs[@]}" "$audio"

  shopt -s nullglob
  produced=( "${base}"_*.ass )
  shopt -u nullglob
  if [ ${#produced[@]} -eq 0 ]; then
    echo "  ! 未找到转写产物 ${base}_*.ass，跳过翻译"
    continue
  fi
  src="$(ls -t "${produced[@]}" | head -1)"   # 取最新生成的

  # ③ 翻译成双语（输出与视频同名的 <base>.ass，VLC 打开视频时自动加载）
  echo ">>> [3/3] translate: $src"
  bash "$HERE/translate.sh" --lang="$lang" --out="${base}.ass" "$src"
done
echo "完成，用时 $(( $(date +%s) - t0 ))s"
