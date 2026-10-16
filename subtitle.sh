#!/bin/bash
# subtitle.sh — 一步生成双语字幕：视频 → 抽音频 → transcribe → translate → 双语 ASS
#
# 用法:
#   bash subtitle.sh                    扫描当前目录的视频，逐个处理
#   bash subtitle.sh 视频1.mp4 [视频2.mp4 ...]
#   bash subtitle.sh --offset=0.3 视频.mp4
#   bash subtitle.sh --lang=ja 视频.mp4  # 指定目标语言（默认 zh）
#   bash subtitle.sh --no-separate 视频.mp4  # 关闭人声分离（默认开启）
#
# 步骤: ① ffmpeg 抽取音频(<名字>.wav，16k 单声道无损，体积小、便于复用)
#       ①' Demucs 分离人声(<名字>.vocals.wav)，过滤 BGM/环境音降低低信噪比幻觉
#       ② transcribe 生成 <名字>_<语言>.ass   ③ translate 生成双语 <名字>.ass
# 结果: <名字>_<语言>.ass（原文）+ <名字>.ass（双语，与视频同名 → VLC 打开视频自动加载）
# 依赖 StaticPy 的部分用绝对路径（/opt/StaticPy）；翻译需 llama.cpp 服务（未启动则自动拉起）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLM_HOST_URL="${LLM_HOST:-http://127.0.0.1:8080/v1}"

offset=""
lang="zh"
src_lang=""
separate="${SUB_SEPARATE:-1}"          # 1=送识别前先用 Demucs 分离人声（去 BGM/环境音）
sub_python="${SUB_PYTHON:-/data/venv/bin/python}"
demucs_model="${SUB_DEMUCS_MODEL:-htdemucs}"
sep_chunk="${SUB_SEPARATE_CHUNK:-600}"  # 分离分块秒数（长音频防 OOM）
files=()
for a in "$@"; do
  case "$a" in
    --offset=*)    offset="${a#--offset=}" ;;
    --lang=*)      lang="${a#--lang=}" ;;
    --src-lang=*)  src_lang="${a#--src-lang=}" ;;
    --separate)    separate=1 ;;
    --no-separate) separate=0 ;;
    *)             files+=("$a") ;;
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

  # ①' 人声分离：Demucs 去掉 BGM/环境音，显著降低低信噪比下的复读/错听幻觉
  in_audio="$audio"
  if [ "$separate" = 1 ] && [ -x "$sub_python" ] && \
     "$sub_python" -c 'import demucs' >/dev/null 2>&1; then
    vocals="${base}.vocals.wav"
    if [ -s "$vocals" ]; then
      echo ">>> [1'/3] 复用已分离人声: $vocals"
    else
      echo ">>> [1'/3] Demucs 分离人声（$demucs_model，分块 ${sep_chunk}s 防 OOM）..."
      sepout="$(mktemp -d)"
      # 长音频一次性载入会吃爆内存，按块分割后逐块分离，再拼接
      ffmpeg -nostdin -v error -y -i "$audio" -f segment -segment_time "$sep_chunk" \
             -reset_timestamps 1 -c:a pcm_s16le "$sepout/chunk_%05d.wav"
      : > "$sepout/list.txt"
      for c in "$sepout"/chunk_*.wav; do
        [ -s "$c" ] || continue
        "$sub_python" -m demucs --two-stems=vocals -n "$demucs_model" -o "$sepout/out" "$c" >/dev/null 2>&1 || true
        vc="$sepout/out/$demucs_model/$(basename "${c%.wav}")/vocals.wav"
        [ -s "$vc" ] && printf "file '%s'\n" "$vc" >> "$sepout/list.txt"
      done
      if [ -s "$sepout/list.txt" ]; then
        ffmpeg -nostdin -v error -y -f concat -safe 0 -i "$sepout/list.txt" -ac 1 -ar 16000 -c:a pcm_s16le "$vocals"
      fi
      rm -rf "$sepout"
    fi
    if [ -s "$vocals" ]; then in_audio="$vocals"; else echo "  ! 分离失败，改用原始音轨"; fi
  fi

  # ② 语音识别（输出 <base>_<语言>.ass）
  echo ">>> [2/3] transcribe: $in_audio"
  targs=()
  [ -n "$offset" ] && targs+=("--offset=$offset")
  bash "$HERE/transcribe.sh" "${targs[@]}" "$in_audio"

  # 识别产物名跟随输入文件名，分离后需改回原基准名（<base>.vocals_<lang>.ass → <base>_<lang>.ass）
  if [ "$in_audio" != "$audio" ]; then
    shopt -s nullglob
    for f in "${base}".vocals_*.ass; do mv -f "$f" "${base}_${f#"${base}".vocals_}"; done
    shopt -u nullglob
  fi

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
