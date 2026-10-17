#!/usr/bin/env bash
# nas_subtitle.sh — 直接处理 NAS(WebDAV) 上的视频，生成中日/中英双语字幕并打开 VLC 播放。
#
# 特点：不下载整包！用 ffmpeg 经 HTTP Range 只拉音轨；字幕按「视频同名」回传 NAS 同目录，
#       VLC 打开原视频即自动加载；最后流式打开 VLC 播放。
#
# 用法: 只需给出 NAS 上的完整路径（或 dav:// URL）即可：
#   nas_subtitle.sh "/sata11-18072858035/av/.../video.mp4"
#   nas_subtitle.sh "dav://192.168.1.53:5005/sata11-18072858035/av/.../video.mp4"
#   nas_subtitle.sh "/sata11-.../video.mp4" --lang=ja      # 目标语言（默认 zh）
#   nas_subtitle.sh "/sata11-.../video.mp4" --src-lang=ja  # 强制源语言（跳过自动检测）
#   nas_subtitle.sh "/sata11-.../video.mp4" --offset=0.3   # 透传给转写的时间偏移
#   nas_subtitle.sh "/sata11-.../video.mp4" --no-vlc       # 只生成字幕，不打开 VLC
#   nas_subtitle.sh "/sata11-.../video.mp4" --dry-run      # 只解析 URL，不实际处理
#
# 依赖: ~/.env 中的 WEBDAV_URL / WEBDAV_USER / WEBDAV_PASS
# 语言: 源语言自动检测（Whisper），目标语言默认中文，可用 --lang 指定。
set -euo pipefail
export LC_ALL=C.UTF-8

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${NAS_WORKDIR:-$HOME/nas}"
mkdir -p "$WORK"

# --- 读取 NAS 凭据 ---
if [ -f "$HOME/.env" ]; then set -a; . "$HOME/.env"; set +a; fi
: "${WEBDAV_URL:?缺少 WEBDAV_URL（见 ~/.env）}"
: "${WEBDAV_USER:?缺少 WEBDAV_USER（见 ~/.env）}"
: "${WEBDAV_PASS:?缺少 WEBDAV_PASS（见 ~/.env）}"

# --- 参数 ---
src="${1:?用法: nas_subtitle.sh <dav://host:port/路径/video.mp4> [--lang=zh] [--no-vlc] [--offset=秒] [--dry-run]}"
shift
LANG_TGT="zh"; OPEN_VLC=1; DRY=0; PASS=()
for a in "$@"; do
  case "$a" in
    --lang=*)  LANG_TGT="${a#--lang=}";;
    --no-vlc)  OPEN_VLC=0;;
    --dry-run) DRY=1;;
    *)         PASS+=("$a");;
  esac
done

# --- 解析输入（只给 NAS 完整路径即可；也支持 dav:// / http:// / host:port/path） ---
rest="${src#dav://}"; rest="${rest#http://}"; rest="${rest#https://}"
if [ "${rest#/}" = "$rest" ]; then
  hostport="${rest%%/*}"                       # 形如 host:port/path
  path="/${rest#*/}"
else
  hostport="${WEBDAV_URL#http://}"; hostport="${hostport#https://}"; hostport="${hostport%%/*}"
  path="$rest"                                 # 纯路径 /sata11-.../video.mp4
fi

enc()      { python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
enc_path() { python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe="/"))' "$1"; }

EPATH="$(enc_path "$path")"
DIRU="${EPATH%/*}/"                       # 视频所在目录（用于回传字幕）
base="$(basename "$path")"; base="${base%.*}"   # 去扩展名，用于本地文件名

PLAIN="http://${hostport}${EPATH}"                                    # 明文 URL
AUTH="http://$(enc "$WEBDAV_USER"):$(enc "$WEBDAV_PASS")@${hostport}${EPATH}"   # 带凭据（流式/下载）
PUTURL="http://${hostport}${DIRU}${base}.ass"                         # 字幕回传目标

echo "源   : $src"
echo "基准 : $base   (工作目录 $WORK)"
if [ "$DRY" = 1 ]; then
  echo "明文 : $PLAIN"
  echo "字幕 : $PUTURL"
  exit 0
fi

# --- ① 抽音轨（HTTP Range，仅拉音频，远小于视频） ---
wav="$WORK/$base.wav"
if [ -s "$wav" ]; then
  echo ">>> [1/4] 复用已有音轨: $wav"
else
  echo ">>> [1/4] 抽音轨（仅拉音频，不下载整包）..."
  t0=$(date +%s)
  ffmpeg -nostdin -v warning -i "$AUTH" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$wav"
  echo "    抽音轨用时 $(( $(date +%s) - t0 ))s  ($(du -h "$wav" | cut -f1))"
fi

# --- ②③ 转写 + 翻译（对 .wav 输入，subtitle.sh 会直接转写并翻译） ---
echo ">>> [2-3/4] 转写 + 翻译（源语言自动检测 → $LANG_TGT）..."
bash "$HERE/subtitle.sh" --lang="$LANG_TGT" ${PASS[@]+"${PASS[@]}"} "$wav"

ass="$WORK/$base.ass"
[ -s "$ass" ] || { echo "未生成双语字幕: $ass" >&2; exit 1; }

# --- ④ 回传 NAS 同目录同名（VLC 打开原视频即自动加载） ---
echo ">>> [4/4] 回传字幕到 NAS..."
code="$(curl -s -T "$ass" -u "$WEBDAV_USER:$WEBDAV_PASS" "$PUTURL" -o /dev/null -w '%{http_code}')"
echo "    PUT -> HTTP $code"

# --- 流式打开 VLC 播放 ---
if [ "$OPEN_VLC" = 1 ]; then
  echo ">>> 打开 VLC 播放（NAS 流式）..."
  pkill -x vlc 2>/dev/null || true
  sleep 1
  DISPLAY="${DISPLAY:-:0}" setsid vlc --sub-file="$ass" "$AUTH" >/dev/null 2>&1 </dev/null &
  disown 2>/dev/null || true
fi
echo "完成。"
