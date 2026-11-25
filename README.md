# my-trans

**本地视频字幕工具**：给一段视频，产出与视频同名的**双语 ASS 字幕**。

两件事，别的都不做：

1. **本地视频** → 抽音轨 → Whisper 转写 → Hunyuan-MT 翻译 → `<视频名>.ass`（原文 + 译文）
2. **NAS / WebDAV 上的视频** → 只经 HTTP Range 拉音轨（不下载整包）→ 同上 → 字幕回传 NAS 同目录

源语言自动检测（英/日/韩/俄…33 种），目标语言默认中文。时间轴用音频能量做一次校正，
解决「还没开口字幕先出来」的老毛病。

> 全部本地跑：语音识别是 **CTranslate2 C++ Whisper**，翻译是 **llama.cpp + 腾讯 Hunyuan-MT-7B**，
> 两层都用 **StaticPy** 编译成独立 ELF 二进制。运行时不需要 Python、不需要联网、不需要 pip。

---

## 快速开始

```bash
# 本地视频（当前目录下所有视频，或指定文件）
bash subtitle.sh "我的视频.mkv"
bash subtitle.sh --lang=ja --offset=0.3 "我的视频.mp4"

# NAS 上的视频：给一个完整路径就够
bash nas_subtitle.sh "/sata11-18072858035/某电影/S01/E01.mkv"

# 只出字幕，不打开播放器
bash nas_subtitle.sh "/sata11-.../E01.mkv" --no-vlc
```

产出（与视频同目录 / 同名，VLC 打开视频自动加载）：

```
视频名.mkv          ← 原视频
视频名_en.ass       ← 原文（源语言自动检测）
视频名.ass          ← 双语（原text\N译文），播放器加载这一个
```

---

## 一步流水线 `subtitle.sh`

| 步骤 | 动作 | 产物 |
|---|---|---|
| ① | `ffmpeg` 抽取音频：16k 单声道 **无损 WAV** | `视频名.wav` |
| ② | `transcribe`（Whisper large-v3，源语言自动检测） | `视频名_<src>.ass` |
| ③ | `translate`（Hunyuan-MT，本地 llama.cpp 服务） | `视频名.ass`（双语） |
| ④ | `ass_retime`（音频能量重定时，`--no-retime` 可关） | 原地校正 ③ 的时间轴 |

要点：

- **抽成无损 WAV 而不是直接喂视频**：体积小、可复用；有损 AAC 会把短句粘在一起，字幕条数会变少。
- **翻译服务没起会自动拉起**（`start_server.sh`），不必手工开。
- 双语拼接用 `\N` 换行，**时间轴不变、条数不变**；样式在脚本里的 `Style` 行。

### 参数

| 参数 | 说明 |
|---|---|
| `--lang=xx` | 目标语言（默认 `zh`） |
| `--src-lang=xx` | 强制源语言，跳过自动检测（如日语被误判成英语时） |
| `--offset=0.3` | 转写时间偏移校正（秒） |
| `--no-retime` | 跳过第 ④ 步时间轴校正 |

### 分步执行（想人工校对原文时）

```bash
bash transcribe.sh video.mkv     # ① ② 出 video_en.ass
# … 手工改 video_en.ass …
bash translate.sh                # ③ 扫当前目录 ASS → video.ass
bash ass_retime.sh video.ass video.wav   # ④ 可选
```

---

## NAS / WebDAV：超大视频直接出字幕

> 场景：4K 原盘动辄十几 GB ~ 几十 GB，放在 NAS 上，不想先拷到本地。

```bash
bash nas_subtitle.sh "/sata11-18072858035/某剧/S01/E01.mkv" --no-vlc
```

| 步骤 | 动作 |
|---|---|
| `[1/4]` | `ffmpeg` 经 **HTTP Range** 只拉音轨 → `~/nas/<视频名>.wav`（十几 GB 的视频只传几百 MB）；已存在则复用 |
| `[2-3/4]` | 调 `subtitle.sh` → `transcribe`（自动检测源语）+ `translate`（双语）+ `ass_retime`（校正） |
| `[4/4]` | 双语字幕 **PUT 回 NAS 同目录同名** → 播放器打开原视频即自动加载 |
| 播放 | （默认）`vlc` 用带凭据的 URL 流式打开，拖动进度也走 Range；`--no-vlc` 只产出不播 |

**依赖**：`~/.env` 里的 `WEBDAV_URL`、`WEBDAV_USER`、`WEBDAV_PASS`。
**工作目录**：默认 `~/nas`，可用 `NAS_WORKDIR=/path` 覆盖。
**支持的输入**：NAS 完整路径（`/sata11-.../a.mkv`）、`dav://host:port/路径`、`http(s)://host:port/路径`。

| 参数 | 说明 |
|---|---|
| `--lang=xx` | 目标语言（默认 `zh`） |
| `--src-lang=xx` | 强制源语言 |
| `--offset=0.3` | 转写时间偏移 |
| `--no-vlc` | 只生成并回传字幕，不播 |
| `--dry-run` | 只解析 URL，不实际处理 |

### 手工等价写法

```bash
set -a; . ~/.env; set +a
P='/sata11-18072858035/.../video.mkv'

# URL 百分号编码（纯 bash，按字节；$2 是要保留的字符）
urlenc() { local s=$1 keep=${2-} out= c i; local LC_ALL=C
  for (( i=0; i<${#s}; i++ )); do c=${s:i:1}
    case $c in [A-Za-z0-9.~_-]) out+=$c ;;
      *) case $keep in *"$c"*) out+=$c ;; *) printf -v c '%%%02X' "'$c"; out+=$c ;; esac ;;
    esac; done; printf '%s' "$out"; }

EP=$(urlenc "$P" "/")
URL="http://$WEBDAV_USER:$WEBDAV_PASS@${WEBDAV_URL#http://}$EP"
PUT="http://${WEBDAV_URL#http://}${EP%/*}/$(basename "$P" | sed 's/\.[^.]*$//').ass"

ffmpeg -i "$URL" -vn -ac 1 -ar 16000 -c:a pcm_s16le video.wav   # ①
bash subtitle.sh video.wav                                     # ② ③ ④
curl -T video.ass -u "$WEBDAV_USER:$WEBDAV_PASS" "$PUT"
```

---

## 时间轴校正（retime）

> 症状：**还没开口字幕先出来**、**说完了字还挂着**。根因是 Whisper 的段边界有 ±0.5s 抖动，
> 时间戳来自声学对齐，而不是精确的语音起止。

`subtitle.sh` 第 ④ 步自动做（`--no-retime` 可关），也可单独跑：

```bash
bash ass_retime.sh video.ass video.wav            # 原地改写
bash ass_retime.sh --dry-run video.ass video.wav  # 只看统计，不写回
bash ass_retime.sh --stats  video.ass video.wav   # 打印位移分布
```

算法（`ass_retime.static.py`，AOT 编译成 `./ass_retime`）：

1. 10ms 一帧算 RMS 能量 → 阈值 `max(30 百分位 × 3, 0.006)` 判「有声 / 静音」
2. 短于 `0.30s` 的静音视为同一停顿，短于 `0.08s` 的碎片发声丢弃 → 得到真实发声段
3. 每条字幕的起止吸附到最近的真实语音边界（提前 `0.06s`、延后 `0.12s`，搜索半径 `0.40s`），
   总位移限制在 ±0.65s 内；若仍落在静音里则兜底修正
4. 强制去重叠

实测（Yellowstone S01E01，1061 段）：起点落在静音里的 319 → ~150 段，终点 294 → ~47 段，
段间重叠 0 处，并丢掉片头 18s 的幻听段。

二进制音频 I/O 这一小块由本项目的 C ABI 库 `cpp/asswav/` 提供
（`aw_open / aw_len / aw_rate / aw_rms / aw_bom`，`mmap` 只读映射，不复制整条音轨）。

---

## 翻译策略：整句翻译 + 确定性回切

逐条翻译 Whisper 的碎片会丢上下文、断句生硬。默认：

- **整句翻译**（拉丁语系）：把连续字幕按「句末标点 / 停顿 / 说话人切换」聚合成完整句子。
  组内每行加 `[1] … [n]` 行号标记**一次送模型**，上下文完整，模型逐行回同样号译文，
  于是逐行天然对齐；模型若丢标记/合并行，缺行再逐条补译，整组标记全丢才退回
  「整句 + 字数比例确定性回切」（只落在合法缝：不切单词、不悬标点、数字与单位之间不切）。
- **逐条翻译**（中日韩等不分词语言）：Whisper 分段往往已是短语级，聚合再回切会拆散专名
  （「武藤」→「武 / 藤」），所以逐条翻译。默认按源语文字**自动选择**。

温度 0，确定性输出；只改写 Dialogue 文本，时间轴与条数不变。

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `SUB_GROUP` | 整句翻译：`-1` 自动、`0` 关、`1` 强开 | `-1` |
| `SUB_PAUSE` | 句间停顿阈值（秒），越大越不易聚合 | `0.7` |
| `SUB_MAX_LINES` | 单句最多聚合几条字幕 | `8` |
| `SUB_MAX_CHARS` | 单句最多聚合多少字 | `400` |
| `SUB_ORDER_HINT` | 提示模型贴近源语小句顺序（同传做法） | `0` |
| `SUB_MARK` | 组内加 `[n]` 行号标记、整组一次翻译并逐行对齐 | `1` |

为什么不整篇丢给模型：全文翻译会改结构、改条数。批量（无标记）会让模型合并条目、
把整段译文压到前几行（整句译文错位到相邻行）——所以组内必须带 `[n]` 标记，
条数与时间轴才能和原文 1:1 对齐。

C ABI 侧（`cpp/llm/`）是工具链 `llm` 的**本项目副本**，唯一改动就是 `SUB_MARK`
标记翻译；产物 `libllm.so` 落在本项目根目录，由 `translate.sh` 的 `LD_LIBRARY_PATH` 加载。

---

## 翻译服务（llama.cpp + Hunyuan-MT-7B）

```bash
bash start_server.sh
```

等价手工命令：

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /data/models/Hunyuan-MT-7B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -a hunyuan-mt
```

- OpenAI 兼容 API：`http://127.0.0.1:8080/v1`（可用 `LLM_HOST` 覆盖）
- Hunyuan-MT-7B：腾讯翻译专用模型，7B / Q4_K_M，33 种语言互译；消费级显卡（8GB+）可跑
- 官方推荐 prompt：`Translate the following segment into Chinese, without additional explanation.`

---

## 语音识别配置

默认 **Whisper large-v3**（CTranslate2，CUDA FP16）。`transcribe.sh` / `subtitle.sh` /
`nas_subtitle.sh` 均认下列环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `WHISPER_MODEL_DIR` | 模型目录 | `/data/models/faster-whisper-large-v3` |
| `WHISPER_MODEL_DIR_JA` | 日语模型目录（可选） | 同 `WHISPER_MODEL_DIR` |
| `WHISPER_LANG` | 强制源语言（`ja`/`zh`/`en`…），跳过自动检测 | 自动 |
| `WHISPER_DEVICE` | 设为 `cpu` 强制 CPU | CUDA/FP16 |
| `WHISPER_CHUNK_SEC` | 解码窗长（秒）；kotoba/distil 系建议 `15` | `30` |
| `WHISPER_BEAM` | 束搜索宽度（1=贪心，略快） | `5` |
| `WHISPER_VAD` | VAD 语音门控：`1`/`0` 强制开关；未设则**仅日语开** | 按语言 |
| `WHISPER_VAD_MODEL` | Silero VAD 模型路径 | `/data/models/silero-vad/silero_vad.onnx` |
| `WHISPER_VAD_THRESHOLD` | 语音概率阈值 | `0.5` |
| `WHISPER_MAX_SEG_SEC` | 丢弃长于该秒数的段（去整窗填充伪影） | `25` |
| `WHISPER_LOGPROB_THRESHOLD` | 与 `no_speech` 配合：仅无语音概率高且置信度低于此值才丢窗 | `-1.0` |
| `WHISPER_MAX_CHAR_RUN` | 同一字连续重复 ≥ 该值才丢弃（默认关，避免误删真实发声） | `0`（关） |
| `WHISPER_DROP_PHRASES` | 幻觉短语黑名单（逗号分隔，默认空=不删） | 空 |
| `WHISPER_PROFILE` | 非空则打印分段耗时 | 关 |

> **VAD 语音门控**：BGM / 静音段不送 Whisper，从根源减少「没人说话却配字幕」的幻觉。
> 依赖 `libonnxruntime` + Silero VAD 模型，缺任一自动降级为无 VAD。
>
> 默认**不做**短语黑名单：仅凭文本无法区分「真说了」和幻觉，误删真台词不可接受。
> 需要时自行设 `WHISPER_DROP_PHRASES="ご視聴ありがとうございました,チャンネル登録"`。

### 支持语言（常用）

`zh` 中文 · `en` 英语 · `ja` 日语 · `ko` 韩语 · `fr` 法语 · `de` 德语 · `es` 西语 ·
`ru` 俄语 · `it` 意语 · `pt` 葡语 · `ar` 阿拉伯语 · `th` 泰语 · `vi` 越南语 · `id` 印尼语

---

## 脚本与目录

```
my-trans/
├── subtitle.sh             # 一步：本地视频 → 抽音频 → 转写 → 翻译 → 重定时 → 双语 ASS
├── nas_subtitle.sh         # 一步：NAS 视频 → 只抽音轨 → 同上 → 回传 NAS → VLC 播放
├── transcribe.sh           # 运行 transcribe（设置 LD_LIBRARY_PATH）
├── translate.sh            # 运行 translate（扫目录 / 按参数批量）
├── ass_retime.sh           # 运行 ass_retime（时间轴校正）
├── start_server.sh         # 启动 llama.cpp 翻译服务
├── build.sh                # 编译全部二进制（依赖 /opt/StaticPy）
├── transcribe.static.py    # 语音识别源码（StaticPy，Python 子集）
├── translate.static.py     # 字幕翻译源码（整句聚合 + 确定性回切）
├── ass_retime.static.py    # 时间轴校正源码
├── *_ffi.scm               # FFI：load-shared-object libwhisper.so / libllm.so / libasswav.so
└── cpp/asswav/             # C ABI 库：WAV → PCM（mmap）给 ass_retime 用
```

运行时产物：`transcribe` / `translate` / `ass_retime`（各约 16KB ELF）+ 各自 `.so` + `libasswav.so`。

---

## 编译

```bash
bash build.sh
```

依次：

1. `$ROOT/cpp/whisper/build.sh` → `libwhisper.so`（CTranslate2 + 自写 log-mel 前端 fftw3f）
2. `$ROOT/cpp/llm/build.sh` → `libllm.so`（自写 HTTP/1.1 客户端打 llama.cpp）
3. `cpp/asswav/build.sh` → `libasswav.so`（本项目自带，WAV 采样读取）
4. `$ROOT/staticpy/static_build.sh` AOT 编译 `transcribe` / `translate` / `ass_retime`
   （`STATICPY_OUTDIR` 指到本项目，产物不落工具链仓库）

`ROOT` 默认 `/opt/StaticPy`，可用 `STATICPY_ROOT=/path bash build.sh` 覆盖。
`build.sh` 自己不实现构建逻辑，只按顺序调上述入口。

- 产物走工具链默认共享运行时（`rpath → $ROOT/dist/libstaticpyrt.so`，约 16KB stub）；
  要单文件 ELF：`STATICPY_STATIC_RT=1 bash build.sh`（约 4.3MB）。
- `libwhisper.so` 的 `DT_RPATH` 指向原位 `thirdparty/ctranslate2/lib`，**不要移动**该目录。

### 编译依赖

- **StaticPy** 工具链（`/opt/StaticPy`）、`g++`（C++17）
- CTranslate2 + 头文件、`fftw3f`、`nlohmann/json.hpp`
- `libonnxruntime` + Silero VAD 模型（可选，用于 VAD）
- llama.cpp 服务端 + Hunyuan-MT GGUF 模型

运行时**完全不依赖 Python**：三个入口都是 StaticPy 编译的 ELF，URL 百分号编码也用纯 bash 实现
（`nas_subtitle.sh` 里的 `urlenc`）。

---

## 字体

默认 **仓耳今楷02**（TsangerJinKai02）。装到 `~/.local/share/fonts/` 后 `fc-cache -f`。

| 样式 | 字号 |
|---|---|
| Default | 46 |
| Top | 40 |
| Comment | 34 |

换字体改脚本里的 `Style` 行。

---

## 实测（RTX 3080）

| 视频 | 时长 | 大小 | 抽音轨 | 转写 | 翻译 | 总计 |
|---|---|---|---|---|---|---|
| JUQ-510 | 4h47m | 12.97GB | 1m54s | 314s | 197s | ~10m24s |
| mida-040-4k | 2h04m | 10.5GB (4K) | 1m35s | 133s | 95s | ~6m03s |
| ACHJ-052 | 2h32m | 6.88GB | 1m03s | 172s | 183s | ~6m58s |

平均约 **0.04× 实时**（1 小时视频约 2m25s）。抽音轨只传音轨体积（13GB 视频仅拉 ~550MB）。
