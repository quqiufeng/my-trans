# my-trans

AI 时代已经来临，解放双手！

基于 AI 的视频字幕生成与翻译工具，支持多语言自动检测，生成中文双语字幕。

AI-powered video subtitle generation and translation tool. Supports automatic language detection for multilingual videos, generating Chinese bilingual subtitles.

> 本项目已本地化：语音识别与翻译均为 **C++ 库 + [StaticPy](https://github.com/) 编译的独立二进制**，
> 翻译走本地 **llama.cpp + 腾讯 Hunyuan-MT-7B** 服务，无需 pip / Python 运行时。

---

## 推荐工作流程

### 高质量双语字幕制作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  步骤 0：yt-dlp      →  下载 YouTube 视频                        │
│      ↓                                                           │
│  步骤 1：transcribe   →  生成英文字幕 (Whisper)                 │
│      ↓                                                           │
│  步骤 2：人工校对    →  修正识别错误、专有名词                    │
│      ↓                                                           │
│  步骤 3：translate   →  Hunyuan-MT 翻译成中文                    │
│      ↓                                                           │
│  输出：视频名.ass   →  精准双语字幕 (对齐时间轴)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 步骤详解

#### 步骤 0：下载 YouTube 视频

使用 **yt-dlp** 下载高质量视频素材。

```bash
# 下载最高质量视频
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" "https://www.youtube.com/watch?v=..."

# 下载视频 + 自动字幕
yt-dlp --write-subs --sub-langs "en" "URL"

# 下载整个播放列表
yt-dlp -cit "https://www.youtube.com/playlist?list=PL..."

# 仅下载音频（适合播客）
yt-dlp -x --audio-format mp3 "URL"
```

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-f best` | 最佳质量 |
| `-o "%(title)s.%(ext)s"` | 自定义文件名 |
| `--cookies frombrowser` | 使用浏览器 cookies |
| `-v` | 详细输出 |

**输出示例**：
```
$ yt-dlp "https://www.youtube.com/watch?v=abc123"
[youtube] abc123: Downloading webpage
[youtube] abc123: Downloading player 0f3f2b0
[download] Destination: OCaml Tutorial Part 1 - Getting Started.mp4
```

#### 步骤 1：生成英文字幕

```bash
# 转录音频，生成英文字幕
bash transcribe.sh video.mp4

# 常用参数
bash transcribe.sh video.mp4 --offset=0.3    # 时间偏移校正
```

输出：`video_en.ass`

#### 步骤 2：人工校对英文

**重要性**：英文准确 = 翻译基础

**校对要点**：
- 修正 Whisper 识别错误（特别是专业术语）
- 统一专有名词翻译（如 OCaml, Cornell, CS3110）
- 修正语法和拼写
- 调整字幕断句（过长/过短）

**工具**：
- VS Code + 支持 ASS 语法高亮
- 或任意文本编辑器

**示例校对**：
```diff
- # 原始识别（可能有误）
- few years, I've taught OCaml to about 3 ,700
+ # 校对后
+ few years, I've taught OCaml to about 3,700
```

#### 步骤 3：翻译成中文

```bash
# 扫描当前目录所有 ASS 文件
bash translate.sh

# 或指定单个文件
bash translate.sh video_en.ass

# 指定目标语言（默认 zh）
bash translate.sh --lang=ja video_en.ass
```

**特点**：
- **整句翻译**（拉丁语系源，如英/法/德）：把连续字幕按「句末标点 / 停顿 /
  说话人切换」聚合成完整句子，整句送 Hunyuan-MT，再按原文字数比例**确定性回切**
  成同样的条数。解决 Whisper 把一句切成多条导致的碎片翻译。
- **逐条翻译**（中日韩等不分词源）：Whisper 分段往往已是短语级，逐条翻译避免
  聚合回切拆散专名、错位时间轴。默认按源语文字**自动选择**。
- 温度 0：确定性输出，不合并不拆分
- 时间轴不变：只改写 Dialogue 文本，双语以 `\N` 换行

输出：`video_en.ass` → `video.ass`（双语）

### 完整命令

```bash
# 一步：视频 → 双语字幕
bash subtitle.sh "我的视频.mp4"

# 或分步：
# 1) 生成英文字幕   2) 人工校对 video_en.ass   3) 翻译成中文
bash transcribe.sh "我的视频.mp4"
bash translate.sh

# 4. 得到双语字幕
ls *.ass
# Introduction_OCaml_Programming_Chapter_1_Video_1_en.ass  (原版)
# Introduction_OCaml_Programming_Chapter_1_Video_1.ass     (双语)
```

### 为什么这个流程质量高？

| 环节 | 作用 | 质量保障 |
|------|------|----------|
| Whisper 语音识别 | 生成原始字幕 | AI 自动处理 |
| 人工校对英文 | 修正原文错误 | 人工确保准确 |
| 整句翻译 | 生成中文 | Hunyuan-MT 专用翻译模型 + 确定性回切保证 1:1 对齐 |

**核心优势**：
1. **英文准确** - 人工校对确保原文正确
2. **中文地道** - Hunyuan-MT 专用翻译模型
3. **时间轴对齐** - 翻译后按字数比例回切，保证条数与时轴 1:1 对应

---

## 快速开始 / Quick Start

### 一步生成双语字幕

```bash
# 扫描当前目录的视频，自动 转写 + 翻译，输出双语 ASS
bash subtitle.sh

# 或指定视频 / 目标语言
bash subtitle.sh video.mp4
bash subtitle.sh --lang=ja video.mp4
bash subtitle.sh --offset=0.3 video.mp4
```

`subtitle.sh` 会自动完成：**① ffmpeg 抽取音频**（`video.wav`，16k 单声道无损，体积远小于视频）
→ **② `transcribe`**（生成 `video_en.ass`）→ **③ `translate`**（生成双语 `video.ass`）；
翻译服务（llama.cpp Hunyuan-MT）未启动时会自动拉起。

### 分步执行（可控）

```bash
# 1. 语音识别（生成英文字幕）
bash transcribe.sh video.mp4

# 2. 启动本地翻译服务（Hunyuan-MT，若尚未启动）
bash start_server.sh &

# 3. 翻译字幕（生成双语 ASS）
bash translate.sh video.ass
```

---

## 超大视频 / NAS(WebDAV)：直接生成双语字幕

> 场景：视频动辄几 GB ~ 十几 GB（甚至数小时），存放在 NAS / WebDAV 上。
> 目标：**不下载整包**，直接生成中日/中英双语字幕，并让 VLC 流式播放时自动加载。

### 思路

1. **字幕只需要音频**：十几 GB 的 4K 视频，音轨往往只有几百 MB。
   ffmpeg 支持 **HTTP Range**，可直接从 WebDAV URL 边读边抽，只取音轨，不落地整段视频。
2. **抽成 16k 单声道无损 WAV**：体积小，且避免有损重编码影响分句
   （实测有损 AAC 会把短句合并，字幕条数变少）。
3. **源语言自动检测**（Whisper），目标语言默认中文，一步完成转写 + 翻译成双语 ASS。
4. **字幕按「视频同名」回传 NAS 同目录**：VLC 打开原视频即自动加载，无需手动选字幕。
5. **VLC 用带凭据的 URL 流式播放**，拖动进度也走 HTTP Range。

### 操作：只给一个 NAS 完整路径

```bash
# 只需 NAS 上的完整路径（或 dav:// URL），其余全自动
bash nas_subtitle.sh "/sata11-18072858035/av/.../video.mp4"

# 等价写法（dav:// URL）
bash nas_subtitle.sh "dav://192.168.1.53:5005/sata11-18072858035/av/.../video.mp4"

# 选项
#   --lang=ja     目标语言（默认 zh）
#   --offset=0.3  转写时间偏移校正
#   --no-vlc      只生成字幕，不打开 VLC
#   --dry-run     只解析 URL，不实际处理
```

脚本自动完成 4 步：

| 步骤 | 动作 |
|------|------|
| ① 抽音轨 | ffmpeg 经 **HTTP Range** 从 NAS 只拉音轨 → `~/nas/<视频名>.wav`（16k 单声道无损） |
| ② 转写 | `transcribe`，**源语言自动检测** → `<视频名>_<lang>.ass` |
| ③ 翻译 | `translate` → 双语 `<视频名>.ass` |
| ④ 回传 + 播放 | 字幕 **PUT 回 NAS 同目录同名**；`vlc` 流式打开原视频（自动加载字幕） |

依赖 `~/.env` 中的 `WEBDAV_URL / WEBDAV_USER / WEBDAV_PASS`；
工作目录默认 `~/nas`（可用环境变量 `NAS_WORKDIR` 覆盖）。

### 手工分步（与脚本等价）

```bash
set -a; . ~/.env; set +a
P='/sata11-18072858035/av/.../video.mp4'          # NAS 上的完整路径
URL=$(python3 -c "import urllib.parse,sys;u,user,pw,path=sys.argv[1:5];print(u.replace('http://','http://%s:%s@'%(urllib.parse.quote(user,safe=''),urllib.parse.quote(pw,safe='')))+urllib.parse.quote(path))" "$WEBDAV_URL" "$WEBDAV_USER" "$WEBDAV_PASS" "$P")

# ① 只抽音轨（不下载整包）
ffmpeg -i "$URL" -vn -ac 1 -ar 16000 -c:a pcm_s16le video.wav
# ②③ 转写 + 翻译（自动检测源语言 → 中文）
bash subtitle.sh video.wav
# ④ 回传字幕（同目录同名）+ VLC 流式播放
curl -T video.ass -u "$WEBDAV_USER:$WEBDAV_PASS" "http://192.168.1.53:5005/.../video.ass"
vlc "http://用户名:密码@192.168.1.53:5005/.../video.mp4"
```

### 实测耗时（RTX 3080，Whisper turbo + Hunyuan-MT）

| 视频 | 时长 | 大小 | 抽音轨 | 转写 | 翻译 | 总计 | 折合 |
|------|------|------|--------|------|------|------|------|
| JUQ-510 | 4h47m | 12.97GB | 1m54s | 314s | 197s | ~10m24s | 2m10s / 小时 |
| mida-040-4k | 2h04m | 10.5GB (4K) | 1m35s | 133s | 95s | ~6m03s | 2m36s / 小时 |
| ACHJ-052 | 2h32m | 6.88GB | 1m03s | 172s | 183s | ~6m58s | 2m45s / 小时 |

**平均约 0.04× 实时（每 1 小时视频约 2m25s，约 25× 加速）。**
抽音轨只传输音轨体积（如 13GB 视频仅拉 ~550MB）。

---

## llama.cpp 服务配置

本地通过 **llama.cpp** 加载腾讯 **Hunyuan-MT-7B** 翻译专用模型，提供 OpenAI 兼容 API。

- **服务端**: `/opt/llama.cpp/build/bin/llama-server`
- **模型**: `/data/models/Hunyuan-MT-7B-Q4_K_M.gguf`
- **API**: `http://127.0.0.1:8080/v1` (OpenAI 兼容)

### 启动服务

```bash
bash start_server.sh
```

或手动启动：

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /data/models/Hunyuan-MT-7B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -a hunyuan-mt
```

### Hunyuan-MT-7B 模型介绍

**官方链接**: https://huggingface.co/tencent/Hunyuan-MT-7B

**模型架构**:
- 类型: Decoder-only Transformer (hunyuan-dense)
- 参数规模: 7B (70亿参数)，此处使用 Q4_K_M GGUF 量化
- 语言: 支持 33 种语言互译（含中英日韩等）

**性能特点**:
- 专用机器翻译模型，翻译质量高
- GGUF 量化 + llama.cpp CUDA 加速，消费级显卡可跑
- 偏好完整句子输入；整句翻译后由代码确定性回切最稳定

**字幕翻译用法**:
- 官方推荐 prompt：`Translate the following segment into Chinese, without additional explanation.`
- `translate` 整句翻译 + 确定性回切，保证时间轴 1:1 对齐；不分词语言自动逐条

---

## 使用方法

### 1. 语音识别（生成英文字幕）

```bash
bash transcribe.sh video.mp4
```

输出: `video_en.ass` 或 `video_zh.ass`（根据检测语言）

**参数**:
- `--offset=0.3` - 字幕时间偏移校正（秒）
- `文件...` - 指定文件（不带则扫描当前目录）

### 2. 翻译字幕

```bash
# 翻译英文字幕
bash translate.sh video_en.ass

# 指定目标语言（默认 zh）
bash translate.sh --lang=ja video_en.ass

# 扫描当前目录所有 ASS 文件
bash translate.sh
```

输出: `video.ass`（双语，与视频同名 → VLC 打开视频时自动加载）

### 支持语言

| 语言代码 | Language |
|---------|----------|
| `zh` | Chinese |
| `en` | English |
| `ja` | Japanese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ru` | Russian |
| `it` | Italian |
| `pt` | Portuguese |
| `ar` | Arabic |
| `th` | Thai |
| `vi` | Vietnamese |
| `id` | Indonesian |

---

## 脚本说明

语音识别与翻译均由 **StaticPy** 编译为独立 ELF 二进制，底层是自写的 C++ C-ABI 库。
源码是 Python 子集（`*.static.py`），用 `bash build.sh` 编译；运行时经包装脚本（设置 `LD_LIBRARY_PATH`）。

### transcribe（语音识别）

| 文件 | 说明 |
|------|------|
| `transcribe.static.py` | StaticPy 源码：扫描目录/按参数批量转写 |
| `transcribe_ffi.scm` | FFI：`load-shared-object libwhisper.so` |
| `transcribe.sh` | 运行包装器（设置 `LD_LIBRARY_PATH`） |

底层 `/opt/StaticPy/cpp/whisper/`：CTranslate2 C++ Whisper + 自写 log-mel（fftw3f）前端。

### translate（字幕翻译）

| 文件 | 说明 |
|------|------|
| `translate.static.py` | StaticPy 源码：扫描目录/按参数批量翻译 |
| `translate_ffi.scm` | FFI：`load-shared-object libllm.so` |
| `translate.sh` | 运行包装器（设置 `LD_LIBRARY_PATH`） |

### subtitle（一步流水线）

| 文件 | 说明 |
|------|------|
| `subtitle.sh` | 视频 →（抽音频）→ `transcribe` → `translate`，一步产出双语 ASS；服务未启动则自动拉起 |

### nas_subtitle（NAS / 超大视频一步流水线）

| 文件 | 说明 |
|------|------|
| `nas_subtitle.sh` | 只给 NAS 完整路径：经 **HTTP Range 只抽音轨**（不下载整包）→ `transcribe` → `translate` → 双语字幕**回传 NAS 同目录同名** → **VLC 流式播放**（自动加载字幕） |

底层 `/opt/StaticPy/cpp/llm/`：`libllm.so`，自写 HTTP/1.1 客户端调用 llama.cpp `/v1/chat/completions`，
C ABI 为 `llm_translate(text, target_name)`、`llm_translate_ass(in, out, lang)`、`llm_host()`。

### 编译

```bash
bash build.sh
```

会依次：
1. 编译 `libwhisper.so`（`/opt/StaticPy/cpp/whisper`）
2. 编译 `libllm.so`（`/opt/StaticPy/cpp/llm`）
3. StaticPy AOT 编译 `transcribe` / `translate` 两个二进制

> 依赖 StaticPy 工具链与 C++ 源码，路径固定为 `/opt/StaticPy`。

---

## 目录结构

```
my-trans/
├── transcribe.static.py       # 语音识别源码（StaticPy，Python 子集）
├── transcribe_ffi.scm         # FFI：加载 libwhisper.so
├── transcribe.sh              # 运行 transcribe（设置 LD_LIBRARY_PATH）
├── translate.static.py        # 字幕翻译源码（StaticPy）
├── translate_ffi.scm          # FFI：加载 libllm.so
├── translate.sh               # 运行 translate（设置 LD_LIBRARY_PATH）
├── subtitle.sh                # 一步流水线：视频 → 转写 → 翻译 → 双语 ASS
├── nas_subtitle.sh            # NAS/超大视频：只抽音轨 → 双语字幕 → 回传 NAS → 播放 VLC
├── build.sh                   # 编译两个二进制（依赖 /opt/StaticPy）
├── start_server.sh            # 启动 llama.cpp 翻译服务
├── res/                       # 原始文件目录
│   ├── video.mp4              # 视频文件
│   └── video_en.ass           # 英文字幕
└── output/                    # 输出文件目录
    └── video.ass              # 双语字幕
```

---

## 环境配置

### 运行时
- **无需 Python 运行时**：`transcribe` / `translate` 为静态编译的独立二进制
- C++ 依赖：CTranslate2（Whisper）、fftw3f（log-mel）、libcurl/自写 HTTP（翻译）

### 编译依赖
- **StaticPy** 工具链（`/opt/StaticPy`）
- **g++**（C++17）
- Nativa 头文件：`nlohmann/json.hpp`（翻译）、CT2 头文件（识别）
- **llama.cpp** 服务端（Hunyuan-MT GGUF 模型）
  - `/opt/llama.cpp/build/bin/llama-server`

---

## 字体配置

默认使用 **仓耳今楷02** 字体。

**安装**:
1. 下载仓耳今楷02 字体（TsangerJinKai02）
2. 安装到系统（Linux: 复制到 `~/.local/share/fonts/` 后执行 `fc-cache -f`）

**字幕样式**:
| 样式 | 字号 |
|------|------|
| Default | 46 |
| Top | 40 |
| Comment | 34 |

如需更换字体，编辑各脚本中的 `Style` 行。

---

## 技术架构

### 处理流程

```
视频.mp4 → ffmpeg 抽音频 video.wav → transcribe → video_en.ass → translate → video.ass（双语）
```

### 核心模型

| 功能 | 模型 | 说明 |
|------|------|------|
| 语音识别 | Whisper **large-v3-turbo**（CT2） | CTranslate2 C++ 引擎 + 自写 log-mel 前端，编译为独立二进制 |
| 字幕翻译 | Hunyuan-MT-7B-Q4_K_M | 腾讯混元翻译专用模型，llama.cpp + CUDA |

### 语音识别模型

- **`large-v3-turbo`**：encoder 同 large-v3、decoder 仅 4 层，推理快约 **3×**
  （实测 145s 音频 1.75s，RTX 3080），多语言，支持自动检测源语言。

**环境变量**（`transcribe.sh` / `subtitle.sh` 均适用）：

| 变量 | 说明 | 默认 |
|------|------|------|
| `WHISPER_MODEL_DIR` | 模型目录 | `/data/models/faster-whisper-large-v3-turbo` |
| `WHISPER_BEAM` | 束搜索宽度（1=贪心，略快） | `5` |
| `WHISPER_DEVICE` | 设为 `cpu` 强制 CPU | CUDA/FP16 |
| `WHISPER_LANG` | 强制源语言（`ja`/`zh`/`en`…），跳过自动检测 | 自动 |
| `WHISPER_VAD` | 设为 `0` 关闭 Silero VAD 语音门控 | 开 |
| `WHISPER_VAD_MODEL` | VAD 模型路径 | `/data/models/silero-vad/silero_vad.onnx` |
| `WHISPER_VAD_THRESHOLD` | 语音概率阈值 | `0.5` |
| `WHISPER_MAX_SEG_SEC` | 丢弃长于该秒数的段（整窗填充伪影） | `15` |
| `WHISPER_PROFILE` | 非空则打印分段耗时 | 关 |

> **VAD 语音门控**：背景音乐/静音段不送 Whisper，从根源消除「无人说话却配字幕」
> 的幻觉（尤其是高频 BGM 的影片）。依赖 `libonnxruntime` 与 Silero VAD 模型；
> 缺任一则自动降级为无 VAD。`subtitle.sh` / `nas_subtitle.sh` 可用 `--src-lang=ja`
> 强制源语言；`transcribe.sh` 直接用 `WHISPER_LANG=ja bash transcribe.sh …`。

### 人声分离前处理（Demucs）

对白被很响的 BGM/环境音盖住（信噪比低）时，VAD 只能去掉纯静音，盖住人声的部分
仍会大量**复读/错听幻觉**。默认在送识别前先用 **Demucs** 分离出纯人声
（`<名字>.vocals.wav`，缓存复用），再送去 Whisper，可显著降低幻觉并修正错字。

**环境变量 / 参数**（`subtitle.sh` / `nas_subtitle.sh` 适用）：

| 变量 | 说明 | 默认 |
|------|------|------|
| `SUB_SEPARATE` | 设为 `0` 关闭人声分离 | `1`（开） |
| `SUB_PYTHON` | 跑 Demucs 的解释器（需含 torch/demucs） | `/data/venv/bin/python` |
| `SUB_DEMUCS_MODEL` | Demucs 模型名 | `htdemucs` |

也可用 `--no-separate` 单次关闭。解释器缺 `demucs` 时自动降级为不分离。

> 安装（一次性）：`/data/venv/bin/pip install demucs sphn julius lameenc dora-search`。

### 技术栈

- **CTranslate2 (C++)**：Whisper 语音识别，StaticPy 编译成独立二进制
- **fftw3f**：自写 log-mel 频谱前端
- **StaticPy**：Python 子集 → Scheme → AOT ELF，挂载 C 库 FFI
- **llama.cpp**: 本地 LLM/MT 推理服务（OpenAI 兼容 API + CUDA）

---

## AI 协作

| 角色 | 工具 | 说明 |
|------|------|------|
| 需求与开发 | OpenCode | AI 编程助手 |
| 语音识别 | StaticPy + CTranslate2 | CTranslate2 C++ Whisper（默认 large-v3-turbo），自写 log-mel 前端 |
| 字幕翻译 | StaticPy + llama.cpp + Hunyuan-MT | 腾讯翻译模型，CUDA 加速 |

---

## 字幕翻译：整句翻译 + 确定性回切

> **策略**（借鉴 BaoCut 字幕管线）
>
> 逐条翻译字幕碎片（Whisper 常把一句话切成多条）会丢失上下文、断句生硬。
> 现在默认**整句翻译**：把连续字幕聚合成完整句子整句送模型，再按原文字数比例
> **确定性回切**成同样的条数，时间轴与条数不变。回切只在合法缝切（不切单词、
> 不悬标点、数字与单位之间不切），并优先落在标点处。
>
> 中日韩等不分词语言的 Whisper 分段往往已是短语级，强行聚合再回切会拆散专名
> （如「武藤」被切成「武 / 藤」），因此**按源语文字自动选择**：拉丁语系整句、
> CJK 逐条；仍可用环境变量手动覆盖。

### 环境变量（`translate.sh` / `subtitle.sh` / `nas_subtitle.sh` 均适用）

| 变量 | 说明 | 默认 |
|------|------|------|
| `SUB_GROUP` | 整句翻译：`-1` 自动、`0` 关、`1` 强开 | `-1` |
| `SUB_PAUSE` | 句间停顿阈值（秒），越大越不易聚合 | `0.7` |
| `SUB_MAX_LINES` | 单句最多聚合几条字幕 | `8` |
| `SUB_MAX_CHARS` | 单句最多聚合多少字 | `400` |
| `SUB_ORDER_HINT` | 提示模型贴近源语小句顺序（同声传译做法） | `0` |

### 为什么不用「全文/批量」翻译

| 方案 | 结果 |
|------|------|
| 批量翻译（多条/批） | 模型会合并翻译，位置错乱 |
| 全文翻译 | 模型修改原文结构 |
| **整句翻译 + 确定性回切** | **上下文完整、条数与时间轴不变、最稳定** |

### 实现要点

1. **温度 0** - 确定性输出，字幕翻译不需要创意
2. **整句翻译** - 以完整句子（而非字幕碎片）为翻译单位，上下文完整
3. **确定性回切** - 代码按字数比例切回同样条数，不依赖模型分句
4. **双语拼接** - 原文 + `\N` + 译文，时间轴保持不变
5. **Prompt 固定** - `Translate the following segment into <目标语言>, without additional explanation.`
