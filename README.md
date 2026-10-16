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
- 逐条翻译：每条字幕独立发送给 Hunyuan-MT，严格 1:1
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
| 逐条翻译 | 生成中文 | Hunyuan-MT 专用翻译模型 + 1:1 对齐 |

**核心优势**：
1. **英文准确** - 人工校对确保原文正确
2. **中文地道** - Hunyuan-MT 专用翻译模型
3. **时间轴对齐** - 逐条翻译保证 1:1 对应

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
- 偏好完整句子输入，逐条 1:1 翻译最稳定

**字幕翻译用法**:
- 官方推荐 prompt：`Translate the following segment into Chinese, without additional explanation.`
- `translate` 逐条翻译，保证时间轴 1:1 对齐

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
| 语音识别 | Whisper large-v3-**turbo**（CT2） | CTranslate2 C++ 引擎 + 自写 log-mel 前端，编译为独立二进制；默认 turbo，快 ~3× |
| 语音识别（高精度） | Whisper large-v3（CT2） | 分句更细、更精细；用 `WHISPER_MODEL_DIR` 切换 |
| 字幕翻译 | Hunyuan-MT-7B-Q4_K_M | 腾讯混元翻译专用模型，llama.cpp + CUDA |

### 语音识别模型与速度

- **默认 `large-v3-turbo`**：encoder 相同、decoder 仅 4 层，推理快约 **3×**，多语言。
- **`large-v3`**：decoder 32 层，分句更细（更短、更碎），质量略高、速度较慢。
- 实测（145s 音频，RTX 3080）：

| 模型 | 分句 | 推理耗时 | 相对速度 |
|------|------|---------|---------|
| large-v3-turbo（默认） | 24 条 | 1.75s | **~3×** |
| large-v3 | 34 条 | 5.14s | 1× |

**环境变量**（`transcribe.sh` / `subtitle.sh` 均适用）：

| 变量 | 说明 | 默认 |
|------|------|------|
| `WHISPER_MODEL_DIR` | 模型目录 | `/data/models/faster-whisper-large-v3-turbo` |
| `WHISPER_BEAM` | 束搜索宽度（1=贪心，略快） | `5` |
| `WHISPER_DEVICE` | 设为 `cpu` 强制 CPU | CUDA/FP16 |
| `WHISPER_PROFILE` | 非空则打印分段耗时 | 关 |

```bash
# 用高精度 large-v3 转写
WHISPER_MODEL_DIR=/data/models/faster-whisper-large-v3 bash subtitle.sh video.mp4
```

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

## 字幕翻译：逐条 1:1 对齐

> **模型差异说明**
>
> 当前默认模型为**腾讯 Hunyuan-MT-7B**（专用翻译模型），它对碎片化字幕会“自作主张”补全句子。
> 一旦传入前文上下文（无论放在 user 还是 system 消息），容易产生**跨行重复/错位**。
> 因此 `translate` 采用**逐条翻译**（每条独立请求），最稳定。

### 为什么不用上下文窗口

| 方案 | 结果 |
|------|------|
| 批量翻译（多条/批） | 模型会合并翻译，位置错乱 |
| 全文翻译 | 模型修改原文结构 |
| **逐条翻译（1 条/请求）** | **对齐正确、最稳定** |

### 实现要点

1. **温度 0** - 确定性输出，字幕翻译不需要创意
2. **严格 1:1** - 每条 Dialogue 独立翻译，不合并不拆分
3. **双语拼接** - 原文 + `\N` + 译文，时间轴保持不变
4. **Prompt 固定** - `Translate the following segment into <目标语言>, without additional explanation.`

### 验证结果

```
测试文件: 27 条字幕
翻译耗时: 4 秒（RTX 3080）
语言: en → zh
时间戳对齐: ✓ 完全正确
```
