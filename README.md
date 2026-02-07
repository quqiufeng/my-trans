# my-trans

> **本脚本用于 Windows 10/11 + GPU 环境运行**

AI 时代已经来临，解放双手！

基于 AI 的视频字幕生成与翻译工具，支持多语言自动检测，生成中文双语字幕。

AI-powered video subtitle generation and translation tool. Supports automatic language detection for multilingual videos, generating Chinese bilingual subtitles.

---

## YouTube 视频字幕一键生成:

```bash
# 1. 安装 yt-dlp
pip install yt-dlp

# 2. 设置代理并下载视频（最佳画质，自动合并）
export http_proxy="http://192.168.124.3:7897"
export https_proxy="http://192.168.124.3:7897"
yt-dlp -o "%(title)s.%(ext)s" -f b --restrict-filenames "https://www.youtube.com/watch?v=xxxxx"

# 3. 用 AI 生成字幕（自动检测语言）
python transcribe.py video.mp4

# 4. 翻译成中文双语字幕
python translate.py video.ass
```

---

## 示例文件 / Example Files

```
res/
├── Introduction_OCaml_Programming_Chapter_1_Video_1.mp4      # 原始视频
├── Introduction_OCaml_Programming_Chapter_1_Video_1.ass         # Whisper 识别的英文字幕
└── Introduction_OCaml_Programming_Chapter_1_Video_1_en.ass      # NLLB 翻译的中英双语字幕
```

---

## 如何使用

### 1. 从服务器下载脚本

```bash
cd E:\downloads\res
scp quqiufeng@192.168.124.10:/home/quqiufeng/my-trans/*.py .
```

### 2. 语音识别（生成英文字幕）

```bash
python transcribe.py Introduction_OCaml_Programming_Chapter_1_Video_1.mp4
```

### 3. 翻译字幕

```bash
# 方式一：使用 1.3B 模型（稳定）
python translate_nllb_official.py Introduction_OCaml_Programming_Chapter_1_Video_1_en.ass

# 方式二：使用 3.3B 模型（速度快）
python translate.py Introduction_OCaml_Programming_Chapter_1_Video_1_en.ass
```

---

## AI 协作 / AI Collaboration

| 角色 / Role | 工具 / Tool | 说明 / Description |
|-----------|-------------|------------------|
| 需求与开发 / Requirement & Development | OpenCode | AI 编程助手 |
| 语音识别 / Speech Recognition | faster-whisper | OpenAI Whisper 的 CTranslate2 优化版 |
| 翻译引擎 / Translation Engine | CTranslate2 | 高性能 Transformer 推理引擎 |
| 多语言翻译 / Multilingual Translation | NLLB-200 | Meta AI 200+ 语言翻译模型 |

---

## 技术架构 / Technical Architecture

### 核心模型 / Core Models

| 功能 / Function | 模型 / Model | 说明 / Description |
|----------------|-------------|-------------------|
| 语音识别 / Speech Recognition | faster-whisper | OpenAI Whisper 的 CTranslate2 优化版 |
| 翻译 / Translation | NLLB-200-3.3B | Meta 多语言翻译模型，支持 200+ 语言 |

### 技术栈 / Tech Stack

- **faster-whisper**: 基于 CTranslate2 的 Whisper 推理加速
- **CTranslate2**: 高性能 Transformer 推理引擎
- **transformers**: Hugging Face 模型库

---

## 模型下载 / Model Download

### 使用下载脚本 (推荐 / Recommended)

```bash
python download_models.py
```

### Whisper 模型 / Whisper Model

```bash
# 模型路径 / Model path: e:/cuda/faster-whisper-medium
```

### NLLB 翻译模型 / NLLB Translation Model

```bash
huggingface-cli download Derur/nllb-200-3.3B-ct2-float16 --local-dir E:/cuda/nllb-200-3.3B-ct2-float16 --local-dir-use-symlinks false
```

---

## 环境配置 / Environment Setup

### Python 版本
- Python 3.9+

### 依赖安装

```bash
# PyTorch (GPU) - Windows CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 核心依赖
pip install faster-whisper==1.2.1
pip install ctranslate2==4.7.1
pip install transformers==4.35.0
pip install huggingface_hub==0.16.4
pip install tokenizers==0.14.0
```

---

## 使用方法 / Usage

### 1. 生成字幕

```bash
python transcribe.py video.mp4
```
输出: `video.ass`

### 2. 翻译为双语字幕

```bash
python translate.py video.ass
```
输出: `video.bilingual.vtt`

### 批量处理

```bash
python transcribe.py video1.mp4 video2.mp4 video3.mp4
```

### 支持语言

| 语言代码 | Language |
|---------|----------|
| `ja` | Japanese |
| `en` | English |
| `zh` | Chinese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `es` | Spanish |

---

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `download_models.py` | 下载模型到本地 |
| `transcribe.py` | 使用 faster-whisper 生成 ASS 字幕 |
| `my_whisper.py` | OpenAI Whisper 识别（备用） |
| `translate.py` | 使用 CTranslate2+NLLB 翻译为双语字幕 |
| `translate_nllb_official.py` | 官方 transformers 版本的翻译脚本 |

---

## License

MIT
