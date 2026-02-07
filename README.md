# my-trans

> 🤖 **AI 时代已经来临，解放双手！**
> 
> 本项目由 [OpenCode](https://opencode.ai) 基于 **MiniMax-M2.1** 模型生成，全程使用 AI 协助开发。
> 
> 这是一个全 AI 驱动的项目，展示了 AI 在实际应用中的强大能力。

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
python translate_vtt.py video.ass
```

# 3. 翻译成中文双语字幕
python translate_vtt.py video.ass
```

---

## AI 协作 / AI Collaboration

| 角色 / Role | 工具 / Tool | 说明 / Description |
|-----------|-------------|------------------|
| 需求与开发 / Requirement & Development | [OpenCode](https://opencode.ai) | AI 编程助手，基于 MiniMax-M2.1 模型 |
| 语音识别 / Speech Recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | OpenAI Whisper 的 CTranslate2 优化版 |
| 翻译引擎 / Translation Engine | [CTranslate2](https://github.com/OpenNMT/CTranslate2) | 高性能 Transformer 推理引擎 |
| 多语言翻译 / Multilingual Translation | [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-3.3B) | Meta AI 200+ 语言翻译模型 |

> 💡 **从想法到实现，全程由 AI 协助完成**  
> **From idea to implementation, all assisted by AI**

---

## 技术架构 / Technical Architecture

### 核心模型 / Core Models

| 功能 / Function | 模型 / Model | 说明 / Description |
|----------------|-------------|-------------------|
| 语音识别 / Speech Recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | OpenAI Whisper 的 CTranslate2 优化版 |
| 翻译 / Translation | [NLLB-200-3.3B](https://huggingface.co/facebook/nllb-200-distilled-3.3B) | Meta 多语言翻译模型，支持 200+ 语言 |

### 技术栈 / Tech Stack

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**: 基于 CTranslate2 的 Whisper 推理加速
- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)**: 高性能 Transformer 推理引擎
- **[transformers](https://huggingface.co/docs/transformers)**: Hugging Face 模型库

---

## 模型下载 / Model Download

### 使用下载脚本 (推荐 / Recommended)

```bash
# 运行下载脚本 / Run download script
python download_models.py

# 选项 / Options:
#   1. 下载 faster-whisper 模型
#   2. 下载 NLLB 翻译模型
#   3. 下载全部模型
```

### 1. Whisper 模型 / Whisper Model

```bash
# 使用 faster-whisper 默认配置，自动下载模型 / Use faster-whisper default config, auto-download model
# 模型路径 / Model path: e:/cuda/faster-whisper-medium
```

### 2. NLLB 翻译模型 / NLLB Translation Model

```bash
# 方法1: 使用下载脚本 / Method 1: Use download script
python download_models.py

# 方法2: 使用 huggingface-cli / Method 2: Use huggingface-cli
huggingface-cli download Derur/nllb-200-3.3B-ct2-float16 --local-dir E:/cuda/nllb-200-3.3B-ct2-float16 --local-dir-use-symlinks false
```

**模型介绍 / Model Introduction:**

| 项目 / Item | 说明 / Description |
|------------|------------------|
| 模型 / Model | [Derur/nllb-200-3.3B-ct2-float16](https://huggingface.co/Derur/nllb-200-3.3B-ct2-float16) |
| 来源 / Source | 社区预转换模型 / Community pre-converted model |
| 原始模型 / Original Model | [facebook/nllb-200-distilled-3.3B](https://huggingface.co/facebook/nllb-200-distilled-3.3B) |
| 量化方式 / Quantization | FP16 (Float16) |
| 模型大小 / Size | ~6.5GB |
| 语言数量 / Languages | 200+ 种语言 / languages |

**量化效果 / Quantization Effects:**

| 量化方式 / Type | 模型大小 / Size | 显存占用 / VRAM | 速度 / Speed | 精度损失 / Quality Loss |
|---------------|---------------|----------------|-------------|---------------------|
| FP32 | ~13GB | ~10GB | 基准 / Base | 无 / None |
| FP16 | ~6.5GB | ~5GB | 快 1.5x | 几乎无 / Almost none |
| INT8 | ~3.3GB | ~3GB | 快 2-3x | ~1-2% |
| INT4 | ~1.7GB | ~2GB | 快 4x | ~3-5% |

**推荐配置 / Recommended Config:**
- **FP16 (本项目使用)**: 最佳平衡，速度快、精度高
- 适合 RTX 3080 (10GB) 及以上显卡

原始模型 / Original Model:
- [facebook/nllb-200-distilled-3.3B](https://huggingface.co/facebook/nllb-200-distilled-3.3B)

---

## 环境配置 / Environment Setup

### Python 版本 / Python Version
- Python 3.9+

### 依赖安装 / Dependencies Installation

```bash
# PyTorch (GPU) - Windows CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 核心依赖 / Core dependencies
pip install faster-whisper==1.2.1
pip install ctranslate2==4.7.1
pip install transformers==4.35.0
pip install huggingface_hub==0.16.4
pip install tokenizers==0.14.0
```

### 完整安装命令 / Complete Installation Command

```bash
# 1. 安装 PyTorch (GPU)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装其他依赖
pip install faster-whisper==1.2.1 ctranslate2==4.7.1 transformers==4.35.0 huggingface_hub==0.16.4 tokenizers==0.14.0
```

### 验证安装 / Verify Installation

```powershell
# 查看已安装的 CUDA 相关包
pip list | findstr cuda
```

输出示例 / Output example:
```
torch              2.7.1+cu118
torchaudio         2.7.1+cu118
torchvision        0.22.1+cu118
ctranslate2        4.7.1
```

---

## 使用方法 / Usage

### 1. 生成英文字幕 / Generate English Subtitles

```bash
python transcribe.py video.mp4
```
输出 / Output: `video.vtt` (英文字幕 / English subtitles)

### 2. 翻译为双语字幕 / Translate to Bilingual Subtitles

```bash
python translate_vtt.py video.vtt
```
输出 / Output: `video.bilingual.vtt` (双语字幕 / Bilingual subtitles)

### 自动语言检测 / Auto Language Detection

支持自动检测字幕语言并翻译为中文：

```bash
# 自动检测语言并翻译为中文双语字幕
python translate_vtt.py video.vtt
```

支持的语言 / Supported Languages:

| 语言代码 | Language | 说明 / Description |
|---------|----------|-------------------|
| `ja` | Japanese | 日语 |
| `en` | English | 英语 |
| `zh` | Chinese | 中文 |
| `ko` | Korean | 韩语 |
| `fr` | French | 法语 |
| `de` | German | 德语 |
| `es` | Spanish | 西班牙语 |

### 手动指定语言 / Manual Language Selection

如需指定源语言，使用 `--lang` 参数：

```bash
# 日语视频
python translate_vtt.py --lang=ja video.vtt

# 韩语视频
python translate_vtt.py --lang=ko video.vtt

# 英语视频
python translate_vtt.py --lang=en video.vtt
```

### 批量处理 / Batch Processing

```bash
# 多个视频 / Multiple videos
python transcribe.py video1.mp4 video2.mp4 video3.mp4
```

---

## 脚本说明 / Script Description

| 脚本 / Script | 功能 / Function |
|-------------|--------------|
| `download_models.py` | 下载模型到本地 / Download models to local |
| `transcribe.py` | 使用 faster-whisper 生成英文字幕 / Generate English subtitles with faster-whisper |
| `translate_vtt.py` | 使用 CTranslate2+NLLB 翻译为双语字幕 / Translate to bilingual subtitles with CTranslate2+NLLB |
| `translate_nllb_official.py` | 官方 transformers 版本的翻译脚本 / Official transformers version translation script |

---

## 性能对比 / Performance Comparison

| 版本 / Version | 速度 / Speed | 说明 / Description |
|--------------|-------------|------------------|
| CTranslate2 (推荐 / Recommended) | ~390ms/条 | GPU 加速，兼容性好 / GPU acceleration, good compatibility |
| 官方 transformers | ~405ms/条 | 不依赖 CTranslate2 / No CTranslate2 dependency |

---

## 常见问题 / FAQ

### 1. CUDA 内存不足 / CUDA Memory Insufficient

降低 `batch_size` 或重启 Python 进程后重试。

Reduce `batch_size` or restart Python process and try again.

### 2. 模型加载失败 / Model Loading Failed

检查模型路径是否正确，清理缓存：

Check model path is correct, clear cache:

```bash
del %LOCALAPPDATA%\huggingface\hub
```

### 3. 翻译结果异常 / Translation Result Abnormal

确认使用正确的模型版本 / Make sure to use correct model version:
`Derur/nllb-200-3.3B-ct2-float16`

---

## License

MIT
