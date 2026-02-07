# my-trans

基于 AI 的视频字幕生成与翻译工具，支持生成英文字幕并翻译为中文双语字幕。

AI-powered video subtitle generation and translation tool. Generate English subtitles and translate them into Chinese bilingual subtitles.

---

## 参考项目 / Reference Projects

### [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

> **CN:** faster-whisper 是 OpenAI Whisper 模型的高性能实现，基于 CTranslate2 优化。
> 
> **特点：** 比原始 Whisper 快 4 倍，内存占用减少 50%，支持 GPU 加速。
>
> ---
>
> **EN:** faster-whisper is a high-performance implementation of OpenAI Whisper model, optimized with CTranslate2.
> 
> **Features:** 4x faster than original Whisper, 50% less memory usage, GPU acceleration support.

### [CTranslate2](https://github.com/OpenNMT/CTranslate2)

> **CN:** CTranslate2 是 OpenNMT 开发的高性能 Transformer 模型推理引擎。
> 
> **特点：** 支持多种模型架构，INT8/FP16 量化，GPU/CPU 优化，批量推理加速。
>
> ---
>
> **EN:** CTranslate2 is a high-performance Transformer model inference engine developed by OpenNMT.
> 
> **Features:** Supports multiple model architectures, INT8/FP16 quantization, GPU/CPU optimization, batch inference acceleration.

### [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-3.3B)

> **CN:** No Language Left Behind (NLLB) 是 Meta AI 开发的多语言翻译模型。
> 
> **特点：** 支持 200 种语言，蒸馏版模型更小更快，开源免费使用。
>
> ---
>
> **EN:** No Language Left Behind (NLLB) is a multilingual translation model developed by Meta AI.
> 
> **Features:** Supports 200 languages, distilled models are smaller and faster, open source and free to use.

---

## 技术架构 / Technical Architecture

### 核心模型 / Core Models

| 功能 / Function | 模型 / Model | 说明 / Description |
|----------------|-------------|-------------------|
| 语音识别 / Speech Recognition | [faster-whisper-medium](https://github.com/SYSTRAN/faster-whisper) | OpenAI Whisper 的 CTranslate2 优化版 / CTranslate2-optimized OpenAI Whisper |
| 翻译 / Translation | [NLLB-200-3.3B](https://huggingface.co/facebook/nllb-200-distilled-3.3B) | Meta 多语言翻译模型，支持 200+ 语言 / Meta multilingual translation model, 200+ languages |

### 技术栈 / Tech Stack

- **faster-whisper**: 基于 CTranslate2 的 Whisper 推理加速 / Whisper inference acceleration with CTranslate2
- **ctranslate2**: 高性能 Transformer 推理引擎 / High-performance Transformer inference engine
- **transformers**: Hugging Face 模型库 / Hugging Face model library

---

## 模型下载 / Model Download

### 1. Whisper 模型 / Whisper Model

```bash
# 使用 faster-whisper 默认配置，自动下载模型 / Use faster-whisper default config, auto-download model
# 模型路径 / Model path: e:/cuda/faster-whisper-medium
```

### 2. NLLB 翻译模型 / NLLB Translation Model

```bash
# 下载预转换的 CTranslate2 模型 (推荐) / Download pre-converted CTranslate2 model (recommended)
huggingface-cli download Derur/nllb-200-3.3B-ct2-float16 --local-dir E:/cuda/nllb-200-3.3B-ct2-float16
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
# 核心依赖 / Core dependencies
pip install faster-whisper==1.2.1
pip install ctranslate2==4.7.1
pip install transformers==4.35.0
pip install huggingface_hub==0.16.4
pip install tokenizers==0.14.0
```

### 完整安装命令 / Complete Installation Command

```bash
pip install faster-whisper==1.2.1 ctranslate2==4.7.1 transformers==4.35.0 huggingface_hub==0.16.4 tokenizers==0.14.0
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

### 批量处理 / Batch Processing

```bash
# 多个视频 / Multiple videos
python transcribe.py video1.mp4 video2.mp4 video3.mp4
```

---

## 脚本说明 / Script Description

| 脚本 / Script | 功能 / Function |
|-------------|--------------|
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
