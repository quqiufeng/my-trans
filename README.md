# AI 视频字幕生成与翻译工具

**运行环境：Windows 10/11 + GPU（推荐 NVIDIA RTX 3060+）**

## 支持功能

| 功能 | 语言 | 模型 |
|------|------|------|
| 语音识别 | 英文、日文、韩文 | faster-whisper-large-v3 |
| 字幕翻译 | 多语言 | NLLB-200 1.3B |

## 快速开始

### 1. 安装依赖

```bash
pip install faster-whisper ctranslate2 transformers langdetect
```

### 2. 下载模型

```bash
python download_models.py
```

模型存放路径：`E:/cuda/`

### 3. 生成字幕

```bash
# 英文/日文/韩文视频 -> 识别 + 生成 ASS 字幕
python transcribe.py 视频.mp4

# 只翻译现有字幕
python translate.py 字幕.ass
python translate_nllb_official.py 字幕.ass
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `transcribe.py` | 语音识别，生成 ASS 字幕 |
| `my_whisper.py` | OpenAI Whisper 识别（备用） |
| `translate.py` | CTranslate2 翻译（推荐，速度快） |
| `translate_nllb_official.py` | Transformers 翻译（CPU 模式） |
| `download_models.py` | 下载模型文件 |

## 模型路径

| 模型 | 路径 |
|------|------|
| Whisper | `E:/cuda/faster-whisper-large-v3/` |
| NLLB | `E:/cuda/nllb-200-3.3B-ct2-float16/` |

## 注意事项

- 模型文件较大，确保磁盘空间充足
- GPU 内存建议 8GB 以上
- 首次运行会自动下载模型
