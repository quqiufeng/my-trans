# my-trans

> **本脚本用于 Windows 10/11 + GPU 环境运行**

AI 时代已经来临，解放双手！

基于 AI 的视频字幕生成与翻译工具，支持多语言自动检测，生成中文双语字幕。

AI-powered video subtitle generation and translation tool. Supports automatic language detection for multilingual videos, generating Chinese bilingual subtitles.

---

## 快速开始 / Quick Start

```bash
# 1. 语音识别（生成英文字幕）
python transcribe.py video.mp4

# 2. 翻译字幕（选择一种方式）
# 方式一：NLLB 离线翻译（速度快）
python translate.py video.ass

# 方式二：LM Studio 千问翻译（质量好，推荐）
python llm.py video.ass
```

---

## 翻译方案对比

| 特性 | NLLB (translate.py) | LM Studio (llm.py) |
|------|---------------------|--------------------|
| 部署方式 | 本地模型文件 | 本地 API 服务 |
| 模型 | NLLB-200-3.3B-ct2 | Qwen2.5-7B-Instruct |
| 翻译质量 | 一般 | **优秀** |
| 中文地道程度 | 机械 | 流畅自然 |
| 速度 | 快（batch=128） | 较慢（逐条发送） |
| 显存占用 | GPU 8GB+ | 无（CPU 调用） |
| 专有名词处理 | 直译 | 首次标注原文 |
| 适用场景 | 批量预览 | 最终精翻 |

---

## LM Studio 配置

**官方主页**: https://lmstudio.ai/

- **模型**: [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-GGUF)
- **量化版本**: `Qwen2.5-7B-Instruct.Q5_K_M.gguf`
- **模型路径**: `E:\model\mradermacher\Qwen2.5-7B-Instruct-GGUF`
- **服务器**: `http://192.168.124.3:11434/v1` (OpenAI 兼容 API)

### Qwen2.5-7B 模型介绍

**官方链接**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**模型架构**:
- 类型: Decoder-only Transformer
- 参数规模: 7B (70亿参数)
- 上下文长度: 131,072 tokens
- 语言: 多语言（含中英文）

**性能特点**:
- 优秀的中英文理解与生成能力
- 指令遵循能力强
- 适合内容生成和翻译任务
- GGUF 量化版在消费级硬件上推理速度快

**字幕翻译优势**:
- 中文表达更地道自然
- 专有名词处理更智能
- 能理解上下文语境

---

## NLLB 翻译模型（离线备用）

- **模型**: [NLLB-200-3.3B](https://huggingface.co/facebook/nllb-200-distilled-3.3B)
- **模型路径**: `E:/cuda/nllb-200-3.3B-ct2-float16`
- **格式**: CTranslate2 量化 (FP16)
- **加速**: CUDA GPU 推理

### NLLB 模型介绍

**官方链接**: https://huggingface.co/facebook/nllb-200-distilled-3.3B

**模型架构**:
- 类型: Encoder-Decoder Transformer
- 参数规模: 3.3B (33亿参数，蒸馏版)
- 语言支持: 200 种语言
- 训练数据: 多语言平行语料

**性能特点**:
- 支持 200 种语言，包括罕见语言
- 翻译质量稳定
- **离线运行**，无需网络
- CTranslate2 量化加速，GPU 推理

**使用场景**:
- 网络不可用时
- 需要批量快速翻译
- 小语种翻译需求

---

## 使用方法

### 1. 语音识别（生成英文字幕）

```bash
python transcribe.py video.mp4
```

输出: `video_en.ass` 或 `video_zh.ass`（根据检测语言）

**参数**:
- `--offset=0.3` - 字幕时间偏移校正（秒）

### 2. 翻译字幕

#### 方式一：NLLB 离线翻译（速度快）

```bash
# 翻译英文字幕
python translate.py video_en.ass

# 指定语言
python translate.py --lang=ja video.ass
```

输出: `video_en_bilingual.ass`

#### 方式二：LM Studio 千问翻译（质量好）

```bash
# 翻译英文字幕
python llm.py video_en.ass

# 指定语言
python llm.py --lang=ja video.ass

# 扫描当前目录所有 ASS 文件
python llm.py
```

输出: `video_en_llm.ass`

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

## 脚本函数说明

### transcribe.py

| 函数 | 说明 |
|------|------|
| `transcribe_video(video_path, model, batched_model, offset)` | 转录音频并保存为 ASS 格式 |
| `format_time_ass(seconds)` | 将秒数转换为 ASS 时间格式 |
| `split_sentences(text)` | 按句子边界分割文本 |
| `wrap_text(text, max_chars)` | 长文本自动换行 |
| `create_ass_header()` | 创建 ASS 字幕头部 |
| `create_ass_dialogue()` | 创建 ASS 对话行 |

### translate.py (NLLB)

| 函数 | 说明 |
|------|------|
| `load_translator()` | 加载 CTranslate2 NLLB 翻译模型 |
| `translate_batch_fast()` | 批量翻译 - 优化版 (batch_size=128) |
| `parse_ass(ass_path)` | 解析 ASS 文件 |
| `parse_vtt(vtt_path)` | 解析 VTT 文件 |
| `create_bilingual_ass()` | 创建双语 ASS |
| `create_bilingual_vtt()` | 创建双语 VTT |

### llm.py (LM Studio)

| 函数 | 说明 |
|------|------|
| `translate_batch(blocks, source_lang, target_lang)` | 调用 LM Studio API 批量翻译 |
| `parse_ass(ass_path)` | 解析 ASS 文件 |
| `parse_translations(content, total_count)` | 解析翻译结果 |
| `create_bilingual_ass()` | 创建双语 ASS |
| `detect_language_simple(texts)` | 简单的语言检测 |

### translate_nllb_official.py (备用)

| 函数 | 说明 |
|------|------|
| `load_translator()` | 加载 HuggingFace Transformers NLLB 模型 |
| `translate_batch()` | 批量翻译 |
| `parse_ass(ass_path)` | 解析 ASS 文件 |
| `create_bilingual_ass()` | 创建双语 ASS |

---

## 目录结构

```
my-trans/
├── transcribe.py              # 语音识别（faster-whisper）
├── translate.py               # NLLB 翻译（CTranslate2）
├── llm.py                     # LM Studio 千问翻译（质量优先）
├── translate_nllb_official.py # NLLB 翻译（Transformers 备用）
├── my_whisper.py              # OpenAI Whisper 字幕生成
├── download_models.py         # 模型下载工具
├── res/                       # 原始文件目录
│   ├── video.mp4              # 视频文件
│   └── video_en.ass           # 英文字幕
└── output/                    # 输出文件目录
    └── video_en_llm.ass       # 双语字幕
```

---

## 环境配置

### Python 版本
- Python 3.9+

### 依赖安装

#### 基础依赖（NLLB 离线翻译）
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

#### LM Studio 翻译依赖
```bash
pip install requests
```

---

## 字体配置

默认使用 **LXGW WenKai** 字体。

**Windows 安装**:
1. 下载 LXGW WenKai 字体：https://github.com/lxgw/LxgwWenKai
2. 安装到系统

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
# 方式一：NLLB 离线翻译（快速预览）
视频.mp4 → transcribe.py → 视频_en.ass → translate.py → 视频_en_bilingual.ass

# 方式二：LM Studio 千问翻译（质量精翻）
视频.mp4 → transcribe.py → 视频_en.ass → llm.py → 视频_en_llm.ass
```

### 核心模型

| 功能 | 模型 | 说明 |
|------|------|------|
| 语音识别 | faster-whisper-large-v3 | OpenAI Whisper 的 CTranslate2 优化版 |
| 离线翻译 | NLLB-200-3.3B-ct2 | Meta 多语言翻译模型，支持 200+ 语言 |
| 在线翻译 | Qwen2.5-7B-Instruct | 阿里千问，中文优化 |

### 技术栈

- **faster-whisper**: 基于 CTranslate2 的 Whisper 推理加速
- **CTranslate2**: 高性能 Transformer 推理引擎
- **transformers**: Hugging Face 模型库
- **LM Studio**: 本地 LLM 服务（OpenAI 兼容 API）

---

## AI 协作

| 角色 | 工具 | 说明 |
|------|------|------|
| 需求与开发 | OpenCode | AI 编程助手 |
| 语音识别 | faster-whisper | OpenAI Whisper 的 CTranslate2 优化版 |
| 离线翻译 | CTranslate2 + NLLB | 高性能 Transformer 推理引擎 |
| 在线翻译 | LM Studio + Qwen2.5 | 本地 LLM，中文优化 |

---

## LLM 翻译顺序问题与解决方案

### 问题背景

在使用 LLM 批量翻译字幕时，发现一个关键问题：**LLM 输出的翻译结果顺序不一定与输入顺序一致**。

### 问题表现

假设输入 15 条字幕：
```
原文1: "Hello world"
原文2: "This is a test"
原文3: "Good morning"
...
原文15: "Thank you"
```

LLM 可能返回：
```json
[
  {"translation": "谢谢"},
  {"translation": "早上好"},
  {"translation": "这是一个测试"},
  ...
]
```

即翻译的顺序是打乱的，导致：
- 翻译内容与原文对不上
- 字幕时间轴错位（中文和英文不匹配）
- 播放时出现文不对时的问题

### 问题原因

LLM 输出顺序不可靠的根本原因：

1. **注意力机制特性** - LLM 生成时不是严格按输入位置顺序，而是根据注意力权重动态决定
2. **语义重组倾向** - LLM 倾向于按"语义相关性"组织输出，可能重新排列
3. **压缩优化** - LLM 可能合并相似句子或省略重复内容
4. **训练数据模式** - LLM 从数据中学到的模式不一定保持位置对应

**这不是 LLM 的 bug，而是其工作原理的固有特性。**

### 解决方案：序号排序算法

#### 核心思路

在翻译请求中为每条原文添加序号，LLM 返回时携带相同序号，解析后按序号重新排序。

#### 实现步骤

**1. 输入格式（带序号）**
```json
[
  {"index": 1, "text": "原文1"},
  {"index": 2, "text": "原文2"},
  ...
]
```

**2. 输出格式（带序号）**
```json
[
  {"index": 1, "translation": "翻译1"},
  {"index": 3, "translation": "翻译3"},
  {"index": 2, "translation": "翻译2"}
]
```

**3. 解析与排序算法**
```python
def parse_translations(content, expected_count):
    # 解析 JSON
    parsed = json.loads(content)
    
    # 检查是否是带序号的格式
    if isinstance(parsed, list) and len(parsed) > 0:
        first = parsed[0]
        if isinstance(first, dict) and 'index' in first:
            # 按 index 排序
            parsed_sorted = sorted(parsed, key=lambda x: x.get('index', 0))
            # 提取 translation 字段
            translations = [item.get('translation', '') for item in parsed_sorted]
            return translations
    
    return parsed  # 其他格式直接返回
```

### 代码实现

详见 `llm.py` 中的 `parse_translations()` 函数（第 39-118 行）：

```python
def parse_translations(content, expected_count):
    """解析翻译结果，支持多种格式，确保顺序正确"""
    # ... 清理代码块 ...
    
    # 尝试解析带序号的 JSON
    try:
        parsed = json.loads(content_clean)
        if isinstance(parsed, list) and len(parsed) > 0:
            first = parsed[0]
            if isinstance(first, dict) and 'index' in first and 'translation' in first:
                # 按 index 排序
                parsed_sorted = sorted(parsed, key=lambda x: x.get('index', 0))
                # 提取 translation 字段
                translations = [item.get('translation', '') for item in parsed_sorted]
                return translations
    except json.JSONDecodeError:
        pass
    
    # ... 其他回退方法 ...
```

### 提示词设计

在 `translate_batch()` 函数中明确要求 LLM 输出带序号的 JSON：

```
请严格按以下 JSON 格式输出（序号必须对应）：
[
  {"index": 1, "translation": "翻译1"},
  {"index": 2, "translation": "翻译2"},
  ...
]
```

### 回退策略

如果 LLM 未按要求格式返回，解析函数提供多种回退方案：

| 方法 | 触发条件 | 说明 |
|------|----------|------|
| 带序号 JSON | 标准格式 | 按 index 排序，提取 translation |
| 普通 JSON | 无 index 字段 | 直接返回列表 |
| 提取 [...] | JSON 解析失败 | 从文本中提取数组部分 |
| 正则提取 | 上述都失败 | 提取所有 "xxx" 内容 |
| 逐行解析 | 上述都失败 | 按行分割，匹配编号格式 |

### 验证方法

```bash
cd /home/quqiufeng/my-trans
python3 llm.py res/视频_en.ass
```

对比输入字幕和输出字幕，检查中英文是否按时间轴对齐。

### 经验总结

1. **不要假设 LLM 输出有序** - 任何批量处理都需要考虑顺序问题
2. **序号是可靠方案** - 通过外部排序绕过 LLM 的不确定性
3. **保留回退机制** - LLM 可能不按格式要求输出，多种解析方法提高鲁棒性
4. **测试验证** - 字幕翻译顺序错乱很难肉眼发现，建议生成后检查几处关键时间点
