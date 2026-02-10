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

## LLM 翻译字幕：窗口滑动翻译算法

### 问题背景

在使用 LLM 批量翻译字幕时，面临两个冲突需求：

1. **上下文连贯** - 需要批量翻译保留上下文
2. **位置对齐** - 每条字幕必须对应原文字幕的时间戳

### 问题表现

| 方案 | 结果 |
|------|------|
| 批量翻译（15条/批） | LLM 合并翻译，位置错乱 |
| 逐条翻译（1条/批） | 对齐正确，但无上下文 |
| 全文翻译 | LLM 修改原文结构 |
| 序号排序 | LLM 不按格式返回 |

### 解决方案：窗口滑动翻译算法

#### 核心思路

每条字幕翻译时，**仅使用前 N 条已翻译的中文作为上下文**，而不是完整全文。

```
┌─────────────────────────────────────────────────────────┐
│  翻译第 i 条时                                           │
│  ├─ 参考: 前 2 条的中文译文 (窗口)                       │
│  ├─ 输入: 当前英文字幕 + 前2条中文                       │
│  └─ 输出: 当前中文翻译                                    │
│                                                          │
│  窗口滑动: [译文1, 译文2] → [译文2, 译文3] → ...         │
└─────────────────────────────────────────────────────────┘
```

#### 算法流程

```
1. 初始化空窗口 prev_translations = []

2. 遍历每条字幕 i:
   a. 构建 prompt:
      - 包含 prev_translations (最多2条)
      - 包含当前英文字幕
      - 强调"只翻译当前这一条"
   
   b. 调用 LLM 翻译
   
   c. 将译文加入 prev_translations
   
   d. 如果窗口超过2条，移除最旧的
      prev_translations = prev_translations[-2:]

3. 按时间戳顺序输出双语 ASS
```

#### 代码实现

```python
CONTEXT_SIZE = 2  # 窗口大小

def translate_with_context(english_text, prev_translations, idx, total):
    # 构建上下文
    context = ""
    if prev_translations:
        context = "[参考]\n" + "\n".join(prev_translations) + "\n"
    
    prompt = f"""{context}

只翻译这一句（不要多译，不要漏译）：

{english_text}

输出中文："""

    # 调用 LLM
    response = requests.post(f"{LLMS_HOST}/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0  # 确定性输出
    })
    
    return response

# 主循环
translations = []
prev_translations = []

for i, text in enumerate(texts):
    trans = translate_with_context(text, prev_translations, i+1, len(texts))
    translations.append(trans)
    
    prev_translations.append(trans)
    if len(prev_translations) > CONTEXT_SIZE:
        prev_translations = prev_translations[-CONTEXT_SIZE:]
```

#### 窗口大小对比

| 窗口大小 | 上下文连贯性 | 位置对齐 | 风险 |
|----------|-------------|----------|------|
| 0 (逐条) | 差 | ★★★★★ | 无 |
| 2 | ★★★★☆ | ★★★★★ | 低 |
| 3 | ★★★★★ | ★★★☆☆ | 中 (合并风险) |
| 全文 | ★★★★★ | ★★☆☆☆ | 高 (错乱) |

**推荐: 窗口大小 = 2**

#### 关键技巧

1. **温度设为 0** - 减少随机性，确保输出稳定
   ```python
   "temperature": 0
   ```

2. **简化 prompt** - 避免复杂指令导致格式混乱
   ```
   只翻译这一句（不要多译，不要漏译）
   输出中文：
   ```

3. **添加延迟** - 防止 API 过载
   ```python
   time.sleep(0.1)
   ```

### 为什么这个方案有效？

1. **限制上下文长度** - LLM 不会因为看到太多内容而"自作主张"
2. **保持局部连贯** - 前后2条足以为当前句子提供语境
3. **1:1 严格对应** - 每条字幕独立翻译，不合并不拆分
4. **确定性输出** - temperature=0 避免随机扰动

### 验证结果

```
测试文件: 45 条字幕
窗口大小: 2
翻译耗时: 12 秒
成功率: 45/45 (100%)
时间戳对齐: ✓ 完全正确
```

### 完整代码

详见 `llm.py` 中的 `translate_with_context()` 函数：

```python
def translate_with_context(english_text, prev_translations, idx, total):
    """使用滑动窗口上下文翻译单条字幕"""
    context = ""
    if prev_translations:
        context = "[参考]\n" + "\n".join(prev_translations) + "\n"
    
    prompt = f"""{context}

只翻译这一句（不要多译，不要漏译）：

{english_text}

输出中文："""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0
    }
    
    try:
        response = requests.post(f"{LLMS_HOST}/chat/completions", json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"    错误: {e}")
        return None
```

### 适用场景

本算法适用于：

- **视频字幕翻译** - 保持时间轴对齐
- **长文章分段翻译** - 保持段落对应
- **对话内容翻译** - 保持说话人顺序
- **任何需要位置对齐的批量翻译任务**

### 经验总结

1. **不要让 LLM 看太多** - 上下文越多，顺序越乱
2. **窗口要小** - 2-3条是最佳平衡点
3. **温度设为 0** - 字幕翻译不需要创意
4. **Prompt 要简单** - 复杂指令适得其反
