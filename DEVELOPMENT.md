# my-trans 开发规范

> 基于项目实际 bug 和教训总结

---

## 1. 字幕生成核心规则

### ⚠️ 必须启用 word_timestamps=True

```python
segments, info = model.transcribe(
    audio,
    word_timestamps=True  # ⚠️ 必须！否则字幕时间轴不准
)
```

**教训**：
- 初期未启用此参数，导致字幕与语音严重不同步
- 用户反馈"27秒显示一条字幕"的问题
- 这是字幕工具的基本要求，必须默认开启

---

## 2. 字幕格式规范

### ASS 格式标准

```python
def create_ass_dialogue(start, end, text, style="Default"):
    # 每行必须以换行符结尾
    return f"Dialogue: 0,{start_fmt},{end_fmt},{style},,0,0,0,,{text_escaped}\n"
```

**教训**：
- 初期 Dialogue 行缺少 `\n`，导致所有行合并成一个超长行
- 播放器无法正确解析

### VTT 格式标准

```python
vtt_content = "WEBVTT\n\n"
# 每条字幕格式
vtt_content += f"{start} --> {end}\n{text}\n\n"
```

---

## 3. 时间戳格式

### ASS 时间格式 (H:MM:SS.cc)
```python
def format_time_ass(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
```

### VTT 时间格式 (HH:MM:SS.mmm)
```python
def format_time_vtt(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
```

---

## 4. 字幕分段规则

### 禁止：按句子拆分后重新分配时间

```python
# ❌ 错误示范：时间分配不准确
for i, sentence in enumerate(sentences):
    current_end = start + (i + 1) * avg_duration  # 不准！
    ass_content += create_ass_dialogue(current_start, current_end, sentence)
```

### 正确：长句子用原始时间戳

```python
# ✅ 正确：用原始 segment 时间戳
for sentence in sentences:
    ass_content += create_ass_dialogue(segment.start, segment.end, sentence)
```

**教训**：
- 尝试按句子拆分并重新计算时间，导致时间轴错乱
- 解决：句子拆分但使用原始时间戳，或直接用 word-level timestamps

---

## 5. GPU 内存管理

### 必须添加内存清理

```python
def cleanup_model(model):
    """释放 GPU 内存"""
    import torch
    del model
    torch.cuda.empty_cache()

# 使用后清理
cleanup_model(model)
```

**教训**：
- 初期没有内存清理，处理大文件后 OOM
- 添加 print_memory_usage() 便于监控

### 必须使用 torch.no_grad()

```python
@torch.no_grad()
def translate_text(tokenizer, model, text):
    # 防止梯度累积
    ...
```

---

## 6. 模型配置

### Whisper 模型选择

| 模型 | VRAM | 推荐场景 |
|-----|------|---------|
| base | ~1GB | CPU/低配 GPU |
| medium | ~2-4GB | RTX 3060+ |
| large-v3 | ~8GB | RTX 3080+ |

### 必须设置 compute_type

```python
model = WhisperModel(model_path, device="cuda", compute_type="float16")
```

**教训**：
- 不同 GPU 需要不同精度
- RTX 3080 用 float16 刚好够

---

## 7. 输出格式

### 默认使用 ASS 格式

```python
output_format = "ass"  # 默认，不再支持 VTT

# 输出路径
output_path = video_path.with_suffix('.ass')
```

**教训**：
- VTT 格式字幕时间长不同步问题难以解决
- ASS 格式支持更精准的时间控制和样式

---

## 8. 错误处理

### 必须检查模型兼容性

```python
# ❌ 错误：WhisperModel 没有 eval() 方法
model.eval()  # 报错！

# ✅ 正确：跳过
# faster-whisper 不需要此设置
```

### 必须检查数据类型

```python
# 确保时间戳是浮点数
start = float(segment.start)
end = float(segment.end)
```

---

## 9. 文档规范

### 必须标注重要参数

```markdown
> ⚠️ **重要提示**: `word_timestamps=True` 是字幕同步的关键参数，务必保持开启！
```

### 必须提供完整示例

```bash
# 完整工作流
export http_proxy="http://proxy:7897"
export https_proxy="http://proxy:7897"
yt-dlp -o "%(title)s.%(ext)s" -f b "url"
python transcribe.py video.mp4
python translate_vtt.py video.ass
```

---

## 10. 测试清单

### 发布前必须测试

- [ ] 字幕时间轴与语音同步
- [ ] 字幕格式正确（ASS/VTT）
- [ ] GPU 内存正常释放
- [ ] 长视频不 OOM
- [ ] 代理配置正确
- [ ] README 示例可运行

---

## 11. Git 提交规范

### 提交信息格式

```
<类型>: <简短描述>

[可选的详细描述]

- <具体改动1>
- <改动2>
```

### 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `perf`: 性能优化
- `refactor`: 重构

### 示例

```
fix: 修复 ASS 格式 Dialogue 行缺少换行符

- Dialogue 末尾添加 \n
- 测试播放正常
```

---

## 12. 禁止事项

### ❌ 禁止未经测试的复杂逻辑

- 句子拆分 + 时间重算 → 时间错乱
- 多个 format 参数混用 → 格式错误

### ❌ 禁止硬编码关键参数

```python
# ❌ 错误
batch_size = 16  # 可能导致 OOM

# ✅ 正确
batch_size = 8  # 安全值
```

### ❌ 禁止忽略用户反馈

- "字幕慢半拍" → 必须解决
- "格式错误" → 必须修复

---

## 总结

本规范基于以下实际教训总结：

1. **word_timestamps=True** 是字幕同步的基本要求
2. **时间戳必须准确**，禁止重新计算
3. **内存管理** 必须到位，防止 OOM
4. **格式标准** 必须严格遵守
5. **用户反馈** 必须认真对待

违反以上规则将导致严重问题，请严格遵守。
