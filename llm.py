#!/usr/bin/env python3
"""
使用 LM Studio (Qwen2.5-7B) 翻译 ASS 字幕
生成中英双语字幕
"""

import warnings
warnings.filterwarnings('ignore')

import requests
import re
import json
import time
import sys
from pathlib import Path

LLMS_HOST = "http://192.168.124.3:11434/v1"
MODEL = "qwen2.5-7b-instruct"

LANG_CODE_MAP = {
    'ja': 'jpn', 'japanese': 'jpn',
    'en': 'eng', 'english': 'eng',
    'zh': 'zh', 'chinese': 'zh',
    'ko': 'kor', 'korean': 'kor',
    'fr': 'fra', 'french': 'fra',
    'de': 'deu', 'german': 'deu',
    'es': 'spa', 'spanish': 'spa',
}

def format_elapsed(seconds):
    """将秒数转换为易读格式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds//60}分{seconds%60:.0f}秒"
    else:
        return f"{seconds//3600}小时{(seconds%3600)//60}分"

def parse_translations(content, expected_count):
    """解析翻译结果，返回 original_index -> translation 的映射"""
    content_clean = content.strip()
    if content_clean.startswith('```'):
        content_clean = content_clean[content_clean.find('\n')+1:]
    if content_clean.endswith('```'):
        content_clean = content_clean[:content_clean.rfind('```')]
    content_clean = content_clean.strip()

    mapping = {}

    try:
        parsed = json.loads(content_clean)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    indices = item.get('original_indices', [])
                    trans = item.get('translation', '')
                    for idx in indices:
                        mapping[idx] = trans
            if mapping:
                print(f"    解析成功: {len(mapping)} 条映射")
                return mapping
    except json.JSONDecodeError:
        pass

    try:
        start = content_clean.find('[')
        end = content_clean.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = content_clean[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        indices = item.get('original_indices', [])
                        trans = item.get('translation', '')
                        for idx in indices:
                            mapping[idx] = trans
                if mapping:
                    print(f"    提取成功: {len(mapping)} 条映射")
                    return mapping
    except:
        pass

    print(f"    解析失败，返回空映射")
    return mapping

def translate_batch(blocks, source_lang='eng', target_lang='zh'):
    """翻译字幕 - 分批确保完整"""
    texts = [b['text'] for b in blocks]
    total = len(texts)
    
    # 每批 15 条，带上下文翻译
    BATCH_SIZE = 15
    all_translations = []
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  翻译批次 {batch_num}/{total_batches}...")
        
        prompt = f"""请将以下{source_lang}字幕翻译成{target_lang}。

【翻译策略】
- **可以合并多条原文一起翻译**，让翻译更自然流畅
- 也可以单独翻译，每条原文独立翻译

【输出格式】
请返回JSON数组，每个元素表示一组翻译：

[
  {{"original_indices": [1, 2], "translation": "合并翻译1"}},
  {{"original_indices": [3], "translation": "单独翻译2"}},
  {{"original_indices": [4, 5], "translation": "合并翻译3"}}
]

说明：
1. original_indices 是数组，表示这组翻译对应哪些原文序号
2. 可以合并多条原文（如 [1, 2] 表示原文第1和第2条合并翻译）
3. **必须覆盖所有原文**，不能遗漏
4. 简洁明了，适合字幕显示
5. 专有名词首次出现时标注原文
6. 保持原意和说话语气

原文列表：
"""
        for i, text in enumerate(batch_texts):
            prompt += f'{i+1}. "{text}"\n'

        prompt += f"""

直接输出 JSON 数组，不要其他内容："""

        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32768,
            "temperature": 0.1
        }

        try:
            response = requests.post(f"{LLMS_HOST}/chat/completions", json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            print(f"    LLM返回: {len(content)} 字符")
            
            mapping = parse_translations(content, len(batch_texts))
            
            # 根据映射填充翻译列表
            batch_translations = []
            for i in range(1, len(batch_texts) + 1):
                if i in mapping:
                    batch_translations.append(mapping[i])
                else:
                    batch_translations.append(batch_texts[i-1])
            
            all_translations.extend(batch_translations)
            print(f"    批次 {batch_num}: {len(batch_translations)}/{len(batch_texts)} 条")
            
        except Exception as e:
            print(f"    批次 {batch_num} 错误: {e}")
            all_translations.extend(batch_texts)
    
    return all_translations

def parse_ass(ass_path):
    """解析 ASS 文件"""
    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    blocks = []
    in_events = False
    
    for line in content.split('\n'):
        if line.startswith('[Events]'):
            in_events = True
            continue
        if line.startswith('['):
            in_events = False
            continue
        if in_events and line.startswith('Dialogue:'):
            parts = line.split(',', 9)
            if len(parts) >= 10:
                start = parts[1]
                end = parts[2]
                text = parts[9].replace('\\N', ' ').replace('\\n', ' ')
                text = re.sub(r'\{[^}]*\}', '', text).strip()
                if text:
                    blocks.append({'start': start, 'end': end, 'text': text})
    
    return blocks

def create_bilingual_ass(blocks, translations, original_content=""):
    """创建双语 ASS"""
    if not original_content:
        header = """[Script Info]
Title: Bilingual Subtitles (LLM)
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,LXGW WenKai,46,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Top,LXGW WenKai,40,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,9,10,10,10,1
Style: Comment,LXGW WenKai,34,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,1,0,0,100,100,0,0,1,1,0,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    else:
        lines = original_content.split('\n')
        header = ""
        events_started = False
        for line in lines:
            if line.startswith('[Events]'):
                events_started = True
                header += line + '\n'
                header += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                break
            if not events_started:
                header += line + '\n'
    
    events = ""
    for block, trans in zip(blocks, translations):
        start = block['start']
        end = block['end']
        if trans:
            text = f"{block['text']}\\N{trans}"
        else:
            text = block['text']
        text = text.replace("{", "\\{").replace("}", "\\}")
        events += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    
    return header + events

def translate_ass(ass_path, source_lang='eng'):
    """翻译 ASS 字幕"""
    ass_path = Path(ass_path)
    
    print(f"读取字幕: {ass_path.name}")
    
    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        original_content = f.read()
    
    blocks = parse_ass(ass_path)
    
    if len(blocks) == 0:
        print("错误: 无法解析字幕文件！")
        return
    
    print(f"共 {len(blocks)} 条字幕")
    print(f"模型: {MODEL}")
    print()
    
    print("开始翻译...")
    start_time = time.time()
    
    translations = translate_batch(blocks, source_lang, 'zh')
    
    elapsed = time.time() - start_time
    print(f"\n翻译耗时: {format_elapsed(elapsed)}")
    
    success_count = sum(1 for t in translations if t is not None)
    print(f"成功翻译: {success_count}/{len(blocks)} ({success_count*100//len(blocks)}%)")
    
    # 查找视频文件，使用视频名作为输出名
    video_name = None
    video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.m4v']
    for ext in video_exts:
        potential_video = ass_path.parent / (ass_path.stem.replace('_en', '') + ext)
        if potential_video.exists():
            video_name = potential_video.stem
            break
    
    if video_name:
        output_path = ass_path.parent / f"{video_name}.ass"
    else:
        output_path = ass_path.parent / f"{ass_path.stem}.ass"
    
    ass_content = create_bilingual_ass(blocks, translations, original_content)
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(ass_content)
    
    print(f"\n完成! 双语字幕: {output_path.name}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")

def detect_language_simple(texts):
    """简单的语言检测"""
    sample = texts[0][:200] if texts else ""
    chinese = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    japanese = sum(1 for c in sample if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    
    if chinese > japanese:
        return 'zh'
    elif japanese > chinese:
        return 'ja'
    return 'en'

def main():
    ass_files = []
    source_lang = None
    
    args = sys.argv[1:]
    
    if '--lang=' in str(args):
        for arg in args:
            if arg.startswith('--lang='):
                source_lang = arg.replace('--lang=', '')
                args = [a for a in args if a != arg]
                break
    
    if not args:
        current_dir = Path(".")
        ass_files = list(current_dir.glob("*.ass")) + list(current_dir.glob("*.ASS"))
        
        if not ass_files:
            print("用法: python llm.py 字幕.ass [--lang=ja]")
            print(f"LM Studio: {LLMS_HOST}")
            print("当前目录没有找到 ASS 文件")
            return
        
        ass_files = [a.resolve() for a in ass_files]
        print(f"扫描当前目录，找到 {len(ass_files)} 个 ASS 文件")
    else:
        path = Path(args[0])
        if path.exists():
            ass_files = [path.resolve()]
        else:
            print(f"文件不存在: {args[0]}")
            return
    
    for ass_path in ass_files:
        print()
        print("=" * 60)
        
        if source_lang is None:
            blocks = parse_ass(ass_path)
            if blocks:
                source_lang = detect_language_simple([b['text'] for b in blocks])
                lang_names = {'zh': '中文', 'ja': '日语', 'en': '英语'}
                print(f"检测到语言: {lang_names.get(source_lang, source_lang)}")
        
        translate_ass(ass_path, source_lang or 'en')

if __name__ == "__main__":
    main()
