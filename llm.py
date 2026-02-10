#!/usr/bin/env python3
"""
使用 LM Studio (Qwen2.5-7B) 翻译 ASS 字幕
生成中英双语字幕
"""

import warnings
warnings.filterwarnings('ignore')

import requests
import re
import time
import sys
from pathlib import Path

LLMS_HOST = "http://192.168.124.3:11434/v1"
MODEL = "qwen2.5-7b-instruct"

LANG_CODE_MAP = {
    'ja': 'jpn',
    'japanese': 'jpn',
    'en': 'eng',
    'english': 'eng',
    'zh': 'zh',
    'chinese': 'zh',
    'ko': 'kor',
    'korean': 'kor',
    'fr': 'fra',
    'french': 'fra',
    'de': 'deu',
    'german': 'deu',
    'es': 'spa',
    'spanish': 'spa',
}

def format_elapsed(seconds):
    """将秒数转换为易读格式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds//60}分{seconds%60:.0f}秒"
    else:
        return f"{seconds//3600}小时{(seconds%3600)//60}分"

def translate_batch(blocks, source_lang='eng', target_lang='zh'):
    """批量翻译字幕"""
    # 提取所有文本
    texts = [b['text'] for b in blocks]
    
    prompt = f"""请将以下{source_lang}字幕翻译成{target_lang}。

要求：
1. 简洁明了，适合字幕显示（每行尽量短）
2. 专有名词首次出现时标注原文，如：Transformer（转换器）
3. 保持原意和说话语气
4. 不要添加额外解释

请按以下格式输出（每行一条）：
原文|翻译

以下是字幕内容：
"""
    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.3
    }

    try:
        response = requests.post(f"{LLMS_HOST}/chat/completions", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"翻译错误: {e}")
        return None

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
                text = re.sub(r'\{[^}]*\}', '', text)
                text = text.strip()
                if text:
                    blocks.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })
    
    return blocks

def parse_translations(content, total_count):
    """解析翻译结果"""
    translations = [None] * total_count
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        if line.startswith('`'):
            line = line.strip('`').strip()
        if '|' not in line:
            continue
            
        parts = line.split('|', 1)
        if len(parts) != 2:
            continue
            
        left = parts[0].strip()
        right = parts[1].strip()
        
        # 跳过序号
        if left.isdigit():
            idx = int(left) - 1
            if 0 <= idx < total_count:
                translations[idx] = right
        # 跳过原文
        elif left.startswith(f'{total_count + 1}.'):
            continue
        else:
            # 可能是 "1. 原文|翻译" 格式
            match = re.match(r'^(\d+)\.\s*(.+)\|(.+)', line)
            if match:
                idx = int(match.group(1)) - 1
                trans = match.group(3).strip()
                if 0 <= idx < total_count:
                    translations[idx] = trans
    
    return translations

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
            text = f"{trans}\\N{block['text']}"
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
    print(f"源语言: {source_lang}")
    print(f"目标语言: zh")
    print(f"模型: {MODEL}")
    print()
    
    # 翻译
    print("开始翻译...")
    start_time = time.time()
    
    raw_result = translate_batch(blocks, source_lang, 'zh')
    
    if not raw_result:
        print("翻译失败")
        return
    
    elapsed = time.time() - start_time
    print(f"翻译耗时: {format_elapsed(elapsed)}")
    print()
    
    # 解析结果
    translations = parse_translations(raw_result, len(blocks))
    
    # 检查翻译成功率
    success_count = sum(1 for t in translations if t is not None)
    print(f"成功翻译: {success_count}/{len(blocks)} ({success_count*100//len(blocks)}%)")
    
    # 生成双语字幕
    output_path = ass_path.parent / f"{ass_path.stem}_llm.ass"
    ass_content = create_bilingual_ass(blocks, translations, original_content)
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(ass_content)
    
    print(f"\n完成! 双语字幕: {output_path.name}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")

def detect_language_simple(texts):
    """简单的语言检测"""
    sample = texts[0][:200] if texts else ""
    
    chinese_chars = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    japanese_chars = sum(1 for c in sample if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    
    if chinese_chars > japanese_chars:
        return 'zh'
    elif japanese_chars > chinese_chars:
        return 'ja'
    else:
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
            print("用法:")
            print("  python llm.py 字幕.ass")
            print("  python llm.py --lang=ja 字幕.ass")
            print()
            print("支持语言: ja, en, zh, ko, fr, de, es")
            print(f"LM Studio: {LLMS_HOST}")
            print(f"模型: {MODEL}")
            print()
            print("当前目录没有找到 ASS 文件")
            return
        
        ass_files = [a.resolve() for a in ass_files]
        print(f"扫描当前目录，找到 {len(ass_files)} 个 ASS 文件")
    else:
        ass_path = args[0]
        path = Path(ass_path)
        if path.exists():
            ass_files = [path.resolve()]
        else:
            print(f"文件不存在: {ass_path}")
            return
    
    for ass_path in ass_files:
        print()
        print("=" * 60)
        
        # 如果没有指定语言，检测
        if source_lang is None:
            blocks = parse_ass(ass_path)
            if blocks:
                texts = [b['text'] for b in blocks]
                source_lang = detect_language_simple(texts)
                lang_names = {'zh': '中文', 'ja': '日语', 'en': '英语'}
                print(f"检测到语言: {lang_names.get(source_lang, source_lang)}")
        
        translate_ass(ass_path, source_lang or 'en')

if __name__ == "__main__":
    main()
