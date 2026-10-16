#!/usr/bin/env python3
"""
使用本地 llama.cpp (Hunyuan-MT-7B) 翻译 ASS 字幕

Hunyuan-MT 是专用翻译模型，偏好完整句子；对字幕碎片逐条翻译时保持
1:1 对齐最稳定（CONTEXT_SIZE=0 默认）。若开启窗口上下文
（CONTEXT_SIZE>0），前 N 条原文+译文会作为 system 参考传入，
但实测可能引入重复，建议配合人工校对后的完整句字幕使用。

启动服务:
  bash start_server.sh
"""

import warnings
warnings.filterwarnings('ignore')

import requests
import re
import time
import sys
from pathlib import Path

LLMS_HOST = "http://127.0.0.1:8080/v1"
MODEL = "hunyuan-mt"
CONTEXT_SIZE = 0

TARGET_LANG_NAMES = {
    'zh': 'Chinese',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'ru': 'Russian',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ar': 'Arabic',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
}

def format_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds//60}分{seconds%60:.0f}秒"
    else:
        return f"{seconds//3600}小时{(seconds%3600)//60}分"

def parse_ass(ass_path):
    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    lines = content.split('\n')
    header = ""
    events_start = -1
    
    for i, line in enumerate(lines):
        if line.startswith('[Events]'):
            events_start = i
            header = '\n'.join(lines[:i+2])
            break
    
    dialogue_lines = []
    for line in lines[events_start+2:]:
        if line.startswith('Dialogue:'):
            dialogue_lines.append(line)
    
    return header, dialogue_lines

def translate_with_context(english_text, prev_pairs, idx, total, target_name="Chinese"):
    messages = []
    if prev_pairs:
        context = "\n".join(f"{src} => {trans}" for src, trans in prev_pairs)
        messages.append({
            "role": "system",
            "content": "Previous subtitle context (for reference only, do not translate):\n" + context
        })

    messages.append({
        "role": "user",
        "content": f"Translate the following segment into {target_name}, "
                   f"without additional explanation.\n\n{english_text}"
    })

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0
    }
    
    try:
        response = requests.post(f"{LLMS_HOST}/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"    错误: {e}")
        return None

def translate_ass(ass_path, target_lang='zh'):
    ass_path = Path(ass_path)
    target_name = TARGET_LANG_NAMES.get(target_lang, target_lang)
    
    print(f"读取字幕: {ass_path.name}")
    
    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        original_content = f.read()
    
    header, dialogue_lines = parse_ass(ass_path)
    
    print(f"共 {len(dialogue_lines)} 条字幕")
    print(f"目标语言: {target_name} ({target_lang})")
    if CONTEXT_SIZE > 0:
        print(f"上下文窗口: 前{CONTEXT_SIZE}条(原文+译文)")
    else:
        print("上下文窗口: 关闭 (逐条 1:1 翻译)")
    print()
    
    texts = []
    for line in dialogue_lines:
        if line.startswith('Dialogue:'):
            parts = line.split(',', 9)
            if len(parts) >= 10:
                text = parts[9].replace('\\N', ' ').replace('\\n', ' ')
                text = re.sub(r'\{[^}]*\}', '', text).strip()
                texts.append(text)
    
    print("开始翻译...")
    start_time = time.time()
    
    translations = []
    prev_pairs = []
    
    for i, text in enumerate(texts):
        if i % 10 == 0 or i == len(texts) - 1:
            print(f"  翻译 {i+1}/{len(texts)}...")
        
        trans = translate_with_context(text, prev_pairs, i+1, len(texts), target_name)
        
        if trans:
            translations.append(trans)
            prev_pairs.append((text, trans))
        else:
            translations.append(text)
            prev_pairs.append((text, text))

        if CONTEXT_SIZE > 0:
            prev_pairs = prev_pairs[-CONTEXT_SIZE:]
        else:
            prev_pairs = []
        
        time.sleep(0.05)
    
    elapsed = time.time() - start_time
    print(f"\n翻译耗时: {format_elapsed(elapsed)}")
    
    success_count = sum(1 for i, t in enumerate(translations) if t and t != texts[i])
    print(f"成功翻译: {success_count}/{len(dialogue_lines)} ({success_count*100//len(dialogue_lines)}%)")
    
    events = header + '\n'
    for i, line in enumerate(dialogue_lines):
        if line.startswith('Dialogue:'):
            parts = line.split(',', 9)
            if len(parts) >= 10:
                prefix = ','.join(parts[:9]) + ','
                original_text = parts[9].rstrip('\n')
                trans = translations[i] if i < len(translations) else ""
                
                if trans and trans != texts[i]:
                    new_text = f"{original_text}\\N{trans}"
                else:
                    new_text = original_text
                
                events += f"{prefix}{new_text}\n"
            else:
                events += line + '\n'
        else:
            events += line + '\n'
    
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
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(events)
    
    print(f"\n完成! 双语字幕: {output_path.name}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")

def main():
    args = sys.argv[1:]
    target_lang = 'zh'

    if '--lang=' in str(args):
        for arg in args:
            if arg.startswith('--lang='):
                target_lang = arg.replace('--lang=', '')
                args = [a for a in args if a != arg]
                break
    elif '--lang' in args:
        idx = args.index('--lang')
        if idx + 1 < len(args):
            target_lang = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if not args:
        current_dir = Path(".")
        ass_files = list(current_dir.glob("*.ass")) + list(current_dir.glob("*.ASS"))
        
        if not ass_files:
            print("用法: python llm.py 字幕.ass")
            print(f"llama.cpp 服务: {LLMS_HOST}")
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
        translate_ass(ass_path, target_lang=target_lang)

if __name__ == "__main__":
    main()
