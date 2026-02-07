#!/usr/bin/env python3
"""
使用 Hugging Face Transformers 直接翻译 ASS 字幕
无需 CTranslate2
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import transformers
import re
import time
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

def cleanup_model(tokenizer, model):
    """释放 GPU 内存"""
    import torch
    del model
    del tokenizer
    torch.cuda.empty_cache()

def print_memory_usage():
    """打印当前 GPU 内存使用情况"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  GPU Memory: 已分配 {allocated:.2f}GB, 保留 {reserved:.2f}GB")

def load_translator():
    """加载翻译模型到 GPU"""
    import torch
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to("cuda")
    model.eval()
    return tokenizer, model

@torch.no_grad()
def translate_text(tokenizer, model, text, source_lang="eng_Latn", target_lang="zho_Hans"):
    """翻译单条文本"""
    import torch
    source_token = source_lang
    target_token = target_lang
    
    encoded = tokenizer(source_token + " " + text, return_tensors="pt")
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    
    target_lang_id = tokenizer.convert_tokens_to_ids(target_token)
    
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=target_lang_id,
        max_new_tokens=512
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    del encoded, generated_tokens
    return result[0]

@torch.no_grad()
def translate_batch(tokenizer, model, texts, source_lang="eng_Latn", target_lang="zho_Hans"):
    """批量翻译"""
    results = []
    for text in texts:
        results.append(translate_text(tokenizer, model, text, source_lang, target_lang))
    return results

def parse_vtt(vtt_path):
    """解析 VTT 文件"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = []
    header, content = content.split('\n\n', 1)
    
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        start = match.group(1)
        end = match.group(2)
        text = match.group(3).strip().replace('\n', ' ')
        blocks.append({
            'start': start,
            'end': end,
            'text': text
        })
    
    return blocks

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

def create_bilingual_vtt(blocks, translations):
    """创建双语 VTT"""
    vtt_content = "WEBVTT\n\n"
    
    for i, (block, trans) in enumerate(zip(blocks, translations), 1):
        vtt_content += f"{block['start']} --> {block['end']}\n"
        vtt_content += f"{trans}\n"
        vtt_content += f"{block['text']}\n\n"
    
    return vtt_content

def create_bilingual_ass(blocks, translations, original_content=""):
    """创建双语 ASS"""
    if not original_content:
        header = """[Script Info]
Title: Bilingual Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Source Han Sans CN,42,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Top,Source Han Sans CN,36,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,9,10,10,10,1
Style: Comment,Source Han Sans CN,30,&H0000A5FF,&H000000FF,&H00000000,&H00000000,-1,1,0,0,100,100,0,0,1,1,0,7,10,10,10,1

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
        text = f"{trans}\\N{block['text']}"
        text = text.replace("{", "\\{").replace("}", "\\}")
        events += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    
    return header + events

def translate_vtt(vtt_path, output_path=None, batch_size=8):
    """翻译 VTT/ASS 文件为双语字幕"""
    import torch
    vtt_path = Path(vtt_path)
    
    if output_path is None:
        if vtt_path.suffix.lower() == '.ass':
            output_path = vtt_path
        else:
            output_path = vtt_path.with_suffix('.bilingual.vtt')
    
    print(f"加载翻译模型: {MODEL_NAME}")
    tokenizer, model = load_translator()
    print("模型加载成功! (使用 Transformers)\n")
    print_memory_usage()
    
    print(f"\n解析字幕: {vtt_path.name}")
    blocks = parse_vtt(vtt_path)
    print(f"共 {len(blocks)} 条字幕\n")
    
    print("开始翻译 (English -> 中文)...")
    start_time = time.time()
    
    translations = []
    for i in range(0, len(blocks), batch_size):
        batch_blocks = blocks[i:i+batch_size]
        batch_texts = [b['text'] for b in batch_blocks]
        
        batch_trans = translate_batch(
            tokenizer, model, batch_texts,
            source_lang="eng_Latn",
            target_lang="zho_Hans"
        )
        
        translations.extend(batch_trans)
        
        progress = min(i + batch_size, len(blocks))
        print(f"  进度: {progress}/{len(blocks)} ({progress*100//len(blocks)}%)", end='\r')
    
    print(f"  进度: {len(blocks)}/{len(blocks)} (100%)")
    
    elapsed = time.time() - start_time
    if len(blocks) > 0:
        print(f"\n翻译耗时: {elapsed:.2f}秒 ({elapsed/len(blocks)*1000:.0f}ms/条)")
    else:
        print(f"\n翻译耗时: {elapsed:.2f}秒")
    
    print(f"\n生成双语字幕...")
    
    if vtt_path.suffix.lower() == '.ass':
        with open(vtt_path, 'r', encoding='utf-8-sig') as f:
            original_content = f.read()
        
        ass_content = create_bilingual_ass(blocks, translations, original_content)
        
        bilingual_path = vtt_path.parent / f"{vtt_path.stem.rsplit('_', 1)[0]}.ass"
        with open(bilingual_path, 'w', encoding='utf-8-sig') as f:
            f.write(ass_content)
        
        print(f"\n完成! 双语字幕: {bilingual_path.name}")
    else:
        vtt_content = create_bilingual_vtt(blocks, translations)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        
        print(f"\n完成! 保存至: {output_path}")
    
    print("释放 GPU 内存...")
    cleanup_model(tokenizer, model)
    print_memory_usage()
    
    return output_path

def main():
    import sys
    from pathlib import Path
    
    vtt_files = []
    
    if len(sys.argv) < 2:
        # 不带参数时，扫描当前目录的 .vtt 文件
        current_dir = Path(".")
        vtt_files = list(current_dir.glob("*.vtt")) + list(current_dir.glob("*.VTT"))
        
        if not vtt_files:
            print("用法:")
            print("  python translate_nllb_official.py 字幕.vtt")
            print()
            print(f"模型: {MODEL_NAME}")
            print()
            print("当前目录没有找到 .vtt 字幕文件")
            return
        
        vtt_files = [v.resolve() for v in vtt_files]
        print(f"扫描当前目录，找到 {len(vtt_files)} 个字幕文件")
    else:
        vtt_path = sys.argv[1]
        path = Path(vtt_path)
        if path.exists():
            vtt_files = [path.resolve()]
        else:
            print(f"文件不存在: {vtt_path}")
            return
    
    for vtt_path in vtt_files:
        print()
        print("=" * 60)
        translate_vtt(vtt_path)

if __name__ == "__main__":
    main()
