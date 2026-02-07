#!/usr/bin/env python3
"""
使用 Hugging Face Transformers 直接翻译 VTT 字幕
无需 CTranslate2
"""

import warnings
warnings.filterwarnings('ignore')

import transformers
import re
import time
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

def load_translator():
    """加载翻译模型到 GPU"""
    import torch
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to("cuda")
    return tokenizer, model

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
    return result[0]

def translate_batch(tokenizer, model, texts, source_lang="eng_Latn", target_lang="zho_Hans"):
    """逐条翻译"""
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

def create_bilingual_vtt(blocks, translations):
    """创建双语 VTT"""
    vtt_content = "WEBVTT\n\n"
    
    for i, (block, trans) in enumerate(zip(blocks, translations), 1):
        vtt_content += f"{block['start']} --> {block['end']}\n"
        vtt_content += f"{trans}\n"
        vtt_content += f"{block['text']}\n\n"
    
    return vtt_content

def translate_vtt(vtt_path, output_path=None, batch_size=8):
    """翻译 VTT 文件为双语字幕"""
    vtt_path = Path(vtt_path)
    
    if output_path is None:
        output_path = vtt_path.with_suffix('.bilingual.vtt')
    
    print(f"加载翻译模型: {MODEL_NAME}")
    tokenizer, model = load_translator()
    print("模型加载成功! (使用 Transformers)\n")
    
    print(f"解析字幕: {vtt_path.name}")
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
        print(f"  进度: {progress}/{len(blocks)} ({progress*100//len(blocks)}%)")
    
    elapsed = time.time() - start_time
    print(f"\n翻译耗时: {elapsed:.2f}秒 ({elapsed/len(blocks)*1000:.0f}ms/条)")
    
    print(f"\n生成双语字幕...")
    
    vtt_content = create_bilingual_vtt(blocks, translations)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(vtt_content)
    
    print(f"\n完成! 保存至: {output_path}")
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
