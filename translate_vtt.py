#!/usr/bin/env python3
"""
使用 CTranslate2 加载 NLLB 模型翻译 VTT 字幕
GPU 加速版本 - 批量优化
"""

import warnings
warnings.filterwarnings('ignore')

import ctranslate2
import transformers
import re
import time
from pathlib import Path

MODEL_DIR = "E:/cuda/nllb-200-3.3B-ct2-float16"

def load_translator():
    """加载 CTranslate2 翻译模型"""
    translator = ctranslate2.Translator(MODEL_DIR, device="cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    return translator, tokenizer

def translate_batch_fast(translator, tokenizer, texts, source_lang="eng_Latn", target_lang="zho_Hans", batch_size=128):
    """批量翻译 - 优化版"""
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 批量 tokenization
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"]
        
        batch_results = []
        
        for j in range(len(batch_texts)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            
            # 翻译
            results = translator.translate_batch([tokens], target_prefix=[[target_lang]])
            
            # 清理输出
            result_tokens = results[0].hypotheses[0]
            if result_tokens and result_tokens[0] == target_lang:
                result_tokens = result_tokens[1:]
            
            result = tokenizer.convert_tokens_to_string(result_tokens).strip()
            batch_results.append(result)
        
        all_results.extend(batch_results)
        
        progress = min(i + batch_size, len(texts))
        print(f"  进度: {progress}/{len(texts)} ({progress*100//len(texts)}%)")
    
    return all_results

def translate_batch(translator, tokenizer, texts, source_lang="eng_Latn", target_lang="zho_Hans"):
    """批量翻译"""
    return translate_batch_fast(translator, tokenizer, texts, source_lang, target_lang, batch_size=128)

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

def create_single_vtt(blocks, translations):
    """创建单语 VTT"""
    vtt_content = "WEBVTT\n\n"
    
    for i, (block, trans) in enumerate(zip(blocks, translations), 1):
        vtt_content += f"{block['start']} --> {block['end']}\n"
        vtt_content += f"{trans}\n\n"
    
    return vtt_content

def translate_vtt(vtt_path, output_path=None, batch_size=128, mode="both"):
    """翻译 VTT 文件"""
    vtt_path = Path(vtt_path)
    
    if output_path is None:
        if mode == "both":
            output_path = vtt_path.with_suffix('.bilingual.vtt')
        else:
            output_path = vtt_path.with_suffix('.zh.vtt')
    
    print(f"加载翻译模型: {MODEL_DIR}")
    translator, tokenizer = load_translator()
    print(f"设备: {translator.device}\n")
    
    print(f"解析字幕: {vtt_path.name}")
    blocks = parse_vtt(vtt_path)
    print(f"共 {len(blocks)} 条字幕\n")
    
    print("开始翻译 (English -> 中文)...")
    start_time = time.time()
    
    # 提取所有文本
    all_texts = [b['text'] for b in blocks]
    
    # 批量翻译
    translations = translate_batch_fast(
        translator, tokenizer, all_texts,
        source_lang="eng_Latn",
        target_lang="zho_Hans",
        batch_size=batch_size
    )
    
    elapsed = time.time() - start_time
    print(f"\n翻译耗时: {elapsed:.2f}秒 ({elapsed/len(blocks)*1000:.0f}ms/条)")
    
    print(f"\n生成{'双语' if mode == 'both' else '中文'}字幕...")
    
    if mode == "both":
        vtt_content = create_bilingual_vtt(blocks, translations)
    else:
        vtt_content = create_single_vtt(blocks, translations)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(vtt_content)
    
    print(f"\n完成! 保存至: {output_path}")
    return output_path

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python translate_vtt.py 字幕.vtt")
        print("  python translate_vtt.py 字幕.vtt --single")
        print()
        print(f"模型目录: {MODEL_DIR}")
        print()
        print("参数:")
        print("  (默认)  生成双语字幕")
        print("  --single  只生成中文译文")
        return
    
    vtt_path = sys.argv[1]
    mode = "both" if "--single" not in sys.argv else "single"
    
    translate_vtt(vtt_path, mode=mode)

if __name__ == "__main__":
    main()
