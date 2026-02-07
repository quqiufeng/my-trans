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
import sys
from pathlib import Path

MODEL_DIR = "E:/cuda/nllb-200-3.3B-ct2-float16"

LANG_CODE_MAP = {
    'ja': 'jpn_Jpan',
    'japanese': 'jpn_Jpan',
    'en': 'eng_Latn',
    'english': 'eng_Latn',
    'zh': 'zho_Hans',
    'chinese': 'zho_Hans',
    'ko': 'kor_Hang',
    'korean': 'kor_Hang',
    'fr': 'fra_Latn',
    'french': 'fra_Latn',
    'de': 'deu_Latn',
    'german': 'deu_Latn',
    'es': 'spa_Latn',
    'spanish': 'spa_Latn',
}

SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())

def print_memory_usage():
    """打印当前 GPU 内存使用情况"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  GPU Memory: 已分配 {allocated:.2f}GB, 保留 {reserved:.2f}GB")

def cleanup_translator(translator, tokenizer):
    """释放 GPU 内存"""
    import torch
    del translator
    del tokenizer
    torch.cuda.empty_cache()

def detect_language_simple(texts):
    """简单的语言检测"""
    try:
        from langdetect import detect, LangDetectException

        sample = texts[0][:200] if texts else ""
        lang = detect(sample)
        return LANG_CODE_MAP.get(lang, 'eng_Latn')

    except ImportError:
        return None

def load_translator():
    """加载 CTranslate2 翻译模型"""
    translator = ctranslate2.Translator(MODEL_DIR, device="cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    return translator, tokenizer

def translate_batch_fast(translator, tokenizer, texts, source_lang="eng_Latn", target_lang="zho_Hans", batch_size=128):
    """批量翻译 - 优化版"""
    import torch
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"]
        
        batch_results = []
        
        for j in range(len(batch_texts)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            results = translator.translate_batch([tokens], target_prefix=[[target_lang]])
            
            result_tokens = results[0].hypotheses[0]
            if result_tokens and result_tokens[0] == target_lang:
                result_tokens = result_tokens[1:]
            
            result = tokenizer.convert_tokens_to_string(result_tokens).strip()
            batch_results.append(result)
        
        all_results.extend(batch_results)
        
        del encoded, input_ids
        torch.cuda.empty_cache()
        
        progress = min(i + batch_size, len(texts))
        print(f"  进度: {progress}/{len(texts)} ({progress*100//len(texts)}%)", end='\r')
    
    print(f"  进度: {len(texts)}/{len(texts)} (100%)")
    
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

def translate_vtt(vtt_path, output_path=None, batch_size=128, source_lang=None):
    """翻译 VTT 文件为双语字幕"""
    import torch
    vtt_path = Path(vtt_path)
    
    if output_path is None:
        output_path = vtt_path.with_suffix('.bilingual.vtt')
    
    print(f"加载翻译模型: {MODEL_DIR}")
    translator, tokenizer = load_translator()
    print(f"设备: {translator.device}\n")
    print_memory_usage()
    
    print(f"\n解析字幕: {vtt_path.name}")
    blocks = parse_vtt(vtt_path)
    print(f"共 {len(blocks)} 条字幕\n")
    
    all_texts = [b['text'] for b in blocks]
    
    if source_lang is None:
        print("检测字幕语言...")
        source_lang = detect_language_simple(all_texts)
        
        if source_lang is None:
            print("未安装 langdetect，使用关键词检测...")
            sample = all_texts[0].lower() if all_texts else ""
            
            japanese_chars = sum(1 for c in sample if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
            chinese_chars = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
            
            if japanese_chars > chinese_chars:
                source_lang = 'jpn_Jpan'
                print("检测到语言: 日语 (jpn_Jpan)")
            elif chinese_chars > 0:
                source_lang = 'zho_Hans'
                print("检测到语言: 中文 (zho_Hans)")
            else:
                source_lang = 'eng_Latn'
                print("检测到语言: 英语 (eng_Latn)")
    
    target_lang = 'zho_Hans'
    lang_name = {'jpn_Jpan': '日语', 'eng_Latn': '英语', 'zho_Hans': '中文'}.get(source_lang, source_lang)
    print(f"翻译: {lang_name} -> 中文\n")
    
    print("开始翻译...")
    start_time = time.time()
    
    translations = translate_batch_fast(
        translator, tokenizer, all_texts,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size
    )
    
    elapsed = time.time() - start_time
    print(f"\n翻译耗时: {elapsed:.2f}秒 ({elapsed/len(blocks)*1000:.0f}ms/条)")
    
    print(f"\n生成双语字幕...")
    
    vtt_content = create_bilingual_vtt(blocks, translations)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(vtt_content)
    
    print(f"\n完成! 保存至: {output_path}")
    
    print("释放 GPU 内存...")
    cleanup_translator(translator, tokenizer)
    print_memory_usage()
    
    return output_path

def main():
    import sys
    from pathlib import Path
    
    vtt_files = []
    source_lang = None
    
    if len(sys.argv) >= 2 and sys.argv[1].startswith('--lang='):
        source_lang = sys.argv[1].replace('--lang=', '')
        args = sys.argv[2:]
    elif len(sys.argv) >= 3 and sys.argv[1] == '--lang':
        source_lang = sys.argv[2]
        args = sys.argv[3:]
    else:
        args = sys.argv[1:]
    
    if not args:
        current_dir = Path(".")
        vtt_files = list(current_dir.glob("*.vtt")) + list(current_dir.glob("*.VTT"))
        
        if not vtt_files:
            print("用法:")
            print("  python translate_vtt.py 字幕.vtt")
            print("  python translate_vtt.py --lang=ja 字幕.vtt")
            print()
            print("支持语言: ja(日语), en(英语), zh(中文), ko(韩语), fr(法语), de(德语), es(西班牙语)")
            print()
            print(f"模型目录: {MODEL_DIR}")
            print()
            print("当前目录没有找到 .vtt 字幕文件")
            return
        
        vtt_files = [v.resolve() for v in vtt_files]
        print(f"扫描当前目录，找到 {len(vtt_files)} 个字幕文件")
    else:
        vtt_path = args[0]
        path = Path(vtt_path)
        if path.exists():
            vtt_files = [path.resolve()]
        else:
            print(f"文件不存在: {vtt_path}")
            return
    
    for vtt_path in vtt_files:
        print()
        print("=" * 60)
        translate_vtt(vtt_path, source_lang=source_lang)

if __name__ == "__main__":
    main()
