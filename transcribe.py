#!/usr/bin/env python3
"""
使用 faster-whisper 本地模型生成英文字幕
支持单个或批量视频文件
"""

import warnings
warnings.filterwarnings('ignore')

import time
import sys
import re
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline

WHISPER_MODEL_PATH = "e:/cuda/faster-whisper-medium"

def print_memory_usage():
    """打印当前 GPU 内存使用情况"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  GPU Memory: 已分配 {allocated:.2f}GB, 保留 {reserved:.2f}GB")

def cleanup_whisper(model):
    """释放 Whisper 模型 GPU 内存"""
    import torch
    del model
    torch.cuda.empty_cache()

def wrap_text(text, max_chars=50):
    """长文本自动换行"""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return "\n".join(lines)

def split_sentences(text):
    """按句子边界分割文本"""
    end_markers = ['. ', '? ', '! ', '。', '？', '！', '.\n', '?\n', '!\n']
    
    paragraphs = text.split('\n')
    result = []
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        sentences = []
        current = ""
        
        i = 0
        while i < len(para):
            char = para[i]
            current += char
            
            for marker in end_markers:
                if i - len(marker) + 1 >= 0 and para[i - len(marker) + 1 : i + 1] == marker:
                    if current.strip():
                        sentences.append(current.strip())
                    current = ""
                    break
            else:
                if len(current) > 50 and char == ' ':
                    if current.strip():
                        sentences.append(current.strip())
                    current = ""
            
            i += 1
        
        if current.strip():
            sentences.append(current.strip())
        
        result.extend(sentences)
    
    return result

def format_time(seconds):
    """将秒数转换为 VTT 时间格式 (00:00:00.000)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def format_elapsed(seconds):
    """将秒数转换为易读格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"

def transcribe_video(video_path, model, batched_model):
    """转录音频并保存为 VTT 格式"""
    video_path = Path(video_path)
    output_path = video_path.with_suffix('.vtt')
    
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    start_time = time.time()
    
    print(f"转录: {video_path.name} ({file_size_mb:.1f} MB)")
    
    segments, info = batched_model.transcribe(str(video_path), batch_size=8)
    
    vtt_content = "WEBVTT\n\n"
    
    for segment in segments:
        start_time_fmt = format_time(segment.start)
        end_time_fmt = format_time(segment.end)
        text = segment.text.strip()
        
        sentences = split_sentences(text)
        
        if len(sentences) == 1:
            wrapped = wrap_text(text, 50)
            vtt_content += f"{start_time_fmt} --> {end_time_fmt}\n{wrapped}\n\n"
        else:
            start_sec = segment.start
            duration = segment.end - segment.start
            avg_duration = duration / len(sentences)
            
            for i, sentence in enumerate(sentences):
                sentence_start = start_time_fmt
                sentence_end = format_time(start_sec + avg_duration)
                
                wrapped = wrap_text(sentence, 50)
                vtt_content += f"{sentence_start} --> {sentence_end}\n{wrapped}\n\n"
                
                start_sec += avg_duration
                start_time_fmt = sentence_end
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(vtt_content)
    
    elapsed = time.time() - start_time
    elapsed_str = format_elapsed(elapsed)
    
    print(f"  → {output_path.name} ({elapsed_str})")
    return output_path, elapsed, file_size_mb

def main():
    video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.m4v']
    
    if len(sys.argv) < 2:
        # 不带参数时，扫描当前目录下的所有视频文件
        current_dir = Path(".")
        video_files = []
        for ext in video_exts:
            video_files.extend(current_dir.glob(f"*{ext}"))
            video_files.extend(current_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print("用法:")
            print("  python transcribe.py 视频1.mp4")
            print("  python transcribe.py 视频1.mp4 视频2.mp4")
            print()
            print("支持格式:", ", ".join(video_exts))
            print()
            print("当前目录没有找到视频文件")
            return
        
        video_files = [v.resolve() for v in video_files]
        print(f"扫描当前目录，找到 {len(video_files)} 个视频文件")
        print()
    else:
        video_files = []
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.exists():
                video_files.append(path.resolve())
            else:
                print(f"文件不存在: {arg}")
        
        if not video_files:
            print("没有有效视频文件")
            return
    
    print("=" * 60)
    print("加载 Whisper 模型...")
    model = WhisperModel(WHISPER_MODEL_PATH, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    model.eval()
    print("模型加载完成!\n")
    print_memory_usage()
    
    print(f"\n处理 {len(video_files)} 个视频")
    print("-" * 60)
    
    total_elapsed = 0
    total_size = 0
    results = []
    
    for i, video in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}]")
        output_path, elapsed, file_size = transcribe_video(video, model, batched_model)
        results.append((video.name, elapsed, file_size))
        total_elapsed += elapsed
        total_size += file_size
        print_memory_usage()
    
    print("-" * 60)
    print(f"完成! 共 {len(results)} 个视频")
    print(f"总耗时: {format_elapsed(total_elapsed)}")
    print(f"总大小: {total_size:.1f} MB")
    print()
    print("各视频详情:")
    print(f"{'文件名':<50} {'大小':<10} {'耗时'}")
    print("-" * 70)
    for name, elapsed, file_size in results:
        print(f"{name:<50} {file_size:<10.1f}MB {format_elapsed(elapsed)}")
    
    print("\n释放 GPU 内存...")
    cleanup_whisper(model)
    print_memory_usage()

if __name__ == "__main__":
    main()
