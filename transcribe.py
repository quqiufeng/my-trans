#!/usr/bin/env python3
"""
使用 faster-whisper 本地模型生成 ASS 字幕
支持单个或批量视频文件，默认输出 ASS 格式
"""

import warnings
warnings.filterwarnings('ignore')

import time
import sys
import re
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline

WHISPER_MODEL_PATH = "e:/cuda/faster-whisper-large-v3"

def get_model_size():
    """根据模型路径返回推荐配置"""
    if "large" in WHISPER_MODEL_PATH.lower():
        return "large"
    elif "medium" in WHISPER_MODEL_PATH.lower():
        return "medium"
    else:
        return "base"

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

def format_time_ass(seconds):
    """将秒数转换为 ASS 时间格式 (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def format_time_vtt(seconds):
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
    
    return "\\N".join(lines)

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

def create_ass_header():
    """创建 ASS 字幕头部"""
    return """[Script Info]
Title: Auto Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Source Han Sans CN,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1
Style: Top,Source Han Sans CN,32,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,9,10,10,10,1
Style: Comment,Source Han Sans CN,28,&H00AAAAAA,&H000000FF,&H00000000,&H00000000,-1,1,0,0,100,100,0,0,1,1,0,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def create_ass_dialogue(start, end, text, style="Default"):
    """创建 ASS 对话行"""
    start_fmt = format_time_ass(start)
    end_fmt = format_time_ass(end)
    text_escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return f"Dialogue: 0,{start_fmt},{end_fmt},{style},,0,0,0,,{text_escaped}\n"

def transcribe_video(video_path, model, batched_model, output_format="vtt"):
    """转录音频并保存为字幕格式"""
    video_path = Path(video_path)
    
    if output_format == "ass":
        output_path = video_path.with_suffix('.ass')
    else:
        output_path = video_path.with_suffix('.vtt')
    
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    start_time = time.time()
    
    print(f"转录: {video_path.name} ({file_size_mb:.1f} MB)")
    
    model_size = get_model_size()
    print(f"模型: {model_size}, 使用高精度模式...")
    
    segments, info = batched_model.transcribe(
        str(video_path),
        batch_size=8,
        beam_size=5,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        patience=1.0
    )
    
    if output_format == "ass":
        ass_content = create_ass_header()
        dialogue_count = 0
        
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            
            wrapped = wrap_text(text, 45)
            ass_content += create_ass_dialogue(start, end, wrapped)
            dialogue_count += 1
        
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write(ass_content)
        
        elapsed = time.time() - start_time
        elapsed_str = format_elapsed(elapsed)
        print(f"  → {output_path.name} ({dialogue_count} 条, {elapsed_str})")
        return output_path, elapsed, file_size_mb, dialogue_count
    
    else:
        vtt_content = "WEBVTT\n\n"
        dialogue_count = 0
        
        for segment in segments:
            start_time_fmt = format_time_vtt(segment.start)
            end_time_fmt = format_time_vtt(segment.end)
            text = segment.text.strip()
            
            wrapped = wrap_text(text, 50)
            vtt_content += f"{start_time_fmt} --> {end_time_fmt}\n{wrapped}\n\n"
            dialogue_count += 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        
        elapsed = time.time() - start_time
        elapsed_str = format_elapsed(elapsed)
        print(f"  → {output_path.name} ({dialogue_count} 条, {elapsed_str})")
        return output_path, elapsed, file_size_mb, dialogue_count

def main():
    video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.m4v']
    
    output_format = "vtt"
    
    args = sys.argv[1:]
    if '--vtt' in args:
        output_format = "vtt"
        args = [a for a in args if a != '--vtt']
    
    if not args:
        current_dir = Path(".")
        video_files = []
        for ext in video_exts:
            video_files.extend(current_dir.glob(f"*{ext}"))
            video_files.extend(current_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print("用法:")
            print("  python transcribe.py 视频1.mp4")
            print("  python transcribe.py 视频1.mp4 视频2.mp4")
            print("  python transcribe.py --vtt 视频.mp4  # 输出 VTT 格式")
            print()
            print("支持格式:", ", ".join(video_exts))
            print()
            print("输出格式: ASS (默认, 思源字体)")
            print("         --vtt: 输出 VTT 格式")
            print()
            print("当前目录没有找到视频文件")
            return
        
        video_files = [v.resolve() for v in video_files]
        print(f"扫描当前目录，找到 {len(video_files)} 个视频文件")
        print(f"输出格式: {'ASS (思源字体)' if output_format == 'ass' else 'VTT'}")
        print()
    else:
        video_files = []
        for arg in args:
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
    print("模型加载完成!\n")
    print_memory_usage()
    
    print("\n高精度转录模式: beam_size=5, patience=1.0")
    
    print(f"\n处理 {len(video_files)} 个视频")
    print("-" * 60)
    
    total_elapsed = 0
    total_size = 0
    results = []
    
    for i, video in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}]")
        output_path, elapsed, file_size, count = transcribe_video(video, model, batched_model, output_format)
        results.append((video.name, elapsed, file_size, count))
        total_elapsed += elapsed
        total_size += file_size
        print_memory_usage()
    
    print("-" * 60)
    print(f"完成! 共 {len(results)} 个视频")
    print(f"总耗时: {format_elapsed(total_elapsed)}")
    print(f"总大小: {total_size:.1f} MB")
    print()
    print("各视频详情:")
    print(f"{'文件名':<50} {'大小':<10} {'耗时':<15} {'字幕数'}")
    print("-" * 85)
    for name, elapsed, file_size, count in results:
        print(f"{name:<50} {file_size:<10.1f}MB {format_elapsed(elapsed):<15} {count}")
    
    print("\n释放 GPU 内存...")
    cleanup_whisper(model)
    print_memory_usage()

if __name__ == "__main__":
    main()
