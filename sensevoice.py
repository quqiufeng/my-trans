#!/usr/bin/env python3
"""
使用阿里 SenseVoice 模型识别音频并生成 ASS 字幕
轻量版 - 只依赖 torch 和 transformers
"""

import warnings
warnings.filterwarnings('ignore')

import time
import sys
import os
from pathlib import Path

def format_time_ass(seconds):
    """将秒数转换为 ASS 时间格式 (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def create_ass_header():
    """创建 ASS 字幕头部"""
    return """[Script Info]
Title: SenseVoice Subtitles
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

def create_ass_dialogue(start, end, text, style="Default"):
    """创建 ASS 对话行"""
    start_fmt = format_time_ass(start)
    end_fmt = format_time_ass(end)
    text_escaped = text.replace("{", "\\{").replace("}", "\\}")
    return f"Dialogue: 0,{start_fmt},{end_fmt},{style},,0,0,0,,{text_escaped}\n"

def sensevoice_to_ass(audio_path, output_path=None):
    """使用 SenseVoice 识别并生成 ASS 字幕"""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    
    if output_path is None:
        output_path = str(Path(audio_path).with_suffix('.ass'))
    
    # 模型路径
    model_path = os.environ.get("SENSEVOICE_MODEL", "E:\\cuda\\SenseVoiceSmall")
    
    print(f"加载 SenseVoice 模型...")
    print(f"模型路径: {model_path}\n")
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("模型加载完成!\n")
    
    print(f"识别音频: {Path(audio_path).name}")
    print("这可能需要几分钟，请耐心等待...\n")
    
    # 加载音频
    from transformers import AutoProcessor
    import librosa
    
    start_time = time.time()
    
    # 加载音频文件
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # 处理输入
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(device, dtype=torch_dtype)
    
    # 识别
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=1024,
            language="zh",
        )
    
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    elapsed = time.time() - start_time
    
    print(f"\n识别完成! 耗时: {elapsed:.1f}秒\n")
    print(f"识别结果: {result[0][:100]}...\n")
    
    # 创建 ASS 字幕（整段字幕）
    ass_content = create_ass_header()
    dialogue_count = 0
    
    text = result[0] if result else ""
    if text:
        # 创建一个长字幕
        import soundfile as sf
        info = sf.info(audio_path)
        duration = info.duration
        
        ass_content += create_ass_dialogue(0, duration, text)
        dialogue_count = 1
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(ass_content)
    
    print(f"完成! 生成 ASS 字幕: {output_path}")
    print(f"字幕条数: {dialogue_count}")
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用阿里 SenseVoice 识别音频并生成 ASS 字幕')
    parser.add_argument('audio', nargs='?', help='音频文件路径')
    parser.add_argument('-o', '--output', help='输出 ASS 文件路径')
    
    args = parser.parse_args()
    
    if args.audio is None:
        print("用法:")
        print("  python sensevoice.py 音频.mp3")
        print("  python sensevoice.py 音频.mp4")
        print("  python sensevoice.py 音频.mp3 -o 输出.ass")
        print()
        print("安装依赖:")
        print("  pip install torch transformers librosa soundfile")
        print()
        print("下载模型:")
        print("  huggingface-cli download iic/SenseVoiceSmall --local-dir E:\\cuda\\SenseVoiceSmall")
        return
    
    sensevoice_to_ass(args.audio, args.output)

if __name__ == "__main__":
    main()
