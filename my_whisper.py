#!/usr/bin/env python3
"""
使用 OpenAI Whisper 识别音频并生成 ASS 字幕
中文支持最好
"""

import warnings
warnings.filterwarnings('ignore')

import time
import torch
from pathlib import Path
import whisper

def format_time_ass(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def create_ass_header():
    return """[Script Info]
Title: Whisper Subtitles
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
    start_fmt = format_time_ass(start)
    end_fmt = format_time_ass(end)
    text_escaped = text.replace("{", "\\{").replace("}", "\\}")
    return f"Dialogue: 0,{start_fmt},{end_fmt},{style},,0,0,0,,{text_escaped}\n"

def whisper_to_ass(audio_path, output_path=None):
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.with_suffix('.ass')

    print(f"加载 Whisper 模型...")
    model = whisper.load_model("large")
    print("模型加载完成!\n")

    print(f"识别音频: {audio_path.name}")
    print("这可能需要几分钟...\n")

    start_time = time.time()

    result = model.transcribe(
        str(audio_path),
        language="Chinese",
        word_timestamps=True
    )

    elapsed = time.time() - start_time
    print(f"\n识别完成! 耗时: {elapsed:.1f}秒\n")

    ass_content = create_ass_header()
    dialogue_count = 0

    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        if text:
            ass_content += create_ass_dialogue(start, end, text)
            dialogue_count += 1

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(ass_content)

    print(f"生成 ASS 字幕: {output_path}")
    print(f"字幕条数: {dialogue_count}")

    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='使用 OpenAI Whisper 识别音频并生成 ASS 字幕')
    parser.add_argument('audio', nargs='?', help='音频文件路径')
    parser.add_argument('-o', '--output', help='输出 ASS 文件路径')
    args = parser.parse_args()

    if args.audio is None:
        print("用法:")
        print("  python whisper.py 音频.mp3")
        print("  python whisper.py 视频.mp4 -o 输出.ass")
        print()
        print("安装依赖:")
        print("  pip install openai-whisper ffmpeg-python")
        return

    whisper_to_ass(args.audio, args.output)

if __name__ == "__main__":
    main()
