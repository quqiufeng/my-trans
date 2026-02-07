#!/usr/bin/env python3
"""
使用阿里达摩院 SenseVoice 模型识别音频并生成 ASS 字幕
专门优化中文识别效果
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
    from modelscope.pipelines import pipeline
    
    print(f"加载 SenseVoice 模型...")
    
    # 默认本地模型路径（Windows）
    model_path = "E:\\cuda\\SenseVoiceSmall"
    
    if os.path.exists(model_path):
        model_id = model_path
        print(f"使用本地模型: {model_path}")
    else:
        model_id = "iic/SenseVoiceSmall"
        print("下载模型中...")
    
    recognition = pipeline(
        model_id,
        model_revision="master"
    )
    
    print("模型加载完成!\n")
    
    if output_path is None:
        output_path = str(Path(audio_path).with_suffix('.ass'))
    
    print(f"识别音频: {Path(audio_path).name}")
    print("这可能需要几分钟，请耐心等待...\n")
    
    start_time = time.time()
    
    # 识别音频
    result = recognition(
        audio_path,
        return_raw_text=False,
        language="zh"
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n识别完成! 耗时: {elapsed:.1f}秒\n")
    
    # 解析结果
    text = result.get("text", "")
    print(f"识别结果: {text[:100]}...\n")
    
    # 创建 ASS 字幕
    ass_content = create_ass_header()
    dialogue_count = 0
    
    # 解析时间戳
    raw_result = result.get("raw_result", [])
    if raw_result:
        for item in raw_result:
            if isinstance(item, dict) and "timestamp" in item:
                for ts in item["timestamp"]:
                    if len(ts) >= 3:
                        start_ms = ts[0]
                        end_ms = ts[1]
                        sentence = ts[2]
                        
                        start_sec = start_ms / 1000.0
                        end_sec = end_ms / 1000.0
                        
                        ass_content += create_ass_dialogue(start_sec, end_sec, sentence)
                        dialogue_count += 1
    
    # 如果没有时间戳，创建一个整段的字幕
    if dialogue_count == 0 and text:
        ass_content += create_ass_dialogue(0, 60, text)
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
    parser.add_argument('-l', '--lang', default='zh', help='语言 (默认: zh)')
    
    args = parser.parse_args()
    
    if args.audio is None:
        print("用法:")
        print("  python sensevoice.py 音频.mp3")
        print("  python sensevoice.py 音频.mp3 -o 输出.ass")
        print()
        print("支持格式: mp3, wav, m4a, flac, ogg, avi, mp4 等")
        print()
        print("安装依赖:")
        print("  pip install modelscope torch")
        print()
        print("下载模型:")
        print("  首次运行会自动从 ModelScope 下载模型")
        print()
        print("或者设置环境变量使用本地模型:")
        print("  set SENSEVOICE_MODEL=本地模型路径")
        return
    
    os.environ["SENSEVOICE_LANGUAGE"] = args.lang
    
    sensevoice_to_ass(args.audio, args.output)

if __name__ == "__main__":
    main()
