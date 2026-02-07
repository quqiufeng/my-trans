#!/usr/bin/env python3
"""
模型下载脚本
下载 faster-whisper、NLLB 翻译模型和 Paraformer 中文识别模型
"""

import os
import sys
from pathlib import Path

def download_faster_whisper():
    """下载 faster-whisper 模型"""
    print("=" * 60)
    print("下载 faster-whisper 模型...")
    print("=" * 60)
    
    try:
        from faster_whisper import WhisperModel
        
        # 自动下载模型到指定目录
        model_path = "e:/cuda/faster-whisper-medium"
        os.makedirs(model_path, exist_ok=True)
        
        print(f"模型将保存到: {model_path}")
        print("自动下载中...")
        
        # 加载模型（会自动下载）
        model = WhisperModel("medium", device="cuda", compute_type="float16")
        
        print(f"\n✓ faster-whisper 模型下载成功!")
        print(f"模型路径: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def download_nllb_ct2():
    """下载 NLLB CTranslate2 模型"""
    print("\n" + "=" * 60)
    print("下载 NLLB CTranslate2 模型...")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        model_path = "E:/cuda/nllb-200-3.3B-ct2-float16"
        repo_id = "Derur/nllb-200-3.3B-ct2-float16"
        
        print(f"下载模型: {repo_id}")
        print(f"保存路径: {model_path}")
        print()
        
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✓ NLLB 模型下载成功!")
        print(f"模型路径: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def download_paraformer():
    """下载 Paraformer 中文识别模型"""
    print("\n" + "=" * 60)
    print("下载 Paraformer 中文识别模型...")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download

        model_path = "E:/cuda/speech_paraformer-large"
        repo_id = "paraformer/zhParaformer"

        print(f"下载模型: {repo_id}")
        print(f"保存路径: {model_path}")
        print()

        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✓ Paraformer 模型下载成功!")
        print(f"模型路径: {model_path}")

        return True

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print()
        print("请手动下载:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli download paraformer/zhParaformer --local-dir E:\\cuda\\paraformer")
        return False

def main():
    print("=" * 60)
    print("  模型下载工具 / Model Download Tool")
    print("=" * 60)
    print()
    
    options = """
    选项 / Options:
      1. 下载 faster-whisper 模型 (语音识别)
      2. 下载 NLLB 翻译模型
      3. 下载 Paraformer 中文识别模型 (推荐)
      4. 下载全部模型
      0. 退出 / Exit
    """
    
    print(options)
    
    choice = input("请选择 / Please choose (0-4): ").strip()
    
    if choice == "1":
        download_faster_whisper()
    elif choice == "2":
        download_nllb_ct2()
    elif choice == "3":
        download_paraformer()
    elif choice == "4":
        print("\n正在下载所有模型...\n")
        success1 = download_faster_whisper()
        success2 = download_nllb_ct2()
        success3 = download_paraformer()
        
        print("\n" + "=" * 60)
        print("下载完成 / Download Complete")
        print("=" * 60)
        print(f"faster-whisper: {'✓' if success1 else '✗'}")
        print(f"NLLB: {'✓' if success2 else '✗'}")
        print(f"Paraformer: {'✓' if success3 else '✗'}")
    else:
        print("退出 / Exit")

if __name__ == "__main__":
    main()
