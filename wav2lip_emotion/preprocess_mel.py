"""
预处理脚本：预计算所有音频的 mel 频谱并保存
这样训练时就不需要每次都重新计算，大幅提升速度
"""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.audio import load_wav, melspectrogram

def preprocess_mel_for_dataset(root_dir, overwrite=False):
    """
    为数据集中的所有音频预计算 mel 频谱
    
    Args:
        root_dir: 数据集根目录
        overwrite: 是否覆盖已存在的 mel 文件
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 数据目录不存在: {root_dir}")
        return
    
    total_audio = 0
    processed_audio = 0
    skipped_audio = 0
    failed_audio = 0
    
    print("=" * 60)
    print("预计算 Mel 频谱")
    print("=" * 60)
    print(f"数据目录: {root_dir}")
    print("=" * 60)
    
    # 遍历所有 speaker
    for speaker_dir in sorted(root_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        print(f"\n处理 speaker: {speaker_dir.name}")
        
        # 遍历所有 utterance
        utterances = sorted([d for d in speaker_dir.iterdir() if d.is_dir()])
        
        for utt_dir in tqdm(utterances, desc=f"  处理 utterances"):
            audio_path = utt_dir / "audio.wav"
            
            if not audio_path.exists():
                continue
            
            total_audio += 1
            
            # mel 保存路径
            mel_path = utt_dir / "mel.npy"
            
            # 如果已存在且不覆盖，跳过
            if mel_path.exists() and not overwrite:
                skipped_audio += 1
                continue
            
            # 计算 mel 频谱
            try:
                wav = load_wav(str(audio_path), sr=16000)
                mel = melspectrogram(wav)  # (n_mels, T)
                
                # 保存
                np.save(mel_path, mel)
                processed_audio += 1
                
            except Exception as e:
                tqdm.write(f"    错误: 处理 {audio_path.name} 时出错: {e}")
                failed_audio += 1
                continue
    
    # 统计
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"总音频数:       {total_audio}")
    print(f"成功处理:       {processed_audio}")
    print(f"跳过 (已存在):  {skipped_audio}")
    print(f"失败:           {failed_audio}")
    print("=" * 60)
    
    if processed_audio > 0:
        print("\n✓ 预计算完成！训练时将自动使用预计算的 mel 频谱。")
    elif skipped_audio > 0:
        print("\n✓ 所有 mel 频谱已存在，无需重新计算。")

def main():
    parser = argparse.ArgumentParser(description="预计算 mel 频谱")
    parser.add_argument("--root_dir", type=str, default="data/MEAD_pairs",
                        help="数据集根目录")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 mel 文件")
    
    args = parser.parse_args()
    
    preprocess_mel_for_dataset(args.root_dir, args.overwrite)

if __name__ == "__main__":
    main()
