"""
预处理脚本：预计算所有帧的 mouth mask 并保存
这样训练时就不需要实时生成，大幅提升速度
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.mouth_mask_landmark import landmarks_to_mouth_mask

def preprocess_masks_for_dataset(root_dir, emotions, img_size=96, overwrite=False):
    """
    为数据集中的所有帧预计算 mouth mask
    
    Args:
        root_dir: 数据集根目录
        emotions: 情绪列表
        img_size: 图像尺寸
        overwrite: 是否覆盖已存在的 mask 文件
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 数据目录不存在: {root_dir}")
        return
    
    total_frames = 0
    processed_frames = 0
    skipped_frames = 0
    failed_frames = 0
    
    print("=" * 60)
    print("预计算 Mouth Masks")
    print("=" * 60)
    print(f"数据目录: {root_dir}")
    print(f"图像尺寸: {img_size}")
    print(f"情绪列表: {emotions}")
    print("=" * 60)
    
    # 遍历所有 speaker
    for speaker_dir in sorted(root_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        print(f"\n处理 speaker: {speaker_dir.name}")
        
        # 遍历所有 utterance
        for utt_dir in sorted(speaker_dir.iterdir()):
            if not utt_dir.is_dir():
                continue
            
            # 遍历所有情绪
            for emotion in emotions:
                emotion_dir = utt_dir / emotion
                if not emotion_dir.exists():
                    continue
                
                # 获取所有 landmark 文件
                landmark_files = sorted(emotion_dir.glob("*_landmark.npy"))
                
                if not landmark_files:
                    continue
                
                for landmark_path in tqdm(landmark_files, desc=f"  {utt_dir.name}/{emotion}", leave=False):
                    total_frames += 1
                    
                    # mask 保存路径
                    mask_path = landmark_path.with_name(f"{landmark_path.stem.replace('_landmark', '')}_mask.npy")
                    
                    # 如果已存在且不覆盖，跳过
                    if mask_path.exists() and not overwrite:
                        skipped_frames += 1
                        continue
                    
                    try:
                        # 加载 landmark
                        landmark = np.load(landmark_path)
                        
                        if landmark.shape != (68, 2):
                            failed_frames += 1
                            continue
                        
                        # 生成 mask
                        mask = landmarks_to_mouth_mask(
                            landmark, 
                            img_size, 
                            img_size, 
                            device="cpu"
                        )  # (1, 1, H, W)
                        
                        # 保存为 numpy 数组
                        mask_np = mask.squeeze().numpy()  # (H, W)
                        np.save(mask_path, mask_np)
                        processed_frames += 1
                        
                    except Exception as e:
                        tqdm.write(f"    错误: 处理 {landmark_path.name} 时出错: {e}")
                        failed_frames += 1
                        continue
    
    # 统计
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"总帧数:       {total_frames}")
    print(f"成功处理:     {processed_frames}")
    print(f"跳过 (已存在): {skipped_frames}")
    print(f"失败:         {failed_frames}")
    print("=" * 60)
    
    if processed_frames > 0:
        print("\n✓ 预计算完成！训练时将自动使用预计算的 mask。")
    elif skipped_frames > 0:
        print("\n✓ 所有 mask 已存在，无需重新计算。")

def main():
    parser = argparse.ArgumentParser(description="预计算 mouth masks")
    parser.add_argument("--root_dir", type=str, default="data/MEAD_pairs",
                        help="数据集根目录")
    parser.add_argument("--emotions", nargs="+", 
                        default=["neutral", "happy", "sad", "angry"],
                        help="情绪列表")
    parser.add_argument("--img_size", type=int, default=96,
                        help="图像尺寸")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 mask 文件")
    
    args = parser.parse_args()
    
    preprocess_masks_for_dataset(
        args.root_dir, 
        args.emotions, 
        args.img_size,
        args.overwrite
    )

if __name__ == "__main__":
    main()
