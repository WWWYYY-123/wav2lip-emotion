"""
预处理脚本：为数据集中的每一帧提取并保存人脸关键点
使用父目录的 face_detection 模块（S3FD + landmark）
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加父目录到路径以导入 face_detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_landmarks_for_dataset(root_dir, emotions, overwrite=False):
    """
    为数据集中的所有帧提取关键点
    
    Args:
        root_dir: 数据集根目录
        emotions: 情绪列表
        overwrite: 是否覆盖已存在的关键点文件
    """
    # 导入 face_alignment
    try:
        import torch
        import face_alignment
        print("✓ 成功导入 face_alignment 模块")
    except ImportError as e:
        print("✗ 无法导入 face_alignment 模块")
        print("请先安装: pip install face-alignment")
        print(f"错误: {e}")
        return
    
    # 初始化检测器
    print("初始化关键点检测器 (face-alignment)...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device
        )
        print("✓ 检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 数据目录不存在: {root_dir}")
        return
    
    total_frames = 0
    processed_frames = 0
    skipped_frames = 0
    failed_frames = 0
    
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
                
                # 获取所有图片
                image_files = sorted([
                    f for f in emotion_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
                ])
                
                if not image_files:
                    continue
                
                print(f"  {utt_dir.name}/{emotion}: {len(image_files)} 帧")
                
                for img_path in tqdm(image_files, desc=f"    提取关键点", leave=False):
                    total_frames += 1
                    
                    # 关键点保存路径
                    landmark_path = img_path.with_name(f"{img_path.stem}_landmark.npy")
                    
                    # 如果已存在且不覆盖，跳过
                    if landmark_path.exists() and not overwrite:
                        skipped_frames += 1
                        continue
                    
                    # 读取图片
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"    警告: 无法读取图片 {img_path}")
                        failed_frames += 1
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 检测关键点
                    try:
                        preds = fa.get_landmarks(img_rgb)
                        
                        if preds is None or len(preds) == 0:
                            print(f"    警告: 未检测到人脸 {img_path.name}")
                            failed_frames += 1
                            continue
                        
                        # 取第一个人脸的关键点
                        landmarks = preds[0].astype(np.float32)  # (68, 2)
                        
                        if landmarks.shape != (68, 2):
                            print(f"    警告: 关键点格式错误 {landmarks.shape} at {img_path.name}")
                            failed_frames += 1
                            continue
                        
                        # 保存关键点
                        np.save(landmark_path, landmarks)
                        processed_frames += 1
                        
                    except Exception as e:
                        print(f"    错误: 处理 {img_path.name} 时出错: {e}")
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

def visualize_landmarks(root_dir, emotions, num_samples=5):
    """
    可视化一些样本的关键点（用于验证）
    
    Args:
        root_dir: 数据集根目录
        emotions: 情绪列表
        num_samples: 可视化样本数
    """
    from utils.mouth_mask_landmark import visualize_mouth_mask
    
    root_path = Path(root_dir)
    output_dir = Path("landmark_visualization")
    output_dir.mkdir(exist_ok=True)
    
    samples_found = 0
    
    for speaker_dir in root_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        
        for utt_dir in speaker_dir.iterdir():
            if not utt_dir.is_dir():
                continue
            
            for emotion in emotions:
                emotion_dir = utt_dir / emotion
                if not emotion_dir.exists():
                    continue
                
                # 随机选一帧
                image_files = [
                    f for f in emotion_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
                ]
                
                if not image_files:
                    continue
                
                img_path = image_files[len(image_files) // 2]  # 取中间帧
                landmark_path = img_path.with_name(f"{img_path.stem}_landmark.npy")
                
                if not landmark_path.exists():
                    continue
                
                # 读取图片和关键点
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                landmarks = np.load(landmark_path)
                
                # 生成 mask
                from utils.mouth_mask_landmark import landmarks_to_mouth_mask
                mask = landmarks_to_mouth_mask(landmarks, img.shape[0], img.shape[1], device="cpu")
                mask_np = mask.squeeze().cpu().numpy()
                
                # 可视化
                vis = visualize_mouth_mask(img_rgb, landmarks, mask_np)
                
                # 保存
                save_path = output_dir / f"{speaker_dir.name}_{utt_dir.name}_{emotion}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                samples_found += 1
                print(f"可视化: {save_path}")
                
                if samples_found >= num_samples:
                    print(f"\n可视化完成！结果保存在: {output_dir}")
                    return
    
    print(f"\n可视化完成！共 {samples_found} 个样本，结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="为 Emotion-Wav2Lip 数据集提取人脸关键点")
    parser.add_argument("--root_dir", default="data/MEAD_pairs", help="数据集根目录")
    parser.add_argument("--emotions", default="neutral,happy,sad,angry", help="情绪列表，逗号分隔")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的关键点文件")
    parser.add_argument("--visualize", action="store_true", help="可视化一些样本")
    parser.add_argument("--num_vis", type=int, default=5, help="可视化样本数")
    
    args = parser.parse_args()
    
    emotions = args.emotions.split(",")
    
    print("=" * 60)
    print("Emotion-Wav2Lip 关键点预处理")
    print("=" * 60)
    print(f"数据目录: {args.root_dir}")
    print(f"情绪列表: {emotions}")
    print(f"覆盖模式: {args.overwrite}")
    print("=" * 60)
    
    if args.visualize:
        print("\n可视化模式")
        visualize_landmarks(args.root_dir, emotions, args.num_vis)
    else:
        print("\n开始提取关键点...")
        extract_landmarks_for_dataset(args.root_dir, emotions, args.overwrite)
        print("\n提示: 使用 --visualize 参数可以查看关键点提取效果")

if __name__ == "__main__":
    # 示例用法：
    # 1. 提取关键点：
    #    python preprocess_landmarks.py --root_dir data/MEAD_pairs
    #
    # 2. 覆盖已存在的关键点：
    #    python preprocess_landmarks.py --root_dir data/MEAD_pairs --overwrite
    #
    # 3. 可视化验证：
    #    python preprocess_landmarks.py --root_dir data/MEAD_pairs --visualize --num_vis 10
    
    main()
