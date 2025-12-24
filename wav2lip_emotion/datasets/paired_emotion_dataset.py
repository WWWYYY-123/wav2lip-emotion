import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.audio import load_wav, melspectrogram

def list_frames(folder):
    frames = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    frames.sort()
    return [os.path.join(folder, f) for f in frames]

class PairedEmotionDataset(Dataset):
    def __init__(self, root_dir, emotions, img_size=96, fps=25, mel_step_size=16, num_frames=5,
                 source_emotion="neutral", use_landmarks=False):
        self.root = root_dir
        self.emotions = list(emotions)
        self.img_size = img_size
        self.fps = fps
        self.mel_step_size = mel_step_size
        self.num_frames = num_frames
        self.source_emotion = source_emotion
        self.use_landmarks = use_landmarks  # 是否加载关键点
        
        self.items = []
        # 收集 utt 目录
        if not os.path.exists(self.root):
            print(f"Warning: root_dir {self.root} does not exist!")
            return
            
        for spk in sorted(os.listdir(self.root)):
            spk_dir = os.path.join(self.root, spk)
            if not os.path.isdir(spk_dir):
                continue
            for utt in sorted(os.listdir(spk_dir)):
                utt_dir = os.path.join(spk_dir, utt)
                if not os.path.isdir(utt_dir):
                    continue
                wav = os.path.join(utt_dir, "audio.wav")
                if not os.path.isfile(wav):
                    continue
                # 要求至少有source情绪帧
                src_dir = os.path.join(utt_dir, self.source_emotion)
                if not os.path.isdir(src_dir):
                    continue
                self.items.append((utt_dir, wav))
        
        self.emo2id = {e:i for i,e in enumerate(self.emotions)}
        print(f"Loaded {len(self.items)} utterances from {self.root}")
    
    def __len__(self):
        return len(self.items)
    
    def _read_img(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img / 255.0).astype(np.float32)
        img = torch.from_numpy(img).permute(2,0,1)  # (3,H,W)
        return img
    
    def __getitem__(self, idx):
        utt_dir, wav_path = self.items[idx]
        
        # 选target emotion（不等于source）
        target_candidates = [e for e in self.emotions if e != self.source_emotion]
        target_emotion = random.choice(target_candidates)
        
        src_dir = os.path.join(utt_dir, self.source_emotion)
        tgt_dir = os.path.join(utt_dir, target_emotion)
        if not os.path.isdir(tgt_dir):
            # 如果缺失，退回别的emotion
            existing = [e for e in target_candidates if os.path.isdir(os.path.join(utt_dir, e))]
            target_emotion = random.choice(existing) if existing else self.source_emotion
            tgt_dir = os.path.join(utt_dir, target_emotion)
        
        src_frames = list_frames(src_dir)
        tgt_frames = list_frames(tgt_dir)
        n = min(len(src_frames), len(tgt_frames))
        assert n > 5, f"Not enough frames in {utt_dir}"
        
        # 随机选择中心帧 t
        t = random.randint(2, n - 3)
        
        # Wav2Lip 输入格式：单帧分成上下半部分 (6 通道)
        # 读取中心帧
        center_img = self._read_img(src_frames[t])  # (3, H, W)
        H, W = center_img.shape[1], center_img.shape[2]
        
        # 分成上下半部分
        upper_half = center_img[:, :H//2, :]  # (3, H/2, W)
        lower_half = center_img[:, H//2:, :]  # (3, H/2, W)
        
        # 将下半部分 resize 到与上半部分相同大小
        lower_half_resized = torch.nn.functional.interpolate(
            lower_half.unsqueeze(0), 
            size=(H//2, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 拼接成 6 通道 (3+3, H/2, W) 然后 resize 回原尺寸
        frames_in_half = torch.cat([upper_half, lower_half_resized], dim=0)  # (6, H/2, W)
        frames_in = torch.nn.functional.interpolate(
            frames_in_half.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (6, H, W)
        
        # target单帧
        target_img = self._read_img(tgt_frames[t])
        
        # mel窗口（按Wav2Lip常用：基于fps定位）
        # 优先使用预计算的 mel，如果不存在则实时计算
        mel_cache_path = os.path.join(utt_dir, "mel.npy")
        if os.path.exists(mel_cache_path):
            mel = np.load(mel_cache_path)  # (n_mels, T)
        else:
            wav = load_wav(wav_path, sr=16000)
            mel = melspectrogram(wav)  # (n_mels, T)
        
        mel_idx = int((t / self.fps) * 80)  # 经验：16000Hz时mel hop约200 samples => 80fps
        start = mel_idx
        end = start + self.mel_step_size
        if end >= mel.shape[1]:
            end = mel.shape[1]
            start = max(0, end - self.mel_step_size)
        mel_seg = mel[:, start:end]
        if mel_seg.shape[1] < self.mel_step_size:
            pad = self.mel_step_size - mel_seg.shape[1]
            mel_seg = np.pad(mel_seg, ((0,0),(0,pad)), mode="edge")
        mel_seg = torch.from_numpy(mel_seg).unsqueeze(0)  # (1, n_mels, step) - batch 后变成 (B, 1, n_mels, step)
        
        emo_id = torch.tensor(self.emo2id[target_emotion], dtype=torch.long)
        
        # 加载关键点或预计算的 mask（如果启用）
        landmark = None
        if self.use_landmarks:
            target_frame_name = os.path.splitext(os.path.basename(tgt_frames[t]))[0]
            
            # 优先使用预计算的 mask
            mask_path = os.path.join(tgt_dir, f"{target_frame_name}_mask.npy")
            if os.path.exists(mask_path):
                try:
                    # 直接加载预计算的 mask，跳过 landmark
                    # 返回 None 作为 landmark，训练时会直接使用预加载的 mask
                    landmark = "precomputed"  # 标记为预计算
                except Exception as e:
                    print(f"Warning: Failed to load mask from {mask_path}: {e}")
                    landmark = None
            else:
                # 如果没有预计算 mask，加载 landmark
                landmark_path = os.path.join(tgt_dir, f"{target_frame_name}_landmark.npy")
                if os.path.exists(landmark_path):
                    try:
                        landmark = np.load(landmark_path)  # (68,2)
                        if landmark.shape != (68, 2):
                            print(f"Warning: Invalid landmark shape {landmark.shape} at {landmark_path}")
                            landmark = None
                        else:
                            # 读取原始图像尺寸并缩放关键点到目标尺寸
                            orig_img = cv2.imread(tgt_frames[t])
                            if orig_img is not None:
                                orig_h, orig_w = orig_img.shape[:2]
                                scale_x = self.img_size / orig_w
                                scale_y = self.img_size / orig_h
                                landmark = landmark.copy()
                                landmark[:, 0] *= scale_x
                                landmark[:, 1] *= scale_y
                    except Exception as e:
                        print(f"Warning: Failed to load landmark from {landmark_path}: {e}")
                        landmark = None
        
        if self.use_landmarks and landmark is not None:
            return frames_in, mel_seg, target_img, emo_id, landmark
        else:
            return frames_in, mel_seg, target_img, emo_id
