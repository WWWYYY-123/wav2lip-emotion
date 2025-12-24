import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置使用双 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 添加父目录到 sys.path，以便导入父目录的 models/wav2lip
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from emotion_wav2lip.config import TrainConfig
from emotion_wav2lip.datasets.paired_emotion_dataset import PairedEmotionDataset
from emotion_wav2lip.models.emotion_wav2lip import EmotionWav2Lip
from emotion_wav2lip.utils.face_mask import simple_mouth_mask
from emotion_wav2lip.utils.mouth_mask_landmark import batch_mouth_mask_from_landmarks
from emotion_wav2lip.wav2lip_backbone import build_wav2lip_backbone

# ---------- 可选：情绪分类器（冻结，用于L_emo） ----------
class SimpleEmotionClassifier(nn.Module):
    """
    占位：你后续应替换成预训练表情/情绪分类器（如FER模型）。
    为了让训练先跑通，这里可以先不用 L_emo。
    """
    def __init__(self, num_emotions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_emotions)
    
    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)

def main():
    cfg = TrainConfig()
    
    # 检查可用 GPU
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        print(f"Using device: cuda")
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 如果只检测到 1 个 GPU，提示用户
        if num_gpus == 1:
            print("\n⚠️  Warning: Only 1 GPU detected!")
            print("   To use both GPUs, make sure CUDA_VISIBLE_DEVICES is set correctly")
            print("   Current CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))
    else:
        device = "cpu"
        num_gpus = 0
        print(f"Using device: cpu")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 检查数据目录
    if not os.path.exists(cfg.root_dir):
        print(f"Error: 数据目录 {cfg.root_dir} 不存在！")
        print("请先准备配对情绪数据，参考 README.md")
        return
    
    ds = PairedEmotionDataset(
        root_dir=cfg.root_dir,
        emotions=cfg.emotions,
        img_size=cfg.img_size,
        fps=cfg.fps,
        mel_step_size=cfg.mel_step_size,
        num_frames=cfg.num_frames,
        source_emotion="neutral",
        use_landmarks=cfg.use_landmarks,  # 是否使用关键点
    )
    
    if len(ds) == 0:
        print("Error: 数据集为空！请检查数据目录结构。")
        return
    
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, 
                    num_workers=cfg.num_workers, drop_last=True)
    
    # build backbone
    # 注意：修改为你实际的 checkpoint 路径，如 "checkpoints/wav2lip_gan.pth"
    wav2lip_ckpt = "../checkpoints/wav2lip_gan.pth"
    if not os.path.exists(wav2lip_ckpt):
        print(f"Error: Wav2Lip checkpoint not found: {wav2lip_ckpt}")
        print("请下载 Wav2Lip 预训练权重并放到正确路径")
        return
    
    try:
        wav2lip = build_wav2lip_backbone(pretrained_path=wav2lip_ckpt, device=device)
    except Exception as e:
        print(f"\nError loading Wav2Lip backbone: {e}\n")
        return
    
    model = EmotionWav2Lip(
        wav2lip_backbone=wav2lip,
        num_emotions=len(cfg.emotions),
        emo_dim=cfg.emo_dim,
        use_expr_residual=True
    ).to(device)
    
    if cfg.freeze_wav2lip_backbone:
        model.freeze_backbone()
        print("Wav2Lip backbone frozen.")
    
    # 使用 DataParallel 进行多 GPU 训练
    if num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
        # 注意：使用 DataParallel 后，访问模型方法需要通过 model.module
    
    # emotion classifier (optional)
    emo_cls = SimpleEmotionClassifier(len(cfg.emotions)).to(device)
    for p in emo_cls.parameters():
        p.requires_grad = False
    emo_cls.eval()
    
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()
    
    step = 0
    pbar = tqdm(total=cfg.max_steps)
    it = iter(dl)
    
    print(f"\nMask mode: {'Landmark-based (精确)' if cfg.use_landmarks else 'Ellipse-based (几何近似)'}")
    
    while step < cfg.max_steps:
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(dl)
            batch_data = next(it)
        
        # 解包数据（支持有/无关键点两种格式）
        if cfg.use_landmarks and len(batch_data) == 5:
            frames_in, mel_seg, target_img, emo_id, landmarks = batch_data
            landmarks = landmarks.numpy()  # (B, 68, 2)
        else:
            frames_in, mel_seg, target_img, emo_id = batch_data[:4]
            landmarks = None
        
        frames_in = frames_in.to(device)
        mel_seg = mel_seg.to(device)
        target_img = target_img.to(device)
        emo_id = emo_id.to(device)
        
        B, _, H, W = target_img.shape
        
        # 生成嘴部 mask（关键点 or 椭圆）
        if cfg.use_landmarks and landmarks is not None:
            mouth_mask = batch_mouth_mask_from_landmarks(
                [landmarks[i] for i in range(B)], H, W, device=device
            )
        else:
            mouth_mask = simple_mouth_mask(B, H, W, device)
        
        out, base = model(frames_in, mel_seg, emo_id, mouth_mask, strength=None)
        
        # losses
        L_mouth = l1(mouth_mask * out, mouth_mask * target_img)
        L_nonmouth = l1((1.0 - mouth_mask) * out, (1.0 - mouth_mask) * target_img)
        
        # optional emotion loss (placeholder classifier)
        logits = emo_cls(out.detach())  # 先detach跑通；你换成真预训练分类器后可不detach
        L_emo = ce(logits, emo_id)
        
        # identity / temporal：这里先给你占位（跑通后你再加ArcFace/时间loss）
        L_id = torch.tensor(0.0, device=device)
        L_temp = torch.tensor(0.0, device=device)
        
        loss = (
            cfg.lambda_mouth * L_mouth +
            cfg.lambda_nonmouth * L_nonmouth +
            cfg.lambda_emo * L_emo +
            cfg.lambda_id * L_id +
            cfg.lambda_temp * L_temp
        )
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # 实时显示 loss
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mouth': f'{L_mouth.item():.4f}',
            'nonmouth': f'{L_nonmouth.item():.4f}'
        })
        
        if step % cfg.log_every == 0:
            tqdm.write(
                f"step {step} | loss {loss.item():.4f} "
                f"| mouth {L_mouth.item():.4f} nonmouth {L_nonmouth.item():.4f} emo {L_emo.item():.4f}"
            )
        
        if step % cfg.save_every == 0 and step > 0:
            ckpt_path = f"checkpoints/ewav2lip_step{step}.pth"
            # 保存时需要处理 DataParallel
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model": model_state}, ckpt_path)
            tqdm.write(f"Saved: {ckpt_path}")
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
