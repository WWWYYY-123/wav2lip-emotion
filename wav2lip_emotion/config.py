from dataclasses import dataclass

@dataclass
class TrainConfig:
    # data
    root_dir: str = "data/MEAD_pairs"  # 你自己的配对数据目录
    img_size: int = 96                   # 与Wav2Lip一致
    fps: int = 25
    mel_step_size: int = 16              # 与Wav2Lip常用一致
    num_frames: int = 5                  # Wav2Lip输入帧窗口（可调）
    emotions: tuple = ("neutral", "happy")  # 实际训练的情绪
    use_landmarks: bool = True           # 是否使用关键点生成精确嘴部mask（需要预处理）
    
    # training
    batch_size: int = 32                 # 减小到 32，更适合小数据集
    num_workers: int = 8                 # 增加到 8，加速数据加载
    lr: float = 1e-4                     # 减小 batch 后降低学习率
    max_steps: int = 50000               # 50k 步
    log_every: int = 20                  # 更频繁地打印
    save_every: int = 200                # 每 200 步保存一次（约 3 分钟）
    
    # loss weights
    lambda_mouth: float = 1.0
    lambda_nonmouth: float = 1.0
    lambda_emo: float = 0.0              # 先关闭，等接入真实 FER 分类器再启用
    lambda_id: float = 0.1
    lambda_temp: float = 0.2
    # sync loss（如果你集成Wav2Lip的SyncNet，这里可启用）
    use_sync_loss: bool = False
    lambda_sync: float = 0.03
    
    # model
    emo_dim: int = 64
    film_layers: tuple = ("dec3", "dec4", "dec5")  # 在decoder后半段注入
    freeze_wav2lip_backbone: bool = False          # 改为 False，让整个模型一起训练
