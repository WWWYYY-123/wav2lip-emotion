import torch
import torch.nn as nn
from .film import FiLM
from .expr_resnet import ExprResidualNet

class EmotionEncoder(nn.Module):
    def __init__(self, num_emotions: int, emo_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_emotions, emo_dim)
    
    def forward(self, emo_id: torch.LongTensor, strength: torch.FloatTensor=None):
        """
        emo_id: (B,)
        strength: (B,) optional in [0,1], scales emotion intensity
        """
        z = self.emb(emo_id)
        if strength is not None:
            z = z * strength.unsqueeze(-1)
        return z

class EmotionWav2Lip(nn.Module):
    """
    输出策略：
    1) base = Wav2Lip(input_frames, mel)
    2) r = ExprResidualNet(base, z)
    3) out = base + (1 - mouth_mask) * r   (不改嘴)
    """
    def __init__(self, wav2lip_backbone: nn.Module,
                 num_emotions: int,
                 emo_dim: int = 64,
                 use_expr_residual: bool = True):
        super().__init__()
        self.wav2lip = wav2lip_backbone
        self.emo_enc = EmotionEncoder(num_emotions, emo_dim)
        self.use_expr_residual = use_expr_residual
        self.expr = ExprResidualNet(emo_dim=emo_dim) if use_expr_residual else None
        
        # 你如果想在 backbone 中注入FiLM，需要你在 Wav2Lip backbone 里暴露出对应层的feature。
        # 为了让你先跑通，我这里默认：不改 backbone 内部结构，只用残差分支实现情绪。
        # 如果你要做论文更强版本，我后面给你"如何注入FiLM到Wav2Lip解码器"的补丁思路。
    
    def freeze_backbone(self):
        for p in self.wav2lip.parameters():
            p.requires_grad = False
    
    def unfreeze_backbone(self):
        for p in self.wav2lip.parameters():
            p.requires_grad = True
    
    def forward(self, frames, mels, emo_id, mouth_mask, strength=None):
        """
        frames: (B, 3*num_frames, H, W)  # 与Wav2Lip一致：多帧stack成通道
        mels:   (B, 1, n_mels, mel_step_size)
        emo_id: (B,)
        mouth_mask: (B,1,H,W)  1=mouth region
        """
        z = self.emo_enc(emo_id, strength=strength)
        # 注意：Wav2Lip 的参数顺序是 (audio, face)
        base = self.wav2lip(mels, frames)  # (B,3,H,W)
        
        if self.use_expr_residual:
            r = self.expr(base, z)  # (B,3,H,W) in [-1,1]
            out = base + (1.0 - mouth_mask) * r
        else:
            out = base
        return out, base
