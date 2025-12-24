import torch
import torch.nn as nn

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation:
        h' = gamma(z) * h + beta(z)
    """
    def __init__(self, channels: int, emo_dim: int):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(emo_dim, channels),
            nn.Sigmoid()
        )
        self.beta = nn.Linear(emo_dim, channels)
    
    def forward(self, h, z):
        # h: (B,C,H,W), z: (B,emo_dim)
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        return g * h + b
