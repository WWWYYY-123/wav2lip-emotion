import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(x + self.net(x))

class ExprResidualNet(nn.Module):
    """
    输入：生成的base帧（或源帧） + 情绪向量
    输出：表情残差 R (B,3,H,W)
    """
    def __init__(self, emo_dim=64, base_c=64):
        super().__init__()
        self.emo_proj = nn.Linear(emo_dim, base_c)
        
        self.enc = nn.Sequential(
            nn.Conv2d(3, base_c, 7, 1, 3),
            nn.ReLU(inplace=True),
            ResBlock(base_c),
            ResBlock(base_c),
        )
        self.dec = nn.Sequential(
            ResBlock(base_c),
            nn.Conv2d(base_c, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, img, z):
        # img: (B,3,H,W)
        h = self.enc(img)
        e = self.emo_proj(z).unsqueeze(-1).unsqueeze(-1)
        h = h + e
        r = self.dec(h)  # [-1,1]
        return r
