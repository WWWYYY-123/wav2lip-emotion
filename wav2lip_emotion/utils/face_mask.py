import torch

def simple_mouth_mask(batch_size, H, W, device):
    """
    返回 (B,1,H,W) 的嘴部mask（1=mouth）。这里用几何近似：
    mouth区域大致在脸下半部中间。
    """
    y = torch.linspace(0, 1, H, device=device).view(1,1,H,1).expand(batch_size,1,H,W)
    x = torch.linspace(0, 1, W, device=device).view(1,1,1,W).expand(batch_size,1,H,W)
    
    # 椭圆中心
    cx, cy = 0.5, 0.72
    rx, ry = 0.22, 0.12
    mask = (((x - cx)/rx)**2 + ((y - cy)/ry)**2) <= 1.0
    return mask.float()
