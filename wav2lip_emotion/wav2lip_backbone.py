"""
通用 Wav2Lip Backbone 加载器
支持多种 checkpoint 格式和 DataParallel/DDP 模型
"""
import torch
import sys
import os

# 添加父目录到路径，以便导入 models/wav2lip.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def _extract_state_dict(ckpt):
    """兼容多种保存格式"""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "generator", "g", "netG"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    # 可能ckpt本身就是state_dict
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")

def _strip_module_prefix(state_dict):
    """兼容 DataParallel / DDP 的 'module.' 前缀"""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def build_wav2lip_backbone(pretrained_path: str, device: str = "cpu"):
    """
    适配你这个仓库结构：models/wav2lip.py 里一般有 Wav2Lip 类
    
    Args:
        pretrained_path: Wav2Lip 预训练权重路径
        device: 目标设备 ("cpu" 或 "cuda")
    
    Returns:
        加载好权重的 Wav2Lip 模型（eval 模式）
    """
    try:
        # 尝试多种导入方式
        try:
            from models.wav2lip import Wav2Lip
        except ImportError:
            # 如果直接导入失败，尝试使用 importlib
            import importlib.util
            models_path = os.path.join(parent_dir, "models", "wav2lip.py")
            spec = importlib.util.spec_from_file_location("models.wav2lip", models_path)
            wav2lip_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wav2lip_module)
            Wav2Lip = wav2lip_module.Wav2Lip
            print(f"[Wav2Lip] Loaded from: {models_path}")
    except Exception as e:
        print("=" * 60)
        print("Error: 无法导入 Wav2Lip 模型！")
        print("请确保：")
        print("1. 父目录有 models/wav2lip.py 文件")
        print("2. 或者修改 wav2lip_backbone.py 中的导入路径")
        print("=" * 60)
        raise e
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")
    
    model = Wav2Lip()
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_module_prefix(state_dict)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Wav2Lip] Loaded ckpt: {pretrained_path}")
    print(f"[Wav2Lip] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(missing) < 20 and len(unexpected) < 20:
        # 打印少量即可，太多会刷屏
        if missing: 
            print("  missing:", missing[:10])
        if unexpected: 
            print("  unexpected:", unexpected[:10])
    
    model = model.to(device).eval()
    return model
