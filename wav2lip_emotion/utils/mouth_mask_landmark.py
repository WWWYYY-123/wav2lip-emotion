"""
基于人脸关键点的精确嘴部 mask 生成
兼容 dlib 68-point 格式（嘴部关键点 48-67）
"""
import cv2
import numpy as np
import torch

def landmarks_to_mouth_mask(
    landmarks,
    img_h,
    img_w,
    device="cpu",
    dilate_iter=2,
    original_h=None,
    original_w=None
):
    """
    从 68 点关键点生成嘴部 mask
    
    Args:
        landmarks: (68,2) numpy array, dlib/68-point format
        img_h: 目标图像高度
        img_w: 目标图像宽度
        device: torch device
        dilate_iter: 膨胀迭代次数（避免情绪变化触及嘴唇边缘）
        original_h: 原始图像高度（如果关键点是在原始尺寸上提取的）
        original_w: 原始图像宽度（如果关键点是在原始尺寸上提取的）
    
    Returns:
        (1,1,H,W) torch tensor, 1=mouth region
    """
    # 如果提供了原始尺寸，缩放关键点坐标
    if original_h is not None and original_w is not None:
        scale_x = img_w / original_w
        scale_y = img_h / original_h
        landmarks = landmarks.copy()
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # mouth landmarks (outer + inner lips)
    # 外唇轮廓: 48-59
    # 内唇轮廓: 60-67
    mouth_pts = landmarks[48:68].astype(np.int32)
    
    # 确保坐标在有效范围内
    mouth_pts[:, 0] = np.clip(mouth_pts[:, 0], 0, img_w - 1)
    mouth_pts[:, 1] = np.clip(mouth_pts[:, 1], 0, img_h - 1)
    
    # fill mouth polygon
    cv2.fillPoly(mask, [mouth_pts], 255)
    
    # dilate a bit (avoid emotion touching lips)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    
    mask = (mask > 0).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return mask

def batch_mouth_mask_from_landmarks(
    batch_landmarks,
    img_h,
    img_w,
    device="cpu",
    dilate_iter=2
):
    """
    批量生成嘴部 mask
    
    Args:
        batch_landmarks: list of (68,2) numpy arrays
        img_h: 图像高度
        img_w: 图像宽度
        device: torch device
        dilate_iter: 膨胀迭代次数
    
    Returns:
        (B,1,H,W) torch tensor
    """
    masks = [
        landmarks_to_mouth_mask(lm, img_h, img_w, device, dilate_iter)
        for lm in batch_landmarks
    ]
    return torch.cat(masks, dim=0)

def load_landmark_from_file(landmark_path):
    """
    从文件加载关键点
    
    Args:
        landmark_path: .npy 文件路径
    
    Returns:
        (68,2) numpy array
    """
    landmarks = np.load(landmark_path)
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected (68,2) landmarks, got {landmarks.shape}")
    return landmarks

def visualize_mouth_mask(image, landmarks, mask=None, save_path=None):
    """
    可视化嘴部 mask（用于调试）
    
    Args:
        image: (H,W,3) numpy array, RGB
        landmarks: (68,2) numpy array
        mask: (H,W) numpy array, optional
        save_path: 保存路径, optional
    """
    vis = image.copy()
    
    # 绘制嘴部关键点
    mouth_pts = landmarks[48:68].astype(np.int32)
    for i, pt in enumerate(mouth_pts):
        cv2.circle(vis, tuple(pt), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(48+i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 绘制嘴部轮廓
    cv2.polylines(vis, [mouth_pts], True, (0, 255, 255), 1)
    
    # 叠加 mask
    if mask is not None:
        mask_vis = (mask * 255).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return vis
