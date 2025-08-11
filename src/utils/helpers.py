"""
工具模块
"""
import os
import numpy as np
from datetime import datetime

def ensure_directory_exists(path: str) -> bool:
    """确保目录存在"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False

def generate_output_filename(base_name: str, extension: str, prefix: str = "") -> str:
    """生成输出文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        return f"{prefix}_{base_name}_{timestamp}.{extension}"
    else:
        return f"{base_name}_{timestamp}.{extension}"

def get_image_statistics(image_data: np.ndarray) -> dict:
    """获取图像统计信息"""
    return {
        'min': float(image_data.min()),
        'max': float(image_data.max()),
        'mean': float(image_data.mean()),
        'std': float(image_data.std()),
        'shape': image_data.shape,
        'dtype': str(image_data.dtype)
    }

def normalize_image(image_data: np.ndarray, target_range: tuple = (0, 255)) -> np.ndarray:
    """归一化图像到指定范围"""
    min_val, max_val = target_range
    normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    return (normalized * (max_val - min_val) + min_val).astype(np.uint8)

def is_valid_dicom_file(file_path: str) -> bool:
    """检查是否为有效的DICOM文件"""
    try:
        import pydicom
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False