"""
频域增强处理器模块
"""
import numpy as np
from typing import Tuple, Optional

class FrequencyProcessor:
    """频域增强算法集合"""
    
    @staticmethod
    def _create_frequency_filter(shape: Tuple[int, int], cutoff_ratio: float, 
                                filter_type: str) -> np.ndarray:
        """创建频域滤波器
        
        Args:
            shape: 图像形状 (rows, cols)
            cutoff_ratio: 截止频率比例 (0.01-0.5)
            filter_type: 滤波器类型 ('ideal_low', 'ideal_high', 'gaussian_low', 'gaussian_high')
            
        Returns:
            np.ndarray: 频域滤波器
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建距离矩阵
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # 计算截止频率
        max_distance = np.sqrt(crow**2 + ccol**2)
        cutoff_frequency = cutoff_ratio * max_distance
        
        if filter_type == 'ideal_low':
            # 理想低通滤波器
            filter_mask = (distance <= cutoff_frequency).astype(np.float32)
            
        elif filter_type == 'ideal_high':
            # 理想高通滤波器
            filter_mask = (distance > cutoff_frequency).astype(np.float32)
            
        elif filter_type == 'gaussian_low':
            # 高斯低通滤波器
            filter_mask = np.exp(-(distance**2) / (2 * cutoff_frequency**2))
            
        elif filter_type == 'gaussian_high':
            # 高斯高通滤波器
            filter_mask = 1 - np.exp(-(distance**2) / (2 * cutoff_frequency**2))
            
        else:
            raise ValueError(f"未知的滤波器类型: {filter_type}")
        
        return filter_mask
    
    @staticmethod
    def _apply_frequency_filter(data: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
        """应用频域滤波器
        
        Args:
            data: 输入图像数据
            filter_mask: 频域滤波器
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        # 转换为float32以提高精度
        data_float = data.astype(np.float32)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(data_float)
        f_shift = np.fft.fftshift(f_transform)
        
        # 应用滤波器
        filtered_shift = f_shift * filter_mask
        
        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(filtered_shift)
        filtered_data = np.fft.ifft2(f_ishift)
        
        # 取实部并转换回原始数据类型
        result = np.abs(filtered_data)
        
        # 归一化到原始数据范围
        result_min, result_max = result.min(), result.max()
        if result_max > result_min:
            result = (result - result_min) / (result_max - result_min)
            result = result * (data.max() - data.min()) + data.min()
        else:
            result = data.astype(np.float32)
        
        return np.clip(result, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def ideal_low_pass(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """理想低通滤波
        
        Args:
            data: 输入图像数据
            cutoff_ratio: 截止频率比例 (0.01-0.5)
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        if cutoff_ratio <= 0 or cutoff_ratio >= 1:
            cutoff_ratio = 0.1
            
        filter_mask = FrequencyProcessor._create_frequency_filter(
            data.shape, cutoff_ratio, 'ideal_low')
        
        return FrequencyProcessor._apply_frequency_filter(data, filter_mask)
    
    @staticmethod
    def ideal_high_pass(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """理想高通滤波
        
        Args:
            data: 输入图像数据
            cutoff_ratio: 截止频率比例 (0.01-0.5)
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        if cutoff_ratio <= 0 or cutoff_ratio >= 1:
            cutoff_ratio = 0.1
            
        filter_mask = FrequencyProcessor._create_frequency_filter(
            data.shape, cutoff_ratio, 'ideal_high')
        
        return FrequencyProcessor._apply_frequency_filter(data, filter_mask)
    
    @staticmethod
    def gaussian_low_pass(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """高斯低通滤波
        
        Args:
            data: 输入图像数据
            cutoff_ratio: 截止频率比例 (0.01-0.5)
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        if cutoff_ratio <= 0 or cutoff_ratio >= 1:
            cutoff_ratio = 0.1
            
        filter_mask = FrequencyProcessor._create_frequency_filter(
            data.shape, cutoff_ratio, 'gaussian_low')
        
        return FrequencyProcessor._apply_frequency_filter(data, filter_mask)
    
    @staticmethod
    def gaussian_high_pass(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """高斯高通滤波
        
        Args:
            data: 输入图像数据
            cutoff_ratio: 截止频率比例 (0.01-0.5)
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        if cutoff_ratio <= 0 or cutoff_ratio >= 1:
            cutoff_ratio = 0.1
            
        filter_mask = FrequencyProcessor._create_frequency_filter(
            data.shape, cutoff_ratio, 'gaussian_high')
        
        return FrequencyProcessor._apply_frequency_filter(data, filter_mask)
    
    @staticmethod
    def get_algorithm_info() -> dict:
        """获取频域增强算法信息"""
        return {
            'ideal_low_pass': {
                'name': '理想低通滤波',
                'description': '去除高频噪声，但会产生振铃效应',
                'parameters': {
                    'cutoff_ratio': {
                        'type': 'float', 
                        'range': (0.01, 0.5), 
                        'default': 0.1, 
                        'description': '截止频率比例'
                    }
                }
            },
            'ideal_high_pass': {
                'name': '理想高通滤波',
                'description': '增强边缘和细节，但会产生振铃效应',
                'parameters': {
                    'cutoff_ratio': {
                        'type': 'float', 
                        'range': (0.01, 0.5), 
                        'default': 0.1, 
                        'description': '截止频率比例'
                    }
                }
            },
            'gaussian_low_pass': {
                'name': '高斯低通滤波',
                'description': '平滑去噪，过渡自然',
                'parameters': {
                    'cutoff_ratio': {
                        'type': 'float', 
                        'range': (0.01, 0.5), 
                        'default': 0.1, 
                        'description': '截止频率比例'
                    }
                }
            },
            'gaussian_high_pass': {
                'name': '高斯高通滤波',
                'description': '平滑的边缘增强',
                'parameters': {
                    'cutoff_ratio': {
                        'type': 'float', 
                        'range': (0.01, 0.5), 
                        'default': 0.1, 
                        'description': '截止频率比例'
                    }
                }
            }
        }
