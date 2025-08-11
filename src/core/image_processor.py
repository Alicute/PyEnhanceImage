"""
图像处理算法模块
"""
import numpy as np
from skimage import exposure, filters, morphology, restoration
from typing import Dict, Any, Tuple, Optional

class ImageProcessor:
    """图像处理算法集合"""
    
    @staticmethod
    def gamma_correction(data: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Gamma校正"""
        if gamma <= 0:
            gamma = 0.1
        
        # 归一化到0-1范围
        normalized = data.astype(np.float32)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        # 应用Gamma校正
        corrected = np.power(normalized, gamma)
        
        # 恢复到原始范围
        corrected = corrected * (data.max() - data.min()) + data.min()
        
        return corrected.astype(np.uint16)
    
    @staticmethod
    def histogram_equalization(data: np.ndarray, method: str = 'global') -> np.ndarray:
        """直方图均衡化"""
        if method == 'global':
            # 全局直方图均衡化
            return exposure.equalize_hist(data) * 65535
        elif method == 'adaptive':
            # 自适应直方图均衡化（CLAHE）
            return exposure.equalize_adapthist(data) * 65535
        else:
            return data
    
    @staticmethod
    def gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """高斯滤波"""
        if sigma <= 0:
            sigma = 0.1
        
        # 转换为float类型进行处理
        filtered = filters.gaussian(data.astype(np.float32), sigma=sigma)
        
        # 恢复到原始范围和类型
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
        filtered = filtered * (data.max() - data.min()) + data.min()
        
        return filtered.astype(np.uint16)
    
    @staticmethod
    def median_filter(data: np.ndarray, disk_size: int = 3) -> np.ndarray:
        """中值滤波"""
        if disk_size < 1:
            disk_size = 1
        
        # 使用圆形结构元素
        selem = morphology.disk(disk_size)
        filtered = filters.median(data, selem)
        
        return filtered.astype(np.uint16)
    
    @staticmethod
    def unsharp_mask(data: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
        """非锐化掩模"""
        if radius <= 0:
            radius = 0.1
        if amount < 0:
            amount = 0
        
        # 转换为float类型进行处理
        data_float = data.astype(np.float32)
        
        # 应用非锐化掩模
        sharpened = filters.unsharp_mask(data_float, radius=radius, amount=amount)
        
        # 恢复到原始范围和类型
        sharpened = (sharpened - sharpened.min()) / (sharpened.max() - sharpened.min())
        sharpened = sharpened * (data.max() - data.min()) + data.min()
        
        return sharpened.astype(np.uint16)
    
    @staticmethod
    def morphological_operation(data: np.ndarray, operation: str, disk_size: int = 3) -> np.ndarray:
        """形态学操作"""
        if disk_size < 1:
            disk_size = 1
        
        # 创建结构元素
        selem = morphology.disk(disk_size)
        
        # 根据操作类型处理
        if operation == 'erosion':
            result = morphology.erosion(data, selem)
        elif operation == 'dilation':
            result = morphology.dilation(data, selem)
        elif operation == 'opening':
            result = morphology.opening(data, selem)
        elif operation == 'closing':
            result = morphology.closing(data, selem)
        else:
            result = data
        
        return result.astype(np.uint16)
    
    @staticmethod
    def low_pass_filter(data: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
        """低通滤波（频域）"""
        # 转换为float类型
        data_float = data.astype(np.float32)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(data_float)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 创建低通滤波器
        rows, cols = data.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高斯低通滤波器
        y, x = np.ogrid[:rows, :cols]
        mask = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (cutoff_frequency * min(rows, cols)) ** 2))
        
        # 应用滤波器
        filtered = f_transform_shifted * mask
        
        # 逆变换
        filtered_shifted = np.fft.ifftshift(filtered)
        filtered_result = np.fft.ifft2(filtered_shifted)
        filtered_result = np.real(filtered_result)
        
        # 恢复到原始范围和类型
        filtered_result = (filtered_result - filtered_result.min()) / (filtered_result.max() - filtered_result.min())
        filtered_result = filtered_result * (data.max() - data.min()) + data.min()
        
        return filtered_result.astype(np.uint16)
    
    @staticmethod
    def high_pass_filter(data: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
        """高通滤波（频域）"""
        # 转换为float类型
        data_float = data.astype(np.float32)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(data_float)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 创建高通滤波器
        rows, cols = data.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高斯高通滤波器
        y, x = np.ogrid[:rows, :cols]
        mask = 1 - np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (cutoff_frequency * min(rows, cols)) ** 2))
        
        # 应用滤波器
        filtered = f_transform_shifted * mask
        
        # 逆变换
        filtered_shifted = np.fft.ifftshift(filtered)
        filtered_result = np.fft.ifft2(filtered_shifted)
        filtered_result = np.real(filtered_result)
        
        # 恢复到原始范围和类型
        filtered_result = (filtered_result - filtered_result.min()) / (filtered_result.max() - filtered_result.min())
        filtered_result = filtered_result * (data.max() - data.min()) + data.min()
        
        return filtered_result.astype(np.uint16)
    
    @staticmethod
    def get_algorithm_info() -> Dict[str, Dict[str, Any]]:
        """获取算法信息"""
        return {
            'gamma_correction': {
                'name': 'Gamma校正',
                'description': '调整图像的亮度分布',
                'parameters': {
                    'gamma': {'type': 'float', 'range': (0.1, 5.0), 'default': 1.0, 'description': 'Gamma值'}
                }
            },
            'histogram_equalization': {
                'name': '直方图均衡化',
                'description': '改善图像对比度',
                'parameters': {
                    'method': {'type': 'string', 'options': ['global', 'adaptive'], 'default': 'global', 'description': '均衡化方法'}
                }
            },
            'gaussian_filter': {
                'name': '高斯滤波',
                'description': '高斯降噪',
                'parameters': {
                    'sigma': {'type': 'float', 'range': (0.1, 10.0), 'default': 1.0, 'description': '高斯核标准差'}
                }
            },
            'median_filter': {
                'name': '中值滤波',
                'description': '中值降噪',
                'parameters': {
                    'disk_size': {'type': 'int', 'range': (1, 10), 'default': 3, 'description': '滤波器大小'}
                }
            },
            'unsharp_mask': {
                'name': '非锐化掩模',
                'description': '图像锐化',
                'parameters': {
                    'radius': {'type': 'float', 'range': (0.1, 5.0), 'default': 1.0, 'description': '锐化半径'},
                    'amount': {'type': 'float', 'range': (0.0, 3.0), 'default': 1.0, 'description': '锐化强度'}
                }
            },
            'morphological_operation': {
                'name': '形态学操作',
                'description': '形态学处理',
                'parameters': {
                    'operation': {'type': 'string', 'options': ['erosion', 'dilation', 'opening', 'closing'], 'default': 'erosion', 'description': '操作类型'},
                    'disk_size': {'type': 'int', 'range': (1, 10), 'default': 3, 'description': '结构元素大小'}
                }
            }
        }