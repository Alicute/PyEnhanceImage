"""
边缘检测处理器模块
"""
import numpy as np
from skimage import feature, filters
from typing import Tuple, Optional

class EdgeProcessor:
    """边缘检测算法集合"""
    
    @staticmethod
    def sobel_edge(data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Sobel边缘检测
        
        Args:
            data: 输入图像数据
            normalize: 是否归一化到原始数据范围
            
        Returns:
            np.ndarray: 边缘检测结果
        """
        # 转换为float32以提高精度
        data_float = data.astype(np.float32)
        
        # Sobel算子检测水平和垂直边缘
        sobel_h = filters.sobel_h(data_float)
        sobel_v = filters.sobel_v(data_float)
        
        # 计算梯度幅值
        sobel_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        if normalize:
            # 归一化到原始数据范围
            result_min, result_max = sobel_magnitude.min(), sobel_magnitude.max()
            if result_max > result_min:
                sobel_magnitude = (sobel_magnitude - result_min) / (result_max - result_min)
                sobel_magnitude = sobel_magnitude * (data.max() - data.min()) + data.min()
        
        return np.clip(sobel_magnitude, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def canny_edge(data: np.ndarray, sigma: float = 1.0, 
                   low_threshold: float = 0.1, high_threshold: float = 0.2) -> np.ndarray:
        """Canny边缘检测
        
        Args:
            data: 输入图像数据
            sigma: 高斯滤波标准差
            low_threshold: 低阈值
            high_threshold: 高阈值
            
        Returns:
            np.ndarray: 边缘检测结果
        """
        # 参数验证
        if sigma <= 0:
            sigma = 1.0
        if low_threshold <= 0 or low_threshold >= 1:
            low_threshold = 0.1
        if high_threshold <= low_threshold or high_threshold >= 1:
            high_threshold = 0.2
        
        # 归一化到0-1范围进行Canny检测
        data_normalized = data.astype(np.float32) / 65535.0
        
        # Canny边缘检测
        edges = feature.canny(data_normalized, sigma=sigma, 
                             low_threshold=low_threshold, 
                             high_threshold=high_threshold)
        
        # 转换为16位图像
        result = edges.astype(np.uint16) * 65535
        
        return result
    
    @staticmethod
    def laplacian_edge(data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Laplacian边缘检测
        
        Args:
            data: 输入图像数据
            normalize: 是否归一化到原始数据范围
            
        Returns:
            np.ndarray: 边缘检测结果
        """
        # 转换为float32以提高精度
        data_float = data.astype(np.float32)
        
        # Laplacian算子
        laplacian = filters.laplace(data_float)
        
        # 取绝对值
        laplacian_abs = np.abs(laplacian)
        
        if normalize:
            # 归一化到原始数据范围
            result_min, result_max = laplacian_abs.min(), laplacian_abs.max()
            if result_max > result_min:
                laplacian_abs = (laplacian_abs - result_min) / (result_max - result_min)
                laplacian_abs = laplacian_abs * (data.max() - data.min()) + data.min()
        
        return np.clip(laplacian_abs, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def edge_enhancement(data: np.ndarray, edge_strength: float = 1.0, 
                        edge_method: str = 'sobel') -> np.ndarray:
        """边缘增强（原图 + 边缘信息）
        
        Args:
            data: 输入图像数据
            edge_strength: 边缘增强强度 (0.1-3.0)
            edge_method: 边缘检测方法 ('sobel', 'laplacian')
            
        Returns:
            np.ndarray: 边缘增强结果
        """
        # 参数验证
        if edge_strength <= 0:
            edge_strength = 1.0
        edge_strength = min(max(edge_strength, 0.1), 3.0)
        
        # 获取边缘信息
        if edge_method == 'sobel':
            edges = EdgeProcessor.sobel_edge(data, normalize=False)
        elif edge_method == 'laplacian':
            edges = EdgeProcessor.laplacian_edge(data, normalize=False)
        else:
            edges = EdgeProcessor.sobel_edge(data, normalize=False)
        
        # 归一化边缘信息
        if edges.max() > 0:
            edges_normalized = edges.astype(np.float32) / edges.max()
        else:
            edges_normalized = edges.astype(np.float32)
        
        # 原图 + 边缘增强
        data_float = data.astype(np.float32)
        enhanced = data_float + edge_strength * edges_normalized * data_float.max() * 0.1
        
        return np.clip(enhanced, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def roberts_edge(data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Roberts边缘检测
        
        Args:
            data: 输入图像数据
            normalize: 是否归一化到原始数据范围
            
        Returns:
            np.ndarray: 边缘检测结果
        """
        # 转换为float32以提高精度
        data_float = data.astype(np.float32)
        
        # Roberts算子
        roberts = filters.roberts(data_float)
        
        if normalize:
            # 归一化到原始数据范围
            result_min, result_max = roberts.min(), roberts.max()
            if result_max > result_min:
                roberts = (roberts - result_min) / (result_max - result_min)
                roberts = roberts * (data.max() - data.min()) + data.min()
        
        return np.clip(roberts, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def get_algorithm_info() -> dict:
        """获取边缘检测算法信息"""
        return {
            'sobel_edge': {
                'name': 'Sobel边缘检测',
                'description': '基于梯度的边缘检测，对噪声敏感度适中',
                'parameters': {}
            },
            'canny_edge': {
                'name': 'Canny边缘检测',
                'description': '精确的边缘检测，具有良好的定位性能',
                'parameters': {
                    'sigma': {
                        'type': 'float', 
                        'range': (0.1, 3.0), 
                        'default': 1.0, 
                        'description': '高斯滤波标准差'
                    },
                    'low_threshold': {
                        'type': 'float', 
                        'range': (0.01, 0.3), 
                        'default': 0.1, 
                        'description': '低阈值'
                    },
                    'high_threshold': {
                        'type': 'float', 
                        'range': (0.1, 0.5), 
                        'default': 0.2, 
                        'description': '高阈值'
                    }
                }
            },
            'laplacian_edge': {
                'name': 'Laplacian边缘检测',
                'description': '二阶导数边缘检测，对噪声敏感',
                'parameters': {}
            },
            'edge_enhancement': {
                'name': '边缘增强',
                'description': '在原图基础上增强边缘信息',
                'parameters': {
                    'edge_strength': {
                        'type': 'float', 
                        'range': (0.1, 3.0), 
                        'default': 1.0, 
                        'description': '边缘增强强度'
                    },
                    'edge_method': {
                        'type': 'choice', 
                        'choices': ['sobel', 'laplacian'], 
                        'default': 'sobel', 
                        'description': '边缘检测方法'
                    }
                }
            },
            'roberts_edge': {
                'name': 'Roberts边缘检测',
                'description': '简单快速的边缘检测算子',
                'parameters': {}
            }
        }
