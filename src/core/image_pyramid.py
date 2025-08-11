"""
图像金字塔缓存模块

实现多级图像缓存，优化缩放性能
支持智能缓存策略和内存管理
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict
from dataclasses import dataclass
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QObject, pyqtSignal
import cv2


@dataclass
class PyramidLevel:
    """金字塔级别数据"""
    level: int
    scale_factor: float
    image_data: np.ndarray
    pixmap: Optional[QPixmap] = None
    created_time: float = 0.0
    access_count: int = 0
    last_access_time: float = 0.0
    
    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()
        self.last_access_time = self.created_time


class ImagePyramid(QObject):
    """图像金字塔缓存类
    
    实现多级图像缓存，根据缩放级别智能选择最优图像
    """
    
    # 信号定义
    pyramid_updated = pyqtSignal(int)  # 金字塔级别数量
    cache_stats_changed = pyqtSignal(dict)  # 缓存统计变化
    
    def __init__(self, max_levels: int = 8, max_memory_mb: int = 200):
        """初始化图像金字塔
        
        Args:
            max_levels: 最大金字塔级别数
            max_memory_mb: 最大内存使用（MB）
        """
        super().__init__()
        
        self.max_levels = max_levels
        self.max_memory_mb = max_memory_mb
        
        # 金字塔数据
        self.pyramid_levels: OrderedDict[int, PyramidLevel] = OrderedDict()
        self.original_image: Optional[np.ndarray] = None
        self.original_size: Tuple[int, int] = (0, 0)
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_memory_usage = 0
        self.creation_times = []
        
        # 配置参数
        self.min_image_size = 64  # 最小图像尺寸
        self.downsample_factor = 2  # 下采样因子
        self.quality_threshold = 0.5  # 质量阈值
        
    def set_image(self, image_data: np.ndarray) -> bool:
        """设置原始图像并生成金字塔
        
        Args:
            image_data: 原始图像数据
            
        Returns:
            bool: 是否成功生成金字塔
        """
        if image_data is None or image_data.size == 0:
            return False
        
        try:
            # 清除现有金字塔
            self.clear_pyramid()
            
            # 保存原始图像
            self.original_image = image_data.copy()
            self.original_size = image_data.shape
            
            # 生成金字塔
            self._generate_pyramid()
            
            # 发出信号
            self.pyramid_updated.emit(len(self.pyramid_levels))
            self._emit_cache_stats()
            
            return True
            
        except Exception as e:
            print(f"生成图像金字塔失败: {e}")
            return False
    
    def get_optimal_level(self, scale_factor: float) -> Optional[PyramidLevel]:
        """获取最优的金字塔级别
        
        Args:
            scale_factor: 当前缩放因子
            
        Returns:
            PyramidLevel: 最优的金字塔级别，如果没有则返回None
        """
        if not self.pyramid_levels:
            return None
        
        # 计算目标级别
        if scale_factor >= 1.0:
            # 放大时使用原始图像（级别0）
            target_level = 0
        else:
            # 缩小时计算最优级别
            target_level = int(np.log2(1.0 / scale_factor))
            target_level = min(target_level, max(self.pyramid_levels.keys()))
        
        # 获取最接近的可用级别
        available_levels = list(self.pyramid_levels.keys())
        best_level = min(available_levels, key=lambda x: abs(x - target_level))
        
        # 更新访问统计
        pyramid_level = self.pyramid_levels[best_level]
        pyramid_level.access_count += 1
        pyramid_level.last_access_time = time.time()
        
        # 更新缓存统计
        if best_level == target_level:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        return pyramid_level
    
    def get_pixmap_for_scale(self, scale_factor: float) -> Optional[QPixmap]:
        """获取指定缩放因子的QPixmap
        
        Args:
            scale_factor: 缩放因子
            
        Returns:
            QPixmap: 对应的QPixmap，如果没有则返回None
        """
        pyramid_level = self.get_optimal_level(scale_factor)
        if pyramid_level is None:
            return None
        
        # 如果QPixmap不存在，创建它
        if pyramid_level.pixmap is None:
            pyramid_level.pixmap = self._create_pixmap(pyramid_level.image_data)
        
        return pyramid_level.pixmap
    
    def _generate_pyramid(self):
        """生成图像金字塔"""
        if self.original_image is None:
            return
        
        start_time = time.time()
        
        # 添加原始图像（级别0）
        level_0 = PyramidLevel(
            level=0,
            scale_factor=1.0,
            image_data=self.original_image.copy()
        )
        self.pyramid_levels[0] = level_0
        
        # 生成下采样级别
        current_image = self.original_image
        current_level = 1
        
        while current_level < self.max_levels:
            # 计算新尺寸
            new_height = current_image.shape[0] // self.downsample_factor
            new_width = current_image.shape[1] // self.downsample_factor
            
            # 检查最小尺寸限制
            if new_height < self.min_image_size or new_width < self.min_image_size:
                break
            
            # 检查内存限制
            if self._estimate_memory_usage(new_height, new_width) > self.max_memory_mb * 1024 * 1024:
                break
            
            # 下采样图像
            downsampled = self._downsample_image(current_image, (new_width, new_height))
            
            # 创建金字塔级别
            scale_factor = 1.0 / (self.downsample_factor ** current_level)
            pyramid_level = PyramidLevel(
                level=current_level,
                scale_factor=scale_factor,
                image_data=downsampled
            )
            
            self.pyramid_levels[current_level] = pyramid_level
            
            # 更新内存使用统计
            self.total_memory_usage += downsampled.nbytes
            
            # 准备下一级别
            current_image = downsampled
            current_level += 1
        
        # 记录创建时间
        creation_time = time.time() - start_time
        self.creation_times.append(creation_time)
        
        # 限制创建时间历史记录
        if len(self.creation_times) > 10:
            self.creation_times.pop(0)
    
    def _downsample_image(self, image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """下采样图像
        
        Args:
            image: 原始图像
            new_size: 新尺寸 (width, height)
            
        Returns:
            np.ndarray: 下采样后的图像
        """
        # 使用OpenCV进行高质量下采样
        if len(image.shape) == 2:
            # 灰度图像
            downsampled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            # 彩色图像
            downsampled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        return downsampled
    
    def _create_pixmap(self, image_data: np.ndarray) -> QPixmap:
        """从图像数据创建QPixmap
        
        Args:
            image_data: 图像数据
            
        Returns:
            QPixmap: 创建的QPixmap
        """
        # 确保数据是uint8类型
        if image_data.dtype != np.uint8:
            # 归一化到0-255
            normalized = ((image_data - image_data.min()) / 
                         (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        else:
            normalized = image_data
        
        # 创建QImage
        height, width = normalized.shape
        bytes_per_line = width
        
        qimage = QImage(normalized.data, width, height, bytes_per_line, 
                       QImage.Format.Format_Grayscale8)
        
        # 创建QPixmap
        return QPixmap.fromImage(qimage)
    
    def _estimate_memory_usage(self, height: int, width: int) -> int:
        """估算内存使用量
        
        Args:
            height: 图像高度
            width: 图像宽度
            
        Returns:
            int: 估算的内存使用量（字节）
        """
        # 图像数据 + QPixmap估算
        image_bytes = height * width * np.dtype(np.uint8).itemsize
        pixmap_bytes = image_bytes * 4  # QPixmap通常使用32位
        return image_bytes + pixmap_bytes
    
    def clear_pyramid(self):
        """清除金字塔缓存"""
        self.pyramid_levels.clear()
        self.original_image = None
        self.original_size = (0, 0)
        self.total_memory_usage = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def optimize_cache(self):
        """优化缓存，移除不常用的级别"""
        if len(self.pyramid_levels) <= 2:
            return  # 保留至少2个级别
        
        # 按访问频率排序
        levels_by_usage = sorted(
            self.pyramid_levels.items(),
            key=lambda x: (x[1].access_count, x[1].last_access_time),
            reverse=True
        )
        
        # 保留最常用的级别
        keep_count = max(2, len(levels_by_usage) // 2)
        levels_to_keep = dict(levels_by_usage[:keep_count])
        
        # 移除不常用的级别
        removed_count = 0
        for level_id in list(self.pyramid_levels.keys()):
            if level_id not in levels_to_keep:
                removed_level = self.pyramid_levels.pop(level_id)
                self.total_memory_usage -= removed_level.image_data.nbytes
                removed_count += 1
        
        if removed_count > 0:
            self._emit_cache_stats()
    
    def get_cache_stats(self) -> Dict[str, any]:
        """获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        avg_creation_time = (sum(self.creation_times) / len(self.creation_times) 
                           if self.creation_times else 0.0)
        
        return {
            'pyramid_levels': len(self.pyramid_levels),
            'max_levels': self.max_levels,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'memory_usage_mb': self.total_memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'avg_creation_time_ms': avg_creation_time * 1000,
            'original_size': self.original_size,
            'level_details': {
                level: {
                    'scale_factor': data.scale_factor,
                    'size': data.image_data.shape,
                    'access_count': data.access_count,
                    'memory_mb': data.image_data.nbytes / (1024 * 1024)
                }
                for level, data in self.pyramid_levels.items()
            }
        }
    
    def _emit_cache_stats(self):
        """发出缓存统计信号"""
        stats = self.get_cache_stats()
        self.cache_stats_changed.emit(stats)


# 全局金字塔实例管理
_pyramid_instances: Dict[str, ImagePyramid] = {}


def get_pyramid_for_image(image_id: str) -> ImagePyramid:
    """获取指定图像的金字塔实例
    
    Args:
        image_id: 图像ID
        
    Returns:
        ImagePyramid: 金字塔实例
    """
    if image_id not in _pyramid_instances:
        _pyramid_instances[image_id] = ImagePyramid()
    
    return _pyramid_instances[image_id]


def clear_all_pyramids():
    """清除所有金字塔实例"""
    for pyramid in _pyramid_instances.values():
        pyramid.clear_pyramid()
    _pyramid_instances.clear()
