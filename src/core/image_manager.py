"""
图像数据管理模块
"""
import numpy as np
import pydicom
import os
import warnings
import uuid
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from .window_level_lut import get_global_lut

# 过滤DICOM字符编码警告
warnings.filterwarnings('ignore', category=UserWarning, message='Incorrect value for Specific Character Set')

class ImageDataType(Enum):
    """图像数据类型"""
    ORIGINAL = "original"
    CURRENT = "current"
    PROCESSED = "processed"

@dataclass
class ImageData:
    """图像数据结构"""
    data: np.ndarray  # 16位图像数据
    metadata: Dict[str, Any]  # DICOM元数据
    window_width: float = 400.0
    window_level: float = 40.0
    name: str = "未命名图像"
    description: str = ""
    id: str = ""  # 唯一标识符，用于比较对象

class ImageManager:
    """图像数据管理器"""
    
    def __init__(self):
        self.original_image: Optional[ImageData] = None
        self.current_image: Optional[ImageData] = None
        self.processing_history: List[Dict[str, Any]] = []
        
        # 显示缓存 - 优化窗宽窗位调节性能
        self.original_display_cache: Optional[np.ndarray] = None
        self.current_display_cache: Optional[np.ndarray] = None
        self.last_window_settings: Tuple[float, float] = (400.0, 40.0)
        
        # 图像ID映射
        self.original_image_id: Optional[str] = None
        self.current_image_id: Optional[str] = None
        
    def load_dicom(self, file_path: str) -> bool:
        """加载DICOM文件"""
        try:
            ds = pydicom.dcmread(file_path)
            
            # 获取像素数据
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                
                # 确保数据是16位的
                if pixel_array.dtype != np.uint16:
                    if pixel_array.max() <= 255:
                        pixel_array = (pixel_array * 256).astype(np.uint16)
                    else:
                        pixel_array = pixel_array.astype(np.uint16)
                
                # 提取元数据
                metadata = {}
                if hasattr(ds, 'PatientName'):
                    metadata['patient_name'] = str(ds.PatientName)
                if hasattr(ds, 'StudyDescription'):
                    metadata['study_description'] = str(ds.StudyDescription)
                if hasattr(ds, 'SeriesDescription'):
                    metadata['series_description'] = str(ds.SeriesDescription)
                if hasattr(ds, 'ImageComments'):
                    metadata['image_comments'] = str(ds.ImageComments)
                
                # 读取窗宽窗位信息
                window_width = 400.0
                window_level = 40.0
                
                # 尝试从DICOM文件中读取窗宽窗位
                if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                    try:
                        # 处理多个窗宽窗位值的情况
                        if isinstance(ds.WindowWidth, (list, tuple, np.ndarray)):
                            window_width = float(ds.WindowWidth[0])
                        else:
                            window_width = float(ds.WindowWidth)
                        
                        if isinstance(ds.WindowCenter, (list, tuple, np.ndarray)):
                            window_level = float(ds.WindowCenter[0])
                        else:
                            window_level = float(ds.WindowCenter)
                    except (IndexError, ValueError, TypeError):
                        # 如果读取失败，使用自动计算的值
                        pass
                
                # 如果没有有效的窗宽窗位，根据数据范围自动计算
                if window_width <= 0 or window_level <= 0:
                    data_min, data_max = pixel_array.min(), pixel_array.max()
                    window_width = data_max - data_min
                    window_level = (data_min + data_max) / 2
                
                # 确保窗宽不为零
                if window_width <= 0:
                    window_width = 400.0
                
                # 将窗宽窗位信息添加到元数据
                metadata['window_width'] = window_width
                metadata['window_level'] = window_level
                
                # 创建图像数据对象
                image_id = str(uuid.uuid4())
                image_data = ImageData(
                    data=pixel_array,
                    metadata=metadata,
                    window_width=window_width,
                    window_level=window_level,
                    name=os.path.basename(file_path),
                    description=f"从 {file_path} 加载的DICOM图像",
                    id=image_id
                )
                
                # 设置原始图像和当前图像
                self.original_image = image_data
                self.original_image_id = image_id
                
                current_image_id = str(uuid.uuid4())
                self.current_image = ImageData(
                    data=pixel_array.copy(),
                    metadata=metadata.copy(),
                    window_width=image_data.window_width,
                    window_level=image_data.window_level,
                    name=image_data.name,
                    description=image_data.description,
                    id=current_image_id
                )
                self.current_image_id = current_image_id
                
                # 清空处理历史
                self.processing_history = []
                
                # 初始化显示缓存
                self._refresh_display_cache()
                
                return True
                
        except Exception as e:
            print(f"加载DICOM文件失败: {e}")
            return False
    
    def reset_to_original(self):
        """重置为原始图像"""
        if self.original_image:
            current_image_id = str(uuid.uuid4())
            self.current_image = ImageData(
                data=self.original_image.data.copy(),
                metadata=self.original_image.metadata.copy(),
                window_width=self.original_image.window_width,
                window_level=self.original_image.window_level,
                name=self.original_image.name,
                description=self.original_image.description,
                id=current_image_id
            )
            self.current_image_id = current_image_id
            self.processing_history = []
            
            # 重置显示缓存
            self._refresh_display_cache()
    
    def apply_processing(self, algorithm_name: str, parameters: Dict[str, Any], 
                        processed_data: np.ndarray, description: str = ""):
        """应用图像处理算法"""
        if self.current_image is None:
            return False
        
        # 更新当前图像
        self.current_image.data = processed_data
        if description:
            self.current_image.description = description
        
        # 清除当前图像的显示缓存（数据已改变）
        self.current_display_cache = None
        
        # 添加到处理历史
        self.processing_history.append({
            'algorithm': algorithm_name,
            'parameters': parameters,
            'timestamp': np.datetime64('now'),
            'description': description
        })
        
        return True
    
    def get_windowed_image(self, image_data: ImageData, invert: bool = False) -> np.ndarray:
        """应用窗宽窗位调整 - 优化版本，使用缓存

        Args:
            image_data: 图像数据
            invert: 是否反相显示
        """
        if image_data is None:
            return np.zeros((100, 100), dtype=np.uint8)

        # 检查是否使用缓存（包括反相状态）
        current_settings = (image_data.window_width, image_data.window_level, invert)
        if current_settings == self.last_window_settings:
            # 返回缓存的显示数据
            if image_data.id == self.original_image_id and self.original_display_cache is not None:
                return self.original_display_cache
            elif image_data.id == self.current_image_id and self.current_display_cache is not None:
                return self.current_display_cache

        # 重新计算显示数据
        display_data = self._calculate_windowed_display(image_data, invert)

        # 更新缓存（不需要copy，显示数据是只读的）
        if image_data.id == self.original_image_id:
            self.original_display_cache = display_data
        elif image_data.id == self.current_image_id:
            self.current_display_cache = display_data

        # 更新最后使用的窗宽窗位设置
        self.last_window_settings = current_settings

        return display_data
    
    def _calculate_windowed_display(self, image_data: ImageData, invert: bool = False) -> np.ndarray:
        """计算窗宽窗位显示数据 - 使用LUT优化

        Args:
            image_data: 图像数据
            invert: 是否反相显示
        """
        data = image_data.data
        window_width = image_data.window_width
        window_level = image_data.window_level

        # 使用LUT优化的窗宽窗位计算
        lut = get_global_lut()
        windowed_data = lut.apply_lut(data, window_width, window_level)

        # 如果启用反相，对显示数据进行反相
        if invert:
            windowed_data = 255 - windowed_data

        return windowed_data

    def calculate_smart_slider_ranges(self, image_data: ImageData) -> tuple:
        """计算智能滑块范围

        Args:
            image_data: 图像数据

        Returns:
            tuple: ((ww_min, ww_max), (wl_min, wl_max))
        """
        if image_data is None:
            return (1, 65535), (0, 65535)

        data = image_data.data
        current_ww = image_data.window_width
        current_wl = image_data.window_level

        # 方法1：基于直方图找有效数据范围
        hist, bins = np.histogram(data.flatten(), bins=1000, range=(data.min(), data.max()))
        total_pixels = data.size
        noise_threshold = total_pixels * 0.001  # 0.1%以下认为是噪声

        # 找到有效的bins（排除噪声）
        effective_bins = np.where(hist > noise_threshold)[0]

        if len(effective_bins) > 0:
            effective_min = bins[effective_bins[0]]
            effective_max = bins[effective_bins[-1]]
        else:
            # 回退到数据范围
            effective_min = float(data.min())
            effective_max = float(data.max())

        print(f"🎯 智能范围计算:")
        print(f"   原始数据范围: {data.min()} - {data.max()}")
        print(f"   有效数据范围: {effective_min:.1f} - {effective_max:.1f}")
        print(f"   当前窗宽窗位: {current_ww:.1f}, {current_wl:.1f}")

        # 方法2：基于当前值和有效范围计算智能范围
        if current_ww > 0 and current_wl > 0:
            # 窗位范围：围绕当前值，考虑有效数据范围
            effective_range = effective_max - effective_min

            # 窗位范围：当前值 ± 窗宽的2倍，但不超出有效范围太多
            wl_margin = max(current_ww * 2, effective_range * 0.1)
            wl_min = max(effective_min - effective_range * 0.1, current_wl - wl_margin)
            wl_max = min(effective_max + effective_range * 0.1, current_wl + wl_margin)

            # 窗宽范围：当前值的0.1倍到5倍，但有合理上限
            ww_min = max(1, current_ww // 10)
            ww_max = min(current_ww * 5, effective_range * 3)

            # 确保范围不会太小
            if ww_max - ww_min < 1000:
                ww_max = ww_min + 1000
            if wl_max - wl_min < current_ww:
                wl_center = (wl_min + wl_max) / 2
                wl_min = wl_center - current_ww
                wl_max = wl_center + current_ww

        else:
            # 回退策略：使用有效数据范围
            wl_min, wl_max = effective_min, effective_max
            ww_min = 1
            ww_max = effective_max - effective_min

        # 确保范围合理
        ww_min = max(1, int(ww_min))
        ww_max = min(65535, int(ww_max))
        wl_min = max(0, int(wl_min))
        wl_max = min(65535, int(wl_max))

        # 确保最小值小于最大值
        if ww_min >= ww_max:
            ww_max = ww_min + 1000
        if wl_min >= wl_max:
            wl_max = wl_min + 1000

        print(f"   智能窗宽范围: {ww_min} - {ww_max}")
        print(f"   智能窗位范围: {wl_min} - {wl_max}")

        return (ww_min, ww_max), (wl_min, wl_max)

    def _refresh_display_cache(self):
        """刷新显示缓存"""
        if self.original_image:
            self.original_display_cache = self._calculate_windowed_display(self.original_image)
        if self.current_image:
            self.current_display_cache = self._calculate_windowed_display(self.current_image)
    
    def update_window_settings(self, window_width: float, window_level: float):
        """更新窗宽窗位设置 - 优化版本，只更新必要的缓存"""
        if self.current_image:
            # 只有当窗宽窗位真的改变时才更新
            if (self.current_image.window_width != window_width or 
                self.current_image.window_level != window_level):
                
                self.current_image.window_width = window_width
                self.current_image.window_level = window_level
                
                # 清除当前图像的显示缓存（强制重新计算）
                self.current_display_cache = None
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        return {
            'has_image': self.original_image is not None,
            'original_image': self.original_image,
            'current_image': self.current_image,
            'processing_history': self.processing_history.copy(),
            'history_length': len(self.processing_history)
        }

    def get_lut_performance_stats(self) -> Dict[str, Any]:
        """获取LUT性能统计信息"""
        lut = get_global_lut()
        return lut.get_cache_stats()

    def optimize_lut_cache(self):
        """优化LUT缓存大小"""
        lut = get_global_lut()
        lut.optimize_cache_size()