"""
平滑窗宽窗位控制器

实现防抖动和平滑更新，减少不必要的重绘
提供60FPS的流畅调节体验
"""

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from typing import Optional
import time


class SmoothWindowLevelController(QObject):
    """平滑窗宽窗位控制器
    
    实现防抖动调节和平滑值更新
    减少UI重绘频率，提升性能
    """
    
    # 信号：当值需要更新时发出
    values_changed = pyqtSignal(float, float)  # window_width, window_level
    
    def __init__(self, parent=None):
        """初始化控制器
        
        Args:
            parent: 父对象
        """
        super().__init__(parent)
        
        # 当前值和目标值
        self.current_ww = 400.0
        self.current_wl = 40.0
        self.target_ww = 400.0
        self.target_wl = 40.0
        
        # 平滑参数
        self.smoothing_factor = 0.3  # 平滑因子，越大越快
        self.update_threshold = 0.5  # 更新阈值，小于此值不触发更新
        
        # 防抖动参数
        self.debounce_delay_ms = 50  # 防抖动延迟（毫秒）
        self.last_input_time = 0.0
        
        # 定时器设置
        self.smooth_timer = QTimer()
        self.smooth_timer.timeout.connect(self._smooth_update)
        self.smooth_timer.start(16)  # 60 FPS
        
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._apply_debounced_values)
        
        # 状态标志
        self.is_updating = False
        self.has_pending_update = False
        
        # 性能统计
        self.update_count = 0
        self.last_update_time = 0.0
        
    def set_target_values(self, window_width: float, window_level: float):
        """设置目标窗宽窗位值
        
        Args:
            window_width: 目标窗宽
            window_level: 目标窗位
        """
        self.target_ww = float(window_width)
        self.target_wl = float(window_level)
        self.last_input_time = time.time()
        
        # 重启防抖动定时器
        self.debounce_timer.stop()
        self.debounce_timer.start(self.debounce_delay_ms)
        
        self.has_pending_update = True
    
    def set_immediate_values(self, window_width: float, window_level: float):
        """立即设置窗宽窗位值（跳过平滑和防抖动）
        
        Args:
            window_width: 窗宽值
            window_level: 窗位值
        """
        self.current_ww = float(window_width)
        self.current_wl = float(window_level)
        self.target_ww = self.current_ww
        self.target_wl = self.current_wl
        
        # 停止所有定时器
        self.debounce_timer.stop()
        self.has_pending_update = False
        
        # 立即发出信号
        self.values_changed.emit(self.current_ww, self.current_wl)
        self.update_count += 1
        self.last_update_time = time.time()
    
    def _apply_debounced_values(self):
        """应用防抖动后的值"""
        # 检查是否有足够的时间间隔
        current_time = time.time()
        if current_time - self.last_input_time >= self.debounce_delay_ms / 1000.0:
            self.has_pending_update = True
    
    def _smooth_update(self):
        """平滑更新当前值"""
        if not self.has_pending_update or self.is_updating:
            return
        
        # 计算当前值与目标值的差距
        ww_diff = abs(self.current_ww - self.target_ww)
        wl_diff = abs(self.current_wl - self.target_wl)
        
        # 如果差距很小，直接设置为目标值
        if ww_diff < self.update_threshold and wl_diff < self.update_threshold:
            if ww_diff > 0.01 or wl_diff > 0.01:  # 避免无意义的微小更新
                self.current_ww = self.target_ww
                self.current_wl = self.target_wl
                self._emit_values_changed()
            self.has_pending_update = False
            return
        
        # 指数平滑更新
        self.current_ww += (self.target_ww - self.current_ww) * self.smoothing_factor
        self.current_wl += (self.target_wl - self.current_wl) * self.smoothing_factor
        
        # 发出更新信号
        self._emit_values_changed()
    
    def _emit_values_changed(self):
        """发出值变化信号"""
        if self.is_updating:
            return
        
        self.is_updating = True
        try:
            self.values_changed.emit(self.current_ww, self.current_wl)
            self.update_count += 1
            self.last_update_time = time.time()
        finally:
            self.is_updating = False
    
    def get_current_values(self) -> tuple[float, float]:
        """获取当前窗宽窗位值
        
        Returns:
            tuple: (window_width, window_level)
        """
        return (self.current_ww, self.current_wl)
    
    def get_target_values(self) -> tuple[float, float]:
        """获取目标窗宽窗位值
        
        Returns:
            tuple: (window_width, window_level)
        """
        return (self.target_ww, self.target_wl)
    
    def is_animating(self) -> bool:
        """检查是否正在动画中
        
        Returns:
            bool: 是否正在平滑更新
        """
        ww_diff = abs(self.current_ww - self.target_ww)
        wl_diff = abs(self.current_wl - self.target_wl)
        return ww_diff > self.update_threshold or wl_diff > self.update_threshold
    
    def set_smoothing_factor(self, factor: float):
        """设置平滑因子
        
        Args:
            factor: 平滑因子，范围0.1-1.0，越大越快
        """
        self.smoothing_factor = max(0.1, min(1.0, factor))
    
    def set_debounce_delay(self, delay_ms: int):
        """设置防抖动延迟
        
        Args:
            delay_ms: 延迟时间（毫秒）
        """
        self.debounce_delay_ms = max(10, min(500, delay_ms))
    
    def set_update_threshold(self, threshold: float):
        """设置更新阈值
        
        Args:
            threshold: 更新阈值，小于此值不触发更新
        """
        self.update_threshold = max(0.1, min(10.0, threshold))
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        current_time = time.time()
        return {
            'update_count': self.update_count,
            'last_update_time': self.last_update_time,
            'time_since_last_update': current_time - self.last_update_time,
            'is_animating': self.is_animating(),
            'has_pending_update': self.has_pending_update,
            'smoothing_factor': self.smoothing_factor,
            'debounce_delay_ms': self.debounce_delay_ms,
            'update_threshold': self.update_threshold,
            'current_values': self.get_current_values(),
            'target_values': self.get_target_values()
        }
    
    def reset(self):
        """重置控制器状态"""
        self.debounce_timer.stop()
        self.has_pending_update = False
        self.is_updating = False
        self.update_count = 0
        self.last_update_time = time.time()
    
    def stop(self):
        """停止控制器"""
        self.smooth_timer.stop()
        self.debounce_timer.stop()
        self.has_pending_update = False
        self.is_updating = False
    
    def start(self):
        """启动控制器"""
        if not self.smooth_timer.isActive():
            self.smooth_timer.start(16)  # 60 FPS
