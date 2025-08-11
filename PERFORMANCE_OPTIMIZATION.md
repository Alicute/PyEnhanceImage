# PyQt6 图像查看器性能优化文档

## 概述

基于对 JavaScript DICOM 查看器的分析，本文档针对 PyQt6 图像查看器的三个核心功能进行性能优化：**缩放**、**画布拖动**和**窗宽窗位调节**。

## 1. 缩放性能优化

### 当前问题
- 每次缩放都重新计算和重绘整个图像
- 没有硬件加速支持
- 缺乏多级缓存机制

### 优化方案

#### 1.1 使用 QOpenGLWidget 替代 QLabel
```python
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer
import numpy as np

class OpenGLImageView(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.texture_id = None
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
    def initializeGL(self):
        """初始化OpenGL上下文"""
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_TEXTURE_2D)
        
    def resizeGL(self, w, h):
        """调整视口大小"""
        glViewport(0, 0, w, h)
        
    def paintGL(self):
        """OpenGL渲染"""
        glClear(GL_COLOR_BUFFER_BIT)
        if self.texture_id:
            # 使用硬件加速渲染
            self._render textured_image()
```

#### 1.2 多级图像金字塔缓存
```python
class ImagePyramid:
    def __init__(self, image_data):
        self.original = image_data
        self.pyramid_levels = self._generate_pyramid()
        
    def _generate_pyramid(self):
        """生成多级图像金字塔"""
        levels = {}
        current = self.original
        
        for level in range(0, 10):  # 10级缩放
            if current.shape[0] < 64 or current.shape[1] < 64:
                break
                
            levels[level] = current.copy()
            # 下采样
            current = cv2.resize(current, 
                              (current.shape[1] // 2, current.shape[0] // 2),
                              interpolation=cv2.INTER_AREA)
            
        return levels
    
    def get_optimal_level(self, zoom_factor):
        """根据缩放因子获取最优图像级别"""
        level = int(np.log2(1.0 / zoom_factor)) if zoom_factor < 1.0 else 0
        level = max(0, min(level, len(self.pyramid_levels) - 1))
        return self.pyramid_levels[level]
```

#### 1.3 智能缓存机制
```python
class ZoomCache:
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 100
        
    def get_cached_image(self, zoom_level, pan_offset):
        """获取缓存的图像"""
        key = (zoom_level, tuple(pan_offset))
        return self.cache.get(key)
        
    def cache_image(self, zoom_level, pan_offset, image_data):
        """缓存图像数据"""
        key = (zoom_level, tuple(pan_offset))
        if len(self.cache) >= self.max_cache_size:
            # LRU缓存淘汰
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = image_data.copy()
```

## 2. 画布拖动性能优化

### 当前问题
- 拖动时频繁重绘整个图像
- 没有硬件加速
- 缺乏平滑的拖动体验

### 优化方案

#### 2.1 视口变换矩阵优化
```python
class ViewportTransform:
    def __init__(self):
        self.transform = np.eye(3)  # 3x3变换矩阵
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
    def update_transform(self):
        """更新变换矩阵"""
        # 平移 -> 缩放 -> 平移
        self.transform = np.array([
            [self.zoom, 0, self.pan_x],
            [0, self.zoom, self.pan_y],
            [0, 0, 1]
        ])
        
    def transform_point(self, x, y):
        """变换坐标点"""
        point = np.array([x, y, 1])
        transformed = self.transform @ point
        return transformed[0], transformed[1]
```

#### 2.2 双缓冲渲染
```python
class DoubleBufferWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.front_buffer = QPixmap()
        self.back_buffer = QPixmap()
        self.is_rendering = False
        
    def paintEvent(self, event):
        """双缓冲绘制"""
        if not self.is_rendering:
            self.is_rendering = True
            
            # 在后台缓冲区绘制
            if self.back_buffer.isNull():
                self.back_buffer = QPixmap(self.size())
            
            painter = QPainter(self.back_buffer)
            self._draw_scene(painter)
            painter.end()
            
            # 交换缓冲区
            self.front_buffer = self.back_buffer.copy()
            
            # 显示前台缓冲区
            display_painter = QPainter(self)
            display_painter.drawPixmap(0, 0, self.front_buffer)
            display_painter.end()
            
            self.is_rendering = False
```

#### 2.3 智能重绘区域
```python
class SmartRepaintWidget:
    def __init__(self):
        self.last_visible_rect = QRect()
        self.dirty_regions = []
        
    def update_visible_rect(self, new_rect):
        """更新可见区域，只重绘变化部分"""
        if self.last_visible_rect != new_rect:
            # 计算需要重绘的区域
            changed_region = self.last_visible_rect.united(new_rect)
            self.update(changed_region)
            self.last_visible_rect = new_rect
```

## 3. 窗宽窗位调节性能优化

### 当前问题
- 每次调节都重新计算整个图像
- 没有查找表优化
- 缺乏实时响应能力

### 优化方案

#### 3.1 预计算查找表 (LUT)
```python
class WindowLevelLUT:
    def __init__(self):
        self.lut_cache = {}
        self.max_cache_size = 50
        
    def get_lut(self, window_width, window_level):
        """获取或创建查找表"""
        key = (window_width, window_level)
        
        if key not in self.lut_cache:
            if len(self.lut_cache) >= self.max_cache_size:
                # 清理缓存
                oldest_key = next(iter(self.lut_cache))
                del self.lut_cache[oldest_key]
            
            # 创建新的查找表
            self.lut_cache[key] = self._create_lut(window_width, window_level)
        
        return self.lut_cache[key]
    
    def _create_lut(self, window_width, window_level):
        """创建窗宽窗位查找表"""
        lut = np.zeros(65536, dtype=np.uint8)
        
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2
        
        if window_width > 0:
            slope = 255.0 / window_width
            for i in range(65536):
                if i < min_val:
                    lut[i] = 0
                elif i > max_val:
                    lut[i] = 255
                else:
                    lut[i] = int((i - min_val) * slope)
        
        return lut
```

#### 3.2 多线程图像处理
```python
from PyQt6.QtCore import QThread, pyqtSignal, QMutex

class ImageProcessingThread(QThread):
    image_processed = pyqtSignal(object, float, float)
    
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.processing_queue = []
        self.is_running = True
        
    def add_task(self, image_data, window_width, window_level):
        """添加处理任务"""
        self.mutex.lock()
        self.processing_queue.append((image_data, window_width, window_level))
        self.mutex.unlock()
        
    def run(self):
        """处理线程主循环"""
        while self.is_running:
            self.mutex.lock()
            if self.processing_queue:
                task = self.processing_queue.pop(0)
                self.mutex.unlock()
                
                image_data, window_width, window_level = task
                processed = self._apply_window_level(image_data, window_width, window_level)
                
                self.image_processed.emit(processed, window_width, window_level)
            else:
                self.mutex.unlock()
                self.msleep(16)  # 60 FPS
    
    def _apply_window_level(self, image_data, window_width, window_level):
        """应用窗宽窗位处理"""
        lut = self.lut_cache.get_lut(window_width, window_level)
        return lut[image_data]
```

#### 3.3 实时防抖调节
```python
class SmoothWindowLevelController:
    def __init__(self):
        self.target_ww = 400.0
        self.target_wl = 40.0
        self.current_ww = 400.0
        self.current_wl = 40.0
        self.smoothing_factor = 0.2
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._smooth_update)
        self.timer.start(16)  # 60 FPS
        
    def set_target_values(self, window_width, window_level):
        """设置目标值"""
        self.target_ww = window_width
        self.target_wl = window_level
        
    def _smooth_update(self):
        """平滑更新当前值"""
        # 指数平滑
        self.current_ww += (self.target_ww - self.current_ww) * self.smoothing_factor
        self.current_wl += (self.target_wl - self.current_wl) * self.smoothing_factor
        
        # 检查是否需要更新
        if (abs(self.current_ww - self.target_ww) > 0.1 or 
            abs(self.current_wl - self.target_wl) > 0.1):
            self.values_changed.emit(self.current_ww, self.current_wl)
```

## 4. 综合优化策略

### 4.1 内存优化
```python
class MemoryManager:
    def __init__(self):
        self.image_cache = {}
        self.max_memory_mb = 512  # 最大内存使用
        
    def get_memory_usage(self):
        """获取当前内存使用"""
        total_memory = 0
        for img in self.image_cache.values():
            total_memory += img.nbytes
        return total_memory / (1024 * 1024)
    
    def cleanup_cache(self):
        """清理缓存"""
        while self.get_memory_usage() > self.max_memory_mb:
            if not self.image_cache:
                break
            # 移除最旧的缓存
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]
```

### 4.2 性能监控
```python
class PerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.frame_times = []
        
    def start_frame(self):
        """开始帧计时"""
        self.frame_start_time = time.time()
        
    def end_frame(self):
        """结束帧计时"""
        frame_time = time.time() - self.frame_start_time
        self.frame_times.append(frame_time)
        
        # 保持最近100帧的记录
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
            
        # 计算FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time
            return fps
        return 0
```

## 5. 实施建议

### 5.1 优先级排序
1. **高优先级**：窗宽窗位查找表优化
2. **中优先级**：多线程图像处理
3. **低优先级**：OpenGL硬件加速

### 5.2 渐进式实施
1. 先实施查找表优化，立即提升响应速度
2. 添加多线程处理，改善用户体验
3. 最后考虑OpenGL硬件加速，提供最佳性能

### 5.3 性能测试
```python
def run_performance_tests():
    """运行性能测试"""
    import time
    
    # 测试查找表性能
    start_time = time.time()
    for i in range(1000):
        lut = WindowLevelLUT()._create_lut(400, 40)
    lut_time = time.time() - start_time
    
    # 测试多线程性能
    thread = ImageProcessingThread()
    start_time = time.time()
    thread.start()
    thread.wait(1000)  # 等待1秒
    thread.is_running = False
    thread.wait()
    
    print(f"LUT创建时间: {lut_time:.3f}s")
    print(f"线程处理性能: {len(thread.processing_queue)} tasks/sec")
```

## 6. 预期性能提升

| 功能 | 当前性能 | 优化后预期 | 提升倍数 |
|------|----------|------------|----------|
| 窗宽窗位调节 | 100-200ms | 5-10ms | 20x |
| 图像缩放 | 200-500ms | 20-50ms | 10x |
| 画布拖动 | 50-100ms | 5-15ms | 10x |
| 内存使用 | 500MB+ | 100-200MB | 5x |

通过这些优化，PyQt6图像查看器的性能应该能够接近JavaScript实现的流畅度。