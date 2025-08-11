"""
图像显示控件
"""
import numpy as np
import time
import uuid
from typing import Optional
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QTransform
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from ..core.image_pyramid import ImagePyramid, get_pyramid_for_image

class ImageView(QGraphicsView):
    """图像显示控件，支持缩放、拖放等交互

    集成图像金字塔缓存，优化大图像的缩放性能
    """

    def __init__(self, title="图像显示"):
        super().__init__()
        self.title = title
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # 图像数据和金字塔
        self.pixmap_item = None
        self.original_pixmap = None
        self.image_id = None
        self.pyramid: Optional[ImagePyramid] = None
        self.current_image_data: Optional[np.ndarray] = None

        # 缩放和拖动状态
        self.scale_factor = 1.0
        self.last_mouse_pos = QPointF()
        self.is_dragging = False
        self.drag_start_pos = QPointF()

        # 性能优化参数
        self.zoom_factor = 1.15
        self.min_scale = 0.05
        self.max_scale = 20.0
        self.update_delay_ms = 16  # 60 FPS

        # 延迟更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update_pixmap)
        self.pending_scale_factor = None

        # 设置属性
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 无限画布模式设置
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 启用无限滚动
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 设置场景大小为无限大
        self.setSceneRect(-10000, -10000, 20000, 20000)

        # 背景色
        self.setBackgroundBrush(QColor(53, 53, 53))

        # 最小缩放
        self.setMinimumSize(200, 200)

        # 性能统计
        self.zoom_operations = 0
        self.last_zoom_time = 0.0
        
    def set_image(self, image_data: np.ndarray):
        """设置图像数据并生成金字塔缓存"""
        if image_data is None:
            self.clear()
            return

        # 生成唯一的图像ID
        self.image_id = str(uuid.uuid4())

        # 保存当前图像数据引用（不拷贝，节省内存）
        self.current_image_data = image_data

        # 快速类型检查和转换
        if image_data.dtype == np.uint8:
            # 已经是正确类型，直接使用
            display_data = image_data
        else:
            # 需要转换，但避免昂贵的min/max计算
            # 假设数据已经在合理范围内（来自窗宽窗位处理）
            display_data = np.clip(image_data, 0, 255).astype(np.uint8)

        # 先创建基础pixmap
        pixmap = self._create_pixmap_from_data(display_data)
        self.original_pixmap = pixmap

        # 暂时禁用图像金字塔以解决内存问题
        # TODO: 重新设计金字塔以减少内存使用
        self.pyramid = None

        # 清除场景并添加新的pixmap
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)

        # 将图像放置在场景中心
        self.pixmap_item.setPos(-pixmap.width()/2, -pixmap.height()/2)

        # 重置视图
        self.reset_view()
        
    def clear(self):
        """清除图像"""
        self.scene.clear()
        self.pixmap_item = None
        self.original_pixmap = None
        self.image_id = None
        self.pyramid = None
        self.current_image_data = None
        self.update_timer.stop()
        self.pending_scale_factor = None
        
    def reset_view(self):
        """重置视图以适应窗口"""
        if self.pixmap_item:
            # 重置变换
            self.resetTransform()
            self.scale_factor = 1.0

            # 居中显示图像
            self.centerOn(self.pixmap_item)

            # 适应窗口大小
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            
    def wheelEvent(self, event):
        """鼠标滚轮缩放 - 无限画布模式"""
        if not self.pixmap_item:
            return

        # 性能统计
        current_time = time.time()
        self.zoom_operations += 1
        self.last_zoom_time = current_time

        # 计算缩放因子
        if event.angleDelta().y() > 0:
            # 放大
            scale_delta = self.zoom_factor
        else:
            # 缩小
            scale_delta = 1.0 / self.zoom_factor

        # 计算新的缩放因子
        new_scale_factor = self.scale_factor * scale_delta

        # 限制缩放范围
        new_scale_factor = max(self.min_scale, min(new_scale_factor, self.max_scale))

        # 如果缩放因子没有实际变化，直接返回
        if abs(new_scale_factor - self.scale_factor) < 0.001:
            return

        # 获取鼠标在场景中的位置（缩放中心）
        mouse_scene_pos = self.mapToScene(event.position().toPoint())

        # 应用缩放变换
        actual_scale_delta = new_scale_factor / self.scale_factor
        self.scale(actual_scale_delta, actual_scale_delta)

        # 更新缩放因子
        self.scale_factor = new_scale_factor

        # 将鼠标位置重新居中（保持缩放中心不变）
        new_mouse_scene_pos = self.mapToScene(event.position().toPoint())
        delta = mouse_scene_pos - new_mouse_scene_pos
        self.translate(delta.x(), delta.y())

        # 延迟更新pixmap以提高响应性
        self.pending_scale_factor = new_scale_factor
        self.update_timer.start(self.update_delay_ms)
            
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        super().mouseMoveEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        """双击重置视图"""
        self.reset_view()
        
    def get_current_transform(self) -> QTransform:
        """获取当前的变换矩阵"""
        return self.transform()
        
    def set_sync_mode(self, enabled: bool):
        """设置同步模式"""
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def _delayed_update_pixmap(self):
        """延迟更新pixmap，使用金字塔缓存"""
        if self.pending_scale_factor is None:
            return

        try:
            # 如果有金字塔，使用金字塔获取最优pixmap
            if self.pyramid:
                optimal_pixmap = self.pyramid.get_pixmap_for_scale(self.pending_scale_factor)

                if optimal_pixmap and self.pixmap_item:
                    # 更新场景中的pixmap
                    self.pixmap_item.setPixmap(optimal_pixmap)

            # 清除待处理的缩放因子
            self.pending_scale_factor = None

        except Exception as e:
            print(f"更新pixmap失败: {e}")
            # 清除待处理的缩放因子，避免重复错误
            self.pending_scale_factor = None

    def _create_pixmap_from_data(self, image_data: np.ndarray) -> QPixmap:
        """从图像数据创建QPixmap

        Args:
            image_data: 图像数据

        Returns:
            QPixmap: 创建的QPixmap
        """
        # 转换为QImage
        height, width = image_data.shape
        bytes_per_line = width

        # 创建灰度图像
        qimage = QImage(image_data.data, width, height, bytes_per_line,
                       QImage.Format.Format_Grayscale8)

        # 创建QPixmap
        return QPixmap.fromImage(qimage)

    def get_pyramid_stats(self) -> dict:
        """获取金字塔统计信息

        Returns:
            dict: 统计信息
        """
        if not self.pyramid:
            return {}

        stats = self.pyramid.get_cache_stats()
        stats.update({
            'zoom_operations': self.zoom_operations,
            'last_zoom_time': self.last_zoom_time,
            'current_scale_factor': self.scale_factor,
            'pending_update': self.pending_scale_factor is not None
        })

        return stats

    def optimize_pyramid_cache(self):
        """优化金字塔缓存"""
        if self.pyramid:
            self.pyramid.optimize_cache()

    def force_update_pixmap(self):
        """强制更新pixmap"""
        if self.pyramid and self.pixmap_item:
            optimal_pixmap = self.pyramid.get_pixmap_for_scale(self.scale_factor)
            if optimal_pixmap:
                self.pixmap_item.setPixmap(optimal_pixmap)