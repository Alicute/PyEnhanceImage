"""
图像显示控件
"""
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QTransform
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal

class ImageView(QGraphicsView):
    """图像显示控件，支持缩放、拖放等交互"""
    
    def __init__(self, title="图像显示"):
        super().__init__()
        self.title = title
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 图像数据
        self.pixmap_item = None
        self.original_pixmap = None
        
        # 缩放和拖动状态
        self.scale_factor = 1.0
        self.last_mouse_pos = QPointF()
        self.is_dragging = False
        self.drag_start_pos = QPointF()
        
        # 设置属性
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # 背景色
        self.setBackgroundBrush(QColor(53, 53, 53))
        
        # 最小缩放
        self.setMinimumSize(200, 200)
        
    def set_image(self, image_data: np.ndarray):
        """设置图像数据"""
        if image_data is None:
            self.clear()
            return
        
        # 确保数据是uint8
        if image_data.dtype != np.uint8:
            # 归一化到0-255
            image_data = ((image_data - image_data.min()) / 
                         (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        # 转换为QImage
        height, width = image_data.shape
        bytes_per_line = width
        
        # 创建灰度图像
        qimage = QImage(image_data.data, width, height, bytes_per_line, 
                       QImage.Format.Format_Grayscale8)
        
        # 创建QPixmap
        pixmap = QPixmap.fromImage(qimage)
        self.original_pixmap = pixmap
        
        # 清除场景并添加新的pixmap
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        
        # 重置视图
        self.reset_view()
        
    def clear(self):
        """清除图像"""
        self.scene.clear()
        self.pixmap_item = None
        self.original_pixmap = None
        
    def reset_view(self):
        """重置视图以适应窗口"""
        if self.original_pixmap:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.scale_factor = 1.0
            
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if not self.original_pixmap:
            return
            
        # 获取缩放因子
        zoom_factor = 1.15
        
        # 根据滚轮方向确定缩放
        if event.angleDelta().y() > 0:
            # 放大
            self.scale(zoom_factor, zoom_factor)
            self.scale_factor *= zoom_factor
        else:
            # 缩小
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
            self.scale_factor /= zoom_factor
            
        # 限制缩放范围
        if self.scale_factor < 0.1:
            self.scale(0.1 / self.scale_factor, 0.1 / self.scale_factor)
            self.scale_factor = 0.1
        elif self.scale_factor > 10.0:
            self.scale(10.0 / self.scale_factor, 10.0 / self.scale_factor)
            self.scale_factor = 10.0
            
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)
        
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_dragging and self.original_pixmap:
            # 计算移动距离
            delta = event.position() - self.drag_start_pos
            
            # 移动视图
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            
            self.drag_start_pos = event.position()
            
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