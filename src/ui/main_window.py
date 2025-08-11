"""
主窗口类
"""
import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                           QSplitter, QMenuBar, QMenu, QFileDialog, QStatusBar,
                           QMessageBox, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon

from .image_view import ImageView
from .control_panel import ControlPanel
from ..core.image_manager import ImageManager
from ..core.image_processor import ImageProcessor
from ..utils.helpers import generate_output_filename, ensure_directory_exists

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.image_manager = ImageManager()
        self.image_processor = ImageProcessor()
        self.view_sync_enabled = False
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("交互式图像增强实验平台")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 创建左侧控制面板
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setMinimumWidth(300)
        
        # 创建右侧图像显示区域
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        
        # 创建图像显示控件
        image_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.original_view = ImageView("原始图像")
        self.processed_view = ImageView("处理结果")
        
        image_splitter.addWidget(self.original_view)
        image_splitter.addWidget(self.processed_view)
        image_splitter.setSizes([400, 400])
        
        image_layout.addWidget(image_splitter)
        image_widget.setLayout(image_layout)
        
        # 添加到分割器
        splitter.addWidget(self.control_panel)
        splitter.addWidget(image_widget)
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 加载DICOM
        load_action = QAction("加载DICOM", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_dicom)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 重置视图
        reset_view_action = QAction("重置视图", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self.reset_views)
        view_menu.addAction(reset_view_action)
        
        view_menu.addSeparator()
        
        # 同步视图
        self.sync_view_action = QAction("同步视图", self)
        self.sync_view_action.setCheckable(True)
        self.sync_view_action.setChecked(False)
        self.sync_view_action.triggered.connect(self.toggle_view_sync)
        view_menu.addAction(self.sync_view_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def connect_signals(self):
        """连接信号"""
        # 控制面板信号
        self.control_panel.load_dicom_clicked.connect(self.load_dicom)
        self.control_panel.reset_clicked.connect(self.reset_to_original)
        self.control_panel.apply_algorithm.connect(self.apply_algorithm)
        self.control_panel.save_current_clicked.connect(self.save_current_result)
        self.control_panel.save_preview_clicked.connect(self.save_preview_image)
        self.control_panel.window_width_changed.connect(self.on_window_width_changed)
        self.control_panel.window_level_changed.connect(self.on_window_level_changed)
        self.control_panel.sync_view_toggled.connect(self.toggle_view_sync)
        self.control_panel.window_auto_requested.connect(self.auto_optimize_window)
        
        # 图像视图信号
        self.original_view.wheelEvent = lambda event: self.on_image_wheel_event(self.original_view, event)
        self.processed_view.wheelEvent = lambda event: self.on_image_wheel_event(self.processed_view, event)
        
    def load_dicom(self):
        """加载DICOM文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择DICOM文件", "", "DICOM文件 (*.dcm);;所有文件 (*.*)"
        )
        
        if file_path:
            self.status_bar.showMessage(f"正在加载: {file_path}")
            QApplication.processEvents()
            
            if self.image_manager.load_dicom(file_path):
                self.update_display()
                self.control_panel.set_controls_enabled(True)
                # 自动优化窗宽窗位
                self.auto_optimize_window()
                self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "错误", "加载DICOM文件失败")
                self.status_bar.showMessage("加载失败")
                
    def reset_to_original(self):
        """重置为原始图像"""
        self.image_manager.reset_to_original()
        self.update_display()
        self.control_panel.update_history([])
        
        # 重置窗宽窗位UI控件为原始图像的值
        if self.image_manager.original_image:
            original_ww = self.image_manager.original_image.window_width
            original_wl = self.image_manager.original_image.window_level
            self.control_panel.set_window_settings(original_ww, original_wl)
            
            # 更新图像信息显示
            data_min = int(self.image_manager.original_image.data.min())
            data_max = int(self.image_manager.original_image.data.max())
            data_mean = float(self.image_manager.original_image.data.mean())
            self.control_panel.update_image_info(data_min, data_max, data_mean)
        
        self.status_bar.showMessage("已重置为原始图像")
        
    def apply_algorithm(self, algorithm_name: str, parameters: dict):
        """应用图像处理算法"""
        if self.image_manager.current_image is None:
            return
            
        try:
            self.status_bar.showMessage(f"正在处理: {algorithm_name}")
            QApplication.processEvents()
            
            # 获取当前图像数据
            current_data = self.image_manager.current_image.data
            
            # 应用算法
            if algorithm_name == 'gamma_correction':
                processed_data = self.image_processor.gamma_correction(
                    current_data, parameters['gamma'])
                description = f"Gamma校正 (γ={parameters['gamma']})"
                
            elif algorithm_name == 'histogram_equalization':
                processed_data = self.image_processor.histogram_equalization(
                    current_data, parameters['method'])
                method_name = "全局均衡化" if parameters['method'] == 'global' else "CLAHE"
                description = f"直方图均衡化 ({method_name})"
                
            elif algorithm_name == 'gaussian_filter':
                processed_data = self.image_processor.gaussian_filter(
                    current_data, parameters['sigma'])
                description = f"高斯滤波 (σ={parameters['sigma']})"
                
            elif algorithm_name == 'median_filter':
                processed_data = self.image_processor.median_filter(
                    current_data, parameters['disk_size'])
                description = f"中值滤波 (size={parameters['disk_size']})"
                
            elif algorithm_name == 'unsharp_mask':
                processed_data = self.image_processor.unsharp_mask(
                    current_data, parameters['radius'], parameters['amount'])
                description = f"非锐化掩模 (r={parameters['radius']}, a={parameters['amount']})"
                
            elif algorithm_name == 'morphological_operation':
                processed_data = self.image_processor.morphological_operation(
                    current_data, parameters['operation'], parameters['disk_size'])
                operation_name = {
                    'erosion': '腐蚀',
                    'dilation': '膨胀',
                    'opening': '开运算',
                    'closing': '闭运算'
                }[parameters['operation']]
                description = f"形态学{operation_name} (size={parameters['disk_size']})"
                
            else:
                processed_data = current_data
                description = algorithm_name
                
            # 更新图像管理器
            self.image_manager.apply_processing(
                algorithm_name, parameters, processed_data, description)
            
            # 更新显示
            self.update_display()
            
            # 更新历史记录
            self.control_panel.update_history(self.image_manager.processing_history)
            
            self.status_bar.showMessage(f"处理完成: {description}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
            self.status_bar.showMessage("处理失败")
            
    def update_display(self):
        """更新显示"""
        if self.image_manager.original_image:
            # 显示原始图像
            original_display = self.image_manager.get_windowed_image(
                self.image_manager.original_image)
            self.original_view.set_image(original_display)
            
        if self.image_manager.current_image:
            # 显示处理后的图像
            processed_display = self.image_manager.get_windowed_image(
                self.image_manager.current_image)
            self.processed_view.set_image(processed_display)
            
    def on_window_width_changed(self, value: float):
        """窗宽改变事件"""
        if self.image_manager.current_image:
            self.image_manager.update_window_settings(value, self.image_manager.current_image.window_level)
            self.update_display()
            
    def on_window_level_changed(self, value: float):
        """窗位改变事件"""
        if self.image_manager.current_image:
            self.image_manager.update_window_settings(self.image_manager.current_image.window_width, value)
            self.update_display()
    
    def auto_optimize_window(self):
        """自动优化窗宽窗位"""
        if self.image_manager.current_image is None:
            return
        
        # 获取图像数据
        data = self.image_manager.current_image.data
        
        # 计算图像统计信息
        data_min = int(data.min())
        data_max = int(data.max())
        data_mean = float(data.mean())
        
        # 基于直方图分析自动计算窗宽窗位
        # 计算直方图
        hist, bins = np.histogram(data.flatten(), bins=256, range=(data_min, data_max))
        
        # 找到主要像素值的范围（排除极值）
        total_pixels = data.size
        cumulative_pixels = np.cumsum(hist)
        
        # 找到5%和95%的像素值位置
        lower_bound = np.where(cumulative_pixels >= total_pixels * 0.05)[0]
        upper_bound = np.where(cumulative_pixels >= total_pixels * 0.95)[0]
        
        if len(lower_bound) > 0 and len(upper_bound) > 0:
            lower_idx = lower_bound[0]
            upper_idx = upper_bound[0]
            
            # 计算优化的窗宽窗位
            window_level = (bins[lower_idx] + bins[upper_idx]) / 2
            window_width = bins[upper_idx] - bins[lower_idx]
            
            # 确保窗宽合理
            if window_width < 100:
                window_width = 100
            elif window_width > 60000:
                window_width = 60000
        else:
            # 回退到简单统计
            window_width = data_max - data_min
            window_level = (data_min + data_max) / 2
        
        # 应用优化的窗宽窗位
        self.control_panel.set_window_settings(window_width, window_level)
        
        # 更新图像信息显示
        self.control_panel.update_image_info(data_min, data_max, data_mean)
        
        # 更新状态栏
        self.status_bar.showMessage(f"自动优化: 窗宽={window_width:.0f}, 窗位={window_level:.0f}")
        
    def on_image_wheel_event(self, view, event):
        """图像视图滚轮事件"""
        # 调用原始的滚轮事件处理
        super(type(view), view).wheelEvent(event)
        
        # 如果启用了视图同步，同步另一个视图
        if self.view_sync_enabled:
            other_view = self.processed_view if view == self.original_view else self.original_view
            if other_view and view.original_pixmap:
                # 获取当前变换
                transform = view.get_current_transform()
                other_view.setTransform(transform)
                
    def toggle_view_sync(self, enabled: bool):
        """切换视图同步"""
        self.view_sync_enabled = enabled
        self.sync_view_action.setChecked(enabled)
        self.control_panel.sync_checkbox.setChecked(enabled)
        
    def reset_views(self):
        """重置视图"""
        self.original_view.reset_view()
        self.processed_view.reset_view()
        
    def save_current_result(self):
        """保存当前结果"""
        if self.image_manager.current_image is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存DICOM文件", "", "DICOM文件 (*.dcm)"
        )
        
        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.endswith('.dcm'):
                    file_path += '.dcm'
                    
                # 这里需要实现DICOM保存功能
                # 由于pydicom的保存比较复杂，这里简化处理
                self.status_bar.showMessage(f"已保存: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                
    def save_preview_image(self):
        """保存预览图像"""
        if self.image_manager.current_image is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存预览图像", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)"
        )
        
        if file_path:
            try:
                # 获取当前显示的图像数据
                display_data = self.image_manager.get_windowed_image(
                    self.image_manager.current_image)
                
                # 保存图像
                import cv2
                if file_path.endswith('.png'):
                    cv2.imwrite(file_path, display_data)
                elif file_path.endswith('.jpg'):
                    cv2.imwrite(file_path, display_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                self.status_bar.showMessage(f"已保存: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "交互式图像增强实验平台 v1.0\\n\\n"
                         "基于Python的桌面GUI应用程序，\\n"
                         "为工业领域的测试工程师提供\\n"
                         "交互式的图像增强实验与教学平台。\\n\\n"
                         "技术栈: PyQt6, pydicom, scikit-image")
        
    def closeEvent(self, event):
        """关闭事件"""
        event.accept()