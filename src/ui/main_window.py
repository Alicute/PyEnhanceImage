"""
主窗口类
"""
import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                           QSplitter, QMenuBar, QMenu, QFileDialog, QStatusBar,
                           QMessageBox, QApplication, QProgressBar, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon

from .image_view import ImageView
from .control_panel import ControlPanel
from .smooth_controller import SmoothWindowLevelController
from ..core.image_manager import ImageManager
from ..core.image_processor import ImageProcessor
from ..core.image_processing_thread import ImageProcessingThread
from ..utils.helpers import generate_output_filename, ensure_directory_exists
from ..utils.memory_monitor import get_memory_monitor

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.image_manager = ImageManager()
        self.image_processor = ImageProcessor()
        self.view_sync_enabled = False

        # 初始化多线程处理
        self.processing_thread = ImageProcessingThread()
        self.processing_thread.start()

        # 当前处理任务ID
        self.current_task_id = None

        # 初始化平滑窗宽窗位控制器
        self.smooth_controller = SmoothWindowLevelController()
        self.smooth_controller.values_changed.connect(self._apply_smooth_window_level)

        # 初始化内存监控
        self.memory_monitor = get_memory_monitor()
        self.memory_monitor.start_tracing()

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
        self.image_splitter = QSplitter(Qt.Orientation.Vertical)

        self.original_view = ImageView("原始图像")
        self.processed_view = ImageView("处理结果")

        self.image_splitter.addWidget(self.original_view)
        self.image_splitter.addWidget(self.processed_view)

        # 默认只显示处理结果窗口（下面的窗口）
        self.image_splitter.setSizes([0, 800])  # 上面窗口高度为0
        self.is_split_view = False  # 分窗状态标志
        
        image_layout.addWidget(self.image_splitter)
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

        # 添加进度指示器
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # 添加队列状态标签
        self.queue_label = QLabel("队列: 0/0")
        self.status_bar.addPermanentWidget(self.queue_label)

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
        
        # 分窗菜单
        window_menu = menubar.addMenu("分窗")

        # 分窗切换
        self.split_view_action = QAction("切换到双窗口", self)
        self.split_view_action.setCheckable(True)
        self.split_view_action.setChecked(False)  # 默认单窗口
        self.split_view_action.triggered.connect(self.toggle_split_view)
        window_menu.addAction(self.split_view_action)

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
        self.control_panel.invert_changed.connect(self.on_invert_changed)
        


        # 多线程处理信号
        self.processing_thread.task_started.connect(self.on_task_started)
        self.processing_thread.task_progress.connect(self.on_task_progress)
        self.processing_thread.task_completed.connect(self.on_task_completed)
        self.processing_thread.task_failed.connect(self.on_task_failed)
        self.processing_thread.queue_status_changed.connect(self.on_queue_status_changed)
        
    def load_dicom(self):
        """加载DICOM文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择DICOM文件", "", "DICOM文件 (*.dcm);;所有文件 (*.*)"
        )
        
        if file_path:
            # 清理之前的缓存，释放内存
            self._clear_memory_caches()

            self.status_bar.showMessage(f"正在加载: {file_path}")
            QApplication.processEvents()

            if self.image_manager.load_dicom(file_path):
                self.update_display()
                self.control_panel.set_controls_enabled(True)

                # 更新控制面板的图像数据
                self.control_panel.update_image_data(
                    self.image_manager.current_image.data,
                    self.image_manager.original_image.data
                )

                # 自动优化窗宽窗位（内部会更新智能滑块范围）
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

            # 更新控制面板的图像数据
            self.control_panel.update_image_data(
                self.image_manager.current_image.data,
                self.image_manager.original_image.data
            )

            # 更新图像信息显示
            data_min = int(self.image_manager.original_image.data.min())
            data_max = int(self.image_manager.original_image.data.max())
            data_mean = float(self.image_manager.original_image.data.mean())
            self.control_panel.update_image_info(data_min, data_max, data_mean)

        self.status_bar.showMessage("已重置为原始图像")
        
    def apply_algorithm(self, algorithm_name: str, parameters: dict):
        """应用图像处理算法 - 使用多线程异步处理"""
        if self.image_manager.current_image is None:
            return

        # 取消当前任务（如果有）
        if self.current_task_id:
            self.processing_thread.cancel_task(self.current_task_id)

        # 获取当前图像数据
        current_data = self.image_manager.current_image.data

        # 如果是基于窗宽窗位的增强，添加窗宽窗位参数
        if algorithm_name == 'window_based_enhance':
            current_ww = self.image_manager.current_image.window_width
            current_wl = self.image_manager.current_image.window_level
            parameters['window_width'] = current_ww
            parameters['window_level'] = current_wl

        # 生成任务描述
        description = self._generate_task_description(algorithm_name, parameters)

        # 添加任务到处理队列
        self.current_task_id = self.processing_thread.add_task(
            algorithm_name, parameters, current_data, description
        )

        # 禁用控制面板，防止重复提交
        self.control_panel.set_controls_enabled(False)

        self.status_bar.showMessage(f"已添加到处理队列: {description}")

    def _generate_task_description(self, algorithm_name: str, parameters: dict) -> str:
        """生成任务描述

        Args:
            algorithm_name: 算法名称
            parameters: 算法参数

        Returns:
            str: 任务描述
        """
        if algorithm_name == 'gamma_correction':
            return f"Gamma校正 (γ={parameters['gamma']})"
        elif algorithm_name == 'histogram_equalization':
            method_name = "全局均衡化" if parameters['method'] == 'global' else "CLAHE"
            return f"直方图均衡化 ({method_name})"
        elif algorithm_name == 'gaussian_filter':
            return f"高斯滤波 (σ={parameters['sigma']})"
        elif algorithm_name == 'median_filter':
            return f"中值滤波 (size={parameters['disk_size']})"
        elif algorithm_name == 'unsharp_mask':
            return f"非锐化掩模 (r={parameters['radius']}, a={parameters['amount']})"
        elif algorithm_name == 'morphological_operation':
            operation_name = {
                'erosion': '腐蚀',
                'dilation': '膨胀',
                'opening': '开运算',
                'closing': '闭运算'
            }[parameters['operation']]
            return f"形态学{operation_name} (size={parameters['disk_size']})"
        elif algorithm_name == 'window_based_enhance':
            return "窗位增强"
        elif algorithm_name == 'paper_enhance':
            return "论文算法处理"
        else:
            return algorithm_name
            
    def update_display(self, reset_view: bool = True):
        """更新显示 - 性能优化版本

        Args:
            reset_view: 是否重置视图缩放，默认True
        """
        # 获取当前反相状态
        is_inverted = self.control_panel.get_invert_state()

        # 只更新可见的视图，提高性能
        if self.is_split_view and self.image_manager.original_image:
            # 双窗口模式：更新原始图像视图
            original_display = self.image_manager.get_windowed_image(
                self.image_manager.original_image, invert=is_inverted)
            self.original_view.set_image(original_display, reset_view)

        if self.image_manager.current_image:
            # 总是更新处理后的图像视图（主要视图）
            processed_display = self.image_manager.get_windowed_image(
                self.image_manager.current_image, invert=is_inverted)
            self.processed_view.set_image(processed_display, reset_view)
            
    def on_window_width_changed(self, value: float):
        """窗宽改变事件 - 直接处理，提高响应速度"""
        if self.image_manager.current_image:
            # 获取当前窗位值
            current_wl = self.image_manager.current_image.window_level
            # 直接更新，不使用防抖动（提高响应速度）
            self._apply_window_level_fast(value, current_wl)

    def on_window_level_changed(self, value: float):
        """窗位改变事件 - 直接处理，提高响应速度"""
        if self.image_manager.current_image:
            # 获取当前窗宽值
            current_ww = self.image_manager.current_image.window_width
            # 直接更新，不使用防抖动（提高响应速度）
            self._apply_window_level_fast(current_ww, value)

    def _apply_window_level_fast(self, window_width: float, window_level: float):
        """快速应用窗宽窗位调节 - 优化性能"""
        if self.image_manager.current_image:
            # 简化处理，减少开销
            self.image_manager.update_window_settings(window_width, window_level)
            # 窗宽窗位调节时不重置视图缩放
            self.update_display(reset_view=False)

    def _apply_smooth_window_level(self, window_width: float, window_level: float):
        """应用平滑的窗宽窗位调节（保留用于特殊情况）"""
        if self.image_manager.current_image:
            # 简化版本，减少监控开销
            self.image_manager.update_window_settings(window_width, window_level)
            # 窗宽窗位调节时不重置视图缩放
            self.update_display(reset_view=False)

            # 减少垃圾回收频率
            if not hasattr(self, '_gc_counter'):
                self._gc_counter = 0
            self._gc_counter += 1

            # 每5次调节才进行一次垃圾回收
            if self._gc_counter % 5 == 0:
                import gc
                gc.collect()
    
    def auto_optimize_window(self):
        """智能自动优化窗宽窗位 - 基于专业建议的改进算法"""
        if self.image_manager.current_image is None:
            return

        # 获取图像数据
        data = self.image_manager.current_image.data
        data_min = int(data.min())
        data_max = int(data.max())
        data_mean = float(data.mean())
        total_pixels = data.size

        print(f"\n🎯 自动优化分析:")
        print(f"   数据范围: {data_min} - {data_max}")
        print(f"   数据均值: {data_mean:.1f}")
        print(f"   图像大小: {data.shape}")

        # 计算直方图
        hist, bins = np.histogram(data.flatten(), bins=65536, range=(data_min, data_max))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 修正：使用真正的bin中心
        cumulative_pixels = np.cumsum(hist)

        # 检测过曝峰值
        pixel_ratios = hist / total_pixels
        major_peaks = np.where(pixel_ratios > 0.05)[0]  # 超过5%的bins

        overexposed_peaks = []
        for peak_idx in major_peaks:
            peak_value = bin_centers[peak_idx]
            peak_ratio = pixel_ratios[peak_idx]
            # 过曝判断：灰度值 > 80%范围 且 像素数 > 5%
            if peak_value > (data_min + (data_max - data_min) * 0.8):
                overexposed_peaks.append((peak_value, peak_ratio))
                print(f"   🔥 过曝峰值: {peak_value:.1f} ({peak_ratio*100:.1f}%像素)")

        # 根据是否有过曝背景选择算法
        if overexposed_peaks:
            print(f"   🎯 检测到过曝背景，使用工件优化算法")

            # 排除过曝区域，只在有效区域计算
            overexposed_threshold = min(peak[0] for peak in overexposed_peaks)
            noise_threshold = total_pixels * 0.0001

            # 找到有效的工件数据区域
            valid_bins = np.where((bin_centers < overexposed_threshold) & (hist > noise_threshold))[0]

            if len(valid_bins) > 10:  # 确保有足够的有效数据
                # 在有效区域内计算5%-95%
                valid_pixels = np.sum(hist[valid_bins])
                valid_cumulative = np.cumsum(hist[valid_bins])

                lower_threshold = valid_pixels * 0.05
                upper_threshold = valid_pixels * 0.95

                lower_idx = np.where(valid_cumulative >= lower_threshold)[0]
                upper_idx = np.where(valid_cumulative >= upper_threshold)[0]

                if len(lower_idx) > 0 and len(upper_idx) > 0:
                    lower_value = bin_centers[valid_bins[lower_idx[0]]]
                    upper_value = bin_centers[valid_bins[upper_idx[0]]]

                    window_level = (lower_value + upper_value) / 2
                    window_width = (upper_value - lower_value) * 1.5  # 扩展50%

                    print(f"   工件区域: {lower_value:.1f} - {upper_value:.1f}")
                else:
                    # 回退：使用有效区域的全范围
                    lower_value = bin_centers[valid_bins[0]]
                    upper_value = bin_centers[valid_bins[-1]]
                    window_level = (lower_value + upper_value) / 2
                    window_width = (upper_value - lower_value) * 2
                    print(f"   回退算法: 使用有效区域全范围")
            else:
                # 最终回退：使用中位数算法
                median_value = np.median(data)
                window_level = median_value * 0.8
                window_width = data.std() * 3
                print(f"   最终回退: 使用中位数算法")
        else:
            print(f"   ✅ 未检测到过曝背景，使用标准算法")

            # 标准5%-95%算法
            lower_bound = np.where(cumulative_pixels >= total_pixels * 0.05)[0]
            upper_bound = np.where(cumulative_pixels >= total_pixels * 0.95)[0]

            if len(lower_bound) > 0 and len(upper_bound) > 0:
                lower_value = bin_centers[lower_bound[0]]
                upper_value = bin_centers[upper_bound[0]]
                window_level = (lower_value + upper_value) / 2
                window_width = upper_value - lower_value
            else:
                # 回退到全范围
                window_width = data_max - data_min
                window_level = (data_min + data_max) / 2

        # 限制窗宽范围
        window_width = max(100, min(window_width, 60000))

        print(f"   ✅ 最终设置: 窗宽={window_width:.0f}, 窗位={window_level:.0f}")

        # 应用设置
        self.control_panel.set_window_settings(window_width, window_level)
        self.control_panel.update_image_info(data_min, data_max, data_mean)

        # 自动优化后更新智能滑块范围
        self.update_smart_slider_ranges()

        self.status_bar.showMessage(f"自动优化: 窗宽={window_width:.0f}, 窗位={window_level:.0f}")

    def on_image_wheel_event(self, view, event):
        """图像视图滚轮事件"""
        # 直接调用ImageView的wheelEvent方法
        view.wheelEvent(event)

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
        
    def update_smart_slider_ranges(self):
        """更新智能滑块范围"""
        if self.image_manager.current_image:
            # 计算智能范围
            ww_range, wl_range = self.image_manager.calculate_smart_slider_ranges(
                self.image_manager.current_image)

            # 更新控制面板的滑块范围
            self.control_panel.update_slider_ranges(ww_range, wl_range)

    def on_invert_changed(self, is_inverted: bool):
        """反相状态改变处理"""
        # 更新显示，不重置视图缩放
        self.update_display(reset_view=False)

        status = "启用" if is_inverted else "禁用"
        self.status_bar.showMessage(f"反相显示已{status}")

    def toggle_split_view(self):
        """切换分窗显示"""
        self.is_split_view = self.split_view_action.isChecked()

        if self.is_split_view:
            # 显示双窗口：上面原图，下面处理结果
            self.image_splitter.setSizes([400, 400])
            self.status_bar.showMessage("已切换到双窗口显示")
            self.split_view_action.setText("切换到单窗口")
        else:
            # 显示单窗口：只显示处理结果
            self.image_splitter.setSizes([0, 800])
            self.status_bar.showMessage("已切换到单窗口显示")
            self.split_view_action.setText("切换到双窗口")

    def reset_views(self):
        """重置视图"""
        self.original_view.reset_view()
        self.processed_view.reset_view()
        
    def save_current_result(self):
        """保存当前结果"""
        if self.image_manager.current_image is None:
            return

        # 生成默认文件名：原始图像名 + 增强步骤名称
        original_name = self.image_manager.original_image.name if self.image_manager.original_image else "image"
        # 移除原始文件的扩展名
        base_name = os.path.splitext(original_name)[0]

        # 获取最后一个处理步骤的描述
        if self.image_manager.processing_history:
            last_step = self.image_manager.processing_history[-1]['description']
            # 清理描述中的特殊字符，用于文件名
            step_name = "".join(c for c in last_step if c.isalnum() or c in (' ', '-', '_')).strip()
            step_name = step_name.replace(' ', '_')
            default_filename = f"{base_name}_{step_name}.dcm"
        else:
            default_filename = f"{base_name}_processed.dcm"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存DICOM文件", default_filename, "DICOM文件 (*.dcm)"
        )

        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.endswith('.dcm'):
                    file_path += '.dcm'

                # 实现DICOM保存功能
                self._save_dicom_file(file_path)
                self.status_bar.showMessage(f"已保存: {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def _save_dicom_file(self, file_path: str):
        """保存DICOM文件的具体实现"""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import generate_uid
        import tempfile

        # 获取当前图像数据
        current_image = self.image_manager.current_image

        # 如果有原始DICOM文件，基于它创建新的DICOM
        if hasattr(current_image, 'metadata') and 'dicom_dataset' in current_image.metadata:
            # 复制原始DICOM数据集
            original_ds = current_image.metadata['dicom_dataset']
            ds = pydicom.dcmread(tempfile.NamedTemporaryFile().name, force=True) if hasattr(original_ds, 'copy') else Dataset()

            # 复制重要的元数据
            for tag in ['PatientName', 'PatientID', 'StudyDate', 'StudyTime',
                       'Modality', 'StudyInstanceUID', 'SeriesInstanceUID',
                       'ImageOrientationPatient', 'ImagePositionPatient',
                       'PixelSpacing', 'SliceThickness']:
                if hasattr(original_ds, tag):
                    setattr(ds, tag, getattr(original_ds, tag))
        else:
            # 创建新的DICOM数据集
            ds = Dataset()
            ds.PatientName = "Anonymous"
            ds.PatientID = "000000"
            ds.Modality = "OT"  # Other
            ds.StudyInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()

        # 设置图像相关的必要字段
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage

        # 设置图像数据
        pixel_data = current_image.data.astype(np.uint16)
        ds.PixelData = pixel_data.tobytes()

        # 设置图像属性
        ds.Rows, ds.Columns = pixel_data.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # 设置窗宽窗位
        ds.WindowWidth = current_image.window_width
        ds.WindowCenter = current_image.window_level

        # 添加处理历史到DICOM注释中
        if self.image_manager.processing_history:
            history_text = "Processing History: " + "; ".join([
                step['description'] for step in self.image_manager.processing_history
            ])
            ds.ImageComments = history_text[:1024]  # DICOM字段长度限制

        # 创建文件数据集并保存
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        file_ds = FileDataset(file_path, ds, file_meta=file_meta, preamble=b"\0" * 128)
        file_ds.save_as(file_path, write_like_original=False)
                
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
                         "交互式图像增强实验平台 v1.0"
                         "基于Python的桌面GUI应用程序，"
                         "为工业领域的测试工程师提供"
                         "交互式的图像增强实验与教学平台。"
                         "技术栈: PyQt6, pydicom, scikit-image")
        
    def on_task_started(self, task_id: str):
        """任务开始处理"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("正在处理...")

    def on_task_progress(self, task_id: str, progress: float):
        """任务进度更新"""
        self.progress_bar.setValue(int(progress * 100))

    def on_task_completed(self, task_id: str, result_data: np.ndarray, description: str):
        """任务完成处理"""
        try:
            # 更新图像管理器
            if self.image_manager.current_image:
                # 获取当前任务的算法名称和参数（从描述中解析或使用默认值）
                algorithm_name = "processed"
                parameters = {}

                self.image_manager.apply_processing(
                    algorithm_name, parameters, result_data, description)

                # 更新显示
                self.update_display()

                # 更新控制面板的图像数据
                self.control_panel.update_image_data(
                    self.image_manager.current_image.data,
                    self.image_manager.original_image.data
                )

                # 如果是窗位增强，自动优化窗宽窗位
                if 'window_based_enhance' in description:
                    print("🎯 检测到窗位增强，自动优化窗宽窗位...")
                    self.auto_optimize_window()

                # 更新历史记录
                self.control_panel.update_history(self.image_manager.processing_history)

            # 隐藏进度条
            self.progress_bar.setVisible(False)

            # 重新启用控制面板
            self.control_panel.set_controls_enabled(True)

            # 清除当前任务ID
            if self.current_task_id == task_id:
                self.current_task_id = None

            self.status_bar.showMessage(f"处理完成: {description}")

        except Exception as e:
            self.on_task_failed(task_id, str(e))

    def on_task_failed(self, task_id: str, error_message: str):
        """任务失败处理"""
        # 隐藏进度条
        self.progress_bar.setVisible(False)

        # 重新启用控制面板
        self.control_panel.set_controls_enabled(True)

        # 清除当前任务ID
        if self.current_task_id == task_id:
            self.current_task_id = None

        # 显示错误信息
        QMessageBox.critical(self, "处理错误", f"图像处理失败:\n{error_message}")
        self.status_bar.showMessage("处理失败")

    def on_queue_status_changed(self, pending_count: int, total_count: int):
        """队列状态变化"""
        self.queue_label.setText(f"队列: {pending_count}/{total_count}")

    def _clear_memory_caches(self):
        """清理内存缓存"""
        try:
            # 清理LUT缓存
            from ..core.window_level_lut import get_global_lut
            lut = get_global_lut()
            lut.clear_cache()

            # 清理图像管理器缓存
            self.image_manager.original_display_cache = None
            self.image_manager.current_display_cache = None

            # 清理图像金字塔缓存
            from ..core.image_pyramid import clear_all_pyramids
            clear_all_pyramids()

            # 强制垃圾回收
            import gc
            gc.collect()

            print("内存缓存已清理")

        except Exception as e:
            print(f"清理内存缓存时出错: {e}")

    def closeEvent(self, event):
        """关闭事件"""
        # 清理内存缓存
        self._clear_memory_caches()

        # 停止处理线程
        if hasattr(self, 'processing_thread'):
            self.processing_thread.stop_processing()
            self.processing_thread.wait(3000)  # 等待最多3秒
            if self.processing_thread.isRunning():
                self.processing_thread.terminate()

        # 停止平滑控制器
        if hasattr(self, 'smooth_controller'):
            self.smooth_controller.stop()

        event.accept()