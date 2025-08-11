"""
控制面板组件
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QSlider, QPushButton, QComboBox, QSpinBox,
                           QDoubleSpinBox, QCheckBox, QScrollArea, QMenu)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from typing import Dict, Any, Tuple

class ControlPanel(QWidget):
    """控制面板"""
    
    # 信号定义
    load_dicom_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    apply_algorithm = pyqtSignal(str, object)
    save_current_clicked = pyqtSignal()
    save_preview_clicked = pyqtSignal()
    window_width_changed = pyqtSignal(float)
    window_level_changed = pyqtSignal(float)
    sync_view_toggled = pyqtSignal(bool)
    window_auto_requested = pyqtSignal()
    invert_changed = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.algorithm_widgets = {}
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        # 文件操作组
        file_group = self.create_file_group()
        layout.addWidget(file_group)
        
        # 窗宽窗位调节组
        window_group = self.create_window_group()
        layout.addWidget(window_group)
        
        # 图像增强算法组
        algorithm_group = self.create_algorithm_group()
        layout.addWidget(algorithm_group)
        
        # 历史记录和操作组
        history_group = self.create_history_group()
        layout.addWidget(history_group)
        
        # 保存选项组
        save_group = self.create_save_group()
        layout.addWidget(save_group)
        
        # 视图同步选项
        sync_widget = self.create_sync_widget()
        layout.addWidget(sync_widget)
        
        # 添加弹簧
        layout.addStretch()
        
        self.setLayout(layout)
        
    def create_file_group(self) -> QGroupBox:
        """创建文件操作组"""
        group = QGroupBox("文件操作")
        layout = QVBoxLayout()
        
        # 加载DICOM按钮
        self.load_btn = QPushButton("加载DICOM")
        self.load_btn.clicked.connect(self.load_dicom_clicked.emit)
        layout.addWidget(self.load_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("重置为原始图像")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)
        
        group.setLayout(layout)
        return group
        
    def create_window_group(self) -> QGroupBox:
        """创建窗宽窗位调节组"""
        group = QGroupBox("窗宽窗位调节")
        layout = QVBoxLayout()
        
        # 反相复选框
        invert_layout = QHBoxLayout()
        self.invert_checkbox = QCheckBox("反相显示")
        self.invert_checkbox.setChecked(False)  # 默认不反相
        self.invert_checkbox.stateChanged.connect(self.on_invert_changed)
        invert_layout.addWidget(self.invert_checkbox)
        invert_layout.addStretch()  # 添加弹性空间
        layout.addLayout(invert_layout)

        # 预设模式按钮
        preset_layout = QHBoxLayout()
        self.auto_btn = QPushButton("自动优化")
        self.auto_btn.clicked.connect(self.auto_window)
        self.defect_btn = QPushButton("缺陷检测")
        self.defect_btn.clicked.connect(lambda: self.apply_preset('defect'))
        self.overview_btn = QPushButton("整体观察")
        self.overview_btn.clicked.connect(lambda: self.apply_preset('overview'))
        preset_layout.addWidget(self.auto_btn)
        preset_layout.addWidget(self.defect_btn)
        preset_layout.addWidget(self.overview_btn)
        layout.addLayout(preset_layout)
        
        # 窗宽滑块
        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("窗宽:"))
        self.ww_slider = QSlider(Qt.Orientation.Horizontal)
        self.ww_slider.setRange(1, 65535)
        self.ww_slider.setValue(1000)
        self.ww_slider.valueChanged.connect(self.on_window_width_changed)
        ww_layout.addWidget(self.ww_slider)

        # 窗宽数值输入框（带上下按钮）
        self.ww_spinbox = QSpinBox()
        self.ww_spinbox.setRange(1, 65535)
        self.ww_spinbox.setValue(1000)
        self.ww_spinbox.setMinimumWidth(80)
        self.ww_spinbox.valueChanged.connect(self.on_ww_spinbox_changed)
        # 加速设置
        self.ww_spinbox.setAccelerated(True)  # 启用长按加速
        self.ww_spinbox.setSingleStep(1)      # 单次步进值
        self.ww_spinbox.setKeyboardTracking(False)  # 减少实时触发
        ww_layout.addWidget(self.ww_spinbox)
        layout.addLayout(ww_layout)
        
        # 窗位滑块
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("窗位:"))
        self.wl_slider = QSlider(Qt.Orientation.Horizontal)
        self.wl_slider.setRange(0, 65535)
        self.wl_slider.setValue(32768)
        self.wl_slider.valueChanged.connect(self.on_window_level_changed)
        wl_layout.addWidget(self.wl_slider)

        # 窗位数值输入框（带上下按钮）
        self.wl_spinbox = QSpinBox()
        self.wl_spinbox.setRange(0, 65535)
        self.wl_spinbox.setValue(32768)
        self.wl_spinbox.setMinimumWidth(80)
        self.wl_spinbox.valueChanged.connect(self.on_wl_spinbox_changed)
        # 加速设置  
        self.wl_spinbox.setAccelerated(True)  # 启用长按加速
        self.wl_spinbox.setSingleStep(1)      # 单次步进值
        self.wl_spinbox.setKeyboardTracking(False)  # 减少实时触发
        wl_layout.addWidget(self.wl_spinbox)
        layout.addLayout(wl_layout)
        
        # 数据信息显示
        self.range_label = QLabel("数据范围: 0 - 65535")
        layout.addWidget(self.range_label)
        
        group.setLayout(layout)
        return group
        
    def create_algorithm_group(self) -> QGroupBox:
        """创建数字影像增强6步骤算法组"""
        group = QGroupBox("数字影像增强6步骤")
        layout = QVBoxLayout()

        # 创建滚动区域
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # 1️⃣ 灰度变换
        step1_group = self.create_step1_gray_transform_group()
        scroll_layout.addWidget(step1_group)

        # 2️⃣ 直方图调整
        step2_group = self.create_step2_histogram_group()
        scroll_layout.addWidget(step2_group)

        # 3️⃣ 空间域滤波
        step3_group = self.create_step3_spatial_filter_group()
        scroll_layout.addWidget(step3_group)

        # 4️⃣ 频域增强
        step4_group = self.create_step4_frequency_group()
        scroll_layout.addWidget(step4_group)

        # 5️⃣ 边缘检测
        step5_group = self.create_step5_edge_detection_group()
        scroll_layout.addWidget(step5_group)

        # 6️⃣ 形态学操作
        step6_group = self.create_step6_morphology_group()
        scroll_layout.addWidget(step6_group)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)
        group.setLayout(layout)
        return group
        
    def create_step1_gray_transform_group(self) -> QGroupBox:
        """创建第1步：灰度变换组"""
        group = QGroupBox("1️⃣ 灰度变换")
        layout = QVBoxLayout()
        
        # Gamma校正
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(1)
        gamma_layout.addWidget(self.gamma_spin)
        
        self.gamma_btn = QPushButton("应用")
        self.gamma_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'gamma_correction', {'gamma': self.gamma_spin.value()}))
        self.gamma_btn.setEnabled(False)
        gamma_layout.addWidget(self.gamma_btn)
        
        layout.addLayout(gamma_layout)
        group.setLayout(layout)
        return group
        
    def create_step2_histogram_group(self) -> QGroupBox:
        """创建第2步：直方图调整组"""
        group = QGroupBox("2️⃣ 直方图调整")
        layout = QVBoxLayout()
        
        # 全局均衡化
        self.hist_global_btn = QPushButton("全局均衡化")
        self.hist_global_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'histogram_equalization', {'method': 'global'}))
        self.hist_global_btn.setEnabled(False)
        layout.addWidget(self.hist_global_btn)
        
        # CLAHE
        clahe_layout = QHBoxLayout()
        clahe_layout.addWidget(QLabel("Clip Limit:"))
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.01, 0.1)
        self.clahe_spin.setValue(0.03)
        self.clahe_spin.setSingleStep(0.01)
        self.clahe_spin.setDecimals(2)
        clahe_layout.addWidget(self.clahe_spin)
        
        self.clahe_btn = QPushButton("应用CLAHE")
        self.clahe_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'histogram_equalization', {'method': 'adaptive'}))
        self.clahe_btn.setEnabled(False)
        clahe_layout.addWidget(self.clahe_btn)
        
        layout.addLayout(clahe_layout)
        group.setLayout(layout)
        return group
        
    def create_step3_spatial_filter_group(self) -> QGroupBox:
        """创建第3步：空间域滤波组"""
        group = QGroupBox("3️⃣ 空间域滤波")
        layout = QVBoxLayout()
        
        # 高斯滤波
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(QLabel("Sigma:"))
        self.gaussian_spin = QDoubleSpinBox()
        self.gaussian_spin.setRange(0.1, 10.0)
        self.gaussian_spin.setValue(1.0)
        self.gaussian_spin.setSingleStep(0.1)
        self.gaussian_spin.setDecimals(1)
        gaussian_layout.addWidget(self.gaussian_spin)
        
        self.gaussian_btn = QPushButton("高斯滤波")
        self.gaussian_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'gaussian_filter', {'sigma': self.gaussian_spin.value()}))
        self.gaussian_btn.setEnabled(False)
        gaussian_layout.addWidget(self.gaussian_btn)
        
        layout.addLayout(gaussian_layout)
        
        # 中值滤波
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("Size:"))
        self.median_spin = QSpinBox()
        self.median_spin.setRange(1, 10)
        self.median_spin.setValue(3)
        median_layout.addWidget(self.median_spin)
        
        self.median_btn = QPushButton("中值滤波")
        self.median_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'median_filter', {'disk_size': self.median_spin.value()}))
        self.median_btn.setEnabled(False)
        median_layout.addWidget(self.median_btn)
        
        layout.addLayout(median_layout)
        
        # 非锐化掩模
        usm_layout = QVBoxLayout()
        
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        self.usm_radius_spin = QDoubleSpinBox()
        self.usm_radius_spin.setRange(0.1, 5.0)
        self.usm_radius_spin.setValue(1.0)
        self.usm_radius_spin.setSingleStep(0.1)
        self.usm_radius_spin.setDecimals(1)
        radius_layout.addWidget(self.usm_radius_spin)
        usm_layout.addLayout(radius_layout)
        
        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("Amount:"))
        self.usm_amount_spin = QDoubleSpinBox()
        self.usm_amount_spin.setRange(0.0, 3.0)
        self.usm_amount_spin.setValue(1.0)
        self.usm_amount_spin.setSingleStep(0.1)
        self.usm_amount_spin.setDecimals(1)
        amount_layout.addWidget(self.usm_amount_spin)
        usm_layout.addLayout(amount_layout)
        
        self.usm_btn = QPushButton("非锐化掩模")
        self.usm_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'unsharp_mask', {
                'radius': self.usm_radius_spin.value(),
                'amount': self.usm_amount_spin.value()
            }))
        self.usm_btn.setEnabled(False)
        usm_layout.addWidget(self.usm_btn)
        
        layout.addLayout(usm_layout)
        group.setLayout(layout)
        return group

    def create_step4_frequency_group(self) -> QGroupBox:
        """创建第4步：频域增强组"""
        group = QGroupBox("4️⃣ 频域增强")
        layout = QVBoxLayout()

        # 滤波器类型选择
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("滤波器类型:"))
        self.frequency_filter_combo = QComboBox()
        self.frequency_filter_combo.addItems([
            "理想低通滤波", "高斯低通滤波",
            "理想高通滤波", "高斯高通滤波"
        ])
        filter_layout.addWidget(self.frequency_filter_combo)
        layout.addLayout(filter_layout)

        # 截止频率
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("截止频率:"))
        self.frequency_cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.frequency_cutoff_slider.setRange(1, 50)  # 0.01-0.5 * 100
        self.frequency_cutoff_slider.setValue(10)  # 默认0.1
        cutoff_layout.addWidget(self.frequency_cutoff_slider)

        self.frequency_cutoff_spinbox = QSpinBox()
        self.frequency_cutoff_spinbox.setRange(1, 50)
        self.frequency_cutoff_spinbox.setValue(10)
        self.frequency_cutoff_spinbox.setSuffix("%")
        cutoff_layout.addWidget(self.frequency_cutoff_spinbox)
        layout.addLayout(cutoff_layout)

        # 同步滑块和数值框
        self.frequency_cutoff_slider.valueChanged.connect(
            lambda v: self.frequency_cutoff_spinbox.setValue(v))
        self.frequency_cutoff_spinbox.valueChanged.connect(
            lambda v: self.frequency_cutoff_slider.setValue(v))

        # 应用按钮
        self.frequency_apply_btn = QPushButton("应用频域滤波")
        self.frequency_apply_btn.clicked.connect(self.apply_frequency_filter)
        layout.addWidget(self.frequency_apply_btn)

        group.setLayout(layout)
        return group

    def create_step5_edge_detection_group(self) -> QGroupBox:
        """创建第5步：边缘检测组"""
        group = QGroupBox("5️⃣ 边缘检测")
        layout = QVBoxLayout()

        # 检测算子选择
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("检测算子:"))
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems([
            "Sobel边缘检测", "Canny边缘检测",
            "Laplacian边缘检测", "边缘增强", "Roberts边缘检测"
        ])
        edge_layout.addWidget(self.edge_method_combo)
        layout.addLayout(edge_layout)

        # 参数调节（根据选择的算法动态显示）
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("参数:"))
        self.edge_param_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_param_slider.setRange(1, 30)  # 0.1-3.0 * 10
        self.edge_param_slider.setValue(10)  # 默认1.0
        param_layout.addWidget(self.edge_param_slider)

        self.edge_param_spinbox = QSpinBox()
        self.edge_param_spinbox.setRange(1, 30)
        self.edge_param_spinbox.setValue(10)
        param_layout.addWidget(self.edge_param_spinbox)
        layout.addLayout(param_layout)

        # 同步滑块和数值框
        self.edge_param_slider.valueChanged.connect(
            lambda v: self.edge_param_spinbox.setValue(v))
        self.edge_param_spinbox.valueChanged.connect(
            lambda v: self.edge_param_slider.setValue(v))

        # 应用按钮
        self.edge_apply_btn = QPushButton("应用边缘检测")
        self.edge_apply_btn.clicked.connect(self.apply_edge_detection)
        layout.addWidget(self.edge_apply_btn)

        group.setLayout(layout)
        return group

    def create_step6_morphology_group(self) -> QGroupBox:
        """创建第6步：形态学操作组"""
        group = QGroupBox("6️⃣ 形态学操作")
        layout = QVBoxLayout()
        
        # 操作类型选择
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("操作:"))
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(['erosion', 'dilation', 'opening', 'closing'])
        op_layout.addWidget(self.morph_combo)
        layout.addLayout(op_layout)
        
        # 结构元素大小
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.morph_size_spin = QSpinBox()
        self.morph_size_spin.setRange(1, 10)
        self.morph_size_spin.setValue(3)
        size_layout.addWidget(self.morph_size_spin)
        layout.addLayout(size_layout)
        
        # 应用按钮
        self.morph_btn = QPushButton("应用形态学操作")
        self.morph_btn.clicked.connect(lambda: self.apply_algorithm.emit(
            'morphological_operation', {
                'operation': self.morph_combo.currentText(),
                'disk_size': self.morph_size_spin.value()
            }))
        self.morph_btn.setEnabled(False)
        layout.addWidget(self.morph_btn)
        
        group.setLayout(layout)
        return group
        
    def create_history_group(self) -> QGroupBox:
        """创建历史记录组"""
        group = QGroupBox("处理历史")
        layout = QVBoxLayout()
        
        # 历史记录标签
        self.history_label = QLabel("暂无处理记录")
        self.history_label.setWordWrap(True)
        layout.addWidget(self.history_label)
        
        group.setLayout(layout)
        return group
        
    def create_save_group(self) -> QGroupBox:
        """创建保存选项组"""
        group = QGroupBox("保存选项")
        layout = QVBoxLayout()
        
        # 保存当前结果
        self.save_current_btn = QPushButton("保存此步结果(DICOM)")
        self.save_current_btn.clicked.connect(self.save_current_clicked.emit)
        self.save_current_btn.setEnabled(False)
        layout.addWidget(self.save_current_btn)
        
        # 保存预览图
        self.save_preview_btn = QPushButton("保存预览图(PNG)")
        self.save_preview_btn.clicked.connect(self.save_preview_clicked.emit)
        self.save_preview_btn.setEnabled(False)
        layout.addWidget(self.save_preview_btn)
        
        group.setLayout(layout)
        return group
        
    def create_sync_widget(self) -> QWidget:
        """创建同步控件"""
        widget = QWidget()
        layout = QHBoxLayout()
        
        self.sync_checkbox = QCheckBox("同步视图")
        self.sync_checkbox.toggled.connect(self.sync_view_toggled.emit)
        layout.addWidget(self.sync_checkbox)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def on_window_width_changed(self, value):
        """窗宽滑块改变事件"""
        # 同步更新数值输入框，但不触发其事件
        self.ww_spinbox.blockSignals(True)
        self.ww_spinbox.setValue(value)
        self.ww_spinbox.blockSignals(False)

        self.window_width_changed.emit(float(value))

    def on_window_level_changed(self, value):
        """窗位滑块改变事件"""
        # 同步更新数值输入框，但不触发其事件
        self.wl_spinbox.blockSignals(True)
        self.wl_spinbox.setValue(value)
        self.wl_spinbox.blockSignals(False)

        self.window_level_changed.emit(float(value))

    def on_ww_spinbox_changed(self, value):
        """窗宽数值输入框改变事件"""
        # 同步更新滑块，但不触发其事件
        self.ww_slider.blockSignals(True)
        self.ww_slider.setValue(value)
        self.ww_slider.blockSignals(False)

        self.window_width_changed.emit(float(value))

    def on_wl_spinbox_changed(self, value):
        """窗位数值输入框改变事件"""
        # 同步更新滑块，但不触发其事件
        self.wl_slider.blockSignals(True)
        self.wl_slider.setValue(value)
        self.wl_slider.blockSignals(False)

        self.window_level_changed.emit(float(value))
        
    def set_controls_enabled(self, enabled: bool):
        """设置控件启用状态"""
        self.reset_btn.setEnabled(enabled)
        self.gamma_btn.setEnabled(enabled)
        self.hist_global_btn.setEnabled(enabled)
        self.clahe_btn.setEnabled(enabled)
        self.gaussian_btn.setEnabled(enabled)
        self.median_btn.setEnabled(enabled)
        self.usm_btn.setEnabled(enabled)
        self.morph_btn.setEnabled(enabled)
        self.save_current_btn.setEnabled(enabled)
        self.save_preview_btn.setEnabled(enabled)
        
        # 启用窗宽窗位控件
        self.auto_btn.setEnabled(enabled)
        self.defect_btn.setEnabled(enabled)
        self.overview_btn.setEnabled(enabled)
        self.ww_slider.setEnabled(enabled)
        self.wl_slider.setEnabled(enabled)
        
    def update_history(self, history: list):
        """更新历史记录显示"""
        if not history:
            self.history_label.setText("暂无处理记录")
        else:
            text = "处理历史:\\n"
            for i, record in enumerate(history[-5:], 1):  # 显示最近5条
                text += f"{i}. {record['algorithm']}\\n"
            self.history_label.setText(text)
            
    def get_window_settings(self) -> tuple:
        """获取窗宽窗位设置"""
        return (self.ww_slider.value(), self.wl_slider.value())
        
    def set_window_settings(self, window_width: float, window_level: float):
        """设置窗宽窗位"""
        self.ww_slider.setValue(int(window_width))
        self.wl_slider.setValue(int(window_level))

        # 同步更新数值输入框
        self.ww_spinbox.setValue(int(window_width))
        self.wl_spinbox.setValue(int(window_level))
    
    def auto_window(self):
        """自动优化窗宽窗位"""
        # 发送信号请求自动优化
        self.window_auto_requested.emit()
    
    def apply_preset(self, preset_type: str):
        """应用预设窗宽窗位"""
        if preset_type == 'defect':
            # 缺陷检测模式：窄窗宽，高对比度
            self.ww_slider.setValue(500)
            self.wl_slider.setValue(3000)
        elif preset_type == 'overview':
            # 整体观察模式：宽窗宽，显示全范围
            self.ww_slider.setValue(60000)
            self.wl_slider.setValue(32768)
    
    def update_image_info(self, data_min: int, data_max: int, data_mean: float):
        """更新图像信息显示"""
        self.range_label.setText(f"数据范围: {data_min} - {data_max} (均值: {data_mean:.1f})")
    
    def set_controls_enabled(self, enabled: bool):
        """设置控件启用状态"""
        self.reset_btn.setEnabled(enabled)
        self.gamma_btn.setEnabled(enabled)
        self.hist_global_btn.setEnabled(enabled)
        self.clahe_btn.setEnabled(enabled)
        self.gaussian_btn.setEnabled(enabled)
        self.median_btn.setEnabled(enabled)
        self.usm_btn.setEnabled(enabled)
        self.morph_btn.setEnabled(enabled)
        self.save_current_btn.setEnabled(enabled)
        self.save_preview_btn.setEnabled(enabled)
        
        # 启用窗宽窗位控件
        self.auto_btn.setEnabled(enabled)
        self.defect_btn.setEnabled(enabled)
        self.overview_btn.setEnabled(enabled)
        self.ww_slider.setEnabled(enabled)
        self.wl_slider.setEnabled(enabled)
        self.ww_spinbox.setEnabled(enabled)
        self.wl_spinbox.setEnabled(enabled)
        self.invert_checkbox.setEnabled(enabled)

    def on_invert_changed(self, state):
        """反相状态改变处理"""
        is_inverted = state == Qt.CheckState.Checked.value
        self.invert_changed.emit(is_inverted)

    def get_invert_state(self) -> bool:
        """获取当前反相状态"""
        return self.invert_checkbox.isChecked()

    def update_slider_ranges(self, ww_range: tuple, wl_range: tuple):
        """更新滑块范围

        Args:
            ww_range: (ww_min, ww_max) 窗宽范围
            wl_range: (wl_min, wl_max) 窗位范围
        """
        ww_min, ww_max = ww_range
        wl_min, wl_max = wl_range

        print(f"📊 更新滑块范围:")
        print(f"   窗宽: {ww_min} - {ww_max}")
        print(f"   窗位: {wl_min} - {wl_max}")

        # 暂时断开信号连接，避免触发事件
        self.ww_slider.valueChanged.disconnect()
        self.wl_slider.valueChanged.disconnect()
        self.ww_spinbox.valueChanged.disconnect()
        self.wl_spinbox.valueChanged.disconnect()

        # 更新滑块和数值输入框范围
        self.ww_slider.setRange(ww_min, ww_max)
        self.wl_slider.setRange(wl_min, wl_max)
        self.ww_spinbox.setRange(ww_min, ww_max)
        self.wl_spinbox.setRange(wl_min, wl_max)

        # 重新连接信号
        self.ww_slider.valueChanged.connect(self.on_window_width_changed)
        self.wl_slider.valueChanged.connect(self.on_window_level_changed)
        self.ww_spinbox.valueChanged.connect(self.on_ww_spinbox_changed)
        self.wl_spinbox.valueChanged.connect(self.on_wl_spinbox_changed)

        # 不再在标签中显示范围，为输入框腾出空间
        # 范围信息在控制台日志中可以看到

    def apply_frequency_filter(self):
        """应用频域滤波"""
        filter_type = self.frequency_filter_combo.currentText()
        cutoff_ratio = self.frequency_cutoff_spinbox.value() / 100.0  # 转换为0.01-0.5

        # 映射UI选择到算法名称
        algorithm_map = {
            "理想低通滤波": "ideal_low_pass_filter",
            "高斯低通滤波": "gaussian_low_pass_filter",
            "理想高通滤波": "ideal_high_pass_filter",
            "高斯高通滤波": "gaussian_high_pass_filter"
        }

        algorithm = algorithm_map.get(filter_type, "ideal_low_pass_filter")
        parameters = {"cutoff_ratio": cutoff_ratio}

        print(f"🔄 应用频域滤波: {filter_type}, 截止频率: {cutoff_ratio:.2f}")
        self.algorithm_applied.emit(algorithm, parameters)

    def apply_edge_detection(self):
        """应用边缘检测"""
        method = self.edge_method_combo.currentText()
        param_value = self.edge_param_spinbox.value() / 10.0  # 转换为0.1-3.0

        # 映射UI选择到算法名称
        algorithm_map = {
            "Sobel边缘检测": "sobel_edge_detection",
            "Canny边缘检测": "canny_edge_detection",
            "Laplacian边缘检测": "laplacian_edge_detection",
            "边缘增强": "edge_enhancement",
            "Roberts边缘检测": "roberts_edge_detection"
        }

        algorithm = algorithm_map.get(method, "sobel_edge_detection")

        # 根据不同算法设置参数
        if algorithm == "canny_edge_detection":
            parameters = {"sigma": param_value}
        elif algorithm == "edge_enhancement":
            parameters = {"edge_strength": param_value}
        else:
            parameters = {}

        print(f"🔄 应用边缘检测: {method}, 参数: {param_value:.1f}")
        self.algorithm_applied.emit(algorithm, parameters)