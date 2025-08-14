#!/usr/bin/env python3
"""
直方图显示窗口
用于显示图像的灰度直方图，帮助用户理解图像的灰度分布特征
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QCheckBox, QSpinBox, QGroupBox)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')

class HistogramWindow(QDialog):
    """直方图显示窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像直方图分析")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        # 数据存储
        self.current_image = None
        self.original_image = None

        # 性能优化：缓存系统
        self.histogram_cache = {}  # 缓存直方图计算结果
        self.stats_cache = {}      # 缓存统计信息
        self.last_image_hash = {}  # 图像数据哈希，用于检测变化

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # matplotlib高性能优化（针对大数据量直方图）
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 0.001  # 极激进简化
        plt.rcParams['agg.path.chunksize'] = 100000      # 超大块处理
        plt.rcParams['figure.max_open_warning'] = 0      # 禁用警告
        plt.rcParams['axes.formatter.useoffset'] = False # 禁用科学计数法
        plt.rcParams['figure.dpi'] = 80                  # 降低DPI提升性能
        plt.rcParams['savefig.dpi'] = 150                # 保存时使用合理DPI

        # 绘制缓存
        self.plot_cache = {}  # 缓存绘制结果

        self.setup_ui()

    def _get_image_hash(self, image):
        """计算图像数据的快速哈希值，用于缓存检测"""
        if image is None:
            return None
        # 使用图像的形状、数据类型、均值和标准差作为快速哈希
        # 这比计算完整哈希快得多，但足够检测图像变化
        return hash((image.shape, image.dtype.name,
                    float(np.mean(image)), float(np.std(image))))

    def _compute_histogram_fast(self, image, bins, data_range):
        """高性能直方图计算（学习C#实现），带缓存"""
        if image is None:
            return None, None

        # 生成缓存键
        image_hash = self._get_image_hash(image)
        cache_key = (image_hash, bins, data_range)

        # 检查缓存
        if cache_key in self.histogram_cache:
            return self.histogram_cache[cache_key]

        # 准备数据
        flat_data = image.flatten()

        # 高性能直方图计算（类似C# Cv2.CalcHist）
        try:
            import cv2
            # 使用OpenCV计算直方图（最快）
            flat_data_uint16 = flat_data.astype(np.uint16)
            hist_range = [int(data_range[0]), int(data_range[1])]

            # OpenCV直方图计算
            hist_values = cv2.calcHist([flat_data_uint16.reshape(-1, 1)], [0], None,
                                     [bins], hist_range).flatten()

            # 生成bin centers
            bin_centers = np.linspace(data_range[0], data_range[1], bins, endpoint=False)

        except ImportError:
            # 回退到优化的numpy实现
            # 使用numpy的bincount（比histogram更快）
            if data_range == (0, 65535) and bins == 65536:
                # 特殊优化：16位全范围直方图
                hist_values = np.bincount(flat_data.astype(np.uint16), minlength=65536)
                bin_centers = np.arange(65536)
            else:
                # 通用情况
                hist_values, bin_edges = np.histogram(flat_data, bins=bins, range=data_range)
                bin_centers = bin_edges[:-1]

        # 缓存结果
        result = (bin_centers, hist_values)
        self.histogram_cache[cache_key] = result

        # 限制缓存大小
        if len(self.histogram_cache) > 10:
            oldest_key = next(iter(self.histogram_cache))
            del self.histogram_cache[oldest_key]

        return result

    def _compute_stats_fast(self, image):
        """高性能统计计算，带缓存"""
        if image is None:
            return None

        image_hash = self._get_image_hash(image)

        # 检查缓存
        if image_hash in self.stats_cache:
            return self.stats_cache[image_hash]

        # 一次性计算所有统计量（向量化操作）
        flat_data = image.flatten()
        stats = {
            'mean': float(np.mean(flat_data)),
            'std': float(np.std(flat_data)),
            'min': int(np.min(flat_data)),
            'max': int(np.max(flat_data)),
            'median': float(np.median(flat_data)),
            'size': flat_data.size
        }

        # 16位图像的特殊统计
        if image.dtype == np.uint16:
            overexp_threshold = 60000
            overexp_mask = flat_data >= overexp_threshold
            stats['overexposed_count'] = int(np.sum(overexp_mask))
            stats['overexposed_ratio'] = float(stats['overexposed_count'] / stats['size'] * 100)

            # 有效数据统计
            if np.any(~overexp_mask):
                valid_data = flat_data[~overexp_mask]
                stats['valid_mean'] = float(np.mean(valid_data))
                stats['valid_std'] = float(np.std(valid_data))
                stats['valid_min'] = int(np.min(valid_data))
                stats['valid_max'] = int(np.max(valid_data))
            else:
                stats['valid_mean'] = stats['mean']
                stats['valid_std'] = stats['std']
                stats['valid_min'] = stats['min']
                stats['valid_max'] = stats['max']

        # 缓存结果
        self.stats_cache[image_hash] = stats

        # 限制缓存大小
        if len(self.stats_cache) > 10:
            oldest_key = next(iter(self.stats_cache))
            del self.stats_cache[oldest_key]

        return stats



    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout()
        
        # 标题和说明
        title_label = QLabel("📊 图像直方图分析")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)
        
        info_label = QLabel(
            "直方图显示图像中每个灰度级的像素数量分布：\n"
            "• 左侧集中：图像偏暗  • 右侧集中：图像偏亮  • 中间窄范围：对比度低"
        )
        info_label.setStyleSheet("color: #7f8c8d; margin: 5px 10px;")
        layout.addWidget(info_label)
        
        # 控制面板
        control_group = QGroupBox("显示控制")
        control_layout = QHBoxLayout()
        
        # 显示选项
        self.show_original_cb = QCheckBox("显示原始图像")
        self.show_original_cb.setChecked(True)
        self.show_original_cb.stateChanged.connect(self.update_histogram)
        control_layout.addWidget(self.show_original_cb)
        
        self.show_current_cb = QCheckBox("显示当前图像")
        self.show_current_cb.setChecked(True)
        self.show_current_cb.stateChanged.connect(self.update_histogram)
        control_layout.addWidget(self.show_current_cb)
        
        # 分箱数量（16位全精度）
        control_layout.addWidget(QLabel("分箱数量:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(256, 65536)  # 恢复16位全范围
        self.bins_spinbox.setValue(65536)  # 默认16位全精度
        self.bins_spinbox.valueChanged.connect(self.update_histogram)
        control_layout.addWidget(self.bins_spinbox)

        # Y轴显示模式
        self.log_scale_cb = QCheckBox("对数坐标(Y轴)")
        self.log_scale_cb.setChecked(True)  # 默认使用对数坐标（商业软件标准）
        self.log_scale_cb.stateChanged.connect(self.update_histogram)
        control_layout.addWidget(self.log_scale_cb)

        # 智能范围显示（商业软件标配）
        self.smart_range_cb = QCheckBox("智能范围显示")
        self.smart_range_cb.setChecked(True)
        self.smart_range_cb.stateChanged.connect(self.update_histogram)
        control_layout.addWidget(self.smart_range_cb)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 创建matplotlib图表（分列显示）
        self.figure = Figure(figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(
            "background-color: #ecf0f1; padding: 10px; border-radius: 5px; "
            "font-family: 'Consolas', 'Monaco', monospace;"
        )
        layout.addWidget(self.stats_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.update_histogram)
        button_layout.addWidget(refresh_btn)
        
        save_btn = QPushButton("💾 保存图表")
        save_btn.clicked.connect(self.save_histogram)
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def set_images(self, current_image, original_image=None):
        """设置要分析的图像"""
        # 检查图像是否真的改变了
        current_hash = self._get_image_hash(current_image)
        original_hash = self._get_image_hash(original_image)

        current_changed = current_hash != self.last_image_hash.get('current')
        original_changed = original_hash != self.last_image_hash.get('original')

        if not current_changed and not original_changed:
            return

        # 更新图像数据
        self.current_image = current_image
        if original_image is not None:
            self.original_image = original_image
        else:
            self.original_image = current_image.copy() if current_image is not None else None

        # 更新哈希记录
        self.last_image_hash['current'] = current_hash
        self.last_image_hash['original'] = original_hash

        self.update_histogram()
        
    def update_histogram(self):
        """更新直方图显示 - 高性能版本，保持65536分箱精度"""
        if self.current_image is None:
            return

        # 检查是否需要重新绘制
        current_hash = self._get_image_hash(self.current_image)
        original_hash = self._get_image_hash(self.original_image)

        display_settings = (
            self.bins_spinbox.value(),
            self.log_scale_cb.isChecked(),
            self.smart_range_cb.isChecked(),
            self.show_original_cb.isChecked(),
            self.show_current_cb.isChecked()
        )

        cache_key = (current_hash, original_hash, display_settings)

        # 检查绘制缓存
        if cache_key in self.plot_cache:
            return

        # 清除之前的图表
        self.figure.clear()

        bins = self.bins_spinbox.value()
        use_log_scale = self.log_scale_cb.isChecked()
        use_smart_range = self.smart_range_cb.isChecked()

        # 计算数据范围（16位图像）
        if self.current_image.dtype == np.uint16:
            data_range = (0, 65535)
        else:
            data_range = (0, 255)

        # 确定显示的子图数量
        show_original = self.show_original_cb.isChecked() and self.original_image is not None
        show_current = self.show_current_cb.isChecked()

        if show_original and show_current:
            # 两个子图，分列显示
            ax1 = self.figure.add_subplot(121)  # 左侧
            ax2 = self.figure.add_subplot(122)  # 右侧
            axes = [ax1, ax2]
            titles = ['原始图像直方图', '当前图像直方图']
            images = [self.original_image, self.current_image]
            colors = ['blue', 'red']
        elif show_current:
            # 只显示当前图像
            ax = self.figure.add_subplot(111)
            axes = [ax]
            titles = ['当前图像直方图']
            images = [self.current_image]
            colors = ['red']
        elif show_original:
            # 只显示原始图像
            ax = self.figure.add_subplot(111)
            axes = [ax]
            titles = ['原始图像直方图']
            images = [self.original_image]
            colors = ['blue']
        else:
            return

        # 为每个子图绘制直方图
        for ax, title, image, color in zip(axes, titles, images, colors):
            # 使用高性能缓存计算
            result = self._compute_histogram_fast(image, bins, data_range)
            if result is None:
                continue

            bin_centers, hist_values = result



            # 处理零值（对数坐标需要）
            if use_log_scale:
                hist_values = np.maximum(hist_values, 1)

            # 稀疏数据优化：只处理非零值
            nonzero_mask = hist_values > 0
            nonzero_indices = np.where(nonzero_mask)[0]

            if len(nonzero_indices) == 0:
                continue

            # 只保留非零数据点
            sparse_centers = bin_centers[nonzero_indices]
            sparse_values = hist_values[nonzero_indices]

            # 使用稀疏数据
            sampled_centers = sparse_centers
            sampled_values = sparse_values

            # 专业显示
            ax.plot(sampled_centers, sampled_values,
                   color=color, alpha=0.9, linewidth=1.2,
                   label=title.replace('直方图', ''))

            # 添加填充效果
            ax.fill_between(sampled_centers, 0, sampled_values,
                           color=color, alpha=0.3, interpolate=True)

            # Y轴刻度设置
            y_max = np.max(sampled_values)

            if use_log_scale:
                # 过滤零值
                nonzero_mask = sampled_values > 0
                if np.any(nonzero_mask):
                    y_min_nonzero = np.min(sampled_values[nonzero_mask])
                    y_max_nonzero = np.max(sampled_values[nonzero_mask])

                    # 设置对数刻度
                    ax.set_yscale('log')
                    ax.set_ylabel('像素数量 (对数)')

                    # 确保Y轴范围足够大
                    log_min = max(0.5, y_min_nonzero * 0.1)
                    log_max = y_max_nonzero * 10
                    ax.set_ylim(bottom=log_min, top=log_max)
                else:
                    ax.set_yscale('linear')
                    ax.set_ylabel('像素数量')
                    ax.set_ylim(0, y_max * 1.05)
            else:
                ax.set_yscale('linear')
                ax.set_ylabel('像素数量')
                ax.set_ylim(0, y_max * 1.05)

            # 设置图表属性
            ax.set_xlabel('灰度值')
            ax.set_title(f'{title} ({bins}分箱)')
            ax.grid(True, alpha=0.3)

            # 使用缓存的统计信息设置范围和标记
            stats = self._compute_stats_fast(image)
            if stats:
                # 设置x轴范围（智能范围优化）
                if use_smart_range and image.dtype == np.uint16:
                    if stats['max'] > 50000:  # 检测过曝
                        # 使用统计信息，避免重复计算百分位数
                        display_max = min(stats['max'], 65535)
                        ax.set_xlim(max(0, stats['min']), display_max)

                        # 添加过曝阈值线
                        if stats['overexposed_count'] > 0:
                            overexp_threshold = 60000  # 使用固定阈值，避免重复计算
                            ax.axvline(overexp_threshold, color='orange', linestyle=':',
                                     alpha=0.6, linewidth=1)
                            ax.text(overexp_threshold, ax.get_ylim()[1] * 0.9,
                                   f'过曝阈值', rotation=90, fontsize=8, color='orange')
                    else:
                        ax.set_xlim(stats['min'], stats['max'])
                else:
                    # 全范围显示
                    if image.dtype == np.uint16:
                        ax.set_xlim(0, 65535)
                    else:
                        ax.set_xlim(0, 255)

                # 添加统计线（使用缓存的统计信息）
                y_max = ax.get_ylim()[1]
                line_height = y_max * 0.8 if use_log_scale else y_max * 0.9

                ax.axvline(stats['mean'], color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax.axvline(stats['median'], color='green', linestyle='--', alpha=0.7, linewidth=1)

                # 添加统计文本
                ax.text(stats['mean'], line_height, f'均值\n{stats["mean"]:.0f}',
                       ha='center', va='top', color='red', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.text(stats['median'], line_height * 0.6, f'中位数\n{stats["median"]:.0f}',
                       ha='center', va='top', color='green', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 调整子图间距
        self.figure.tight_layout()

        # 绘制到画布
        self.canvas.draw()

        # 缓存绘制结果
        self.plot_cache[cache_key] = True

        # 限制缓存大小
        if len(self.plot_cache) > 5:
            oldest_key = next(iter(self.plot_cache))
            del self.plot_cache[oldest_key]

        # 更新统计信息（使用缓存）
        self.update_statistics()
        
    def update_statistics(self):
        """更新统计信息显示 - 使用缓存的高性能版本"""
        if self.current_image is None:
            return

        # 计算统计信息
        stats_text = f"📊 图像统计信息 (分箱数: {self.bins_spinbox.value()}):\n"

        if self.original_image is not None and self.show_original_cb.isChecked():
            # 使用缓存的统计信息
            orig_stats = self._compute_stats_fast(self.original_image)
            if orig_stats:
                stats_text += f"原始图像: 均值={orig_stats['mean']:.0f}, 标准差={orig_stats['std']:.0f}, 范围=[{orig_stats['min']}-{orig_stats['max']}]\n"

        if self.show_current_cb.isChecked():
            # 使用缓存的统计信息
            curr_stats = self._compute_stats_fast(self.current_image)
            if curr_stats:
                stats_text += f"当前图像: 均值={curr_stats['mean']:.0f}, 标准差={curr_stats['std']:.0f}, 中位数={curr_stats['median']:.0f}\n"
                stats_text += f"         范围=[{curr_stats['min']}-{curr_stats['max']}], 总像素={curr_stats['size']:,}\n"

                # 计算动态范围利用率
                if self.current_image.dtype == np.uint16:
                    utilization = (curr_stats['max'] - curr_stats['min']) / 65535 * 100
                    max_possible = 65535
                else:
                    utilization = (curr_stats['max'] - curr_stats['min']) / 255 * 100
                    max_possible = 255

                stats_text += f"动态范围利用率: {utilization:.1f}% ({curr_stats['max'] - curr_stats['min'] + 1}/{max_possible + 1}个灰度级)\n"

                # 工业X射线图像分析（使用缓存的结果）
                if self.current_image.dtype == np.uint16 and 'overexposed_count' in curr_stats:
                    # 背景过曝分析
                    background_threshold = 45000
                    background_pixels = np.sum(self.current_image >= background_threshold)
                    background_ratio = background_pixels / curr_stats['size'] * 100

                    if background_ratio > 30:
                        stats_text += f"🏭 工业X射线: 背景区域(≥{background_threshold})={background_pixels:,}像素 ({background_ratio:.1f}%)\n"

                    # 真正过曝分析
                    if curr_stats['overexposed_count'] > 0:
                        stats_text += f"⚠️  真正过曝: {curr_stats['overexposed_count']:,}像素 ({curr_stats['overexposed_ratio']:.2f}%), 阈值≥60000\n"

                        # 工件数据分析（排除背景和过曝）
                        if 'valid_mean' in curr_stats:
                            stats_text += f"📊 工件数据: 均值={curr_stats['valid_mean']:.0f}, 标准差={curr_stats['valid_std']:.0f}, 范围=[{curr_stats['valid_min']}-{curr_stats['valid_max']}]\n"

                # 图像特征分析
                mid_range = (curr_stats['max'] + curr_stats['min']) / 2
                if curr_stats['mean'] < mid_range - (curr_stats['max'] - curr_stats['min']) * 0.2:
                    stats_text += "🔍 亮度特征: 偏暗，建议Gamma校正(γ=0.6-0.8)提亮\n"
                elif curr_stats['mean'] > mid_range + (curr_stats['max'] - curr_stats['min']) * 0.2:
                    stats_text += "🔍 亮度特征: 偏亮，建议Gamma校正(γ=1.2-1.5)压缩\n"
                else:
                    stats_text += "🔍 亮度特征: 分布适中\n"

                # 对比度分析
                data_range = curr_stats['max'] - curr_stats['min']
                if data_range > 0:
                    contrast_ratio = curr_stats['std'] / data_range
                    if contrast_ratio < 0.15:
                        stats_text += "📈 对比度: 较低，建议直方图均衡化\n"
                    elif contrast_ratio > 0.4:
                        stats_text += "📈 对比度: 很高，可能需要平滑处理\n"
                    else:
                        stats_text += "📈 对比度: 良好，适合缺陷检测\n"

                # 性能信息
                bins_count = self.bins_spinbox.value()
                cache_hits = len(self.histogram_cache)
                stats_text += f"⚡ 性能信息: 分箱数={bins_count}, 缓存命中={cache_hits}项\n"

                if bins_count == 65536:
                    stats_text += "🎯 精度模式: 16位全精度分析，每个灰度级独立统计\n"

        self.stats_label.setText(stats_text)
        
    def save_histogram(self):
        """保存直方图到文件"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存直方图", "histogram.png", 
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')

        except Exception:
            pass
            
    def clear_cache(self):
        """清理缓存，释放内存"""
        self.histogram_cache.clear()
        self.stats_cache.clear()
        self.last_image_hash.clear()
        self.plot_cache.clear()

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 清理缓存和matplotlib资源
        self.clear_cache()
        plt.close(self.figure)
        event.accept()
