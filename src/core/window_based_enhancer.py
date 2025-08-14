"""
基于窗宽窗位的DICOM图像增强处理器
专门用于突出细小缺陷的图像增强算法
"""
import numpy as np
import cv2
from typing import Optional, Callable


class WindowBasedEnhancer:
    """基于窗宽窗位的DICOM图像增强处理器"""

    @staticmethod
    def _detect_effective_range(data: np.ndarray) -> tuple:
        """检测有效数据范围（复用自动优化算法的逻辑）

        Args:
            data: 图像数据

        Returns:
            tuple: (effective_min, effective_max)
        """
        data_min = int(data.min())
        data_max = int(data.max())
        total_pixels = data.size

        # 计算直方图
        hist, bins = np.histogram(data.flatten(), bins=65536, range=(data_min, data_max))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # 检测过曝峰值（与自动优化算法相同的逻辑）
        pixel_ratios = hist / total_pixels
        major_peaks = np.where(pixel_ratios > 0.05)[0]  # 超过5%的bins

        overexposed_peaks = []
        for peak_idx in major_peaks:
            peak_value = bin_centers[peak_idx]
            peak_ratio = pixel_ratios[peak_idx]
            # 过曝判断：灰度值 > 80%范围 且 像素数 > 5%
            if peak_value > (data_min + (data_max - data_min) * 0.8):
                overexposed_peaks.append((peak_value, peak_ratio))

        if overexposed_peaks:
            # 检测到过曝背景，使用工件检测算法
            overexposed_threshold = min(peak[0] for peak in overexposed_peaks)
            noise_threshold = total_pixels * 0.0001

            # 找到有效的工件数据区域
            valid_bins = np.where((bin_centers < overexposed_threshold) & (hist > noise_threshold))[0]

            if len(valid_bins) > 10:
                # 在有效区域内计算5%-95%
                valid_pixels = np.sum(hist[valid_bins])
                valid_cumulative = np.cumsum(hist[valid_bins])

                lower_threshold = valid_pixels * 0.05
                upper_threshold = valid_pixels * 0.95

                lower_idx = np.where(valid_cumulative >= lower_threshold)[0]
                upper_idx = np.where(valid_cumulative >= upper_threshold)[0]

                if len(lower_idx) > 0 and len(upper_idx) > 0:
                    effective_min = bin_centers[valid_bins[lower_idx[0]]]
                    effective_max = bin_centers[valid_bins[upper_idx[0]]]
                    return effective_min, effective_max

        # 回退：使用标准5%-95%算法
        cumulative_pixels = np.cumsum(hist)
        lower_bound = np.where(cumulative_pixels >= total_pixels * 0.05)[0]
        upper_bound = np.where(cumulative_pixels >= total_pixels * 0.95)[0]

        if len(lower_bound) > 0 and len(upper_bound) > 0:
            effective_min = bin_centers[lower_bound[0]]
            effective_max = bin_centers[upper_bound[0]]
        else:
            # 最终回退
            effective_min = float(data_min)
            effective_max = float(data_max)

        return effective_min, effective_max
    
    @staticmethod
    def window_based_enhance(data: np.ndarray, window_width: float, window_level: float, 
                           progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        基于窗宽窗位的缺陷检测增强算法
        
        Args:
            data: 输入图像数据（16位DICOM原始数据）
            window_width: 窗宽
            window_level: 窗位
            progress_callback: 进度回调函数
            
        Returns:
            增强后的图像数据
        """
        try:
            if progress_callback:
                progress_callback(5)

            print(f"\n🔍 窗位增强Debug日志:")
            print(f"   输入数据范围: {data.min()} - {data.max()}")
            print(f"   输入数据均值: {data.mean():.2f}")
            print(f"   输入数据标准差: {data.std():.2f}")
            print(f"   窗宽: {window_width}, 窗位: {window_level}")
            print(f"   🔧 使用全范围处理策略（避免动态范围压缩）")

            # 1. 检测感兴趣区域（用于自适应处理，但不裁剪数据）
            wl_min = window_level - window_width / 2
            wl_max = window_level + window_width / 2
            roi_mask = (data >= wl_min) & (data <= wl_max)
            roi_ratio = np.sum(roi_mask) / data.size

            print(f"   感兴趣区域: {wl_min} - {wl_max}")
            print(f"   感兴趣像素比例: {roi_ratio*100:.1f}%")

            if progress_callback:
                progress_callback(15)

            # 2. 全范围归一化（保持完整动态范围）
            data_min = float(data.min())
            data_max = float(data.max())
            img_norm = (data.astype(np.float32) - data_min) / (data_max - data_min)

            print(f"   全范围归一化: {data_min} - {data_max}")
            print(f"   归一化后范围: {img_norm.min():.4f} - {img_norm.max():.4f}")
            print(f"   归一化后均值: {img_norm.mean():.4f}")
            print(f"   归一化后标准差: {img_norm.std():.4f}")

            # 3. 创建感兴趣区域的权重图（用于自适应增强）
            roi_weight = np.zeros_like(img_norm)
            roi_weight[roi_mask] = 1.0
            # 对权重图进行高斯模糊，创建平滑过渡
            roi_weight = cv2.GaussianBlur(roi_weight, (21, 21), 7)

            print(f"   感兴趣区域权重范围: {roi_weight.min():.4f} - {roi_weight.max():.4f}")

            if progress_callback:
                progress_callback(25)

            # 4. 噪声检测（在感兴趣区域内）
            var_map = cv2.GaussianBlur(img_norm**2, (7, 7), 0) - cv2.GaussianBlur(img_norm, (7, 7), 0)**2
            var_map = np.clip(var_map / (var_map.max() + 1e-8), 0, 1)

            # 在感兴趣区域内计算噪声统计
            roi_var = var_map[roi_mask]
            noise_level = np.mean(roi_var) if len(roi_var) > 0 else np.mean(var_map)
            high_noise_ratio = np.sum(roi_var > 0.5) / len(roi_var) if len(roi_var) > 0 else 0

            print(f"   感兴趣区域噪声水平: {noise_level:.4f}")
            print(f"   感兴趣区域高噪声比例: {high_noise_ratio*100:.1f}%")

            # 5. 自适应多尺度高频增强
            blur_small = cv2.GaussianBlur(img_norm, (0, 0), 1)
            high_freq_small = img_norm - blur_small
            blur_large = cv2.GaussianBlur(img_norm, (0, 0), 5)
            high_freq_large = img_norm - blur_large

            print(f"   小尺度高频范围: {high_freq_small.min():.4f} - {high_freq_small.max():.4f}")
            print(f"   大尺度高频范围: {high_freq_large.min():.4f} - {high_freq_large.max():.4f}")

            # 根据噪声水平调整增强强度
            if noise_level > 0.1:
                base_strength_small, base_strength_large = 0.8, 0.4
                print(f"   🔧 检测到高噪声，使用保守增强")
            elif noise_level > 0.05:
                base_strength_small, base_strength_large = 1.2, 0.6
                print(f"   🔧 检测到中等噪声，使用中等增强")
            else:
                base_strength_small, base_strength_large = 1.5, 0.8
                print(f"   🔧 检测到低噪声，使用正常增强")

            # 使用权重图进行空间自适应增强
            # 感兴趣区域内：使用设定的增强强度
            # 感兴趣区域外：使用较弱的增强强度
            strength_small = base_strength_small * roi_weight + 0.3 * (1 - roi_weight)
            strength_large = base_strength_large * roi_weight + 0.2 * (1 - roi_weight)

            print(f"   空间自适应增强: ROI内={base_strength_small:.1f}/{base_strength_large:.1f}, ROI外=0.3/0.2")

            # 应用空间自适应增强
            img_detail = img_norm + strength_small * high_freq_small + strength_large * high_freq_large
            img_detail = np.clip(img_detail, 0, 1)

            print(f"   增强后范围: {img_detail.min():.4f} - {img_detail.max():.4f}")
            print(f"   增强后均值: {img_detail.mean():.4f}")

            if progress_callback:
                progress_callback(45)

            # 5. 安全的光照归一化（防止数值爆炸）
            illum = cv2.GaussianBlur(img_detail, (0, 0), 50)

            print(f"   光照归一化前: {img_detail.min():.4f} - {img_detail.max():.4f}")
            print(f"   光照图范围: {illum.min():.4f} - {illum.max():.4f}")

            # 安全的光照归一化：限制除法结果
            illum_safe = np.clip(illum, 0.1, 1.0)  # 防止除以过小的数
            img_light_norm = img_detail / illum_safe
            img_light_norm = np.clip(img_light_norm, 0, 2.0)  # 限制最大值为2倍

            print(f"   安全光照归一化后: {img_light_norm.min():.4f} - {img_light_norm.max():.4f}")

            # 弱化光照归一化效果，主要保留原图
            img_light_norm = 0.3 * img_light_norm + 0.7 * img_detail  # 降低光照归一化权重
            img_light_norm = np.clip(img_light_norm, 0, 1)

            print(f"   混合后范围: {img_light_norm.min():.4f} - {img_light_norm.max():.4f}")

            if progress_callback:
                progress_callback(65)

            # 6. Gamma曲线调整（根据噪声水平调整）
            if noise_level > 0.1:
                gamma = 0.9  # 高噪声：保守的gamma
            else:
                gamma = 0.8  # 低噪声：正常gamma

            print(f"   Gamma值: {gamma}")
            img_gamma = np.power(img_light_norm, gamma)
            print(f"   Gamma调整后: {img_gamma.min():.4f} - {img_gamma.max():.4f}")

            if progress_callback:
                progress_callback(80)

            # 7. 温和的CLAHE增强（避免过度拉伸）
            img_16bit = (img_gamma * 65535).astype(np.uint16)

            if noise_level > 0.1:
                # 高噪声：非常温和的CLAHE
                clip_limit = 1.2
                tile_size = (16, 16)
                print(f"   🔧 高噪声模式：温和CLAHE clipLimit={clip_limit}, tileSize={tile_size}")
            else:
                # 低噪声：温和CLAHE
                clip_limit = 1.5
                tile_size = (8, 8)
                print(f"   🔧 正常模式：温和CLAHE clipLimit={clip_limit}, tileSize={tile_size}")

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            img_clahe = clahe.apply(img_16bit)

            print(f"   CLAHE后范围: {img_clahe.min()} - {img_clahe.max()}")

            if progress_callback:
                progress_callback(95)

            # 8. 输出全范围数据（映射回原始数据范围）
            clahe_min = float(img_clahe.min())
            clahe_max = float(img_clahe.max())

            # 映射回原始数据的全范围
            result_float = (img_clahe.astype(np.float32) - clahe_min) / (clahe_max - clahe_min)
            result_float = result_float * (data_max - data_min) + data_min

            print(f"   CLAHE范围: {clahe_min:.0f} - {clahe_max:.0f}")
            print(f"   映射回全范围: {data_min:.0f} - {data_max:.0f}")

            # 9. 轻微混合原图（保留质感和全范围）
            alpha = 0.85  # 增强结果权重
            result = cv2.addWeighted(result_float, alpha, data.astype(np.float32), 1 - alpha, 0)
            result = np.clip(result, data_min, data_max).astype(np.uint16)

            print(f"   混合后范围: {result.min()} - {result.max()}")

            print(f"   最终输出范围: {result.min()} - {result.max()}")
            print(f"   最终输出均值: {result.mean():.2f}")
            print(f"   最终输出标准差: {result.std():.2f}")
            print(f"   ✅ 窗位增强完成\n")

            if progress_callback:
                progress_callback(100)

            return result
            
        except Exception as e:
            raise RuntimeError(f"基于窗宽窗位的增强处理失败: {str(e)}")
    
    @staticmethod
    def get_algorithm_info() -> dict:
        """获取算法信息"""
        return {
            'name': '基于窗宽窗位的平衡增强',
            'version': '2.0',
            'description': '平衡版缺陷检测增强算法，视觉效果更佳',
            'features': [
                '基于窗宽窗位的数据筛选',
                '平衡的多尺度高频增强',
                '弱化光照归一化（防止过亮）',
                'Gamma曲线调整（压亮提暗）',
                'CLAHE对比度增强',
                '原图混合（保留质感）'
            ],
            'advantages': [
                '只处理用户关心的灰度范围',
                '避免背景噪声干扰',
                '平衡的增强效果，不过度处理',
                '保留原始图像质感',
                '更好的视觉效果'
            ],
            'improvements': [
                '简化高频增强（固定强度）',
                '弱化光照归一化效果',
                '新增Gamma曲线调整',
                '新增原图混合保质感'
            ]
        }
    
    @staticmethod
    def _debug_info(data: np.ndarray, window_width: float, window_level: float) -> dict:
        """调试信息（可选）"""
        wl_min = window_level - window_width / 2
        wl_max = window_level + window_width / 2
        
        # 统计原始数据
        original_min = float(data.min())
        original_max = float(data.max())
        original_mean = float(data.mean())
        
        # 统计窗宽窗位后的数据
        windowed_data = np.clip(data, wl_min, wl_max)
        windowed_pixels = np.sum((data >= wl_min) & (data <= wl_max))
        total_pixels = data.size
        effective_ratio = windowed_pixels / total_pixels
        
        return {
            'original_range': (original_min, original_max),
            'original_mean': original_mean,
            'window_range': (wl_min, wl_max),
            'effective_pixels_ratio': effective_ratio,
            'total_pixels': total_pixels,
            'effective_pixels': windowed_pixels
        }
