"""
图像质量分析模块 - 专门检测马赛克效应和图像质量变化
"""
import numpy as np
from scipy import ndimage, fft
from skimage import filters, feature, measure
import functools
import time
from typing import Dict, Any, Tuple, Optional


class ImageQualityAnalyzer:
    """图像质量分析器 - 检测马赛克效应、纹理变化等"""
    
    @staticmethod
    def analyze_image_quality(image: np.ndarray, name: str = "图像") -> Dict[str, Any]:
        """
        全面分析图像质量，特别关注马赛克效应

        Args:
            image: 输入图像
            name: 图像名称（用于报告）

        Returns:
            Dict: 包含各种质量指标的字典
        """
        print(f"\n📊 {name}质量分析:")

        # 大图像优化：如果图像太大，进行下采样分析
        original_shape = image.shape
        total_pixels = image.size

        if total_pixels > 2000000:  # 2M像素以上进行下采样
            print(f"   🔧 检测到大图像({total_pixels:,}像素)，使用下采样分析以提高速度...")
            # 计算下采样比例，目标约1M像素
            scale = np.sqrt(1000000 / total_pixels)
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)

            # 使用OpenCV进行高质量下采样
            from skimage import transform
            img_resized = transform.resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)
            img = img_resized.astype(np.float32)
            print(f"      下采样: {original_shape} → {img.shape}")
        else:
            # 确保是浮点数
            if image.dtype != np.float32:
                img = image.astype(np.float32)
            else:
                img = image.copy()

        # 归一化到[0,1]以便统一分析
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img.copy()
        
        analysis = {
            'name': name,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'range': (float(img_min), float(img_max)),
        }
        
        # 1. 基础统计
        analysis.update(ImageQualityAnalyzer._basic_statistics(img_norm))
        
        # 2. 纹理复杂度分析（马赛克的关键指标）
        analysis.update(ImageQualityAnalyzer._texture_complexity(img_norm))
        
        # 3. 边缘质量分析
        analysis.update(ImageQualityAnalyzer._edge_quality(img_norm))
        
        # 4. 频域分析
        analysis.update(ImageQualityAnalyzer._frequency_analysis(img_norm))
        
        # 5. 空间相关性分析（检测块状效应）
        analysis.update(ImageQualityAnalyzer._spatial_correlation(img_norm))
        
        # 6. 马赛克效应检测
        analysis.update(ImageQualityAnalyzer._mosaic_detection(img_norm))
        
        # 打印关键指标
        ImageQualityAnalyzer._print_key_metrics(analysis)

        return analysis

    @staticmethod
    def compare_analyses(original: Dict[str, Any], processed: Dict[str, Any]):
        """对比两个分析结果"""
        print(f"   📈 均值变化: {original['mean']:.3f} → {processed['mean']:.3f} ({processed['mean']/original['mean']:.2f}x)")
        print(f"   📊 标准差变化: {original['std']:.3f} → {processed['std']:.3f} ({processed['std']/original['std']:.2f}x)")
        print(f"   🎨 纹理复杂度变化: {original['local_var_mean']:.4f} → {processed['local_var_mean']:.4f} ({processed['local_var_mean']/original['local_var_mean']:.2f}x)")
        print(f"   🔍 边缘密度变化: {original['edge_density']:.4f} → {processed['edge_density']:.4f} ({processed['edge_density']/original['edge_density']:.2f}x)")
        print(f"   📊 高频能量变化: {original['high_freq_energy']:.3f} → {processed['high_freq_energy']:.3f} ({processed['high_freq_energy']/original['high_freq_energy']:.2f}x)")
        print(f"   🚨 马赛克指数变化: {original['mosaic_index']:.4f} → {processed['mosaic_index']:.4f} ({processed['mosaic_index']/original['mosaic_index']:.2f}x)")

        # 质量评估
        if processed['mosaic_index'] > original['mosaic_index'] * 2:
            print(f"   ⚠️  警告：马赛克效应显著增加！")
        if processed['texture_irregularity'] > original['texture_irregularity'] * 3:
            print(f"   ⚠️  警告：纹理变得异常不规律！")
        if processed['high_freq_energy'] > original['high_freq_energy'] * 5:
            print(f"   ⚠️  警告：高频噪声显著增加！")
    
    @staticmethod
    def _basic_statistics(img: np.ndarray) -> Dict[str, float]:
        """基础统计信息"""
        return {
            'mean': float(np.mean(img)),
            'std': float(np.std(img)),
            'entropy': float(measure.shannon_entropy(img)),
            'dynamic_range': float(img.max() - img.min()),
        }
    
    @staticmethod
    def _texture_complexity(img: np.ndarray) -> Dict[str, float]:
        """纹理复杂度分析 - 马赛克会导致纹理异常复杂"""
        # 局部方差（窗口大小5x5）
        local_var = ndimage.generic_filter(img, np.var, size=5)
        
        # 梯度幅值
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
        grad_mag = np.sqrt(gx*gx + gy*gy)
        
        # 纹理能量（灰度共生矩阵的简化版本）
        # 计算水平和垂直方向的纹理变化
        h_diff = np.abs(np.diff(img, axis=1))
        v_diff = np.abs(np.diff(img, axis=0))
        
        return {
            'local_var_mean': float(np.mean(local_var)),
            'local_var_std': float(np.std(local_var)),
            'local_var_max': float(np.max(local_var)),
            'grad_mag_mean': float(np.mean(grad_mag)),
            'grad_mag_std': float(np.std(grad_mag)),
            'texture_energy_h': float(np.mean(h_diff)),
            'texture_energy_v': float(np.mean(v_diff)),
            'texture_uniformity': float(1.0 / (1.0 + np.std(local_var))),  # 越小越不均匀
        }
    
    @staticmethod
    def _edge_quality(img: np.ndarray) -> Dict[str, float]:
        """边缘质量分析"""
        # Canny边缘检测
        edges = feature.canny(img, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        
        # 边缘连续性（通过形态学操作评估）
        from skimage import morphology
        edge_dilated = morphology.dilation(edges, morphology.disk(2))
        edge_continuity = np.sum(edge_dilated) / np.sum(edges) if np.sum(edges) > 0 else 0
        
        # 梯度方向一致性
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
        grad_angle = np.arctan2(gy, gx)
        
        # 计算梯度方向的局部一致性
        angle_diff = np.abs(np.diff(grad_angle, axis=1))
        angle_consistency = 1.0 - np.mean(angle_diff) / np.pi
        
        return {
            'edge_density': float(edge_density),
            'edge_continuity': float(edge_continuity),
            'gradient_consistency': float(angle_consistency),
        }
    
    @staticmethod
    def _frequency_analysis(img: np.ndarray) -> Dict[str, float]:
        """频域分析 - 马赛克会产生异常的高频成分"""
        # 对于大图像，进一步下采样以加速FFT
        if img.size > 500000:  # 500K像素以上再次下采样
            from skimage import transform
            scale = np.sqrt(500000 / img.size)
            new_h = max(64, int(img.shape[0] * scale))  # 最小64像素
            new_w = max(64, int(img.shape[1] * scale))
            img_small = transform.resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)
        else:
            img_small = img

        # 2D FFT
        f_transform = fft.fft2(img_small)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # 计算频域能量分布
        h, w = img_small.shape
        center_h, center_w = h // 2, w // 2
        
        # 创建频率掩码
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # 低频、中频、高频区域
        max_dist = min(center_h, center_w)
        low_freq_mask = dist_from_center <= max_dist * 0.1
        mid_freq_mask = (dist_from_center > max_dist * 0.1) & (dist_from_center <= max_dist * 0.5)
        high_freq_mask = dist_from_center > max_dist * 0.5
        
        total_energy = np.sum(magnitude**2)
        low_freq_energy = np.sum(magnitude[low_freq_mask]**2) / total_energy
        mid_freq_energy = np.sum(magnitude[mid_freq_mask]**2) / total_energy
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2) / total_energy
        
        return {
            'low_freq_energy': float(low_freq_energy),
            'mid_freq_energy': float(mid_freq_energy),
            'high_freq_energy': float(high_freq_energy),
            'freq_energy_ratio': float(high_freq_energy / (low_freq_energy + 1e-8)),
        }
    
    @staticmethod
    def _spatial_correlation(img: np.ndarray) -> Dict[str, float]:
        """空间相关性分析 - 检测块状效应"""
        # 水平和垂直方向的自相关
        h_corr = np.corrcoef(img[:-1].flatten(), img[1:].flatten())[0, 1]
        v_corr = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
        
        # 对角线相关性
        d1_corr = np.corrcoef(img[:-1, :-1].flatten(), img[1:, 1:].flatten())[0, 1]
        d2_corr = np.corrcoef(img[:-1, 1:].flatten(), img[1:, :-1].flatten())[0, 1]
        
        # 局部相关性变化（检测块状效应）
        block_size = 8
        h_blocks = img.shape[0] // block_size
        w_blocks = img.shape[1] // block_size
        
        block_correlations = []
        for i in range(h_blocks - 1):
            for j in range(w_blocks - 1):
                block1 = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block2 = img[(i+1)*block_size:(i+2)*block_size, j*block_size:(j+1)*block_size]
                if block1.size > 0 and block2.size > 0:
                    corr = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
                    if not np.isnan(corr):
                        block_correlations.append(corr)
        
        block_corr_std = np.std(block_correlations) if block_correlations else 0
        
        return {
            'horizontal_correlation': float(h_corr) if not np.isnan(h_corr) else 0.0,
            'vertical_correlation': float(v_corr) if not np.isnan(v_corr) else 0.0,
            'diagonal_correlation_1': float(d1_corr) if not np.isnan(d1_corr) else 0.0,
            'diagonal_correlation_2': float(d2_corr) if not np.isnan(d2_corr) else 0.0,
            'block_correlation_std': float(block_corr_std),
        }
    
    @staticmethod
    def _mosaic_detection(img: np.ndarray) -> Dict[str, float]:
        """马赛克效应检测"""
        # 块状效应检测：计算8x8块的内部方差vs块间方差
        block_size = 8
        h_blocks = img.shape[0] // block_size
        w_blocks = img.shape[1] // block_size
        
        intra_block_vars = []  # 块内方差
        inter_block_vars = []  # 块间方差
        
        block_means = np.zeros((h_blocks, w_blocks))
        
        # 计算每个块的均值和内部方差
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_mean = np.mean(block)
                block_var = np.var(block)
                
                block_means[i, j] = block_mean
                intra_block_vars.append(block_var)
        
        # 计算块间方差
        if h_blocks > 1 and w_blocks > 1:
            inter_block_var = np.var(block_means)
        else:
            inter_block_var = 0
        
        avg_intra_var = np.mean(intra_block_vars)
        
        # 马赛克指数：块内方差高 + 块间变化大 = 马赛克效应
        mosaic_index = avg_intra_var * (1 + inter_block_var)
        
        # 纹理不规律性：局部方差的方差
        local_var = ndimage.generic_filter(img, np.var, size=3)
        texture_irregularity = np.var(local_var)
        
        return {
            'avg_intra_block_variance': float(avg_intra_var),
            'inter_block_variance': float(inter_block_var),
            'mosaic_index': float(mosaic_index),
            'texture_irregularity': float(texture_irregularity),
        }
    
    @staticmethod
    def _print_key_metrics(analysis: Dict[str, Any]):
        """打印关键指标"""
        print(f"   📈 基础统计: 均值={analysis['mean']:.3f}, 标准差={analysis['std']:.3f}, 熵={analysis['entropy']:.3f}")
        print(f"   🎨 纹理复杂度: 局部方差均值={analysis['local_var_mean']:.4f}, 梯度幅值={analysis['grad_mag_mean']:.4f}")
        print(f"   🔍 边缘质量: 边缘密度={analysis['edge_density']:.4f}, 梯度一致性={analysis['gradient_consistency']:.3f}")
        print(f"   📊 频域分析: 高频能量占比={analysis['high_freq_energy']:.3f}, 频率比={analysis['freq_energy_ratio']:.2f}")
        print(f"   🧩 空间相关性: 水平={analysis['horizontal_correlation']:.3f}, 垂直={analysis['vertical_correlation']:.3f}")
        print(f"   🚨 马赛克检测: 马赛克指数={analysis['mosaic_index']:.4f}, 纹理不规律性={analysis['texture_irregularity']:.4f}")
        
        # 马赛克效应警告
        if analysis['mosaic_index'] > 0.01:  # 阈值需要根据实际情况调整
            print(f"   ⚠️  检测到可能的马赛克效应！")
        if analysis['texture_irregularity'] > 0.005:
            print(f"   ⚠️  检测到纹理异常不规律！")


def image_analysis_decorator(func):
    """
    图像分析装饰器 - 自动分析处理前后的图像质量
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 假设第一个参数是图像数据
        if len(args) > 0:
            input_image = args[0]
            
            # 分析原始图像
            print("\n" + "="*60)
            print("🔍 图像质量分析开始")
            print("="*60)
            
            original_analysis = ImageQualityAnalyzer.analyze_image_quality(
                input_image, "原始图像"
            )
            
            # 执行原始函数
            print(f"\n🚀 开始执行 {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 分析处理后的图像
            if isinstance(result, np.ndarray):
                processed_analysis = ImageQualityAnalyzer.analyze_image_quality(
                    result, "处理后图像"
                )
                
                # 对比分析
                print(f"\n📊 处理前后对比分析:")
                ImageQualityAnalyzer.compare_analyses(original_analysis, processed_analysis)
            
            print(f"\n⏱️  总处理时间: {execution_time:.2f}秒")
            print("="*60)
            
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper



