"""
图像处理算法模块
"""
import numpy as np
from skimage import exposure, filters, morphology, restoration
from typing import Dict, Any, Tuple, Optional

# 导入新的处理器模块
from .frequency_processor import FrequencyProcessor
from .edge_processor import EdgeProcessor
from .dicom_enhancer import DicomEnhancer
from .window_based_enhancer import WindowBasedEnhancer
from .paper_enhance import enhance_xray_poisson_nlm_strict

class ImageProcessor:
    """图像处理算法集合"""
    
    @staticmethod
    def gamma_correction(data: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Gamma校正"""
        if gamma <= 0:
            gamma = 0.1
        
        # 归一化到0-1范围
        normalized = data.astype(np.float32)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        # 应用Gamma校正
        corrected = np.power(normalized, gamma)
        
        # 恢复到原始范围
        corrected = corrected * (data.max() - data.min()) + data.min()
        
        return corrected.astype(np.uint16)
    
    @staticmethod
    def histogram_equalization(data: np.ndarray, method: str = 'global') -> np.ndarray:
        """直方图均衡化"""
        if method == 'global':
            # 全局直方图均衡化
            return exposure.equalize_hist(data) * 65535
        elif method == 'adaptive':
            # 自适应直方图均衡化（CLAHE）
            return exposure.equalize_adapthist(data) * 65535
        else:
            return data
    
    @staticmethod
    def gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """高斯滤波"""
        if sigma <= 0:
            sigma = 0.1
        
        # 转换为float类型进行处理
        filtered = filters.gaussian(data.astype(np.float32), sigma=sigma)
        
        # 恢复到原始范围和类型
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
        filtered = filtered * (data.max() - data.min()) + data.min()
        
        return filtered.astype(np.uint16)
    
    @staticmethod
    def median_filter(data: np.ndarray, disk_size: int = 3) -> np.ndarray:
        """中值滤波"""
        if disk_size < 1:
            disk_size = 1
        
        # 使用圆形结构元素
        selem = morphology.disk(disk_size)
        filtered = filters.median(data, selem)
        
        return filtered.astype(np.uint16)
    
    @staticmethod
    def unsharp_mask(data: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
        """非锐化掩模"""
        if radius <= 0:
            radius = 0.1
        if amount < 0:
            amount = 0
        
        # 转换为float类型进行处理
        data_float = data.astype(np.float32)
        
        # 应用非锐化掩模
        sharpened = filters.unsharp_mask(data_float, radius=radius, amount=amount)
        
        # 恢复到原始范围和类型（避免除零错误）
        sharpened_min = sharpened.min()
        sharpened_max = sharpened.max()

        if sharpened_max > sharpened_min:
            sharpened = (sharpened - sharpened_min) / (sharpened_max - sharpened_min)
            sharpened = sharpened * (data.max() - data.min()) + data.min()
        else:
            # 如果图像是常数，直接返回原图像
            sharpened = data.astype(np.float64)

        return np.clip(sharpened, 0, 65535).astype(np.uint16)
    
    @staticmethod
    def morphological_operation(data: np.ndarray, operation: str, disk_size: int = 3) -> np.ndarray:
        """形态学操作"""
        if disk_size < 1:
            disk_size = 1
        
        # 创建结构元素
        selem = morphology.disk(disk_size)
        
        # 根据操作类型处理
        if operation == 'erosion':
            result = morphology.erosion(data, selem)
        elif operation == 'dilation':
            result = morphology.dilation(data, selem)
        elif operation == 'opening':
            result = morphology.opening(data, selem)
        elif operation == 'closing':
            result = morphology.closing(data, selem)
        else:
            result = data
        
        return result.astype(np.uint16)
    
    @staticmethod
    def low_pass_filter(data: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
        """低通滤波（频域）"""
        # 转换为float类型
        data_float = data.astype(np.float32)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(data_float)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 创建低通滤波器
        rows, cols = data.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高斯低通滤波器
        y, x = np.ogrid[:rows, :cols]
        mask = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (cutoff_frequency * min(rows, cols)) ** 2))
        
        # 应用滤波器
        filtered = f_transform_shifted * mask
        
        # 逆变换
        filtered_shifted = np.fft.ifftshift(filtered)
        filtered_result = np.fft.ifft2(filtered_shifted)
        filtered_result = np.real(filtered_result)
        
        # 恢复到原始范围和类型
        filtered_result = (filtered_result - filtered_result.min()) / (filtered_result.max() - filtered_result.min())
        filtered_result = filtered_result * (data.max() - data.min()) + data.min()
        
        return filtered_result.astype(np.uint16)
    
    @staticmethod
    def high_pass_filter(data: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
        """高通滤波（频域）"""
        # 转换为float类型
        data_float = data.astype(np.float32)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(data_float)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 创建高通滤波器
        rows, cols = data.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高斯高通滤波器
        y, x = np.ogrid[:rows, :cols]
        mask = 1 - np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (cutoff_frequency * min(rows, cols)) ** 2))
        
        # 应用滤波器
        filtered = f_transform_shifted * mask
        
        # 逆变换
        filtered_shifted = np.fft.ifftshift(filtered)
        filtered_result = np.fft.ifft2(filtered_shifted)
        filtered_result = np.real(filtered_result)
        
        # 恢复到原始范围和类型
        filtered_result = (filtered_result - filtered_result.min()) / (filtered_result.max() - filtered_result.min())
        filtered_result = filtered_result * (data.max() - data.min()) + data.min()
        
        return filtered_result.astype(np.uint16)
    
    @staticmethod
    def get_algorithm_info() -> Dict[str, Dict[str, Any]]:
        """获取算法信息"""
        return {
            'gamma_correction': {
                'name': 'Gamma校正',
                'description': '调整图像的亮度分布',
                'parameters': {
                    'gamma': {'type': 'float', 'range': (0.1, 5.0), 'default': 1.0, 'description': 'Gamma值'}
                }
            },
            'histogram_equalization': {
                'name': '直方图均衡化',
                'description': '改善图像对比度',
                'parameters': {
                    'method': {'type': 'string', 'options': ['global', 'adaptive'], 'default': 'global', 'description': '均衡化方法'}
                }
            },
            'gaussian_filter': {
                'name': '高斯滤波',
                'description': '高斯降噪',
                'parameters': {
                    'sigma': {'type': 'float', 'range': (0.1, 10.0), 'default': 1.0, 'description': '高斯核标准差'}
                }
            },
            'median_filter': {
                'name': '中值滤波',
                'description': '中值降噪',
                'parameters': {
                    'disk_size': {'type': 'int', 'range': (1, 10), 'default': 3, 'description': '滤波器大小'}
                }
            },
            'unsharp_mask': {
                'name': '非锐化掩模',
                'description': '图像锐化',
                'parameters': {
                    'radius': {'type': 'float', 'range': (0.1, 5.0), 'default': 1.0, 'description': '锐化半径'},
                    'amount': {'type': 'float', 'range': (0.0, 3.0), 'default': 1.0, 'description': '锐化强度'}
                }
            },
            'morphological_operation': {
                'name': '形态学操作',
                'description': '形态学处理',
                'parameters': {
                    'operation': {'type': 'string', 'options': ['erosion', 'dilation', 'opening', 'closing'], 'default': 'erosion', 'description': '操作类型'},
                    'disk_size': {'type': 'int', 'range': (1, 10), 'default': 3, 'description': '结构元素大小'}
                }
            }
        }

    # ==================== 频域增强方法 ====================

    @staticmethod
    def ideal_low_pass_filter(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """理想低通滤波"""
        return FrequencyProcessor.ideal_low_pass(data, cutoff_ratio)

    @staticmethod
    def ideal_high_pass_filter(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """理想高通滤波"""
        return FrequencyProcessor.ideal_high_pass(data, cutoff_ratio)

    @staticmethod
    def gaussian_low_pass_filter(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """高斯低通滤波"""
        return FrequencyProcessor.gaussian_low_pass(data, cutoff_ratio)

    @staticmethod
    def gaussian_high_pass_filter(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
        """高斯高通滤波"""
        return FrequencyProcessor.gaussian_high_pass(data, cutoff_ratio)

    # ==================== 边缘检测方法 ====================

    @staticmethod
    def sobel_edge_detection(data: np.ndarray) -> np.ndarray:
        """Sobel边缘检测"""
        return EdgeProcessor.sobel_edge(data)

    @staticmethod
    def canny_edge_detection(data: np.ndarray, sigma: float = 1.0,
                           low_threshold: float = 0.1, high_threshold: float = 0.2) -> np.ndarray:
        """Canny边缘检测"""
        return EdgeProcessor.canny_edge(data, sigma, low_threshold, high_threshold)

    @staticmethod
    def laplacian_edge_detection(data: np.ndarray) -> np.ndarray:
        """Laplacian边缘检测"""
        return EdgeProcessor.laplacian_edge(data)

    @staticmethod
    def edge_enhancement(data: np.ndarray, edge_strength: float = 1.0,
                        edge_method: str = 'sobel') -> np.ndarray:
        """边缘增强"""
        return EdgeProcessor.edge_enhancement(data, edge_strength, edge_method)

    @staticmethod
    def roberts_edge_detection(data: np.ndarray) -> np.ndarray:
        """Roberts边缘检测"""
        return EdgeProcessor.roberts_edge(data)

    # ==================== DICOM增强方法 ====================

    @staticmethod
    def dicom_basic_enhance(data: np.ndarray) -> np.ndarray:
        """DICOM普通增强"""
        return DicomEnhancer.basic_enhance(data)

    @staticmethod
    def dicom_advanced_enhance(data: np.ndarray) -> np.ndarray:
        """DICOM高级增强"""
        return DicomEnhancer.advanced_enhance(data)

    @staticmethod
    def dicom_super_enhance(data: np.ndarray) -> np.ndarray:
        """DICOM超级增强"""
        return DicomEnhancer.super_enhance(data)

    @staticmethod
    def dicom_auto_enhance(data: np.ndarray) -> np.ndarray:
        """DICOM一键处理"""
        return DicomEnhancer.auto_enhance(data)

    @staticmethod
    def window_based_enhance(data: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
        """基于窗宽窗位的缺陷检测增强"""
        return WindowBasedEnhancer.window_based_enhance(data, window_width, window_level)

    @staticmethod
    def paper_enhance(data: np.ndarray, progress_callback=None) -> np.ndarray:
        """论文算法：基于梯度场和非局部均值的复杂工件图像增强算法"""
        print(f"\n📄 论文算法处理:")
        print(f"   输入数据范围: {data.min()} - {data.max()}")
        print(f"   输入数据类型: {data.dtype}")
        print(f"   图像大小: {data.shape}")

        if progress_callback:
            progress_callback(0.1)

        try:
            print(f"   🔄 开始执行论文算法（预计需要10-30秒）...")

            if progress_callback:
                progress_callback(0.2)

            print(f"   📊 Step1: 开始梯度场自适应增强...")

            # 为了调试，我们先尝试更快的参数
            print(f"   🔧 使用加速参数进行测试...")

            # 调用论文算法（使用新的优化接口）
            def progress_wrapper(progress):
                if progress_callback:
                    progress_callback(progress)

            I_enh, (Gx_p, Gy_p), (Gx, Gy), nctx = enhance_xray_poisson_nlm_strict(
                data,
                # 归一化参数
                norm_mode="percentile", p_lo=0.5, p_hi=99.5,
                # Step1: 梯度场增强参数
                epsilon_8bit=2.3, mu=10.0, ksize_var=5,
                # Step2: NLM参数（快速模式会自动映射）
                rho=1.5, search_radius=3, patch_radius=2, topk=15,
                count_target_mean=18.0,  # 0.3 * 60.0，对应原来的count_scale=0.3
                lam_quant=0.02,
                # Step3: 变分重建参数
                gamma=0.2, delta=0.8, iters=5, dt=0.15,
                # 输出参数
                out_dtype=np.uint16,
                # 进度回调
                progress_callback=progress_wrapper,
                # 快速模式（自动判断）
                use_fast_nlm=None  # None=自动判断，大图像会自动使用快速模式
            )

            print(f"   📊 论文算法核心处理完成，开始后处理...")

            if progress_callback:
                progress_callback(0.9)

            print(f"   输出数据范围: {I_enh.min()} - {I_enh.max()}")
            print(f"   输出数据类型: {I_enh.dtype}")
            print(f"   梯度场范围: Gx_p[{Gx_p.min():.2f}, {Gx_p.max():.2f}], Gy_p[{Gy_p.min():.2f}, {Gy_p.max():.2f}]")
            print(f"   处理后梯度: Gx[{Gx.min():.2f}, {Gx.max():.2f}], Gy[{Gy.min():.2f}, {Gy.max():.2f}]")
            print(f"   归一化上下文: vmin={nctx['vmin']:.1f}, vmax={nctx['vmax']:.1f}")

            # 新函数直接返回16位结果，无需转换
            result = I_enh

            if progress_callback:
                progress_callback(1.0)

            print(f"   ✅ 论文算法处理完成")
            return result

        except Exception as e:
            print(f"   ❌ 论文算法处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回原始数据
            return data