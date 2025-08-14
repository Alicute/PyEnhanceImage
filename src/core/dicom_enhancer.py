"""
DICOM图像增强处理器
提供4个不同级别的增强算法：普通增强、高级增强、超级增强、一键处理
"""
import numpy as np
import cv2
from typing import Optional, Callable


class DicomEnhancer:
    """DICOM图像增强处理器"""
    
    @staticmethod
    def basic_enhance(data: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        普通增强：基础CLAHE + 简单高频增强
        
        Args:
            data: 输入图像数据
            progress_callback: 进度回调函数
            
        Returns:
            增强后的图像数据
        """
        try:
            if progress_callback:
                progress_callback(10)
            
            # 归一化到 0~1
            img_norm = DicomEnhancer._normalize_image(data)
            
            if progress_callback:
                progress_callback(30)
            
            # 简单高频增强
            blur = cv2.GaussianBlur(img_norm, (0, 0), 2.0)
            high_freq = img_norm - blur
            img_enhanced = np.clip(img_norm + 0.3 * high_freq, 0, 1)
            
            if progress_callback:
                progress_callback(60)
            
            # 基础CLAHE
            img_16 = (img_enhanced * 65535).astype(np.uint16)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(img_16)
            
            if progress_callback:
                progress_callback(100)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"普通增强处理失败: {str(e)}")
    
    @staticmethod
    def advanced_enhance(data: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        高级增强：完整的多步骤算法（基于tests/dicom.py）
        
        Args:
            data: 输入图像数据
            progress_callback: 进度回调函数
            
        Returns:
            增强后的图像数据
        """
        try:
            if progress_callback:
                progress_callback(5)
            
            # 归一化到 0~1
            img_norm = DicomEnhancer._normalize_image(data)
            
            if progress_callback:
                progress_callback(15)
            
            # 1. 自适应高频增强
            # 局部方差（噪声检测）
            var_map = cv2.GaussianBlur(img_norm**2, (7, 7), 0) - cv2.GaussianBlur(img_norm, (7, 7), 0)**2
            var_map = np.clip(var_map / var_map.max(), 0, 1)
            
            if progress_callback:
                progress_callback(30)
            
            # 多尺度高频
            blur_small = cv2.GaussianBlur(img_norm, (0, 0), 1)
            high_freq_small = img_norm - blur_small
            blur_large = cv2.GaussianBlur(img_norm, (0, 0), 5)
            high_freq_large = img_norm - blur_large
            
            # 增强强度随方差变化（平坦区增强少）
            img_detail = img_norm + (0.5 + 1.0 * var_map) * high_freq_small + (0.3 + 0.5 * var_map) * high_freq_large
            img_detail = np.clip(img_detail, 0, 1)
            
            if progress_callback:
                progress_callback(50)
            
            # 2. 光照归一化
            illum = cv2.GaussianBlur(img_detail, (0, 0), 30)
            img_light_norm = img_detail / (illum + 1e-6)
            img_light_norm = np.clip(img_light_norm / img_light_norm.max(), 0, 1)
            
            # 保留原亮度
            img_fused = np.clip(0.7 * img_detail + 0.3 * img_light_norm, 0, 1)
            
            if progress_callback:
                progress_callback(70)
            
            # 3. 双边滤波降噪（保边）
            img_fused_8 = (img_fused * 255).astype(np.uint8)
            img_denoised_8 = cv2.bilateralFilter(img_fused_8, d=5, sigmaColor=50, sigmaSpace=5)
            img_denoised = (img_denoised_8.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            
            if progress_callback:
                progress_callback(85)
            
            # 4. CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            result = clahe.apply(img_denoised)
            
            if progress_callback:
                progress_callback(100)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"高级增强处理失败: {str(e)}")
    
    @staticmethod
    def super_enhance(data: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        超级增强：更复杂的多层次处理
        
        Args:
            data: 输入图像数据
            progress_callback: 进度回调函数
            
        Returns:
            增强后的图像数据
        """
        try:
            if progress_callback:
                progress_callback(5)
            
            # 归一化到 0~1
            img_norm = DicomEnhancer._normalize_image(data)
            
            if progress_callback:
                progress_callback(10)
            
            # 1. 多层次噪声检测
            var_map_fine = cv2.GaussianBlur(img_norm**2, (3, 3), 0) - cv2.GaussianBlur(img_norm, (3, 3), 0)**2
            var_map_coarse = cv2.GaussianBlur(img_norm**2, (15, 15), 0) - cv2.GaussianBlur(img_norm, (15, 15), 0)**2
            var_map_fine = np.clip(var_map_fine / var_map_fine.max(), 0, 1)
            var_map_coarse = np.clip(var_map_coarse / var_map_coarse.max(), 0, 1)
            
            if progress_callback:
                progress_callback(25)
            
            # 2. 多尺度高频增强
            scales = [0.5, 1.0, 2.0, 4.0]
            enhanced_components = []
            
            for i, sigma in enumerate(scales):
                blur = cv2.GaussianBlur(img_norm, (0, 0), sigma)
                high_freq = img_norm - blur
                # 自适应增强强度
                strength = 0.8 - 0.1 * i  # 细节越细，增强越强
                enhanced_components.append(strength * high_freq)
                
                if progress_callback:
                    progress_callback(25 + 15 * (i + 1) / len(scales))
            
            # 组合多尺度增强
            img_detail = img_norm
            for component in enhanced_components:
                img_detail = img_detail + (0.3 + 0.7 * var_map_fine) * component
            img_detail = np.clip(img_detail, 0, 1)
            
            if progress_callback:
                progress_callback(50)
            
            # 3. 高级光照归一化
            illum_fine = cv2.GaussianBlur(img_detail, (0, 0), 20)
            illum_coarse = cv2.GaussianBlur(img_detail, (0, 0), 50)
            
            img_light_fine = img_detail / (illum_fine + 1e-6)
            img_light_coarse = img_detail / (illum_coarse + 1e-6)
            
            img_light_fine = np.clip(img_light_fine / img_light_fine.max(), 0, 1)
            img_light_coarse = np.clip(img_light_coarse / img_light_coarse.max(), 0, 1)
            
            # 融合不同尺度的光照归一化
            img_fused = np.clip(0.5 * img_detail + 0.3 * img_light_fine + 0.2 * img_light_coarse, 0, 1)
            
            if progress_callback:
                progress_callback(70)
            
            # 4. 边缘保护的降噪
            img_fused_8 = (img_fused * 255).astype(np.uint8)

            # 多次双边滤波，参数递减
            img_denoised = img_fused_8
            for i, (d, sigma_color, sigma_space) in enumerate([(7, 80, 7), (5, 60, 5), (3, 40, 3)]):
                img_denoised = cv2.bilateralFilter(img_denoised, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                if progress_callback:
                    progress_callback(70 + 10 * (i + 1) / 3)

            # 转换回16位
            img_denoised = (img_denoised.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            
            if progress_callback:
                progress_callback(85)
            
            # 5. 自适应CLAHE
            clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(16, 16))
            result = clahe.apply(img_denoised)
            
            if progress_callback:
                progress_callback(100)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"超级增强处理失败: {str(e)}")
    
    @staticmethod
    def auto_enhance(data: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        一键处理：自动参数优化
        
        Args:
            data: 输入图像数据
            progress_callback: 进度回调函数
            
        Returns:
            增强后的图像数据
        """
        try:
            if progress_callback:
                progress_callback(5)
            
            # 分析图像特征
            img_norm = DicomEnhancer._normalize_image(data)
            
            # 计算图像统计特征
            mean_val = np.mean(img_norm)
            std_val = np.std(img_norm)
            contrast = std_val / (mean_val + 1e-6)
            
            if progress_callback:
                progress_callback(15)
            
            # 根据图像特征选择处理策略
            if contrast < 0.2:  # 低对比度图像
                # 使用更强的增强
                result = DicomEnhancer._enhance_low_contrast(img_norm, progress_callback)
            elif contrast > 0.8:  # 高对比度图像
                # 使用保守的增强
                result = DicomEnhancer._enhance_high_contrast(img_norm, progress_callback)
            else:  # 中等对比度图像
                # 使用标准增强
                result = DicomEnhancer._enhance_normal_contrast(img_norm, progress_callback)
            
            if progress_callback:
                progress_callback(100)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"一键处理失败: {str(e)}")
    
    @staticmethod
    def _normalize_image(data: np.ndarray) -> np.ndarray:
        """归一化图像到0-1范围"""
        data_float = data.astype(np.float32)
        return (data_float - data_float.min()) / (data_float.max() - data_float.min())
    
    @staticmethod
    def _enhance_low_contrast(img_norm: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """低对比度图像增强"""
        # 强CLAHE + 高频增强
        blur = cv2.GaussianBlur(img_norm, (0, 0), 1.5)
        high_freq = img_norm - blur
        img_enhanced = np.clip(img_norm + 0.8 * high_freq, 0, 1)
        
        if progress_callback:
            progress_callback(60)
        
        img_16 = (img_enhanced * 65535).astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img_16)
    
    @staticmethod
    def _enhance_high_contrast(img_norm: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """高对比度图像增强"""
        # 轻微增强 + 降噪
        blur = cv2.GaussianBlur(img_norm, (0, 0), 2.0)
        high_freq = img_norm - blur
        img_enhanced = np.clip(img_norm + 0.2 * high_freq, 0, 1)
        
        if progress_callback:
            progress_callback(50)
        
        img_8 = (img_enhanced * 255).astype(np.uint8)
        img_denoised_8 = cv2.bilateralFilter(img_8, d=5, sigmaColor=30, sigmaSpace=5)
        img_denoised = (img_denoised_8.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        
        if progress_callback:
            progress_callback(80)
        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        return clahe.apply(img_denoised)
    
    @staticmethod
    def _enhance_normal_contrast(img_norm: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """中等对比度图像增强"""
        # 使用高级增强的简化版本
        blur = cv2.GaussianBlur(img_norm, (0, 0), 1.5)
        high_freq = img_norm - blur
        img_enhanced = np.clip(img_norm + 0.5 * high_freq, 0, 1)
        
        if progress_callback:
            progress_callback(60)
        
        img_8 = (img_enhanced * 255).astype(np.uint8)
        img_denoised_8 = cv2.bilateralFilter(img_8, d=5, sigmaColor=50, sigmaSpace=5)
        img_denoised = (img_denoised_8.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        
        if progress_callback:
            progress_callback(80)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(img_denoised)
