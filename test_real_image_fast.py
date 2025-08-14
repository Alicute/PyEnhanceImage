"""
快速测试真实DICOM图像的处理效果（不进行详细分析）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core.image_processor import ImageProcessor
from src.core.image_manager import ImageManager

def test_real_dicom_fast():
    """快速测试真实DICOM图像"""
    dicom_path = r"D:\Projects\PyEnhanceImage\钢板-原始图.dcm"
    
    print(f'🔍 开始快速测试真实DICOM图像')
    print(f'   文件路径: {dicom_path}')
    print(f'   文件存在: {os.path.exists(dicom_path)}')
    
    try:
        # 加载DICOM图像
        image_manager = ImageManager()
        success = image_manager.load_dicom(dicom_path)
        
        if not success:
            print(f'❌ DICOM文件加载失败')
            return
            
        dicom_data = image_manager.current_image.data
        metadata = image_manager.current_image.metadata
        
        print(f'✅ DICOM加载成功!')
        print(f'   图像大小: {dicom_data.shape}')
        print(f'   数据类型: {dicom_data.dtype}')
        print(f'   数据范围: {dicom_data.min()} - {dicom_data.max()}')
        print(f'   总像素数: {dicom_data.size:,}')
        
        # 简单的质量分析（不使用装饰器）
        print(f'\n📊 原始图像简单分析:')
        print(f'   均值: {np.mean(dicom_data):.1f}')
        print(f'   标准差: {np.std(dicom_data):.1f}')
        print(f'   动态范围: {dicom_data.max() - dicom_data.min()}')
        
        # 计算简单的纹理指标
        gx = np.abs(np.diff(dicom_data, axis=1))
        gy = np.abs(np.diff(dicom_data, axis=0))
        grad_mag = np.mean(gx) + np.mean(gy)
        print(f'   平均梯度幅值: {grad_mag:.2f}')
        
        # 测试论文算法处理（临时移除装饰器）
        print(f'\n🚀 开始论文算法处理（无详细分析）...')
        
        # 直接调用处理函数，绕过装饰器
        from src.core.paper_enhance import enhance_xray_poisson_nlm_strict
        
        def progress_wrapper(progress):
            if progress in [0.1, 0.3, 0.5, 0.7, 0.9]:
                print(f'   📈 处理进度: {progress*100:.0f}%')
        
        I_enh, (Gx_p, Gy_p), (Gx, Gy), nctx = enhance_xray_poisson_nlm_strict(
            dicom_data,
            # 归一化参数
            norm_mode="percentile", p_lo=0.5, p_hi=99.5,
            # Step1: 梯度场增强参数
            epsilon_8bit=2.3, mu=10.0, ksize_var=5,
            # Step2: NLM参数
            rho=1.5, search_radius=1, patch_radius=1, topk=5,
            count_target_mean=18.0,
            lam_quant=0.02,
            # Step3: 变分重建参数
            gamma=0.2, delta=0.8, iters=3, dt=0.15,
            # 输出参数
            out_dtype=np.uint16,
            # 进度回调
            progress_callback=progress_wrapper,
            # 强制使用原始算法
            use_fast_nlm=False
        )
        
        print(f'✅ 真实图像处理完成!')
        print(f'   输出大小: {I_enh.shape}')
        print(f'   输出范围: {I_enh.min()} - {I_enh.max()}')
        
        # 简单的处理后分析
        print(f'\n📊 处理后图像简单分析:')
        print(f'   均值: {np.mean(I_enh):.1f}')
        print(f'   标准差: {np.std(I_enh):.1f}')
        print(f'   动态范围: {I_enh.max() - I_enh.min()}')
        
        # 计算处理后的纹理指标
        gx_after = np.abs(np.diff(I_enh, axis=1))
        gy_after = np.abs(np.diff(I_enh, axis=0))
        grad_mag_after = np.mean(gx_after) + np.mean(gy_after)
        print(f'   平均梯度幅值: {grad_mag_after:.2f}')
        
        # 简单对比
        print(f'\n📊 简单对比分析:')
        print(f'   均值变化: {np.mean(dicom_data):.1f} → {np.mean(I_enh):.1f}')
        print(f'   标准差变化: {np.std(dicom_data):.1f} → {np.std(I_enh):.1f}')
        print(f'   梯度幅值变化: {grad_mag:.2f} → {grad_mag_after:.2f} ({grad_mag_after/grad_mag:.2f}x)')
        
        # 马赛克效应检测（简化版）
        # 计算8x8块的方差变化
        def block_variance_analysis(img, block_size=8):
            h, w = img.shape
            h_blocks = h // block_size
            w_blocks = w // block_size
            
            block_vars = []
            for i in range(0, h_blocks*block_size, block_size):
                for j in range(0, w_blocks*block_size, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    if block.size > 0:
                        block_vars.append(np.var(block))
            
            return np.mean(block_vars) if block_vars else 0
        
        # 随机采样分析（避免全图计算）
        sample_size = min(1000000, dicom_data.size)  # 最多采样100万像素
        if dicom_data.size > sample_size:
            indices = np.random.choice(dicom_data.size, sample_size, replace=False)
            sample_original = dicom_data.flat[indices].reshape(-1, int(np.sqrt(sample_size)))
            sample_processed = I_enh.flat[indices].reshape(-1, int(np.sqrt(sample_size)))
        else:
            sample_original = dicom_data
            sample_processed = I_enh
            
        block_var_orig = block_variance_analysis(sample_original)
        block_var_proc = block_variance_analysis(sample_processed)
        
        print(f'   块方差变化: {block_var_orig:.1f} → {block_var_proc:.1f} ({block_var_proc/block_var_orig:.2f}x)')
        
        if block_var_proc > block_var_orig * 2:
            print(f'   ⚠️  警告：检测到可能的马赛克效应（块方差增加{block_var_proc/block_var_orig:.1f}倍）')
        else:
            print(f'   ✅ 块方差变化正常，未检测到明显马赛克效应')
        
    except Exception as e:
        print(f'❌ 处理失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始真实DICOM图像快速测试")
    print("=" * 60)
    test_real_dicom_fast()
    print("=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
