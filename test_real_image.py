"""
测试真实DICOM图像的处理效果
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core.image_processor import ImageProcessor
from src.core.image_manager import ImageManager

def test_real_dicom():
    """测试真实DICOM图像"""
    dicom_path = r"D:\Projects\PyEnhanceImage\钢板-原始图.dcm"

    print(f'🔍 开始测试真实DICOM图像')
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
        
        # 测试论文算法处理
        print(f'\n🚀 开始论文算法处理...')
        result = ImageProcessor.paper_enhance(dicom_data)
        
        print(f'✅ 真实图像处理完成!')
        print(f'   输出大小: {result.shape}')
        print(f'   输出范围: {result.min()} - {result.max()}')
        
    except Exception as e:
        print(f'❌ 处理失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始真实DICOM图像测试")
    print("=" * 60)
    test_real_dicom()
    print("=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
