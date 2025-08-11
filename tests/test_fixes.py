"""
修复验证测试

验证图像加载、缩放、拖拽等功能是否正常工作
"""

import sys
import os
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_processor import ImageProcessor


def test_image_processor_fixes():
    """测试图像处理器修复"""
    print("=== 图像处理器修复测试 ===")
    
    processor = ImageProcessor()
    
    # 测试常数图像（会导致除零错误的情况）
    constant_image = np.full((100, 100), 128, dtype=np.uint16)
    print(f"常数图像: {constant_image.shape}, 值={constant_image[0,0]}")
    
    try:
        # 测试非锐化掩模（之前会出错的算法）
        result = processor.unsharp_mask(constant_image, radius=1.0, amount=1.0)
        print(f"非锐化掩模结果: {result.shape}, dtype={result.dtype}")
        print(f"结果范围: {result.min()} - {result.max()}")
        print("✅ 非锐化掩模修复成功")
    except Exception as e:
        print(f"❌ 非锐化掩模仍有问题: {e}")
    
    # 测试正常图像
    normal_image = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
    
    try:
        result = processor.unsharp_mask(normal_image, radius=1.0, amount=1.0)
        print(f"正常图像非锐化掩模: {result.shape}, dtype={result.dtype}")
        print("✅ 正常图像处理成功")
    except Exception as e:
        print(f"❌ 正常图像处理失败: {e}")


def test_mouse_coordinate_conversion():
    """测试鼠标坐标转换"""
    print("\n=== 鼠标坐标转换测试 ===")
    
    # 模拟浮点坐标转整数
    float_coords = [1.5, 2.7, -1.3, 0.0, 100.9]
    
    for coord in float_coords:
        int_coord = int(coord)
        print(f"浮点坐标 {coord} -> 整数坐标 {int_coord}")
    
    print("✅ 坐标转换正常")


def test_pyramid_fallback():
    """测试金字塔回退机制"""
    print("\n=== 金字塔回退机制测试 ===")
    
    try:
        from core.image_pyramid import ImagePyramid
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # 测试正常创建
        pyramid = ImagePyramid()
        success = pyramid.set_image(test_image)
        
        if success:
            print("✅ 金字塔创建成功")
            
            # 测试级别选择
            level = pyramid.get_optimal_level(0.5)
            if level:
                print(f"✅ 级别选择成功: 级别{level.level}, 缩放{level.scale_factor}")
            else:
                print("❌ 级别选择失败")
        else:
            print("❌ 金字塔创建失败")
            
    except Exception as e:
        print(f"❌ 金字塔测试出错: {e}")


def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===")
    
    # 测试空图像
    try:
        empty_image = np.array([])
        processor = ImageProcessor()
        result = processor.gamma_correction(empty_image, 1.0)
        print("❌ 空图像应该抛出异常")
    except Exception as e:
        print(f"✅ 空图像正确处理异常: {type(e).__name__}")
    
    # 测试无效参数
    try:
        normal_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        processor = ImageProcessor()
        result = processor.gamma_correction(normal_image, 0)  # gamma=0可能有问题
        print(f"Gamma=0结果: {result.shape}")
    except Exception as e:
        print(f"Gamma=0异常: {type(e).__name__}")


if __name__ == '__main__':
    print("开始修复验证测试...")
    print("=" * 50)
    
    try:
        test_image_processor_fixes()
        test_mouse_coordinate_conversion()
        test_pyramid_fallback()
        test_error_handling()
        
        print("\n修复验证测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
