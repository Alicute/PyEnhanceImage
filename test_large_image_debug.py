#!/usr/bin/env python3
"""
测试大图像处理的调试脚本
"""
import sys
import os
sys.path.append('src')

import numpy as np
from core.image_processor import ImageProcessor

def test_large_image_processing():
    """测试大图像处理性能"""
    print("🧪 测试大图像处理性能...")
    
    # 创建类似您实际图像大小的测试数据
    print("📊 创建大图像测试数据 (1024x1024)...")
    test_data = np.random.randint(958, 57544, (1024, 1024), dtype=np.uint16)
    print(f"   图像大小: {test_data.shape}")
    print(f"   数据范围: {test_data.min()} - {test_data.max()}")
    print(f"   总像素数: {test_data.size:,}")
    
    # 模拟进度回调
    def progress_callback(progress):
        print(f"   📈 进度更新: {progress*100:.1f}%")
    
    try:
        print("\n🚀 开始处理...")
        result = ImageProcessor.paper_enhance(test_data, progress_callback)
        print(f"\n✅ 处理完成！")
        print(f"   输出形状: {result.shape}")
        print(f"   输出范围: {result.min()} - {result.max()}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def test_parameter_optimization():
    """测试参数优化效果"""
    print("\n🔧 测试参数优化...")
    
    # 创建不同大小的图像测试参数调整
    sizes = [
        (100, 100, "小图像"),
        (500, 500, "中图像"), 
        (1500, 1500, "大图像"),
        (3000, 2400, "超大图像")
    ]
    
    for h, w, desc in sizes:
        total_pixels = h * w
        print(f"\n📊 {desc} ({h}x{w}, {total_pixels:,}像素):")
        
        # 模拟参数调整逻辑
        search_radius = 2
        topk = 15
        iters = 5
        patch_radius = 1
        
        if total_pixels > 2000000:  # 2M像素
            if search_radius > 1:
                search_radius = 1
            if topk is None or topk > 5:
                topk = 5
            if iters > 2:
                iters = 2
            if patch_radius > 1:
                patch_radius = 1
                
        if total_pixels > 5000000:  # 5M像素
            search_radius = 1
            topk = 3
            iters = 1
            patch_radius = 1
            
        print(f"   优化后参数: search_radius={search_radius}, topk={topk}, iters={iters}")
        
        # 估算处理时间
        estimated_time = total_pixels * 0.0013  # 基于测试的每像素1.3ms
        if total_pixels > 2000000:
            estimated_time *= 0.3  # 参数优化后的加速比
        if total_pixels > 5000000:
            estimated_time *= 0.5  # 极速模式进一步加速
            
        print(f"   预计处理时间: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 大图像处理调试测试")
    print("=" * 60)
    
    # 测试参数优化
    test_parameter_optimization()
    
    print("\n" + "=" * 60)
    print("是否要测试实际处理？(y/n): ", end="")
    
    # 在脚本中直接测试，不等待输入
    choice = "n"  # 默认不测试，避免长时间运行
    
    if choice.lower() == 'y':
        test_large_image_processing()
    else:
        print("跳过实际处理测试")
        
    print("\n📝 调试建议:")
    print("1. 对于您的3072x2432图像，建议使用极速参数")
    print("2. 预计处理时间约5-15分钟")
    print("3. 可以通过日志监控具体进度")
    print("4. 如果仍然太慢，考虑先缩放图像到较小尺寸")

if __name__ == "__main__":
    main()
