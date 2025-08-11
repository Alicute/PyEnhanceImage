"""
性能测试

测试窗宽窗位调节的性能
"""

import sys
import os
import time
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.window_level_lut import get_global_lut
from core.image_manager import ImageManager


def test_lut_performance():
    """测试LUT性能"""
    print("=== LUT性能测试 ===")
    
    # 创建测试图像
    sizes = [
        (512, 512),      # 0.25M像素
        (1024, 1024),    # 1M像素  
        (2048, 2048),    # 4M像素
        (4096, 4096),    # 16M像素
    ]
    
    lut = get_global_lut()
    
    for size in sizes:
        print(f"\n测试图像大小: {size[0]}x{size[1]} ({size[0]*size[1]/1024/1024:.1f}M像素)")
        
        # 创建测试数据
        test_data = np.random.randint(0, 4096, size, dtype=np.uint16)
        data_size_mb = test_data.nbytes / 1024 / 1024
        print(f"数据大小: {data_size_mb:.2f}MB")
        
        # 测试不同窗宽窗位设置
        test_settings = [
            (400, 40),
            (1500, 300),
            (80, 40),
            (3000, 1500),
        ]
        
        total_time = 0
        for ww, wl in test_settings:
            start_time = time.time()
            result = lut.apply_lut(test_data, ww, wl)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            total_time += processing_time
            
            print(f"  WW:{ww:4d} WL:{wl:4d} -> {processing_time:6.2f}ms")
        
        avg_time = total_time / len(test_settings)
        throughput = data_size_mb / (avg_time / 1000)
        
        print(f"平均处理时间: {avg_time:.2f}ms")
        print(f"处理吞吐量: {throughput:.1f}MB/s")
        
        # 性能评估
        if avg_time < 50:
            print("✅ 性能优秀")
        elif avg_time < 200:
            print("⚠️ 性能一般")
        else:
            print("❌ 性能较差")


def test_image_manager_performance():
    """测试ImageManager性能"""
    print("\n=== ImageManager性能测试 ===")
    
    # 创建ImageManager
    manager = ImageManager()
    
    # 创建测试图像
    test_data = np.random.randint(0, 4096, (2048, 2048), dtype=np.uint16)
    
    manager.original_image = manager.ImageData(
        id="test_image",
        data=test_data,
        window_width=400,
        window_level=40
    )
    manager.current_image = manager.original_image
    
    print(f"测试图像: 2048x2048 ({test_data.nbytes / 1024 / 1024:.2f}MB)")
    
    # 测试连续的窗宽窗位调节
    print("\n连续窗宽窗位调节测试:")
    
    settings_sequence = [
        (400, 40),
        (500, 50),
        (600, 60),
        (700, 70),
        (800, 80),
        (900, 90),
        (1000, 100),
        (1100, 110),
        (1200, 120),
        (1300, 130),
    ]
    
    total_time = 0
    for i, (ww, wl) in enumerate(settings_sequence, 1):
        start_time = time.time()
        
        manager.update_window_settings(ww, wl)
        result = manager.get_windowed_image(manager.current_image)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        total_time += processing_time
        
        print(f"  第{i:2d}次调节 WW:{ww:4d} WL:{wl:3d} -> {processing_time:6.2f}ms")
    
    avg_time = total_time / len(settings_sequence)
    print(f"\n平均调节时间: {avg_time:.2f}ms")
    
    # 性能评估
    if avg_time < 20:
        print("✅ 响应速度优秀 (< 20ms)")
    elif avg_time < 50:
        print("⚠️ 响应速度一般 (20-50ms)")
    else:
        print("❌ 响应速度较慢 (> 50ms)")


def test_cache_effectiveness():
    """测试缓存效果"""
    print("\n=== 缓存效果测试 ===")
    
    lut = get_global_lut()
    lut.clear_cache()
    
    # 创建测试数据
    test_data = np.random.randint(0, 4096, (1024, 1024), dtype=np.uint16)
    
    # 测试重复访问
    ww, wl = 400, 40
    
    print("第一次访问（建立缓存）:")
    start_time = time.time()
    result1 = lut.apply_lut(test_data, ww, wl)
    first_time = (time.time() - start_time) * 1000
    print(f"  处理时间: {first_time:.2f}ms")
    
    print("第二次访问（缓存命中）:")
    start_time = time.time()
    result2 = lut.apply_lut(test_data, ww, wl)
    second_time = (time.time() - start_time) * 1000
    print(f"  处理时间: {second_time:.2f}ms")
    
    if second_time > 0:
        speedup = first_time / second_time
        print(f"缓存加速比: {speedup:.1f}x")
    else:
        print("缓存加速比: ∞x (几乎瞬时)")
    
    # 验证结果一致性
    if np.array_equal(result1, result2):
        print("✅ 缓存结果一致")
    else:
        print("❌ 缓存结果不一致")
    
    # 缓存统计
    stats = lut.get_cache_stats()
    print(f"缓存统计: {stats}")


def test_memory_vs_performance():
    """测试内存与性能的平衡"""
    print("\n=== 内存与性能平衡测试 ===")
    
    # 测试不同大小的图像
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    
    for size in sizes:
        print(f"\n测试图像: {size[0]}x{size[1]}")
        
        test_data = np.random.randint(0, 4096, size, dtype=np.uint16)
        data_size_mb = test_data.nbytes / 1024 / 1024
        
        lut = get_global_lut()
        
        # 测试处理时间
        start_time = time.time()
        result = lut.apply_lut(test_data, 400, 40)
        processing_time = (time.time() - start_time) * 1000
        
        # 估算内存使用
        if test_data.dtype != np.uint8:
            # 需要类型转换的额外内存
            conversion_memory = test_data.nbytes / 1024 / 1024
        else:
            conversion_memory = 0
        
        result_memory = result.nbytes / 1024 / 1024
        total_memory = data_size_mb + conversion_memory + result_memory
        
        print(f"  数据大小: {data_size_mb:.2f}MB")
        print(f"  处理时间: {processing_time:.2f}ms")
        print(f"  内存使用: {total_memory:.2f}MB")
        print(f"  处理效率: {data_size_mb / (processing_time / 1000):.1f}MB/s")
        
        # 性能评级
        if processing_time < 100 and total_memory < data_size_mb * 3:
            print("  ✅ 内存和性能平衡良好")
        elif processing_time < 500:
            print("  ⚠️ 性能可接受，内存使用合理")
        else:
            print("  ❌ 需要进一步优化")


if __name__ == '__main__':
    print("开始性能测试...")
    print("=" * 60)
    
    try:
        test_lut_performance()
        test_image_manager_performance()
        test_cache_effectiveness()
        test_memory_vs_performance()
        
        print("\n" + "=" * 60)
        print("性能测试完成！")
        
        print("\n性能优化建议:")
        print("1. 如果平均处理时间 > 50ms，考虑进一步优化算法")
        print("2. 如果内存使用 > 原图3倍，检查是否有不必要的拷贝")
        print("3. 缓存命中率应该 > 80% 以获得最佳性能")
        print("4. 大图像(>4M像素)处理时间应该 < 200ms")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
