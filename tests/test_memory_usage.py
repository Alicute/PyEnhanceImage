"""
内存使用测试

验证窗宽窗位调节的内存使用情况
"""

import sys
import os
import time
import numpy as np
import psutil
import gc

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.window_level_lut import get_global_lut, clear_all_pyramids
from core.image_manager import ImageManager


def get_memory_usage():
    """获取当前内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_lut_memory_usage():
    """测试LUT内存使用"""
    print("=== LUT内存使用测试 ===")
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存: {initial_memory:.2f}MB")
    
    # 获取LUT实例
    lut = get_global_lut()
    
    # 模拟频繁的窗宽窗位调节
    print("模拟频繁窗宽窗位调节...")
    for i in range(100):
        ww = 400 + i * 10
        wl = 40 + i * 5
        lut.get_lut(ww, wl)
        
        if i % 20 == 0:
            current_memory = get_memory_usage()
            cache_stats = lut.get_cache_stats()
            print(f"  调节{i}次: 内存{current_memory:.2f}MB, 缓存{cache_stats['cache_size']}项")
    
    # 最终内存使用
    final_memory = get_memory_usage()
    cache_stats = lut.get_cache_stats()
    
    print(f"最终内存: {final_memory:.2f}MB")
    print(f"内存增长: {final_memory - initial_memory:.2f}MB")
    print(f"缓存统计: {cache_stats}")
    
    # 清理缓存
    lut.clear_cache()
    gc.collect()
    
    after_clear_memory = get_memory_usage()
    print(f"清理后内存: {after_clear_memory:.2f}MB")
    print(f"释放内存: {final_memory - after_clear_memory:.2f}MB")


def test_image_manager_memory():
    """测试ImageManager内存使用"""
    print("\n=== ImageManager内存使用测试 ===")
    
    initial_memory = get_memory_usage()
    print(f"初始内存: {initial_memory:.2f}MB")
    
    # 创建ImageManager
    manager = ImageManager()
    
    # 创建模拟DICOM数据
    test_data = np.random.randint(0, 4096, (1024, 1024), dtype=np.uint16)
    
    # 模拟加载图像
    print("模拟加载图像...")
    manager.original_image = manager.ImageData(
        id="test_image",
        data=test_data,
        window_width=400,
        window_level=40
    )
    manager.current_image = manager.original_image
    
    load_memory = get_memory_usage()
    print(f"加载后内存: {load_memory:.2f}MB")
    
    # 模拟频繁的窗宽窗位调节
    print("模拟频繁窗宽窗位调节...")
    for i in range(50):
        ww = 400 + i * 20
        wl = 40 + i * 10
        
        manager.update_window_settings(ww, wl)
        display_data = manager.get_windowed_image(manager.current_image)
        
        if i % 10 == 0:
            current_memory = get_memory_usage()
            print(f"  调节{i}次: 内存{current_memory:.2f}MB")
    
    final_memory = get_memory_usage()
    print(f"最终内存: {final_memory:.2f}MB")
    print(f"内存增长: {final_memory - initial_memory:.2f}MB")
    
    # 清理缓存
    manager.original_display_cache = None
    manager.current_display_cache = None
    gc.collect()
    
    after_clear_memory = get_memory_usage()
    print(f"清理后内存: {after_clear_memory:.2f}MB")


def test_memory_leak_simulation():
    """模拟内存泄漏测试"""
    print("\n=== 内存泄漏模拟测试 ===")
    
    initial_memory = get_memory_usage()
    print(f"初始内存: {initial_memory:.2f}MB")
    
    # 模拟连续加载多个图像
    for image_idx in range(5):
        print(f"\n加载图像 {image_idx + 1}...")
        
        # 清理之前的缓存
        lut = get_global_lut()
        lut.clear_cache()
        clear_all_pyramids()
        gc.collect()
        
        # 创建新的图像数据
        size = 512 + image_idx * 256
        test_data = np.random.randint(0, 4096, (size, size), dtype=np.uint16)
        
        # 创建ImageManager
        manager = ImageManager()
        manager.original_image = manager.ImageData(
            id=f"test_image_{image_idx}",
            data=test_data,
            window_width=400,
            window_level=40
        )
        manager.current_image = manager.original_image
        
        # 模拟窗宽窗位调节
        for i in range(20):
            ww = 400 + i * 50
            wl = 40 + i * 25
            manager.update_window_settings(ww, wl)
            display_data = manager.get_windowed_image(manager.current_image)
        
        current_memory = get_memory_usage()
        print(f"  图像{image_idx + 1} ({size}x{size}): 内存{current_memory:.2f}MB")
    
    final_memory = get_memory_usage()
    print(f"\n最终内存: {final_memory:.2f}MB")
    print(f"总内存增长: {final_memory - initial_memory:.2f}MB")
    
    # 最终清理
    lut = get_global_lut()
    lut.clear_cache()
    clear_all_pyramids()
    gc.collect()
    
    after_final_clear = get_memory_usage()
    print(f"最终清理后内存: {after_final_clear:.2f}MB")
    print(f"可释放内存: {final_memory - after_final_clear:.2f}MB")


def test_performance_vs_memory():
    """性能与内存权衡测试"""
    print("\n=== 性能与内存权衡测试 ===")
    
    # 测试不同缓存大小的影响
    cache_sizes = [5, 10, 20, 50]
    
    for cache_size in cache_sizes:
        print(f"\n测试缓存大小: {cache_size}")
        
        # 清理之前的缓存
        lut = get_global_lut()
        lut.clear_cache()
        lut.max_cache_size = cache_size
        gc.collect()
        
        initial_memory = get_memory_usage()
        
        # 模拟窗宽窗位调节
        start_time = time.time()
        for i in range(100):
            ww = 400 + (i % 30) * 10  # 重复一些值以测试缓存效果
            wl = 40 + (i % 20) * 5
            lut.get_lut(ww, wl)
        
        end_time = time.time()
        final_memory = get_memory_usage()
        
        stats = lut.get_cache_stats()
        
        print(f"  处理时间: {(end_time - start_time) * 1000:.3f}ms")
        print(f"  内存使用: {final_memory - initial_memory:.2f}MB")
        print(f"  缓存命中率: {stats['hit_rate']:.1%}")
        print(f"  缓存大小: {stats['cache_size']}")


if __name__ == '__main__':
    print("开始内存使用测试...")
    print("=" * 50)
    
    try:
        test_lut_memory_usage()
        test_image_manager_memory()
        test_memory_leak_simulation()
        test_performance_vs_memory()
        
        print("\n内存测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
