"""
缩放优化测试

验证图像金字塔缓存和缩放性能优化效果
"""

import sys
import os
import time
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_pyramid import ImagePyramid, get_pyramid_for_image, clear_all_pyramids


def test_pyramid_creation():
    """测试金字塔创建"""
    print("=== 图像金字塔创建测试 ===")
    
    # 创建不同大小的测试图像
    test_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096)
    ]
    
    for size in test_sizes:
        print(f"\n测试图像大小: {size}")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, size, dtype=np.uint8)
        
        # 创建金字塔
        pyramid = ImagePyramid(max_levels=6, max_memory_mb=100)
        
        start_time = time.time()
        success = pyramid.set_image(test_image)
        creation_time = time.time() - start_time
        
        if success:
            stats = pyramid.get_cache_stats()
            print(f"  创建时间: {creation_time:.3f}s")
            print(f"  金字塔级别: {stats['pyramid_levels']}")
            print(f"  内存使用: {stats['memory_usage_mb']:.2f}MB")
            print(f"  级别详情:")
            for level, details in stats['level_details'].items():
                print(f"    级别{level}: {details['size']}, 缩放{details['scale_factor']:.3f}, {details['memory_mb']:.2f}MB")
        else:
            print("  金字塔创建失败")


def test_pyramid_performance():
    """测试金字塔性能"""
    print("\n=== 金字塔性能测试 ===")
    
    # 创建大图像
    large_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    pyramid = ImagePyramid(max_levels=8, max_memory_mb=200)
    
    # 创建金字塔
    pyramid.set_image(large_image)
    
    # 测试不同缩放级别的访问性能
    scale_factors = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    
    print("缩放因子访问性能测试:")
    for scale_factor in scale_factors:
        start_time = time.time()
        
        # 多次访问同一级别
        for _ in range(10):
            level = pyramid.get_optimal_level(scale_factor)
            if level:
                pixmap = pyramid.get_pixmap_for_scale(scale_factor)
        
        access_time = (time.time() - start_time) / 10 * 1000  # 平均时间，转换为毫秒
        print(f"  缩放{scale_factor}: {access_time:.3f}ms/次")
    
    # 获取最终统计
    final_stats = pyramid.get_cache_stats()
    print(f"\n性能统计:")
    print(f"  缓存命中率: {final_stats['hit_rate']:.1%}")
    print(f"  总请求数: {final_stats['total_requests']}")
    print(f"  缓存命中: {final_stats['cache_hits']}")
    print(f"  缓存未命中: {final_stats['cache_misses']}")


def test_memory_management():
    """测试内存管理"""
    print("\n=== 内存管理测试 ===")
    
    # 创建多个金字塔实例
    image_ids = []
    pyramids = []
    
    for i in range(5):
        image_id = f"test_image_{i}"
        image_ids.append(image_id)
        
        # 创建不同大小的图像
        size = 512 * (i + 1)
        test_image = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        
        # 获取金字塔实例
        pyramid = get_pyramid_for_image(image_id)
        pyramid.set_image(test_image)
        pyramids.append(pyramid)
        
        stats = pyramid.get_cache_stats()
        print(f"图像{i+1} ({size}x{size}): {stats['memory_usage_mb']:.2f}MB, {stats['pyramid_levels']}级")
    
    # 测试缓存优化
    print("\n缓存优化前:")
    total_memory = sum(p.get_cache_stats()['memory_usage_mb'] for p in pyramids)
    print(f"总内存使用: {total_memory:.2f}MB")
    
    # 执行缓存优化
    for pyramid in pyramids:
        pyramid.optimize_cache()
    
    print("\n缓存优化后:")
    total_memory_after = sum(p.get_cache_stats()['memory_usage_mb'] for p in pyramids)
    print(f"总内存使用: {total_memory_after:.2f}MB")
    print(f"内存节省: {total_memory - total_memory_after:.2f}MB")
    
    # 清理
    clear_all_pyramids()


def test_scale_factor_selection():
    """测试缩放因子选择"""
    print("\n=== 缩放因子选择测试 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    pyramid = ImagePyramid(max_levels=6)
    pyramid.set_image(test_image)
    
    # 测试各种缩放因子
    test_scales = [4.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    
    print("缩放因子 -> 选择级别 (实际缩放因子):")
    for scale in test_scales:
        level = pyramid.get_optimal_level(scale)
        if level:
            print(f"  {scale:6.3f} -> 级别{level.level} ({level.scale_factor:.3f})")
        else:
            print(f"  {scale:6.3f} -> 无可用级别")
    
    # 获取统计信息
    stats = pyramid.get_cache_stats()
    print(f"\n访问统计:")
    print(f"  总访问次数: {stats['total_requests']}")
    print(f"  缓存命中率: {stats['hit_rate']:.1%}")


def test_pixmap_creation():
    """测试QPixmap创建"""
    print("\n=== QPixmap创建测试 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    pyramid = ImagePyramid()
    pyramid.set_image(test_image)
    
    # 测试不同级别的QPixmap创建
    scale_factors = [1.0, 0.5, 0.25, 0.125]
    
    for scale_factor in scale_factors:
        start_time = time.time()
        
        pixmap = pyramid.get_pixmap_for_scale(scale_factor)
        
        creation_time = (time.time() - start_time) * 1000
        
        if pixmap:
            print(f"  缩放{scale_factor}: {creation_time:.3f}ms, 大小{pixmap.width()}x{pixmap.height()}")
        else:
            print(f"  缩放{scale_factor}: 创建失败")


def benchmark_zoom_operations():
    """基准测试缩放操作"""
    print("\n=== 缩放操作基准测试 ===")
    
    # 创建大图像
    large_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    
    # 测试传统方法（直接缩放）
    print("传统方法（直接缩放）:")
    traditional_times = []
    for i in range(10):
        scale_factor = 0.5 ** (i % 4)  # 0.5, 0.25, 0.125, 0.0625
        
        start_time = time.time()
        # 模拟传统缩放（简单下采样）
        new_size = (int(large_image.shape[1] * scale_factor), 
                   int(large_image.shape[0] * scale_factor))
        if new_size[0] > 0 and new_size[1] > 0:
            import cv2
            scaled = cv2.resize(large_image, new_size, interpolation=cv2.INTER_AREA)
        
        operation_time = (time.time() - start_time) * 1000
        traditional_times.append(operation_time)
    
    avg_traditional = np.mean(traditional_times)
    print(f"  平均时间: {avg_traditional:.3f}ms")
    
    # 测试金字塔方法
    print("金字塔方法:")
    pyramid = ImagePyramid()
    pyramid.set_image(large_image)
    
    pyramid_times = []
    for i in range(10):
        scale_factor = 0.5 ** (i % 4)
        
        start_time = time.time()
        level = pyramid.get_optimal_level(scale_factor)
        if level:
            pixmap = pyramid.get_pixmap_for_scale(scale_factor)
        
        operation_time = (time.time() - start_time) * 1000
        pyramid_times.append(operation_time)
    
    avg_pyramid = np.mean(pyramid_times)
    print(f"  平均时间: {avg_pyramid:.3f}ms")
    
    # 性能对比
    speedup = avg_traditional / avg_pyramid if avg_pyramid > 0 else 0
    print(f"\n性能提升: {speedup:.1f}x")
    
    # 获取金字塔统计
    stats = pyramid.get_cache_stats()
    print(f"金字塔统计: {stats['pyramid_levels']}级, {stats['memory_usage_mb']:.2f}MB")


if __name__ == '__main__':
    print("开始缩放优化测试...")
    print("=" * 50)
    
    try:
        test_pyramid_creation()
        test_pyramid_performance()
        test_memory_management()
        test_scale_factor_selection()
        test_pixmap_creation()
        benchmark_zoom_operations()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
