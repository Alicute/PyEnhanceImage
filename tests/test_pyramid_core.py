"""
图像金字塔核心功能测试

专注测试金字塔缓存逻辑，不涉及QPixmap创建
"""

import sys
import os
import time
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_pyramid import ImagePyramid, get_pyramid_for_image, clear_all_pyramids


def test_pyramid_creation_core():
    """测试金字塔创建核心逻辑"""
    print("=== 图像金字塔创建核心测试 ===")
    
    # 创建不同大小的测试图像
    test_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for size in test_sizes:
        print(f"\n测试图像大小: {size}")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, size, dtype=np.uint8)
        
        # 创建金字塔（不创建QPixmap）
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
                print(f"    级别{level}: {details['size']}, 缩放{details['scale_factor']:.3f}")
        else:
            print("  金字塔创建失败")


def test_level_selection():
    """测试级别选择逻辑"""
    print("\n=== 级别选择测试 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    pyramid = ImagePyramid(max_levels=6)
    pyramid.set_image(test_image)
    
    # 测试各种缩放因子
    test_scales = [4.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.25, 0.125, 0.0625]
    
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


def test_performance_core():
    """测试核心性能"""
    print("\n=== 核心性能测试 ===")
    
    # 创建大图像
    large_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    pyramid = ImagePyramid(max_levels=8, max_memory_mb=200)
    
    # 创建金字塔
    start_time = time.time()
    pyramid.set_image(large_image)
    creation_time = time.time() - start_time
    print(f"金字塔创建时间: {creation_time:.3f}s")
    
    # 测试不同缩放级别的访问性能
    scale_factors = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    
    print("缩放因子访问性能测试:")
    for scale_factor in scale_factors:
        start_time = time.time()
        
        # 多次访问同一级别
        for _ in range(100):
            level = pyramid.get_optimal_level(scale_factor)
        
        access_time = (time.time() - start_time) / 100 * 1000  # 平均时间，转换为毫秒
        print(f"  缩放{scale_factor}: {access_time:.3f}ms/次")
    
    # 获取最终统计
    final_stats = pyramid.get_cache_stats()
    print(f"\n性能统计:")
    print(f"  缓存命中率: {final_stats['hit_rate']:.1%}")
    print(f"  总请求数: {final_stats['total_requests']}")
    print(f"  内存使用: {final_stats['memory_usage_mb']:.2f}MB")


def test_memory_efficiency():
    """测试内存效率"""
    print("\n=== 内存效率测试 ===")
    
    # 创建不同大小的图像并比较内存使用
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    for size in sizes:
        test_image = np.random.randint(0, 255, size, dtype=np.uint8)
        original_size_mb = test_image.nbytes / (1024 * 1024)
        
        pyramid = ImagePyramid(max_levels=6)
        pyramid.set_image(test_image)
        
        stats = pyramid.get_cache_stats()
        pyramid_size_mb = stats['memory_usage_mb']
        
        efficiency = pyramid_size_mb / original_size_mb
        
        print(f"图像{size}: 原始{original_size_mb:.2f}MB -> 金字塔{pyramid_size_mb:.2f}MB (效率{efficiency:.2f}x)")


def test_cache_optimization():
    """测试缓存优化"""
    print("\n=== 缓存优化测试 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    pyramid = ImagePyramid(max_levels=8)
    pyramid.set_image(test_image)
    
    # 模拟不均匀的访问模式
    access_patterns = [
        (1.0, 50),    # 原始大小，访问50次
        (0.5, 30),    # 50%缩放，访问30次
        (0.25, 10),   # 25%缩放，访问10次
        (0.125, 5),   # 12.5%缩放，访问5次
        (0.0625, 2),  # 6.25%缩放，访问2次
        (0.03125, 1), # 3.125%缩放，访问1次
    ]
    
    # 执行访问模式
    for scale_factor, count in access_patterns:
        for _ in range(count):
            pyramid.get_optimal_level(scale_factor)
    
    # 获取优化前统计
    stats_before = pyramid.get_cache_stats()
    print(f"优化前: {stats_before['pyramid_levels']}级, {stats_before['memory_usage_mb']:.2f}MB")
    
    # 执行缓存优化
    pyramid.optimize_cache()
    
    # 获取优化后统计
    stats_after = pyramid.get_cache_stats()
    print(f"优化后: {stats_after['pyramid_levels']}级, {stats_after['memory_usage_mb']:.2f}MB")
    
    memory_saved = stats_before['memory_usage_mb'] - stats_after['memory_usage_mb']
    print(f"内存节省: {memory_saved:.2f}MB")


def benchmark_downsample_methods():
    """基准测试下采样方法"""
    print("\n=== 下采样方法基准测试 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    
    # 测试不同的下采样方法
    import cv2
    
    methods = [
        ('INTER_AREA', cv2.INTER_AREA),
        ('INTER_LINEAR', cv2.INTER_LINEAR),
        ('INTER_CUBIC', cv2.INTER_CUBIC),
    ]
    
    target_size = (1024, 1024)
    
    for method_name, method in methods:
        times = []
        for _ in range(5):
            start_time = time.time()
            result = cv2.resize(test_image, target_size, interpolation=method)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000
        print(f"  {method_name}: {avg_time:.3f}ms")


def test_global_pyramid_management():
    """测试全局金字塔管理"""
    print("\n=== 全局金字塔管理测试 ===")
    
    # 创建多个图像的金字塔
    image_ids = []
    for i in range(3):
        image_id = f"test_image_{i}"
        image_ids.append(image_id)
        
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        pyramid = get_pyramid_for_image(image_id)
        pyramid.set_image(test_image)
        
        stats = pyramid.get_cache_stats()
        print(f"图像{i}: {stats['pyramid_levels']}级, {stats['memory_usage_mb']:.2f}MB")
    
    # 测试重复获取
    pyramid_1a = get_pyramid_for_image(image_ids[0])
    pyramid_1b = get_pyramid_for_image(image_ids[0])
    
    print(f"重复获取测试: {pyramid_1a is pyramid_1b}")  # 应该是True
    
    # 清理所有金字塔
    clear_all_pyramids()
    print("已清理所有金字塔")


if __name__ == '__main__':
    print("开始图像金字塔核心功能测试...")
    print("=" * 50)
    
    try:
        test_pyramid_creation_core()
        test_level_selection()
        test_performance_core()
        test_memory_efficiency()
        test_cache_optimization()
        benchmark_downsample_methods()
        test_global_pyramid_management()
        
        print("\n所有核心测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
