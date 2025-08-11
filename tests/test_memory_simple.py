"""
简化内存使用测试

验证窗宽窗位调节的内存使用情况（不依赖psutil）
"""

import sys
import os
import time
import numpy as np
import gc

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.window_level_lut import get_global_lut
from core.image_manager import ImageManager


def test_lut_cache_behavior():
    """测试LUT缓存行为"""
    print("=== LUT缓存行为测试 ===")
    
    lut = get_global_lut()
    lut.clear_cache()
    
    print(f"初始缓存大小: {lut.max_cache_size}")
    
    # 模拟频繁的窗宽窗位调节
    print("模拟频繁窗宽窗位调节...")
    for i in range(50):
        ww = 400 + i * 10
        wl = 40 + i * 5
        lut.get_lut(ww, wl)
        
        if i % 10 == 0:
            stats = lut.get_cache_stats()
            print(f"  调节{i}次: 缓存{stats['cache_size']}项, 命中率{stats['hit_rate']:.1%}")
    
    # 最终统计
    final_stats = lut.get_cache_stats()
    print(f"最终缓存统计: {final_stats}")
    
    # 测试缓存限制
    print(f"缓存是否受限制: {final_stats['cache_size'] <= lut.max_cache_size}")
    
    return final_stats


def test_repeated_access():
    """测试重复访问的缓存效果"""
    print("\n=== 重复访问缓存效果测试 ===")
    
    lut = get_global_lut()
    lut.clear_cache()
    
    # 测试重复访问相同的窗宽窗位
    common_settings = [(400, 40), (1500, 300), (80, 40)]
    
    # 第一轮：建立缓存
    print("第一轮访问（建立缓存）:")
    start_time = time.time()
    for ww, wl in common_settings * 10:  # 重复10次
        lut.get_lut(ww, wl)
    first_round_time = time.time() - start_time
    
    stats_after_first = lut.get_cache_stats()
    print(f"  时间: {first_round_time * 1000:.3f}ms")
    print(f"  缓存命中率: {stats_after_first['hit_rate']:.1%}")
    
    # 第二轮：应该全部命中缓存
    print("第二轮访问（应该全部命中缓存）:")
    start_time = time.time()
    for ww, wl in common_settings * 10:  # 重复10次
        lut.get_lut(ww, wl)
    second_round_time = time.time() - start_time
    
    stats_after_second = lut.get_cache_stats()
    print(f"  时间: {second_round_time * 1000:.3f}ms")
    print(f"  缓存命中率: {stats_after_second['hit_rate']:.1%}")
    
    speedup = first_round_time / second_round_time if second_round_time > 0 else float('inf')
    print(f"  缓存加速比: {speedup:.1f}x")


def test_image_manager_caching():
    """测试ImageManager缓存"""
    print("\n=== ImageManager缓存测试 ===")
    
    # 创建ImageManager
    manager = ImageManager()
    
    # 创建测试图像
    test_data = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
    
    manager.original_image = manager.ImageData(
        id="test_image",
        data=test_data,
        window_width=400,
        window_level=40
    )
    manager.current_image = manager.original_image
    
    print("测试ImageManager缓存效果...")
    
    # 第一次调用
    start_time = time.time()
    result1 = manager.get_windowed_image(manager.current_image)
    first_time = time.time() - start_time
    
    # 第二次调用（应该使用缓存）
    start_time = time.time()
    result2 = manager.get_windowed_image(manager.current_image)
    second_time = time.time() - start_time
    
    print(f"第一次调用: {first_time * 1000:.3f}ms")
    print(f"第二次调用: {second_time * 1000:.3f}ms")
    
    # 验证结果一致性
    if np.array_equal(result1, result2):
        print("✅ 缓存结果一致")
    else:
        print("❌ 缓存结果不一致")
    
    # 改变窗宽窗位
    manager.update_window_settings(500, 50)
    
    # 第三次调用（应该重新计算）
    start_time = time.time()
    result3 = manager.get_windowed_image(manager.current_image)
    third_time = time.time() - start_time
    
    print(f"改变设置后调用: {third_time * 1000:.3f}ms")
    
    if not np.array_equal(result1, result3):
        print("✅ 设置改变后结果正确更新")
    else:
        print("❌ 设置改变后结果未更新")


def test_memory_cleanup():
    """测试内存清理"""
    print("\n=== 内存清理测试 ===")
    
    # 创建大量缓存
    lut = get_global_lut()
    lut.clear_cache()
    
    print("创建大量LUT缓存...")
    for i in range(100):
        ww = 400 + i * 10
        wl = 40 + i * 5
        lut.get_lut(ww, wl)
    
    stats_before = lut.get_cache_stats()
    print(f"清理前缓存: {stats_before['cache_size']}项")
    
    # 清理缓存
    lut.clear_cache()
    gc.collect()
    
    stats_after = lut.get_cache_stats()
    print(f"清理后缓存: {stats_after['cache_size']}项")
    
    if stats_after['cache_size'] == 0:
        print("✅ 缓存清理成功")
    else:
        print("❌ 缓存清理失败")


def test_cache_size_optimization():
    """测试缓存大小优化"""
    print("\n=== 缓存大小优化测试 ===")
    
    # 测试不同缓存大小的效果
    cache_sizes = [5, 10, 20, 50]
    
    for cache_size in cache_sizes:
        print(f"\n测试缓存大小: {cache_size}")
        
        lut = get_global_lut()
        lut.clear_cache()
        lut.max_cache_size = cache_size
        
        # 模拟访问模式：有些重复，有些新的
        access_pattern = []
        for i in range(50):
            if i < cache_size:
                # 前面的访问建立基础缓存
                access_pattern.append((400 + i * 10, 40 + i * 5))
            else:
                # 后面的访问混合重复和新的
                if i % 3 == 0:
                    # 重复访问
                    idx = i % cache_size
                    access_pattern.append((400 + idx * 10, 40 + idx * 5))
                else:
                    # 新访问
                    access_pattern.append((400 + i * 10, 40 + i * 5))
        
        # 执行访问
        start_time = time.time()
        for ww, wl in access_pattern:
            lut.get_lut(ww, wl)
        total_time = time.time() - start_time
        
        stats = lut.get_cache_stats()
        print(f"  总时间: {total_time * 1000:.3f}ms")
        print(f"  缓存命中率: {stats['hit_rate']:.1%}")
        print(f"  实际缓存大小: {stats['cache_size']}")


if __name__ == '__main__':
    print("开始简化内存使用测试...")
    print("=" * 50)
    
    try:
        test_lut_cache_behavior()
        test_repeated_access()
        test_image_manager_caching()
        test_memory_cleanup()
        test_cache_size_optimization()
        
        print("\n简化内存测试完成！")
        print("\n建议:")
        print("1. 缓存大小20已经足够，避免设置过大")
        print("2. 防抖动控制器可以减少不必要的计算")
        print("3. 定期清理缓存可以控制内存使用")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
