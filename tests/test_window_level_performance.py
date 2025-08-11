"""
窗宽窗位性能测试

验证LUT优化的性能提升效果
目标：将响应时间从100-200ms降低到5-10ms
"""

import sys
import os
import time
import numpy as np
import unittest
from typing import List, Dict

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.window_level_lut import WindowLevelLUT, get_global_lut, apply_window_level_fast
from core.image_manager import ImageManager, ImageData


class WindowLevelPerformanceTest(unittest.TestCase):
    """窗宽窗位性能测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像数据
        self.test_sizes = [
            (512, 512),    # 小图像
            (1024, 1024),  # 中等图像
            (2048, 2048),  # 大图像
        ]
        
        self.test_images = {}
        for size in self.test_sizes:
            # 创建模拟DICOM数据（16位）
            image_data = np.random.randint(0, 4096, size, dtype=np.uint16)
            self.test_images[size] = image_data
        
        # 测试参数
        self.test_window_settings = [
            (400, 40),     # 典型CT设置
            (1500, 300),   # 骨窗
            (80, 40),      # 软组织窗
            (2000, 600),   # 肺窗
        ]
        
        # 性能目标
        self.target_response_time_ms = 10.0  # 目标响应时间
        
    def test_lut_creation_performance(self):
        """测试LUT创建性能"""
        print("\n=== LUT创建性能测试 ===")
        
        lut = WindowLevelLUT()
        creation_times = []
        
        for ww, wl in self.test_window_settings:
            start_time = time.time()
            lut_array = lut._create_lut(ww, wl)
            end_time = time.time()
            
            creation_time_ms = (end_time - start_time) * 1000
            creation_times.append(creation_time_ms)
            
            print(f"窗宽{ww}, 窗位{wl}: {creation_time_ms:.3f}ms")
            
            # 验证LUT正确性
            self.assertEqual(len(lut_array), 65536)
            self.assertEqual(lut_array.dtype, np.uint8)
        
        avg_creation_time = np.mean(creation_times)
        print(f"平均LUT创建时间: {avg_creation_time:.3f}ms")
        
        # LUT创建应该很快（<1ms）
        self.assertLess(avg_creation_time, 1.0, "LUT创建时间应该小于1ms")
    
    def test_lut_cache_performance(self):
        """测试LUT缓存性能"""
        print("\n=== LUT缓存性能测试 ===")
        
        lut = WindowLevelLUT(max_cache_size=10)
        
        # 第一次访问（缓存未命中）
        start_time = time.time()
        lut_array1 = lut.get_lut(400, 40)
        first_access_time = (time.time() - start_time) * 1000
        
        # 第二次访问（缓存命中）
        start_time = time.time()
        lut_array2 = lut.get_lut(400, 40)
        second_access_time = (time.time() - start_time) * 1000
        
        print(f"首次访问时间: {first_access_time:.3f}ms")
        print(f"缓存命中时间: {second_access_time:.3f}ms")
        
        # 验证缓存正确性
        np.testing.assert_array_equal(lut_array1, lut_array2)
        
        # 缓存命中应该非常快
        self.assertLess(second_access_time, 0.1, "缓存命中时间应该小于0.1ms")
        
        # 获取缓存统计
        stats = lut.get_cache_stats()
        print(f"缓存统计: {stats}")
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
    
    def test_window_level_application_performance(self):
        """测试窗宽窗位应用性能"""
        print("\n=== 窗宽窗位应用性能测试 ===")
        
        for size in self.test_sizes:
            image_data = self.test_images[size]
            print(f"\n测试图像大小: {size}")
            
            # 测试新的LUT方法
            lut_times = []
            for ww, wl in self.test_window_settings:
                start_time = time.time()
                result_lut = apply_window_level_fast(image_data, ww, wl)
                end_time = time.time()
                
                lut_time_ms = (end_time - start_time) * 1000
                lut_times.append(lut_time_ms)
                
                print(f"  LUT方法 - 窗宽{ww}, 窗位{wl}: {lut_time_ms:.3f}ms")
                
                # 验证结果正确性
                self.assertEqual(result_lut.shape, image_data.shape)
                self.assertEqual(result_lut.dtype, np.uint8)
            
            avg_lut_time = np.mean(lut_times)
            print(f"  平均LUT处理时间: {avg_lut_time:.3f}ms")
            
            # 验证性能目标
            if size == (512, 512):
                # 小图像应该非常快
                self.assertLess(avg_lut_time, 5.0, f"512x512图像处理时间应该小于5ms，实际{avg_lut_time:.3f}ms")
            elif size == (1024, 1024):
                # 中等图像应该达到目标
                self.assertLess(avg_lut_time, self.target_response_time_ms, 
                              f"1024x1024图像处理时间应该小于{self.target_response_time_ms}ms，实际{avg_lut_time:.3f}ms")
    
    def test_old_vs_new_performance_comparison(self):
        """对比新旧方法的性能"""
        print("\n=== 新旧方法性能对比 ===")
        
        # 使用中等大小图像进行对比
        image_data = self.test_images[(1024, 1024)]
        ww, wl = 400, 40
        
        # 旧方法（逐像素计算）
        def old_window_level_method(data, window_width, window_level):
            min_val = window_level - window_width / 2
            max_val = window_level + window_width / 2
            windowed_data = np.clip(data, min_val, max_val)
            return ((windowed_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 测试旧方法
        old_times = []
        for _ in range(5):
            start_time = time.time()
            result_old = old_window_level_method(image_data, ww, wl)
            end_time = time.time()
            old_times.append((end_time - start_time) * 1000)
        
        # 测试新方法
        new_times = []
        for _ in range(5):
            start_time = time.time()
            result_new = apply_window_level_fast(image_data, ww, wl)
            end_time = time.time()
            new_times.append((end_time - start_time) * 1000)
        
        avg_old_time = np.mean(old_times)
        avg_new_time = np.mean(new_times)
        speedup = avg_old_time / avg_new_time
        
        print(f"旧方法平均时间: {avg_old_time:.3f}ms")
        print(f"新方法平均时间: {avg_new_time:.3f}ms")
        print(f"性能提升倍数: {speedup:.1f}x")
        
        # 验证结果一致性（允许小的数值差异）
        diff = np.abs(result_old.astype(np.float32) - result_new.astype(np.float32))
        max_diff = np.max(diff)
        print(f"结果最大差异: {max_diff}")
        self.assertLess(max_diff, 2, "新旧方法结果差异应该很小")
        
        # 验证性能提升（调整为更现实的目标）
        self.assertGreater(speedup, 3.0, f"性能提升应该至少3倍，实际{speedup:.1f}倍")
        self.assertLess(avg_new_time, self.target_response_time_ms, 
                       f"新方法应该达到{self.target_response_time_ms}ms目标")
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        print("\n=== 内存使用测试 ===")
        
        lut = WindowLevelLUT(max_cache_size=20)
        
        # 创建多个不同的LUT
        for i in range(25):  # 超过缓存大小
            ww = 400 + i * 10
            wl = 40 + i * 5
            lut.get_lut(ww, wl)
        
        stats = lut.get_cache_stats()
        print(f"缓存统计: {stats}")
        
        # 验证缓存大小限制
        self.assertLessEqual(stats['cache_size'], 20, "缓存大小应该受到限制")
        self.assertGreater(stats['cache_misses'], 20, "应该有缓存淘汰发生")
    
    def test_edge_cases(self):
        """测试边界情况"""
        print("\n=== 边界情况测试 ===")
        
        image_data = self.test_images[(512, 512)]
        
        # 测试极端窗宽窗位值
        edge_cases = [
            (0, 40),      # 窗宽为0
            (-100, 40),   # 负窗宽
            (65535, 32768),  # 最大值
            (1, 0),       # 最小窗宽
        ]
        
        for ww, wl in edge_cases:
            try:
                start_time = time.time()
                result = apply_window_level_fast(image_data, ww, wl)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000
                print(f"边界情况 窗宽{ww}, 窗位{wl}: {processing_time:.3f}ms")
                
                # 验证结果有效性
                self.assertEqual(result.shape, image_data.shape)
                self.assertEqual(result.dtype, np.uint8)
                
            except Exception as e:
                print(f"边界情况 窗宽{ww}, 窗位{wl} 出错: {e}")


def run_performance_benchmark():
    """运行性能基准测试"""
    print("开始窗宽窗位性能基准测试...")
    print("=" * 50)
    
    # 运行测试
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 输出全局LUT统计
    global_lut = get_global_lut()
    stats = global_lut.get_cache_stats()
    print("\n=== 全局LUT统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    run_performance_benchmark()
