"""
内存监控工具

实时监控内存使用情况，帮助诊断内存泄漏问题
"""

import gc
import sys
import tracemalloc
from typing import Dict, List, Optional
import numpy as np


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.is_tracing = False
        
    def start_tracing(self):
        """开始内存追踪"""
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
            print("内存追踪已启动")
    
    def stop_tracing(self):
        """停止内存追踪"""
        if self.is_tracing:
            tracemalloc.stop()
            self.is_tracing = False
            print("内存追踪已停止")
    
    def take_snapshot(self, label: str = ""):
        """拍摄内存快照"""
        if not self.is_tracing:
            self.start_tracing()
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)
        
        # 获取当前内存使用
        current, peak = tracemalloc.get_traced_memory()
        
        print(f"内存快照 [{label}]: 当前 {current / 1024 / 1024:.2f}MB, 峰值 {peak / 1024 / 1024:.2f}MB")
        
        return len(self.snapshots) - 1
    
    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int, top_n: int = 10):
        """比较两个快照的差异"""
        if snapshot1_idx >= len(self.snapshots) or snapshot2_idx >= len(self.snapshots):
            print("快照索引无效")
            return
        
        snapshot1 = self.snapshots[snapshot1_idx]
        snapshot2 = self.snapshots[snapshot2_idx]
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        print(f"\n内存差异分析 (快照{snapshot1_idx} -> 快照{snapshot2_idx}):")
        print("=" * 60)
        
        for index, stat in enumerate(top_stats[:top_n], 1):
            print(f"{index}. {stat}")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        # 强制垃圾回收
        gc.collect()
        
        # 获取Python对象内存
        if self.is_tracing:
            current, peak = tracemalloc.get_traced_memory()
            traced_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
        else:
            traced_mb = 0
            peak_mb = 0
        
        # 获取NumPy数组内存使用
        numpy_arrays = []
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                numpy_arrays.append(obj)
        
        numpy_memory = sum(arr.nbytes for arr in numpy_arrays) / 1024 / 1024
        
        return {
            'traced_memory_mb': traced_mb,
            'peak_memory_mb': peak_mb,
            'numpy_memory_mb': numpy_memory,
            'numpy_array_count': len(numpy_arrays),
            'total_objects': len(gc.get_objects())
        }
    
    def print_memory_report(self):
        """打印内存报告"""
        stats = self.get_current_memory_usage()
        
        print("\n" + "=" * 50)
        print("内存使用报告")
        print("=" * 50)
        print(f"追踪内存: {stats['traced_memory_mb']:.2f}MB")
        print(f"峰值内存: {stats['peak_memory_mb']:.2f}MB")
        print(f"NumPy数组内存: {stats['numpy_memory_mb']:.2f}MB")
        print(f"NumPy数组数量: {stats['numpy_array_count']}")
        print(f"总对象数量: {stats['total_objects']}")
        print("=" * 50)
    
    def find_large_arrays(self, min_size_mb: float = 1.0) -> List[Dict]:
        """查找大型数组"""
        large_arrays = []
        
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / 1024 / 1024
                if size_mb >= min_size_mb:
                    large_arrays.append({
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'size_mb': size_mb,
                        'id': id(obj)
                    })
        
        # 按大小排序
        large_arrays.sort(key=lambda x: x['size_mb'], reverse=True)
        
        return large_arrays
    
    def print_large_arrays_report(self, min_size_mb: float = 1.0):
        """打印大型数组报告"""
        large_arrays = self.find_large_arrays(min_size_mb)
        
        print(f"\n大型数组报告 (>= {min_size_mb}MB):")
        print("=" * 60)
        
        if not large_arrays:
            print("未发现大型数组")
            return
        
        total_size = 0
        for i, arr_info in enumerate(large_arrays, 1):
            print(f"{i}. 形状: {arr_info['shape']}, "
                  f"类型: {arr_info['dtype']}, "
                  f"大小: {arr_info['size_mb']:.2f}MB, "
                  f"ID: {arr_info['id']}")
            total_size += arr_info['size_mb']
        
        print(f"\n总计: {len(large_arrays)}个数组, {total_size:.2f}MB")
    
    def force_cleanup(self):
        """强制清理内存"""
        print("执行强制内存清理...")
        
        # 多次垃圾回收
        for i in range(3):
            collected = gc.collect()
            print(f"垃圾回收第{i+1}轮: 清理了{collected}个对象")
        
        # 清理NumPy缓存
        try:
            import numpy as np
            # NumPy没有直接的缓存清理方法，但我们可以强制垃圾回收
            pass
        except ImportError:
            pass
        
        print("强制清理完成")


# 全局内存监控器实例
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """获取全局内存监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def monitor_memory_usage(func):
    """装饰器：监控函数的内存使用"""
    def wrapper(*args, **kwargs):
        monitor = get_memory_monitor()
        
        # 执行前快照
        before_idx = monitor.take_snapshot(f"执行前: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
        finally:
            # 执行后快照
            after_idx = monitor.take_snapshot(f"执行后: {func.__name__}")
            
            # 比较差异
            monitor.compare_snapshots(before_idx, after_idx, top_n=5)
        
        return result
    
    return wrapper


def print_memory_summary():
    """打印内存摘要"""
    monitor = get_memory_monitor()
    monitor.print_memory_report()
    monitor.print_large_arrays_report()


if __name__ == '__main__':
    # 测试内存监控器
    monitor = MemoryMonitor()
    monitor.start_tracing()
    
    # 创建一些测试数组
    print("创建测试数组...")
    arr1 = np.random.random((1000, 1000))  # ~8MB
    monitor.take_snapshot("创建arr1后")
    
    arr2 = np.random.random((2000, 2000))  # ~32MB
    monitor.take_snapshot("创建arr2后")
    
    # 删除数组
    del arr1
    monitor.take_snapshot("删除arr1后")
    
    del arr2
    monitor.take_snapshot("删除arr2后")
    
    # 强制清理
    monitor.force_cleanup()
    monitor.take_snapshot("强制清理后")
    
    # 打印报告
    monitor.print_memory_report()
    monitor.print_large_arrays_report()
    
    monitor.stop_tracing()
