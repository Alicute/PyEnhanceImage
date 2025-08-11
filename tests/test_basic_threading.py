"""
基础多线程测试

简单验证多线程处理功能
"""

import sys
import os
import time
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_processing_thread import ImageProcessingThread, TaskStatus


def test_basic_functionality():
    """测试基础功能"""
    print("开始基础多线程测试...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
    print(f"创建测试图像: {test_image.shape}, dtype: {test_image.dtype}")
    
    # 创建处理线程
    thread = ImageProcessingThread()
    print("创建处理线程")
    
    # 测试任务添加
    task_id = thread.add_task(
        'gamma_correction',
        {'gamma': 1.5},
        test_image,
        "基础测试任务"
    )
    print(f"添加任务: {task_id}")
    
    # 检查队列状态
    status = thread.get_queue_status()
    print(f"队列状态: {status}")
    
    # 启动线程
    thread.start()
    print("启动处理线程")
    
    # 等待一段时间
    time.sleep(2)
    
    # 检查最终状态
    final_status = thread.get_queue_status()
    print(f"最终状态: {final_status}")
    
    # 停止线程
    thread.stop_processing()
    thread.wait(1000)
    print("停止处理线程")
    
    print("基础测试完成")


def test_image_processor_directly():
    """直接测试图像处理器"""
    print("\n直接测试图像处理器...")
    
    try:
        from core.image_processor import ImageProcessor
        
        # 创建测试图像
        test_image = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        processor = ImageProcessor()
        
        # 测试各种算法
        print("测试Gamma校正...")
        result1 = processor.gamma_correction(test_image, 1.5)
        print(f"Gamma校正结果: {result1.shape}, dtype: {result1.dtype}")
        
        print("测试高斯滤波...")
        result2 = processor.gaussian_filter(test_image, 1.0)
        print(f"高斯滤波结果: {result2.shape}, dtype: {result2.dtype}")
        
        print("测试直方图均衡化...")
        result3 = processor.histogram_equalization(test_image, 'global')
        print(f"直方图均衡化结果: {result3.shape}, dtype: {result3.dtype}")
        
        print("图像处理器测试完成")
        
    except Exception as e:
        print(f"图像处理器测试失败: {e}")


def test_lut_performance():
    """测试LUT性能"""
    print("\n测试LUT性能...")
    
    try:
        from core.window_level_lut import get_global_lut
        
        # 创建测试图像
        test_image = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        lut = get_global_lut()
        
        # 测试LUT应用
        start_time = time.time()
        result = lut.apply_lut(test_image, 400, 40)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        print(f"LUT处理时间: {processing_time:.3f}ms")
        print(f"LUT结果: {result.shape}, dtype: {result.dtype}")
        
        # 获取统计信息
        stats = lut.get_cache_stats()
        print(f"LUT统计: {stats}")
        
        print("LUT性能测试完成")
        
    except Exception as e:
        print(f"LUT性能测试失败: {e}")


if __name__ == '__main__':
    try:
        test_image_processor_directly()
        test_lut_performance()
        test_basic_functionality()
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
