"""
多线程图像处理测试

验证异步处理功能和UI响应性
"""

import sys
import os
import time
import numpy as np
import unittest
from unittest.mock import MagicMock

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtWidgets import QApplication

from core.image_processing_thread import ImageProcessingThread, ProcessingTask, TaskStatus


class MultithreadingTest(unittest.TestCase):
    """多线程处理测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建QApplication实例（如果不存在）
        if not QCoreApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QCoreApplication.instance()
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像数据
        self.test_image = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        
        # 创建处理线程
        self.processing_thread = ImageProcessingThread()
        
        # 信号接收器
        self.received_signals = {
            'task_started': [],
            'task_progress': [],
            'task_completed': [],
            'task_failed': [],
            'queue_status_changed': []
        }
        
        # 连接信号
        self.processing_thread.task_started.connect(
            lambda task_id: self.received_signals['task_started'].append(task_id))
        self.processing_thread.task_progress.connect(
            lambda task_id, progress: self.received_signals['task_progress'].append((task_id, progress)))
        self.processing_thread.task_completed.connect(
            lambda task_id, result, desc: self.received_signals['task_completed'].append((task_id, result, desc)))
        self.processing_thread.task_failed.connect(
            lambda task_id, error: self.received_signals['task_failed'].append((task_id, error)))
        self.processing_thread.queue_status_changed.connect(
            lambda pending, total: self.received_signals['queue_status_changed'].append((pending, total)))
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'processing_thread'):
            self.processing_thread.stop_processing()
            self.processing_thread.wait(1000)
            if self.processing_thread.isRunning():
                self.processing_thread.terminate()
    
    def test_thread_creation_and_startup(self):
        """测试线程创建和启动"""
        print("\n=== 线程创建和启动测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        
        # 等待线程启动
        self.assertTrue(self.processing_thread.wait(1000), "线程应该能够启动")
        
        # 检查线程状态
        status = self.processing_thread.get_queue_status()
        print(f"线程状态: {status}")
        
        self.assertTrue(status['is_running'], "线程应该处于运行状态")
        self.assertEqual(status['pending_tasks'], 0, "初始队列应该为空")
        self.assertEqual(status['completed_tasks'], 0, "初始完成任务应该为0")
    
    def test_task_queue_management(self):
        """测试任务队列管理"""
        print("\n=== 任务队列管理测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 添加多个任务
        task_ids = []
        for i in range(3):
            task_id = self.processing_thread.add_task(
                'gamma_correction',
                {'gamma': 1.0 + i * 0.1},
                self.test_image,
                f"测试任务 {i+1}"
            )
            task_ids.append(task_id)
        
        # 检查队列状态
        status = self.processing_thread.get_queue_status()
        print(f"添加任务后状态: {status}")
        
        self.assertGreater(status['pending_tasks'], 0, "应该有待处理任务")
        
        # 等待一些信号
        self._wait_for_signals(timeout=2000)
        
        # 检查信号接收
        print(f"接收到的信号: {self.received_signals}")
        self.assertGreater(len(self.received_signals['queue_status_changed']), 0, "应该接收到队列状态变化信号")
    
    def test_task_cancellation(self):
        """测试任务取消"""
        print("\n=== 任务取消测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 添加任务
        task_id = self.processing_thread.add_task(
            'gaussian_filter',
            {'sigma': 2.0},
            self.test_image,
            "可取消的任务"
        )
        
        # 立即取消任务
        cancelled = self.processing_thread.cancel_task(task_id)
        print(f"任务取消结果: {cancelled}")
        
        # 等待处理
        self._wait_for_signals(timeout=1000)
        
        # 检查状态
        status = self.processing_thread.get_queue_status()
        print(f"取消后状态: {status}")
    
    def test_queue_clearing(self):
        """测试队列清空"""
        print("\n=== 队列清空测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 添加多个任务
        for i in range(5):
            self.processing_thread.add_task(
                'median_filter',
                {'disk_size': 3},
                self.test_image,
                f"批量任务 {i+1}"
            )
        
        # 清空队列
        self.processing_thread.clear_queue()
        
        # 检查状态
        status = self.processing_thread.get_queue_status()
        print(f"清空后状态: {status}")
        
        self.assertEqual(status['pending_tasks'], 0, "队列应该被清空")
    
    def test_pause_and_resume(self):
        """测试暂停和恢复"""
        print("\n=== 暂停和恢复测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 暂停处理
        self.processing_thread.pause_processing()
        
        # 添加任务
        task_id = self.processing_thread.add_task(
            'histogram_equalization',
            {'method': 'global'},
            self.test_image,
            "暂停测试任务"
        )
        
        # 等待一段时间（任务不应该被处理）
        self._wait_for_signals(timeout=500)
        
        # 检查任务是否仍在队列中
        status = self.processing_thread.get_queue_status()
        print(f"暂停时状态: {status}")
        self.assertTrue(status['is_paused'], "线程应该处于暂停状态")
        
        # 恢复处理
        self.processing_thread.resume_processing()
        
        # 等待任务处理
        self._wait_for_signals(timeout=2000)
        
        # 检查状态
        status = self.processing_thread.get_queue_status()
        print(f"恢复后状态: {status}")
        self.assertFalse(status['is_paused'], "线程应该不再暂停")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 错误处理测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 添加一个会失败的任务（无效算法）
        task_id = self.processing_thread.add_task(
            'invalid_algorithm',
            {},
            self.test_image,
            "错误测试任务"
        )
        
        # 等待处理
        self._wait_for_signals(timeout=2000)
        
        # 检查是否接收到失败信号
        print(f"失败信号: {self.received_signals['task_failed']}")
        self.assertGreater(len(self.received_signals['task_failed']), 0, "应该接收到任务失败信号")
    
    def test_performance_statistics(self):
        """测试性能统计"""
        print("\n=== 性能统计测试 ===")
        
        # 启动线程
        self.processing_thread.start()
        self.processing_thread.wait(100)
        
        # 添加并处理一些任务
        for i in range(3):
            self.processing_thread.add_task(
                'gamma_correction',
                {'gamma': 1.0},
                self.test_image,
                f"性能测试任务 {i+1}"
            )
        
        # 等待所有任务完成
        self._wait_for_signals(timeout=5000)
        
        # 检查性能统计
        status = self.processing_thread.get_queue_status()
        print(f"性能统计: {status}")
        
        if status['total_processed'] > 0:
            self.assertGreater(status['avg_processing_time'], 0, "平均处理时间应该大于0")
            print(f"平均处理时间: {status['avg_processing_time']:.3f}秒")
    
    def _wait_for_signals(self, timeout=1000):
        """等待信号处理
        
        Args:
            timeout: 超时时间（毫秒）
        """
        start_time = time.time()
        while time.time() - start_time < timeout / 1000.0:
            QCoreApplication.processEvents()
            time.sleep(0.01)


def run_multithreading_tests():
    """运行多线程测试"""
    print("开始多线程图像处理测试...")
    print("=" * 50)
    
    # 运行测试
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_multithreading_tests()
