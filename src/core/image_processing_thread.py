"""
多线程图像处理模块

实现异步图像处理，避免UI阻塞
支持任务队列管理和进度反馈
"""

import numpy as np
import time
import uuid
from typing import Dict, Any, Optional, Callable
from PyQt6.QtCore import QThread, QMutex, QWaitCondition, pyqtSignal, QObject
from dataclasses import dataclass
from enum import Enum

from .image_processor import ImageProcessor


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """图像处理任务"""
    task_id: str
    algorithm_name: str
    parameters: Dict[str, Any]
    image_data: np.ndarray
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[np.ndarray] = None
    error_message: str = ""
    created_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()


class ImageProcessingThread(QThread):
    """图像处理线程类
    
    实现异步图像处理，支持任务队列和进度反馈
    """
    
    # 信号定义
    task_started = pyqtSignal(str)  # task_id
    task_progress = pyqtSignal(str, float)  # task_id, progress
    task_completed = pyqtSignal(str, object, str)  # task_id, result_data, description
    task_failed = pyqtSignal(str, str)  # task_id, error_message
    queue_status_changed = pyqtSignal(int, int)  # pending_count, total_count
    
    def __init__(self, parent=None):
        """初始化处理线程
        
        Args:
            parent: 父对象
        """
        super().__init__(parent)
        
        # 线程控制
        self.is_running = True
        self.is_paused = False
        
        # 任务队列和同步
        self.task_queue = []
        self.completed_tasks = []
        self.current_task: Optional[ProcessingTask] = None
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 图像处理器
        self.processor = ImageProcessor()
        
        # 性能统计
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.max_queue_size = 100  # 最大队列长度
        
        # 进度回调映射
        self.progress_callbacks = {
            'gamma_correction': self._gamma_progress,
            'histogram_equalization': self._histogram_progress,
            'gaussian_filter': self._filter_progress,
            'median_filter': self._filter_progress,
            'unsharp_mask': self._filter_progress,
            'morphological_operation': self._morphology_progress,
        }
    
    def add_task(self, algorithm_name: str, parameters: Dict[str, Any], 
                 image_data: np.ndarray, description: str = "") -> str:
        """添加处理任务到队列
        
        Args:
            algorithm_name: 算法名称
            parameters: 算法参数
            image_data: 图像数据
            description: 任务描述
            
        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            algorithm_name=algorithm_name,
            parameters=parameters,
            image_data=image_data.copy(),  # 复制数据避免并发问题
            description=description
        )
        
        self.mutex.lock()
        try:
            # 检查队列大小限制
            if len(self.task_queue) >= self.max_queue_size:
                # 移除最旧的待处理任务
                self.task_queue.pop(0)
            
            self.task_queue.append(task)
            self.condition.wakeOne()  # 唤醒处理线程
            
            # 发出队列状态变化信号
            pending_count = len(self.task_queue)
            total_count = pending_count + len(self.completed_tasks)
            self.queue_status_changed.emit(pending_count, total_count)
            
        finally:
            self.mutex.unlock()
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        self.mutex.lock()
        try:
            # 查找并移除待处理任务
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.task_queue.pop(i)
                    return True
            
            # 检查是否是当前正在处理的任务
            if self.current_task and self.current_task.task_id == task_id:
                self.current_task.status = TaskStatus.CANCELLED
                return True
                
        finally:
            self.mutex.unlock()
        
        return False
    
    def clear_queue(self):
        """清空任务队列"""
        self.mutex.lock()
        try:
            for task in self.task_queue:
                task.status = TaskStatus.CANCELLED
            self.task_queue.clear()
            
            # 发出队列状态变化信号
            self.queue_status_changed.emit(0, len(self.completed_tasks))
            
        finally:
            self.mutex.unlock()
    
    def pause_processing(self):
        """暂停处理"""
        self.is_paused = True
    
    def resume_processing(self):
        """恢复处理"""
        self.is_paused = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
    
    def stop_processing(self):
        """停止处理线程"""
        self.is_running = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态
        
        Returns:
            Dict: 队列状态信息
        """
        self.mutex.lock()
        try:
            return {
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks),
                'current_task': self.current_task.task_id if self.current_task else None,
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                'total_processed': self.total_tasks_processed,
                'avg_processing_time': (self.total_processing_time / self.total_tasks_processed 
                                      if self.total_tasks_processed > 0 else 0.0)
            }
        finally:
            self.mutex.unlock()
    
    def run(self):
        """线程主循环"""
        while self.is_running:
            # 等待任务或暂停状态
            self.mutex.lock()
            
            while (len(self.task_queue) == 0 or self.is_paused) and self.is_running:
                self.condition.wait(self.mutex, 100)  # 100ms超时
            
            if not self.is_running:
                self.mutex.unlock()
                break
            
            # 获取下一个任务
            if len(self.task_queue) > 0:
                self.current_task = self.task_queue.pop(0)
            else:
                self.current_task = None
                self.mutex.unlock()
                continue
            
            self.mutex.unlock()
            
            # 处理任务
            if self.current_task:
                self._process_task(self.current_task)
                
                # 移动到完成列表
                self.mutex.lock()
                self.completed_tasks.append(self.current_task)
                
                # 限制完成任务列表大小
                if len(self.completed_tasks) > 50:
                    self.completed_tasks.pop(0)
                
                self.current_task = None
                
                # 发出队列状态变化信号
                pending_count = len(self.task_queue)
                total_count = pending_count + len(self.completed_tasks)
                self.queue_status_changed.emit(pending_count, total_count)
                
                self.mutex.unlock()
    
    def _process_task(self, task: ProcessingTask):
        """处理单个任务
        
        Args:
            task: 处理任务
        """
        try:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            # 发出任务开始信号
            self.task_started.emit(task.task_id)
            
            # 执行算法
            result = self._execute_algorithm(task)
            
            if task.status == TaskStatus.CANCELLED:
                return
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            
            # 更新统计信息
            processing_time = task.end_time - task.start_time
            self.total_processing_time += processing_time
            self.total_tasks_processed += 1
            
            # 发出完成信号
            self.task_completed.emit(task.task_id, result, task.description)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = time.time()
            
            # 发出失败信号
            self.task_failed.emit(task.task_id, str(e))
    
    def _execute_algorithm(self, task: ProcessingTask) -> np.ndarray:
        """执行具体算法
        
        Args:
            task: 处理任务
            
        Returns:
            np.ndarray: 处理结果
        """
        algorithm_name = task.algorithm_name
        parameters = task.parameters
        data = task.image_data
        
        # 获取进度回调
        progress_callback = self.progress_callbacks.get(algorithm_name, self._default_progress)
        
        # 执行算法
        if algorithm_name == 'gamma_correction':
            return self._execute_with_progress(
                lambda: self.processor.gamma_correction(data, parameters['gamma']),
                task, progress_callback
            )
        elif algorithm_name == 'histogram_equalization':
            return self._execute_with_progress(
                lambda: self.processor.histogram_equalization(data, parameters['method']),
                task, progress_callback
            )
        elif algorithm_name == 'gaussian_filter':
            return self._execute_with_progress(
                lambda: self.processor.gaussian_filter(data, parameters['sigma']),
                task, progress_callback
            )
        elif algorithm_name == 'median_filter':
            return self._execute_with_progress(
                lambda: self.processor.median_filter(data, parameters['disk_size']),
                task, progress_callback
            )
        elif algorithm_name == 'unsharp_mask':
            return self._execute_with_progress(
                lambda: self.processor.unsharp_mask(data, parameters['radius'], parameters['amount']),
                task, progress_callback
            )
        elif algorithm_name == 'morphological_operation':
            return self._execute_with_progress(
                lambda: self.processor.morphological_operation(
                    data, parameters['operation'], parameters['disk_size']),
                task, progress_callback
            )
        else:
            raise ValueError(f"未知算法: {algorithm_name}")
    
    def _execute_with_progress(self, algorithm_func: Callable, task: ProcessingTask, 
                              progress_callback: Callable) -> np.ndarray:
        """带进度反馈的算法执行
        
        Args:
            algorithm_func: 算法函数
            task: 处理任务
            progress_callback: 进度回调函数
            
        Returns:
            np.ndarray: 处理结果
        """
        # 模拟进度更新
        progress_callback(task, 0.0)
        
        # 执行算法
        result = algorithm_func()
        
        # 检查是否被取消
        if task.status == TaskStatus.CANCELLED:
            raise InterruptedError("任务被取消")
        
        progress_callback(task, 1.0)
        return result
    
    # 进度回调函数
    def _default_progress(self, task: ProcessingTask, progress: float):
        """默认进度回调"""
        task.progress = progress
        self.task_progress.emit(task.task_id, progress)
    
    def _gamma_progress(self, task: ProcessingTask, progress: float):
        """Gamma校正进度回调"""
        self._default_progress(task, progress)
    
    def _histogram_progress(self, task: ProcessingTask, progress: float):
        """直方图均衡化进度回调"""
        self._default_progress(task, progress)
    
    def _filter_progress(self, task: ProcessingTask, progress: float):
        """滤波算法进度回调"""
        self._default_progress(task, progress)
    
    def _morphology_progress(self, task: ProcessingTask, progress: float):
        """形态学操作进度回调"""
        self._default_progress(task, progress)
