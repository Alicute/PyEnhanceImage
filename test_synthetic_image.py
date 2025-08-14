"""
测试合成图像的处理效果
"""
import numpy as np
from src.core.image_processor import ImageProcessor

def create_synthetic_xray():
    """创建合成X光图像"""
    size = 200
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # 模拟X光图像：大部分区域平滑，少数区域有结构
    base = np.ones((size, size)) * 0.3  # 基础背景
    
    # 添加一些结构（模拟骨骼、器官等）
    base += 0.4 * np.exp(-((x-0.3)**2 + (y-0.5)**2) / 0.05)  # 圆形结构
    base += 0.3 * np.exp(-((x-0.7)**2 + (y-0.3)**2) / 0.03)  # 另一个结构
    base[int(size*0.2):int(size*0.8), int(size*0.45):int(size*0.55)] += 0.2  # 线性结构
    
    # 添加少量噪声
    noise = np.random.normal(0, 0.02, (size, size))
    synthetic_image = base + noise
    
    # 转换为uint16格式
    synthetic_image = np.clip(synthetic_image, 0, 1)
    test_data = (synthetic_image * 4000 + 1000).astype(np.uint16)
    
    return test_data

if __name__ == "__main__":
    print('🧪 创建合成X光图像...')
    test_data = create_synthetic_xray()
    print(f'合成图像: {test_data.shape}, 范围: {test_data.min()}-{test_data.max()}')
    
    # 测试处理
    print('🔍 开始处理合成图像...')
    result = ImageProcessor.paper_enhance(test_data)
    print(f'✅ 合成图像测试完成！')
