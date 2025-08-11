"""
功能测试脚本
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试导入是否正常"""
    try:
        from src.core.image_manager import ImageManager
        from src.core.image_processor import ImageProcessor
        from src.ui.main_window import MainWindow
        from src.ui.control_panel import ControlPanel
        from src.ui.image_view import ImageView
        from src.utils.helpers import is_valid_dicom_file
        print("OK 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"ERROR 导入失败: {e}")
        return False

def test_image_processor():
    """测试图像处理算法"""
    try:
        import numpy as np
        from src.core.image_processor import ImageProcessor
        
        # 创建测试图像
        test_image = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
        
        processor = ImageProcessor()
        
        # 测试各种算法
        result1 = processor.gamma_correction(test_image, 1.5)
        result2 = processor.histogram_equalization(test_image, 'global')
        result3 = processor.gaussian_filter(test_image, 1.0)
        result4 = processor.median_filter(test_image, 3)
        result5 = processor.unsharp_mask(test_image, 1.0, 1.0)
        result6 = processor.morphological_operation(test_image, 'erosion', 3)
        
        # 验证结果
        assert result1.shape == test_image.shape
        assert result2.shape == test_image.shape
        assert result3.shape == test_image.shape
        assert result4.shape == test_image.shape
        assert result5.shape == test_image.shape
        assert result6.shape == test_image.shape
        
        print("OK 图像处理算法测试通过")
        return True
    except Exception as e:
        print(f"ERROR 图像处理算法测试失败: {e}")
        return False

def test_image_manager():
    """测试图像管理器"""
    try:
        from src.core.image_manager import ImageManager
        
        manager = ImageManager()
        
        # 测试图像管理器基本功能
        state = manager.get_current_state()
        assert isinstance(state, dict)
        
        print("OK 图像管理器测试通过")
        return True
    except Exception as e:
        print(f"ERROR 图像管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始功能测试...")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("图像处理算法", test_image_processor),
        ("图像管理器", test_image_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n测试 {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"测试 {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("SUCCESS 所有测试通过！应用程序已准备就绪。")
        return True
    else:
        print("FAILED 部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)