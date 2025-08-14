"""
测试基于窗宽窗位的DICOM增强功能
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_window_based_enhancer():
    """测试基于窗宽窗位的增强器"""
    try:
        from src.core.window_based_enhancer import WindowBasedEnhancer
        print("✅ WindowBasedEnhancer导入成功")
        
        # 创建测试数据（模拟DICOM数据）
        test_data = np.random.randint(1000, 4000, (512, 512), dtype=np.uint16)
        # 添加一些"缺陷"（高对比度区域）
        test_data[100:150, 100:150] = 3500
        test_data[200:210, 200:210] = 1500
        
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        print(f"   数据范围: {test_data.min()} - {test_data.max()}")
        
        # 测试窗宽窗位增强
        window_width = 2000
        window_level = 2500
        
        print(f"\n🧪 测试基于窗宽窗位的增强:")
        print(f"   窗宽: {window_width}")
        print(f"   窗位: {window_level}")
        print(f"   窗宽窗位范围: {window_level - window_width/2} - {window_level + window_width/2}")
        
        result = WindowBasedEnhancer.window_based_enhance(test_data, window_width, window_level)
        print(f"✅ 窗宽窗位增强测试成功，输出形状: {result.shape}")
        print(f"   输出范围: {result.min()} - {result.max()}")
        
        # 获取算法信息
        info = WindowBasedEnhancer.get_algorithm_info()
        print(f"\n📋 算法信息:")
        print(f"   名称: {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   特性: {', '.join(info['features'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_image_processor_integration():
    """测试图像处理器集成"""
    try:
        from src.core.image_processor import ImageProcessor
        print("\n✅ ImageProcessor导入成功")
        
        # 创建测试数据
        test_data = np.random.randint(1000, 4000, (256, 256), dtype=np.uint16)
        
        # 测试集成的方法
        window_width = 1500
        window_level = 2500
        
        result = ImageProcessor.window_based_enhance(test_data, window_width, window_level)
        print(f"✅ ImageProcessor集成测试成功，输出形状: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ImageProcessor集成测试失败: {e}")
        return False

def test_button_creation():
    """测试按钮创建"""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.control_panel import ControlPanel
        
        # 创建应用程序（测试需要）
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # 创建控制面板
        control_panel = ControlPanel()
        
        # 检查新按钮是否存在
        if hasattr(control_panel, 'window_based_btn'):
            button = control_panel.window_based_btn
            print(f"✅ 新按钮创建成功: {button.text()}")
            print(f"   工具提示: {button.toolTip()}")
            print(f"   初始状态: {'启用' if button.isEnabled() else '禁用'}")
            return True
        else:
            print("❌ 新按钮不存在")
            return False
            
    except Exception as e:
        print(f"❌ 按钮测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 测试基于窗宽窗位的DICOM增强功能")
    print("=" * 60)
    
    # 测试增强器
    print("📋 测试WindowBasedEnhancer类:")
    test1_result = test_window_based_enhancer()
    
    # 测试ImageProcessor集成
    print("\n📋 测试ImageProcessor集成:")
    test2_result = test_image_processor_integration()
    
    # 测试按钮创建
    print("\n📋 测试按钮创建:")
    test3_result = test_button_creation()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result and test3_result:
        print("🎉 所有测试通过！新的窗宽窗位增强功能已成功添加。")
        print("\n📝 使用说明:")
        print("1. 启动程序: uv run python src/main.py")
        print("2. 加载DICOM文件")
        print("3. 调整窗宽窗位到最佳视觉效果")
        print("4. 点击 '🎯 窗位增强' 按钮")
        print("5. 算法会基于当前窗宽窗位设置进行增强处理")
        print("\n🎯 新功能特点:")
        print("   ✅ 只处理窗宽窗位范围内的数据")
        print("   ✅ 避免背景噪声干扰")
        print("   ✅ 充分利用动态范围")
        print("   ✅ 专门针对细小缺陷检测优化")
        print("\n✨ 现在可以对比测试不同算法的效果了！")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
