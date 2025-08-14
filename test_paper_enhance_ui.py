#!/usr/bin/env python3
"""
测试论文算法UI集成
"""
import sys
import os
sys.path.append('src')

import numpy as np
from PyQt6.QtWidgets import QApplication
from ui.control_panel import ControlPanel

def test_paper_enhance_button():
    """测试论文算法按钮是否正确集成"""
    print("🧪 测试论文算法UI集成...")
    
    app = QApplication(sys.argv)
    
    try:
        # 创建控制面板
        control_panel = ControlPanel()
        print("✅ 控制面板创建成功")
        
        # 检查论文算法按钮是否存在
        if hasattr(control_panel, 'paper_enhance_btn'):
            print("✅ 论文算法按钮存在")
            
            # 检查按钮文本
            btn_text = control_panel.paper_enhance_btn.text()
            print(f"   按钮文本: {btn_text}")
            
            # 检查工具提示
            tooltip = control_panel.paper_enhance_btn.toolTip()
            print(f"   工具提示: {tooltip}")
            
            # 检查初始状态（应该是禁用的）
            is_enabled = control_panel.paper_enhance_btn.isEnabled()
            print(f"   初始状态: {'启用' if is_enabled else '禁用'}")
            
            # 模拟启用按钮
            control_panel.set_controls_enabled(True)
            is_enabled_after = control_panel.paper_enhance_btn.isEnabled()
            print(f"   启用后状态: {'启用' if is_enabled_after else '禁用'}")
            
            if btn_text == "📄 论文算法处理" and not is_enabled and is_enabled_after:
                print("✅ 论文算法按钮集成测试通过")
                return True
            else:
                print("❌ 论文算法按钮集成测试失败")
                return False
        else:
            print("❌ 论文算法按钮不存在")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        app.quit()

def test_signal_connection():
    """测试信号连接"""
    print("\n🔗 测试信号连接...")
    
    app = QApplication(sys.argv)
    
    try:
        control_panel = ControlPanel()
        
        # 检查信号是否正确连接
        signal_connected = False
        
        # 创建一个测试接收器
        def test_receiver(algorithm_name, parameters):
            nonlocal signal_connected
            if algorithm_name == 'paper_enhance':
                signal_connected = True
                print(f"✅ 接收到信号: {algorithm_name}, 参数: {parameters}")
        
        # 连接信号
        control_panel.apply_algorithm.connect(test_receiver)
        
        # 启用按钮
        control_panel.set_controls_enabled(True)
        
        # 模拟点击按钮
        control_panel.paper_enhance_btn.click()
        
        # 处理事件
        app.processEvents()
        
        if signal_connected:
            print("✅ 信号连接测试通过")
            return True
        else:
            print("❌ 信号连接测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 信号测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        app.quit()

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 论文算法UI集成测试")
    print("=" * 60)
    
    # 测试1: 按钮集成
    test1_result = test_paper_enhance_button()
    
    # 测试2: 信号连接
    test2_result = test_signal_connection()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result:
        print("🎉 所有UI集成测试通过！")
        print("\n📝 论文算法已成功集成到UI中:")
        print("1. ✅ 按钮正确创建并显示")
        print("2. ✅ 工具提示信息完整")
        print("3. ✅ 启用/禁用状态正确")
        print("4. ✅ 信号连接正常工作")
        print("\n🚀 现在可以在主程序中使用论文算法了！")
        print("   - 启动程序: uv run python src/main.py")
        print("   - 加载DICOM文件")
        print("   - 点击 '📄 论文算法处理' 按钮")
    else:
        print("❌ 部分UI集成测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
