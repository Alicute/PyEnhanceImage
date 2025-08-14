"""
测试按钮修复是否成功
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_control_panel():
    """测试控制面板按钮启用功能"""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.control_panel import ControlPanel
        
        # 创建应用程序（测试需要）
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # 创建控制面板
        control_panel = ControlPanel()
        
        # 检查DICOM增强按钮是否存在
        buttons = [
            ('dicom_basic_btn', '普通增强'),
            ('dicom_advanced_btn', '高级增强'),
            ('dicom_super_btn', '超级增强'),
            ('dicom_auto_btn', '一键处理')
        ]
        
        print("🔍 检查DICOM增强按钮:")
        all_buttons_exist = True
        
        for attr_name, display_name in buttons:
            if hasattr(control_panel, attr_name):
                button = getattr(control_panel, attr_name)
                print(f"   ✅ {display_name} 按钮存在")
                print(f"      初始状态: {'启用' if button.isEnabled() else '禁用'}")
            else:
                print(f"   ❌ {display_name} 按钮不存在")
                all_buttons_exist = False
        
        if not all_buttons_exist:
            return False
        
        # 测试启用功能
        print("\n🧪 测试按钮启用功能:")
        print("   设置按钮为启用状态...")
        control_panel.set_controls_enabled(True)
        
        for attr_name, display_name in buttons:
            button = getattr(control_panel, attr_name)
            if button.isEnabled():
                print(f"   ✅ {display_name} 按钮已启用")
            else:
                print(f"   ❌ {display_name} 按钮仍然禁用")
                return False
        
        # 测试禁用功能
        print("\n   设置按钮为禁用状态...")
        control_panel.set_controls_enabled(False)
        
        for attr_name, display_name in buttons:
            button = getattr(control_panel, attr_name)
            if not button.isEnabled():
                print(f"   ✅ {display_name} 按钮已禁用")
            else:
                print(f"   ❌ {display_name} 按钮仍然启用")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 DICOM增强按钮修复验证")
    print("=" * 50)
    
    # 测试控制面板
    success = test_control_panel()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 修复成功！")
        print("\n📝 现在的操作步骤:")
        print("1. 重新启动程序（如果还在运行旧版本）")
        print("2. 加载DICOM文件")
        print("3. DICOM增强按钮应该会变为可点击状态")
        print("\n✨ 问题已解决，按钮应该可以正常使用了！")
    else:
        print("❌ 修复失败，请检查代码。")

if __name__ == "__main__":
    main()
