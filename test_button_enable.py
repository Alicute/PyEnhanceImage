"""
测试按钮启用状态
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dicom_files():
    """检查项目中的DICOM文件"""
    dicom_files = []
    for file in os.listdir('.'):
        if file.endswith('.dcm'):
            dicom_files.append(file)
    
    print("📁 项目中的DICOM文件:")
    if dicom_files:
        for i, file in enumerate(dicom_files, 1):
            print(f"   {i}. {file}")
    else:
        print("   ❌ 未找到DICOM文件")
    
    return dicom_files

def test_dicom_loading():
    """测试DICOM加载功能"""
    try:
        from src.core.image_manager import ImageManager
        
        # 检查DICOM文件
        dicom_files = check_dicom_files()
        if not dicom_files:
            print("\n❌ 无法测试：项目中没有DICOM文件")
            return False
        
        # 测试加载第一个DICOM文件
        test_file = dicom_files[0]
        print(f"\n🧪 测试加载: {test_file}")
        
        image_manager = ImageManager()
        success = image_manager.load_dicom(test_file)
        
        if success:
            print(f"✅ DICOM文件加载成功")
            print(f"   图像尺寸: {image_manager.current_image.data.shape}")
            print(f"   数据类型: {image_manager.current_image.data.dtype}")
            print(f"   数值范围: {image_manager.current_image.data.min()} - {image_manager.current_image.data.max()}")
            return True
        else:
            print(f"❌ DICOM文件加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 DICOM增强按钮启用状态检查")
    print("=" * 50)
    
    # 检查DICOM文件
    dicom_files = check_dicom_files()
    
    # 测试DICOM加载
    print("\n📋 测试DICOM加载功能:")
    load_success = test_dicom_loading()
    
    print("\n" + "=" * 50)
    print("📝 使用说明:")
    print("1. 启动程序: uv run python src/main.py")
    print("2. 点击 '加载DICOM' 按钮")
    
    if dicom_files:
        print("3. 选择以下DICOM文件之一:")
        for file in dicom_files:
            print(f"   - {file}")
    else:
        print("3. ❌ 请先添加DICOM文件到项目根目录")
    
    print("4. 加载成功后，DICOM增强按钮就会变为可点击状态")
    print("\n🎯 DICOM增强按钮位置:")
    print("   在控制面板的 '🏥 DICOM增强' 组中")
    print("   包含4个按钮:")
    print("   🔹 普通增强")
    print("   🔸 高级增强") 
    print("   🔶 超级增强")
    print("   ⚡ 一键处理")
    
    if load_success:
        print("\n✅ 系统正常，按钮应该可以正常使用")
    else:
        print("\n⚠️  请检查DICOM文件是否正确")

if __name__ == "__main__":
    main()
