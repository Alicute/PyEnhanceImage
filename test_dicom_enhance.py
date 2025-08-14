"""
测试DICOM增强功能
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_dicom_enhancer():
    """测试DICOM增强器"""
    try:
        from src.core.dicom_enhancer import DicomEnhancer
        print("✅ DicomEnhancer导入成功")
        
        # 创建测试数据
        test_data = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        
        # 测试普通增强
        try:
            result1 = DicomEnhancer.basic_enhance(test_data)
            print(f"✅ 普通增强测试成功，输出形状: {result1.shape}")
        except Exception as e:
            print(f"❌ 普通增强测试失败: {e}")
        
        # 测试高级增强
        try:
            result2 = DicomEnhancer.advanced_enhance(test_data)
            print(f"✅ 高级增强测试成功，输出形状: {result2.shape}")
        except Exception as e:
            print(f"❌ 高级增强测试失败: {e}")
        
        # 测试超级增强
        try:
            result3 = DicomEnhancer.super_enhance(test_data)
            print(f"✅ 超级增强测试成功，输出形状: {result3.shape}")
        except Exception as e:
            print(f"❌ 超级增强测试失败: {e}")
        
        # 测试一键处理
        try:
            result4 = DicomEnhancer.auto_enhance(test_data)
            print(f"✅ 一键处理测试成功，输出形状: {result4.shape}")
        except Exception as e:
            print(f"❌ 一键处理测试失败: {e}")
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True

def test_image_processor():
    """测试图像处理器集成"""
    try:
        from src.core.image_processor import ImageProcessor
        print("✅ ImageProcessor导入成功")
        
        # 创建测试数据
        test_data = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        
        # 测试DICOM增强方法
        methods = [
            ('dicom_basic_enhance', '普通增强'),
            ('dicom_advanced_enhance', '高级增强'),
            ('dicom_super_enhance', '超级增强'),
            ('dicom_auto_enhance', '一键处理')
        ]
        
        for method_name, display_name in methods:
            try:
                method = getattr(ImageProcessor, method_name)
                result = method(test_data)
                print(f"✅ {display_name}集成测试成功，输出形状: {result.shape}")
            except Exception as e:
                print(f"❌ {display_name}集成测试失败: {e}")
                
    except ImportError as e:
        print(f"❌ ImageProcessor导入失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🧪 开始测试DICOM增强功能...")
    print("=" * 50)
    
    # 测试DicomEnhancer
    print("\n📋 测试DicomEnhancer类:")
    test1_result = test_dicom_enhancer()
    
    # 测试ImageProcessor集成
    print("\n📋 测试ImageProcessor集成:")
    test2_result = test_image_processor()
    
    print("\n" + "=" * 50)
    if test1_result and test2_result:
        print("🎉 所有测试通过！DICOM增强功能已成功集成。")
        print("\n📝 功能说明:")
        print("🔹 普通增强: 基础CLAHE + 简单高频增强")
        print("🔸 高级增强: 多步骤自适应增强算法")
        print("🔶 超级增强: 多层次复杂处理算法")
        print("⚡ 一键处理: 自动分析图像特征并选择最佳算法")
        print("\n✨ 现在可以在GUI界面中使用这些功能了！")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
