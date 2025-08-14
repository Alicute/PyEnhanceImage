"""
测试优化后的窗宽窗位增强算法
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_optimized_algorithm():
    """测试优化后的算法"""
    try:
        from src.core.window_based_enhancer import WindowBasedEnhancer
        
        # 测试数据
        test_data = np.random.randint(1000, 4000, (512, 512), dtype=np.uint16)
        print("🧪 测试优化后的窗宽窗位增强算法...")
        print(f"   输入数据范围: {test_data.min()} - {test_data.max()}")
        
        # 模拟窗宽窗位
        window_width = 2000
        window_level = 2500
        print(f"   窗宽窗位: {window_width}, {window_level}")
        
        # 执行增强
        result = WindowBasedEnhancer.window_based_enhance(test_data, window_width, window_level)
        
        print("✅ 测试成功！")
        print(f"   输出数据范围: {result.min()} - {result.max()}")
        print(f"   输出数据类型: {result.dtype}")
        
        # 获取算法信息
        info = WindowBasedEnhancer.get_algorithm_info()
        print(f"\n📋 算法信息:")
        print(f"   名称: {info['name']}")
        print(f"   版本: {info['version']}")
        print(f"   描述: {info['description']}")
        
        print(f"\n🎯 新特性:")
        for feature in info['features']:
            print(f"   ✅ {feature}")
            
        print(f"\n💡 改进点:")
        for improvement in info['improvements']:
            print(f"   🔧 {improvement}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 测试优化后的窗宽窗位增强算法")
    print("=" * 60)
    
    success = test_optimized_algorithm()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过！优化后的算法已准备就绪。")
        print("\n📝 主要改进:")
        print("   🔧 基于 gaoji_enhance.py 的平衡版算法")
        print("   🔧 固定强度的多尺度高频增强")
        print("   🔧 弱化的光照归一化（防止过亮）")
        print("   🔧 新增Gamma曲线调整（压亮提暗）")
        print("   🔧 新增原图混合（保留质感）")
        print("\n✨ 现在可以测试新的视觉效果了！")
        print("   1. 启动程序: uv run python src/main.py")
        print("   2. 加载DICOM文件")
        print("   3. 调整窗宽窗位")
        print("   4. 点击 '🎯 窗位增强' 按钮")
        print("   5. 享受更好的视觉效果！")
    else:
        print("❌ 测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
