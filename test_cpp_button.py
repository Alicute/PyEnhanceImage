"""
测试C++加速按钮功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core.image_processor import ImageProcessor

def test_cpp_acceleration():
    """测试C++加速功能"""
    print("=" * 60)
    print("🧪 测试C++加速论文算法")
    print("=" * 60)
    
    # 创建测试数据
    test_data = np.random.randint(1000, 5000, (300, 300), dtype=np.uint16)
    print(f"测试数据: {test_data.shape}, 范围: {test_data.min()}-{test_data.max()}")
    
    # 测试C++加速版本
    print(f"\n🚀 测试C++加速版本...")
    try:
        result = ImageProcessor.paper_enhance_cpp(test_data)
        print(f"✅ C++加速版本测试成功!")
        print(f"   输出形状: {result.shape}")
        print(f"   输出范围: {result.min()}-{result.max()}")
        return True
    except Exception as e:
        print(f"❌ C++加速版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpp_extension():
    """测试C++扩展是否可用"""
    print(f"\n🔍 检查C++扩展状态...")
    
    try:
        import poisson_nlm_cpp
        print(f"✅ C++扩展可用")
        print(f"   版本: {getattr(poisson_nlm_cpp, '__version__', 'unknown')}")
        print(f"   OpenMP支持: {poisson_nlm_cpp.is_openmp_available()}")
        print(f"   线程数: {poisson_nlm_cpp.get_openmp_threads()}")
        
        # 测试基本功能
        test_gx = np.random.randn(50, 50).astype(np.float32)
        test_gy = np.random.randn(50, 50).astype(np.float32)
        
        result_gx, result_gy, count_scale = poisson_nlm_cpp.poisson_nlm_on_gradient_exact_cpp(
            test_gx, test_gy, search_radius=1, patch_radius=1
        )
        
        print(f"✅ C++函数调用成功")
        print(f"   输入形状: {test_gx.shape}")
        print(f"   输出形状: {result_gx.shape}")
        
        return True
        
    except ImportError:
        print(f"⚠️  C++扩展不可用")
        print(f"   提示: 运行 'python build_cpp.py' 来编译C++扩展")
        return False
    except Exception as e:
        print(f"❌ C++扩展测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始C++加速功能测试")
    
    # 测试C++扩展
    cpp_available = test_cpp_extension()
    
    # 测试C++加速算法
    algorithm_success = test_cpp_acceleration()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"   C++扩展可用: {'✅' if cpp_available else '❌'}")
    print(f"   算法功能正常: {'✅' if algorithm_success else '❌'}")
    
    if algorithm_success:
        print("\n🎉 C++加速功能测试通过!")
        if not cpp_available:
            print("💡 提示: 编译C++扩展可获得更好性能")
            print("   运行: python build_cpp.py")
    else:
        print("\n❌ 测试失败，请检查代码")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
