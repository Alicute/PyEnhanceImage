#!/usr/bin/env python3
"""
C++扩展编译脚本
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """检查编译要求"""
    print("检查编译要求...")

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("需要Python 3.7或更高版本")
        return False
    print(f"Python版本: {sys.version}")

    # 检查pybind11
    try:
        import pybind11
        print(f"pybind11版本: {pybind11.__version__}")
    except ImportError:
        print("未安装pybind11，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
            print("pybind11安装成功")
        except subprocess.CalledProcessError:
            print("pybind11安装失败")
            return False

    # 检查numpy
    try:
        import numpy
        print(f"numpy版本: {numpy.__version__}")
    except ImportError:
        print("未安装numpy")
        return False

    return True

def clean_build():
    """清理构建目录"""
    print("清理构建目录...")

    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   删除目录: {path}")
            elif path.is_file():
                path.unlink()
                print(f"   删除文件: {path}")

def build_extension():
    """编译C++扩展"""
    print("开始编译C++扩展...")

    try:
        # 运行setup.py编译
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("C++扩展编译成功")
            print(result.stdout)
            return True
        else:
            print("C++扩展编译失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"编译过程出错: {e}")
        return False

def test_extension():
    """测试C++扩展"""
    print("测试C++扩展...")
    
    try:
        import numpy as np
        
        # 尝试导入编译的模块
        try:
            import poisson_nlm_cpp
            print("C++模块导入成功")
        except ImportError as e:
            print(f"C++模块导入失败: {e}")
            return False
        
        # 测试基本功能
        print(f"   OpenMP可用: {poisson_nlm_cpp.is_openmp_available()}")
        print(f"   OpenMP线程数: {poisson_nlm_cpp.get_openmp_threads()}")
        
        # 测试函数调用
        test_gx = np.random.randn(100, 100).astype(np.float32)
        test_gy = np.random.randn(100, 100).astype(np.float32)
        
        result_gx, result_gy, count_scale = poisson_nlm_cpp.poisson_nlm_on_gradient_exact_cpp(
            test_gx, test_gy, search_radius=1, patch_radius=1
        )
        
        print(f" 函数调用成功")
        print(f"   输入形状: {test_gx.shape}")
        print(f"   输出形状: {result_gx.shape}")
        print(f"   计数尺度: {count_scale}")
        
        return True
        
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("C++扩展编译脚本")
    print("=" * 60)
    
    # 检查要求
    if not check_requirements():
        print("编译要求检查失败")
        return 1
    
    # 清理构建目录
    clean_build()
    
    # 编译扩展
    if not build_extension():
        print(" 编译失败")
        return 1
    
    # 测试扩展
    if not test_extension():
        print(" 测试失败")
        return 1
    
    print("=" * 60)
    print(" C++扩展编译和测试完成！")
    print("现在可以在Python中使用 import poisson_nlm_cpp")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
