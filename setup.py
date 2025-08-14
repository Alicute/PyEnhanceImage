"""
C++扩展编译脚本 - 用于编译泊松NLM加速模块
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig
import os

# 检查是否安装了pybind11
try:
    import pybind11
    from pybind11 import get_cmake_dir
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("错误：需要安装pybind11")
    print("请运行: pip install pybind11")
    sys.exit(1)

# 编译选项
compile_args = {
    'msvc': ['/O2', '/openmp', '/std:c++14'],         # Visual Studio
    'mingw32': ['-O3', '-fopenmp', '-std=c++14'],    # MinGW
    'unix': ['-O3', '-fopenmp', '-std=c++14'],       # Linux/macOS
}

link_args = {
    'msvc': [],
    'mingw32': ['-fopenmp'],
    'unix': ['-fopenmp'],
}

# 检测编译器类型
def get_compiler_type():
    if sys.platform == 'win32':
        if 'MSC' in sys.version:
            return 'msvc'
        else:
            return 'mingw32'
    else:
        return 'unix'

compiler_type = get_compiler_type()
print(f"检测到编译器类型: {compiler_type}")

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "poisson_nlm_cpp",
        sources=[
            "cpp/poisson_nlm.cpp",
        ],
        include_dirs=[
            # pybind11会自动添加
        ],
        language='c++',
        cxx_std=14,
        extra_compile_args=compile_args.get(compiler_type, ['-O3']),
        extra_link_args=link_args.get(compiler_type, []),
    ),
]

# 自定义构建命令
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # 检查C++编译器是否可用
        try:
            self.compiler.compile(['cpp/test_compile.cpp'], output_dir=self.build_temp)
            print("C++编译器检查通过")
        except Exception as e:
            print(f"C++编译器检查失败: {e}")
            print("请确保已安装C++编译器:")
            if sys.platform == 'win32':
                print("  - Visual Studio Build Tools 2019/2022")
                print("  - 或者 MinGW-w64")
            else:
                print("  - GCC 或 Clang")
            raise
        
        # 检查OpenMP支持
        try:
            test_openmp = """
            #include <omp.h>
            int main() { return omp_get_num_threads(); }
            """
            with open(os.path.join(self.build_temp, 'test_openmp.cpp'), 'w') as f:
                f.write(test_openmp)
            
            self.compiler.compile([os.path.join(self.build_temp, 'test_openmp.cpp')],
                                output_dir=self.build_temp,
                                extra_preargs=compile_args.get(compiler_type, []))
            print("OpenMP支持检查通过")
        except Exception as e:
            print(f"OpenMP支持检查失败: {e}")
            print("将使用单线程版本")
            # 移除OpenMP相关编译选项
            for ext in self.extensions:
                if '/openmp' in ext.extra_compile_args:
                    ext.extra_compile_args.remove('/openmp')
                if '-fopenmp' in ext.extra_compile_args:
                    ext.extra_compile_args.remove('-fopenmp')
                if '-fopenmp' in ext.extra_link_args:
                    ext.extra_link_args.remove('-fopenmp')
        
        super().build_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
