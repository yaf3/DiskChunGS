import os
import torch
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

# Determine project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Get dependencies directories
libtorch_path =  "/workspace/third_party/libtorch"
opencv_path = "/workspace/third_party/install/opencv"
orb_slam_path = os.path.join(project_root, "slam_deps/ORB-SLAM3")

# Try to find Eigen include directory
eigen_include_dirs = [
    # Common Eigen installation locations
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    # Possible locations in the project
    os.path.join(project_root, "slam_deps/eigen"),
    os.path.join(project_root, "slam_deps/eigen3"),
    os.path.join(project_root, "slam_deps/Eigen"),
]

# Try to find Eigen via pkg-config
try:
    pkg_config_eigen = subprocess.check_output(['pkg-config', '--cflags', 'eigen3']).decode('utf-8').strip()
    if pkg_config_eigen:
        # Extract include paths from pkg-config output
        for flag in pkg_config_eigen.split():
            if flag.startswith('-I'):
                eigen_include_dirs.insert(0, flag[2:])  # Remove the -I prefix
except (subprocess.SubprocessError, FileNotFoundError):
    print("Warning: pkg-config not available or eigen3 not found with pkg-config")

# Extra include directories
include_dirs = [
    project_root,
    os.path.join(project_root, "slam_deps"),
    "/workspace/third_party",
    os.path.join(project_root, "python"),  # Directory containing our extension source files
    os.path.join(orb_slam_path),
    os.path.join(orb_slam_path, "include"),
    os.path.join(orb_slam_path, "include/CameraModels"),
    os.path.join(orb_slam_path, "Thirdparty/Sophus"),
    os.path.join(orb_slam_path, "Thirdparty/DBoW2"),
    os.path.join(orb_slam_path, "Thirdparty/g2o"),
    os.path.join(opencv_path, "include/opencv4"),
]

# Add Eigen include directories
include_dirs.extend(eigen_include_dirs)

# Debug: Print include directories to help with troubleshooting
print("Using include directories:")
for inc_dir in include_dirs:
    print(f"  - {inc_dir}")
    # Check if the directory exists and Eigen files are there
    if os.path.isdir(inc_dir):
        eigen_core_path = os.path.join(inc_dir, "Eigen", "Core")
        if os.path.exists(eigen_core_path):
            print(f"    Found Eigen/Core at {eigen_core_path}")
        eigen3_core_path = os.path.join(inc_dir, "eigen3", "Eigen", "Core")
        if os.path.exists(eigen3_core_path):
            print(f"    Found eigen3/Eigen/Core at {eigen3_core_path}")
    else:
        print(f"    Directory does not exist")

# Library directories
library_dirs = [
    os.path.join(project_root, "lib"),
    os.path.join(libtorch_path, "lib"),
    os.path.join(opencv_path, "lib"),
    os.path.join(orb_slam_path, "lib"),
]

# Libraries to link
libraries = [
    "gaussian_mapper",
    "gaussian_viewer",
    "cuda_rasterizer",
    "simple_knn",
    "ORB_SLAM3",
    "opencv_core",
    "opencv_imgproc",
    "opencv_highgui",
    "opencv_imgcodecs",
    "jsoncpp",
]

# CUDA settings
cuda_include_dirs = [
    os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include")
]
cuda_lib_dirs = [
    os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
]

# Add CUDA include dirs to the include dirs
include_dirs.extend(cuda_include_dirs)
library_dirs.extend(cuda_lib_dirs)

# Define preprocessor definitions
define_macros = [
    ('EIGEN_MPL2_ONLY', 1),   # Use only MPL2 licensed parts of Eigen
    ('EIGEN_NO_DEBUG', 1),    # Disable Eigen asserts for performance
]

# Define the extension
extension = CUDAExtension(
    name="gs_render",  # Name of the extension module
    sources=[
        "python/gs_render.cpp",    # Main extension file with pybind11 bindings
        "python/gs_core.cpp",      # Core functionality
        "python/utils.cpp",        # Utility functions
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_compile_args={
        'cxx': ['-std=c++17', '-fopenmp', '-Wno-deprecated-declarations'],
        'nvcc': ['-std=c++17', '-Xcompiler', '-fopenmp', '-DTORCH_USE_CUDA_DSA', '-arch=sm_86']
    },
    # Add runtime path to find shared libraries
    extra_link_args=['-Wl,-rpath,' + os.path.join(project_root, 'lib')]
)

setup(
    name="gs_render",
    version="0.1",
    description="3DGS Rendering C++ Extension",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
)