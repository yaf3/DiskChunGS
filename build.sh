#!/bin/bash
set -e  # Exit on any error

# Get current directory
workdir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
echo "Working directory: $workdir"

export OPENCV_PATH=/workspace/third_party/opencv
export OpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4
export LD_LIBRARY_PATH=/workspace/third_party/install/opencv/lib:$LD_LIBRARY_PATH

# Set compiler flags
export CMAKE_EXPORT_COMPILE_COMMANDS=ON

# --- Build ORB-SLAM3 dependencies ---
echo "Building ORB-SLAM3 dependencies..."

# DBoW2
echo "Building DBoW2..."
cmake -B slam_deps/ORB-SLAM3/Thirdparty/DBoW2/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
      slam_deps/ORB-SLAM3/Thirdparty/DBoW2
cmake --build slam_deps/ORB-SLAM3/Thirdparty/DBoW2/build

# g2o
echo "Building g2o..."
cmake -B slam_deps/ORB-SLAM3/Thirdparty/g2o/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      slam_deps/ORB-SLAM3/Thirdparty/g2o
cmake --build slam_deps/ORB-SLAM3/Thirdparty/g2o/build

# Sophus
echo "Building Sophus..."
cmake -B slam_deps/ORB-SLAM3/Thirdparty/Sophus/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      slam_deps/ORB-SLAM3/Thirdparty/Sophus
cmake --build slam_deps/ORB-SLAM3/Thirdparty/Sophus/build

# Uncompress vocabulary if needed
if [ -f "slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz" ] && [ ! -f "slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt" ]; then
    echo "Uncompressing vocabulary..."
    tar -xf slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz \
        -C slam_deps/ORB-SLAM3/Vocabulary
fi

# Build ORB-SLAM3
echo "Building ORB-SLAM3..."
cmake -B slam_deps/ORB-SLAM3/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
      -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
      slam_deps/ORB-SLAM3
cmake --build slam_deps/ORB-SLAM3/build

# Update PATH for ORB-SLAM3 library
export LD_LIBRARY_PATH=$workdir/slam_deps/ORB-SLAM3/lib:$LD_LIBRARY_PATH
echo "ORB-SLAM3 built successfully!"

# --- Build main application ---
echo "Building DiskChunGS application..."
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="86" \
      -DTorch_DIR=/workspace/third_party/libtorch/share/cmake/Torch \
      -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
      -DCMAKE_CXX_FLAGS="-fopenmp" \
      -DCMAKE_CUDA_FLAGS="-Xcompiler -fopenmp -DTORCH_USE_CUDA_DSA" \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build

echo "✓ Build completed successfully!"
echo "----------------------------------------"
echo "You can run the application with:"
echo "  ./bin/your_application_name"
echo "----------------------------------------"