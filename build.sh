#!/bin/bash

# current directory
workdir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
echo $workdir

# libtorch
if [ ! -d "third_party/libtorch" ]; then
      echo "Downloading libtorch ..."
      wget -c https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip
      unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121.zip -d third_party
      rm libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121.zip
fi

# Fix nvrtc bug
cd third_party/libtorch/lib
ln -s libnvrtc-builtins-6c5639ce.so.12.1 libnvrtc-builtins.so.12.1
cd ../../../

export CMAKE_EXPORT_COMPILE_COMMANDS=ON

# opencv4
echo "Building OpenCV ..."
cmake -B third_party/opencv/build -G Ninja \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="native" \
      -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON \
      -DWITH_CUFFT=ON -DWITH_CUBLAS=ON -DWITH_NVCUVENC=ON\
      -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON \
      -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON \
      -DBUILD_JPEG=ON -DWITH_FFMPEG=ON \
      -DOPENCV_EXTRA_MODULES_PATH=third_party/opencv_contrib/modules \
      -DCMAKE_INSTALL_PREFIX=$workdir/third_party/install/opencv \
      third_party/opencv
cmake --build third_party/opencv/build
cmake --install third_party/opencv/build

echo "Setting up OpenCV environment variables..."
echo "export LD_PRELOAD=$workdir/third_party/install/opencv/lib/libopencv_core.so.410:$workdir/third_party/install/opencv/lib/libopencv_imgproc.so.410:$workdir/third_party/install/opencv/lib/libopencv_imgcodecs.so.410:$workdir/third_party/install/opencv/lib/libopencv_videoio.so.410:$workdir/third_party/install/opencv/lib/libopencv_highgui.so.410" >> /root/.bashrc

# Rebuild cv_bridge against custom OpenCV
echo "Rebuilding cv_bridge..."
mkdir -p /ws_cv_bridge/src
cd /ws_cv_bridge/src
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv
git checkout noetic
cd ../..

source /opt/ros/noetic/setup.bash
catkin_make install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/ros/noetic \
    -DOPENCV_VERSION_MAJOR=4 \
    -DOpenCV_DIR=$workdir/third_party/install/opencv/lib/cmake/opencv4 \
    -DCMAKE_INSTALL_RPATH=$workdir/third_party/install/opencv/lib \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE

cd $workdir

# DBoW2
cmake -B third_party/ORB-SLAM3/Thirdparty/DBoW2/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=$workdir/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3/Thirdparty/DBoW2
cmake --build third_party/ORB-SLAM3/Thirdparty/DBoW2/build

# g2o
cmake -B third_party/ORB-SLAM3/Thirdparty/g2o/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/g2o
cmake --build third_party/ORB-SLAM3/Thirdparty/g2o/build

# Sophus
cmake -B third_party/ORB-SLAM3/Thirdparty/Sophus/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/Sophus
cmake --build third_party/ORB-SLAM3/Thirdparty/Sophus/build

# ORB-SLAM3
echo "Uncompress vocabulary ..."
tar -xf third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz \
    -C third_party/ORB-SLAM3/Vocabulary

cmake -B third_party/ORB-SLAM3/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
      -DOpenCV_DIR=$workdir/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3
cmake --build third_party/ORB-SLAM3/build

# LSGS
echo "Building LSGS ..."
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="86" \
      -DTorch_DIR=$workdir/third_party/libtorch/share/cmake/Torch \
      -DOpenCV_DIR=$workdir/third_party/third_party/install/opencv/lib/cmake/opencv4 \
      -DCMAKE_CXX_FLAGS="-fopenmp" \
      -DCMAKE_CUDA_FLAGS="-Xcompiler -fopenmp -DTORCH_USE_CUDA_DSA" \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build
