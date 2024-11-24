FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget

# gcc
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y

RUN apt-get install -y gcc-11 g++-11

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# dependency
RUN apt-get install -y \
    git \
    build-essential \
    sudo \
    libeigen3-dev \
    libboost-all-dev \
    libjsoncpp-dev \
    libopengl-dev \
    mesa-utils \
    libglfw3-dev \
    libglm-dev \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    zip \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /cmake-3.22.1.sh && \
    chmod +x /cmake-3.22.1.sh && \
    /cmake-3.22.1.sh --skip-license --prefix=/usr/local

RUN apt-get update && apt-get install -y ninja-build

RUN sed -i 's/library_version_type/item_version_type/g' /usr/include/boost/serialization/list.hpp

RUN apt-get install python3-tk -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install evo torchmetrics numpy scipy scikit-image lpips pillow tqdm plyfile opencv-python optuna optuna-dashboard

WORKDIR /cartgs
