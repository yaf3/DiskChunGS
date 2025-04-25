FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Add ROS repository and keys
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    gnupg2 \
    lsb-release

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Remove system OpenCV and all related files
RUN apt-get update && \
    apt-get remove -y libopencv* python3-opencv && \
    apt-get autoremove -y && \
    rm -rf /usr/include/opencv* /usr/include/opencv2 /usr/include/opencv4 && \
    rm -rf /lib/x86_64-linux-gnu/libopencv* /usr/lib/x86_64-linux-gnu/libopencv* && \
    rm -rf /usr/local/include/opencv* /usr/local/include/opencv2

# Install ROS Noetic without recommended packages to avoid OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential

# Initialize rosdep
RUN rosdep init && rosdep update

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
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /cmake-3.22.1.sh && \
    chmod +x /cmake-3.22.1.sh && \
    /cmake-3.22.1.sh --skip-license --prefix=/usr/local

RUN apt-get update && apt-get install -y ninja-build

RUN apt-get install python3-tk -y
RUN pip3 install optuna optuna-dashboard networkx==2.8.8 torchmetrics evo
RUN apt-get install ffmpeg -y

# RUN build.sh
# RUN pip install opencv-python==4.10.0.84

# RUN build_pytorch.sh

# Add ROS setup to bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Set custom OpenCV environment variables
ENV OPENCV_PATH=/workspaces/large_scale_gaussian_slam/third_party/opencv
RUN echo "export OPENCV_PATH=${OPENCV_PATH}" >> /root/.bashrc && \
    echo "export OpenCV_DIR=${OPENCV_PATH}/build" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=${OPENCV_PATH}/build/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

# Other environment variables
RUN echo "LD_LIBRARY_PATH=/opt/ros/noetic/lib:/workspaces/large_scale_gaussian_slam/third_party/ORB-SLAM3/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]
CMD [ "sleep", "infinity" ]