#!/bin/bash

# Build the main project
cd /workspace/repo
./scripts/build/build.sh Debug

# Build the ROS package
cd /root/catkin_ws
catkin build diskchungs_ros --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Source the setup file
source /root/catkin_ws/devel/setup.bash

cd /workspace/repo