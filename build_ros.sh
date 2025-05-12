#!/bin/bash

# Build the main project
cd /workspace/repo
./build.sh Debug

# Build the ROS package
cd /root/catkin_ws
catkin build lsgs_ros --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Source the setup file
source /root/catkin_ws/devel/setup.bash

cd /workspace/repo