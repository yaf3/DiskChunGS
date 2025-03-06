#!/bin/bash

# Your existing commands
echo "Running post installation scripts" 

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init

cd ~/catkin_ws/src
ln -sf /workspaces/large_scale_gaussian_slam/lsgs_ros .

# Keep container running
exec "$@"