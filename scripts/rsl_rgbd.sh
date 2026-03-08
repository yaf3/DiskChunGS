#!/bin/bash

# Optional parameters
exp_name=${1:-"default_experiment"}
num_trials=${2:-1}

# Set ROS environment variables
export ROS_MASTER_URI=http://localhost:11311
export DISPLAY=${DISPLAY}

# Make sure we're in the right working directory
cd /root/catkin_ws

for ((i=0; i<num_trials; i++))
do
    echo "Starting trial $((i+1)) of $num_trials"
    
    # Use the direct path to the executable
    /root/catkin_ws/devel/lib/lsgs_ros/lsgs_ros_node \
        __name:=gaussian_slam \
        _vocabulary_path:=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
        _orb_settings_path:=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
        _gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_train1.yaml \
        _output_directory:=/workspace/repo/results/${exp_name}/RSL_rgbd_$i/train1 \
        _use_viewer:=true \
        _mode:=rgbd \
        _image_topic:=/left_camera_rgb \
        _depth_topic:=/zed2/zed_node/depth/depth_registered \
        _slam_mode:=external \
        _target_frame:=map \
        _source_frame:=zed2_left_camera_optical_frame
    
    echo "Completed trial $((i+1)) of $num_trials"
done

for ((i=0; i<num_trials; i++))
do
    echo "Starting trial $((i+1)) of $num_trials"
    
    # Use the direct path to the executable
    /root/catkin_ws/devel/lib/lsgs_ros/lsgs_ros_node \
        __name:=gaussian_slam \
        _vocabulary_path:=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
        _orb_settings_path:=//workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
        _gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_train2.yaml \
        _output_directory:=/workspace/repo/results/${exp_name}/RSL_rgbd_$i/train2  \
        _use_viewer:=true \
        _mode:=rgbd \
        _image_topic:=/left_camera_rgb \
        _depth_topic:=/zed2/zed_node/depth/depth_registered \
        _slam_mode:=external \
        _target_frame:=map \
        _source_frame:=zed2_left_camera_optical_frame
    
    echo "Completed trial $((i+1)) of $num_trials"
done

for ((i=0; i<num_trials; i++))
do
    echo "Starting trial $((i+1)) of $num_trials"
    
    # Use the direct path to the executable
    /root/catkin_ws/devel/lib/lsgs_ros/lsgs_ros_node \
        __name:=gaussian_slam \
        _vocabulary_path:=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
        _orb_settings_path:=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
        _gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_gazebo1.yaml \
        _output_directory:=/workspace/repo/results/${exp_name}/RSL_rgbd_$i/gazebo1  \
        _use_viewer:=true \
        _mode:=rgbd \
        _image_topic:=/left_camera_rgb \
        _depth_topic:=/zed2/zed_node/depth/depth_registered \
        _slam_mode:=external \
        _target_frame:=map \
        _source_frame:=zed2_left_camera_optical_frame
    
    echo "Completed trial $((i+1)) of $num_trials"
done