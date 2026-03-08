#!/bin/bash

# Optional parameters
exp_name=${1:-"default_experiment"}
num_trials=${2:-1}

# Set ROS environment variables
export ROS_MASTER_URI=http://localhost:11311
export DISPLAY=${DISPLAY}

VOCAB=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG_DIR=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL
GS_CFG_DIR=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL
RESULTS_DIR=/workspace/repo/results/${exp_name}

COMMON_ARGS="vocabulary_path:=${VOCAB}
  mode:=rgbd
  slam_mode:=external
  image_topic:=/left_camera_rgb
  depth_topic:=/zed2/zed_node/depth/depth_registered
  target_frame:=map
  source_frame:=zed2_left_camera_optical_frame
  use_viewer:=false"

for ((i=0; i<num_trials; i++))
do
    echo "=== train1: trial $((i+1)) of $num_trials ==="
    roslaunch diskchungs_ros diskchungs.launch \
        ${COMMON_ARGS} \
        orb_settings_path:=${ORB_CFG_DIR}/arche_train1.yaml \
        gaussian_settings_path:=${GS_CFG_DIR}/arche_train1.yaml \
        output_directory:=${RESULTS_DIR}/RSL_rgbd_${i}/train1
    echo "Completed trial $((i+1)) of $num_trials"
done

for ((i=0; i<num_trials; i++))
do
    echo "=== train2: trial $((i+1)) of $num_trials ==="
    roslaunch diskchungs_ros diskchungs.launch \
        ${COMMON_ARGS} \
        orb_settings_path:=${ORB_CFG_DIR}/arche_train1.yaml \
        gaussian_settings_path:=${GS_CFG_DIR}/arche_train2.yaml \
        output_directory:=${RESULTS_DIR}/RSL_rgbd_${i}/train2
    echo "Completed trial $((i+1)) of $num_trials"
done

for ((i=0; i<num_trials; i++))
do
    echo "=== gazebo1: trial $((i+1)) of $num_trials ==="
    roslaunch diskchungs_ros diskchungs.launch \
        ${COMMON_ARGS} \
        orb_settings_path:=${ORB_CFG_DIR}/arche_train1.yaml \
        gaussian_settings_path:=${GS_CFG_DIR}/arche_gazebo1.yaml \
        output_directory:=${RESULTS_DIR}/RSL_rgbd_${i}/gazebo1
    echo "Completed trial $((i+1)) of $num_trials"
done
