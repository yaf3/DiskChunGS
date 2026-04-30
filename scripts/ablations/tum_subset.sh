#!/bin/bash
# TUM RGBD ablation subset: fr1_desk, fr3_long_office
# (drops fr2_xyz which is 6x slower and skews averages less)
# Usage: same as tum_rgbd.sh — exp, num_trials, mapper_cfg, data_root
exp=$1
num_trials=$2
mapper_cfg=${3:-cfg/triangle_mapper/RGB-D/TUM/tum_rgbd.yaml}
data_root=${4:-./data}

for ((i=0; i<num_trials; i++))
do
bin/tum_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    "$mapper_cfg" \
    $data_root/TUM/rgbd_dataset_freiburg1_desk \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    results/$exp/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/tum_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    "$mapper_cfg" \
    $data_root/TUM/rgbd_dataset_freiburg3_long_office_household \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    results/$exp/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer
done
