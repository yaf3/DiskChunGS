#!/bin/bash
exp=$1
num_trials=$2

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences_modified/00_500f \
    results/$exp/kitti_stereo_modified_$i/00_500f \
    no_viewer
done