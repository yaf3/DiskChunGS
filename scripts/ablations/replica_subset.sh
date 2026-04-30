#!/bin/bash
# Replica RGBD ablation subset: office3, office4, room1
# (best 3-scene representative of the full 8-scene average)
# Usage: same as replica_rgbd.sh — exp, num_trials, mapper_cfg, data_root
exp=$1
num_trials=$2
mapper_cfg=${3:-cfg/triangle_mapper/RGB-D/Replica/replica_rgbd.yaml}
data_root=${4:-./data}

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office3.yaml \
    "$mapper_cfg" \
    $data_root/Replica/office3 \
    results/$exp/replica_rgbd_$i/office3 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office4.yaml \
    "$mapper_cfg" \
    $data_root/Replica/office4 \
    results/$exp/replica_rgbd_$i/office4 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml \
    "$mapper_cfg" \
    $data_root/Replica/room1 \
    results/$exp/replica_rgbd_$i/room1 \
    no_viewer
done
