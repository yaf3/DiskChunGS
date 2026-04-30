#!/bin/bash
exp=$1
num_trials=$2
mapper_cfg=${3:-cfg/triangle_mapper/RGB-D/Replica/replica_rgbd.yaml}
data_root=${4:-./data}

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    "$mapper_cfg" \
    $data_root/Replica/office0 \
    results/$exp/replica_rgbd_$i/office0 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    "$mapper_cfg" \
    $data_root/Replica/office1 \
    results/$exp/replica_rgbd_$i/office1 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml \
    "$mapper_cfg" \
    $data_root/Replica/office2 \
    results/$exp/replica_rgbd_$i/office2 \
    no_viewer
done

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
    cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    "$mapper_cfg" \
    $data_root/Replica/room0 \
    results/$exp/replica_rgbd_$i/room0 \
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

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/room2.yaml \
    "$mapper_cfg" \
    $data_root/Replica/room2 \
    results/$exp/replica_rgbd_$i/room2 \
    no_viewer
done
