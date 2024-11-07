docker build -f Dockerfile -t lsgs-slam .

xhost +local:root

docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e USER=$USER \
-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
-v /home/casimir/ETH/MT/large_scale_gaussian_slam:/large_scale_gaussian_slam \
-v /mnt/auxiliary1/datasets:/data \
--shm-size 20G --net host --gpus all --privileged \
--name lsgs-slam lsgs-slam:latest /bin/bash

./bin/replica_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office0 \
    ./results
    # no_viewer 

./bin/view_result \
    ./cfg/gaussian_mapper/Monocular/Replica/office0.yaml \
    ./cfg/view_only/camera_replica.yaml\
    ./results/replica/mono/office0/6381_shutdown/ply/point_cloud/iteration_6381/point_cloud.ply

./bin/replica_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    ./cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    ./data/Replica/office0 \
    ./results/replica/mono/office0

./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    ./data/TUM/rgbd_dataset_freiburg3_long_office_household \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    ./results/tum_rgbd/rgbd_dataset_freiburg3_long_office_household


xhost -local:root
