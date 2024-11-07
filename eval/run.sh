docker build -f Dockerfile -t lsgs-slam-eval .

xhost +local:root

docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e USER=$USER \
-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
-v /home/casimir/ETH/MT/large_scale_gaussian_slam:/large_scale_gaussian_slam \
-v /mnt/auxiliary1/datasets:/data \
--shm-size 20G --net host --gpus all --privileged \
--name lsgs-slam-eval lsgs-slam-eval:latest /bin/bash

python3 onekey.py --dataset_center_path /data --result_main_folder /large_scale_gaussian_slam/results

xhost -local:root

