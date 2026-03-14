# DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`](https://arxiv.org/abs/2511.23030)] [[`Project`](https://rffr.leggedrobotics.com/works/diskchungs/)] [[`Video`](https://www.youtube.com/watch?v=BFqPBZulrhQ&feature=youtu.be)]

![Pipeline](assets/pipeline.png?raw=true)

## Overview

Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated impressive results for novel view synthesis with real-time rendering capabilities. However, integrating 3DGS with SLAM systems faces a fundamental scalability limitation: methods are constrained by GPU memory capacity, restricting reconstruction to small-scale environments. We present DiskChunGS, a scalable 3DGS SLAM system that overcomes this bottleneck through an out-of-core approach that partitions scenes into spatial chunks and maintains only active regions in GPU memory while storing inactive areas on disk. Our architecture integrates seamlessly with existing SLAM frameworks for pose estimation and loop closure, enabling globally consistent reconstruction at scale. We validate DiskChunGS on indoor scenes (Replica, TUM-RGBD), urban driving scenarios (KITTI), and resource-constrained Nvidia Jetson platforms. Our method uniquely completes all 11 KITTI sequences without memory failures while achieving superior visual quality, demonstrating that algorithmic innovation can overcome the memory constraints that have limited previous 3DGS SLAM methods.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
- [ROS Usage](docs/ros-usage.md)
- [Configuration Options](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Additional Information](docs/additional-info.md)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

```bash
# Clone this repo
git clone --recursive git@github.com:leggedrobotics/DiskChunGS.git

cd DiskChunGS

# Build the development container (for Jetson use docker-compose_jetson.yml)
docker compose -f docker/docker-compose.yml build dev

# Start the development container (for Jetson use docker-compose_jetson.yml)
docker compose -f docker/docker-compose.yml run --rm dev

# Inside the container, build the application
./scripts/build.sh
```

## Getting Started

### Preparing Datasets

The benchmark datasets mentioned in our paper:
- [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam)
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

1. Create a dataset folder which we can bind to the docker container. Then edit the dev section [docker-compose.yml](docker/docker-compose.yml) to set the right path to your datasets.
   E.g. ```- /path/to/your/datasets:/data```

2. Download the desired dataset:
   ```bash
   scripts/download_replica.sh
   scripts/download_tum.sh
   ```

   For KITTI:
   - Download odometry data set (color, 65 GB)
   - Download odometry ground truth poses (4 MB)
   - Download odometry data set (calibration files, 1 MB)

   Then combine these to create the following file structure:
   ```
   kitti
   ├── data_odometry_color
       └── dataset
           |── sequences
           |   |── 00
           |   |   |── calib.txt
           |   |   |── image_2
           |   |   |── image_3
           |   |   |── times.txt
           |   |── 01
           |   ...
           └── poses
               |── 00.txt
               |── 01.txt
               ...
   ```

### Running the System

1. Outside of your container, run the below command to allow the docker container to connect to your display:
   ```bash
   xhost +local:root
   ```
   You can run `xhost -local:root` when you are done using DiskChunGS.

2. For testing, use the below commands to run the system after specifying the paths. Disable the viewer by adding `no_viewer` for evaluation:
   ```bash
   bin/replica_rgbd \
       slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
       cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
       cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
       /data/Replica/office0 \
       results/replica_rgbd/office0
       # no_viewer
   ```

3. We also provide scripts to conduct experiments on all benchmark datasets. In case you use a different data location, change the dataset root lines in scripts/*.sh:
   ```bash
   scripts/replica_mono.sh exp_name num_trials
   scripts/replica_rgbd.sh exp_name num_trials
   scripts/tum_mono.sh exp_name num_trials
   scripts/tum_rgbd.sh exp_name num_trials
   scripts/kitti_stereo.sh exp_name num_trials
   # etc.
   ```

## Evaluation

To use evaluate, your results need to be in the expected format. If you use our `./xxx.sh` scripts to conduct your experiments, the results are stored like this:
```
results
├── replica_mono_0
│   ├── office0
│   ├── ....
│   └── room2
├── replica_rgbd_0
│   ├── office0
│   ├── ....
│   └── room2
│
└── [replica/tum/kitti]_[mono/rgbd/stereo]_num  ....
    ├── scene_1
    ├── ....
    └── scene_n
```

### Run experiments:
Make sure to enable the following in your gaussian_mapper configs:
```
Record.record_rendered_image: 1
Record.record_ground_truth_image: 1
```

### Convert Replica GT camera pose files for EVO package:
```bash
cd eval
python3 shapeReplicaGT.py --replica_dataset_path PATH_TO_REPLICA_DATASET
```

### To get all metrics:
```bash
cd eval
python3 eval.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder RESULTS_PATH
```

- PATH_TO_ALL_DATASET: Should be /data if you've bound your datasets folder to /data
- Results will be summarized in two files: `RESULTS_PATH/log.txt` and `RESULTS_PATH/log.csv`.


## ROS Usage

### Setup

In one terminal launch and build:
```bash
docker compose -f docker/docker-compose.yml run --rm dev
source scripts/build_ros.sh
```

In another terminal launch the roscore if needed:
```bash
docker ps
docker exec -it container_name bash
source /opt/ros/noetic/setup.bash
roscore
```

### Running the ROS Node

In a new terminal, enter the container and source the workspace:
```bash
docker exec -it container_name bash
source /root/catkin_ws/devel/setup.bash
```

Then you can run the node:
```bash
roslaunch diskchungs_ros diskchungs.launch \
vocabulary_path:=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
orb_settings_path:=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_train1.yaml \
output_directory:=/workspace/repo/results/rsl/train1 \
use_viewer:=true \
mode:=rgbd \
image_topic:=/left_camera_rgb \
depth_topic:=/zed2/zed_node/depth/depth_registered \
slam_mode:=external \
target_frame:=map \
source_frame:=zed2_left_camera_optical_frame
```

You may have to publish uncompressed images like:
```bash
rosrun image_transport republish compressed in:=/zed2/zed_node/left/image_rect_color raw out:=/left_camera_rgb
```

You may also have to add ```--clock --pause``` in case you are using rosbags so that tf data can be correctly used.

### ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocabulary_path` | string | (required) | Path to the ORB vocabulary file |
| `orb_settings_path` | string | (required) | Path to the ORB SLAM settings file |
| `gaussian_settings_path` | string | (required) | Path to the Gaussian settings file |
| `output_directory` | string | (required) | Directory where the output will be saved |
| `use_viewer` | bool | `false` | Whether to use the ImGui viewer for visualization |
| `mode` | string | `"stereo"` | Sensor mode. Options: `"mono"`, `"stereo"`, `"rgbd"` |
| `left_topic` | string | `"/camera/rgb/image_raw"` | Topic for left stereo image (stereo mode) |
| `right_topic` | string | `"/camera/rgb/image_raw"` | Topic for right stereo image (stereo mode) |
| `mono_topic` | string | `"/camera/image_raw"` | Topic for monocular image (mono mode) |
| `rgb_topic` | string | `"/camera/rgb/image_raw"` | Topic for RGB image (rgbd mode) |
| `depth_topic` | string | `"/camera/depth/image_raw"` | Topic for depth image (rgbd mode) |
| `slam_mode` | string | `"orbslam"` | SLAM mode. Options: `"orbslam"` (use ORB-SLAM3 for poses), `"external"` (use TF poses from another source, currently doesn't support mono mode) |
| `target_frame` | string | `"map"` | Target frame for TF transformations (external mode) |
| `source_frame` | string | `"zed2i_left_camera_frame"` | Source frame for TF transformations (external mode) |
| `timeout_duration` | double | `20.0` | Duration (in seconds) after which the system considers data stream stopped |

Make sure to set the following in your gaussian_mapper config if you are using external mode:
```
External.min_keyframe_translation: 
External.min_keyframe_rotation: 
External.min_keyframe_time: 
```

## License

DiskChunGS is **source-available for non-commercial and research use only**. It is not open source in the OSI sense due to non-commercial restrictions on incorporated Inria components (3D Gaussian Splatting, On-The-Fly-NVS).

For commercial use, separate licenses must be obtained from Inria: stip-sophia.transfert@inria.fr

See [LICENSE.md](LICENSE.md) for full details including all upstream licenses.

## Acknowledgements

This work incorporates many open-source codes. Thanks for their great work!
- [CaRtGS](https://github.com/DapengFeng/cartgs)
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs)
- [Frustum Culling](https://bruop.github.io/frustum_culling/)
- [On-the-fly](https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/)
- [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- [xfeat_cpp](https://github.com/udaysankar01/xfeat_cpp)
- [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

## Citation

If you find this work useful in your research, consider citing it:
```
@article{feldmann2025diskchungslargescale3dgaussian,
        title = {DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management}, 
        author = {Casimir Feldmann and Maximum Wilder-Smith and Vaishakh Patil and Michael Oechsle and Michael Niemeyer and Keisuke Tateno and Marco Hutter},
        journal = {arXiv preprint arXiv:2511.23030},
        year = {2025},
        eprint = {2511.23030},
        archivePrefix = {arXiv},
        primaryClass = {cs.RO},
        url = {https://arxiv.org/abs/2511.23030}
      }
```