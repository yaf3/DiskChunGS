# DiskChunGS: Memory-Unbounded 3D Gaussian SLAM Through Efficient Disk Chunking
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`]()] [[`Project`]()] [[`Demo`]()] [[`Dataset`]()] [[`BibTeX`]()]

![Pipeline](assets/main_pipeline_resized.png?raw=true)

## Overview

DiskChunGS is a 3D Gaussian Splatting SLAM system that enables unbounded scene reconstruction through dynamic memory management, partitioning environments into spatial chunks that are selectively loaded between GPU and disk storage. This innovative approach achieves substantially higher Gaussian density than previous methods, resulting in significantly improved reconstruction quality across diverse environments while maintaining real-time performance.

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
git clone git@github.com:leggedrobotics/large_scale_gaussian_slam.git

cd large_scale_gaussian_slam

# Build the development container
docker-compose build dev

# Start the development container
docker-compose run --rm dev

# Inside the container, build the application
./build.sh
```

## Getting Started

### Preparing Datasets

The benchmark datasets mentioned in our paper:
- [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam)
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

1. Create a dataset folder which we can bind to the docker container. Then edit the dev section [docker-compose.yml](docker-compose.yml) to set the right path to your datasets.
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

### Install the Python wrapper for rendering:
```bash
python3 setup.py install
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

## Acknowledgements

This work incorporates many open-source codes. Thanks for their great work!
- [CaRtGS](https://github.com/DapengFeng/cartgs)
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs)
- [Frustum Culling](https://bruop.github.io/frustum_culling/)

## Citation

If you find this work useful in your research, consider citing it:
```
[Citation will be added here]
```