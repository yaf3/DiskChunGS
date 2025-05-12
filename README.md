# DiskChunGS: Memory-Unbounded 3D Gaussian SLAM Through Efficient Disk Chunking
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`]()] [[`Project`]()] [[`Demo`]()] [[`Dataset`]()] [[`BibTeX`]()]

## Table of Contents
- [Overview](#overview)
- [Installation](#installation-using-docker-compose)
- [Getting Started](#getting-started)
  - [Preparing Datasets](#preparing-datasets)
  - [Running the System](#running-the-system)
- [Evaluation](#diskchungs-evaluation)
- [ROS Usage](#ros-usage)
- [Configuration Options](#mapping-configuration-options)
- [ROS Parameters](#ros-options)
- [Troubleshooting](#common-issues)
- [Additional Information](#other-info)
- [Acknowledgements](#acknowledgement)
- [Citation](#citation)

![Pipeline](assets/main_pipeline_resized.png?raw=true)

## Overview

DiskChunGS is a 3D Gaussian Splatting SLAM system that enables unbounded scene reconstruction through dynamic memory management, partitioning environments into spatial chunks that are selectively loaded between GPU and disk storage. This innovative approach achieves substantially higher Gaussian density than previous methods, resulting in significantly improved reconstruction quality across diverse environments while maintaining real-time performance.

## Installation using Docker Compose

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

## DiskChunGS Evaluation

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

## ROS Usage

In one terminal launch and build:
```bash
docker-compose run --rm dev
source build_ros.sh
```

In another terminal launch the roscore if needed:
```bash
docker ps
docker exec -it container_name bash
source /opt/ros/noetic/setup.bash
roscore
```

Then you can run the node:
```bash
rosrun lsgs_ros lsgs_ros_node \
__name:=gaussian_slam \
_vocabulary_path:=/workspace/repo/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
_orb_settings_path:=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
_gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_train1.yaml \
_output_directory:=/workspace/repo/results/rsl/train1 \
_use_viewer:=true \
_mode:=rgbd \
_rgb_topic:=/left_camera_rgb \
_depth_topic:=/zed2/zed_node/depth/depth_registered \
_slam_mode:=external \
_target_frame:=map \
_source_frame:=zed2_left_camera_optical_frame
```

You may have to publish uncompressed images like:
```bash
rosrun image_transport republish compressed in:=/zed2/zed_node/left/image_rect_color raw out:=/left_camera_rgb
```

You may also have to add ```--clock --pause``` in case you are using rosbags so that tf data can be correctly used.

## Mapping Configuration Options

This section explains all available configuration options for the system. The configuration file uses YAML format and provides fine-grained control over model behavior, mapping, optimization, visualization, and more.

### Model Settings

| Parameter | Description |
|-----------|-------------|
| `Model.resolution` | Resolution for model rendering. Set to `-1` for automatic resolution. |
| `Model.white_background` | Controls background color. `0`: false (black), `1` or other integer: true (white). |
| `Model.eval` | DEPRECATED. Enables evaluation mode. `0`: false (training mode), `1` or other integer: true (evaluation mode). |
| `Model.sh_degree` | Spherical harmonics degree for appearance modeling. Higher values allow more complex lighting effects. |
| `Model.appearance_embedding` | Controls whether to use appearance embeddings. `0` disables appearance embedding. |

### Camera Settings

| Parameter | Description |
|-----------|-------------|
| `Camera.z_near` | Near clipping plane distance. |
| `Camera.z_far` | Far clipping plane distance. |

### Monocular Settings

| Parameter | Description |
|-----------|-------------|
| `Monocular.inactive_geo_densify_max_pixel_dist` | Maximum squared pixel distance for inactive geometry densification. |

### Stereo Settings

| Parameter | Description |
|-----------|-------------|
| `Stereo.min_disparity` | Minimum disparity value for stereo processing. |
| `Stereo.num_disparity` | Number of disparity levels for stereo matching. |
| `Stereo.do_stereo_loss` | Enables stereo loss calculation. `0`: false, `1` or other integer: true. |

### Mapper Settings

| Parameter | Description |
|-----------|-------------|
| `Mapper.min_depth_` | Minimum depth threshold for mapping. |
| `Mapper.max_depth_` | Maximum depth threshold for mapping. |
| `Mapper.inactive_geo_densify` | Enables inactive geometry densification. `0`: false, `1` or other integer: true. |
| `Mapper.depth_densify` | Enables depth-based densification. `0`: false, `1` or other integer: true. |
| `Mapper.depth_densify_subsample_ratio` | Subsampling ratio for depth-based densification. |
| `Mapper.depth_cache` | Size of the depth cache for mapping. |
| `Mapper.keyframe_similarity_threshold` | DEPRECATED. Threshold for determining keyframe similarity. |
| `Mapper.keyframe_selection_strategy` | Strategy for keyframe selection. `0`: use all keyframes, `1`: use recent k keyframes. |
| `Mapper.min_num_initial_map_kfs` | Minimum number of keyframes required for initial mapping. |
| `Mapper.new_keyframe_times_of_use` | Number of times to use new keyframes. |
| `Mapper.local_BA_increased_times_of_use` | Increased usage count for local bundle adjustment. |
| `Mapper.loop_closure_increased_times_of_use_` | Increased usage count for frames involved in loop closure. |
| `Mapper.large_rotation_threshold` | Threshold (in degrees) for detecting large rotations. |
| `Mapper.large_translation_threshold` | Threshold for detecting large translations. |
| `Mapper.stable_num_iter_existence` | Number of iterations required for a point to be considered stable. |
| `Mapper.cull_keyframes` | Enables keyframe culling. `0`: false, `1` or other integer: true. |

### Gaussian Pyramid Settings

| Parameter | Description |
|-----------|-------------|
| `GausPyramid.do` | Enables Gaussian pyramid processing. `0`: false, `1` or other integer: true. |
| `GausPyramid.num_sub_levels` | Number of sub-levels in the Gaussian pyramid. |
| `GausPyramid.sub_level_times_of_use` | Number of times to use each sub-level. |

### Pipeline Settings

| Parameter | Description |
|-----------|-------------|
| `Pipeline.convert_SHs` | Enables conversion of spherical harmonics. `0`: false, `1` or other integer: true. |
| `Pipeline.compute_cov3D` | Enables computation of 3D covariance. `0`: false, `1` or other integer: true. |

### Recording Settings

| Parameter | Description |
|-----------|-------------|
| `Record.keyframe_record_interval` | Interval for recording keyframes. `0`: never, `1`: always, other values: periodically. |
| `Record.all_keyframes_record_interval` | Interval for recording all keyframes. `0`: never, `1`: always, other values: periodically. |
| `Record.record_rendered_image` | Enables recording of rendered images. `0`: false, `1` or other integer: true. |
| `Record.record_ground_truth_image` | Enables recording of ground truth images. `0`: false, `1` or other integer: true. |
| `Record.record_loss_image` | Enables recording of loss visualization images. `0`: false, `1` or other integer: true. |
| `Record.training_report_interval` | Interval for generating training reports. `0`: never, `1`: always, other values: periodically. |
| `Record.record_loop_ply` | DEPRECATED. Enables recording of loop closure point clouds. `0`: false, `1` or other integer: true. |
| `Record.render_fly_through` | Enables rendering of fly-through sequences. `0`: false, `1` or other integer: true. |
| `Record.render_fly_through_speed` | Speed factor for fly-through rendering. |

### Optimization Settings

| Parameter | Description |
|-----------|-------------|
| `Optimization.max_num_iterations` | Maximum number of optimization iterations. `-1` for stop after SLAM ends. |
| `Optimization.smooth_l1` | Enables smooth L1 loss. `0`: false, `1`: true. |
| `Optimization.opacity_reset_interval` | Interval for resetting opacity. `0`: never, `1`: always, other values: periodically. |
| `Optimization.prune_big_point_after_iter` | Iteration after which to prune large points. `-1` to disable. |
| `Optimization.densify_from_iter` | Iteration to start densification. |
| `Optimization.densify_until_iter` | Iteration to stop densification. `-1` for no limit. |
| `Optimization.auto_distribute` | Enables automatic distribution of points. `0`: false, any other integer: Top 1/auto_distribute % get extra uses. |
| `Optimization.position_lr_init` | Initial learning rate for position optimization. |
| `Optimization.position_lr_final` | Final learning rate for position optimization. |
| `Optimization.position_lr_delay_mult` | Delay multiplier for position learning rate decay. |
| `Optimization.position_lr_max_steps` | Maximum steps for position optimization. |
| `Optimization.feature_lr` | Learning rate for feature optimization. |
| `Optimization.opacity_lr` | Learning rate for opacity optimization. |
| `Optimization.scaling_lr` | Learning rate for scaling optimization. |
| `Optimization.rotation_lr` | Learning rate for rotation optimization. |
| `Optimization.percent_dense` | Threshold for scaling during point densification. |
| `Optimization.lambda_dssim` | Weight for structural dissimilarity loss. |
| `Optimization.lambda_depth` | Weight for depth loss. |
| `Optimization.densification_interval` | Interval between densification operations. |
| `Optimization.densify_min_opacity` | Minimum opacity threshold for densification. |
| `Optimization.densify_grad_threshold` | Gradient threshold for densification. |
| `Optimization.opacity_reg` | Regularization weight for opacity. |

### Chunking Settings

| Parameter | Description |
|-----------|-------------|
| `Chunking.chunk_size` | Size of chunks. |
| `Chunking.max_chunks` | Maximum number of chunks allowed in VRAM. |

### Gaussian Viewer Settings

| Parameter | Description |
|-----------|-------------|
| `GaussianViewer.glfw_window_width` | Width of the GLFW window for visualization. |
| `GaussianViewer.glfw_window_height` | Height of the GLFW window for visualization. |
| `GaussianViewer.image_scale` | Scale factor for displayed images. |
| `GaussianViewer.image_scale_main` | Scale factor for the main displayed image. |
| `GaussianViewer.camera_watch_dist` | Distance for camera watching. |

## ROS Options

This section documents the ROS parameters, topics, and configuration options for the Gaussian SLAM wrapper.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocabulary_path` | string | (required) | Path to the ORB vocabulary file |
| `orb_settings_path` | string | (required) | Path to the ORB SLAM settings file |
| `gaussian_settings_path` | string | (required) | Path to the Gaussian settings file |
| `output_directory` | string | (required) | Directory where the output will be saved |
| `use_viewer` | bool | `false` | Whether to use the ImGui viewer for visualization |
| `mode` | string | `"stereo"` | Sensor mode. Options: `"mono"`, `"stereo"`, `"rgbd"`, `"stereo-imu"`, `"rgbd-imu"` |
| `left_topic` | string | `"/camera/rgb/image_raw"` | Topic for left stereo image (stereo mode) |
| `right_topic` | string | `"/camera/rgb/image_raw"` | Topic for right stereo image (stereo mode) |
| `mono_topic` | string | `"/camera/image_raw"` | Topic for monocular image (mono mode) |
| `rgb_topic` | string | `"/camera/rgb/image_raw"` | Topic for RGB image (rgbd mode) |
| `depth_topic` | string | `"/camera/depth/image_raw"` | Topic for depth image (rgbd mode) |
| `imu_topic` | string | `"/boxi/zed2i/imu/data"` | Topic for IMU data (stereo-imu and rgbd-imu modes) |
| `slam_mode` | string | `"orbslam"` | SLAM mode. Options: `"orbslam"`, `"external"`, `"hybrid"` |
| `target_frame` | string | `"map"` | Target frame for TF transformations (external and hybrid modes) |
| `source_frame` | string | `"zed2i_left_camera_frame"` | Source frame for TF transformations (external and hybrid modes) |
| `timeout_duration` | double | `20.0` | Duration (in seconds) after which the system considers data stream stopped |

### Sensor Modes

- **mono**: Uses a single camera for visual SLAM
- **stereo**: Uses a stereo camera pair for visual SLAM
- **rgbd**: Uses RGB and depth images for visual SLAM
- **stereo-imu**: Uses a stereo camera pair with IMU data for visual-inertial SLAM
- **rgbd-imu**: Uses RGB and depth images with IMU data for visual-inertial SLAM

### SLAM Modes

- **orbslam**: Uses ORB-SLAM3 for tracking and mapping
- **external**: Uses external pose information from TF transformations
- **hybrid**: Combines ORB-SLAM3 with external pose information from TF

### TF Configuration

When using `external` or `hybrid` SLAM modes, the system looks up transforms between the specified frames:

- **target_frame**: Usually the world/map frame
- **source_frame**: Usually the camera frame

The transform between these frames provides the camera pose used for mapping.

### Image Topics

The system accepts various image formats:
- Standard RGB8 encoding
- Bayer RGGB8 encoding (automatically converted to RGB8)
- For depth images: 32FC1 (in meters, converted to millimeters) or 16UC1 format

### Mapping Process Control

The system will automatically shut down the ROS node when:
- No callbacks are received for the duration specified in `timeout_duration`
- The mapping process signals completion through its callback mechanism

This behavior ensures proper termination and output saving without manual intervention.

## Common Issues

- If you get "ImportError: Cannot load backend 'Qt5Agg' which requires the 'qt' interactive framework, as 'headless' is currently running" during eval then make sure the display is forwarded. Run `xhost +local:root`
- If you get any crashes, first look for mistakes in the configuration files. If they are misconfigured, DiskChunGS will not validate them and exhibit weird behavior or crashing.
- If you run out of memory, reduce the number of `Chunking.max_chunks`. For large scenes, you should also use the `Mapper.keyframe_selection_strategy` 1.
- The `rsl_rgbd.sh` script does not play rosbags automatically; this has to be done by the user. You will also have to publish uncompressed topics.
- In case of weird errors in the dependencies, try running `./clean.sh` and building fresh.
- In case of ROS commands not working, run `source /opt/ros/noetic/setup.bash`
- After evaluation, you may encounter a segmentation fault in the python wrapper. This is a known issue but doesn't affect anything right now.

## Other Info

- There exist hypertuning scripts `hypertune_kitty.py` and `hypertune_rsl.py`. These are pretty much thrown together and you will have to change the paths/parameters inside these scripts.
- `rosbag_extractor.py` can be used to generate TUM style datasets from rosbags. It will play the bag but you will have to press space after running the script. It can also work with multiple bags if you add all their paths as arguments.
- `img2vid.py` can be used to create a video from images

## Acknowledgement

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