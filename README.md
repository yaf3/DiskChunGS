# DiskChunGS: Memory-Unbounded 3D Gaussian SLAM Through Efficient Disk Chunking
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`]()] [[`Project`]()] [[`Demo`]()] [[`Dataset`]()] [[`BibTeX`]()]

![Pipeline](assets/main_pipeline_resized.png?raw=true)

DiskChunGS is a 3D Gaussian Splatting SLAM system that enables unbounded scene reconstruction through dynamic memory management, partitioning environments into spatial chunks that are selectively loaded between GPU and disk storage. This innovative approach achieves substantially higher Gaussian density than previous methods, resulting in significantly improved reconstruction quality across diverse environments while maintaining real-time performance.

## Installation

Installation herer (docker image)

## Getting Started

The benchmark datasets mentioned in our paper: [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam), [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

0. Create a dataset folder which we can bind to the docker container. 

1. Download the desired dataset
```
scripts/download_replica.sh
scripts/download_tum.sh
```

For KITTI:
- Download odometry data set (color, 65 GB)
- Download odometry ground truth poses (4 MB)
- Download odometry data set (calibration files, 1 MB)

Then combine these to create the following file structure
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
            ---
```

2. For testing, you could use the below commands to run the system after specifying the `PATH_TO_Replica` and `PATH_TO_SAVE_RESULTS`. We would disable the viewer by adding `no_viewer` during the evaluation.
``` bash
bin/replica_rgbd \
    ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    PATH_TO_Replica/office0 \
    PATH_TO_SAVE_RESULTS
    # no_viewer
```

3. We also provide scripts to conduct experiments on all benchmark datasets mentioned in our paper. In case you use a different data location, you need to change the dataset root lines in scripts/*.sh. Specify the experiment name and the number of trials and then run:
``` bash
scripts/replica_mono.sh exp_name num_trials
scripts/replica_rgbd.sh exp_name num_trials
scripts/tum_mono.sh exp_name num_trials
scripts/tum_rgbd.sh exp_name num_trials
scripts/kitti_stereo.sh exp_name num_trials
scripts/rsl_rgbd.sh exp_name num_trials
# etc.
```

## DiskChunGS Evaluation
To use evaluate, your results need to be in the expected format. If you use our `./xxx.sh` scripts to conduct your experiments, the results are stored like
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

### Install the the python wrapper for rendering:
```
python3 setup.py install
```

### Convert Replica GT camera pose files to suitable pose files to run EVO package
``` bash
python3 eval/shapeReplicaGT.py --replica_dataset_path PATH_TO_REPLICA_DATASET
```

### To get all metrics, you can run
``` bash
python3 eval/eval.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder RESULTS_PATH
```

- PATH_TO_ALL_DATASET: Should be /data if you've bound your datasets folder to /data
Results will be summarized in two files: `RESULTS_PATH/log.txt` and `RESULTS_PATH/log.csv`.


## Configuration Options

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


## Common Issues:
- If you get "ImportError: Cannot load backend 'Qt5Agg' which requires the 'qt' interactive framework, as 'headless' is currently running" during eval then make sure the display is forwarded. Run "xhost +local:root"
- If you get any crashes a first thing to look for are mistakes in the configuration files. If they are misconfigured DiskChunGS will not validate them and exhibit weird behavior or crashing. 

## Acknowledgement
This work incorporates many open-source codes. Thanks for their great work!
- [CaRtGS](https://github.com/DapengFeng/cartgs)
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs)
- [Frustum Culling](https://bruop.github.io/frustum_culling/)

# Citation
If you find this work useful in your research, consider citing it:
```

```