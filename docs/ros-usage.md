# ROS Usage

This document provides comprehensive information about using DiskChunGS with ROS (Robot Operating System).

## Setup

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

## Running the ROS Node

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

## ROS Parameters

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