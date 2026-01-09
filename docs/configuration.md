# Configuration Options

This document explains all available configuration options for DiskChunGS. The configuration file uses YAML format and provides fine-grained control over model behavior, mapping, optimization, visualization, and more.

## Model Settings

| Parameter | Description |
|-----------|-------------|
| `Model.resolution` | Resolution for model rendering. Set to `-1` for automatic resolution. |
| `Model.white_background` | Controls background color. `0`: false (black), `1` or other integer: true (white). |
| `Model.eval` | DEPRECATED. Enables evaluation mode. `0`: false (training mode), `1` or other integer: true (evaluation mode). |
| `Model.sh_degree` | Spherical harmonics degree for appearance modeling. Higher values allow more complex lighting effects. |
| `Model.appearance_embedding` | Controls whether to use appearance embeddings. `0` disables appearance embedding. |

## Camera Settings

| Parameter | Description |
|-----------|-------------|
| `Camera.z_near` | Near clipping plane distance. |
| `Camera.z_far` | Far clipping plane distance. |

## Monocular Settings

| Parameter | Description |
|-----------|-------------|
| `Monocular.inactive_geo_densify_max_pixel_dist` | Maximum squared pixel distance for inactive geometry densification. |

## Stereo Settings

| Parameter | Description |
|-----------|-------------|
| `Stereo.min_disparity` | Minimum disparity value for stereo processing. |
| `Stereo.num_disparity` | Number of disparity levels for stereo matching. |
| `Stereo.do_stereo_loss` | Enables stereo loss calculation. `0`: false, `1` or other integer: true. |

## Mapper Settings

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

## Gaussian Pyramid Settings

| Parameter | Description |
|-----------|-------------|
| `GausPyramid.do` | Enables Gaussian pyramid processing. `0`: false, `1` or other integer: true. |
| `GausPyramid.num_levels` | Number of sub-levels in the Gaussian pyramid. |
| `GausPyramid.level_times_of_use` | Number of times to use each sub-level. |

## Pipeline Settings

| Parameter | Description |
|-----------|-------------|
| `Pipeline.convert_SHs` | Enables conversion of spherical harmonics. `0`: false, `1` or other integer: true. |
| `Pipeline.compute_cov3D` | Enables computation of 3D covariance. `0`: false, `1` or other integer: true. |

## Recording Settings

| Parameter | Description |
|-----------|-------------|
| `Record.keyframe_record_interval` | Interval for recording keyframes. `0`: never, `1`: always, other values: periodically. |
| `Record.all_keyframes_record_interval` | Interval for recording all keyframes. `0`: never, `1`: always, other values: periodically. |
| `Record.record_rendered_image` | Enables recording of rendered images. `0`: false, `1` or other integer: true. |
| `Record.record_ground_truth_image` | Enables recording of ground truth images. `0`: false, `1` or other integer: true. |
| `Record.record_loss_image` | Enables recording of loss visualization images. `0`: false, `1` or other integer: true. |
| `Record.training_report_interval` | Interval for generating training reports. `0`: never, `1`: always, other values: periodically. |
| `Record.record_loop_ply` | DEPRECATED. Enables recording of loop closure point clouds. `0`: false, `1` or other integer: true. |

## Optimization Settings

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

## Chunking Settings

| Parameter | Description |
|-----------|-------------|
| `Chunking.chunk_size` | Size of chunks. |
| `Chunking.max_chunks` | Maximum number of chunks allowed in VRAM. |

## Gaussian Viewer Settings

| Parameter | Description |
|-----------|-------------|
| `GaussianViewer.glfw_window_width` | Width of the GLFW window for visualization. |
| `GaussianViewer.glfw_window_height` | Height of the GLFW window for visualization. |
| `GaussianViewer.image_scale` | Scale factor for displayed images. |
| `GaussianViewer.image_scale_main` | Scale factor for the main displayed image. |
| `GaussianViewer.camera_watch_dist` | Distance for camera watching. |