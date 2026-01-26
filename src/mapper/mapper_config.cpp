/**
 * This file is part of DiskChunGS, modified from CaRtGS/Photo-SLAM.
 *
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See the GNU General Public License for more details:
 * <http://www.gnu.org/licenses/>.
 */

#include <filesystem>
#include <iostream>

#include "gaussian_mapper.h"

void copyFolder(const std::filesystem::path& source,
                const std::filesystem::path& destination) {
  // Create the destination directory if it doesn't exist
  std::filesystem::create_directories(destination);

  // Iterate through the source directory
  for (const auto& entry : std::filesystem::directory_iterator(source)) {
    const auto& path = entry.path();
    const auto destPath = destination / path.filename();

    if (std::filesystem::is_directory(path)) {
      // Recursively copy subdirectories
      copyFolder(path, destPath);
    } else {
      // Copy files
      std::filesystem::copy_file(
          path, destPath, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path) {
  cv::FileStorage settings_file(cfg_path.string().c_str(),
                                cv::FileStorage::READ);
  if (!settings_file.isOpened()) {
    std::cerr << "[Gaussian Mapper]Failed to open settings file at: "
              << cfg_path << std::endl;
    exit(-1);
  }

  std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path
            << std::endl;
  std::unique_lock<std::mutex> lock(mutex_settings_);

  // Model parameters
  model_params_.sh_degree_ = settings_file["Model.sh_degree"].operator int();
  model_params_.white_background_ =
      (settings_file["Model.white_background"].operator int()) != 0;
  model_params_.max_gaussians_in_memory_ =
      settings_file["Model.max_gaussians_in_memory"].operator int();
  init_proba_scaler_ =
      settings_file["Model.init_proba_scaler"].operator float();
  downsample_for_sampling_ =
      (settings_file["Model.downsample_for_sampling"].operator int()) != 0;

  // Pipeline Parameters
  z_near_ = settings_file["Camera.z_near"].operator float();
  z_far_ = settings_file["Camera.z_far"].operator float();

  min_depth_ = settings_file["Mapper.min_depth_"].operator float();
  max_depth_ = settings_file["Mapper.max_depth_"].operator float();

  min_num_initial_map_kfs_ = static_cast<unsigned long>(
      settings_file["Mapper.min_num_initial_map_kfs"].operator int());
  new_keyframe_times_of_use_ =
      settings_file["Mapper.new_keyframe_times_of_use"].operator int();
  local_BA_increased_times_of_use_ =
      settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
  loop_closure_increased_times_of_use_ =
      settings_file["Mapper.loop_closure_increased_times_of_use_"]
          .operator int();
  large_rot_th_ =
      settings_file["Mapper.large_rotation_threshold"].operator float();
  large_trans_th_ =
      settings_file["Mapper.large_translation_threshold"].operator float();
  stable_num_iter_existence_ =
      settings_file["Mapper.stable_num_iter_existence"].operator int();

  min_keyframe_translation_ =
      settings_file["External.min_keyframe_translation"].operator float();
  min_keyframe_rotation_ =
      settings_file["External.min_keyframe_rotation"].operator float();
  min_keyframe_time_ =
      settings_file["External.min_keyframe_time"].operator float();

  pipe_params_.convert_SHs_ =
      (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
  pipe_params_.compute_cov3D_ =
      (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;
  num_gaus_pyramid_sub_levels_ =
      settings_file["GausPyramid.num_levels"].operator int();
  int sub_level_times_of_use =
      settings_file["GausPyramid.level_times_of_use"].operator int();
  kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
  kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    kf_gaus_pyramid_factors_[l] = std::pow(0.5f, l);
  }

  keyframe_record_interval_ =
      settings_file["Record.keyframe_record_interval"].operator int();
  all_keyframes_record_interval_ =
      settings_file["Record.all_keyframes_record_interval"].operator int();
  record_rendered_image_ =
      (settings_file["Record.record_rendered_image"].operator int()) != 0;
  record_ground_truth_image_ =
      (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
  record_loss_image_ =
      (settings_file["Record.record_loss_image"].operator int()) != 0;
  training_report_interval_ =
      settings_file["Record.training_report_interval"].operator int();
  record_loop_ply_ =
      (settings_file["Record.record_loop_ply"].operator int()) != 0;

  // Optimization Parameters
  opt_params_.iterations_ =
      settings_file["Optimization.max_num_iterations"].operator int();
  opt_params_.position_lr_init_ =
      settings_file["Optimization.position_lr_init"].operator float();
  opt_params_.position_lr_decay_ =
      settings_file["Optimization.position_lr_decay"].operator float();
  opt_params_.feature_lr_ =
      settings_file["Optimization.feature_lr"].operator float();
  opt_params_.opacity_lr_ =
      settings_file["Optimization.opacity_lr"].operator float();
  opt_params_.scaling_lr_ =
      settings_file["Optimization.scaling_lr"].operator float();
  opt_params_.rotation_lr_ =
      settings_file["Optimization.rotation_lr"].operator float();
  opt_params_.pose_lr_ = settings_file["Optimization.pose_lr"].operator float();
  opt_params_.exposure_lr_ =
      settings_file["Optimization.exposure_lr"].operator float();
  opt_params_.depth_scale_bias_lr_ =
      settings_file["Optimization.depth_scale_bias_lr"].operator float();
  opt_params_.smooth_l1_ =
      (settings_file["Optimization.smooth_l1"].operator int()) != 0;
  opt_params_.opacity_reg_ =
      settings_file["Optimization.opacity_reg"].operator float();

  opt_params_.lambda_dssim_ =
      settings_file["Optimization.lambda_dssim"].operator float();
  opt_params_.lambda_depth_ =
      settings_file["Optimization.lambda_depth"].operator float();
  opt_params_.auto_distribute_ =
      settings_file["Optimization.auto_distribute"].operator int();
  exposure_optimization_ =
      settings_file["Optimization.exposure_optimization"].operator int();

  // Viewer Parameters
  rendered_image_viewer_scale_ =
      settings_file["GaussianViewer.image_scale"].operator float();
  rendered_image_viewer_scale_main_ =
      settings_file["GaussianViewer.image_scale_main"].operator float();

  chunk_size_ = settings_file["Chunking.chunk_size"].operator float();
}
