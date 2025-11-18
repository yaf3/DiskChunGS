/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM and modified by Dapeng Feng in 2024, as part of CaRtGS.
 */

#pragma once

#include <filesystem>
#include <string>

class GaussianModelParams {
 public:
  GaussianModelParams(std::filesystem::path source_path = "",
                      std::filesystem::path model_path = "",
                      std::filesystem::path exec_path = "",
                      int sh_degree = 3,
                      std::string images = "images",
                      bool white_background = false,
                      std::string data_device = "cuda",
                      long max_gaussians_in_memory = 1500000);

 public:
  int sh_degree_;
  std::filesystem::path source_path_;
  std::filesystem::path model_path_;
  std::string images_;
  bool white_background_;
  std::string data_device_;
  long max_gaussians_in_memory_;
};

class GaussianPipelineParams {
 public:
  GaussianPipelineParams(bool convert_SHs = false,
                         bool compute_cov3D = false,
                         bool separate_sh = true);

 public:
  bool convert_SHs_;
  bool compute_cov3D_;
  bool separate_sh_;
};

class GaussianOptimizationParams {
 public:
  GaussianOptimizationParams(int iterations = 30'000,
                             float position_lr_init = 0.00005f,
                             float position_lr_decay_ = 0.99998f,
                             float feature_lr = 0.0025f,
                             float opacity_lr = 0.05f,
                             float scaling_lr = 0.005f,
                             float rotation_lr = 0.001f,
                             float pose_lr = 0.0001f,
                             float exposure_lr = 0.05f,
                             float depth_scale_bias_lr = 0.0001f,
                             float lambda_dssim = 0.2f,
                             float lambda_depth = 0.001f,
                             int auto_distribute = 0,
                             bool smooth_l1 = false,
                             float opacity_reg = 0.01f);

 public:
  int iterations_;
  float position_lr_init_;
  float position_lr_decay_;
  float feature_lr_;
  float opacity_lr_;
  float scaling_lr_;
  float rotation_lr_;
  float pose_lr_;
  float exposure_lr_;
  float depth_scale_bias_lr_;
  float lambda_dssim_;
  float lambda_depth_;
  int auto_distribute_;
  bool smooth_l1_;
  float opacity_reg_;
};
