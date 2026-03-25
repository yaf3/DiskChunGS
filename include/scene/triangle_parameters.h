/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023
 * as part of Photo-SLAM, modified by Dapeng Feng in 2024 as part of CaRtGS,
 * and further modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 */

#pragma once

#include <filesystem>
#include <string>

/**
 * @brief Parameters for the Triangle model configuration.
 *
 * Contains paths, spherical harmonics settings, and memory constraints.
 */
class TriangleModelParams {
 public:
  TriangleModelParams(std::filesystem::path source_path = "",
                      std::filesystem::path model_path = "",
                      std::filesystem::path exec_path = "",
                      int sh_degree = 3,
                      std::string images = "images",
                      bool white_background = false,
                      std::string data_device = "cuda",
                      long max_triangles_in_memory = 1500000);

  int sh_degree_;                         ///< Degree of spherical harmonics.
  std::filesystem::path source_path_;     ///< Path to source data.
  std::filesystem::path model_path_;      ///< Path to model output.
  std::string images_;                    ///< Subdirectory name for images.
  bool white_background_;                 ///< Use white background if true.
  std::string data_device_;               ///< Device for data storage.
  long max_triangles_in_memory_;          ///< Maximum Triangles to keep in GPU memory.
};

/**
 * @brief Parameters controlling the rendering pipeline.
 */
class TrianglePipelineParams {
 public:
  TrianglePipelineParams(bool convert_SHs = false,
                         bool compute_cov3D = false,
                         bool separate_sh = true);

  bool convert_SHs_;    ///< Convert spherical harmonics to RGB on CPU.
  bool compute_cov3D_;  ///< Precompute 3D covariance matrices.
  bool separate_sh_;    ///< Store SH coefficients separately from other attributes.
};

/**
 * @brief Parameters for Triangle optimization/training.
 *
 * Contains learning rates for different attributes and loss function weights.
 */
class TriangleOptimizationParams {
 public:
  TriangleOptimizationParams(int iterations = 30'000,
                             float position_lr_init = 0.00005f,
                             float position_lr_decay = 0.99998f,
                             float feature_lr = 0.0025f,
                             float opacity_lr = 0.05f,
                             float sigma_lr = 0.005f,
                             float pose_lr = 0.0001f,
                             float exposure_lr = 0.05f,
                             float depth_scale_bias_lr = 0.0001f,
                             float lambda_dssim = 0.2f,
                             float lambda_depth = 0.001f,
                             int auto_distribute = 0,
                             bool smooth_l1 = false);

  // Training settings
  int iterations_;       ///< Total number of optimization iterations.
  int auto_distribute_;  ///< Auto distribute for loss based selection across keyframes (0 = disabled).
  bool smooth_l1_;       ///< Use smooth L1 loss instead of L1.

  // Learning rates
  float position_lr_init_;      ///< Initial learning rate for position.
  float position_lr_decay_;     ///< Per-iteration decay factor for position LR.
  float feature_lr_;            ///< Learning rate for SH features.
  float opacity_lr_;            ///< Learning rate for opacity.
  float sigma_lr_;              ///< Learning rate for triangle sigma (isotropic scale).
  float pose_lr_;               ///< Learning rate for camera pose refinement.
  float exposure_lr_;           ///< Learning rate for exposure compensation.
  float depth_scale_bias_lr_;   ///< Learning rate for depth scale/bias.

  // Loss weights
  float lambda_dssim_;  ///< Weight for D-SSIM loss term.
  float lambda_depth_;  ///< Weight for depth loss term.
};
