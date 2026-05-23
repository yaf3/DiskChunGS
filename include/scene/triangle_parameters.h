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
                      long max_vertices_in_memory = 1500000);

  int sh_degree_;                         ///< Degree of spherical harmonics.
  std::filesystem::path source_path_;     ///< Path to source data.
  std::filesystem::path model_path_;      ///< Path to model output.
  std::string images_;                    ///< Subdirectory name for images.
  bool white_background_;                 ///< Use white background if true.
  std::string data_device_;               ///< Device for data storage.
  long max_vertices_in_memory_;           ///< Maximum vertices to keep in GPU memory.
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
                             float weight_lr = 0.05f,
                             float pose_lr = 0.0001f,
                             float exposure_lr = 0.05f,
                             float depth_scale_bias_lr = 0.0001f,
                             float lambda_dssim = 0.2f,
                             float lambda_depth = 0.001f,
                             int auto_distribute = 0,
                             bool smooth_l1 = false,
                             float lambda_normal = 0.0f,
                             int normal_loss_mode = 0,
                             float lambda_weight = 0.0f,
                             float sigma_init = 0.3f,
                             float sigma_final = 0.0001f,
                             int sigma_start_iter = 0,
                             int sigma_until_iter = 30000,
                             int opacity_floor_start_iter = 800,
                             int opacity_floor_end_iter = 4000,
                             float opacity_floor_init = 0.1f,
                             float opacity_floor_final = 0.9999f,
                             int rdt_iter = 3000,
                             bool enable_rdt = true);

  // Training settings
  int iterations_;       ///< Total number of optimization iterations.
  int auto_distribute_;  ///< Auto distribute for loss based selection across keyframes (0 = disabled).
  bool smooth_l1_;       ///< Use smooth L1 loss instead of L1.

  // Learning rates
  float position_lr_init_;      ///< Initial learning rate for position.
  float position_lr_decay_;     ///< Per-iteration decay factor for position LR.
  float feature_lr_;            ///< Learning rate for SH features.
  float weight_lr_;             ///< Learning rate for per-vertex weight.
  float pose_lr_;               ///< Learning rate for camera pose refinement.
  float exposure_lr_;           ///< Learning rate for exposure compensation.
  float depth_scale_bias_lr_;   ///< Learning rate for depth scale/bias.

  // Loss weights
  float lambda_dssim_;        ///< Weight for D-SSIM loss term.
  float lambda_depth_;        ///< Weight for depth loss term.
  float lambda_normal_;       ///< Weight for normal loss term (0 = disabled).
  int normal_loss_mode_;      ///< 0=off, 1=self-consistency, 2=GT-anchored (RGBD only).
  float lambda_weight_;       ///< Weight for vertex weight regularization (0 = disabled).

  // Sigma schedule
  float sigma_init_;          ///< Initial sigma value for schedule.
  float sigma_final_;         ///< Final sigma value for schedule.
  int sigma_start_iter_;      ///< Iteration to begin sigma annealing.
  int sigma_until_iter_;      ///< Iteration when sigma reaches final value.

  // Stage 2: Opacity floor (thresholds are per-chunk opt step counts)
  int opacity_floor_start_iter_;
  int opacity_floor_end_iter_;
  float opacity_floor_init_;
  float opacity_floor_final_;

  // Stage 2: Restricted Delaunay Triangulation (per-chunk opt step threshold)
  int rdt_iter_;
  bool enable_rdt_;
};
