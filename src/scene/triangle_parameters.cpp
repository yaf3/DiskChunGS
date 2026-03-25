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

#include "scene/triangle_parameters.h"

TriangleModelParams::TriangleModelParams(std::filesystem::path source_path,
                                         std::filesystem::path model_path,
                                         std::filesystem::path exec_path,
                                         int sh_degree,
                                         std::string images,
                                         bool white_background,
                                         std::string data_device,
                                         long max_triangles_in_memory)
    : sh_degree_(sh_degree),
      images_(images),
      white_background_(white_background),
      data_device_(data_device),
      max_triangles_in_memory_(max_triangles_in_memory) {
  if (source_path.is_absolute())
    source_path_ = source_path;
  else
    source_path_ = exec_path / source_path;

  if (model_path.is_absolute())
    model_path_ = model_path;
  else
    model_path_ = exec_path / model_path;
}

TrianglePipelineParams::TrianglePipelineParams(bool convert_SHs,
                                               bool compute_cov3D,
                                               bool separate_sh)
    : convert_SHs_(convert_SHs),
      compute_cov3D_(compute_cov3D),
      separate_sh_(separate_sh) {}

TriangleOptimizationParams::TriangleOptimizationParams(
    int iterations,
    float position_lr_init,
    float position_lr_decay,
    float feature_lr,
    float opacity_lr,
    float sigma_lr,
    float pose_lr,
    float exposure_lr,
    float depth_scale_bias_lr,
    float lambda_dssim,
    float lambda_depth,
    int auto_distribute,
    bool smooth_l1)
    : iterations_(iterations),
      auto_distribute_(auto_distribute),
      smooth_l1_(smooth_l1),
      position_lr_init_(position_lr_init),
      position_lr_decay_(position_lr_decay),
      feature_lr_(feature_lr),
      opacity_lr_(opacity_lr),
      sigma_lr_(sigma_lr),
      pose_lr_(pose_lr),
      exposure_lr_(exposure_lr),
      depth_scale_bias_lr_(depth_scale_bias_lr),
      lambda_dssim_(lambda_dssim),
      lambda_depth_(lambda_depth) {}
