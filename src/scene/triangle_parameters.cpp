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
                                         long max_vertices_in_memory)
    : sh_degree_(sh_degree),
      images_(images),
      white_background_(white_background),
      data_device_(data_device),
      max_vertices_in_memory_(max_vertices_in_memory) {
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
    float weight_lr,
    float pose_lr,
    float exposure_lr,
    float depth_scale_bias_lr,
    float lambda_dssim,
    float lambda_depth,
    int auto_distribute,
    bool smooth_l1,
    float lambda_normal,
    int normal_loss_mode,
    float lambda_weight,
    float sigma_init,
    float sigma_final,
    int sigma_start_iter,
    int sigma_until_iter,
    int opacity_floor_start_iter,
    int opacity_floor_end_iter,
    float opacity_floor_init,
    float opacity_floor_final,
    int rdt_iter,
    bool enable_rdt,
    int rdt_update_interval,
    float lambda_vertex_depth,
    float max_vertex_depth_diff,
    int vertex_depth_mode)
    : iterations_(iterations),
      auto_distribute_(auto_distribute),
      smooth_l1_(smooth_l1),
      position_lr_init_(position_lr_init),
      position_lr_decay_(position_lr_decay),
      feature_lr_(feature_lr),
      weight_lr_(weight_lr),
      pose_lr_(pose_lr),
      exposure_lr_(exposure_lr),
      depth_scale_bias_lr_(depth_scale_bias_lr),
      lambda_dssim_(lambda_dssim),
      lambda_depth_(lambda_depth),
      lambda_normal_(lambda_normal),
      normal_loss_mode_(normal_loss_mode),
      lambda_weight_(lambda_weight),
      sigma_init_(sigma_init),
      sigma_final_(sigma_final),
      sigma_start_iter_(sigma_start_iter),
      sigma_until_iter_(sigma_until_iter),
      opacity_floor_start_iter_(opacity_floor_start_iter),
      opacity_floor_end_iter_(opacity_floor_end_iter),
      opacity_floor_init_(opacity_floor_init),
      opacity_floor_final_(opacity_floor_final),
      rdt_iter_(rdt_iter),
      enable_rdt_(enable_rdt),
      rdt_update_interval_(rdt_update_interval),
      lambda_vertex_depth_(lambda_vertex_depth),
      max_vertex_depth_diff_(max_vertex_depth_diff),
      vertex_depth_mode_(vertex_depth_mode) {}
