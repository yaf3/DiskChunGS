/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * This file is Derivative Works of Gaussian Splatting,
 * modified by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023
 * as part of Photo-SLAM, modified by Dapeng Feng in 2024 as part of CaRtGS,
 * and further modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 *
 * For inquiries contact george.drettakis@inria.fr
 */

#pragma once

#include <torch/torch.h>

#include <memory>
#include <tuple>

#include "model/gaussian_model.h"
#include "rendering/gaussian_rasterizer.h"
#include "scene/gaussian_keyframe.h"
#include "scene/gaussian_parameters.h"
#include "utils/sh_utils.h"

class GaussianRenderer {
 public:
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  render(std::shared_ptr<GaussianModel> pc,
         const torch::Tensor& visible_gaussian_mask,
         std::shared_ptr<GaussianKeyframe> viewpoint_camera,
         int image_height,
         int image_width,
         GaussianPipelineParams& pipe,
         torch::Tensor& bg_color,
         torch::Tensor& override_color,
         float scaling_modifier,
         bool use_override_color,
         float FoVx,
         float FoVy,
         torch::Tensor& world_view_transform,
         torch::Tensor& projection_matrix);
};
