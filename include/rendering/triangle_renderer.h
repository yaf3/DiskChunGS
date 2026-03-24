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

#include "model/triangle_model.h"
#include "rendering/triangle_rasterizer.h"
#include "scene/triangle_keyframe.h"
#include "scene/triangle_parameters.h"
#include "utils/sh_utils.h"

/**
 * @brief Renders 3D Triangle primitives to a 2D image using splatting.
 *
 * This class provides differentiable Triangle splatting rendering, converting
 * a set of 3D Triangles into rendered images with depth information.
 */
class TriangleRenderer {
 public:
  /**
   * @brief Renders visible Triangles from the given viewpoint.
   *
   * Performs differentiable rasterization of 3D Triangles. Colors can be
   * provided directly, computed from spherical harmonics (SH) on CPU, or
   * converted from SH by the rasterizer on GPU.
   *
   * @param model The Triangle model containing 3D Triangle primitives.
   * @param visible_triangle_mask Boolean mask indicating which Triangles are
   *        visible from the current viewpoint.
   * @param viewpoint_camera Camera keyframe containing pose and exposure info.
   * @param image_height Output image height in pixels.
   * @param image_width Output image width in pixels.
   * @param pipe Pipeline parameters controlling SH conversion and covariance.
   * @param bg_color Background color tensor (RGB).
   * @param override_color Pre-computed colors to use (if use_override_color).
   * @param scaling_modifier Scale factor applied to Triangle sizes.
   * @param use_override_color If true, use override_color instead of SH.
   * @param FoVx Horizontal field of view in radians.
   * @param FoVy Vertical field of view in radians.
   * @param world_view_transform 4x4 world-to-camera transformation matrix.
   * @param projection_matrix 4x4 camera projection matrix.
   * @return Tuple of (depth, rendered_image, radii, main_triangle_ids).
   */
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  render(std::shared_ptr<TriangleModel> model,
         const torch::Tensor& visible_triangle_mask,
         std::shared_ptr<TriangleKeyframe> viewpoint_camera,
         int image_height,
         int image_width,
         TrianglePipelineParams& pipe,
         torch::Tensor& bg_color,
         torch::Tensor& override_color,
         float scaling_modifier,
         bool use_override_color,
         float FoVx,
         float FoVy,
         torch::Tensor& world_view_transform,
         torch::Tensor& projection_matrix);
};
