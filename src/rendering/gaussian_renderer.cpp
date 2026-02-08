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

#include "rendering/gaussian_renderer.h"

#include "utils/profiling.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(std::shared_ptr<GaussianModel> model,
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
                         torch::Tensor& projection_matrix) {
  int active_sh_degree = model->sh_degree_;

  torch::Tensor camera_center = viewpoint_camera->getCenter();
  torch::Tensor visible_indices = torch::where(visible_gaussian_mask)[0];

  // Prepare color data: either use override colors, convert SH to RGB on CPU,
  // or pass SH coefficients to the rasterizer for GPU conversion.
  torch::Tensor dc, shs, colors_precomp;
  if (use_override_color) {
    colors_precomp = override_color.index({visible_indices}).contiguous();
  } else {
    if (pipe.convert_SHs_) {
      int max_sh_degree = model->sh_degree_ + 1;

      // Extract visible features first
      torch::Tensor visible_features =
          model->getFeatures().index({visible_indices});
      torch::Tensor visible_xyz = model->getXYZ().index({visible_indices});

      torch::Tensor shs_view = visible_features.transpose(1, 2).view(
          {-1, 3, max_sh_degree * max_sh_degree});

      // Use visible gaussians count, not full model count
      torch::Tensor dir_pp =
          (visible_xyz -
           viewpoint_camera->camera_center_.repeat({visible_xyz.size(0), 1}));

      auto dir_pp_normalized =
          dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
      auto sh2rgb =
          sh_utils::eval_sh(model->sh_degree_, shs_view, dir_pp_normalized);
      colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0).contiguous();
    } else {
      if (pipe.separate_sh_) {
        dc = model->features_dc_.index({visible_indices}).clone().contiguous();
        shs =
            model->features_rest_.index({visible_indices}).clone().contiguous();
      } else {
        shs = model->getFeatures().index({visible_indices}).contiguous();
      }
    }
  }

  auto means3D = model->getXYZ().index({visible_indices}).contiguous();

  auto screenspace_points =
      torch::zeros_like(means3D, torch::TensorOptions()
                                     .dtype(means3D.dtype())
                                     .requires_grad(true)
                                     .device(torch::kCUDA))
          .contiguous();
  try {
    screenspace_points.retain_grad();
  } catch (const std::exception& e) {
    // pass
  }
  auto means2D = screenspace_points;
  auto opacity =
      model->getOpacityActivation().index({visible_indices}).contiguous();

  // Prepare Gaussian shape: either precompute 3D covariance or use
  // scale/rotation.
  torch::Tensor scales, rotations, cov3D_precomp;
  if (pipe.compute_cov3D_) {
    cov3D_precomp =
        model->getCovarianceActivation().index({visible_indices}).contiguous();
  } else {
    scales =
        model->getScalingActivation().index({visible_indices}).contiguous();
    rotations =
        model->getRotationActivation().index({visible_indices}).contiguous();
  }

  // Setup and run rasterization
  float tanfovx = std::tan(FoVx * 0.5f);
  float tanfovy = std::tan(FoVy * 0.5f);

  GaussianRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      projection_matrix, active_sh_degree, camera_center, false, false);

  GaussianRasterizer rasterizer(raster_settings);

  auto rasterizer_result = rasterizer.forward(
      means3D, means2D, opacity, dc, shs, colors_precomp, scales, rotations,
      cov3D_precomp, world_view_transform);

  auto rendered_image = std::get<0>(rasterizer_result);
  auto rendered_depth = std::get<1>(rasterizer_result);
  auto mainGaussID = std::get<2>(rasterizer_result);
  auto radii = std::get<3>(rasterizer_result);

  rendered_image = viewpoint_camera->applyExposureTransform(rendered_image);

  return std::make_tuple(rendered_depth,  // depth
                         rendered_image,  // render
                         radii,           // radii
                         mainGaussID      // mainGaussID
  );
}