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

#include "rendering/triangle_renderer.h"

#include "utils/loss_utils.h"
#include "utils/profiling.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
TriangleRenderer::render(std::shared_ptr<TriangleModel> model,
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
                         torch::Tensor& projection_matrix) {
  int active_sh_degree = model->sh_degree_;

  torch::Tensor camera_center = viewpoint_camera->getCenter();
  torch::Tensor visible_indices = torch::where(visible_triangle_mask)[0];

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
      // Use centroids for SH direction computation
      torch::Tensor visible_xyz =
          model->getXYZ().index({visible_indices});

      torch::Tensor shs_view = visible_features.transpose(1, 2).view(
          {-1, 3, max_sh_degree * max_sh_degree});

      // Use visible triangles count, not full model count
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

  // Extract triangle vertices [V,3,3] and sigma [V,1] for visible triangles
  auto tri_pts =
      model->getTrianglesPoints().index({visible_indices}).contiguous();
  auto sigma =
      model->getSigmaActivation().index({visible_indices}).contiguous();

  // Centroid for screen-space tracking (means2D gradient used in densification)
  auto centroids = tri_pts.mean(/*dim=*/1);  // [V,3]
  auto screenspace_points =
      torch::zeros_like(centroids, torch::TensorOptions()
                                       .dtype(centroids.dtype())
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

  // No cov3D or rotation/scale extraction needed — triangle API uses tri_pts+sigma
  torch::Tensor rotations, cov3D_precomp;

  // Setup and run rasterization
  float tanfovx = std::tan(FoVx * 0.5f);
  float tanfovy = std::tan(FoVy * 0.5f);

  TriangleRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      projection_matrix, active_sh_degree, camera_center, false, false);

  TriangleRasterizer rasterizer(raster_settings);

  // Pass tri_pts [V,3,3] as means3D (first arg), sigma [V,1] as scales
  auto rasterizer_result = rasterizer.forward(
      tri_pts, means2D, opacity, dc, shs, colors_precomp, sigma, rotations,
      cov3D_precomp, world_view_transform);

  auto rendered_image = std::get<0>(rasterizer_result);
  auto rendered_depth = std::get<1>(rasterizer_result);
  auto mainGaussID = std::get<2>(rasterizer_result);
  auto radii = std::get<3>(rasterizer_result);
  auto scaling_visible = std::get<4>(rasterizer_result);  // [V] kernel-computed screen extent
  auto rend_normal = std::get<5>(rasterizer_result);  // [3, H, W]

  // Scatter scaling from visible-triangle space [V] to full-model space [N]
  int64_t N = model->getTrianglesPoints().size(0);
  auto scaling = torch::zeros(
      {N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  scaling.index_put_({visible_indices}, scaling_visible.squeeze(1));

  rendered_image = viewpoint_camera->applyExposureTransform(rendered_image);

  // surf_normal: normals derived from rendered depth via finite differences.
  // Used for the self-consistency normal loss (mode 1).
  float fx = viewpoint_camera->intr_[0];
  float fy = viewpoint_camera->intr_[1];
  float cx = viewpoint_camera->intr_[2];
  float cy = viewpoint_camera->intr_[3];
  auto surf_normal = loss_utils::computeNormalsFromDepth(rendered_depth, fx, fy, cx, cy);

  return std::make_tuple(rendered_depth,  // [0] depth
                         rendered_image,  // [1] render
                         radii,           // [2] radii
                         mainGaussID,     // [3] mainGaussID
                         scaling,         // [4] full-model screen extent [N]
                         rend_normal,     // [5] alpha-weighted triangle plane normals [3,H,W]
                         surf_normal      // [6] normals from rendered depth [3,H,W]
  );
}