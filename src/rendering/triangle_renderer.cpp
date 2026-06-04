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

  auto vertices = model->getVertices();
  auto triangle_indices = model->getTriangleIndices();
  int64_t V_full = vertices.size(0);

  auto vis_tri_idx = triangle_indices.index({visible_indices});
  auto vis_vert_set = std::get<0>(torch::_unique(vis_tri_idx.flatten().to(torch::kLong)));

  auto vert_remap = torch::full({V_full}, -1, torch::TensorOptions().dtype(torch::kLong).device(vertices.device()));
  vert_remap.index_put_({vis_vert_set}, torch::arange(vis_vert_set.size(0), torch::TensorOptions().dtype(torch::kLong).device(vertices.device())));

  auto vis_vertices = vertices.index({vis_vert_set});
  auto vis_local_tri = vert_remap.index({vis_tri_idx.flatten().to(torch::kLong)}).reshape(vis_tri_idx.sizes()).to(torch::kInt32);

  auto vis_vertex_weights = model->getVertexWeightActivation().index({vis_vert_set});
  float sigma = model->getSigmaActivation();

  torch::Tensor dc, shs, colors_precomp;
  if (use_override_color) {
    colors_precomp = override_color.index({vis_vert_set}).contiguous();
  } else {
    if (pipe.convert_SHs_) {
      int max_sh_degree = model->sh_degree_ + 1;

      torch::Tensor visible_features =
          model->getFeatures().index({vis_vert_set});
      torch::Tensor visible_xyz = vis_vertices;

      torch::Tensor shs_view = visible_features.transpose(1, 2).view(
          {-1, 3, max_sh_degree * max_sh_degree});

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
        dc = model->features_dc_.index({vis_vert_set}).clone().contiguous();
        shs =
            model->features_rest_.index({vis_vert_set}).clone().contiguous();
      } else {
        shs = model->getFeatures().index({vis_vert_set}).contiguous();
      }
    }
  }

  float tanfovx = std::tan(FoVx * 0.5f);
  float tanfovy = std::tan(FoVy * 0.5f);

  TriangleRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      projection_matrix, active_sh_degree, camera_center, false, false);

  TriangleRasterizer rasterizer(raster_settings);

  auto rasterizer_result = rasterizer.forward(
      vis_vertices, vis_local_tri, vis_vertex_weights, sigma,
      dc, shs, colors_precomp, world_view_transform);

  auto rendered_image = std::get<0>(rasterizer_result);
  auto rendered_depth = std::get<1>(rasterizer_result);
  auto mainGaussID = std::get<2>(rasterizer_result);
  auto radii = std::get<3>(rasterizer_result);
  auto scaling_visible = std::get<4>(rasterizer_result);
  auto rend_normal = std::get<5>(rasterizer_result);

  auto rendered_inv_depth = torch::where(
      rendered_depth > 1e-6f,
      1.0f / rendered_depth,
      torch::zeros_like(rendered_depth));

  int64_t N = triangle_indices.size(0);
  auto scaling = torch::zeros(
      {N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  scaling.index_put_({visible_indices}, scaling_visible.squeeze(1));

  rendered_image = viewpoint_camera->applyExposureTransform(rendered_image);

  float fx = viewpoint_camera->intr_[0];
  float fy = viewpoint_camera->intr_[1];
  float cx = viewpoint_camera->intr_[2];
  float cy = viewpoint_camera->intr_[3];
  auto surf_normal = loss_utils::computeNormalsFromDepth(rendered_inv_depth, fx, fy, cx, cy);

  return std::make_tuple(rendered_inv_depth,
                         rendered_image,
                         radii,
                         mainGaussID,
                         scaling,
                         rend_normal,
                         surf_normal
  );
}
