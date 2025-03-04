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

#include "include/gaussian_renderer.h"

/**
 * @brief
 *
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which
 * are all `torch::Tensor`
 */
std::
    tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    GaussianRenderer::render(
        const std::vector<std::shared_ptr<GaussianModel>>& models,
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> pc,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        float scaling_modifier,
        bool use_override_color) {
  /* Render the scene.

     Background tensor (bg_color) must be on GPU!
   */

  std::vector<torch::Tensor> means3D_vec;
  std::vector<torch::Tensor> means2D_vec;
  std::vector<torch::Tensor> opacity_vec;
  std::vector<torch::Tensor> dc_vec;
  std::vector<torch::Tensor> shs_vec;
  std::vector<torch::Tensor> colors_precomp_vec;
  std::vector<torch::Tensor> scales_vec;
  std::vector<torch::Tensor> rotations_vec;
  std::vector<torch::Tensor> cov3D_precomp_vec;
  std::vector<torch::Tensor> screenspace_points_vec;

  // Track model sizes to split radii later
  std::vector<int> model_sizes;
  int total_points = 0;

  int num_models = models.size();

  for (int model_idx = 0; model_idx < num_models; model_idx++) {
    const auto& pc = models[model_idx];

    model_sizes.push_back(pc->getXYZ().sizes()[0]);
    total_points += pc->getXYZ().sizes()[0];

    // Create zero tensor. We will use it to make pytorch return gradients of
    // the 2D (screen-space) means
    auto screenspace_points =
        torch::zeros_like(pc->getXYZ(), torch::TensorOptions()
                                            .dtype(pc->getXYZ().dtype())
                                            .requires_grad(true)
                                            .device(torch::kCUDA));
    try {
      screenspace_points.retain_grad();
    } catch (const std::exception& e) {
      // pass
    }

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be
    computed from scaling / rotation by the rasterizer.
  */
    torch::Tensor scales, rotations, cov3D_precomp;
    if (pipe.compute_cov3D_) {
      cov3D_precomp = pc->getCovarianceActivation();
    } else {
      scales = pc->getScalingActivation();
      rotations = pc->getRotationActivation();
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired
       to precompute colors from SHs in Python, do it. If not, then SH -> RGB
       conversion will be done by rasterizer.
     */
    torch::Tensor dc, shs, colors_precomp;
    if (use_override_color) {
      colors_precomp = override_color;
    } else {
      if (pipe.convert_SHs_) {
        int max_sh_degree = pc->max_sh_degree_ + 1;
        torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view(
            {-1, 3, max_sh_degree * max_sh_degree});
        torch::Tensor dir_pp =
            (pc->getXYZ() - viewpoint_camera->camera_center_.repeat(
                                {pc->getFeatures().size(0), 1}));
        auto dir_pp_normalized =
            dir_pp /
            torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
        auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view,
                                        dir_pp_normalized);
        colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
      } else {
        if (pipe.separate_sh_) {
          dc = pc->features_dc_.clone();
          shs = pc->features_rest_.clone();
        } else {
          shs = pc->getFeatures();
        }
      }
    }

    // Use push_back instead of indexing into unsized vectors
    means3D_vec.push_back(means3D);
    means2D_vec.push_back(means2D);
    opacity_vec.push_back(opacity);

    // Only push_back non-empty tensors
    if (!dc.numel() == 0) dc_vec.push_back(dc);
    if (!shs.numel() == 0) shs_vec.push_back(shs);
    if (!colors_precomp.numel() == 0)
      colors_precomp_vec.push_back(colors_precomp);
    if (!scales.numel() == 0) scales_vec.push_back(scales);
    if (!rotations.numel() == 0) rotations_vec.push_back(rotations);
    if (!cov3D_precomp.numel() == 0) cov3D_precomp_vec.push_back(cov3D_precomp);

    screenspace_points_vec.push_back(screenspace_points);
  }

  torch::Tensor means3D = torch::cat(means3D_vec, 0);
  torch::Tensor means2D = torch::cat(means2D_vec, 0);
  torch::Tensor opacity = torch::cat(opacity_vec, 0);
  torch::Tensor dc, shs, colors_precomp, scales, rotations, cov3D_precomp;

  if (!dc_vec.empty()) dc = torch::cat(dc_vec, 0);
  if (!shs_vec.empty()) shs = torch::cat(shs_vec, 0);
  if (!colors_precomp_vec.empty())
    colors_precomp = torch::cat(colors_precomp_vec, 0);
  if (!scales_vec.empty()) scales = torch::cat(scales_vec, 0);
  if (!rotations_vec.empty()) rotations = torch::cat(rotations_vec, 0);
  if (!cov3D_precomp_vec.empty())
    cov3D_precomp = torch::cat(cov3D_precomp_vec, 0);

  torch::Tensor screenspace_points = torch::cat(screenspace_points_vec, 0);

  // Set up rasterization configuration
  float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
  float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

  GaussianRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      viewpoint_camera->world_view_transform_,
      viewpoint_camera->full_proj_transform_, pc->active_sh_degree_,
      viewpoint_camera->camera_center_, false, false);

  GaussianRasterizer rasterizer(raster_settings);

  // Rasterize visible Gaussians to image, obtain their radii (on screen).
  auto rasterizer_result =
      rasterizer.forward(means3D, means2D, opacity, dc, shs, colors_precomp,
                         scales, rotations, cov3D_precomp);
  auto rendered_image = std::get<0>(rasterizer_result);
  auto radii = std::get<1>(rasterizer_result);

  // Split the radii tensor into separate tensors per model
  std::vector<torch::Tensor> radii_vec;
  int offset = 0;
  for (int model_idx = 0; model_idx < num_models; model_idx++) {
    int size = model_sizes[model_idx];
    radii_vec.push_back(radii.slice(0, offset, offset + size));
    offset += size;
  }

  /* Those Gaussians that were frustum culled or had a radius of 0 were not
     visible. They will be excluded from value updates used in the splitting
     criteria.
   */
  return std::make_tuple(rendered_image,         /*render*/
                         screenspace_points_vec, /*viewspace_points*/
                         radii_vec /*radii*/);
}
