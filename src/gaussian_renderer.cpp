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

#include "include/profiling.h"

void assertTensorDims(const std::vector<torch::Tensor>& tensors,
                      const std::string& tensor_name) {
  if (tensors.empty()) return;

  int first_dim = tensors[0].dim();
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].dim() != first_dim) {
      std::stringstream ss;
      ss << "Dimension mismatch in " << tensor_name << " tensors. ";
      ss << "Expected dim=" << first_dim << ", but tensor at index " << i
         << " has dim=" << tensors[i].dim() << ". ";
      ss << "First tensor shape: [";
      for (size_t d = 0; d < tensors[0].dim(); d++) {
        ss << tensors[0].size(d);
        if (d < tensors[0].dim() - 1) ss << ", ";
      }
      ss << "], Mismatched tensor shape: [";
      for (size_t d = 0; d < tensors[i].dim(); d++) {
        ss << tensors[i].size(d);
        if (d < tensors[i].dim() - 1) ss << ", ";
      }
      ss << "]";
      throw std::runtime_error(ss.str());
    }
  }
}

/**
 * @brief
 *
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>,
 which
 * are all `torch::Tensor`
 */
std::tuple<torch::Tensor,
           torch::Tensor,
           std::vector<torch::Tensor>,
           std::vector<torch::Tensor>,
           torch::Tensor>
GaussianRenderer::render(
    const std::vector<std::shared_ptr<GaussianModel>>& models,
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
  /* Render the scene.

     Background tensor (bg_color) must be on GPU!
   */

  // auto timer_render = ProfilingUtils::Timer("render");

  // auto timer_initialization = ProfilingUtils::Timer("initialization");

  // int active_sh_degree = models[0]->active_sh_degree_;

  int active_sh_degree = 0;
  for (size_t i = 0; i < models.size(); i++) {
    const auto& pc = models[i];
    if (pc) {  // Safety check
      if (pc->getXYZ().sizes()[0] == 0) {
        throw std::runtime_error("Empty model");
      }
      active_sh_degree = std::max(active_sh_degree, pc->active_sh_degree_);
    }
  }

  // torch::Tensor camera_center =
  // world_view_transform.detach().inverse().index(
  //     {3, torch::indexing::Slice(0, 3)});
  torch::Tensor camera_center = viewpoint_camera->getCenter();

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

  // timer_initialization.stop();

  // auto timer_loop = ProfilingUtils::Timer("loop");

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

    /* If precomputed 3d covariance is provided, use it. If not, then it will
    be computed from scaling / rotation by the rasterizer.
  */
    torch::Tensor scales, rotations, cov3D_precomp;
    if (pipe.compute_cov3D_) {
      cov3D_precomp = pc->getCovarianceActivation();
    } else {
      scales = pc->getScalingActivation();
      rotations = pc->getRotationActivation();
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is
    desired
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
            (pc->getXYZ() -
             camera_center.repeat({pc->getFeatures().size(0), 1}));
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

  // timer_loop.stop();

  // auto timer_concat = ProfilingUtils::Timer("concat");

  // Debug statements before concatenation
  // std::cout << "===== Concatenation Debug Info =====" << std::endl;

  // // Debug means3D dimensions
  // std::cout << "means3D_vec sizes: " << means3D_vec.size() << std::endl;
  // for (size_t i = 0; i < means3D_vec.size(); i++) {
  //   std::cout << "  Model " << i << " means3D dims: [";
  //   for (size_t d = 0; d < means3D_vec[i].dim(); d++) {
  //     std::cout << means3D_vec[i].size(d);
  //     if (d < means3D_vec[i].dim() - 1) std::cout << ", ";
  //   }
  //   std::cout << "]" << std::endl;
  // }

  // // Debug means2D dimensions
  // std::cout << "means2D_vec sizes: " << means2D_vec.size() << std::endl;
  // for (size_t i = 0; i < means2D_vec.size(); i++) {
  //   std::cout << "  Model " << i << " means2D dims: [";
  //   for (size_t d = 0; d < means2D_vec[i].dim(); d++) {
  //     std::cout << means2D_vec[i].size(d);
  //     if (d < means2D_vec[i].dim() - 1) std::cout << ", ";
  //   }
  //   std::cout << "]" << std::endl;
  // }

  // // Debug opacity dimensions
  // std::cout << "opacity_vec sizes: " << opacity_vec.size() << std::endl;
  // for (size_t i = 0; i < opacity_vec.size(); i++) {
  //   std::cout << "  Model " << i << " opacity dims: [";
  //   for (size_t d = 0; d < opacity_vec[i].dim(); d++) {
  //     std::cout << opacity_vec[i].size(d);
  //     if (d < opacity_vec[i].dim() - 1) std::cout << ", ";
  //   }
  //   std::cout << "]" << std::endl;
  // }

  // // Debug dc dimensions
  // if (!dc_vec.empty()) {
  //   std::cout << "dc_vec sizes: " << dc_vec.size() << std::endl;
  //   for (size_t i = 0; i < dc_vec.size(); i++) {
  //     std::cout << "  Model " << i << " dc dims: [";
  //     for (size_t d = 0; d < dc_vec[i].dim(); d++) {
  //       std::cout << dc_vec[i].size(d);
  //       if (d < dc_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug shs dimensions
  // if (!shs_vec.empty()) {
  //   std::cout << "shs_vec sizes: " << shs_vec.size() << std::endl;
  //   for (size_t i = 0; i < shs_vec.size(); i++) {
  //     std::cout << "  Model " << i << " shs dims: [";
  //     for (size_t d = 0; d < shs_vec[i].dim(); d++) {
  //       std::cout << shs_vec[i].size(d);
  //       if (d < shs_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug colors_precomp dimensions
  // if (!colors_precomp_vec.empty()) {
  //   std::cout << "colors_precomp_vec sizes: " << colors_precomp_vec.size()
  //             << std::endl;
  //   for (size_t i = 0; i < colors_precomp_vec.size(); i++) {
  //     std::cout << "  Model " << i << " colors_precomp dims: [";
  //     for (size_t d = 0; d < colors_precomp_vec[i].dim(); d++) {
  //       std::cout << colors_precomp_vec[i].size(d);
  //       if (d < colors_precomp_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug scales dimensions
  // if (!scales_vec.empty()) {
  //   std::cout << "scales_vec sizes: " << scales_vec.size() << std::endl;
  //   for (size_t i = 0; i < scales_vec.size(); i++) {
  //     std::cout << "  Model " << i << " scales dims: [";
  //     for (size_t d = 0; d < scales_vec[i].dim(); d++) {
  //       std::cout << scales_vec[i].size(d);
  //       if (d < scales_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug rotations dimensions
  // if (!rotations_vec.empty()) {
  //   std::cout << "rotations_vec sizes: " << rotations_vec.size() <<
  //   std::endl; for (size_t i = 0; i < rotations_vec.size(); i++) {
  //     std::cout << "  Model " << i << " rotations dims: [";
  //     for (size_t d = 0; d < rotations_vec[i].dim(); d++) {
  //       std::cout << rotations_vec[i].size(d);
  //       if (d < rotations_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug cov3D_precomp dimensions
  // if (!cov3D_precomp_vec.empty()) {
  //   std::cout << "cov3D_precomp_vec sizes: " << cov3D_precomp_vec.size()
  //             << std::endl;
  //   for (size_t i = 0; i < cov3D_precomp_vec.size(); i++) {
  //     std::cout << "  Model " << i << " cov3D_precomp dims: [";
  //     for (size_t d = 0; d < cov3D_precomp_vec[i].dim(); d++) {
  //       std::cout << cov3D_precomp_vec[i].size(d);
  //       if (d < cov3D_precomp_vec[i].dim() - 1) std::cout << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // }

  // // Debug screenspace_points dimensions
  // std::cout << "screenspace_points_vec sizes: " <<
  // screenspace_points_vec.size()
  //           << std::endl;
  // for (size_t i = 0; i < screenspace_points_vec.size(); i++) {
  //   std::cout << "  Model " << i << " screenspace_points dims: [";
  //   for (size_t d = 0; d < screenspace_points_vec[i].dim(); d++) {
  //     std::cout << screenspace_points_vec[i].size(d);
  //     if (d < screenspace_points_vec[i].dim() - 1) std::cout << ", ";
  //   }
  //   std::cout << "]" << std::endl;
  // }

  // std::cout << "===== End Debug Info =====" << std::endl;

  // Check dimensions of all tensor vectors before concatenation
  assertTensorDims(means3D_vec, "means3D");
  assertTensorDims(means2D_vec, "means2D");
  assertTensorDims(opacity_vec, "opacity");
  if (!dc_vec.empty()) assertTensorDims(dc_vec, "dc");
  if (!shs_vec.empty()) assertTensorDims(shs_vec, "shs");
  if (!colors_precomp_vec.empty())
    assertTensorDims(colors_precomp_vec, "colors_precomp");
  if (!scales_vec.empty()) assertTensorDims(scales_vec, "scales");
  if (!rotations_vec.empty()) assertTensorDims(rotations_vec, "rotations");
  if (!cov3D_precomp_vec.empty())
    assertTensorDims(cov3D_precomp_vec, "cov3D_precomp");
  assertTensorDims(screenspace_points_vec, "screenspace_points");

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

  // timer_concat.stop();

  // auto timer_raster = ProfilingUtils::Timer("raster");

  // Set up rasterization configuration
  float tanfovx = std::tan(FoVx * 0.5f);
  float tanfovy = std::tan(FoVy * 0.5f);

  GaussianRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      projection_matrix, active_sh_degree, camera_center, false, false);

  // std::cout << " Image height: " << image_height
  //           << "Image width: " << image_width << "Tanfovx: " << tanfovx
  //           << "Tanfovy: " << tanfovy << "BG color: " << bg_color
  //           << "Scaling_modifier: " << scaling_modifier
  //           << "Projection matrix: " << projection_matrix
  //           << "Active SH degree: " << active_sh_degree
  //           << "Camera center: " << camera_center << std::endl;

  GaussianRasterizer rasterizer(raster_settings);

  // std::cout << "View matrix: " << world_view_transform << std::endl;

  // Rasterize visible Gaussians to image, obtain their radii (on screen).
  auto rasterizer_result = rasterizer.forward(
      means3D, means2D, opacity, dc, shs, colors_precomp, scales, rotations,
      cov3D_precomp, world_view_transform);

  // timer_raster.stop();
  auto rendered_image = std::get<0>(rasterizer_result);

  auto inverse_depth = std::get<1>(rasterizer_result);

  // std::cout << "Inverse depth min: " << inverse_depth.min().item<float>()
  //           << ", max: " << inverse_depth.max().item<float>() << std::endl;
  torch::Tensor rendered_depth = 1.0f / torch::clamp_min(inverse_depth, 1e-6f);
  // std::cout << "Rendered depth min: " << rendered_depth.min().item<float>()
  //           << ", max: " << rendered_depth.max().item<float>() << std::endl;

  auto mainGaussID = std::get<2>(rasterizer_result);

  auto radii = std::get<3>(rasterizer_result);

  rendered_image = viewpoint_camera->applyExposureTransform(rendered_image);

  // auto timer_final_loop = ProfilingUtils::Timer("final_loop");
  // Split the radii tensor into separate tensors per model
  std::vector<torch::Tensor> radii_vec;
  int offset = 0;
  for (int model_idx = 0; model_idx < num_models; model_idx++) {
    int size = model_sizes[model_idx];
    radii_vec.push_back(radii.slice(0, offset, offset + size));
    offset += size;
  }

  // timer_final_loop.stop();

  /* Those Gaussians that were frustum culled or had a radius of 0 were not
     visible. They will be excluded from value updates used in the splitting
     criteria.
   */

  // timer_render.stop();
  return std::make_tuple(rendered_depth, /*depth*/ rendered_image, /*render*/
                         screenspace_points_vec, /*viewspace_points*/
                         radii_vec /*radii*/, mainGaussID);
}
