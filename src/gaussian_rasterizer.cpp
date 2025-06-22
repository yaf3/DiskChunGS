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

#include "include/gaussian_rasterizer.h"

// torch::Tensor GaussianRasterizer::markVisibleGaussians(
//     torch::Tensor& positions,
//     torch::Tensor& viewmatrix) {
//   // Mark visible points (based on frustum culling for camera) with a boolean
//   torch::NoGradGuard no_grad;
//   auto raster_settings = this->raster_settings_;
//   return markVisible(positions, viewmatrix, raster_settings.projmatrix_);
// }

torch::autograd::tensor_list GaussianRasterizerFunction::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor dc,
    torch::Tensor sh,
    torch::Tensor colors_precomp,
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3Ds_precomp,
    torch::Tensor viewmatrix,
    GaussianRasterizationSettings raster_settings) {
  // Invoke C++/CUDA rasterizer
  auto [num_rendered, num_buckets, color, invdepth, mainGaussID, radii,
        geomBuffer, binningBuffer, imgBuffer, sampleBuffer] =
      RasterizeGaussiansCUDA(
          raster_settings.bg_, means3D, colors_precomp, opacities, scales,
          rotations, raster_settings.scale_modifier_, cov3Ds_precomp,
          viewmatrix, raster_settings.projmatrix_, raster_settings.tanfovx_,
          raster_settings.tanfovy_, raster_settings.image_height_,
          raster_settings.image_width_, dc, sh, raster_settings.sh_degree_,
          raster_settings.campos_, raster_settings.prefiltered_,
          raster_settings.debug_);

  // Keep relevant tensors for backward - store raster_settings for later use
  ctx->saved_data["num_rendered"] = num_rendered;
  ctx->saved_data["num_buckets"] = num_buckets;
  ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
  ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
  ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
  ctx->saved_data["image_height"] = raster_settings.image_height_;
  ctx->saved_data["image_width"] = raster_settings.image_width_;
  ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
  ctx->saved_data["prefiltered"] = raster_settings.prefiltered_;
  ctx->saved_data["debug"] = raster_settings.debug_;

  // Save tensors in the SAME ORDER as Python version
  ctx->save_for_backward({colors_precomp, means3D, scales, rotations,
                          cov3Ds_precomp, radii, dc, sh, opacities, geomBuffer,
                          binningBuffer, imgBuffer, sampleBuffer, viewmatrix,
                          raster_settings.bg_, raster_settings.projmatrix_,
                          raster_settings.campos_});

  return {color, invdepth, mainGaussID, radii};
}

torch::autograd::tensor_list GaussianRasterizerFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  // Restore necessary values from context
  auto num_rendered = ctx->saved_data["num_rendered"].toInt();
  auto num_buckets = ctx->saved_data["num_buckets"].toInt();
  auto scale_modifier =
      static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
  auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
  auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
  auto image_height = ctx->saved_data["image_height"].toInt();
  auto image_width = ctx->saved_data["image_width"].toInt();
  auto sh_degree = ctx->saved_data["sh_degree"].toInt();
  auto prefiltered = ctx->saved_data["prefiltered"].toBool();
  auto debug = ctx->saved_data["debug"].toBool();

  auto saved = ctx->get_saved_variables();

  // Restore tensors in the SAME ORDER as Python version
  auto colors_precomp = saved[0];  // matches Python: colors_precomp
  auto means3D = saved[1];         // matches Python: means3D
  auto scales = saved[2];          // matches Python: scales
  auto rotations = saved[3];       // matches Python: rotations
  auto cov3Ds_precomp = saved[4];  // matches Python: cov3Ds_precomp
  auto radii = saved[5];           // matches Python: radii
  auto dc = saved[6];              // matches Python: dc
  auto sh = saved[7];              // matches Python: sh
  auto opacities = saved[8];       // matches Python: opacities
  auto geomBuffer = saved[9];      // matches Python: geomBuffer
  auto binningBuffer = saved[10];  // matches Python: binningBuffer
  auto imgBuffer = saved[11];      // matches Python: imgBuffer
  auto sampleBuffer = saved[12];   // matches Python: sampleBuffer
  auto viewmatrix = saved[13];     // matches Python: viewmatrix
  auto bg = saved[14];             // raster_settings.bg_
  auto projmatrix = saved[15];     // raster_settings.projmatrix_
  auto campos = saved[16];         // raster_settings.campos_

  // Compute gradients for relevant tensors by invoking backward method
  auto grad_out_color = grad_outputs[0];
  auto grad_out_invdepth = grad_outputs[1];

  auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
        grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations,
        grad_viewmatrix] =
      RasterizeGaussiansBackwardCUDA(
          bg, means3D, radii, colors_precomp, opacities, scales, rotations,
          scale_modifier, cov3Ds_precomp, viewmatrix, projmatrix, tanfovx,
          tanfovy, grad_out_color, dc, sh, grad_out_invdepth, sh_degree, campos,
          geomBuffer, num_rendered, binningBuffer, imgBuffer, num_buckets,
          sampleBuffer, debug);

  // Return gradients in the SAME ORDER as Python version
  return {
      grad_means3D,         // matches Python position 0
      grad_means2D,         // matches Python position 1
      grad_dc,              // matches Python position 2
      grad_sh,              // matches Python position 3
      grad_colors_precomp,  // matches Python position 4
      grad_opacities,       // matches Python position 5
      grad_scales,          // matches Python position 6
      grad_rotations,       // matches Python position 7
      grad_cov3Ds_precomp,  // matches Python position 8
      grad_viewmatrix,      // matches Python position 9
      torch::Tensor()       // matches Python position 10 (None)
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer::forward(torch::Tensor means3D,
                            torch::Tensor means2D,
                            torch::Tensor opacities,
                            torch::Tensor dc,
                            torch::Tensor shs,
                            torch::Tensor colors_precomp,
                            torch::Tensor scales,
                            torch::Tensor rotations,
                            torch::Tensor cov3D_precomp,
                            torch::Tensor viewmatrix) {
  auto raster_settings = this->raster_settings_;

  // Remove the validation checks to match Python version (which has them
  // commented out) The Python version doesn't validate these conditions, so C++
  // shouldn't either

  // Create empty tensors for undefined parameters to match Python behavior
  torch::TensorOptions options;
  if (!shs.defined()) shs = torch::tensor({}, options.device(torch::kCUDA));
  if (!colors_precomp.defined())
    colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
  if (!scales.defined())
    scales = torch::tensor({}, options.device(torch::kCUDA));
  if (!rotations.defined())
    rotations = torch::tensor({}, options.device(torch::kCUDA));
  if (!cov3D_precomp.defined())
    cov3D_precomp = torch::tensor({}, options.device(torch::kCUDA));

  auto result = rasterizeGaussians(means3D, means2D, dc, shs, colors_precomp,
                                   opacities, scales, rotations, cov3D_precomp,
                                   viewmatrix, raster_settings);

  return std::make_tuple(result[0] /*color*/, result[1] /*invdepth*/,
                         result[2] /*mainGaussID*/, result[3] /*radii*/);
}

// void SparseGaussianAdam::step(torch::Tensor& visibility, const uint32_t N) {
//   torch::NoGradGuard no_grad;
//   // std::cout << "SparseGaussianAdam::step() called with N=" << N <<
//   std::endl;

//   int group_idx = 0;
//   for (auto& group : this->param_groups()) {
//     // std::cout << "Processing group " << group_idx << std::endl;

//     auto options = static_cast<torch::optim::AdamOptions&>(group.options());
//     // auto lr = torch::tensor(options.lr());
//     auto eps = options.eps();
//     // std::cout << "  lr=" << lr << ", eps=" << eps << std::endl;

//     auto& param = group.params()[0];
//     // std::cout << "  param size: " << param.sizes()
//     //           << ", numel: " << param.numel() << std::endl;

//     if (!param.grad().defined()) {
//       // std::cout << "  grad not defined, skipping" << std::endl;
//       group_idx++;
//       continue;
//     }

//     // std::cout << "  About to access state..." << std::endl;
//     auto tensor_impl = param.unsafeGetTensorImpl();
//     // std::cout << "  Got tensor impl" << std::endl;

//     auto& optimizer_state = this->state();
//     // std::cout << "  Got optimizer state map" << std::endl;

//     auto state_iter = optimizer_state.find(tensor_impl);
//     if (state_iter == optimizer_state.end()) {
//       // std::cout << "  State not found, creating new state..." <<
//       std::endl;
//       // Initialize state for this parameter (normally done by base
//       Adam::step) auto new_state =
//       std::make_unique<torch::optim::AdamParamState>(); new_state->step(0);
//       new_state->exp_avg(torch::zeros_like(param));
//       new_state->exp_avg_sq(torch::zeros_like(param));
//       optimizer_state[tensor_impl] = std::move(new_state);
//       state_iter = optimizer_state.find(tensor_impl);
//     }
//     // std::cout << "  Found/created state in map" << std::endl;

//     auto& state =
//         static_cast<torch::optim::AdamParamState&>(*state_iter->second);
//     // std::cout << "  Cast state successfully" << std::endl;

//     if (!state.exp_avg().defined()) {
//       // std::cout << "  Initializing state..." << std::endl;
//       state.step(0);
//       state.exp_avg(
//           torch::zeros_like(param, {}, torch::MemoryFormat::Preserve));
//       state.exp_avg_sq(
//           torch::zeros_like(param, {}, torch::MemoryFormat::Preserve));
//       // std::cout << "  State initialized" << std::endl;
//     }

//     auto exp_avg = state.exp_avg();
//     auto exp_avg_sq = state.exp_avg_sq();
//     auto grad = param.grad();

//     torch::Tensor lr_tensor;
//     if (group_idx == 0) {
//       // Use per-primitive learning rates for positions
//       lr_tensor = position_lrs_;
//     } else {
//       // Use scalar learning rate for other parameters
//       float scalar_lr = group.options().get_lr();
//       lr_tensor = torch::tensor(scalar_lr,
//                                 torch::TensorOptions().device(param.device()));
//     }

//     const uint32_t M = param.numel() / N;
//     // std::cout << "  M=" << M << ", calling adamUpdate..." << std::endl;

//     adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility, lr_tensor,
//                std::get<0>(options.betas()), std::get<1>(options.betas()),
//                eps, N, M);
//     // std::cout << "  adamUpdate completed for group " << group_idx <<
//     // std::endl;
//     group_idx++;
//   }
//   // std::cout << "SparseGaussianAdam::step() completed" << std::endl;
// }
