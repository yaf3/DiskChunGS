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

#include "rendering/triangle_rasterizer.h"

torch::autograd::tensor_list TriangleRasterizerFunction::forward(
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
    TriangleRasterizationSettings raster_settings) {
  // means3D is now triangles_points [P, 3, 3] (3 vertices per triangle)
  // scales is now sigma [P, 1] (already activated by the renderer)
  const int P = static_cast<int>(means3D.size(0));
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  auto int_opts = means3D.options().dtype(torch::kInt32);

  // Flatten 3 vertices per triangle: [P, 3, 3] → [P*3, 3]
  auto triangles_points = means3D.reshape({P * 3, 3}).contiguous();

  // sigma passed directly from the renderer (already activated)
  torch::Tensor sigma;
  if (scales.defined() && scales.numel() > 0) {
    sigma = scales.reshape({P, 1}).to(float_opts.dtype());  // [P, 1]
  } else {
    sigma = torch::ones({P, 1}, float_opts);
  }

  // 3 vertices per triangle
  auto num_points_per_triangle = torch::full({P}, 3, int_opts);
  auto cumsum_of_points_per_triangle = torch::arange(P, int_opts) * 3;

  // Output buffers filled by the CUDA preprocess kernel
  auto scaling_buf = torch::zeros({P, 1}, float_opts);
  auto density_factor_buf = torch::zeros({P, 1}, float_opts);

  // Combine dc [P,1,3] and sh [P,K,3] into a single SH tensor for the CUDA
  // API.  When colors_precomp is provided the CUDA kernel ignores SH entirely.
  torch::Tensor combined_sh;
  bool dc_valid = dc.defined() && dc.numel() > 0;
  bool sh_valid = sh.defined() && sh.numel() > 0;
  if (dc_valid && sh_valid) {
    combined_sh = torch::cat({dc, sh}, /*dim=*/1);  // [P, 1+K, 3]
  } else if (dc_valid) {
    combined_sh = dc;
  } else if (sh_valid) {
    combined_sh = sh;
  } else {
    combined_sh =
        torch::zeros({P, 0, 3}, float_opts);  // empty – precomp path
  }

  // --- Invoke the triangle CUDA rasterizer ---
  auto [rendered, out_color, out_others, radii, geomBuffer, binningBuffer,
        imgBuffer, scaling_ret, density_ret, max_blending] =
      RasterizeTrianglesCUDA(
          raster_settings.bg_,
          triangles_points,
          sigma,
          num_points_per_triangle,
          cumsum_of_points_per_triangle,
          colors_precomp,
          opacities,
          scaling_buf,
          density_factor_buf,
          viewmatrix,
          raster_settings.projmatrix_,
          P,
          raster_settings.tanfovx_,
          raster_settings.tanfovy_,
          raster_settings.image_height_,
          raster_settings.image_width_,
          combined_sh,
          raster_settings.sh_degree_,
          raster_settings.campos_,
          raster_settings.prefiltered_,
          raster_settings.debug_);

  // --- Save state for backward ---
  ctx->saved_data["num_rendered"] = rendered;
  ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
  ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
  ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
  ctx->saved_data["debug"] = raster_settings.debug_;
  // Record how the combined_sh was constructed so the backward can split grads
  ctx->saved_data["dc_valid"] = dc_valid;
  ctx->saved_data["sh_valid"] = sh_valid;
  ctx->saved_data["dc_bands"] = dc_valid ? static_cast<int>(dc.size(1)) : 0;

  ctx->save_for_backward(
      {means3D,                      // [0]  triangles_points [P,3,3]
       sigma,                         // [1]  sigma [P,1]
       num_points_per_triangle,       // [2]
       cumsum_of_points_per_triangle, // [3]
       colors_precomp,                // [4]
       opacities,                     // [5]
       radii,                         // [6]
       combined_sh,                   // [7]
       viewmatrix,                    // [8]
       raster_settings.bg_,           // [9]
       raster_settings.projmatrix_,   // [10]
       raster_settings.campos_,       // [11]
       geomBuffer,                    // [12]
       binningBuffer,                 // [13]
       imgBuffer});                   // [14]

  // out_others is [7, H, W]: channels 0-2 normals, 3-5 offsets, 6 depth.
  // Expose channel 6 as the depth output; return zeros for per-pixel ID.
  auto invdepth = out_others.slice(/*dim=*/0, /*start=*/6, /*end=*/7);
  // mainTriangleID must be an integer type: it propagates through the mapper
  // as indices into the visible triangle list (e.g. visible_indices.index({...}))
  auto mainTriangleID = torch::full(
      {1, raster_settings.image_height_, raster_settings.image_width_},
      -1,
      means3D.options().dtype(torch::kLong));

  return {out_color, invdepth, mainTriangleID, radii};
}

torch::autograd::tensor_list TriangleRasterizerFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  // Restore scalars
  auto num_rendered = ctx->saved_data["num_rendered"].toInt();
  auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
  auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
  auto sh_degree = ctx->saved_data["sh_degree"].toInt();
  auto debug = ctx->saved_data["debug"].toBool();
  auto dc_valid = ctx->saved_data["dc_valid"].toBool();
  auto sh_valid = ctx->saved_data["sh_valid"].toBool();
  auto dc_bands = ctx->saved_data["dc_bands"].toInt();

  // Restore tensors
  auto saved = ctx->get_saved_variables();
  auto means3D = saved[0];   // triangles_points [P,3,3]
  auto sigma = saved[1];
  auto num_points_per_triangle = saved[2];
  auto cumsum_of_points_per_triangle = saved[3];
  auto colors_precomp = saved[4];
  auto opacities = saved[5];
  auto radii = saved[6];
  auto combined_sh = saved[7];
  auto viewmatrix = saved[8];
  auto bg = saved[9];
  auto projmatrix = saved[10];
  auto campos = saved[11];
  auto geomBuffer = saved[12];
  auto binningBuffer = saved[13];
  auto imgBuffer = saved[14];

  const int P = static_cast<int>(means3D.size(0));
  auto grad_out_color = grad_outputs[0];  // [3, H, W]
  // grad_outputs[1] is grad for the invdepth slice (channel 6 of out_others)
  // grad_outputs[2] is grad for mainTriangleID (zeros output, ignore)
  // grad_outputs[3] is grad for radii (int tensor, no grad)

  // Reconstruct full [7, H, W] gradient for out_others
  auto dL_dout_others =
      torch::zeros({7, grad_out_color.size(1), grad_out_color.size(2)},
                   grad_out_color.options());
  if (grad_outputs[1].defined() && grad_outputs[1].numel() > 0) {
    dL_dout_others.slice(/*dim=*/0, /*start=*/6, /*end=*/7) = grad_outputs[1];
  }

  // Flatten triangles_points [P,3,3] → [P*3,3] for the CUDA backward
  auto triangles_pts_flat = means3D.reshape({P * 3, 3}).contiguous();

  // --- Invoke the triangle CUDA backward ---
  auto [dL_dtriangle, dL_dsigma, dL_dcolors, dL_dopacity, dL_dsh,
        dL_dmeans2D] =
      RasterizeTrianglesBackwardCUDA(
          bg,
          triangles_pts_flat,           // triangles_points [P*3,3]
          sigma,
          num_points_per_triangle,
          cumsum_of_points_per_triangle,
          radii,
          colors_precomp,
          viewmatrix,
          projmatrix,
          P,
          tanfovx,
          tanfovy,
          grad_out_color,
          dL_dout_others,
          combined_sh,
          sh_degree,
          campos,
          geomBuffer,
          num_rendered,  // R
          binningBuffer,
          imgBuffer,
          debug);

  // --- Split combined SH gradient back to dc / sh gradients ---
  torch::Tensor grad_dc, grad_sh;
  bool sh_grad_valid = dL_dsh.defined() && dL_dsh.numel() > 0;

  if (dc_valid && sh_valid && sh_grad_valid) {
    // combined_sh = cat(dc, sh, dim=1) → split gradient along dim 1
    grad_dc = dL_dsh.slice(/*dim=*/1, /*start=*/0, /*end=*/dc_bands);
    grad_sh = dL_dsh.slice(/*dim=*/1, /*start=*/dc_bands);
  } else if (dc_valid && sh_grad_valid) {
    grad_dc = dL_dsh;
    grad_sh = torch::Tensor();
  } else if (sh_valid && sh_grad_valid) {
    grad_dc = torch::Tensor();
    grad_sh = dL_dsh;
  } else {
    grad_dc = torch::Tensor();
    grad_sh = torch::Tensor();
  }

  // dL_dtriangle is [P*3, 3] — reshape to [P, 3, 3] to match triangles_points_
  torch::Tensor grad_triangles;
  if (dL_dtriangle.defined() && dL_dtriangle.numel() > 0) {
    grad_triangles = dL_dtriangle.reshape({P, 3, 3});
  } else {
    grad_triangles = torch::Tensor();
  }

  // dL_dsigma is [P, 1] — returned directly (sigma already activated, passed as-is)
  torch::Tensor grad_sigma = (dL_dsigma.defined() && dL_dsigma.numel() > 0)
      ? dL_dsigma.reshape({P, 1})
      : torch::Tensor();

  // Return one gradient per forward input tensor (10) + 1 for raster_settings
  return {
      grad_triangles,  // [0] grad for means3D (= triangles_points [P,3,3])
      dL_dmeans2D,     // [1] grad for means2D
      grad_dc,         // [2] grad for dc
      grad_sh,         // [3] grad for sh (rest SH bands)
      dL_dcolors,      // [4] grad for colors_precomp
      dL_dopacity,     // [5] grad for opacities
      grad_sigma,      // [6] grad for scales (= sigma [P,1], directly)
      torch::Tensor(), // [7] grad for rotations (not used in triangle API)
      torch::Tensor(), // [8] grad for cov3Ds_precomp (not used)
      torch::Tensor(), // [9] grad for viewmatrix
      torch::Tensor()  // [10] grad for raster_settings (not a tensor)
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TriangleRasterizer::forward(torch::Tensor means3D,
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

  // Create empty tensors for undefined parameters
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

  auto result = rasterizeTriangles(means3D, means2D, dc, shs, colors_precomp,
                                   opacities, scales, rotations, cov3D_precomp,
                                   viewmatrix, raster_settings);

  return std::make_tuple(result[0] /*color*/, result[1] /*invdepth*/,
                         result[2] /*mainTriangleID*/, result[3] /*radii*/);
}
