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
    torch::Tensor vertices,
    torch::Tensor triangle_indices,
    torch::Tensor vertex_weights,
    float sigma,
    torch::Tensor dc,
    torch::Tensor sh,
    torch::Tensor colors_precomp,
    torch::Tensor viewmatrix,
    TriangleRasterizationSettings raster_settings) {
  const int T = static_cast<int>(triangle_indices.size(0));
  const int V = static_cast<int>(vertices.size(0));
  auto float_opts = vertices.options().dtype(torch::kFloat32);

  auto scaling_buf = torch::zeros({T, 1}, float_opts);

  torch::Tensor combined_sh;
  bool dc_valid = dc.defined() && dc.numel() > 0;
  bool sh_valid = sh.defined() && sh.numel() > 0;
  if (dc_valid && sh_valid) {
    combined_sh = torch::cat({dc, sh}, /*dim=*/1);
  } else if (dc_valid) {
    combined_sh = dc;
  } else if (sh_valid) {
    combined_sh = sh;
  } else {
    combined_sh = torch::zeros({V, 0, 3}, float_opts);
  }

  auto [rendered, out_color, out_others, radii, was_rendered, geomBuffer,
        binningBuffer, imgBuffer, scaling_ret, max_blending] =
      RasterizeTrianglesCUDA(
          raster_settings.bg_,
          vertices,
          triangle_indices,
          vertex_weights.flatten(),
          sigma,
          colors_precomp,
          scaling_buf,
          viewmatrix,
          raster_settings.projmatrix_,
          raster_settings.tanfovx_,
          raster_settings.tanfovy_,
          raster_settings.image_height_,
          raster_settings.image_width_,
          combined_sh,
          raster_settings.sh_degree_,
          raster_settings.campos_,
          raster_settings.prefiltered_,
          raster_settings.debug_);

  ctx->saved_data["num_rendered"] = rendered;
  ctx->saved_data["sigma"] = static_cast<double>(sigma);
  ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
  ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
  ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
  ctx->saved_data["debug"] = raster_settings.debug_;
  ctx->saved_data["dc_valid"] = dc_valid;
  ctx->saved_data["sh_valid"] = sh_valid;
  ctx->saved_data["dc_bands"] = dc_valid ? static_cast<int>(dc.size(1)) : 0;

  ctx->save_for_backward(
      {vertices,              // [0]
       triangle_indices,      // [1]
       vertex_weights,        // [2]
       colors_precomp,        // [3]
       radii,                 // [4]
       combined_sh,           // [5]
       viewmatrix,            // [6]
       raster_settings.bg_,   // [7]
       raster_settings.projmatrix_,  // [8]
       raster_settings.campos_,      // [9]
       geomBuffer,            // [10]
       binningBuffer,         // [11]
       imgBuffer});           // [12]

  auto invdepth = out_others.slice(/*dim=*/0, /*start=*/0, /*end=*/1);
  auto rend_normal = out_others.slice(/*dim=*/0, /*start=*/2, /*end=*/5);

  auto mainTriangleID = torch::full(
      {1, raster_settings.image_height_, raster_settings.image_width_},
      -1,
      vertices.options().dtype(torch::kLong));

  return {out_color, invdepth, mainTriangleID, radii, scaling_ret, rend_normal};
}

torch::autograd::tensor_list TriangleRasterizerFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto num_rendered = ctx->saved_data["num_rendered"].toInt();
  auto sigma = static_cast<float>(ctx->saved_data["sigma"].toDouble());
  auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
  auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
  auto sh_degree = ctx->saved_data["sh_degree"].toInt();
  auto debug = ctx->saved_data["debug"].toBool();
  auto dc_valid = ctx->saved_data["dc_valid"].toBool();
  auto sh_valid = ctx->saved_data["sh_valid"].toBool();
  auto dc_bands = ctx->saved_data["dc_bands"].toInt();

  auto saved = ctx->get_saved_variables();
  auto vertices = saved[0];
  auto triangle_indices = saved[1];
  auto vertex_weights = saved[2];
  auto colors_precomp = saved[3];
  auto radii = saved[4];
  auto combined_sh = saved[5];
  auto viewmatrix = saved[6];
  auto bg = saved[7];
  auto projmatrix = saved[8];
  auto campos = saved[9];
  auto geomBuffer = saved[10];
  auto binningBuffer = saved[11];
  auto imgBuffer = saved[12];

  auto grad_out_color = grad_outputs[0];

  auto dL_dout_others =
      torch::zeros({7, grad_out_color.size(1), grad_out_color.size(2)},
                   grad_out_color.options());
  if (grad_outputs[1].defined() && grad_outputs[1].numel() > 0) {
    dL_dout_others.slice(/*dim=*/0, /*start=*/0, /*end=*/1) = grad_outputs[1];
  }
  if (grad_outputs[5].defined() && grad_outputs[5].numel() > 0) {
    dL_dout_others.slice(/*dim=*/0, /*start=*/2, /*end=*/5) = grad_outputs[5];
  }

  auto [dL_dvertices, dL_dvertex_weight, dL_dcolors, dL_dsh] =
      RasterizeTrianglesBackwardCUDA(
          bg,
          vertices,
          triangle_indices,
          vertex_weights.flatten(),
          sigma,
          radii,
          colors_precomp,
          viewmatrix,
          projmatrix,
          tanfovx,
          tanfovy,
          grad_out_color,
          dL_dout_others,
          combined_sh,
          sh_degree,
          campos,
          geomBuffer,
          num_rendered,
          binningBuffer,
          imgBuffer,
          debug);

  torch::Tensor grad_dc, grad_sh;
  bool sh_grad_valid = dL_dsh.defined() && dL_dsh.numel() > 0;

  if (dc_valid && sh_valid && sh_grad_valid) {
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

  torch::Tensor grad_vertex_weights;
  if (dL_dvertex_weight.defined() && dL_dvertex_weight.numel() > 0) {
    grad_vertex_weights = dL_dvertex_weight.reshape(vertex_weights.sizes());
  } else {
    grad_vertex_weights = torch::Tensor();
  }

  return {
      dL_dvertices,        // [0] grad for vertices
      torch::Tensor(),     // [1] grad for triangle_indices (int, no grad)
      grad_vertex_weights, // [2] grad for vertex_weights
      torch::Tensor(),     // [3] grad for sigma (float scalar, no grad)
      grad_dc,             // [4] grad for dc
      grad_sh,             // [5] grad for sh
      dL_dcolors,          // [6] grad for colors_precomp
      torch::Tensor(),     // [7] grad for viewmatrix
      torch::Tensor()      // [8] grad for raster_settings
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TriangleRasterizer::forward(torch::Tensor vertices,
                            torch::Tensor triangle_indices,
                            torch::Tensor vertex_weights,
                            float sigma,
                            torch::Tensor dc,
                            torch::Tensor shs,
                            torch::Tensor colors_precomp,
                            torch::Tensor viewmatrix) {
  auto raster_settings = this->raster_settings_;

  torch::TensorOptions options;
  if (!shs.defined()) shs = torch::tensor({}, options.device(torch::kCUDA));
  if (!colors_precomp.defined())
    colors_precomp = torch::tensor({}, options.device(torch::kCUDA));

  auto result = rasterizeTriangles(vertices, triangle_indices, vertex_weights,
                                   sigma, dc, shs, colors_precomp,
                                   viewmatrix, raster_settings);

  return std::make_tuple(result[0] /*color*/, result[1] /*invdepth*/,
                         result[2] /*mainTriangleID*/, result[3] /*radii*/,
                         result[4] /*scaling*/, result[5] /*rend_normal*/);
}
