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

#pragma once

#include <torch/all.h>

#include <tuple>
#include <utility>
#include <vector>

#include "cuda_rasterizer/rasterize_points.h"
#include "model/triangle_model.h"

/**
 * @brief Configuration parameters for Triangle rasterization.
 *
 * Encapsulates all settings needed to rasterize 3D Triangles onto a 2D image
 * plane, including camera intrinsics, projection matrices, and rendering
 * options.
 */
struct TriangleRasterizationSettings {
  TriangleRasterizationSettings(int image_height,
                                int image_width,
                                float tanfovx,
                                float tanfovy,
                                torch::Tensor& bg,
                                float scale_modifier,
                                torch::Tensor& projmatrix,
                                int sh_degree,
                                torch::Tensor& campos,
                                bool prefiltered,
                                bool debug)
      : image_height_(image_height),
        image_width_(image_width),
        tanfovx_(tanfovx),
        tanfovy_(tanfovy),
        bg_(bg),
        scale_modifier_(scale_modifier),
        projmatrix_(projmatrix),
        sh_degree_(sh_degree),
        campos_(campos),
        prefiltered_(prefiltered),
        debug_(debug) {}

  int image_height_;
  int image_width_;
  float tanfovx_;   ///< Tangent of horizontal field of view
  float tanfovy_;   ///< Tangent of vertical field of view
  torch::Tensor bg_;
  float scale_modifier_;
  torch::Tensor projmatrix_;
  int sh_degree_;   ///< Spherical harmonics degree
  torch::Tensor campos_;
  bool prefiltered_;
  bool debug_;
};

/**
 * @brief PyTorch autograd function for differentiable Triangle rasterization.
 *
 * Implements forward and backward passes for rasterizing 3D Triangles,
 * enabling gradient-based optimization of Triangle parameters.
 */
class TriangleRasterizerFunction
    : public torch::autograd::Function<TriangleRasterizerFunction> {
 public:
  /**
   * @brief Rasterizes 3D Triangles to produce a rendered image.
   * @return Tensor list containing: [color, invdepth, mainGaussID, radii]
   */
  static torch::autograd::tensor_list forward(
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
      TriangleRasterizationSettings raster_settings);

  /**
   * @brief Computes gradients for all rasterization inputs.
   */
  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

/**
 * @brief Convenience wrapper to invoke TriangleRasterizerFunction::apply().
 */
inline torch::autograd::tensor_list rasterizeTriangles(
    torch::Tensor& means3D,
    torch::Tensor& means2D,
    torch::Tensor& dc,
    torch::Tensor& sh,
    torch::Tensor& colors_precomp,
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& cov3Ds_precomp,
    torch::Tensor& viewmatrix,
    TriangleRasterizationSettings& raster_settings) {
  return TriangleRasterizerFunction::apply(
      means3D, means2D, dc, sh, colors_precomp, opacities, scales, rotations,
      cov3Ds_precomp, viewmatrix, raster_settings);
}

/**
 * @brief PyTorch module wrapper for Triangle rasterization.
 *
 * Provides a module-based interface for rasterizing 3D Triangles, handling
 * optional tensor initialization and delegating to TriangleRasterizerFunction.
 */
class TriangleRasterizer : public torch::nn::Module {
 public:
  explicit TriangleRasterizer(TriangleRasterizationSettings& raster_settings)
      : raster_settings_(raster_settings) {}

  /**
   * @brief Identifies which Triangles are visible from the current viewpoint.
   */
  torch::Tensor markVisibleTriangles(torch::Tensor& positions,
                                     torch::Tensor& viewmatrix) {
    return markVisible(positions, viewmatrix, raster_settings_.projmatrix_);
  }

  /**
   * @brief Renders 3D Triangles to produce color, depth, and auxiliary outputs.
   * @return Tuple of [color, invdepth, mainGaussID, radii, scaling, rend_normal]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor>
  forward(torch::Tensor means3D,
          torch::Tensor means2D,
          torch::Tensor opacities,
          torch::Tensor dc,
          torch::Tensor shs,
          torch::Tensor colors_precomp,
          torch::Tensor scales,
          torch::Tensor rotations,
          torch::Tensor cov3D_precomp,
          torch::Tensor viewmatrix);

  TriangleRasterizationSettings raster_settings_;
};

// Note: SparseTriangleAdam is defined here for convenience as it shares
// dependencies with the rasterizer components.

/**
 * @brief Adam optimizer variant for sparse Triangle parameter updates.
 */
class SparseTriangleAdam : public torch::optim::Adam {
 public:
  explicit SparseTriangleAdam(std::vector<torch::Tensor> parameters,
                              const torch::optim::AdamOptions& options)
      : torch::optim::Adam(parameters, options) {}

  // void step(torch::Tensor& visibility, const uint32_t N);
};