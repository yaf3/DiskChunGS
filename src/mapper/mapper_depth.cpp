/**
 * This file is part of DiskChunGS, incorporating code from multiple sources.
 *
 * Original Copyright (C) 2025, Inria (On-The-Fly-NVS)
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This file incorporates code from:
 * - On-The-Fly-NVS (Inria, non-commercial research license)
 * - CaRtGS/Photo-SLAM (GPL v3)
 *
 * This software is licensed under GPL v3, with the following restrictions:
 * Portions derived from Inria-licensed works (3DGS, On-The-Fly-NVS) are
 * subject to non-commercial use restrictions. Commercial use requires
 * separate licensing from Inria.
 *
 * For non-commercial inquiries: george.drettakis@inria.fr
 * For commercial licensing: stip-sophia.transfert@inria.fr
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version, subject to the non-commercial restrictions
 * above.
 *
 * See <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <filesystem>
#include <iostream>

#include "gaussian_mapper.h"
#include "geometry/operate_points.h"
#include "utils/tensor_utils.h"

torch::Tensor GaussianMapper::computeLoGProbability(
    const torch::Tensor& image) {
  // Create Laplacian kernel for edge detection
  torch::Tensor laplacian_kernel = torch::tensor(
      {{{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));

  // Replicate kernel for each input channel
  laplacian_kernel = laplacian_kernel.repeat({1, image.size(0), 1, 1});

  // Apply Laplacian convolution with same padding
  torch::Tensor laplacian = torch::nn::functional::conv2d(
      image.unsqueeze(0),  // Add batch dimension: [1, C, H, W]
      laplacian_kernel,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute L1 norm across channels to get edge magnitude
  torch::Tensor laplacian_norm =
      torch::linalg_vector_norm(laplacian, 1, /*dim=*/1, /*keepdim=*/true);

  // Zero out image borders to avoid edge artifacts
  int H = laplacian_norm.size(-2);
  int W = laplacian_norm.size(-1);

  using namespace torch::indexing;
  laplacian_norm.index_put_({Ellipsis, 0, Slice()}, 0);      // Top row
  laplacian_norm.index_put_({Ellipsis, H - 1, Slice()}, 0);  // Bottom row
  laplacian_norm.index_put_({Ellipsis, Slice(), 0}, 0);      // Left column
  laplacian_norm.index_put_({Ellipsis, Slice(), W - 1}, 0);  // Right column

  // Smooth with disc kernel
  int pad_h = disc_kernel_.size(2) / 2;
  int pad_w = disc_kernel_.size(3) / 2;

  torch::Tensor result = torch::nn::functional::conv2d(
      laplacian_norm, disc_kernel_,
      torch::nn::functional::Conv2dFuncOptions().padding({pad_h, pad_w}));

  // Remove batch and channel dimensions, clamp to valid probability range
  result = result[0][0];
  return torch::clamp(result, 0.0f, 1.0f);
}

void GaussianMapper::initializeLaplacianOfGaussianKernel() {
  constexpr int radius = LOG_KERNEL_RADIUS;
  constexpr int kernel_size = 2 * radius + 1;

  // Create coordinate grids centered at origin
  torch::Tensor coords = torch::arange(
      -radius, radius + 1, torch::TensorOptions().dtype(torch::kFloat32));

  // Generate 2D coordinate grids
  auto meshgrid = torch::meshgrid({coords, coords}, "ij");
  torch::Tensor X = meshgrid[1];  // x coordinates
  torch::Tensor Y = meshgrid[0];  // y coordinates

  // Create disc mask: include pixels within radius + 0.5
  torch::Tensor distances = torch::sqrt(X * X + Y * Y);
  torch::Tensor disc_mask = distances <= (radius + 0.5f);

  // Initialize kernel and apply disc mask
  disc_kernel_ = torch::zeros({1, 1, kernel_size, kernel_size},
                              torch::TensorOptions().dtype(torch::kFloat32));
  disc_kernel_[0][0] = disc_mask.to(torch::kFloat32);

  // Normalize kernel to sum to 1
  disc_kernel_ = disc_kernel_ / disc_kernel_.sum();
  disc_kernel_ = disc_kernel_.to(device_type_);
}

void GaussianMapper::initializeStereoDepthEstimator() {
  // Construct model path for configured resolution
  std::string onnx_path =
      std::string(DEPTH_MODEL_BASE_DIR) +
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(STEREO_MODEL_HEIGHT) + "x" +
      std::to_string(STEREO_MODEL_WIDTH) + ".onnx";

  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(onnx_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  std::string onnx_path =
      std::string(DEPTH_MODEL_BASE_DIR) + "depth_anything_v2_vitl.onnx";

  this->monocular_depth_estimator_ = std::make_shared<MonoDepth>(onnx_path);
}

torch::Tensor GaussianMapper::sampleConf(const torch::Tensor& mono_depth_conf,
                                         const torch::Tensor& uv,
                                         int width,
                                         int height) {
  // Reshape UV coordinates for grid_sample: [1, 1, N, 2]
  torch::Tensor uv_reshaped = uv.view({1, 1, -1, 2});

  // Normalize pixel coordinates from [0, width/height] to [-1, 1]
  torch::Tensor normalized_uv = uv_reshaped.clone();
  normalized_uv.select(-1, 0) =
      (normalized_uv.select(-1, 0) / (width - 1)) * 2.0f - 1.0f;  // x
  normalized_uv.select(-1, 1) =
      (normalized_uv.select(-1, 1) / (height - 1)) * 2.0f - 1.0f;  // y

  // Sample using bilinear interpolation
  torch::Tensor sampled = torch::nn::functional::grid_sample(
      mono_depth_conf, normalized_uv,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .padding_mode(torch::kZeros)
          .align_corners(true));

  // Remove batch and channel dimensions, return [N]
  return sampled[0][0][0];
}
