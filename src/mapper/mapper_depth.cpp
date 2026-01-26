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
  // Step 1: Create Laplacian kernel
  torch::Tensor laplacian_kernel = torch::tensor(
      {{{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));

  // Step 2: Repeat kernel for each channel of input image
  // laplacian_kernel shape: [1, input_channels, 3, 3]
  laplacian_kernel = laplacian_kernel.repeat({1, image.size(0), 1, 1});

  // Step 3: Apply Laplacian convolution with "same" padding
  // For 3x3 kernel, "same" padding = 1 on all sides
  torch::Tensor laplacian = torch::nn::functional::conv2d(
      image.unsqueeze(0),  // Add batch dimension: [1, C, H, W]
      laplacian_kernel,
      torch::nn::functional::Conv2dFuncOptions().padding(
          1)  // "same" padding for 3x3 kernel
  );

  // Step 4: Compute L1 norm across channels (dim=1), keep dimension
  torch::Tensor laplacian_norm =
      torch::linalg_vector_norm(laplacian, 1, /*dim=*/1, /*keepdim=*/true);

  // Step 5: Zero out the borders
  // laplacian_norm shape: [1, 1, H, W]
  int H = laplacian_norm.size(-2);  // Second to last dimension
  int W = laplacian_norm.size(-1);  // Last dimension

  // Zero out borders
  using namespace torch::indexing;

  // Zero out top and bottom rows
  laplacian_norm.index_put_({Ellipsis, 0, Slice()}, 0);      // Top row
  laplacian_norm.index_put_({Ellipsis, H - 1, Slice()}, 0);  // Bottom row

  // Zero out left and right columns
  laplacian_norm.index_put_({Ellipsis, Slice(), 0}, 0);      // Left column
  laplacian_norm.index_put_({Ellipsis, Slice(), W - 1}, 0);  // Right column

  // Step 6: Convolve with disc kernel and clamp
  // For disc_kernel_ size calculation: if radius=3, kernel is 7x7, so
  // padding=3
  int pad_h = disc_kernel_.size(2) / 2;
  int pad_w = disc_kernel_.size(3) / 2;

  torch::Tensor result = torch::nn::functional::conv2d(
      laplacian_norm, disc_kernel_,
      torch::nn::functional::Conv2dFuncOptions().padding({pad_h, pad_w}));

  // Step 7: Extract result and clamp to [0, 1]
  result = result[0][0];  // Remove batch and channel dimensions -> [H, W]
  return torch::clamp(result, 0.0f, 1.0f);
}

void GaussianMapper::initializeLaplacianOfGaussianKernel() {
  int radius = 3;
  int kernel_size = 2 * radius + 1;

  // Create coordinate grids
  torch::Tensor y = torch::arange(
      -radius, radius + 1, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor x = y.clone();

  // Create 2D grids
  auto meshgrid = torch::meshgrid({x, y}, "ij");
  torch::Tensor X = meshgrid[1];  // x coordinates
  torch::Tensor Y = meshgrid[0];  // y coordinates

  // Create disc kernel: 1 where distance <= radius + 0.5, 0 elsewhere
  torch::Tensor distances = torch::sqrt(X * X + Y * Y);
  torch::Tensor disc_mask = distances <= (radius + 0.5);

  // Initialize kernel with zeros and set disc region to 1
  disc_kernel_ = torch::zeros({1, 1, kernel_size, kernel_size},
                              torch::TensorOptions().dtype(torch::kFloat32));
  disc_kernel_[0][0] = disc_mask.to(torch::kFloat32);

  // Normalize kernel (divide by sum)
  disc_kernel_ = disc_kernel_ / disc_kernel_.sum();
  disc_kernel_ = disc_kernel_.to(device_type_);
}

void GaussianMapper::initializeStereoDepthEstimator() {
  cv::Size model_resolution(
      1280, 384);  // Other resolutions would need to be downloaded separately

  // ONNX model path
  std::string onnx_path =
      "/workspace/repo/models/"
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(model_resolution.height) + "x" +
      std::to_string(model_resolution.width) + ".onnx";

  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(onnx_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  // ONNX model path
  std::string onnx_path =
      "/workspace/repo/models/"
      "depth_anything_v2_vitl.onnx";

  this->monocular_depth_estimator_ = std::make_shared<MonoDepth>(onnx_path);
}

torch::Tensor GaussianMapper::sampleConf(const torch::Tensor& mono_depth_conf,
                                         const torch::Tensor& uv,
                                         int width,
                                         int height) {
  // mono_depth_conf shape: [1, 1, H, W]
  // uv shape: [N, 2] where N is number of points
  // Returns: [N] confidence values

  // Reshape uv to [1, 1, N, 2] for grid_sample
  torch::Tensor uv_reshaped = uv.view({1, 1, -1, 2});

  // Convert UV coordinates to normalized coordinates [-1, 1]
  // grid_sample expects coordinates in [-1, 1] range
  torch::Tensor normalized_uv = uv_reshaped.clone();
  normalized_uv.select(-1, 0) =
      (normalized_uv.select(-1, 0) / (width - 1)) * 2.0f - 1.0f;  // x
  normalized_uv.select(-1, 1) =
      (normalized_uv.select(-1, 1) / (height - 1)) * 2.0f - 1.0f;  // y

  // Use grid_sample for bilinear interpolation
  torch::Tensor sampled = torch::nn::functional::grid_sample(
      mono_depth_conf, normalized_uv,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .padding_mode(torch::kZeros)
          .align_corners(true));

  // Return flattened result [N]
  return sampled[0][0][0];  // Remove batch and channel dimensions
}
