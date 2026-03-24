/*
 * Copyright (C) 2025, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * This file is Derivative Works of On-The-Fly-NVS,
 * modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 *
 * For inquiries contact george.drettakis@inria.fr
 */

#include "depth/guided_mvs.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Forward declaration of CUDA kernel wrapper
extern "C" void launch_uvToDepth(const void* uvs,
                                 const void* refFeatMap,
                                 const void* otherFeatMaps,
                                 const void* Rts,
                                 const void* intrinsics_,
                                 const float* idepthMap,
                                 float* depth,
                                 float* idist,
                                 float range,
                                 int nPts,
                                 int featMapH,
                                 int featMapW,
                                 int depthMapH,
                                 int depthMapW,
                                 int H,
                                 int W,
                                 int num_depth_candidates);

// Constructor implementation
GuidedMVS::GuidedMVS(int num_prev_keyframes,
                     int num_depth_candidates,
                     float inverse_depth_range)
    : n_cams(num_prev_keyframes),
      num_depth_candidates(num_depth_candidates),
      idepth_range(inverse_depth_range) {}

// Implementation of guided MVS operator (see header for API documentation)
std::pair<torch::Tensor, torch::Tensor> GuidedMVS::operator()(
    const torch::Tensor& uv,
    const std::shared_ptr<TriangleKeyframe> refKeyframe,
    const std::vector<std::shared_ptr<TriangleKeyframe>>& keyframes) {
  // Input validation
  if (uv.ndimension() != 2 || uv.size(1) != 2) {
    AT_ERROR("uv must have dimensions (num_points, 2)");
  }

  // Ensure tensors are contiguous and on CUDA
  auto uv_cuda = uv.contiguous().cuda();

  // Compute relative poses: transform other keyframes to reference frame coordinates
  std::vector<torch::Tensor> other2ref_list;
  for (const auto& keyframe : keyframes) {
    auto rel_pose = torch::matmul(keyframe->getRT(),
                                  torch::linalg::inv(refKeyframe->getRT()));
    // Extract rotation and translation [3x4] from 4x4 transformation matrix
    other2ref_list.push_back(
        rel_pose.slice(0, 0, 3).slice(1, 0, 4));
  }
  auto other2ref = torch::stack(other2ref_list, 0).contiguous().cuda();

  // Gather feature maps from all keyframes
  auto refFeatMap = refKeyframe->feature_map_.contiguous().cuda();
  std::vector<torch::Tensor> featMaps_list;
  for (const auto& keyframe : keyframes) {
    featMaps_list.push_back(keyframe->feature_map_.cuda().contiguous());
  }
  auto featMaps = torch::stack(featMaps_list, 0);

  // Upscale feature maps to full image resolution for precise matching
  auto interpolate_options = torch::nn::functional::InterpolateFuncOptions()
                                 .size(std::vector<int64_t>{refKeyframe->image_height_,
                                                            refKeyframe->image_width_})
                                 .mode(torch::kBilinear)
                                 .align_corners(true);

  refFeatMap = torch::nn::functional::interpolate(
                   refFeatMap.unsqueeze(0),
                   interpolate_options)
                   .squeeze(0);

  featMaps = torch::nn::functional::interpolate(featMaps, interpolate_options);

  // Extract camera intrinsics: fx, cx, cy (fy assumed equal to fx)
  auto intrinsics = torch::tensor({refKeyframe->intr_[0], refKeyframe->intr_[2],
                                   refKeyframe->intr_[3]})
                        .contiguous()
                        .cuda();

  // Get monocular inverse depth prior from finest pyramid level
  auto mono_idepth = refKeyframe->gaus_pyramid_inv_depth_image_[0]
                         .unsqueeze(0)
                         .unsqueeze(0)
                         .contiguous()
                         .cuda();

  // Initialize output tensors (negative values indicate invalid/uncomputed)
  const int num_points = uv_cuda.size(0);
  auto cuda_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto depth = -torch::ones({num_points}, cuda_float_options);
  auto idist = -torch::ones({num_points}, cuda_float_options);

  if (num_points != 0) {
    // Launch CUDA kernel for depth estimation
    launch_uvToDepth(
        uv_cuda.data_ptr<float>(), refFeatMap.data_ptr<at::Half>(),
        featMaps.data_ptr<at::Half>(), other2ref.data_ptr<float>(),
        intrinsics.data_ptr<float>(), mono_idepth.data_ptr<float>(),
        depth.data_ptr<float>(), idist.data_ptr<float>(), idepth_range, num_points,
        static_cast<int>(refFeatMap.size(1)),
        static_cast<int>(refFeatMap.size(2)),
        static_cast<int>(mono_idepth.size(-2)),
        static_cast<int>(mono_idepth.size(-1)),
        static_cast<int>(refKeyframe->gaus_pyramid_original_image_[0].size(1)),
        static_cast<int>(refKeyframe->gaus_pyramid_original_image_[0].size(2)),
        num_depth_candidates);
  }

  // Valid mask identifies points where depth estimation succeeded (non-negative idist)
  auto valid_mask = idist >= 0;
  return std::make_pair(depth, valid_mask);
}