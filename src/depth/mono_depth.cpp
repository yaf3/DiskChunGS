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

#include "depth/mono_depth.h"

#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

MonoDepth::MonoDepth(const std::string& model_path) {
  initialize_model(model_path);
}

void MonoDepth::initialize_model(const std::string& user_model_path) {
  std::string base_filepath = user_model_path;
  size_t pos = base_filepath.rfind(".onnx");
  if (pos != std::string::npos) {
    base_filepath = base_filepath.substr(0, pos);
  }

  std::string engine_path = base_filepath + ".engine";

  std::string model_path;
  if (std::filesystem::exists(engine_path)) {
    model_path = engine_path;
  } else {
    model_path = user_model_path;
  }

  // Check if model file exists
  std::ifstream file(model_path);
  if (!file.good()) {
    throw std::runtime_error("Model file not found: " + model_path);
  }
  file.close();

  std::cout << "Loading TensorRT model: " << model_path << std::endl;

  try {
    // Initialize TensorRT DepthAnything
    depth_anything_ = std::make_unique<DepthAnything>();

    // Create logger for TensorRT
    static Logger logger;

    // Initialize the model
    depth_anything_->init(model_path, logger);

    std::cout << "TensorRT DepthAnything model initialized successfully"
              << std::endl;

    // Set input dimensions (these should match the DepthAnything model)
    input_height_ = 518;
    input_width_ = 518;

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize TensorRT model: " +
                             std::string(e.what()));
  }

  // Initialize Sobel kernels
  sobel_x_ = torch::tensor(
      {{{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  sobel_y_ = torch::tensor(
      {{{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

std::tuple<torch::Tensor, torch::Tensor> MonoDepth::estimate_depth(
    const cv::Mat& image,
    float focal_length) {
  if (!depth_anything_) {
    throw std::runtime_error("DepthAnything model not initialized");
  }

  img_height_ = image.rows;
  img_width_ = image.cols;

  // Convert image to the format expected by DepthAnything
  cv::Mat input_image;
  if (image.type() != CV_8UC3) {
    if (image.type() == CV_32FC3) {
      // Convert from float [0,1] to uint8 [0,255]
      image.convertTo(input_image, CV_8UC3, 255.0);
    } else {
      image.convertTo(input_image, CV_8UC3);
    }
  } else {
    input_image = image.clone();
  }

  auto start_inf = std::chrono::high_resolution_clock::now();

  // Run TensorRT inference
  cv::Mat raw_depth = depth_anything_->predict(input_image);

  auto end_inf = std::chrono::high_resolution_clock::now();

  // Convert to torch tensor for processing
  torch::Tensor depth =
      tensor_utils::cvMat2TorchTensor_Float32(raw_depth, torch::kCUDA);

  // Apply normalization: (depth - t) / s
  auto [t, s] = get_t_s(depth);
  depth = (depth - t) / s;

  if (depth.dim() == 2) {
    depth = depth.unsqueeze(0).unsqueeze(0);
  } else if (depth.dim() == 3) {
    depth = depth.unsqueeze(0);
  }

  // Compute gradients using Sobel filters
  torch::Tensor grad_x = torch::nn::functional::conv2d(
      depth, sobel_x_, torch::nn::functional::Conv2dFuncOptions().padding(1));

  torch::Tensor grad_y = torch::nn::functional::conv2d(
      depth, sobel_y_, torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute edge magnitude and confidence
  torch::Tensor edges = torch::cat({grad_x, grad_y}, 0);
  torch::Tensor edges_sq_norm = (edges.pow(2)).sum(0, true);

  float var = 0.2f;
  torch::Tensor confidence = torch::exp(-edges_sq_norm / var);

  auto inf_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_inf - start_inf);
  // std::cout << "TensorRT inference time: " << inf_time.count() << "ms"
  //           << std::endl;

  return std::make_tuple(depth, confidence);
}

/**
 * Get median and median absolute deviation for depth normalization
 */
std::tuple<torch::Tensor, torch::Tensor> MonoDepth::get_t_s(
    const torch::Tensor& depth) const {
  torch::Tensor t = depth.median();
  torch::Tensor s = (depth - t).abs().median();
  return std::make_tuple(t, s);
}

/**
 * Align samples by finding scale and offset
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MonoDepth::align_samples(const torch::Tensor& tri_idepth,
                         const torch::Tensor& mono_idepth) const {
  auto [t_tri, s_tri] = get_t_s(tri_idepth);
  auto [t_mono, s_mono] = get_t_s(mono_idepth);

  torch::Tensor scale = s_tri / s_mono;
  torch::Tensor offset = t_tri - t_mono * scale;
  torch::Tensor aligned = mono_idepth * scale + offset;

  return std::make_tuple(aligned, scale, offset);
}

torch::Tensor MonoDepth::align_depth(
    const torch::Tensor& mono_depth_map,  // Normalized depth from model
    const std::vector<float>& keypoint_pixels,
    const std::vector<float>& keypoint_depths,  // Metric depths in meters
    int width,
    int height) const {
  if (keypoint_pixels.empty() || keypoint_depths.empty() ||
      keypoint_pixels.size() != keypoint_depths.size() * 2) {
    std::cerr << "Warning: No valid keypoints for depth alignment" << std::endl;
    return mono_depth_map;
  }

  int num_keypoints = keypoint_depths.size();

  // Convert keypoint data to tensors
  torch::Tensor pixel_coords =
      torch::from_blob(const_cast<float*>(keypoint_pixels.data()),
                       {num_keypoints, 2},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(mono_depth_map.device());

  torch::Tensor metric_depths =
      torch::from_blob(const_cast<float*>(keypoint_depths.data()),
                       {num_keypoints},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(mono_depth_map.device());

  // Step 1: Sample normalized depth values at keypoint locations
  torch::Tensor mono_idepth =
      sample_depth_at_pixels(mono_depth_map, pixel_coords, width, height);

  // Step 2: Convert metric depths to inverse depths
  torch::Tensor tri_idepth = 1.0f / metric_depths;

  // Step 3: First alignment
  auto [mono_idepth_aligned, scale, offset] =
      align_samples(tri_idepth, mono_idepth);

  // Step 4: Outlier filtering
  torch::Tensor err = (mono_idepth_aligned - tri_idepth).abs();
  torch::Tensor err_median = err.median();
  torch::Tensor valid_mask = err < (5.0f * err_median);

  // Step 5: Re-align with filtered data
  if (valid_mask.sum().item<int>() >= 3) {
    torch::Tensor tri_idepth_valid = tri_idepth.masked_select(valid_mask);
    torch::Tensor mono_idepth_valid = mono_idepth.masked_select(valid_mask);

    auto [mono_idepth_aligned_final, scale_final, offset_final] =
        align_samples(tri_idepth_valid, mono_idepth_valid);

    // Step 6: Apply to entire depth map
    torch::Tensor mono_depth_map_aligned =
        mono_depth_map * scale_final + offset_final;

    return mono_depth_map_aligned;

  } else {
    std::cout << "Warning: Not enough valid keypoints after filtering"
              << std::endl;
    // Fall back to first alignment
    torch::Tensor mono_depth_map_aligned = mono_depth_map * scale + offset;
    return mono_depth_map_aligned;
  }
}

/**
 * Sample depth values at given pixel coordinates
 */
torch::Tensor MonoDepth::sample_depth_at_pixels(
    const torch::Tensor& depth_map,
    const torch::Tensor& pixel_coords,
    int width,
    int height) const {
  // Convert pixel coordinates to normalized coordinates [-1, 1]
  torch::Tensor normalized_coords = pixel_coords.clone().to(torch::kFloat32);
  normalized_coords.select(1, 0) =
      (normalized_coords.select(1, 0) / (width - 1)) * 2.0 - 1.0;
  normalized_coords.select(1, 1) =
      (normalized_coords.select(1, 1) / (height - 1)) * 2.0 - 1.0;

  // Reshape for grid_sample: [1, 1, N, 2]
  torch::Tensor grid = normalized_coords.view({1, 1, -1, 2});

  // Sample using bilinear interpolation
  torch::Tensor sampled = torch::nn::functional::grid_sample(
      depth_map, grid,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .align_corners(true));

  return sampled.view({-1});
}