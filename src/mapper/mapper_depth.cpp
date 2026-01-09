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
  // Step 1: Create Laplacian kernel (same as Python)
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

  // Step 5: Zero out the borders (exactly like Python)
  // laplacian_norm shape: [1, 1, H, W]
  int H = laplacian_norm.size(-2);  // Second to last dimension
  int W = laplacian_norm.size(-1);  // Last dimension

  // Zero out borders using proper LibTorch indexing
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
  int radius = 3;  // Match Python version
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
  cv::Size model_resolution(1280, 384);

  // ONNX model in Docker image (rebuilt on each Docker build)
  std::string model_path =
      "/workspace/repo/models/"
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(model_resolution.height) + "x" +
      std::to_string(model_resolution.width) + ".onnx";

  // Engine will be saved to /workspace/repo/engines/ (persistent)
  // by StereoDepth::initialize_model()

  // std::string model_path =
  //     "/workspace/models/crestereo/"
  //     "crestereo_init_iter20_720x1280.onnx";
  // std::string model_path =
  //     "/workspace/models/IGEV-plusplus/"
  //     "IGEVplusplusRT_fp32_iter6_kitti.onnx";
  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(model_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  // std::string model_path =
  // "/workspace/repo/models/metric3dv2/metric3d-vit-large.onnx";
  // std::string model_path =
  // "/workspace/repo/models/depth_anything/"
  // "depth_anything_v2_vitl.onnx";

  // ONNX model in Docker image (rebuilt on each Docker build)
  std::string onnx_path =
      "/workspace/repo/models/"
      "depth_anything_v2_vitl.onnx";

  // Engine in persistent volume mount (survives Docker rebuilds)
  std::string persistent_engine_path =
      "/workspace/repo/engines/depth_anything_v2_vitl.engine";

  // Temporary engine path (where DepthAnything initially saves it)
  std::string temp_engine_path =
      "/workspace/models/depth_anything_v2_vitl.engine";

  std::string model_path;

  // Check if persistent engine file exists
  if (std::filesystem::exists(persistent_engine_path)) {
    std::cout << "Using cached TensorRT engine: " << persistent_engine_path
              << std::endl;
    model_path = persistent_engine_path;
  } else {
    std::cout << "Persistent engine not found. Building from ONNX: "
              << onnx_path << std::endl;

    // Check if temporary engine exists from previous run
    if (std::filesystem::exists(temp_engine_path)) {
      std::cout << "Found temporary engine, moving to persistent location..."
                << std::endl;
      // Create engines directory if it doesn't exist
      std::filesystem::create_directories("/workspace/repo/engines");
      std::filesystem::copy_file(
          temp_engine_path, persistent_engine_path,
          std::filesystem::copy_options::overwrite_existing);
      model_path = persistent_engine_path;
    } else {
      // Build from ONNX (DepthAnything will save to temp location)
      std::cout << "Building TensorRT engine from ONNX (this may take a few "
                   "minutes)..."
                << std::endl;
      model_path = onnx_path;

      // Initialize with ONNX to trigger build
      this->monocular_depth_estimator_ =
          std::make_shared<MonoDepth>(model_path);

      // Copy the built engine to persistent location
      if (std::filesystem::exists(temp_engine_path)) {
        std::cout << "Saving engine to persistent location: "
                  << persistent_engine_path << std::endl;
        std::filesystem::create_directories("/workspace/repo/engines");
        std::filesystem::copy_file(
            temp_engine_path, persistent_engine_path,
            std::filesystem::copy_options::overwrite_existing);
        std::cout << "Engine saved! Future runs will use the cached engine."
                  << std::endl;
      }
      return;  // Already initialized
    }
  }

  this->monocular_depth_estimator_ = std::make_shared<MonoDepth>(model_path);
}

void GaussianMapper::projectRgbDepthToPointCloud(
    torch::Tensor& rgb_tensor,
    torch::Tensor& depth_tensor,
    std::vector<float>& camera_intrinsics,
    float min_depth,
    float max_depth,
    Sophus::SE3f& pose,
    std::string& output_path,
    int subsample_factor) {
  int height = rgb_tensor.size(1);
  int width = rgb_tensor.size(2);

  std::cout << "Projecting " << width << "x" << height
            << " image to point cloud..." << std::endl;

  std::cout << "RGB tensor size: " << rgb_tensor.sizes() << std::endl;
  std::cout << "Depth tensor size: " << depth_tensor.sizes() << std::endl;

  // Create validity mask for depth
  // torch::Tensor valid_depth =
  //     (depth_tensor >= min_depth) & (depth_tensor <= max_depth);
  torch::Tensor valid_depth = torch::ones_like(depth_tensor, torch::kBool);

  // Optional: Add subsampling for performance
  if (subsample_factor > 1) {
    torch::Tensor subsample_mask = torch::zeros_like(valid_depth);
    for (int v = 0; v < height; v += subsample_factor) {
      for (int u = 0; u < width; u += subsample_factor) {
        if (v < height && u < width) {
          subsample_mask[v][u] = true;
        }
      }
    }
    valid_depth = valid_depth & subsample_mask;
  }

  // Flatten for processing (following your existing pattern)
  torch::Tensor sample_mask = valid_depth.flatten();
  torch::Tensor depth_flat = depth_tensor.flatten();
  torch::Tensor rgb_flat =
      rgb_tensor.permute({1, 2, 0}).flatten(0, 1);  // HWC -> (H*W)C

  // Get valid data
  torch::Tensor sampled_colors = rgb_flat.index({sample_mask});
  torch::Tensor sampled_depths = depth_flat.index({sample_mask});

  std::cout << "Valid points after filtering: " << sampled_depths.size(0)
            << std::endl;

  if (sampled_depths.size(0) == 0) {
    std::cerr << "No valid depth points found!" << std::endl;
    return;
  }

  // Reproject to 3D using your existing function
  torch::Tensor points3D =
      reprojectDepthPinhole(depth_flat, sample_mask, camera_intrinsics, width);
  points3D = points3D.index({sample_mask});

  // Transform to world coordinates if pose is provided
  if (!pose.matrix().isIdentity()) {
    Sophus::SE3f Twc = pose.inverse();  // Convert camera-to-world
    torch::Tensor Twc_tensor =
        tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
            .transpose(0, 1);
    transformPoints(points3D, Twc_tensor);
  }

  // Visualize using your existing function
  visualizePointCloud(points3D, sampled_colors, output_path);

  // Print some statistics
  auto points_cpu = points3D.cpu();
  auto points_accessor = points_cpu.accessor<float, 2>();

  float min_x = points_accessor[0][0], max_x = points_accessor[0][0];
  float min_y = points_accessor[0][1], max_y = points_accessor[0][1];
  float min_z = points_accessor[0][2], max_z = points_accessor[0][2];

  int num_points = points_cpu.size(0);
  for (int i = 0; i < num_points; i++) {
    min_x = std::min(min_x, points_accessor[i][0]);
    max_x = std::max(max_x, points_accessor[i][0]);
    min_y = std::min(min_y, points_accessor[i][1]);
    max_y = std::max(max_y, points_accessor[i][1]);
    min_z = std::min(min_z, points_accessor[i][2]);
    max_z = std::max(max_z, points_accessor[i][2]);
  }

  std::cout << "Point cloud bounds:" << std::endl;
  std::cout << "  X: [" << min_x << ", " << max_x << "]" << std::endl;
  std::cout << "  Y: [" << min_y << ", " << max_y << "]" << std::endl;
  std::cout << "  Z: [" << min_z << ", " << max_z << "]" << std::endl;
}

void GaussianMapper::projectKeypointsToPointCloud(
    std::shared_ptr<GaussianKeyframe> pkf,
    const std::string& output_path) {
  std::vector<float> valid_points_3d;  // Will store [x1,y1,z1, x2,y2,z2, ...]
  std::vector<float> valid_colors;     // Will store [r1,g1,b1, r2,g2,b2, ...]

  int num_keypoints = pkf->kps_pixel_.size() / 2;

  for (int i = 0; i < num_keypoints; i++) {
    float u = pkf->kps_pixel_[2 * i];
    float v = pkf->kps_pixel_[2 * i + 1];
    float x = pkf->kps_point_local_[3 * i];
    float y = pkf->kps_point_local_[3 * i + 1];
    float z = pkf->kps_point_local_[3 * i + 2];

    bool has_valid_3d =
        (z > 0.1f && z < 100.0f) && (u >= 0 && u < pkf->image_width_) &&
        (v >= 0 && v < pkf->image_height_) && std::isfinite(x) &&
        std::isfinite(y) && std::isfinite(z);

    if (has_valid_3d) {
      // Add 3D point
      valid_points_3d.push_back(x);
      valid_points_3d.push_back(y);
      valid_points_3d.push_back(z);

      // Add red color
      valid_colors.push_back(1.0f);  // R
      valid_colors.push_back(0.0f);  // G
      valid_colors.push_back(0.0f);  // B
    }
  }

  if (valid_points_3d.empty()) {
    std::cerr << "No valid keypoints found!" << std::endl;
    return;
  }

  int num_valid = valid_points_3d.size() / 3;
  std::cout << "Found " << num_valid << " valid keypoints out of "
            << num_keypoints << " total" << std::endl;

  // Create tensors from vectors
  torch::Tensor points3D =
      torch::from_blob(valid_points_3d.data(), {num_valid, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .clone();  // Clone to own the memory

  torch::Tensor colors =
      torch::from_blob(valid_colors.data(), {num_valid, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .clone();  // Clone to own the memory

  // Transform to world coordinates if needed
  Sophus::SE3f pose = pkf->getPosef();
  if (!pose.matrix().isIdentity()) {
    Sophus::SE3f Twc = pose.inverse();
    torch::Tensor Twc_tensor =
        tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
            .transpose(0, 1);
    transformPoints(points3D, Twc_tensor);
  }

  // Use your existing visualization function
  visualizePointCloud(points3D, colors, output_path);

  // Print statistics
  // std::cout << "Keypoint cloud statistics:" << std::endl;
  // auto points_cpu = points3D.cpu();
  // auto mins = points_cpu.min(0).values;
  // auto maxs = points_cpu.max(0).values;
  // std::cout << " X: [" << mins[0].item<float>() << ", " <<
  // maxs[0].item<float>()
  //           << "]" << std::endl;
  // std::cout << " Y: [" << mins[1].item<float>() << ", " <<
  // maxs[1].item<float>()
  //           << "]" << std::endl;
  // std::cout << " Z: [" << mins[2].item<float>() << ", " <<
  // maxs[2].item<float>()
  //           << "]" << std::endl;
}

void GaussianMapper::updateORBSLAMPoses() {
  if (!pSLAM_) return;

  auto* atlas = pSLAM_->getAtlas();
  auto* map = atlas->GetCurrentMap();

  // Get all ORB-SLAM keyframes
  std::vector<ORB_SLAM3::KeyFrame*> orb_keyframes;
  {
    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
    orb_keyframes = map->GetAllKeyFrames();
  }

  // Update each ORB-SLAM keyframe with optimized pose
  for (auto* orb_kf : orb_keyframes) {
    unsigned long kf_id = orb_kf->mnId;

    // Find corresponding Gaussian keyframe
    auto gaussian_kf_it = scene_->keyframes().find(kf_id);
    if (gaussian_kf_it != scene_->keyframes().end()) {
      auto gaussian_kf = gaussian_kf_it->second;

      // Get optimized pose from Gaussian keyframe
      Sophus::SE3f optimized_pose = gaussian_kf->getPosef();

      // Convert to ORB-SLAM format and update
      orb_kf->SetPose(optimized_pose);
    }
  }
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
