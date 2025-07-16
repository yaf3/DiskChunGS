/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_keyframe.h"

void GaussianKeyframe::setPose(const double qw,
                               const double qx,
                               const double qy,
                               const double qz,
                               const double tx,
                               const double ty,
                               const double tz) {
  // Convert quaternion to rotation matrix
  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();
  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Vector3d t(tx, ty, tz);

  // Initialize tensor representation directly
  rW2C_ = torch::zeros({3, 2}, torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCUDA)
                                   .requires_grad(true));

  tW2C_ = torch::zeros({3}, torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(torch::kCUDA)
                                .requires_grad(true));

  // Copy pose to tensor parameters
  {
    torch::NoGradGuard no_grad;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        rW2C_[i][j] = static_cast<float>(R(i, j));
      }
      tW2C_[i] = static_cast<float>(t(i));
    }
  }

  this->set_pose_ = true;
}

void GaussianKeyframe::setPose(const Eigen::Quaterniond& q,
                               const Eigen::Vector3d& t) {
  // Normalize and convert to rotation matrix
  Eigen::Quaterniond q_norm = q.normalized();
  Eigen::Matrix3d R = q_norm.toRotationMatrix();

  // Initialize tensor representation
  rW2C_ = torch::zeros({3, 2}, torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCUDA)
                                   .requires_grad(true));

  tW2C_ = torch::zeros({3}, torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(torch::kCUDA)
                                .requires_grad(true));

  // Copy current pose to parameters
  {
    torch::NoGradGuard no_grad;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        rW2C_[i][j] = static_cast<float>(R(i, j));
      }
      tW2C_[i] = static_cast<float>(t(i));
    }
  }

  this->set_pose_ = true;
}

Sophus::SE3d GaussianKeyframe::getPose() {
  // Convert tensor representation to SE3
  torch::Tensor R_tensor = sixD2RotationMatrix(rW2C_);

  // Convert to Eigen
  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;

  auto R_cpu = R_tensor.detach().cpu();
  auto t_cpu = tW2C_.detach().cpu();

  for (int i = 0; i < 3; i++) {
    t_eigen(i) = t_cpu[i].item<float>();
    for (int j = 0; j < 3; j++) {
      R_eigen(i, j) = R_cpu[i][j].item<float>();
    }
  }

  Eigen::Quaterniond q(R_eigen);
  return Sophus::SE3d(q, t_eigen);
}

Sophus::SE3f GaussianKeyframe::getPosef() {
  return this->getPose().cast<float>();
}

void GaussianKeyframe::setCameraParams(const Camera& camera) {
  this->camera_id_ = camera.camera_id_;
  this->camera_model_id_ = camera.model_id_;
  this->image_height_ = camera.height_;
  this->image_width_ = camera.width_;

  this->num_gaus_pyramid_sub_levels_ = camera.num_gaus_pyramid_sub_levels_;
  this->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
  this->gaus_pyramid_width_ = camera.gaus_pyramid_width_;

  this->intr_.resize(camera.params_.size());
  for (std::size_t i = 0; i < camera.params_.size(); ++i)
    this->intr_[i] = static_cast<float>(camera.params_[i]);

  switch (this->camera_model_id_) {
    case 1:  // Pinhole
    {
      float focal_length_x = static_cast<float>(camera.params_[0]);
      float focal_length_y = static_cast<float>(camera.params_[1]);
      this->FoVx_ = graphics_utils::focal2fov(focal_length_x, camera.width_);
      this->FoVy_ = graphics_utils::focal2fov(focal_length_y, camera.height_);
      this->set_camera_ = true;
    } break;

    default: {
      throw std::runtime_error(
          "Colmap camera model not handled: only undistorted datasets (PINHOLE "
          "or SIMPLE_PINHOLE cameras) supported!");
    } break;
  }
}

void GaussianKeyframe::setPoints2D(
    const std::vector<Eigen::Vector2d>& points2D) {
  this->points2D_.clear();
  auto num_points2D = points2D.size();
  this->points2D_.resize(num_points2D);
  for (point2D_idx_t point2D_idx = 0; point2D_idx < num_points2D;
       ++point2D_idx) {
    points2D_[point2D_idx].xy_ = points2D[point2D_idx];
  }
}

void GaussianKeyframe::setPoint3DIdxForPoint2D(const point2D_idx_t point2D_idx,
                                               const point3D_id_t point3D_id) {
  points2D_.at(point2D_idx).point3D_id_ = point3D_id;
}

void GaussianKeyframe::computeTransformTensors() {
  if (this->set_pose_ && this->set_camera_) {
    this->world_view_transform_ =
        tensor_utils::EigenMatrix2TorchTensor(
            this->getWorld2View2(this->trans_, this->scale_), torch::kCUDA)
            .transpose(0, 1);

    if (!this->set_projection_matrix_) {
      this->projection_matrix_ =
          this->getProjectionMatrix(this->znear_, this->zfar_, this->FoVx_,
                                    this->FoVy_, torch::kCUDA)
              .transpose(0, 1);
      this->set_projection_matrix_ = true;
    }

    this->full_proj_transform_ = (this->world_view_transform_.unsqueeze(0).bmm(
                                      this->projection_matrix_.unsqueeze(0)))
                                     .squeeze(0);

    this->camera_center_ = this->world_view_transform_.inverse().index(
        {3, torch::indexing::Slice(0, 3)});
  } else if (!this->set_pose_ && this->set_camera_) {
    std::cerr << "Could not compute transform tensors for keyframe "
              << this->fid_ << " because POSE is not set!" << std::endl;
  } else if (!this->set_camera_) {
    std::cerr << "Could not compute transform tensors for keyframe "
              << this->fid_ << " because CAMERA is not set!" << std::endl;
  } else {
    std::cerr << "Could not compute transform tensors for keyframe "
              << this->fid_ << " because POSE and CAMERA are not set!"
              << std::endl;
  }
}

Eigen::Matrix4f GaussianKeyframe::getWorld2View2(const Eigen::Vector3f& trans,
                                                 float scale) {
  // Get current pose from tensors
  torch::Tensor R_tensor = sixD2RotationMatrix(rW2C_);

  Eigen::Matrix4f Rt;
  Rt.setZero();

  // Convert tensor to Eigen matrix
  auto R_cpu = R_tensor.detach().cpu();
  auto t_cpu = tW2C_.detach().cpu();

  Eigen::Matrix3f R;
  Eigen::Vector3f t;

  for (int i = 0; i < 3; i++) {
    t(i) = t_cpu[i].item<float>();
    for (int j = 0; j < 3; j++) {
      R(i, j) = R_cpu[i][j].item<float>();
    }
  }

  Rt.topLeftCorner<3, 3>() = R;
  Rt.topRightCorner<3, 1>() = t;
  Rt(3, 3) = 1.0f;

  Eigen::Matrix4f C2W = Rt.inverse();
  Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
  cam_center += trans;
  cam_center *= scale;
  C2W.block<3, 1>(0, 3) = cam_center;
  Rt = C2W.inverse();
  return Rt;
}

Eigen::Matrix3d GaussianKeyframe::getRotationMatrix() {
  torch::Tensor R_tensor = sixD2RotationMatrix(rW2C_);

  Eigen::Matrix3d R_eigen;
  auto R_cpu = R_tensor.detach().cpu();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_eigen(i, j) = R_cpu[i][j].item<float>();
    }
  }

  return R_eigen;
}

Eigen::Matrix3f GaussianKeyframe::getRotationMatrixf() {
  return getRotationMatrix().cast<float>();
}

Eigen::Vector3d GaussianKeyframe::getTranslation() {
  Eigen::Vector3d t_eigen;
  auto t_cpu = tW2C_.detach().cpu();

  for (int i = 0; i < 3; i++) {
    t_eigen(i) = t_cpu[i].item<float>();
  }

  return t_eigen;
}

Eigen::Vector3f GaussianKeyframe::getTranslationf() {
  return getTranslation().cast<float>();
}

Eigen::Quaterniond GaussianKeyframe::getQuaternion() {
  Eigen::Matrix3d R = getRotationMatrix();
  return Eigen::Quaterniond(R);
}

Eigen::Quaternionf GaussianKeyframe::getQuaternionf() {
  return getQuaternion().cast<float>();
}

torch::Tensor GaussianKeyframe::getProjectionMatrix(
    float znear,
    float zfar,
    float fovX,
    float fovY,
    torch::DeviceType device_type) {
  float tanHalfFovY = std::tan(fovY / 2);
  float tanHalfFovX = std::tan(fovX / 2);

  float top = tanHalfFovY * znear;
  float bottom = -top;
  float right = tanHalfFovX * znear;
  float left = -right;

  torch::Tensor P =
      torch::zeros({4, 4}, torch::TensorOptions().device(device_type));

  float z_sign = 1.0f;

  P.index({0, 0}) = 2.0 * znear / (right - left);
  P.index({1, 1}) = 2.0 * znear / (top - bottom);
  P.index({0, 2}) = (right + left) / (right - left);
  P.index({1, 2}) = (top + bottom) / (top - bottom);
  P.index({3, 2}) = z_sign;
  P.index({2, 2}) = z_sign * zfar / (zfar - znear);
  P.index({2, 3}) = -(zfar * znear) / (zfar - znear);
  return P;
}

int GaussianKeyframe::getCurrentGausPyramidLevel() {
  // Start from the highest level (smallest image) and work down
  for (int i = num_gaus_pyramid_sub_levels_ - 1; i >= 0; --i) {
    if (gaus_pyramid_times_of_use_[i]) {
      --gaus_pyramid_times_of_use_[i];
      return i;
    }
  }
  // If all sub levels have been used up, default to level 0 (largest/original)
  return 0;
}

// Initialize appearance parameters with defaults
void GaussianKeyframe::initOptimizer(torch::DeviceType device_type,
                                     float pose_lr,
                                     float exposure_lr,
                                     float depth_scale_bias_lr) {
  std::vector<torch::Tensor> params_to_optimize;

  pose_lr_ = pose_lr;

  // Initialize as 3x4 identity matrix [I|0]
  exposure_transform_ = torch::zeros(
      {3, 4},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type));

  // Set identity for 3x3 part
  exposure_transform_.slice(1, 0, 3) = torch::eye(
      3, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));

  exposure_transform_.requires_grad_(true);

  depth_scale_ = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));
  depth_scale_.requires_grad_(true);

  depth_bias_ = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));
  depth_bias_.requires_grad_(true);

  Tensor_vec_rW2C_ = {rW2C_};
  Tensor_vec_tW2C_ = {tW2C_};
  Tensor_vec_exposure_ = {exposure_transform_};
  Tensor_vec_depth_scale_ = {depth_scale_};
  Tensor_vec_depth_bias_ = {depth_bias_};

  torch::optim::AdamOptions adam_options;
  adam_options.lr(pose_lr);
  optimizer_ =
      std::make_shared<torch::optim::Adam>(Tensor_vec_rW2C_, adam_options);
  optimizer_->param_groups()[0].options().set_lr(pose_lr);

  optimizer_->add_param_group(Tensor_vec_tW2C_);
  optimizer_->param_groups()[1].options().set_lr(pose_lr);

  optimizer_->add_param_group(Tensor_vec_exposure_);
  optimizer_->param_groups()[2].options().set_lr(exposure_lr);

  optimizer_->add_param_group(Tensor_vec_depth_scale_);
  optimizer_->param_groups()[3].options().set_lr(depth_scale_bias_lr);

  optimizer_->add_param_group(Tensor_vec_depth_bias_);
  optimizer_->param_groups()[4].options().set_lr(depth_scale_bias_lr);
}

// Simplified step function
void GaussianKeyframe::step() {
  if (!optimizer_) return;

  // Debug: Check gradients before step
  // if (local_iterations_ % 10 == 0) {
  //   std::cout << "=== Pose Optimization Debug (Iteration " <<
  //   local_iterations_
  //             << ") ===" << "(ID: " << fid_ << " )===" << std::endl;

  //   // Check depth_scale_ gradients
  //   if (depth_scale_.defined() && depth_scale_.grad().defined()) {
  //     auto depth_scale_grad_norm =
  //         torch::norm(depth_scale_.grad()).item<float>();
  //     std::cout << "depth_scale_ gradient norm: " << depth_scale_grad_norm
  //               << std::endl;
  //     std::cout << "depth_scale_ values: " << depth_scale_.detach().cpu()
  //               << std::endl;
  //     std::cout << "depth_scale_ gradient: "
  //               << depth_scale_.grad().detach().cpu() << std::endl;
  //   } else {
  //     std::cout << "depth_scale_ gradient not defined!" << std::endl;
  //   }

  //   // Check depth_bias_ gradients
  //   if (depth_bias_.defined() && depth_bias_.grad().defined()) {
  //     auto depth_bias_grad_norm =
  //     torch::norm(depth_bias_.grad()).item<float>(); std::cout <<
  //     "depth_bias_ gradient norm: " << depth_bias_grad_norm
  //               << std::endl;
  //     std::cout << "depth_bias_ values: " << depth_bias_.detach().cpu()
  //               << std::endl;
  //     std::cout << "depth_bias_ gradient: " <<
  //     depth_bias_.grad().detach().cpu()
  //               << std::endl;
  //   } else {
  //     std::cout << "depth_bias_ gradient not defined!" << std::endl;
  //   }

  // if (rW2C_.defined() && rW2C_.grad().defined()) {
  //   auto rW2C_grad_norm = torch::norm(rW2C_.grad()).item<float>();
  //   std::cout << "rW2C gradient norm: " << rW2C_grad_norm << std::endl;
  //   std::cout << "rW2C values: " << rW2C_.detach().cpu() << std::endl;
  // } else {
  //   std::cout << "rW2C gradient not defined!" << std::endl;
  // }

  // if (tW2C_.defined() && tW2C_.grad().defined()) {
  //   auto tW2C_grad_norm = torch::norm(tW2C_.grad()).item<float>();
  //   std::cout << "tW2C gradient norm: " << tW2C_grad_norm << std::endl;
  //   std::cout << "tW2C values: " << tW2C_.detach().cpu() << std::endl;
  // } else {
  //   std::cout << "tW2C gradient not defined!" << std::endl;
  // }
  // }

  depth_loss_weight *= depth_loss_weight_decay_;

  optimizer_->step();
  optimizer_->zero_grad();

  local_iterations_++;
}

// Apply appearance transform to rendered colors
torch::Tensor GaussianKeyframe::applyExposureTransform(torch::Tensor& colors) {
  if (!exposure_transform_.defined()) {
    return colors;
  }
  // Permute from [C, H, W] to [H, W, C]
  auto colors_hwc = colors.permute({1, 2, 0});
  auto original_shape = colors_hwc.sizes();  // [H, W, C]

  // Flatten to [H*W, C] for matrix multiplication
  auto colors_flat = colors_hwc.view({-1, 3});  // [H*W, 3]

  // Extract 3x3 transform and bias from 3x4 matrix
  auto transform_3x3 = exposure_transform_.slice(1, 0, 3);    // [3, 3]
  auto bias = exposure_transform_.slice(1, 3, 4).squeeze(1);  // [3]

  // Apply transform: (H*W, 3) @ (3, 3) -> (H*W, 3)
  auto transformed =
      torch::mm(colors_flat, transform_3x3.t()) + bias.unsqueeze(0);

  // Reshape back to [H, W, C] then permute to [C, H, W]
  auto result = transformed.view(original_shape).permute({2, 0, 1});

  // Clamp to [0, 1] like the Python version
  return result.clamp(0.0f, 1.0f);
}

torch::Tensor GaussianKeyframe::sixD2RotationMatrix(const torch::Tensor& rW2C) {
  // Convert 6D representation to rotation matrix
  // Input: rW2C [3, 2] - first two columns of rotation matrix
  // Output: R [3, 3] - full rotation matrix

  auto a1 = rW2C.select(1, 0);  // First column
  auto a2 = rW2C.select(1, 1);  // Second column

  // Normalize first column
  auto b1 = torch::nn::functional::normalize(
      a1, torch::nn::functional::NormalizeFuncOptions().dim(0));

  // Gram-Schmidt orthogonalization for second column
  auto b2 = a2 - torch::sum(b1 * a2) * b1;
  b2 = torch::nn::functional::normalize(
      b2, torch::nn::functional::NormalizeFuncOptions().dim(0));

  // Cross product for third column
  auto b3 = torch::cross(b1, b2, 0);

  // Stack to form rotation matrix
  return torch::stack({b1, b2, b3}, 1);  // [3, 3]
}

torch::Tensor GaussianKeyframe::getR() { return sixD2RotationMatrix(rW2C_); }

torch::Tensor GaussianKeyframe::getT() { return tW2C_; }

torch::Tensor GaussianKeyframe::getRT() {
  torch::Tensor RT = torch::eye(
      {4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  RT.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
                getR());
  RT.index_put_({torch::indexing::Slice(0, 3), 3}, getT());
  return RT;
}

torch::Tensor GaussianKeyframe::getCenter() {
  return -getR().transpose(0, 1).mv(getT());
}

void GaussianKeyframe::setupStereoData(
    float baseline,
    torch::DeviceType device_type,
    std::shared_ptr<StereoDepth> depth_estimator,
    float min_depth,
    float max_depth) {
  if (img_auxiliary_undist_.empty()) {
    return;  // No stereo image available
  }

  // FIX: Convert float32 [0,1] images to uint8 [0,255] images
  cv::Mat left_img_uint8, right_img_uint8;
  this->img_undist_.convertTo(left_img_uint8, CV_8UC3, 255.0);
  this->img_auxiliary_undist_.convertTo(right_img_uint8, CV_8UC3, 255.0);

  // Verify conversion worked
  // std::cout << "Converted left image - Type: " << left_img_uint8.type()
  //           << " Size: " << left_img_uint8.size() << std::endl;
  // double min_val, max_val;
  // cv::minMaxLoc(left_img_uint8, &min_val, &max_val);
  // std::cout << "Converted left image range: " << min_val << " to " << max_val
  // << std::endl;

  // Now estimate depth with properly formatted images
  cv::Mat depth = depth_estimator->estimate_metric_depth(
      left_img_uint8, right_img_uint8, this->intr_[0], baseline);

  // Clamp to minimum value (e.g., 0.1 meters)
  float min_depth_clamp = 1e-8f;
  cv::max(depth, min_depth_clamp, depth);

  // Invert the depth values
  cv::Mat inverted_depth;
  cv::divide(1.0f, depth, inverted_depth);

  // cv::max(depth, min_depth,
  //         depth);  // Set values < min_depth to min_depth
  // cv::min(depth, max_depth,
  //         depth);  // Set values > max_depth to max_depth

  // cv::Mat min_depth_mask, max_depth_mask;
  // cv::threshold(depth, min_depth_mask, min_depth, 1.0, cv::THRESH_BINARY);
  // cv::threshold(depth, max_depth_mask, max_depth, 1.0,
  // cv::THRESH_BINARY_INV);

  // cv::Mat combined_mask;
  // cv::multiply(max_depth_mask, min_depth_mask, combined_mask);

  // // Apply final mask to depth map
  // cv::cuda::multiply(depth, combined_mask, depth);
  // Get some depth statistics
  // if (!depth.empty()) {
  //   double min_depth, max_depth;
  //   cv::minMaxLoc(depth, &min_depth, &max_depth);
  //   cv::Scalar mean_depth = cv::mean(depth);
  //   std::cout << "Depth range: " << min_depth << " - " << max_depth
  //             << " meters " << std::endl;
  //   std::cout << " Mean depth: " << mean_depth[0] << " meters " << std::endl;
  // }

  // float max_dist = 80;
  // cv::Mat norm_depth_map = 255.0 * (1.0 - depth / max_dist);

  // // Clamp values
  // cv::threshold(norm_depth_map, norm_depth_map, 0, 0, cv::THRESH_TOZERO);
  // cv::threshold(norm_depth_map, norm_depth_map, 255, 0,
  // cv::THRESH_TOZERO_INV);

  // cv::Mat depth_8u;
  // norm_depth_map.convertTo(depth_8u, CV_8U);

  // cv::Mat colored_depth;
  // cv::applyColorMap(depth_8u, colored_depth, cv::COLORMAP_JET);

  // cv::imwrite("kitti_depth_map_pipeline.png", colored_depth);
  // Store depth image as tensor

  // std::string gt_filename =
  //     "./debug_mono/gt_depth_" + std::to_string(this->fid_) + ".png";
  // colorize_and_save_depth(depth_image_.detach().cpu(), gt_filename,
  // min_depth,
  //                         max_depth);

  torch::Tensor depth_image =
      tensor_utils::cvMat2TorchTensor_Float32(inverted_depth, torch::kCUDA)
          .unsqueeze(0)
          .unsqueeze(0);

  // Initialize Sobel kernels
  torch::Tensor sobel_x = torch::tensor(
      {{{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor sobel_y = torch::tensor(
      {{{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  // Compute gradients using Sobel filters
  torch::Tensor grad_x = torch::nn::functional::conv2d(
      depth_image, sobel_x,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  torch::Tensor grad_y = torch::nn::functional::conv2d(
      depth_image, sobel_y,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute edge magnitude and confidence
  torch::Tensor edges = torch::cat({grad_x, grad_y}, 0);
  torch::Tensor edges_sq_norm = (edges.pow(2)).sum(0, true);

  float var = 0.2f;
  depth_confidence_ = torch::exp(-edges_sq_norm / var);

  // Create multi-resolution depth images for pyramid training
  generateInverseDepthPyramid(inverted_depth);
}

/**
 * Extract valid keypoints with 3D coordinates for depth alignment
 * Similar to how the Python code filters keypoints with has_pt3d
 */
std::tuple<std::vector<float>, std::vector<float>>
GaussianKeyframe::extractValidKeypointsForDepthAlignment() const {
  std::vector<float> valid_pixel_coords;
  std::vector<float> valid_depths;

  assert(kps_pixel_.size() % 2 == 0);
  assert(kps_point_local_.size() % 3 == 0);

  int num_keypoints = kps_pixel_.size() / 2;

  for (int i = 0; i < num_keypoints; i++) {
    float u = kps_pixel_[2 * i];      // u coordinate
    float v = kps_pixel_[2 * i + 1];  // v coordinate

    // Get 3D point in local camera frame
    float x = kps_point_local_[3 * i];
    float y = kps_point_local_[3 * i + 1];
    float z = kps_point_local_[3 * i + 2];

    // Check if keypoint has valid 3D coordinates
    // Following the pattern from the Python code where has_pt3d checks for
    // valid points
    // bool has_valid_3d = (z > 0.1f && z < 100.0f) &&  // reasonable depth
    // range
    //                     (u >= 0 && u < image_width_) &&  // within image
    //                     bounds (v >= 0 && v < image_height_) &&
    //                     std::isfinite(x) && std::isfinite(y) &&
    //                     std::isfinite(z);
    bool has_valid_3d = (z > 0.0) &&
                        (u >= 0 && u < image_width_) &&  // within image bounds
                        (v >= 0 && v < image_height_) && std::isfinite(x) &&
                        std::isfinite(y) && std::isfinite(z);

    if (has_valid_3d) {
      valid_pixel_coords.push_back(u);
      valid_pixel_coords.push_back(v);
      valid_depths.push_back(z);  // depth in camera coordinate system
    }
  }

  // std::cout << "Found " << valid_depths.size() << " valid keypoints out of "
  //           << num_keypoints << " total keypoints" << std::endl;

  return std::make_tuple(valid_pixel_coords, valid_depths);
}

void GaussianKeyframe::setupMonoData(torch::DeviceType device_type,
                                     std::shared_ptr<MonoDepth> depth_estimator,
                                     float min_depth,
                                     float max_depth) {
  auto [relative_depth, depth_confidence] =
      depth_estimator->estimate_depth(img_undist_, intr_[0]);

  // std::cout << "Depth info right after prediction" << std::endl;
  // std::cout << relative_depth.sizes() << std::endl;
  // std::cout << relative_depth.mean().item() << std::endl;
  // std::cout << relative_depth.max().item() << std::endl;
  // std::cout << relative_depth.min().item() << std::endl;

  depth_confidence_ = depth_confidence;
  // std::cout << "Relative depth size: " << relative_depth.sizes() <<
  // std::endl;

  // std::string keypoint_pcd_path = "slam_keypoints.ply";
  // projectKeypointsToPointCloud(pkf, keypoint_pcd_path);

  // Extract keypoint pixels and depths
  auto [valid_pixel_coords, valid_depths] =
      extractValidKeypointsForDepthAlignment();

  if (valid_depths.size() < 5) {
    std::cout << "Not enough valid depths for monocular depth alignment: "
              << valid_depths.size() << std::endl;
    return;
  }

  // Align depth to keypoints
  torch::Tensor aligned_inv_depth = depth_estimator->align_depth_equivalent(
      relative_depth, valid_pixel_coords, valid_depths, image_width_,
      image_height_);

  // std::cout << "Aligned depth stats:" << std::endl;
  // std::cout << "  Min: " << aligned_depth.min().item<float>() << std::endl;
  // std::cout << "  Max: " << aligned_depth.max().item<float>() << std::endl;
  // std::cout << "  Mean: " << aligned_depth.mean().item<float>() << std::endl;
  // std::cout << "  Median: " << aligned_depth.median().item<float>()
  //           << std::endl;

  // std::string gt_filename =
  //     "./debug_mono/gt_depth_" + std::to_string(this->fid_) + ".png";
  // colorize_and_save_depth(idepth_.detach().cpu(), gt_filename, 0.0f, 6.0f);

  torch::Tensor inv_depth =
      torch::nn::functional::interpolate(
          aligned_inv_depth,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{image_height_, image_width_})
              .mode(torch::kBilinear)
              .align_corners(true))
          .squeeze(0)
          .squeeze(0);

  // std::filesystem::create_directories("./debug_mono");
  // colorize_and_save_depth(relative_depth.detach().cpu(),
  //                         "./debug_mono/depth_prealigned.png", min_depth,
  //                         max_depth);
  // colorize_and_save_depth(depth_image_.detach().cpu(),
  //                         "./debug_mono/depth_aligned.png", min_depth,
  //                         max_depth);

  // torch::Tensor aligned_depth = relative_depth.squeeze(0).squeeze(0);
  // pkf->depth_image_ = aligned_depth;
  // std::cout << "Aligned depth min value: "
  //           << aligned_depth.min().item<float>()
  //           << ", max value: " << aligned_depth.max().item<float>()
  //           << std::endl;

  // // Convert tensors to cv::Mat for processing
  // torch::Tensor rgb_image =
  //     tensor_utils::cvMat2TorchTensor_Float32(pkf->img_undist_,
  //     device_type_);

  // // Get camera pose (world-to-camera)
  // Sophus::SE3f Tcw = pkf->getPosef();

  // std::string render_filename = "aligned_depth.png";
  // colorize_and_save_depth(aligned_depth.detach().cpu(), render_filename,
  //                         aligned_depth.min().item<float>(),
  //                         aligned_depth.max().item<float>());

  // // Project to point cloud
  // std::string pcd_path = "depth_pcd.ply";
  // projectRgbDepthToPointCloud(rgb_image, aligned_depth, pkf->intr_,
  //                             min_depth_, max_depth_, Tcw, pcd_path, 2);

  // torch::Tensor manual_depth =
  //     relative_depth.squeeze(0).squeeze(0) * 0.41558 - 1.29237;

  // std::cout << "Manual depths range: " << manual_depth.min().item<float>()
  //           << " - " << manual_depth.max().item<float>() << std::endl;

  // pcd_path = "depth_pcd_C.ply";
  // projectRgbDepthToPointCloud(rgb_image, manual_depth, pkf->intr_,
  // min_depth_,
  //                             max_depth_, Tcw, pcd_path, 2);

  cv::Mat inverted_depth_mat =
      tensor_utils::torchTensor2CvMat_Float32(inv_depth);

  generateInverseDepthPyramid(inverted_depth_mat);
}

void GaussianKeyframe::setupRGBDData() {
  cv::Mat depth = img_auxiliary_undist_;

  // Clamp minimum to 1e-8
  cv::Mat clamped_depth;
  cv::max(depth, 1e-8, clamped_depth);

  // Take inverse (1 / clamped_depth)
  cv::Mat inverse_depth;
  cv::divide(1.0, clamped_depth, inverse_depth);

  torch::Tensor depth_image =
      tensor_utils::cvMat2TorchTensor_Float32(inverse_depth, torch::kCUDA)
          .unsqueeze(0)
          .unsqueeze(0);

  // Initialize Sobel kernels
  torch::Tensor sobel_x = torch::tensor(
      {{{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor sobel_y = torch::tensor(
      {{{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  // Compute gradients using Sobel filters
  torch::Tensor grad_x = torch::nn::functional::conv2d(
      depth_image, sobel_x,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  torch::Tensor grad_y = torch::nn::functional::conv2d(
      depth_image, sobel_y,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute edge magnitude and confidence
  torch::Tensor edges = torch::cat({grad_x, grad_y}, 0);
  torch::Tensor edges_sq_norm = (edges.pow(2)).sum(0, true);

  float var = 0.2f;
  depth_confidence_ = torch::exp(-edges_sq_norm / var);

  generateInverseDepthPyramid(inverse_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
GaussianKeyframe::getTrainingData(
    const torch::Tensor& undistort_mask,
    const std::vector<torch::Tensor>& pyramid_masks) {
  int training_level = getCurrentGausPyramidLevel();

  torch::Tensor gt_image, gt_inv_depth, mask;
  int image_height, image_width;

  // Pyramid level
  image_height = gaus_pyramid_height_[training_level];
  image_width = gaus_pyramid_width_[training_level];
  gt_image = gaus_pyramid_original_image_[training_level].cuda();
  mask = pyramid_masks[training_level];

  if (!gaus_pyramid_inv_depth_image_.empty() &&
      training_level < gaus_pyramid_inv_depth_image_.size()) {
    gt_inv_depth = gaus_pyramid_inv_depth_image_[training_level].cuda();
  }

  if (gt_inv_depth.defined()) {
    gt_inv_depth = gt_inv_depth * depth_scale_ + depth_bias_;
  }

  return std::make_tuple(gt_image, gt_inv_depth, mask, image_height,
                         image_width);
}

// In GaussianKeyframe class
void GaussianKeyframe::generateImagePyramid() {
  cv::cuda::GpuMat img_gpu;
  img_gpu.upload(img_undist_);
  gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);

  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    cv::cuda::GpuMat img_resized;
    cv::cuda::resize(img_gpu, img_resized,
                     cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]));
    gaus_pyramid_original_image_[l] =
        tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
  }
}

// In GaussianKeyframe class
void GaussianKeyframe::generateInverseDepthPyramid(const cv::Mat& depth_mat) {
  if (!depth_mat.empty()) {
    gaus_pyramid_inv_depth_image_.resize(num_gaus_pyramid_sub_levels_);

    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(depth_mat);

    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
      cv::cuda::GpuMat depth_resized;
      cv::cuda::resize(
          depth_gpu, depth_resized,
          cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]), 0, 0,
          cv::INTER_NEAREST);
      gaus_pyramid_inv_depth_image_[l] =
          tensor_utils::cvGpuMat2TorchTensor_Float32(depth_resized);
    }
  }
}

void GaussianKeyframe::saveDataToDisk() {
  auto start_time = std::chrono::steady_clock::now();
  if (!loaded_) {
    throw std::runtime_error("Can't save keyframe to disk that isn't loaded");
  }
  // Create directory if it doesn't exist
  std::filesystem::create_directories(keyframe_save_dir_);

  // Save heavy image/depth tensors
  torch::serialize::OutputArchive archive;

  if (depth_confidence_.defined()) {
    archive.write("depth_confidence_", depth_confidence_);
  }

  // Save pyramid image data (these can be large)
  if (!gaus_pyramid_original_image_.empty()) {
    archive.write("pyramid_size", torch::tensor(static_cast<int64_t>(
                                      gaus_pyramid_original_image_.size())));
    for (size_t i = 0; i < gaus_pyramid_original_image_.size(); ++i) {
      if (gaus_pyramid_original_image_[i].defined()) {
        archive.write("pyramid_image_" + std::to_string(i),
                      gaus_pyramid_original_image_[i]);
      }
    }
  }

  // Save pyramid depth data
  if (!gaus_pyramid_inv_depth_image_.empty()) {
    archive.write("pyramid_depth_size",
                  torch::tensor(static_cast<int64_t>(
                      gaus_pyramid_inv_depth_image_.size())));
    for (size_t i = 0; i < gaus_pyramid_inv_depth_image_.size(); ++i) {
      if (gaus_pyramid_inv_depth_image_[i].defined()) {
        archive.write("pyramid_depth_" + std::to_string(i),
                      gaus_pyramid_inv_depth_image_[i]);
      }
    }
  }

  std::filesystem::path data_path =
      keyframe_save_dir_ / ("keyframe_data_" + std::to_string(fid_) + ".pt");
  archive.save_to(data_path.string());

  if (depth_confidence_.defined()) {
    depth_confidence_.reset();
  }

  // Clear pyramid data
  for (auto& img : gaus_pyramid_original_image_) {
    if (img.defined()) {
      img.reset();
    }
  }
  gaus_pyramid_original_image_.clear();

  for (auto& depth : gaus_pyramid_inv_depth_image_) {
    if (depth.defined()) {
      depth.reset();
    }
  }
  gaus_pyramid_inv_depth_image_.clear();

  // c10::cuda::CUDACachingAllocator::emptyCache();

  loaded_ = false;

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Keyframe " << fid_ << " saved. Save completed in "
            << duration.count() << "ms" << std::endl;

  std::cout << "Keyframe data saved and cleared from memory for keyframe "
            << fid_ << std::endl;
}

void GaussianKeyframe::loadDataFromDisk() {
  if (loaded_) {
    std::cout << "WARN: Loading keyframe that is already marked as loaded!"
              << std::endl;
  }
  std::filesystem::path data_path =
      keyframe_save_dir_ / ("keyframe_data_" + std::to_string(fid_) + ".pt");

  if (!std::filesystem::exists(data_path)) {
    std::cout << "No data file found for keyframe " << fid_ << " at "
              << data_path << std::endl;
    return;
  }

  torch::serialize::InputArchive archive;

  try {
    archive.load_from(data_path.string());

    try {
      archive.read("depth_confidence_", depth_confidence_);
      depth_confidence_ = depth_confidence_.to(torch::kCUDA);
    } catch (const std::exception& e) {
      // Silent fail
    }

    // Load pyramid images
    try {
      torch::Tensor pyramid_size_tensor;
      archive.read("pyramid_size", pyramid_size_tensor);
      int pyramid_size = pyramid_size_tensor.item<int64_t>();

      gaus_pyramid_original_image_.resize(pyramid_size);
      for (int i = 0; i < pyramid_size; ++i) {
        try {
          archive.read("pyramid_image_" + std::to_string(i),
                       gaus_pyramid_original_image_[i]);
          gaus_pyramid_original_image_[i] =
              gaus_pyramid_original_image_[i].to(torch::kCUDA);
        } catch (const std::exception& e) {
          // Silent fail for individual pyramid levels
        }
      }
    } catch (const std::exception& e) {
      // Silent fail
    }

    // Load pyramid depths
    try {
      torch::Tensor pyramid_depth_size_tensor;
      archive.read("pyramid_depth_size", pyramid_depth_size_tensor);
      int pyramid_depth_size = pyramid_depth_size_tensor.item<int64_t>();

      gaus_pyramid_inv_depth_image_.resize(pyramid_depth_size);
      for (int i = 0; i < pyramid_depth_size; ++i) {
        try {
          archive.read("pyramid_depth_" + std::to_string(i),
                       gaus_pyramid_inv_depth_image_[i]);
          gaus_pyramid_inv_depth_image_[i] =
              gaus_pyramid_inv_depth_image_[i].to(torch::kCUDA);
        } catch (const std::exception& e) {
          // Silent fail for individual pyramid levels
        }
      }
    } catch (const std::exception& e) {
      // Silent fail
    }

    std::cout << "Data loaded from disk for keyframe " << fid_ << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error loading data for keyframe " << fid_ << ": " << e.what()
              << std::endl;
  }

  loaded_ = true;
}