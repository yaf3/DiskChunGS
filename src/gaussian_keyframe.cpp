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
  this->R_quaternion_.w() = qw;
  this->R_quaternion_.x() = qx;
  this->R_quaternion_.y() = qy;
  this->R_quaternion_.z() = qz;
  this->R_quaternion_.normalize();
  this->t_.x() = tx;
  this->t_.y() = ty;
  this->t_.z() = tz;

  this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

  this->set_pose_ = true;
}

void GaussianKeyframe::setPose(const Eigen::Quaterniond& q,
                               const Eigen::Vector3d& t) {
  this->R_quaternion_ = q;
  this->R_quaternion_.normalize();
  this->t_ = t;

  this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

  this->set_pose_ = true;
}

Sophus::SE3d GaussianKeyframe::getPose() { return this->Tcw_; }

Sophus::SE3f GaussianKeyframe::getPosef() { return this->Tcw_.cast<float>(); }

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
  Eigen::Matrix4f Rt;
  Rt.setZero();
  Eigen::Matrix3f R = this->R_quaternion_.toRotationMatrix().cast<float>();
  Rt.topLeftCorner<3, 3>() = R;
  Eigen::Vector3f t = this->t_.cast<float>();
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
  for (int i = 0; i < gaus_pyramid_times_of_use_.size(); ++i) {
    if (gaus_pyramid_times_of_use_[i]) {
      --gaus_pyramid_times_of_use_[i];
      return i;
    }
  }
  // If all sub levels has been used up
  return num_gaus_pyramid_sub_levels_;
}

// Initialize appearance parameters with defaults
void GaussianKeyframe::initAppearanceParams(torch::DeviceType device_type,
                                            float exposure_lr_init,
                                            float exposure_lr_final,
                                            float lr_delay_mult,
                                            int lr_delay_steps,
                                            int max_iterations) {
  if (!has_appearance_params_) {
    // Initialize as 3x4 identity matrix [I|0]
    appearance_transform_ = torch::zeros(
        {3, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type));

    // Set identity for 3x3 part
    appearance_transform_.slice(1, 0, 3) = torch::eye(
        3, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));

    appearance_transform_.requires_grad_(true);

    // Create optimizer with initial learning rate
    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(exposure_lr_init);
    std::vector<torch::Tensor> appearance_params = {appearance_transform_};
    appearance_optimizer_ =
        std::make_shared<torch::optim::Adam>(appearance_params, adam_options);

    // Create learning rate scheduler
    exposure_scheduler_ = std::make_unique<ExponentialLRScheduler>(
        exposure_lr_init, exposure_lr_final, lr_delay_mult, lr_delay_steps,
        max_iterations);

    has_appearance_params_ = true;
  }
}

void GaussianKeyframe::stepAppearanceOptimizer() {
  if (!has_appearance_params_) return;

  // Use local iteration counter for this keyframe
  float current_lr = exposure_scheduler_->getLR(local_iterations_);
  // std::cout << "Keyframe " << fid_ << ": Iter: " << local_iterations_
  //           << " | LR: " << current_lr << std::endl;

  // Update LR and step
  for (auto& param_group : appearance_optimizer_->param_groups()) {
    static_cast<torch::optim::AdamOptions&>(param_group.options())
        .lr(current_lr);
  }

  appearance_optimizer_->step();
  appearance_optimizer_->zero_grad();

  local_iterations_++;  // Increment per-keyframe counter
}

// Apply appearance transform to rendered colors
torch::Tensor GaussianKeyframe::applyAppearanceTransform(
    torch::Tensor& colors) {
  if (!has_appearance_params_) {
    return colors;
  }

  // Permute from [C, H, W] to [H, W, C]
  auto colors_hwc = colors.permute({1, 2, 0});
  auto original_shape = colors_hwc.sizes();  // [H, W, C]

  // Flatten to [H*W, C] for matrix multiplication
  auto colors_flat = colors_hwc.view({-1, 3});  // [H*W, 3]

  // Extract 3x3 transform and bias from 3x4 matrix
  auto transform_3x3 = appearance_transform_.slice(1, 0, 3);    // [3, 3]
  auto bias = appearance_transform_.slice(1, 3, 4).squeeze(1);  // [3]

  // Apply transform: (H*W, 3) @ (3, 3) -> (H*W, 3)
  auto transformed =
      torch::mm(colors_flat, transform_3x3.t()) + bias.unsqueeze(0);

  // Reshape back to [H, W, C] then permute to [C, H, W]
  auto result = transformed.view(original_shape).permute({2, 0, 1});

  // Clamp to [0, 1] like the Python version
  return result.clamp(0.0f, 1.0f);
}

void GaussianKeyframe::setupStereoData(
    float baseline,
    torch::DeviceType device_type,
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm,
    float min_depth,
    float max_depth) {
  if (img_auxiliary_undist_.empty()) {
    return;  // No stereo image available
  }

  is_stereo_ = true;

  // Calculate right camera pose from left camera
  Sophus::SE3f Tcw_left = this->getPosef();
  Sophus::SE3f Twc_left = Tcw_left.inverse();

  // Right camera is offset along camera's x-axis by baseline
  Eigen::Vector3f baseline_offset(baseline, 0, 0);

  // Transform baseline from camera to world coordinates
  Eigen::Vector3f baseline_in_world =
      Twc_left.rotationMatrix() * baseline_offset;

  // Right camera position = left camera position - baseline in world
  Eigen::Vector3f right_pos = Twc_left.translation() + baseline_in_world;

  // Create right camera world-to-camera transform (same rotation, different
  // position)
  Sophus::SE3f Twc_right(Twc_left.rotationMatrix(), right_pos);
  Sophus::SE3f Tcw_right = Twc_right.inverse();

  // Compute and store right camera transformation matrices
  Eigen::Matrix4f right_world_view = Tcw_right.matrix();
  this->world_view_transform_right_ =
      tensor_utils::EigenMatrix2TorchTensor(right_world_view, device_type)
          .transpose(0, 1);

  // The projection matrix is the same for both cameras
  this->full_proj_transform_right_ =
      (this->world_view_transform_right_.unsqueeze(0).bmm(
           this->projection_matrix_.unsqueeze(0)))
          .squeeze(0);

  // Calculate and store right camera center
  this->camera_center_right_ =
      this->world_view_transform_right_.inverse().index(
          {3, torch::indexing::Slice(0, 3)});

  // Preprocess and store right image tensor
  if (device_type == torch::kCUDA) {
    cv::cuda::GpuMat right_gpu;
    right_gpu.upload(this->img_auxiliary_undist_);
    this->right_original_image_ =
        tensor_utils::cvGpuMat2TorchTensor_Float32(right_gpu);

    // Create disparity and compute depth image
    cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;
    cv::cuda::GpuMat left_gpu;
    left_gpu.upload(this->img_undist_);

    // Convert to grayscale for disparity computation
    cv::cuda::cvtColor(left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
    cv::cuda::cvtColor(right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

    // Convert to uint8 required by stereo algorithm
    gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
    gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

    // Compute disparity
    cv::cuda::GpuMat disparity_gpu;
    // Assuming stereo_cv_sgm_ is accessible through external function
    stereo_cv_sgm->compute(gray_left_gpu, gray_right_gpu, disparity_gpu);
    disparity_gpu.convertTo(disparity_gpu, CV_32F, 1.0 / 16.0);

    // Convert disparity to depth
    float focal_length = this->intr_[0];  // fx
    float bf = baseline * focal_length;  // baseline * focal_length (stereo_bf_)

    // Create a valid disparity mask (disparity > 0.1)
    cv::cuda::GpuMat valid_mask;
    cv::cuda::threshold(disparity_gpu, valid_mask, 0.1, 1.0, cv::THRESH_BINARY);
    valid_mask.convertTo(valid_mask, CV_32F);

    // Create constant bf matrix
    cv::cuda::GpuMat bf_mat(disparity_gpu.size(), CV_32FC1, cv::Scalar(bf));

    // Compute depth = bf / disparity for valid disparities
    cv::cuda::GpuMat depth_gpu(disparity_gpu.size(), CV_32FC1);
    cv::cuda::divide(bf_mat, disparity_gpu, depth_gpu);

    // Apply valid mask to eliminate invalid disparities
    cv::cuda::multiply(depth_gpu, valid_mask, depth_gpu);

    // Create min/max depth masks and apply them
    cv::cuda::GpuMat min_depth_mask, max_depth_mask;
    cv::cuda::threshold(depth_gpu, min_depth_mask, min_depth, 1.0,
                        cv::THRESH_BINARY);
    cv::cuda::threshold(depth_gpu, max_depth_mask, max_depth, 1.0,
                        cv::THRESH_BINARY_INV);

    // 2. Create a mask to exclude the top portion of the image (sky region)
    cv::Mat cpu_height_mask(disparity_gpu.size(), CV_8UC1, cv::Scalar(0));
    // Only keep the bottom 60% of the image (adjust this value based on your
    // scenes)
    int valid_start_y =
        static_cast<int>(cpu_height_mask.rows * 0.4);  // Skip top 40%
    cv::rectangle(cpu_height_mask, cv::Point(0, valid_start_y),
                  cv::Point(cpu_height_mask.cols, cpu_height_mask.rows),
                  cv::Scalar(255), -1);

    // Upload to GPU
    cv::cuda::GpuMat height_mask;
    height_mask.upload(cpu_height_mask);
    height_mask.convertTo(height_mask, CV_32F, 1.0 / 255.0);

    // Combine all masks
    cv::cuda::GpuMat combined_mask;
    cv::cuda::multiply(valid_mask, min_depth_mask, combined_mask);
    cv::cuda::multiply(combined_mask, max_depth_mask, combined_mask);
    cv::cuda::multiply(combined_mask, height_mask, combined_mask);

    // Apply final mask to depth map
    cv::cuda::multiply(depth_gpu, combined_mask, depth_gpu);

    // Store depth image as tensor
    this->depth_image_ = tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);

    // Also handle multi-resolution if needed
    if (!gaus_pyramid_original_image_.empty()) {
      gaus_pyramid_right_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::cuda::GpuMat img_resized;
        cv::cuda::resize(
            right_gpu, img_resized,
            cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]));
        gaus_pyramid_right_original_image_[l] =
            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
      }
    }
  } else {
    this->right_original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
        this->img_auxiliary_undist_, device_type);

    // Also handle multi-resolution pyramid for right image
    if (!gaus_pyramid_original_image_.empty()) {
      gaus_pyramid_right_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::Mat img_resized;
        cv::resize(this->img_auxiliary_undist_, img_resized,
                   cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]));
        gaus_pyramid_right_original_image_[l] =
            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type);
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianKeyframe::getRightCameraTransforms() const {
  if (!is_stereo_) {
    throw std::runtime_error(
        "Attempted to get right camera transforms for non-stereo keyframe");
  }
  return std::make_tuple(world_view_transform_right_,
                         full_proj_transform_right_, camera_center_right_);
}