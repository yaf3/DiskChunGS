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

#pragma once

#include <torch/torch.h>

#include <Eigen/Geometry>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "camera.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "point2d.h"
#include "stereo_depth.h"
#include "tensor_utils.h"
#include "types.h"

class ExponentialLRScheduler {
 private:
  float lr_init_;
  float lr_final_;
  float lr_delay_mult_;
  int lr_delay_steps_;
  int max_steps_;

 public:
  ExponentialLRScheduler(float lr_init,
                         float lr_final,
                         float lr_delay_mult = 1.0f,
                         int lr_delay_steps = 0,
                         int max_steps = 1000000)
      : lr_init_(lr_init),
        lr_final_(lr_final),
        lr_delay_mult_(lr_delay_mult),
        lr_delay_steps_(lr_delay_steps),
        max_steps_(max_steps) {}

  float getLR(int step) {
    if (lr_init_ == 0.0f) return 0.0f;
    if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f)) return 0.0f;

    // Calculate delay rate (reverse cosine decay)
    float delay_rate;
    if (lr_delay_steps_ > 0) {
      float delay_progress =
          std::min(1.0f, float(step) / float(lr_delay_steps_));
      delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) *
                                        std::sin(0.5f * M_PI * delay_progress);
    } else {
      delay_rate = 1.0f;
    }

    // Log-linear interpolation (true exponential decay)
    float t = std::min(1.0f, float(step) / float(max_steps_));
    float log_lerp =
        std::exp(std::log(lr_init_) * (1.0f - t) + std::log(lr_final_) * t);

    return delay_rate * log_lerp;
  }
};

class GaussianKeyframe {
 public:
  GaussianKeyframe() {}

  explicit GaussianKeyframe(std::size_t fid, int creation_iter = 0)
      : fid_(fid), creation_iter_(creation_iter) {}

  void setPose(const double qw,
               const double qx,
               const double qy,
               const double qz,
               const double tx,
               const double ty,
               const double tz);

  void setPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);

  Sophus::SE3d getPose();
  Sophus::SE3f getPosef();

  void setCameraParams(const Camera& camera);

  void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);
  void setPoint3DIdxForPoint2D(const point2D_idx_t point2D_idx,
                               const point3D_id_t point3D_id);

  void computeTransformTensors();

  Eigen::Matrix4f getWorld2View2(const Eigen::Vector3f& trans = {0.0f, 0.0f,
                                                                 0.0f},
                                 float scale = 1.0f);

  torch::Tensor getProjectionMatrix(
      float znear,
      float zfar,
      float fovX,
      float fovY,
      torch::DeviceType device_type = torch::kCUDA);

  int getCurrentGausPyramidLevel();

  void initAppearanceParams(torch::DeviceType device_type,
                            float exposure_lr_init = 0.001f,
                            float exposure_lr_final = 0.0001f,
                            float lr_delay_mult = 0.01f,
                            int lr_delay_steps = 0,
                            int max_iterations = 30000);

  void stepAppearanceOptimizer();

  torch::Tensor applyAppearanceTransform(torch::Tensor& colors);

  void setupStereoData(float baseline,
                       torch::DeviceType device_type,
                       std::shared_ptr<StereoDepth> depth_estimator,
                       float min_depth,
                       float max_depth);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  getRightCameraTransforms() const;

 public:
  std::size_t fid_;
  int creation_iter_;
  int remaining_times_of_use_ = 0;

  bool set_camera_ = false;

  camera_id_t camera_id_;
  int camera_model_id_ = 0;

  std::string img_filename_;
  cv::Mat img_undist_, img_auxiliary_undist_;
  torch::Tensor original_image_, depth_image_;  ///< image, depth
  int image_width_;                             ///< image
  int image_height_;                            ///< image

  int num_gaus_pyramid_sub_levels_;
  std::vector<int> gaus_pyramid_times_of_use_;
  std::vector<std::size_t> gaus_pyramid_width_;   ///< gaus_pyramid image
  std::vector<std::size_t> gaus_pyramid_height_;  ///< gaus_pyramid image
  std::vector<torch::Tensor>
      gaus_pyramid_original_image_;  ///< gaus_pyramid image
  std::vector<torch::Tensor> gaus_pyramid_right_original_image_;
  std::vector<torch::Tensor> gaus_pyramid_depth_image_;
  // Tensor gt_alpha_mask_;

  std::vector<float> intr_;  ///< intrinsics

  float FoVx_;  ///< intrinsics
  float FoVy_;  ///< intrinsics

  bool set_pose_ = false;
  bool set_projection_matrix_ = false;

  Eigen::Quaterniond R_quaternion_;  ///< extrinsics
  Eigen::Vector3d t_;                ///< extrinsics
  Sophus::SE3d Tcw_;                 ///< extrinsics

  torch::Tensor R_tensor_;  ///< extrinsics
  torch::Tensor t_tensor_;  ///< extrinsics

  float zfar_ = 100.0f;
  float znear_ = 0.01f;

  Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
  float scale_ = 1.0f;

  torch::Tensor world_view_transform_;  ///< transform tensors
  torch::Tensor projection_matrix_;     ///< transform tensors
  torch::Tensor full_proj_transform_;   ///< transform tensors
  torch::Tensor camera_center_;         ///< transform tensors

  bool is_stereo_ = false;
  torch::Tensor right_original_image_;  // Pre-processed right image
  torch::Tensor
      world_view_transform_right_;  // Right camera world-to-view transform
  torch::Tensor
      full_proj_transform_right_;      // Right camera full projection transform
  torch::Tensor camera_center_right_;  // Right camera center

  std::vector<Point2D> points2D_;
  std::vector<float> kps_pixel_;
  std::vector<float> kps_point_local_;

  bool done_inactive_geo_densify_ = false;

  // Appearance embedding parameters (affine transform)
  torch::Tensor appearance_transform_;  // 3x4 matrix
  bool has_appearance_params_ = false;
  std::shared_ptr<torch::optim::Adam> appearance_optimizer_;
  std::unique_ptr<ExponentialLRScheduler> exposure_scheduler_;
  int local_iterations_ = 0;  // Per-keyframe counter
};
