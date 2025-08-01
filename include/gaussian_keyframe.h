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

#include <c10/cuda/CUDACachingAllocator.h>
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
#include "debugging_utils.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "mono_depth.h"
#include "point2d.h"
#include "stereo_depth.h"
#include "tensor_utils.h"
#include "types.h"

class GaussianKeyframe {
 public:
  GaussianKeyframe() {}

  explicit GaussianKeyframe(std::size_t fid,
                            int creation_iter = 0,
                            std::filesystem::path keyframe_save_dir = "")
      : fid_(fid),
        creation_iter_(creation_iter),
        keyframe_save_dir_(keyframe_save_dir) {}

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

  void initOptimizer(torch::DeviceType device_type,
                     float pose_lr,
                     float exposure_lr,
                     float depth_scale_bias_lr);

  void step();

  torch::Tensor applyExposureTransform(torch::Tensor& colors);

  torch::Tensor sixD2RotationMatrix(const torch::Tensor& rW2C);

  torch::Tensor getR();
  torch::Tensor getT();
  torch::Tensor getRT();
  torch::Tensor getCenter();

  Eigen::Matrix3d getRotationMatrix();
  Eigen::Matrix3f getRotationMatrixf();
  Eigen::Vector3d getTranslation();
  Eigen::Vector3f getTranslationf();
  Eigen::Quaterniond getQuaternion();
  Eigen::Quaternionf getQuaternionf();

  void updatePoseFromParameters();

  void setupStereoData(float baseline,
                       torch::DeviceType device_type,
                       std::shared_ptr<StereoDepth> depth_estimator,
                       float min_depth,
                       float max_depth);

  std::tuple<std::vector<float>, std::vector<float>>
  extractValidKeypointsForDepthAlignment() const;

  void setupMonoData(torch::DeviceType device_type,
                     std::shared_ptr<MonoDepth> depth_estimator,
                     float min_depth,
                     float max_depth);

  void setupRGBDData();

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
  getTrainingData(const torch::Tensor& undistort_mask,
                  const std::vector<torch::Tensor>& pyramid_masks);

  void generateImagePyramid();
  void generateInverseDepthPyramid(const cv::Mat& depth_mat);

  // Save only memory-heavy data to disk and clear from memory
  void saveDataToDisk();

  // Load memory-heavy data from disk back into memory
  void loadDataFromDisk();

 public:
  std::size_t fid_;
  int creation_iter_;
  int remaining_times_of_use_ = 0;

  bool set_camera_ = false;

  camera_id_t camera_id_;
  int camera_model_id_ = 0;

  std::string img_filename_;
  cv::Mat img_undist_, img_auxiliary_undist_;
  int image_width_;   ///< image
  int image_height_;  ///< image

  int num_gaus_pyramid_sub_levels_;
  std::vector<int> gaus_pyramid_times_of_use_;
  std::vector<std::size_t> gaus_pyramid_width_;   ///< gaus_pyramid image
  std::vector<std::size_t> gaus_pyramid_height_;  ///< gaus_pyramid image
  std::vector<torch::Tensor>
      gaus_pyramid_original_image_;  ///< gaus_pyramid image
  std::vector<torch::Tensor> gaus_pyramid_inv_depth_image_;
  // Tensor gt_alpha_mask_;

  std::vector<float> intr_;  ///< intrinsics

  float FoVx_;  ///< intrinsics
  float FoVy_;  ///< intrinsics

  bool set_pose_ = false;
  bool set_projection_matrix_ = false;

  // Optimizable pose parameters (similar to Python keyframe)
  torch::Tensor rW2C_;  // 3x2 rotation parameters (6D representation)
  torch::Tensor tW2C_;  // 3x1 translation parameters

  float zfar_ = 100.0f;
  float znear_ = 0.01f;

  Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
  float scale_ = 1.0f;

  torch::Tensor world_view_transform_;  ///< transform tensors
  torch::Tensor projection_matrix_;     ///< transform tensors
  torch::Tensor full_proj_transform_;   ///< transform tensors
  torch::Tensor camera_center_;         ///< transform tensors

  std::vector<Point2D> points2D_;
  std::vector<float> kps_pixel_;
  std::vector<float> kps_point_local_;

  bool done_inactive_geo_densify_ = false;

  torch::Tensor exposure_transform_;  // 3x4 matrix
  torch::Tensor depth_scale_, depth_bias_;
  std::vector<torch::Tensor> Tensor_vec_rW2C_, Tensor_vec_tW2C_,
      Tensor_vec_exposure_, Tensor_vec_depth_scale_,
      Tensor_vec_depth_bias_;  // For optimizer
  bool exposure_optimization_enabled_ = false;
  bool pose_optimization_enabled_ = false;
  float pose_lr_ = 1e-4f;  // Learning rate for pose optimization
  std::shared_ptr<torch::optim::Adam> optimizer_;
  int local_iterations_ = 0;  // Per-keyframe counter
  float depth_loss_weight = 1e-2f;
  float depth_loss_weight_decay_ = 0.9;

  torch::Tensor feature_map_;
  torch::Tensor depth_confidence_;

  std::filesystem::path keyframe_save_dir_;
  bool loaded_ = false;
  bool allow_eviction_ = false;
};
