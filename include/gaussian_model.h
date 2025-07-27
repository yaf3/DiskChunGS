/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "gaussian_parameters.h"
#include "general_utils.h"

// Forward declaration to avoid circular dependency
class SparseGaussianAdam;
#include "operate_points.h"
#include "point3d.h"
#include "sh_utils.h"
#include "slam_deps/simple-knn/spatial.h"
#include "slam_deps/tinyply/tinyply.h"
#include "tensor_utils.h"
#include "types.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                      \
  this->Tensor_vec_xyz_ = {this->xyz_};                    \
  this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
  this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
  this->Tensor_vec_opacity_ = {this->opacity_};            \
  this->Tensor_vec_scaling_ = {this->scaling_};            \
  this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                              \
  this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));   \
  this->features_dc_ =                                                        \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->features_rest_ =                                                      \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->scaling_ =                                                            \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->rotation_ =                                                           \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->opacity_ =                                                            \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->max_radii2D_ =                                                        \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->xyz_gradient_accum_ =                                                 \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
  GAUSSIAN_MODEL_TENSORS_TO_VEC

// Enhanced transfer structures to include optimizer states
struct GaussianTransferData {
  torch::Tensor points;
  torch::Tensor features_dc;
  torch::Tensor features_rest;
  torch::Tensor opacities;
  torch::Tensor scaling;
  torch::Tensor rotation;
  torch::Tensor exist_since;

  // Optimizer states
  torch::Tensor position_lrs;
  torch::Tensor xyz_gradient_accum;
  torch::Tensor denom;
  torch::Tensor max_radii2D;

  // Adam optimizer states for each parameter group
  std::vector<torch::Tensor> exp_avg_states;     // 6 parameter groups
  std::vector<torch::Tensor> exp_avg_sq_states;  // 6 parameter groups
  std::vector<torch::Tensor> step_states;        // 6 parameter groups
};

class GaussianModel {
 public:
  explicit GaussianModel(const int sh_degree);
  explicit GaussianModel(const GaussianModelParams& model_params);

  torch::Tensor getScalingActivation();
  torch::Tensor getRotationActivation();
  torch::Tensor getXYZ();
  torch::Tensor getFeatures();
  torch::Tensor getOpacityActivation();
  torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

  int getLocalIteration() const { return local_iteration_; }
  void incrementLocalIteration(int inc = 1) { local_iteration_ += inc; }
  void setLocalIteration(int iter) { local_iteration_ = iter; }

  void createFromPcd(const torch::Tensor& fused_point_cloud,
                     const torch::Tensor& color,
                     const torch::Tensor& new_scales,
                     const torch::Tensor& new_opacities,
                     const int iteration,
                     const float spatial_lr_scale);

  void increasePcd(const torch::Tensor& new_point_cloud,
                   const torch::Tensor& new_colors,
                   const torch::Tensor& new_scales,
                   const torch::Tensor& new_opacities,
                   const int iteration);

  void applyScaledTransformation(
      const float s = 1.0,
      const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                          Eigen::Vector3f::Zero()));
  void scaledTransformationPostfix(torch::Tensor& new_xyz,
                                   torch::Tensor& new_scaling);

  void scaledTransformVisiblePointsOfKeyframe(
      torch::Tensor& point_transformed_flags,
      const torch::Tensor& diff_pose,
      torch::Tensor& kf_world_view_transform,
      torch::Tensor& kf_full_proj_transform,
      const int kf_creation_iter,
      const int stable_num_iter_existence,
      int& num_transformed,
      const float scale = 1.0f);

  void trainingSetup(const GaussianOptimizationParams& training_args);
  void updateLearningRates(const torch::Tensor& visibility);
  void optimizerStep(torch::Tensor& visibility, const uint32_t N);
  void setFeatureLearningRate(float feature_lr);
  void setOpacityLearningRate(float opacity_lr);
  void setScalingLearningRate(float scaling_lr);
  void setRotationLearningRate(float rot_lr);

  void resetOpacity();
  torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

  void prunePoints(torch::Tensor& mask);

  void prune(float min_opacity, int max_screen_size);

  void densificationPostfix(torch::Tensor& new_xyz,
                            torch::Tensor& new_features_dc,
                            torch::Tensor& new_features_rest,
                            torch::Tensor& new_opacities,
                            torch::Tensor& new_scaling,
                            torch::Tensor& new_rotation,
                            torch::Tensor& new_exist_since_iter);

  void densifyAndSplit(torch::Tensor& grads,
                       float grad_threshold,
                       float scene_extent,
                       int N = 2);

  void densifyAndClone(torch::Tensor& grads,
                       float grad_threshold,
                       float scene_extent);

  void densifyAndPrune(float max_grad,
                       float min_opacity,
                       float extent,
                       int max_screen_size);

  void addDensificationStats(const torch::Tensor& viewspace_point_tensor,
                             const torch::Tensor& update_filter);

  // void increasePointsIterationsOfExistence(const int i = 1);

  void loadPly(std::filesystem::path ply_path);
  void savePly(std::filesystem::path result_path);
  void saveSparsePointsPly(std::filesystem::path result_path);

  float percentDense();
  void setPercentDense(const float percent_dense);

  void save_checkpoint(const std::string& path);
  void load_checkpoint_incremental(
      const std::string& path,
      const GaussianOptimizationParams& training_args,
      bool load_auxiliary_tensors = false,
      bool load_optimizer_state = false,
      bool load_existence_info = false,
      bool normalize_quaternions = true,
      bool clear_cache_after_load = true);

  // New methods for optimizer state transfer
  GaussianTransferData extractGaussiansWithStates(const torch::Tensor& mask);
  void addGaussiansWithStates(const GaussianTransferData& transfer_data);
  void initializeFromTransferData(
      const GaussianTransferData& transfer_data,
      const GaussianOptimizationParams& training_args,
      const float spatial_lr_scale);

  struct TensorHeader {
    uint32_t dims;
    uint32_t sizes[8];   // Support up to 8D tensors
    uint32_t dtype;      // torch::ScalarType as uint32_t
    uint64_t data_size;  // Size in bytes
  };

  struct OptimizerHeader {
    uint32_t num_param_groups;
    uint32_t param_counts[6];      // Number of parameters per group
    float learning_rates[6];       // LR for each group
    uint64_t state_data_size;      // Total size of state data
    uint32_t has_optimizer_state;  // 1 if state is saved, 0 if not
  };

  // Memory-mapped file format structures
  struct CompleteMMapHeader {
    uint32_t magic = 0x474D4150;  // "GMAP" in hex
    uint32_t version = 3;         // Incremented for complete optimizer support

    // Model metadata
    uint32_t num_points;
    uint32_t sh_degree;
    float spatial_lr_scale;
    float position_lr_init;
    float position_lr_decay;
    float position_lr_min;
    float percent_dense;
    uint32_t local_iteration;

    // Optimizer metadata
    uint32_t has_optimizer_state;
    uint32_t num_param_groups;
    float learning_rates[6];
    uint32_t param_counts[6];
    uint64_t optimizer_state_size;

    // Tensor shapes and offsets
    uint64_t xyz_size[2];
    uint64_t xyz_offset;

    uint64_t features_dc_size[3];
    uint64_t features_dc_offset;

    uint64_t features_rest_size[3];
    uint64_t features_rest_offset;

    uint64_t scaling_size[2];
    uint64_t scaling_offset;

    uint64_t rotation_size[2];
    uint64_t rotation_offset;

    uint64_t opacity_size[2];
    uint64_t opacity_offset;

    uint64_t max_radii2D_size[1];
    uint64_t max_radii2D_offset;

    uint64_t xyz_gradient_accum_size[2];
    uint64_t xyz_gradient_accum_offset;

    uint64_t denom_size[2];
    uint64_t denom_offset;

    uint64_t exist_since_iter_size[1];
    uint64_t exist_since_iter_offset;

    uint64_t position_lrs_size[1];
    uint64_t position_lrs_offset;

    // Optimizer state offsets
    uint64_t optimizer_state_offset;
    uint64_t step_data_offset;
    uint64_t exp_avg_offsets[6];
    uint64_t exp_avg_sq_offsets[6];

    uint64_t total_file_size;

    // Reserved space for future extensions
    uint64_t reserved[32];
  };

  struct OptimizerStateLayout {
    std::vector<uint64_t> step_offsets;
    std::vector<uint64_t> exp_avg_offsets;
    std::vector<uint64_t> exp_avg_sq_offsets;
    std::vector<std::vector<int64_t>> param_shapes;
  };

  void saveTensorBinary(const torch::Tensor& tensor, std::ofstream& file);
  torch::Tensor loadTensorBinary(std::ifstream& file);
  void save_checkpoint_fast(const std::string& path);
  void load_checkpoint_fast(const std::string& path,
                            const GaussianOptimizationParams& training_args,
                            bool load_auxiliary_tensors = false,
                            bool load_optimizer_state = false,
                            bool load_existence_info = false,
                            bool normalize_quaternions = true);

  // Memory-mapped checkpoint functions for Ubuntu
  void save_checkpoint_mmap(const std::string& path);
  void load_checkpoint_mmap(const std::string& path,
                            const GaussianOptimizationParams& training_args,
                            bool load_auxiliary_tensors = true,
                            bool load_optimizer_state = true,
                            bool load_existence_info = true,
                            bool normalize_quaternions = true);

 protected:
  float exponLrFunc(int step);

 public:
  torch::DeviceType device_type_;

  int sh_degree_;

  torch::Tensor xyz_;
  torch::Tensor features_dc_;
  torch::Tensor features_rest_;
  torch::Tensor scaling_;
  torch::Tensor rotation_;
  torch::Tensor opacity_;
  torch::Tensor max_radii2D_;
  torch::Tensor xyz_gradient_accum_;
  torch::Tensor denom_;
  torch::Tensor exist_since_iter_;

  std::vector<torch::Tensor> Tensor_vec_xyz_, Tensor_vec_feature_dc_,
      Tensor_vec_feature_rest_, Tensor_vec_opacity_, Tensor_vec_scaling_,
      Tensor_vec_rotation_;

  std::shared_ptr<SparseGaussianAdam> optimizer_;
  float percent_dense_;
  float spatial_lr_scale_;

 protected:
  int local_iteration_;
  float position_lr_init_;
  float position_lr_decay_;
  float position_lr_min_;

  // Store per-primitive position learning rates
  torch::Tensor position_lrs_;

  std::mutex mutex_settings_;
};
