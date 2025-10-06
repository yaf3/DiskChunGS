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
#include <linux/falloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "gaussian_keyframe.h"
#include "gaussian_parameters.h"
#include "general_utils.h"

// Forward declaration to avoid circular dependency
class SparseGaussianAdam;
#include "frustum_culler.h"
#include "operate_points.h"
#include "point3d.h"
#include "sh_utils.h"
#include "slam_deps/simple-knn/spatial.h"
#include "slam_deps/tinyply/tinyply.h"
#include "tensor_utils.h"
#include "types.h"

// Ultra-fast memory-mapped chunk storage - replace saveSingleChunkToDisk &
// loadSingleChunkFromDisk

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                      \
  this->Tensor_vec_xyz_ = {this->xyz_};                    \
  this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
  this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
  this->Tensor_vec_opacity_ = {this->opacity_};            \
  this->Tensor_vec_scaling_ = {this->scaling_};            \
  this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                            \
  this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
  this->features_dc_ =                                                      \
      torch::empty(0, torch::TensorOptions().device(device_type));          \
  this->features_rest_ =                                                    \
      torch::empty(0, torch::TensorOptions().device(device_type));          \
  this->scaling_ =                                                          \
      torch::empty(0, torch::TensorOptions().device(device_type));          \
  this->rotation_ =                                                         \
      torch::empty(0, torch::TensorOptions().device(device_type));          \
  this->opacity_ =                                                          \
      torch::empty(0, torch::TensorOptions().device(device_type));          \
  GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel {
 public:
  explicit GaussianModel(const GaussianModelParams& model_params,
                         std::string storage_base_path = "",
                         float chunk_size = 20.0f);

  torch::Tensor getScalingActivation();
  torch::Tensor getRotationActivation();
  torch::Tensor getXYZ();
  torch::Tensor getFeatures();
  torch::Tensor getOpacityActivation();
  torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

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

  void resetOpacity();
  void resetOpacityForMask(const torch::Tensor& gaussian_mask);
  void resetPositionLRAndOptimizerState(const torch::Tensor& gaussian_mask);
  void maskGradients(const torch::Tensor& keep_mask);
  torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

  void prunePoints(torch::Tensor& mask);

  void densificationPostfix(torch::Tensor& new_xyz,
                            torch::Tensor& new_features_dc,
                            torch::Tensor& new_features_rest,
                            torch::Tensor& new_opacities,
                            torch::Tensor& new_scaling,
                            torch::Tensor& new_rotation,
                            torch::Tensor& new_exist_since_iter,
                            torch::Tensor& new_chunk_ids,
                            torch::Tensor& new_position_lrs,
                            torch::Tensor& new_lod_levels,
                            torch::Tensor& new_gaussian_ids,
                            const std::vector<torch::Tensor>& loaded_exp_avg = {},
                            const std::vector<torch::Tensor>& loaded_exp_avg_sq = {},
                            const std::vector<int64_t>& loaded_step_counts = {});

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
  torch::Tensor exist_since_iter_;
  torch::Tensor gaussian_chunk_ids_;
  torch::Tensor gaussian_lod_levels_;  // LoD level assignment for each gaussian

  std::vector<torch::Tensor> Tensor_vec_xyz_, Tensor_vec_feature_dc_,
      Tensor_vec_feature_rest_, Tensor_vec_opacity_, Tensor_vec_scaling_,
      Tensor_vec_rotation_;

  std::shared_ptr<SparseGaussianAdam> optimizer_;
  float spatial_lr_scale_;

 protected:
  int local_iteration_;
  float position_lr_init_;
  float position_lr_decay_;
  float position_lr_min_;

  // Store per-primitive position learning rates
  torch::Tensor position_lrs_;

  std::mutex mutex_settings_;

 public:
  float chunk_size_;

  // LoD system parameters
  bool enable_lod_;
  float lod_distance_multiplier_;

  // Cache for keyframe visibility results
  FrustumCullingCache gaussian_visibility_cache_;

  std::vector<ChunkCoord> frustumCullChunks(
      std::shared_ptr<GaussianKeyframe> keyframe,
      bool use_cache);
  torch::Tensor cullVisibleGaussians(std::shared_ptr<GaussianKeyframe> keyframe,
                                     bool use_lod = true,
                                     bool manage_memory = true);

  // LoD system methods
  torch::Tensor assignLoDByPercentiles(
      const torch::Tensor& nearest_distances,
      float lod0_percentile = 75.0f,  // Top 25% get LoD 0
      float lod2_percentile = 25.0f   // Bottom 25% get LoD 2
  );

  torch::Tensor selectScreenSpaceLoD(const torch::Tensor& visible_gaussian_mask,
                                     const torch::Tensor& camera_position,
                                     float focal_length,
                                     int image_width);

  torch::Tensor selectCumulativeLoD(const torch::Tensor& visible_gaussian_mask,
                                    const torch::Tensor& camera_position);

  torch::Tensor createGaussianMaskFromChunks(
      const torch::Tensor& visible_chunk_ids);

  void pruneLowOpacityGaussians(std::shared_ptr<GaussianKeyframe> pkf,
                                const torch::Tensor& visible_gaussian_mask);

  void updateChunkIDs();

  // Recompute chunk IDs for gaussians after loop closure transformations
  void recomputeChunkIdsAfterLoopClosure();

  bool is_initialized_ = false;

  void addPoints(const torch::Tensor& new_xyz,
                 const torch::Tensor& new_colors,
                 const torch::Tensor& new_scales,
                 const torch::Tensor& new_opacities,
                 int iteration,
                 float spatial_lr_scale);

  void initializeFromPoints(const torch::Tensor& initial_xyz,
                            const torch::Tensor& initial_colors,
                            const torch::Tensor& initial_scales,
                            const torch::Tensor& initial_opacities,
                            int iteration,
                            float spatial_lr_scale);

  void appendPoints(const torch::Tensor& new_xyz,
                    const torch::Tensor& new_colors,
                    const torch::Tensor& new_scales,
                    const torch::Tensor& new_opacities,
                    int iteration);

  // Storage tracking
  torch::Tensor chunks_loaded_from_disk_;
  torch::Tensor chunks_on_disk_;
  torch::Tensor chunk_gaussian_counts_;
  torch::Tensor gaussian_ids_;

  int64_t next_gaussian_id_ = 0;

  // For chunk-based save/load operations
  std::string storage_base_path_;

  // Memory management
  float max_memory_gb_ = 8.0f;  // Configurable
  int64_t max_gaussians_in_memory_ = 3000000;
  std::chrono::steady_clock::time_point last_memory_check_;

  int64_t debug_expected_gaussian_count_ = 0;
  std::unordered_map<int64_t, int64_t> debug_chunk_gaussian_counts_;

  torch::Tensor
      chunk_last_used_;  // [N] - float tensor of timestamps (as float seconds)
  std::unordered_map<int64_t, float>
      chunk_access_times_;  // chunk_id -> timestamp
  float memory_pressure_threshold_ = 0.85f;
  size_t min_chunks_to_evict_ = 5;
  int new_gaussian_chunk_density_ = 100;

  size_t getCurrentGPUMemoryUsage() const;

  struct ChunkData {
    // Main tensors
    torch::Tensor xyz, features_dc, features_rest;
    torch::Tensor scaling, rotation, opacity;
    torch::Tensor exist_since, position_lrs;
    torch::Tensor lod_levels, gaussian_ids;

    // Optimizer states
    std::vector<torch::Tensor> exp_avg_states;     // [6] tensors
    std::vector<torch::Tensor> exp_avg_sq_states;  // [6] tensors
    std::vector<int64_t> step_counts;              // [6] step counts

    int num_points;
    int64_t chunk_id;
  };

  struct TensorHeader {
    uint32_t dims;
    uint32_t sizes[8];   // Support up to 8D tensors
    uint32_t dtype;      // torch::ScalarType as uint32_t
    uint64_t data_size;  // Size in bytes
  };

  void saveTensorBinary(const torch::Tensor& tensor, std::ofstream& file);
  torch::Tensor loadTensorBinary(std::ifstream& file);

  std::string getChunkFilename(const ChunkCoord& coord);

  void loadChunks(const torch::Tensor& chunk_ids_to_load);
  void saveSingleChunkToDisk(int64_t chunk_id, const ChunkData& chunk_data);
  std::optional<ChunkData> loadSingleChunkFromDisk(int64_t chunk_id);
  void appendLoadedChunks(const std::vector<ChunkData>& chunks_data,
                          const std::vector<int64_t>& chunk_ids);
  void saveChunks(const torch::Tensor& chunk_ids_to_save);
  ChunkData extractChunkData(const torch::Tensor& chunk_mask, int64_t chunk_id);
  void saveAndEvictChunks(const torch::Tensor& chunk_ids);

  torch::Tensor findLRUChunks(const torch::Tensor& candidate_chunks,
                              int64_t target_gaussian_count);

  void checkMemoryPressure();
  void testSaveLoadEvictCycle();

  void updateChunkAccess(const torch::Tensor& accessed_chunk_ids);
  void saveAllChunks();
  int64_t countAllGaussians();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  filterPointsByChunkDensity(const torch::Tensor& xyz,
                             const torch::Tensor& colors,
                             const torch::Tensor& scales,
                             const torch::Tensor& opacities,
                             int min_gaussians_per_chunk);
  void initializeEmpty(float spatial_lr_scale);
  void deleteSparseChunks(int min_gaussians_per_chunk);
  void deleteSparseChunkFiles(const torch::Tensor& chunk_ids);

  int min_chunk_occupancy_for_loaded_ = 50;

  void handleBatchChunkRedistribution(const torch::Tensor& processed_chunk_ids);

  void assertChunkTrackingConsistency(const std::string& location);
  void assertGaussianCountInvariant(const std::string& location,
                                    bool should_increase = false);
  void assertChunkGaussianCounts(const std::string& location);
  void assertNoDuplicateGaussians(const std::string& location);
  void assertTensorSizesConsistent(const std::string& location);
  void runFullConsistencyCheck(const std::string& location);

  void prune(float min_opacity, float extent, int max_screen_size);
};