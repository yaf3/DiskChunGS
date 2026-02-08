/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023
 * as part of Photo-SLAM, modified by Dapeng Feng in 2024 as part of CaRtGS,
 * and further modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 */

/**
 * @file gaussian_model.h
 * @brief 3D Gaussian Splatting model with disk-based chunk streaming.
 *
 * This class implements a 3D Gaussian Splatting model with support for
 * memory-efficient chunk-based storage. Gaussians are organized into spatial
 * chunks that can be loaded/evicted from GPU memory on demand, enabling
 * processing of large-scale scenes that exceed GPU memory capacity.
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
#include "geometry/operate_points.h"
#include "geometry/point3d.h"
#include "rendering/frustum_culler.h"
#include "scene/gaussian_keyframe.h"
#include "scene/gaussian_parameters.h"
#include "slam_deps/simple-knn/spatial.h"
#include "slam_deps/tinyply/tinyply.h"
#include "types.h"
#include "utils/general_utils.h"
#include "utils/sh_utils.h"
#include "utils/tensor_utils.h"

// Forward declaration to avoid circular dependency
class SparseGaussianAdam;

/**
 * @brief Macro to populate tensor vectors from individual tensors.
 *
 * Used to update the std::vector wrappers that the optimizer requires
 * after any operation that modifies the underlying tensors.
 */
#define GAUSSIAN_MODEL_TENSORS_TO_VEC                      \
  this->Tensor_vec_xyz_ = {this->xyz_};                    \
  this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
  this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
  this->Tensor_vec_opacity_ = {this->opacity_};            \
  this->Tensor_vec_scaling_ = {this->scaling_};            \
  this->Tensor_vec_rotation_ = {this->rotation_};

/**
 * @brief Macro to initialize all Gaussian tensors as empty on the specified
 * device.
 * @param device_type The torch device (kCUDA or kCPU) for tensor allocation.
 */
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

/**
 * @class GaussianModel
 * @brief Manages a collection of 3D Gaussians with disk-based chunk streaming.
 *
 * The GaussianModel represents a scene as a set of 3D Gaussian primitives,
 * each parameterized by position (xyz), color (spherical harmonics features),
 * opacity, scale, and rotation. The model supports:
 *
 * - **Chunk-based storage**: Gaussians are spatially partitioned into chunks
 *   that can be independently loaded/saved to disk.
 * - **LRU eviction**: When GPU memory is constrained, least-recently-used
 *   chunks are automatically evicted to disk.
 * - **Frustum culling**: Only chunks visible to a camera are loaded.
 * - **Sparse optimization**: Per-Gaussian learning rates with Adam optimizer.
 *
 * The model maintains optimizer state (Adam momentum) alongside Gaussian
 * parameters, enabling seamless save/restore of training state.
 */
class GaussianModel {
 public:
  /**
   * @brief Constructs a GaussianModel with the given parameters.
   * @param model_params Configuration including SH degree, device, and memory
   * limits.
   * @param storage_base_path Directory path for chunk file storage.
   * @param chunk_size Spatial size of each chunk in world units.
   */
  explicit GaussianModel(const GaussianModelParams& model_params,
                         std::string storage_base_path = "",
                         float chunk_size = 20.0f);

  //============================================================================
  // Activation Getters - Methods to retrieve activated (transformed) Gaussian
  // parameters.
  //============================================================================

  /**
   * @brief Returns scaling with exponential activation applied.
   * @return Tensor of shape [N, 3] with positive scale values.
   */
  torch::Tensor getScalingActivation();

  /**
   * @brief Returns normalized rotation quaternions.
   * @return Tensor of shape [N, 4] with unit quaternions.
   */
  torch::Tensor getRotationActivation();

  /**
   * @brief Returns raw Gaussian positions.
   * @return Tensor of shape [N, 3] with XYZ coordinates.
   */
  torch::Tensor getXYZ();

  /**
   * @brief Returns combined spherical harmonics features (DC + rest).
   * @return Tensor of concatenated SH coefficients.
   */
  torch::Tensor getFeatures();

  /**
   * @brief Returns opacity with sigmoid activation applied.
   * @return Tensor of shape [N, 1] with values in (0, 1).
   */
  torch::Tensor getOpacityActivation();

  /**
   * @brief Computes 3x3 covariance matrices from scale and rotation.
   * @param scaling_modifier Multiplier applied to scaling values.
   * @return Tensor of shape [N, 3, 3] covariance matrices.
   */
  torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

  //============================================================================
  // Transformation Methods - Methods for applying geometric transformations to
  // Gaussians.
  //============================================================================

  /**
   * @brief Applies a scaled rigid transformation to all Gaussians.
   * @param s Scale factor applied before transformation.
   * @param T SE3 transformation (rotation + translation).
   *
   * Transforms positions as: xyz' = s * R * xyz + t
   * Also scales the Gaussian scaling parameters accordingly.
   */
  void applyScaledTransformation(
      const float s = 1.0,
      const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                          Eigen::Vector3f::Zero()));

  /**
   * @brief Updates optimizer state after transformation.
   * @param new_xyz Transformed positions tensor.
   * @param new_scaling Transformed scaling tensor.
   *
   * Replaces the optimizer's tracked tensors and resets their Adam states.
   */
  void scaledTransformationPostfix(torch::Tensor& new_xyz,
                                   torch::Tensor& new_scaling);

  /**
   * @brief Transforms Gaussians visible to a keyframe after pose update.
   * @param point_transformed_flags Output mask of which points were
   * transformed.
   * @param diff_pose Differential pose change.
   * @param kf_world_view_transform Keyframe's world-to-view transform.
   * @param kf_full_proj_transform Keyframe's full projection matrix.
   * @param kf_creation_iter Iteration when keyframe was created.
   * @param stable_num_iter_existence Threshold for "stable" Gaussians.
   * @param num_transformed Output count of transformed Gaussians.
   * @param scale Optional scale factor.
   *
   * Only transforms "unstable" Gaussians (recently created near the keyframe).
   */
  void scaledTransformVisiblePointsOfKeyframe(
      torch::Tensor& point_transformed_flags,
      const torch::Tensor& diff_pose,
      torch::Tensor& kf_world_view_transform,
      torch::Tensor& kf_full_proj_transform,
      const int kf_creation_iter,
      const int stable_num_iter_existence,
      int& num_transformed,
      const float scale = 1.0f);

  //============================================================================
  // Optimization Methods - Methods for training setup and gradient-based
  // optimization.
  //============================================================================

  /**
   * @brief Initializes the optimizer with per-parameter learning rates.
   * @param training_args Learning rate configuration for each parameter group.
   *
   * Sets up a SparseGaussianAdam optimizer with 6 parameter groups:
   * xyz (0), features_dc (1), features_rest (2), opacity (3), scaling (4),
   * rotation (5). Position learning rates are per-Gaussian; others use scalar
   * LRs.
   */
  void trainingSetup(const GaussianOptimizationParams& training_args);

  /**
   * @brief Decays position learning rates for visible Gaussians.
   * @param visibility Boolean mask indicating which Gaussians were rendered.
   *
   * Applies exponential decay to per-Gaussian position learning rates,
   * clamped to a minimum value.
   */
  void updateLearningRates(const torch::Tensor& visibility);

  /**
   * @brief Performs sparse Adam update for visible Gaussians only.
   * @param visibility Boolean mask of visible Gaussians.
   * @param N Total number of Gaussians.
   *
   * Updates only the Gaussians that contributed to the rendered image,
   * using per-Gaussian learning rates for positions.
   */
  void optimizerStep(torch::Tensor& visibility, const uint32_t N);

  /**
   * @brief Resets opacity of all Gaussians to a low value.
   *
   * Used periodically during training to cull Gaussians that don't
   * recover their opacity (indicating they're not needed).
   */
  void resetOpacity();

  /**
   * @brief Resets opacity for a subset of Gaussians.
   * @param gaussian_mask Boolean mask selecting Gaussians to reset.
   */
  void resetOpacityForMask(const torch::Tensor& gaussian_mask);

  /**
   * @brief Resets position learning rates and Adam momentum for selected
   * Gaussians.
   * @param gaussian_mask Boolean mask selecting Gaussians to reset.
   *
   * Used after loop closure to allow affected Gaussians to move freely again.
   */
  void resetPositionLRAndOptimizerState(const torch::Tensor& gaussian_mask);

  /**
   * @brief Replaces a parameter tensor in the optimizer.
   * @param t New tensor value.
   * @param tensor_idx Parameter group index (0-5).
   * @return The new tensor registered with the optimizer.
   *
   * Handles optimizer state bookkeeping when tensors are replaced
   * (e.g., after transformation or pruning).
   */
  torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

  //============================================================================
  // Pruning and Densification - Methods for removing and adding Gaussians.
  //============================================================================

  /**
   * @brief Removes Gaussians indicated by the mask.
   * @param mask Boolean tensor where true indicates points to remove.
   *
   * Updates all parameter tensors and optimizer states accordingly.
   */
  void prunePoints(torch::Tensor& mask);

  /**
   * @brief Appends new Gaussians to the model with optimizer state.
   * @param new_xyz Positions of new Gaussians [M, 3].
   * @param new_features_dc DC spherical harmonics coefficients.
   * @param new_features_rest Higher-order SH coefficients.
   * @param new_opacities Opacity values (pre-sigmoid).
   * @param new_scaling Scale values (pre-exp).
   * @param new_rotation Rotation quaternions.
   * @param new_exist_since_iter Iteration when each Gaussian was created.
   * @param new_chunk_ids Spatial chunk assignment for each Gaussian.
   * @param new_position_lrs Per-Gaussian position learning rates.
   * @param new_gaussian_ids Unique IDs for each Gaussian.
   * @param loaded_exp_avg Optional: Adam first moment from disk.
   * @param loaded_exp_avg_sq Optional: Adam second moment from disk.
   * @param loaded_step_counts Optional: Adam step counts from disk.
   *
   * Concatenates new Gaussians to existing tensors and updates optimizer state.
   * Used both for densification and loading chunks from disk.
   */
  void densificationPostfix(
      torch::Tensor& new_xyz,
      torch::Tensor& new_features_dc,
      torch::Tensor& new_features_rest,
      torch::Tensor& new_opacities,
      torch::Tensor& new_scaling,
      torch::Tensor& new_rotation,
      torch::Tensor& new_exist_since_iter,
      torch::Tensor& new_chunk_ids,
      torch::Tensor& new_position_lrs,
      torch::Tensor& new_gaussian_ids,
      const std::vector<torch::Tensor>& loaded_exp_avg = {},
      const std::vector<torch::Tensor>& loaded_exp_avg_sq = {},
      const std::vector<int64_t>& loaded_step_counts = {});

 protected:
  /**
   * @brief Computes exponentially decaying learning rate.
   * @param step Current optimization step.
   * @return Learning rate value.
   */
  float exponLrFunc(int step);

 public:
  //============================================================================
  // Core Gaussian Parameters
  //============================================================================

  torch::DeviceType device_type_;  ///< CUDA or CPU device for tensors.
  int sh_degree_;                  ///< Spherical harmonics degree (0-3).

  torch::Tensor xyz_;               ///< Gaussian positions [N, 3].
  torch::Tensor features_dc_;       ///< DC SH coefficients [N, 1, 3].
  torch::Tensor features_rest_;     ///< Higher-order SH coefficients [N, K, 3].
  torch::Tensor scaling_;           ///< Log-space scaling [N, 3].
  torch::Tensor rotation_;          ///< Rotation quaternions [N, 4].
  torch::Tensor opacity_;           ///< Logit-space opacity [N, 1].
  torch::Tensor exist_since_iter_;  ///< Creation iteration per Gaussian [N].
  torch::Tensor gaussian_chunk_ids_;  ///< Spatial chunk ID per Gaussian [N].

  //============================================================================
  // Optimizer Interface - Vector wrappers required by the optimizer API.
  //============================================================================

  std::vector<torch::Tensor> Tensor_vec_xyz_, Tensor_vec_feature_dc_,
      Tensor_vec_feature_rest_, Tensor_vec_opacity_, Tensor_vec_scaling_,
      Tensor_vec_rotation_;

  std::shared_ptr<SparseGaussianAdam> optimizer_;  ///< Sparse Adam optimizer.
  float spatial_lr_scale_;  ///< Scale factor for position learning rate.

 protected:
  int local_iteration_;      ///< Current local training iteration.
  float position_lr_init_;   ///< Initial position learning rate.
  float position_lr_decay_;  ///< Per-step decay factor for position LR.
  float position_lr_min_;    ///< Minimum position learning rate.

  torch::Tensor position_lrs_;  ///< Per-Gaussian position learning rates [N].

  std::mutex mutex_settings_;  ///< Mutex for thread-safe settings access.

 public:
  //============================================================================
  // Chunk-Based Visibility - Methods and data for spatial chunking and frustum
  // culling
  //============================================================================

  float chunk_size_;  ///< Spatial size of each chunk in world units.

  FrustumCullingCache
      gaussian_visibility_cache_;  ///< Cache for visibility queries.

  /**
   * @brief Determines which chunks are visible from a keyframe's frustum.
   * @param keyframe The camera keyframe for visibility testing.
   * @param use_cache Whether to use cached visibility results.
   * @return Vector of visible chunk coordinates.
   */
  std::vector<ChunkCoord> frustumCullChunks(
      std::shared_ptr<GaussianKeyframe> keyframe,
      bool use_cache);

  /**
   * @brief Creates a mask of Gaussians visible from a keyframe.
   * @param keyframe The camera keyframe for visibility testing.
   * @param manage_memory If true, loads/evicts chunks as needed.
   * @return Boolean mask [N] indicating visible Gaussians.
   *
   * Performs frustum culling at chunk level, then creates a mask
   * for all Gaussians in visible chunks.
   */
  torch::Tensor cullVisibleGaussians(std::shared_ptr<GaussianKeyframe> keyframe,
                                     bool manage_memory = true);

  /**
   * @brief Creates a Gaussian mask from a set of chunk IDs.
   * @param visible_chunk_ids Tensor of chunk IDs to include.
   * @return Boolean mask [N] for Gaussians in those chunks.
   */
  torch::Tensor createGaussianMaskFromChunks(
      const torch::Tensor& visible_chunk_ids);

  /**
   * @brief Prunes Gaussians with low opacity or excessive screen size.
   * @param pkf Keyframe used for screen-size calculation.
   * @param visible_gaussian_mask Mask of Gaussians to consider.
   */
  void pruneLowOpacityGaussians(std::shared_ptr<GaussianKeyframe> pkf,
                                const torch::Tensor& visible_gaussian_mask);

  /**
   * @brief Recomputes chunk IDs based on current Gaussian positions.
   */
  void updateChunkIDs();

  bool is_initialized_ =
      false;  ///< Whether the model has been initialized with points.

  //============================================================================
  // Point Management - Methods for adding new Gaussians from SLAM observations.
  //============================================================================

  /**
   * @brief Adds new Gaussians from observed 3D points.
   * @param new_xyz Point positions [M, 3].
   * @param new_colors Point colors [M, 3] in RGB [0, 1].
   * @param new_scales Initial scale values [M, 3].
   * @param new_opacities Initial opacity values [M, 1].
   * @param iteration Current training iteration.
   * @param spatial_lr_scale Scale factor for position learning rate.
   *
   * Filters points by chunk density, loads affected disk chunks,
   * then either initializes or appends depending on model state.
   */
  void addPoints(const torch::Tensor& new_xyz,
                 const torch::Tensor& new_colors,
                 const torch::Tensor& new_scales,
                 const torch::Tensor& new_opacities,
                 int iteration,
                 float spatial_lr_scale);

  /**
   * @brief Initializes the model with the first set of points.
   * @param initial_xyz Point positions [N, 3].
   * @param initial_colors Point colors [N, 3] in RGB [0, 1].
   * @param initial_scales Initial scale values [N, 3].
   * @param initial_opacities Initial opacity values [N, 1].
   * @param iteration Current training iteration.
   *
   * Converts colors to spherical harmonics, initializes rotations
   * to identity, and sets up all parameter tensors.
   */
  void initializeFromPoints(const torch::Tensor& initial_xyz,
                            const torch::Tensor& initial_colors,
                            const torch::Tensor& initial_scales,
                            const torch::Tensor& initial_opacities,
                            int iteration);

  /**
   * @brief Appends additional points to an initialized model.
   * @param new_xyz Point positions [M, 3].
   * @param new_colors Point colors [M, 3] in RGB [0, 1].
   * @param new_scales Initial scale values [M, 3].
   * @param new_opacities Initial opacity values [M, 1].
   * @param iteration Current training iteration.
   */
  void appendPoints(const torch::Tensor& new_xyz,
                    const torch::Tensor& new_colors,
                    const torch::Tensor& new_scales,
                    const torch::Tensor& new_opacities,
                    int iteration);

  //============================================================================
  // Storage Tracking -  Tensors tracking chunk state between disk and memory
  //============================================================================

  torch::Tensor
      chunks_loaded_from_disk_;   ///< IDs of chunks currently loaded from disk.
  torch::Tensor chunks_on_disk_;  ///< IDs of all chunks saved to disk.
  torch::Tensor chunk_gaussian_counts_;  ///< Gaussian count per disk chunk.
  torch::Tensor gaussian_ids_;  ///< Unique ID per Gaussian for tracking.

  int64_t next_gaussian_id_ =
      0;  ///< Counter for generating unique Gaussian IDs.

  std::string storage_base_path_;  ///< Directory for chunk file storage.

  //============================================================================
  // Memory Management -  Configuration and state for GPU memory management
  //============================================================================

  int64_t max_gaussians_in_memory_ =
      3000000;  ///< Max Gaussians before eviction.
  std::chrono::steady_clock::time_point
      last_memory_check_;  ///< Rate-limiting for checks.

  std::unordered_map<int64_t, float>
      chunk_access_times_;  ///< Per-chunk access timestamps.
  int new_gaussian_chunk_density_ =
      100;  ///< Min Gaussians/chunk for new points.

  /**
   * @brief Queries current GPU memory usage via CUDA allocator.
   * @return Current allocated bytes on GPU.
   */
  size_t getCurrentGPUMemoryUsage() const;

  //============================================================================
  // Disk Storage Structures
  //============================================================================

  /**
   * @brief Container for all data associated with a spatial chunk.
   *
   * Holds Gaussian parameters and optimizer state for serialization.
   */
  struct ChunkData {
    // Gaussian parameters
    torch::Tensor xyz, features_dc, features_rest;
    torch::Tensor scaling, rotation, opacity;
    torch::Tensor exist_since, position_lrs, gaussian_ids;

    // Adam optimizer states (one per parameter group)
    std::vector<torch::Tensor> exp_avg_states;  ///< First moment estimates [6].
    std::vector<torch::Tensor>
        exp_avg_sq_states;             ///< Second moment estimates [6].
    std::vector<int64_t> step_counts;  ///< Adam step counts [6].

    int num_points;    ///< Number of Gaussians in this chunk.
    int64_t chunk_id;  ///< Encoded spatial coordinate ID.
  };

  /**
   * @brief Binary header for tensor serialization.
   *
   * Precedes tensor data in chunk files to enable reconstruction.
   */
  struct TensorHeader {
    uint32_t dims;       ///< Number of tensor dimensions.
    uint32_t sizes[8];   ///< Size in each dimension (max 8D).
    uint32_t dtype;      ///< torch::ScalarType as uint32_t.
    uint64_t data_size;  ///< Total data size in bytes.
  };

  //============================================================================
  // Disk I/O Methods - Methods for chunk serialization and persistence
  //============================================================================

  /**
   * @brief Writes a tensor to a binary file stream.
   * @param tensor Tensor to serialize.
   * @param file Output file stream.
   */
  void saveTensorBinary(const torch::Tensor& tensor, std::ofstream& file);

  /**
   * @brief Reads a tensor from a binary file stream.
   * @param file Input file stream.
   * @return Reconstructed tensor on the model's device.
   */
  torch::Tensor loadTensorBinary(std::ifstream& file);

  /**
   * @brief Generates filesystem path for a chunk's binary file.
   * @param coord Spatial chunk coordinate.
   * @return Full path to the chunk file.
   */
  std::string getChunkFilename(const ChunkCoord& coord);

  /**
   * @brief Loads chunks from disk into GPU memory.
   * @param chunk_ids_to_load Tensor of chunk IDs to load.
   *
   * Handles pre-emptive eviction if loading would exceed memory limits.
   * Uses parallel I/O for efficiency.
   */
  void loadChunks(const torch::Tensor& chunk_ids_to_load);

  /**
   * @brief Saves a single chunk to disk.
   * @param chunk_id The chunk's encoded spatial ID.
   * @param chunk_data All data for the chunk.
   */
  void saveSingleChunkToDisk(int64_t chunk_id, const ChunkData& chunk_data);

  /**
   * @brief Loads a single chunk from disk.
   * @param chunk_id The chunk's encoded spatial ID.
   * @return ChunkData if successful, nullopt otherwise.
   */
  std::optional<ChunkData> loadSingleChunkFromDisk(int64_t chunk_id);

  /**
   * @brief Appends loaded chunks to the model's tensors.
   * @param chunks_data Vector of loaded chunk data.
   * @param chunk_ids Corresponding chunk IDs.
   */
  void appendLoadedChunks(const std::vector<ChunkData>& chunks_data,
                          const std::vector<int64_t>& chunk_ids);

  /**
   * @brief Saves multiple chunks to disk in parallel.
   * @param chunk_ids_to_save Tensor of chunk IDs to save.
   */
  void saveChunks(const torch::Tensor& chunk_ids_to_save);

  /**
   * @brief Extracts chunk data from model tensors.
   * @param chunk_mask Boolean mask selecting Gaussians in the chunk.
   * @param chunk_id The chunk's encoded spatial ID.
   * @return ChunkData containing all parameters and optimizer state.
   */
  ChunkData extractChunkData(const torch::Tensor& chunk_mask, int64_t chunk_id);

  /**
   * @brief Saves chunks to disk and removes from GPU memory.
   * @param chunk_ids Tensor of chunk IDs to evict.
   *
   * Distinguishes between loaded, spillover, and new chunks for proper
   * handling.
   */
  void saveAndEvictChunks(const torch::Tensor& chunk_ids);

  //============================================================================
  // Memory Management Methods
  //============================================================================

  /**
   * @brief Selects chunks to evict based on LRU policy.
   * @param candidate_chunks Chunks that may be evicted.
   * @param target_gaussian_count Minimum Gaussians to free.
   * @return Tensor of chunk IDs to evict.
   */
  torch::Tensor findLRUChunks(const torch::Tensor& candidate_chunks,
                              int64_t target_gaussian_count);

  /**
   * @brief Checks if memory limit exceeded and evicts if needed.
   */
  void checkMemoryPressure();

  /**
   * @brief Updates access timestamps for chunks.
   * @param accessed_chunk_ids Chunks that were accessed.
   */
  void updateChunkAccess(const torch::Tensor& accessed_chunk_ids);

  /**
   * @brief Saves all in-memory chunks to disk.
   *
   * Used for checkpointing or shutdown. Skips spillover chunks.
   */
  void saveAllChunks();

  /**
   * @brief Counts total Gaussians across memory and disk.
   * @return Total Gaussian count.
   */
  int64_t countAllGaussians();

  /**
   * @brief Filters points to exclude sparse chunks.
   * @param xyz Point positions.
   * @param colors Point colors.
   * @param scales Point scales.
   * @param opacities Point opacities.
   * @param min_gaussians_per_chunk Minimum points required per chunk.
   * @return Tuple of filtered tensors.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  filterPointsByChunkDensity(const torch::Tensor& xyz,
                             const torch::Tensor& colors,
                             const torch::Tensor& scales,
                             const torch::Tensor& opacities,
                             int min_gaussians_per_chunk);

  /**
   * @brief Initializes an empty model for subsequent chunk loading.
   * @param spatial_lr_scale Scale factor for position learning rate.
   */
  void initializeEmpty(float spatial_lr_scale);

  /**
   * @brief Deletes chunks with too few Gaussians.
   * @param min_gaussians_per_chunk Threshold for deletion.
   */
  void deleteSparseChunks(int min_gaussians_per_chunk);

  /**
   * @brief Removes chunk files from disk.
   * @param chunk_ids Tensor of chunk IDs whose files to delete.
   */
  void deleteSparseChunkFiles(const torch::Tensor& chunk_ids);

  /**
   * @brief Handles Gaussians that moved between chunks after loop closure.
   * @param processed_chunk_ids Chunks that were recently optimized.
   *
   * Recomputes chunk assignments and loads destination chunks to avoid
   * spillover.
   */
  void handleBatchChunkRedistribution(const torch::Tensor& processed_chunk_ids);

  /**
   * @brief Prunes Gaussians by opacity and world-space size.
   * @param min_opacity Minimum opacity threshold.
   * @param extent Scene extent for size calculation.
   * @param max_screen_size Maximum allowed screen-space size (0 to disable).
   */
  void prune(float min_opacity, float extent, int max_screen_size);
};