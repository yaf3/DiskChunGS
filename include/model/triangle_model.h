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
 * @file triangle_model.h
 * @brief 3D Gaussian Splatting model with disk-based chunk streaming.
 *
 * This class implements a 3D Gaussian Splatting model with support for
 * memory-efficient chunk-based storage. Triangles are organized into spatial
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
#include "scene/triangle_keyframe.h"
#include "scene/triangle_parameters.h"
#include "types.h"
#include "utils/general_utils.h"
#include "utils/sh_utils.h"
#include "utils/tensor_utils.h"

// Forward declaration to avoid circular dependency
class SparseTriangleAdam;

/**
 * @brief Macro to populate tensor vectors from individual tensors.
 *
 * Used to update the std::vector wrappers that the optimizer requires
 * after any operation that modifies the underlying tensors.
 */
#define TRIANGLE_MODEL_TENSORS_TO_VEC                                      \
  this->Tensor_vec_vertices_ = {this->vertices_};                          \
  this->Tensor_vec_feature_dc_ = {this->features_dc_};                     \
  this->Tensor_vec_feature_rest_ = {this->features_rest_};                 \
  this->Tensor_vec_vertex_weight_ = {this->vertex_weight_};

/**
 * @brief Macro to initialize all Triangle tensors as empty on the specified
 * device.
 * @param device_type The torch device (kCUDA or kCPU) for tensor allocation.
 */
#define TRIANGLE_MODEL_INIT_TENSORS(device_type)                                     \
  this->vertices_ =                                                                  \
      torch::empty(0, torch::TensorOptions().device(device_type));                   \
  this->triangle_indices_ =                                                          \
      torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device_type)); \
  this->features_dc_ =                                                               \
      torch::empty(0, torch::TensorOptions().device(device_type));                   \
  this->features_rest_ =                                                             \
      torch::empty(0, torch::TensorOptions().device(device_type));                   \
  this->vertex_weight_ =                                                             \
      torch::empty(0, torch::TensorOptions().device(device_type));                   \
  TRIANGLE_MODEL_TENSORS_TO_VEC

/**
 * @class TriangleModel
 * @brief Manages a collection of 3D Triangles with disk-based chunk streaming.
 *
 * The TriangleModel represents a scene as a set of 3D Triangle primitives,
 * each parameterized by position (xyz), color (spherical harmonics features),
 * opacity, scale, and rotation. The model supports:
 *
 * - **Chunk-based storage**: Triangles are spatially partitioned into chunks
 *   that can be independently loaded/saved to disk.
 * - **LRU eviction**: When GPU memory is constrained, least-recently-used
 *   chunks are automatically evicted to disk.
 * - **Frustum culling**: Only chunks visible to a camera are loaded.
 * - **Sparse optimization**: Per-Triangle learning rates with Adam optimizer.
 *
 * The model maintains optimizer state (Adam momentum) alongside Triangle
 * parameters, enabling seamless save/restore of training state.
 */
class TriangleModel {
 public:
  /// Number of optimizer parameter groups: vertices, features_dc,
  /// features_rest, vertex_weight.
  static constexpr int kNumParamGroups = 4;

  //============================================================================
  // Nested Types
  //============================================================================

  /**
   * @brief Container for all data associated with a spatial chunk.
   *
   * Holds Triangle parameters and optimizer state for serialization.
   */
  struct ChunkData {
    // Triangle parameters
    torch::Tensor vertices, triangle_indices, features_dc, features_rest;
    torch::Tensor vertex_weight;
    torch::Tensor exist_since, position_lrs, triangle_ids;

    // Adam optimizer states (one per parameter group)
    std::vector<torch::Tensor> exp_avg_states;  ///< First moment estimates [4].
    std::vector<torch::Tensor>
        exp_avg_sq_states;             ///< Second moment estimates [4].
    std::vector<int64_t> step_counts;  ///< Adam step counts [4].

    int num_points;    ///< Number of Triangles in this chunk.
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
  // Construction
  //============================================================================

  /**
   * @brief Constructs a TriangleModel with the given parameters.
   * @param model_params Configuration including SH degree, device, and memory
   * limits.
   * @param storage_base_path Directory path for chunk file storage.
   * @param chunk_size Spatial size of each chunk in world units.
   */
  explicit TriangleModel(const TriangleModelParams& model_params,
                         std::string storage_base_path = "",
                         float chunk_size = 20.0f);

  //============================================================================
  // Activation Getters
  //============================================================================

  /**
   * @brief Returns vertex positions.
   * @return Tensor of shape [V, 3].
   */
  torch::Tensor getVertices();

  /**
   * @brief Returns triangle index buffer.
   * @return Tensor of shape [T, 3], int32.
   */
  torch::Tensor getTriangleIndices();

  /**
   * @brief Assembles [T,3,3] from vertices_ indexed by triangle_indices_.
   * @return Tensor of shape [T, 3, 3] (T triangles, 3 vertices, 3D coords).
   */
  torch::Tensor getTrianglesPoints();

  /**
   * @brief Returns the current sigma scalar value.
   * @return Current sigma value.
   */
  float getSigmaActivation();

  /**
   * @brief Returns Triangle centroids (mean of 3 vertices).
   * @return Tensor of shape [N, 3] with XYZ centroid coordinates.
   */
  torch::Tensor getXYZ();

  /**
   * @brief Returns combined spherical harmonics features (DC + rest).
   * @return Tensor of concatenated SH coefficients.
   */
  torch::Tensor getFeatures();

  /**
   * @brief Returns sigmoid(vertex_weight_), per-vertex weights in (0,1).
   * @return Tensor of shape [V, 1] with values in (0, 1).
   */
  torch::Tensor getVertexWeightActivation();

  torch::Tensor inverseVertexWeightActivation(const torch::Tensor& y);

  void updateOpacityFloor(float new_floor);

  void runRestrictedDelaunay(int current_iter);
  void runRestrictedDelaunayForChunk(int64_t chunk_id, int current_iter);


  //============================================================================
  // Geometric Transformations
  //============================================================================

  /**
   * @brief Applies a scaled rigid transformation to all Triangles.
   * @param s Scale factor applied before transformation.
   * @param T SE3 transformation (rotation + translation).
   *
   * Transforms positions as: xyz' = s * R * xyz + t
   * Also scales the Triangle scaling parameters accordingly.
   */
  void applyScaledTransformation(
      const float s = 1.0,
      const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                          Eigen::Vector3f::Zero()));

  /**
   * @brief Updates optimizer state after transformation.
   * @param new_vertices Transformed vertex positions tensor [V,3].
   *
   * Replaces the optimizer's tracked tensors and resets their Adam states.
   */
  void scaledTransformationPostfix(torch::Tensor& new_vertices);

  /**
   * @brief Transforms Triangles visible to a keyframe after pose update.
   * @param point_transformed_flags Output mask of which points were
   * transformed.
   * @param diff_pose Differential pose change.
   * @param kf_world_view_transform Keyframe's world-to-view transform.
   * @param kf_full_proj_transform Keyframe's full projection matrix.
   * @param kf_creation_iter Iteration when keyframe was created.
   * @param stable_num_iter_existence Threshold for "stable" Triangles.
   * @param num_transformed Output count of transformed Triangles.
   * @param scale Optional scale factor.
   *
   * Only transforms "unstable" Triangles (recently created near the keyframe).
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
  // Optimization
  //============================================================================

  /**
   * @brief Initializes the optimizer with per-parameter learning rates.
   * @param training_args Learning rate configuration for each parameter group.
   *
   * Sets up a SparseTriangleAdam optimizer with 5 parameter groups:
   * triangles_points (0), features_dc (1), features_rest (2), opacity (3),
   * sigma (4). Position learning rates are per-Triangle; others use scalar LRs.
   */
  void trainingSetup(const TriangleOptimizationParams& training_args);

  /**
   * @brief Decays position learning rates for visible Triangles.
   * @param visibility Boolean mask indicating which Triangles were rendered.
   *
   * Applies exponential decay to per-Triangle position learning rates,
   * clamped to a minimum value.
   */
  void updateLearningRates(const torch::Tensor& visibility);

  /**
   * @brief Performs sparse Adam update for vertices of visible triangles.
   * @param triangle_visibility Boolean mask [T] of visible triangles.
   *
   * Maps triangle visibility to per-vertex visibility via triangle_indices_,
   * then runs sparse Adam on all 4 vertex-level param groups.
   */
  void optimizerStep(torch::Tensor& triangle_visibility);

  /**
   * @brief Resets opacity of all Triangles to a low value.
   *
   * Used periodically during training to cull Triangles that don't
   * recover their opacity (indicating they're not needed).
   */
  void resetVertexWeight();

  /**
   * @brief Resets vertex weight for a subset of Triangles.
   * @param triangle_mask Boolean mask selecting Triangles to reset.
   */
  void resetVertexWeightForMask(const torch::Tensor& triangle_mask);

  /**
   * @brief Resets position learning rates and Adam momentum for selected
   * Triangles.
   * @param triangle_mask Boolean mask selecting Triangles to reset.
   *
   * Used after loop closure to allow affected Triangles to move freely again.
   */
  void resetPositionLRAndOptimizerState(const torch::Tensor& triangle_mask);

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
  // Pruning and Densification
  //============================================================================

  /**
   * @brief Removes Triangles indicated by the mask.
   * @param mask Boolean tensor where true indicates points to remove.
   *
   * Updates all parameter tensors and optimizer states accordingly.
   */
  void prunePoints(torch::Tensor& mask);

  /**
   * @brief Appends new Triangles to the model with optimizer state.
   * @param new_xyz Positions of new Triangles [M, 3].
   * @param new_features_dc DC spherical harmonics coefficients.
   * @param new_features_rest Higher-order SH coefficients.
   * @param new_opacities Opacity values (pre-sigmoid).
   * @param new_scaling Scale values (pre-exp).
   * @param new_rotation Rotation quaternions.
   * @param new_exist_since_iter Iteration when each Triangle was created.
   * @param new_chunk_ids Spatial chunk assignment for each Triangle.
   * @param new_position_lrs Per-Triangle position learning rates.
   * @param new_triangle_ids Unique IDs for each Triangle.
   * @param loaded_exp_avg Optional: Adam first moment from disk.
   * @param loaded_exp_avg_sq Optional: Adam second moment from disk.
   * @param loaded_step_counts Optional: Adam step counts from disk.
   *
   * Concatenates new Triangles to existing tensors and updates optimizer state.
   * Used both for densification and loading chunks from disk.
   */
  void densificationPostfix(
      torch::Tensor& new_vertices,
      torch::Tensor& new_triangle_indices,
      torch::Tensor& new_features_dc,
      torch::Tensor& new_features_rest,
      torch::Tensor& new_vertex_weight,
      torch::Tensor& new_exist_since_iter,
      torch::Tensor& new_chunk_ids,
      torch::Tensor& new_position_lrs,
      torch::Tensor& new_triangle_ids,
      const std::vector<torch::Tensor>& loaded_exp_avg = {},
      const std::vector<torch::Tensor>& loaded_exp_avg_sq = {},
      const std::vector<int64_t>& loaded_step_counts = {});

  /**
   * @brief Prunes Triangles with low opacity or excessive screen size.
   * @param pkf Keyframe used for context (image dimensions).
   * @param visible_triangle_mask Mask of Triangles to consider.
   * @param full_model_scaling Kernel-computed screen extent per triangle [N],
   *        0 for triangles not in the current frustum.
   */
  void pruneLowWeightTriangles(std::shared_ptr<TriangleKeyframe> pkf,
                                const torch::Tensor& visible_triangle_mask,
                                const torch::Tensor& full_model_scaling);

  //============================================================================
  // Point Management
  //============================================================================

  /**
   * @brief Adds new Triangles from observed 3D points.
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
                 float spatial_lr_scale,
                 const torch::Tensor& cam_center = {},
                 const torch::Tensor& normals = {});

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
                            int iteration,
                            const torch::Tensor& cam_center = {},
                            const torch::Tensor& normals = {});

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
                    int iteration,
                    const torch::Tensor& cam_center = {},
                    const torch::Tensor& normals = {});

  /**
   * @brief Initializes an empty model for subsequent chunk loading.
   * @param spatial_lr_scale Scale factor for position learning rate.
   */
  void initializeEmpty(float spatial_lr_scale);

  //============================================================================
  // Chunk Visibility and Frustum Culling
  //============================================================================

  /**
   * @brief Determines which chunks are visible from a keyframe's frustum.
   * @param keyframe The camera keyframe for visibility testing.
   * @param use_cache Whether to use cached visibility results.
   * @return Vector of visible chunk coordinates.
   */
  std::vector<ChunkCoord> frustumCullChunks(
      std::shared_ptr<TriangleKeyframe> keyframe,
      bool use_cache);

  /**
   * @brief Creates a mask of Triangles visible from a keyframe.
   * @param keyframe The camera keyframe for visibility testing.
   * @param manage_memory If true, loads/evicts chunks as needed.
   * @return Boolean mask [N] indicating visible Triangles.
   *
   * Performs frustum culling at chunk level, then creates a mask
   * for all Triangles in visible chunks.
   */
  torch::Tensor cullVisibleTriangles(std::shared_ptr<TriangleKeyframe> keyframe,
                                     bool manage_memory = true);

  /**
   * @brief Creates a Triangle mask from a set of chunk IDs.
   * @param visible_chunk_ids Tensor of chunk IDs to include.
   * @return Boolean mask [N] for Triangles in those chunks.
   */
  torch::Tensor createTriangleMaskFromChunks(
      const torch::Tensor& visible_chunk_ids);

  /**
   * @brief Recomputes chunk IDs based on current Triangle positions.
   */
  void updateChunkIDs();

  void incrementChunkOptCounts();
  int getChunkOptCount(int64_t chunk_id) const;
  const torch::Tensor& getLastVisibleChunkIds() const {
    return last_visible_chunk_ids_;
  }

  //============================================================================
  // Disk I/O
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
   * @param chunk_mask Boolean mask selecting Triangles in the chunk.
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

  /**
   * @brief Saves all in-memory chunks to disk.
   *
   * Used for checkpointing or shutdown. Skips spillover chunks.
   */
  void saveAllChunks();

  /**
   * @brief Removes chunk files from disk.
   * @param chunk_ids Tensor of chunk IDs whose files to delete.
   */
  void deleteSparseChunkFiles(const torch::Tensor& chunk_ids);

  //============================================================================
  // Memory Management
  //============================================================================

  /**
   * @brief Queries current GPU memory usage via CUDA allocator.
   * @return Current allocated bytes on GPU.
   */
  size_t getCurrentGPUMemoryUsage() const;

  /**
   * @brief Selects chunks to evict based on LRU policy.
   * @param candidate_chunks Chunks that may be evicted.
   * @param target_triangle_count Minimum Triangles to free.
   * @return Tensor of chunk IDs to evict.
   */
  torch::Tensor findLRUChunks(const torch::Tensor& candidate_chunks,
                              int64_t target_triangle_count);

  /**
   * @brief Checks if memory limit exceeded and evicts if needed.
   */
  void checkMemoryPressure();

  /**
   * @brief Evicts LRU chunks until excess Triangles are freed.
   * @param protected_chunk_ids Chunk IDs that must not be evicted.
   * @param excess_triangles Minimum number of Triangles to free.
   *
   * Applies a 5% hysteresis buffer on top of the requested eviction amount
   * to reduce eviction frequency.
   */
  void evictExcessChunks(const torch::Tensor& protected_chunk_ids,
                         int64_t excess_triangles);

  /**
   * @brief Computes exact Triangle count for chunks to be loaded from disk.
   * @param chunks_ids_needing_load Chunk IDs to look up.
   * @return Total number of Triangles across the requested chunks.
   */
  int64_t countTrianglesToLoad(const torch::Tensor& chunks_ids_needing_load);

  /**
   * @brief Updates access timestamps for chunks.
   * @param accessed_chunk_ids Chunks that were accessed.
   */
  void updateChunkAccess(const torch::Tensor& accessed_chunk_ids);

  /**
   * @brief Counts total Triangles across memory and disk.
   * @return Total Triangle count.
   */
  int64_t countAllTriangles();

  /**
   * @brief Filters points to exclude sparse chunks.
   * @param xyz Point positions.
   * @param colors Point colors.
   * @param scales Point scales.
   * @param opacities Point opacities.
   * @param min_triangles_per_chunk Minimum points required per chunk.
   * @return Tuple of filtered tensors.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  filterPointsByChunkDensity(const torch::Tensor& xyz,
                             const torch::Tensor& colors,
                             const torch::Tensor& scales,
                             const torch::Tensor& opacities,
                             int min_triangles_per_chunk,
                             const torch::Tensor& normals = {});

  /**
   * @brief Deletes chunks with too few Triangles.
   * @param min_triangles_per_chunk Threshold for deletion.
   */
  void deleteSparseChunks(int min_triangles_per_chunk);

  /**
   * @brief Handles Triangles that moved between chunks after loop closure.
   * @param processed_chunk_ids Chunks that were recently optimized.
   *
   * Recomputes chunk assignments and loads destination chunks to avoid
   * spillover.
   */
  void handleBatchChunkRedistribution(const torch::Tensor& processed_chunk_ids);

  //============================================================================
  // Public Data Members
  //============================================================================

  // Device and configuration
  torch::DeviceType device_type_;  ///< CUDA or CPU device for tensors.
  int sh_degree_;                  ///< Spherical harmonics degree (0-3).

  // Core Triangle parameters
  torch::Tensor vertices_;            ///< Vertex positions [V, 3].
  torch::Tensor triangle_indices_;    ///< Triangle vertex indices [T, 3], int32.
  torch::Tensor features_dc_;         ///< DC SH coefficients [V, 1, 3].
  torch::Tensor features_rest_;       ///< Higher-order SH coefficients [V, K, 3].
  torch::Tensor vertex_weight_;       ///< Logit-space per-vertex weight [V, 1].
  float sigma_value_ = 0.0f;          ///< Current sigma (scheduled, not learnable).
  float opacity_floor_ = 0.0f;        ///< Current opacity floor for weight activation.
  torch::Tensor exist_since_iter_;  ///< Creation iteration per Triangle [N].
  torch::Tensor triangle_chunk_ids_;  ///< Spatial chunk ID per Triangle [N].

  // Optimizer interface (vector wrappers required by optimizer API)
  std::vector<torch::Tensor> Tensor_vec_vertices_,
      Tensor_vec_feature_dc_, Tensor_vec_feature_rest_,
      Tensor_vec_vertex_weight_;

  std::shared_ptr<SparseTriangleAdam> optimizer_;  ///< Sparse Adam optimizer.
  float spatial_lr_scale_;  ///< Scale factor for position learning rate.

  // Chunk visibility
  float chunk_size_;  ///< Spatial size of each chunk in world units.
  FrustumCullingCache
      triangle_visibility_cache_;  ///< Cache for visibility queries.
  bool is_initialized_ =
      false;  ///< Whether the model has been initialized with points.

  // Storage tracking
  torch::Tensor
      chunks_loaded_from_disk_;   ///< IDs of chunks currently loaded from disk.
  torch::Tensor chunks_on_disk_;  ///< IDs of all chunks saved to disk.
  torch::Tensor chunk_triangle_counts_;  ///< Triangle count per disk chunk.
  torch::Tensor triangle_ids_;  ///< Unique ID per Triangle for tracking.
  int64_t next_triangle_id_ =
      0;  ///< Counter for generating unique Triangle IDs.
  std::string storage_base_path_;  ///< Directory for chunk file storage.

  // Memory management configuration
  int64_t max_triangles_in_memory_ =
      3000000;  ///< Max Triangles before eviction.
  std::unordered_map<int64_t, float>
      chunk_access_times_;  ///< Per-chunk access timestamps.
  std::unordered_map<int64_t, int>
      chunk_opt_counts_;              ///< Per-chunk training step count.
  torch::Tensor last_visible_chunk_ids_;  ///< Cached from last cullVisibleTriangles.
  int new_triangle_chunk_density_ =
      100;  ///< Min Triangles/chunk for new points.

 private:
  //============================================================================
  // Private Helpers
  //============================================================================

  /**
   * @brief Assigns optimized tensors back to member variables after
   * pruning/densification.
   * @param tensors Vector of 4 tensors corresponding to the parameter groups.
   *
   * Updates vertices_, features_dc_, features_rest_, vertex_weight_
   * and their optimizer vector wrappers.
   */
  void assignOptimizedTensors(const std::vector<torch::Tensor>& tensors);

  /**
   * @brief Removes unreferenced vertices after triangle pruning.
   *
   * Finds vertices not referenced by any triangle in triangle_indices_,
   * remaps indices to be contiguous, and prunes vertex-level data.
   */
  void gcUnreferencedVertices();

  /**
   * @brief Prunes vertex-level optimizer state and parameters.
   * @param vertex_keep_mask Boolean mask [V] — true for vertices to keep.
   */
  void pruneVertexData(const torch::Tensor& vertex_keep_mask);

 protected:
  //============================================================================
  // Protected Members
  //============================================================================

  /**
   * @brief Computes exponentially decaying learning rate.
   * @param step Current optimization step.
   * @return Learning rate value.
   */
  float exponLrFunc(int step);

  int local_iteration_;      ///< Current local training iteration.
  float position_lr_init_;   ///< Initial position learning rate.
  float position_lr_decay_;  ///< Per-step decay factor for position LR.
  float position_lr_min_;    ///< Minimum position learning rate.

  torch::Tensor position_lrs_;  ///< Per-Triangle position learning rates [N].

  std::mutex mutex_settings_;  ///< Mutex for thread-safe settings access.
};