/**
 * This file is part of DiskChunGS, modified from CaRtGS/Photo-SLAM.
 *
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See the GNU General Public License for more details:
 * <http://www.gnu.org/licenses/>.
 */

#include "triangle_mapper.h"
#include "rendering/triangle_renderer.h"
#include "utils/loss_utils.h"
#include "utils/profiling.h"

// ============================================================================
// Helper Methods
// ============================================================================

bool TriangleMapper::isPoseDivergenceLarge(
    const Sophus::SE3f &diff_pose) const {
  bool large_rot = !diff_pose.rotationMatrix().isApprox(
      Eigen::Matrix3f::Identity(), large_rot_th_);
  bool large_trans =
      !diff_pose.translation().isMuchSmallerThan(1.0, large_trans_th_);
  return large_rot || large_trans;
}

torch::Tensor TriangleMapper::filterRelevantChunks(
    const torch::Tensor &visible_chunk_ids) const {
  torch::Tensor loaded_mask =
      torch::isin(visible_chunk_ids, triangles_->chunks_loaded_from_disk_);
  torch::Tensor on_disk_mask =
      torch::isin(visible_chunk_ids, triangles_->chunks_on_disk_);
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(triangles_->triangle_chunk_ids_));
  torch::Tensor has_triangles_mask =
      torch::isin(visible_chunk_ids, spatial_chunks);

  torch::Tensor relevant_mask = loaded_mask | on_disk_mask | has_triangles_mask;
  return visible_chunk_ids.index({relevant_mask});
}

// ============================================================================
// Mapping Operations Processing
// ============================================================================

void TriangleMapper::combineMappingOperations() {
  // Group operations by type
  std::vector<ORB_SLAM3::MappingOperation> localBAOps;
  std::vector<ORB_SLAM3::MappingOperation> loopClosureOps;
  std::vector<ORB_SLAM3::MappingOperation> scaleRefinementOps;

  while (pSLAM_->getAtlas()->hasMappingOperation()) {
    ORB_SLAM3::MappingOperation opr =
        pSLAM_->getAtlas()->getAndPopMappingOperation();

    switch (opr.meOperationType) {
      case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        localBAOps.push_back(std::move(opr));
        break;
      case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        loopClosureOps.push_back(std::move(opr));
        break;
      case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        scaleRefinementOps.push_back(std::move(opr));
        break;
      default:
        throw std::runtime_error("MappingOperation type not supported!");
    }
  }

  // Process local BA operations as a batch
  if (!localBAOps.empty()) {
    processLocalMappingBABatch(localBAOps);
  }

  // Process loop closure operations (more complex, less frequent)
  for (auto &opr : loopClosureOps) {
    auto timer_loopClosure = ProfilingUtils::Timer("processLoopClosure");

    // Pause image ingestion during loop closure
    std::cout << "[Loop Closure] ========================================\n"
              << "[Loop Closure] PAUSING ORB-SLAM3 image ingestion...\n"
              << "[Loop Closure] ========================================"
              << std::endl;
    pause_image_ingestion_.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    processLoopClosureBA(opr);

    // Run extended optimization for loop closure affected areas
    std::cout << "\n[Loop Closure] ========================================\n"
              << "[Loop Closure] Running extended optimization for "
              << loop_closure_optimization_iterations_ << " iterations...\n"
              << "[Loop Closure] ========================================\n"
              << std::endl;

    for (int i = 0; i < loop_closure_optimization_iterations_; i++) {
      trainForOneIteration();
      if ((i + 1) % 100 == 0 || i == 0) {
        std::cout << "[Loop Closure Optimization] Iteration " << (i + 1) << "/"
                  << loop_closure_optimization_iterations_
                  << " (EMA Loss: " << ema_loss_for_log_ << ")" << std::endl;
      }
    }

    // Resume image ingestion
    std::cout << "\n[Loop Closure] ========================================\n"
              << "[Loop Closure] Optimization complete. RESUMING ORB-SLAM3...\n"
              << "[Loop Closure] ========================================\n"
              << std::endl;
    pause_image_ingestion_.store(false, std::memory_order_release);

    timer_loopClosure.stop();
  }

  // Process scale refinement operations
  for (auto &opr : scaleRefinementOps) {
    auto timer_scaleRefinement =
        ProfilingUtils::Timer("processScaleRefinement");
    processScaleRefinement(opr);
    timer_scaleRefinement.stop();
  }
}

void TriangleMapper::processLocalMappingBABatch(
    std::vector<ORB_SLAM3::MappingOperation> &operations) {
  if (operations.empty()) return;

  std::map<std::size_t, std::shared_ptr<TriangleKeyframe>>
      associated_keyframe_map;

  for (auto &opr : operations) {
    auto &associated_kfs = opr.associatedKeyFrames();

    for (auto &kf : associated_kfs) {
      auto kfid = std::get<0>(kf);
      std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);

      if (pkf) {
        auto &orb_pose = std::get<2>(kf);

        // If pose optimization disabled, always use ORB-SLAM poses
        if (opt_params_.pose_lr_ <= 0.0f) {
          pkf->setPose(orb_pose.unit_quaternion().cast<double>(),
                       orb_pose.translation().cast<double>());
        } else {
          // Selectively update if poses have diverged significantly
          Sophus::SE3f triangle_pose = pkf->getPosef();
          Sophus::SE3f diff_pose = orb_pose.inverse() * triangle_pose;

          constexpr float kRotationThreshold = 0.1f;
          constexpr float kTranslationThreshold = 0.05f;
          bool large_divergence =
              !diff_pose.rotationMatrix().isApprox(Eigen::Matrix3f::Identity(),
                                                   kRotationThreshold) ||
              !diff_pose.translation().isMuchSmallerThan(1.0,
                                                         kTranslationThreshold);

          if (large_divergence) {
            pkf->setPose(orb_pose.unit_quaternion().cast<double>(),
                         orb_pose.translation().cast<double>());
          }
        }
        pkf->computeTransformTensors();
      } else {
        handleNewKeyframeFromORBSLAM(kf);
      }
    }
  }
}

// ============================================================================
// Loop Closure Processing
// ============================================================================

void TriangleMapper::processLoopClosureBA(ORB_SLAM3::MappingOperation &opr) {
  std::cout << "[Loop Closure] Starting with scale factor: " << opr.mfScale
            << std::endl;

  float loop_kf_scale = opr.mfScale;
  auto &associated_kfs = opr.associatedKeyFrames();
  std::cout << "[Loop Closure] Processing " << associated_kfs.size()
            << " keyframes" << std::endl;

  auto time_start = std::chrono::steady_clock::now();

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_0_before_loop_correction") /
              "data");
  }

  // First pass: Handle new keyframes
  for (auto &kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);
    if (!pkf) {
      std::cout << "[Loop Closure] New frame in loop-closure" << std::endl;
      handleNewKeyframeFromORBSLAM(kf);
    }
  }

  std::unique_lock<std::mutex> lock_render(mutex_render_);

  // Estimate total triangles needed and collect keyframe-chunk pairs
  int64_t total_triangles_needed = 0;
  std::vector<std::pair<std::shared_ptr<TriangleKeyframe>, torch::Tensor>>
      kf_chunk_pairs;
  std::unordered_set<int64_t> all_unique_chunks;

  for (auto &kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);
    if (!pkf) continue;

    auto &pose = std::get<2>(kf);
    Sophus::SE3f original_pose = pkf->getPosef();
    Sophus::SE3f diff_pose = pose.inverse() * original_pose;

    if (isPoseDivergenceLarge(diff_pose)) {
      std::vector<ChunkCoord> visible_chunk_coords =
          triangles_->frustumCullChunks(pkf, /*use_cache=*/true);

      torch::Tensor visible_chunk_coords_tensor = chunkCoordVectorToTensor(
          visible_chunk_coords, triangles_->device_type_);
      torch::Tensor visible_chunk_ids =
          encodeChunkCoordsTensor(visible_chunk_coords_tensor);

      torch::Tensor relevant_chunk_ids =
          filterRelevantChunks(visible_chunk_ids);

      if (relevant_chunk_ids.size(0) > 0) {
        kf_chunk_pairs.emplace_back(pkf, relevant_chunk_ids);

        auto chunk_ids_cpu = relevant_chunk_ids.cpu();
        auto accessor = chunk_ids_cpu.accessor<int64_t, 1>();
        for (int i = 0; i < chunk_ids_cpu.size(0); ++i) {
          all_unique_chunks.insert(accessor[i]);
        }
      }
    }
  }

  // Save and evict all current chunks for clean memory calculation
  torch::Tensor all_spatial_chunks =
      std::get<0>(torch::_unique2(triangles_->triangle_chunk_ids_));
  if (all_spatial_chunks.size(0) > 0) {
    std::cout << "[Loop Closure] Saving and evicting "
              << all_spatial_chunks.size(0) << " chunks" << std::endl;
    triangles_->saveAndEvictChunks(all_spatial_chunks);
  }

  // Calculate triangle count for needed chunks
  for (int64_t chunk_id : all_unique_chunks) {
    torch::Tensor chunk_id_tensor =
        torch::tensor({chunk_id}, torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(triangles_->device_type_));

    // Check if chunk is on disk and get its triangle count
    auto disk_mask = torch::eq(triangles_->chunks_on_disk_, chunk_id_tensor);
    if (torch::any(disk_mask).item<bool>()) {
      auto indices = torch::where(disk_mask)[0];
      if (indices.size(0) > 0) {
        int64_t count =
            triangles_->chunk_vertex_counts_[indices[0].item<int64_t>()]
                .item<int64_t>();
        total_triangles_needed += count;
      }
    }
    // Note: No else case needed since we evicted all spatial chunks above
  }

  int64_t current_triangles = triangles_->triangle_indices_.size(0);
  int64_t projected_total = total_triangles_needed + current_triangles;

  // =========================================================================
  // Loop Closure Memory Management Strategy:
  //
  // During loop closure, we temporarily increase the triangle memory limit to
  // allow processing more triangles than during normal operation. This is safe
  // because we don't render during loop closure (no VRAM needed for rendering).
  //
  // The increased limit is: original_limit * loop_closure_memory_multiplier_
  // (default multiplier is 8x, configurable via
  // loop_closure_memory_multiplier_)
  //
  // Strategy selection:
  // - If projected_total <= increased_limit: Use BATCHED processing
  //   (loads all affected chunks at once, faster but uses more memory)
  // - If projected_total > increased_limit: Use SEQUENTIAL processing
  //   (processes one keyframe at a time, stays under memory limit)
  //
  // The limit is always restored to the original value at the end.
  // =========================================================================

  int64_t original_limit = triangles_->max_vertices_in_memory_;
  int64_t increased_limit =
      static_cast<int64_t>(original_limit * loop_closure_memory_multiplier_);

  std::cout << "[Loop Closure] Triangle estimation - Current: "
            << current_triangles
            << ", Additional needed: " << total_triangles_needed
            << ", Projected total: " << projected_total
            << ", Original limit: " << original_limit << ", Increased limit (x"
            << loop_closure_memory_multiplier_ << "): " << increased_limit
            << std::endl;

  // Temporarily increase the memory limit for loop closure processing
  triangles_->max_vertices_in_memory_ = increased_limit;

  for (const auto &[index, keyframe] : scene_->keyframes_) {
    if (keyframe->loaded_) keyframe->saveDataToDisk();
  }

  // Choose processing strategy based on projected memory usage
  int total_transformed = 0;
  bool use_batched = (projected_total <= increased_limit);

  if (use_batched) {
    std::cout << "[Loop Closure] Using BATCHED strategy - projected total ("
              << projected_total << ") fits within increased limit ("
              << increased_limit << ")" << std::endl;
    total_transformed = processBatchedLoopClosure(
        associated_kfs, kf_chunk_pairs, all_unique_chunks, loop_kf_scale);
  } else {
    std::cout << "[Loop Closure] Using SEQUENTIAL strategy - projected total ("
              << projected_total << ") exceeds increased limit ("
              << increased_limit << ")" << std::endl;
    total_transformed =
        processSequentialLoopClosure(associated_kfs, loop_kf_scale);
  }

  // Always restore the original memory limit
  triangles_->max_vertices_in_memory_ = original_limit;

  // Update chunk-keyframe mappings
  for (auto &kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);
    keyframe_selector_->updateChunkKeyframeMapping(pkf, false);
  }

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_1_after_loop_correction") /
              "data");
  }

  loop_closure_iteration_ = true;

  auto time_end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start)
          .count();
  std::cout << "[Loop Closure] Completed in " << duration
            << "s - Total triangles transformed: " << total_transformed
            << std::endl;
}

int TriangleMapper::processBatchedLoopClosure(
    std::vector<KeyframeTuple> &associated_kfs,
    const std::vector<std::pair<std::shared_ptr<TriangleKeyframe>,
                                torch::Tensor>> &kf_chunk_pairs,
    const std::unordered_set<int64_t> &all_unique_chunks,
    float loop_kf_scale) {
  int total_transformed = 0;

  // Batch load all required chunks at once
  if (!all_unique_chunks.empty()) {
    torch::Tensor all_chunk_ids = torch::empty(
        {static_cast<int64_t>(all_unique_chunks.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto accessor = all_chunk_ids.accessor<int64_t, 1>();
    int idx = 0;
    for (int64_t chunk_id : all_unique_chunks) {
      accessor[idx++] = chunk_id;
    }
    all_chunk_ids = all_chunk_ids.to(triangles_->device_type_);

    std::cout << "[Batched Loop] Loading " << all_unique_chunks.size()
              << " unique chunks" << std::endl;
    triangles_->loadChunks(all_chunk_ids);
  }

  // Global transform mask to track which triangles have been transformed
  torch::Tensor global_transform_mask = torch::zeros(
      {triangles_->triangle_indices_.size(0)}, torch::TensorOptions()
                                      .dtype(torch::kBool)
                                      .device(triangles_->device_type_));

  std::vector<torch::Tensor> chunks_to_redistribute;

  for (auto &kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);

    if (!pkf) continue;

    auto &pose = std::get<2>(kf);
    Sophus::SE3f original_pose = pkf->getPosef();
    Sophus::SE3f diff_pose = pose.inverse() * original_pose;

    // Handle loop closure keyframes (reset opacity for visible triangles)
    bool is_loop_closure_kf = std::get<4>(kf);
    if (is_loop_closure_kf) {
      std::cout << "[Batched Loop] Loop closure keyframe: " << kfid
                << std::endl;
      torch::Tensor visible_triangles =
          triangles_->cullVisibleTriangles(pkf, false);
      if (torch::any(visible_triangles).item<bool>()) {
        triangles_->resetPositionLRAndOptimizerState(visible_triangles);
      }
    }

    if (isPoseDivergenceLarge(diff_pose)) {
      auto it =
          std::find_if(kf_chunk_pairs.begin(), kf_chunk_pairs.end(),
                       [&](const auto &pair) { return pair.first == pkf; });

      if (it != kf_chunk_pairs.end()) {
        torch::Tensor relevant_chunk_ids = it->second;

        std::cout << "[Batched Loop] Large correction for kf" << kfid
                  << std::endl;

        torch::Tensor diff_pose_tensor =
            tensor_utils::EigenMatrix2TorchTensor(diff_pose.matrix(),
                                                  triangles_->device_type_)
                .transpose(0, 1);

        int triangles_transformed_by_this_kf = 0;
        float scale = 1.0f;  // Scale factor for transformation

        triangles_->scaledTransformVisiblePointsOfKeyframe(
            global_transform_mask, diff_pose_tensor, pkf->world_view_transform_,
            pkf->full_proj_transform_, pkf->creation_iter_,
            stableNumIterExistence(), triangles_transformed_by_this_kf, scale);

        total_transformed += triangles_transformed_by_this_kf;
        std::cout << "[Batched Loop] Keyframe " << kfid << " transformed "
                  << triangles_transformed_by_this_kf << " points" << std::endl;

        chunks_to_redistribute.push_back(relevant_chunk_ids);
        increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
      }
    }

    // Update keyframe pose
    pkf->setPose(pose.unit_quaternion().cast<double>(),
                 pose.translation().cast<double>());
    pkf->computeTransformTensors();
  }

  // Batch redistribute chunks
  if (!chunks_to_redistribute.empty()) {
    torch::Tensor all_redistrib_chunks = torch::cat(chunks_to_redistribute, 0);
    torch::Tensor unique_redistrib_chunks =
        std::get<0>(torch::_unique2(all_redistrib_chunks));

    std::cout << "[Batched Loop] Redistributing "
              << unique_redistrib_chunks.size(0) << " chunks" << std::endl;
    triangles_->handleBatchChunkRedistribution(unique_redistrib_chunks);
  }

  return total_transformed;
}

int TriangleMapper::processSequentialLoopClosure(
    const std::vector<KeyframeTuple> &associated_kfs,
    float loop_kf_scale) {
  int total_transformed = 0;

  int64_t total_triangle_count = triangles_->countAllTriangles();
  int64_t buffer_size = static_cast<int64_t>(total_triangle_count * 1.1);

  // Buffer to track which triangles have been transformed (by stable ID).
  // Used to avoid double-transforming triangles visible from multiple
  // keyframes. We use IDs rather than indices because indices change as chunks
  // load/unload.
  torch::Tensor transformed_triangle_ids =
      torch::full({buffer_size}, -1,
                  torch::TensorOptions()
                      .dtype(torch::kInt64)
                      .device(triangles_->device_type_));
  int next_slot = 0;

  // Helper: Record newly transformed triangles by their IDs
  // Compares old vs new transform flags to find which triangles were just
  // transformed, then stores their stable IDs in the tracking buffer.
  auto updateTransformTracking = [&](const torch::Tensor &old_flags,
                                     const torch::Tensor &new_flags) {
    torch::Tensor newly_transformed_mask = new_flags & (~old_flags);
    torch::Tensor new_indices = torch::where(newly_transformed_mask)[0];

    if (new_indices.size(0) > 0) {
      torch::Tensor new_ids = triangles_->triangle_ids_.index({new_indices});
      int end_slot = next_slot + new_ids.size(0);
      transformed_triangle_ids.slice(0, next_slot, end_slot).copy_(new_ids);
      next_slot = end_slot;
    }
  };

  // Helper: Convert tracked IDs back to current index mask
  // Since indices change as chunks load/unload, we must recompute the mask
  // each time by checking which current triangles have IDs in our tracked set.
  auto getCurrentTransformFlags = [&]() -> torch::Tensor {
    torch::Tensor valid_ids = transformed_triangle_ids.slice(0, 0, next_slot);
    return torch::isin(triangles_->triangle_ids_, valid_ids);
  };

  // Process each keyframe sequentially
  for (const auto &kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<TriangleKeyframe> pkf = scene_->getKeyframe(kfid);

    if (!pkf) continue;

    // Compute pose difference
    const auto &pose = std::get<2>(kf);
    Sophus::SE3f original_pose = pkf->getPosef();
    Sophus::SE3f diff_pose = pose.inverse() * original_pose;

    // Handle loop closure keyframes: reset optimizer state for visible
    // triangles Must reset IMMEDIATELY while triangles are loaded (before
    // potential eviction)
    bool is_loop_closure_kf = std::get<4>(kf);
    if (is_loop_closure_kf) {
      std::cout << "[Sequential Loop] Loop closure keyframe: " << kfid
                << std::endl;
      // cullVisibleTriangles handles chunk loading internally
      torch::Tensor visible_triangles = triangles_->cullVisibleTriangles(pkf);

      if (torch::any(visible_triangles).item<bool>()) {
        triangles_->resetPositionLRAndOptimizerState(visible_triangles);
      }
    }

    // Handle large pose corrections: transform visible triangles
    // Only process if pose changed significantly (rotation or translation).
    if (isPoseDivergenceLarge(diff_pose)) {
      std::cout << "[Sequential Loop] Large correction for kf" << kfid
                << std::endl;

      // Convert pose difference to tensor for GPU operations
      torch::Tensor diff_pose_tensor =
          tensor_utils::EigenMatrix2TorchTensor(diff_pose.matrix(),
                                                triangles_->device_type_)
              .transpose(0, 1);

      // Find and load chunks visible from this keyframe's frustum
      std::vector<ChunkCoord> visible_chunk_coords =
          triangles_->frustumCullChunks(pkf, /*use_cache=*/true);

      torch::Tensor visible_chunk_coords_tensor = chunkCoordVectorToTensor(
          visible_chunk_coords, triangles_->device_type_);
      torch::Tensor visible_chunk_ids =
          encodeChunkCoordsTensor(visible_chunk_coords_tensor);

      torch::Tensor relevant_chunk_ids =
          filterRelevantChunks(visible_chunk_ids);

      if (relevant_chunk_ids.size(0) <= 0) continue;

      triangles_->loadChunks(relevant_chunk_ids);

      // Get current transform state (recomputed since chunks just loaded)
      torch::Tensor old_transform_flags = getCurrentTransformFlags();
      torch::Tensor current_transform_flags = old_transform_flags.clone();

      // Transform visible triangles that haven't been transformed yet
      int triangles_transformed_by_this_kf = 0;
      triangles_->scaledTransformVisiblePointsOfKeyframe(
          current_transform_flags, diff_pose_tensor, pkf->world_view_transform_,
          pkf->full_proj_transform_, pkf->creation_iter_,
          stableNumIterExistence(), triangles_transformed_by_this_kf,
          loop_kf_scale);

      // Record newly transformed triangle IDs
      updateTransformTracking(old_transform_flags, current_transform_flags);

      total_transformed += triangles_transformed_by_this_kf;
      std::cout << "[Sequential Loop] Keyframe " << kfid << " transformed "
                << triangles_transformed_by_this_kf << " points" << std::endl;

      // Reassign triangles to correct spatial chunks after transformation
      triangles_->handleBatchChunkRedistribution(relevant_chunk_ids);
      increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
    }

    // Update keyframe to use corrected ORB-SLAM pose
    pkf->setPose(pose.unit_quaternion().cast<double>(),
                 pose.translation().cast<double>());
    pkf->computeTransformTensors();
  }

  return total_transformed;
}

void TriangleMapper::processScaleRefinement(ORB_SLAM3::MappingOperation &opr) {
  throw std::runtime_error("Scale refinement not implemented!");
}

// ============================================================================
// Keyframe Management
// ============================================================================

void TriangleMapper::createAndInitializeKeyframe(
    std::shared_ptr<TriangleKeyframe> &pkf,
    cv::Mat &rgb_image,
    cv::Mat &aux_image,
    const Camera &camera,
    const std::string &filename) {
  // Set z clipping planes
  pkf->zfar_ = z_far_ * scene_->cameras_extent_;
  pkf->znear_ = z_near_ * scene_->cameras_extent_;

  // Set camera parameters
  pkf->setCameraParams(camera);
  pkf->img_filename_ = filename;
  pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
  pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
  pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

  // Add keyframe to scene
  pkf->computeTransformTensors();
  scene_->addKeyframe(pkf);

  // Update chunk-keyframe mapping
  keyframe_selector_->updateChunkKeyframeMapping(pkf, true);

  // Give new keyframes times of use and add to training sliding window
  increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

  // Extract dense features
  torch::Tensor input_tensor = feat_extractor_->parseInput(rgb_image);
  pkf->feature_map_ = feat_extractor_->extractDenseFeatures(input_tensor);

  // Initialize optimizer
  pkf->initOptimizer(device_type_, opt_params_.pose_lr_,
                     opt_params_.exposure_lr_,
                     opt_params_.depth_scale_bias_lr_);

  // Prepare multi resolution images for training
  pkf->generateImagePyramid(rgb_image);

  // Setup depth data based on sensor type
  if (sensor_type_ == MONOCULAR) {
    pkf->setupMonoData(rgb_image, device_type_, monocular_depth_estimator_,
                       min_depth_, max_depth_);
  } else if (sensor_type_ == STEREO && !aux_image.empty()) {
    pkf->setupStereoData(rgb_image, aux_image, stereo_baseline_length_,
                         device_type_, stereo_depth_estimator_, min_depth_,
                         max_depth_);
  } else if (sensor_type_ == RGBD && !aux_image.empty()) {
    pkf->setupRGBDData(aux_image);
  }

  pkf->loaded_ = true;

  // Sample triangles (requires render lock)
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  sampleTriangles(pkf);

  pkf->allow_eviction_ = true;
}

void TriangleMapper::handleNewKeyframeFromORBSLAM(KeyframeTuple &kf) {
  std::shared_ptr<TriangleKeyframe> pkf = std::make_shared<TriangleKeyframe>(
      std::get<0>(kf), getIteration(), keyframe_save_dir_);

  // Set pose from ORB-SLAM data
  auto &pose = std::get<2>(kf);
  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());

  cv::Mat imgRGB_undistorted = std::get<3>(kf);
  cv::Mat imgAux_undistorted = std::get<5>(kf);

  try {
    Camera &camera = scene_->cameras_.at(std::get<1>(kf));
    std::string filename = std::get<8>(kf);

    // Store ORB-SLAM keypoints
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));

    createAndInitializeKeyframe(pkf, imgRGB_undistorted, imgAux_undistorted,
                                camera, filename);

    // Release ORB-SLAM resources
    pSLAM_->getAtlas()->ReleaseKeyFrameImages(pkf->fid_);

  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[TriangleMapper::handleNewKeyframeFromORBSLAM] KeyFrame Camera not "
        "found!");
  }
}

void TriangleMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<TriangleKeyframe> pkf,
    int times) {
  pkf->remaining_times_of_use_ += times;
}

std::vector<std::shared_ptr<TriangleKeyframe>>
TriangleMapper::getClosestKeyframes(
    std::shared_ptr<TriangleKeyframe> current_kf,
    int n,
    int k) {
  std::vector<std::shared_ptr<TriangleKeyframe>> closest_keyframes;
  if (n <= 0 || k <= 0) return closest_keyframes;

  auto all_keyframes = scene_->getAllKeyframes();
  if (all_keyframes.empty()) return closest_keyframes;

  Eigen::Vector3f current_center = current_kf->getTranslationf();

  // Build candidate list sorted by spatial distance
  std::vector<std::pair<float, std::shared_ptr<TriangleKeyframe>>> candidates;
  for (const auto &kf_pair : all_keyframes) {
    if (kf_pair.second != current_kf) {
      Eigen::Vector3f candidate_center = kf_pair.second->getTranslationf();
      float spatial_distance = (current_center - candidate_center).norm();
      candidates.push_back({spatial_distance, kf_pair.second});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  // Take every k-th keyframe from sorted list
  int selected_count = 0;
  for (int i = 0; i < static_cast<int>(candidates.size()) && selected_count < n;
       i += k) {
    closest_keyframes.push_back(candidates[i].second);
    selected_count++;
  }

  // Fill remaining slots with closest unused keyframes
  if (selected_count < n) {
    std::set<std::shared_ptr<TriangleKeyframe>> selected_set(
        closest_keyframes.begin(), closest_keyframes.end());

    for (int i = 0;
         i < static_cast<int>(candidates.size()) && selected_count < n; ++i) {
      if (selected_set.find(candidates[i].second) == selected_set.end()) {
        closest_keyframes.push_back(candidates[i].second);
        selected_count++;
      }
    }
  }

  return closest_keyframes;
}

// ============================================================================
// Triangle Sampling
// ============================================================================

void TriangleMapper::sampleTriangles(std::shared_ptr<TriangleKeyframe> pkf) {
  torch::NoGradGuard no_grad;

  std::vector<std::shared_ptr<TriangleKeyframe>> newly_loaded_keyframes;
  if (!pkf->loaded_) {
    pkf->loadDataFromDisk();
    newly_loaded_keyframes.push_back(pkf);
  }

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  // Get RGB image and optionally downsample for sampling
  torch::Tensor rgb = pkf->gaus_pyramid_original_image_[0];

  if (downsample_for_sampling_) {
    rgb = rgb.unsqueeze(0);
    rgb = torch::avg_pool2d(rgb, 2);
    rgb = torch::nn::functional::interpolate(
        rgb,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{pkf->image_height_, pkf->image_width_})
            .mode(torch::kBilinear)
            .align_corners(true));
    rgb = rgb.squeeze(0);
  }

  torch::Tensor depth_confidence = pkf->depth_confidence_;

  // Compute initial probability based on image gradients
  torch::Tensor init_proba = computeLoGProbability(rgb);

  // Render current view and compute penalty if scene is initialized
  torch::Tensor penalty = torch::zeros_like(init_proba);
  torch::Tensor rendered_depth;
  torch::Tensor main_triangle_ids;
  torch::Tensor visible_triangle_mask;
  bool has_rendered_depth = false;

  torch::Tensor full_model_scaling;

  if (initial_mapped_) {
    visible_triangle_mask = triangles_->cullVisibleTriangles(pkf);

    torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
    auto render_pkg = TriangleRenderer::render(
        triangles_, visible_triangle_mask, pkf, pkf->image_height_,
        pkf->image_width_, pipe_params_, background_, override_color_, 1.0f,
        false, pkf->FoVx_, pkf->FoVy_, view_matrix, pkf->full_proj_transform_);

    torch::Tensor rendered_image = std::get<1>(render_pkg);
    rendered_depth = 1 / std::get<0>(render_pkg).clamp_min(1e-8);
    has_rendered_depth = true;
    main_triangle_ids = std::get<3>(render_pkg)[0];
    full_model_scaling = std::get<4>(render_pkg);
    penalty = computeLoGProbability(rendered_image);
  }

  // Apply scaling factor and compute sampling probability
  init_proba *= init_proba_scaler_;
  penalty *= init_proba_scaler_;

  // Generate initial sample mask based on probability
  torch::Tensor sample_mask =
      torch::rand_like(init_proba) < init_proba - penalty;
  torch::Tensor flat_sample_mask = sample_mask.flatten();

  // Pre-compute UV grid
  torch::Tensor uv_;
  {
    auto x_coords = torch::arange(0, pkf->image_width_, torch::kFloat32).cuda();
    auto y_coords =
        torch::arange(0, pkf->image_height_, torch::kFloat32).cuda();
    auto meshgrid = torch::meshgrid({x_coords, y_coords}, "xy");
    uv_ = torch::stack({meshgrid[0], meshgrid[1]}, -1);
  }

  // Get UV coordinates of initially sampled points
  torch::Tensor sampled_uv = uv_.view({-1, 2}).index({sample_mask.flatten()});

  // Get closest keyframes for MVS
  std::vector<std::shared_ptr<TriangleKeyframe>> prev_keyframes =
      getClosestKeyframes(pkf, guided_mvs_->getNumCams(), 6);

  for (const auto &kf : prev_keyframes) {
    if (!kf->loaded_) {
      kf->loadDataFromDisk();
      newly_loaded_keyframes.push_back(kf);
    }
  }

  torch::Tensor accurate_mask, depth;

  if (prev_keyframes.size() != guided_mvs_->getNumCams()) {
    std::cout << "Not enough previous keyframes found for MVS." << std::endl;
    assert(pkf->gaus_pyramid_inv_depth_image_[0].defined());

    torch::Tensor depth_map =
        1 / pkf->gaus_pyramid_inv_depth_image_[0].clamp_min(1e-8);
    torch::Tensor sample_indices = torch::nonzero(flat_sample_mask).squeeze(-1);
    torch::Tensor depth_map_flat = depth_map.flatten();
    depth = depth_map_flat.index({sample_indices});
    accurate_mask = torch::ones_like(depth, torch::kBool);
  } else {
    std::tie(depth, accurate_mask) =
        (*guided_mvs_)(sampled_uv, pkf, prev_keyframes);
  }

  // Apply confidence filtering
  torch::Tensor sampled_confidence = sampleConf(
      depth_confidence, sampled_uv, pkf->image_width_, pkf->image_height_);
  torch::Tensor valid_mask = (depth > 1e-6) & (sampled_confidence > 0.5);

  // Update sample_mask with valid positions
  torch::Tensor original_sample_indices =
      torch::nonzero(flat_sample_mask).squeeze(-1);
  torch::Tensor valid_sample_indices =
      original_sample_indices.index({valid_mask});

  sample_mask.fill_(false);
  sample_mask.view(-1).index_put_({valid_sample_indices}, true);

  // Filter tensors
  depth = depth.index({valid_mask});
  sampled_uv = sampled_uv.index({valid_mask});
  accurate_mask = accurate_mask.index({valid_mask});

  // Handle Triangle removal for coarser Triangles
  if (has_rendered_depth) {
    torch::Tensor accurate_sample_mask = torch::zeros_like(sample_mask);
    torch::Tensor current_flat_indices =
        sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
    torch::Tensor accurate_positions =
        current_flat_indices.index({accurate_mask});
    accurate_sample_mask.view(-1).index_put_(
        {accurate_positions.to(torch::kLong)}, true);

    if (accurate_sample_mask.any().item<bool>()) {
      torch::Tensor selected_main_triangles =
          main_triangle_ids.index({accurate_sample_mask});
      torch::Tensor valid_ids_mask = selected_main_triangles >= 0;

      if (valid_ids_mask.any().item<bool>()) {
        selected_main_triangles =
            selected_main_triangles.index({valid_ids_mask});

        auto unique_result =
            torch::_unique2(selected_main_triangles, false, false, true);
        torch::Tensor unique_ids = std::get<0>(unique_result);
        torch::Tensor counts = std::get<2>(unique_result);

        constexpr int kMinCountForRemoval = 10;
        torch::Tensor removal_mask = counts >= kMinCountForRemoval;

        if (removal_mask.any().item<bool>()) {
          torch::Tensor triangles_to_remove_subset =
              unique_ids.index({removal_mask});
          torch::Tensor visible_indices =
              torch::where(visible_triangle_mask)[0];
          torch::Tensor triangles_to_remove_full =
              visible_indices.index({triangles_to_remove_subset});

          torch::Tensor full_model_prune_mask = torch::zeros(
              {triangles_->getXYZ().size(0)},
              torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

          full_model_prune_mask.index_put_({triangles_to_remove_full}, true);
          triangles_->prunePoints(full_model_prune_mask);

          visible_triangle_mask = triangles_->cullVisibleTriangles(pkf);

          torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
          auto updated_render_pkg = TriangleRenderer::render(
              triangles_, visible_triangle_mask, pkf, pkf->image_height_,
              pkf->image_width_, pipe_params_, background_, override_color_,
              1.0f, false, pkf->FoVx_, pkf->FoVy_, view_matrix,
              pkf->full_proj_transform_);

          rendered_depth = 1 / std::get<0>(updated_render_pkg).clamp_min(1e-8);
          full_model_scaling = std::get<4>(updated_render_pkg);
        }
      }
    }
  }

  // Check for occlusions
  if (has_rendered_depth) {
    torch::Tensor current_flat_indices =
        sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
    torch::Tensor rendered_depth_flat = rendered_depth.flatten();
    torch::Tensor rendered_depth_sampled =
        rendered_depth_flat.index({current_flat_indices.to(torch::kLong)});

    torch::Tensor occlusion_mask = depth < rendered_depth_sampled;

    depth = depth.index({occlusion_mask});
    sampled_uv = sampled_uv.index({occlusion_mask});
    accurate_mask = accurate_mask.index({occlusion_mask});

    sample_mask.fill_(false);
    if (depth.size(0) > 0) {
      torch::Tensor final_flat_indices =
          sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
      sample_mask.view(-1).index_put_({final_flat_indices.to(torch::kLong)},
                                      true);
    }
  }

  // Early exit if no samples remain
  if (depth.size(0) == 0) {
    return;
  }

  // Get matched keypoints and their 3D positions
  torch::Tensor match_pts_3d;
  torch::Tensor match_colors;
  torch::Tensor match_init_proba;
  int num_matched_points = 0;

  if (!pkf->kps_pixel_.empty() && !pkf->kps_point_local_.empty()) {
    int num_keypoints = pkf->kps_pixel_.size() / 2;

    torch::Tensor kps_pixel_tensor =
        torch::from_blob(pkf->kps_pixel_.data(), {num_keypoints, 2},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);

    torch::Tensor kps_point_local_tensor =
        torch::from_blob(pkf->kps_point_local_.data(), {num_keypoints, 3},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);

    // Create validity mask
    torch::Tensor u_coords = kps_pixel_tensor.select(1, 0);
    torch::Tensor v_coords = kps_pixel_tensor.select(1, 1);
    torch::Tensor z_coords = kps_point_local_tensor.select(1, 2);

    torch::Tensor kp_valid_mask =
        (z_coords > 1e-6) & (u_coords >= 0) & (u_coords < pkf->image_width_) &
        (v_coords >= 0) & (v_coords < pkf->image_height_) &
        torch::isfinite(kps_point_local_tensor.select(1, 0)) &
        torch::isfinite(kps_point_local_tensor.select(1, 1)) &
        torch::isfinite(z_coords);

    num_matched_points = kp_valid_mask.sum().item<int>();

    if (num_matched_points > 0) {
      torch::Tensor valid_kps_pixel = kps_pixel_tensor.index({kp_valid_mask});
      match_pts_3d = kps_point_local_tensor.index({kp_valid_mask});

      // Convert pixel coordinates to normalized grid coordinates [-1, 1]
      torch::Tensor normalized_coords = torch::zeros(
          {num_matched_points, 2},
          torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
      normalized_coords.select(1, 0) =
          2.0f * valid_kps_pixel.select(1, 0) / (pkf->image_width_ - 1) - 1.0f;
      normalized_coords.select(1, 1) =
          2.0f * valid_kps_pixel.select(1, 1) / (pkf->image_height_ - 1) - 1.0f;

      torch::Tensor grid =
          normalized_coords.view({1, 1, num_matched_points, 2});

      // Sample colors using grid_sample
      torch::Tensor rgb_for_sampling = rgb.unsqueeze(0);
      torch::Tensor sampled_colors_raw = torch::nn::functional::grid_sample(
          rgb_for_sampling, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kBilinear)
              .align_corners(true));

      match_colors = sampled_colors_raw.squeeze(0).squeeze(1).transpose(0, 1);

      // Sample init_proba
      torch::Tensor init_proba_for_sampling =
          init_proba.unsqueeze(0).unsqueeze(0);
      torch::Tensor sampled_proba_raw = torch::nn::functional::grid_sample(
          init_proba_for_sampling, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kBilinear)
              .align_corners(true));

      match_init_proba = sampled_proba_raw.squeeze();
    }
  }

  // Flatten RGB for regular sampled points extraction
  rgb = rgb.permute({1, 2, 0}).flatten(0, 1);
  init_proba = init_proba.flatten();

  // Extract sampled data
  torch::Tensor flat_indices =
      sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
  torch::Tensor sampled_colors = rgb.index({flat_indices.to(torch::kLong)});
  torch::Tensor sampled_init_proba =
      init_proba.index({flat_indices.to(torch::kLong)});

  // Reproject sampled points to 3D
  float fx = pkf->intr_[0];
  float fy = pkf->intr_[1];
  float cx = pkf->intr_[2];
  float cy = pkf->intr_[3];

  torch::Tensor u_coords = sampled_uv.select(1, 0);
  torch::Tensor v_coords = sampled_uv.select(1, 1);

  torch::Tensor sampled_points3D =
      torch::zeros({depth.size(0), 3}, depth.options());
  sampled_points3D.select(1, 0) = (u_coords - cx) * depth / fx;
  sampled_points3D.select(1, 1) = (v_coords - cy) * depth / fy;
  sampled_points3D.select(1, 2) = depth;

  // Compute surface normals from depth map finite differences (4DTAM method).
  // n = normalize((p(u+1,v) - p(u-1,v)) × (p(u,v+1) - p(u,v-1)))
  // where p(u,v) is the back-projected 3D point in camera space.
  // Compute surface normals from depth map (4DTAM method) and sample at the
  // selected UV positions. Degenerate normals (zero) fall back to camera-facing
  // in generateTriangleVertices. Normals are rotated to world space.
  torch::Tensor sampled_normals_world;
  {
    // Dense normal map [3, H, W] in camera space
    torch::Tensor normal_map = loss_utils::computeNormalsFromDepth(
        pkf->gaus_pyramid_inv_depth_image_[0], fx, fy, cx, cy);

    // Sample at the filtered UV positions → [N, 3]
    torch::Tensor u_l = sampled_uv.select(1, 0).to(torch::kLong);
    torch::Tensor v_l = sampled_uv.select(1, 1).to(torch::kLong);
    torch::Tensor normals_cam =
        normal_map.permute({1, 2, 0}).index({v_l, u_l});  // [N, 3]

    // Ensure normals point toward camera (nz < 0 in camera space)
    torch::Tensor nz = normals_cam.select(1, 2);
    torch::Tensor flip = (nz > 0.0f).unsqueeze(1).to(torch::kFloat32);
    normals_cam = normals_cam * (1.0f - 2.0f * flip);

    // Zero out border/degenerate pixels (already zeroed by computeNormalsFromDepth,
    // but re-apply after flip to keep zeros as zeros)
    torch::Tensor lengths = torch::norm(normals_cam, 2, 1, true);
    torch::Tensor valid = (lengths.squeeze(1) > 1e-6f).unsqueeze(1).to(torch::kFloat32);
    normals_cam = normals_cam * valid;

    // Rotate to world space: n_world = Rwc @ n_cam
    Eigen::MatrixXf Rwc = Twc.rotationMatrix();
    torch::Tensor Rwc_t =
        tensor_utils::EigenMatrix2TorchTensor(Rwc, device_type_);
    sampled_normals_world = torch::mm(normals_cam, Rwc_t.transpose(0, 1));
  }

  // Combine sampled and matched points
  torch::Tensor all_points3D;
  torch::Tensor all_colors;
  torch::Tensor all_init_proba;
  torch::Tensor all_normals;

  if (num_matched_points > 0) {
    all_points3D = torch::cat({sampled_points3D, match_pts_3d}, 0);
    all_colors = torch::cat({sampled_colors, match_colors}, 0);
    all_init_proba = torch::cat({sampled_init_proba, match_init_proba}, 0);
    // Matched keypoints have no depth neighborhood: zero normals cause
    // generateTriangleVertices to fall back to camera-facing for those points.
    torch::Tensor matched_normals = torch::zeros(
        {num_matched_points, 3},
        torch::TensorOptions().device(device_type_));
    all_normals = torch::cat({sampled_normals_world, matched_normals}, 0);
  } else {
    all_points3D = sampled_points3D;
    all_colors = sampled_colors;
    all_init_proba = sampled_init_proba;
    all_normals = sampled_normals_world;
  }

  // Transform points to world coordinates
  torch::Tensor Twc_tensor =
      tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(all_points3D, Twc_tensor);

  // Compute scales for all points
  torch::Tensor scales = 1.0f / torch::sqrt(all_init_proba + 1e-8f);
  scales =
      torch::clamp(scales, 1.0f, static_cast<float>(pkf->image_width_) / 10.0f);
  scales *= (1.0f / pkf->intr_[0]);

  // Scale by distance to camera center
  torch::Tensor diff = all_points3D - pkf->getCenter().unsqueeze(0);
  torch::Tensor distances = torch::norm(diff, 2, 1);
  scales *= distances;
  scales = torch::log(torch::clamp(scales, 1e-6f, 1e6f));
  torch::Tensor all_scales = scales.unsqueeze(1).repeat({1, 3});

  // Set opacities based on point type
  torch::Tensor all_opacities = torch::zeros(
      {all_points3D.size(0), 1}, torch::TensorOptions().device(device_type_));

  int num_sampled = sampled_points3D.size(0);

  // Initial opacities match original triangle-splatting set_opacity = 0.28
  constexpr float kAccurateOpacity = 0.28f;
  constexpr float kInaccurateOpacity = 0.14f;
  constexpr float kMatchedOpacity = 0.28f;

  if (num_sampled > 0) {
    torch::Tensor sampled_accurate_opacity =
        torch::full({num_sampled, 1}, kAccurateOpacity,
                    torch::TensorOptions().device(device_type_));
    torch::Tensor sampled_inaccurate_opacity =
        torch::full({num_sampled, 1}, kInaccurateOpacity,
                    torch::TensorOptions().device(device_type_));

    torch::Tensor sampled_opacities =
        torch::where(accurate_mask.unsqueeze(-1), sampled_accurate_opacity,
                     sampled_inaccurate_opacity);

    all_opacities.slice(0, 0, num_sampled) = sampled_opacities;
  }

  if (num_matched_points > 0) {
    torch::Tensor matched_opacities =
        torch::full({num_matched_points, 1}, kMatchedOpacity,
                    torch::TensorOptions().device(device_type_));

    all_opacities.slice(0, num_sampled, num_sampled + num_matched_points) =
        matched_opacities;
  }

  if (initial_mapped_) {
    triangles_->pruneLowWeightTriangles(pkf, visible_triangle_mask, full_model_scaling);
  }

  // Add all points to scene
  torch::Tensor final_opacities = general_utils::inverse_sigmoid(all_opacities);

  // init_strategy_: 0=normal-based, 1=camera-facing, 2=fibonacci.
  // generateTriangleVertices selects the mode based on which tensors are defined,
  // so we control the strategy by passing or withholding normals/cam_center.
  torch::Tensor eff_normals    = (init_strategy_ != 2) ? all_normals      : torch::Tensor{};
  torch::Tensor eff_cam_center = (init_strategy_ != 2) ? pkf->getCenter() : torch::Tensor{};
  if (init_strategy_ == 1) eff_normals = torch::Tensor{};  // camera-facing: drop normals

  triangles_->addPoints(all_points3D, all_colors, all_scales, final_opacities,
                        getIteration(), scene_->cameras_extent_,
                        eff_cam_center, eff_normals);

  // Save keyframes that were loaded during this operation
  for (const auto &kf : newly_loaded_keyframes) {
    kf->saveDataToDisk();
  }
}