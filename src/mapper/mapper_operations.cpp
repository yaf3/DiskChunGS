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

#include "gaussian_mapper.h"
#include "rendering/gaussian_renderer.h"
#include "utils/profiling.h"

void GaussianMapper::combineMappingOperations() {
  // auto timer_combineMappingOperations =
  //     ProfilingUtils::Timer("combineMappingOperations");

  // Collect and group all operations
  std::vector<ORB_SLAM3::MappingOperation> localBAOps;
  std::vector<ORB_SLAM3::MappingOperation> loopClosureOps;
  std::vector<ORB_SLAM3::MappingOperation> scaleRefinementOps;

  // Collect all operations first
  while (pSLAM_->getAtlas()->hasMappingOperation()) {
    ORB_SLAM3::MappingOperation opr =
        pSLAM_->getAtlas()->getAndPopMappingOperation();

    // Group by operation type
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

  // Process all LocalMappingBA operations together
  if (!localBAOps.empty()) {
    // auto timer_processLocalBA = ProfilingUtils::Timer("processLocalBA");
    processLocalMappingBABatch(localBAOps);
    // timer_processLocalBA.stop();
  }

  // Process loop closure operations (these are usually more complex and less
  // frequent)
  for (auto& opr : loopClosureOps) {
    auto timer_loopClosure = ProfilingUtils::Timer("processLoopClosure");

    // PAUSE IMAGE INGESTION - Stop ORB-SLAM3 from receiving new images
    std::cout << "[Loop Closure] ========================================"
              << std::endl;
    std::cout << "[Loop Closure] PAUSING ORB-SLAM3 image ingestion..."
              << std::endl;
    std::cout << "[Loop Closure] ========================================"
              << std::endl;
    pause_image_ingestion_.store(true, std::memory_order_release);

    // Wait a bit to ensure main thread has stopped feeding images
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Process the loop closure
    processLoopClosureBA(opr);

    // EXTENDED OPTIMIZATION for loop closure affected areas
    std::cout << "\n[Loop Closure] ========================================"
              << std::endl;
    std::cout << "[Loop Closure] Running extended optimization for "
              << loop_closure_optimization_iterations_ << " iterations..."
              << std::endl;
    std::cout << "[Loop Closure] Spatial gradient masking enabled (radius: "
              << max_optimization_distance_ << "m)" << std::endl;
    std::cout << "[Loop Closure] ========================================\n"
              << std::endl;

    // Enable spatial gradient masking during loop closure
    enable_spatial_gradient_masking_ = false;

    for (int i = 0; i < loop_closure_optimization_iterations_; i++) {
      trainForOneIteration();

      // Progress reporting
      if ((i + 1) % 100 == 0 || i == 0) {
        std::cout << "[Loop Closure Optimization] Iteration " << (i + 1) << "/"
                  << loop_closure_optimization_iterations_
                  << " (EMA Loss: " << ema_loss_for_log_ << ")" << std::endl;
      }
    }

    // Disable spatial gradient masking after loop closure
    enable_spatial_gradient_masking_ = false;

    // RESUME - Allow image ingestion to continue
    std::cout << "\n[Loop Closure] ========================================"
              << std::endl;
    std::cout << "[Loop Closure] Optimization complete. RESUMING ORB-SLAM3..."
              << std::endl;
    std::cout << "[Loop Closure] ========================================\n"
              << std::endl;
    pause_image_ingestion_.store(false, std::memory_order_release);

    timer_loopClosure.stop();
  }

  // Process scale refinement operations
  for (auto& opr : scaleRefinementOps) {
    auto timer_scaleRefinement =
        ProfilingUtils::Timer("processScaleRefinement");
    processScaleRefinement(opr);
    timer_scaleRefinement.stop();
  }

  // timer_combineMappingOperations.stop();
}

void GaussianMapper::processLocalMappingBABatch(
    std::vector<ORB_SLAM3::MappingOperation>& operations) {
  // auto timer_LocalMapping_before_addPoints =
  //     ProfilingUtils::Timer("LocalMapping_before_addPoints");
  if (operations.empty()) return;

  // Containers for batching
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
      associated_keyframe_map;
  std::vector<float> all_points;
  std::vector<float> all_colors;

  // Process all keyframes first
  for (auto& opr : operations) {
    auto& associated_kfs = opr.associatedKeyFrames();

    // Add/update keyframes
    for (auto& kf : associated_kfs) {
      auto kfid = std::get<0>(kf);
      std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

      if (pkf) {
        auto& orb_pose = std::get<2>(kf);

        // Option A: If pose optimization is disabled (pose_lr = 0), use ORB -
        // SLAM poses
        if (opt_params_.pose_lr_ <= 0.0f) {
          pkf->setPose(orb_pose.unit_quaternion().cast<double>(),
                       orb_pose.translation().cast<double>());
        }
        // Option B: If pose optimization is enabled, selectively
        // update
        else {
          // Check if pose has diverged significantly from ORB-SLAM
          Sophus::SE3f gaussian_pose = pkf->getPosef();
          Sophus::SE3f diff_pose = orb_pose.inverse() * gaussian_pose;

          bool large_divergence =
              !diff_pose.rotationMatrix().isApprox(Eigen::Matrix3f::Identity(),
                                                   0.1f) ||
              !diff_pose.translation().isMuchSmallerThan(1.0, 0.05f);

          // Only override if poses have diverged too much (geometric BA found
          // better solution)
          if (large_divergence) {
            pkf->setPose(orb_pose.unit_quaternion().cast<double>(),
                         orb_pose.translation().cast<double>());
          }
          // Otherwise keep Gaussian-optimized pose
        }

        pkf->computeTransformTensors();
        // if (keyframe_selection_strategy_ == 1) {
        //   keyframe_queue_->updateChunkKeyframeMapping(pkf);
        // }
      } else {
        // Create a new keyframe
        handleNewKeyframeFromORBSLAM(kf);
      }
    }
  }
}

void GaussianMapper::processLoopClosureBA(ORB_SLAM3::MappingOperation& opr) {
  std::cout << "[DEBUG] Starting loop closure with scale factor: "
            << opr.mfScale << std::endl;

  float loop_kf_scale = opr.mfScale;
  auto& associated_kfs = opr.associatedKeyFrames();

  std::cout << "[DEBUG] Processing " << associated_kfs.size() << " keyframes"
            << std::endl;

  auto time_start = std::chrono::steady_clock::now();

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_0_before_loop_correction") /
              "data");
  }

  int total_transformed = 0;

  // First pass: Handle all new keyframes (this modifies chunks)
  for (auto& kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

    if (!pkf) {
      std::cout << "New frame in loop-closure" << std::endl;
      handleNewKeyframeFromORBSLAM(kf);  // This modifies chunks!
    }
  }

  std::unique_lock<std::mutex> lock_render(mutex_render_);

  // === ADAPTIVE BATCHING STRATEGY ===

  // Step 1: Estimate total gaussians needed for all keyframes
  int64_t total_gaussians_needed = 0;
  std::vector<std::pair<std::shared_ptr<GaussianKeyframe>, torch::Tensor>>
      kf_chunk_pairs;
  std::unordered_set<int64_t> all_unique_chunks;

  for (auto& kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

    if (!pkf) continue;

    auto& pose = std::get<2>(kf);
    Sophus::SE3f original_pose = pkf->getPosef();
    Sophus::SE3f inv_pose = pose.inverse();
    Sophus::SE3f diff_pose = inv_pose * original_pose;

    bool large_rot = !diff_pose.rotationMatrix().isApprox(
        Eigen::Matrix3f::Identity(), large_rot_th_);
    bool large_trans =
        !diff_pose.translation().isMuchSmallerThan(1.0, large_trans_th_);

    if (large_rot || large_trans) {
      // Use frustum culling to get visible chunk coordinates
      std::vector<ChunkCoord> visible_chunk_coords =
          gaussians_->frustumCullChunks(pkf, /*use_cache=*/true);

      // Convert chunk coords to IDs and filter existing chunks
      torch::Tensor visible_chunk_coords_tensor = chunkCoordVectorToTensor(
          visible_chunk_coords, gaussians_->device_type_);
      torch::Tensor visible_chunk_ids =
          encodeChunkCoordsTensor(visible_chunk_coords_tensor);

      // Check which chunks have any relevance
      torch::Tensor loaded_mask =
          torch::isin(visible_chunk_ids, gaussians_->chunks_loaded_from_disk_);
      torch::Tensor on_disk_mask =
          torch::isin(visible_chunk_ids, gaussians_->chunks_on_disk_);
      torch::Tensor spatial_chunks =
          std::get<0>(torch::_unique2(gaussians_->gaussian_chunk_ids_));
      torch::Tensor has_gaussians_mask =
          torch::isin(visible_chunk_ids, spatial_chunks);

      torch::Tensor relevant_mask =
          loaded_mask | on_disk_mask | has_gaussians_mask;
      torch::Tensor relevant_chunk_ids =
          visible_chunk_ids.index({relevant_mask});

      if (relevant_chunk_ids.size(0) > 0) {
        kf_chunk_pairs.emplace_back(pkf, relevant_chunk_ids);

        // Add to global unique chunks set
        auto chunk_ids_cpu = relevant_chunk_ids.cpu();
        auto accessor = chunk_ids_cpu.accessor<int64_t, 1>();
        for (int i = 0; i < chunk_ids_cpu.size(0); ++i) {
          // Only inserts if not already in the set since its a set
          all_unique_chunks.insert(accessor[i]);
        }
      }
    }
  }

  // Step 2: Save and evict all current chunks to get clean slate
  torch::Tensor all_spatial_chunks =
      std::get<0>(torch::_unique2(gaussians_->gaussian_chunk_ids_));
  if (all_spatial_chunks.size(0) > 0) {
    std::cout << "[Loop Closure] Saving and evicting "
              << all_spatial_chunks.size(0)
              << " current chunks for clean memory calculation" << std::endl;
    gaussians_->saveAndEvictChunks(all_spatial_chunks);
  }

  // Step 3: Calculate exact gaussian count for needed chunks (no double
  // counting)
  for (int64_t chunk_id : all_unique_chunks) {
    torch::Tensor chunk_id_tensor =
        torch::tensor({chunk_id}, torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(gaussians_->device_type_));

    // Check if chunk is on disk and get its gaussian count
    auto disk_mask = torch::eq(gaussians_->chunks_on_disk_, chunk_id_tensor);
    if (torch::any(disk_mask).item<bool>()) {
      auto indices = torch::where(disk_mask)[0];
      if (indices.size(0) > 0) {
        int64_t count =
            gaussians_->chunk_gaussian_counts_[indices[0].item<int64_t>()]
                .item<int64_t>();
        total_gaussians_needed += count;
      }
    }
    // Note: No else case needed since we evicted all spatial chunks above
  }

  int64_t current_gaussians =
      gaussians_->xyz_.size(0);  // Should be 0 or minimal after eviction
  int64_t projected_total =
      total_gaussians_needed + current_gaussians;  // Clean calculation

  std::cout << "[Loop Closure] Gaussian estimation - Current: "
            << current_gaussians
            << ", Additional needed: " << total_gaussians_needed
            << ", Projected total: " << projected_total
            << ", Limit: " << gaussians_->max_gaussians_in_memory_ << std::endl;

  // Step 3: Choose strategy based on memory constraints
  bool use_batched_strategy =
      (projected_total <= gaussians_->max_gaussians_in_memory_);

  float temp_max_gaussians_in_memory = gaussians_->max_gaussians_in_memory_;
  gaussians_->max_gaussians_in_memory_ = 100000000000;

  if (keyframe_selection_strategy_ == 1) {
    for (const auto& [index, keyframe] : scene_->keyframes_) {
      if (keyframe->loaded_) keyframe->saveDataToDisk();
    }
  }

  // Force batched strategy for now
  if (true || use_batched_strategy) {
    std::cout << "[Loop Closure] Using BATCHED strategy - sufficient memory"
              << std::endl;
    total_transformed = processBatchedLoopClosure(
        associated_kfs, kf_chunk_pairs, all_unique_chunks, loop_kf_scale);
  } else {
    std::cout << "[Loop Closure] Using SEQUENTIAL strategy - memory limited"
              << std::endl;
    total_transformed =
        processSequentialLoopClosure(associated_kfs, loop_kf_scale);
  }

  gaussians_->max_gaussians_in_memory_ = temp_max_gaussians_in_memory;

  if (keyframe_selection_strategy_ == 1) {
    for (auto& kf : associated_kfs) {
      auto kfid = std::get<0>(kf);
      std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
      keyframe_queue_->updateChunkKeyframeMapping(pkf, false);
    }
  }

  // Delete any mapping BA operations that accumulated during loop closure
  // pSLAM_->getAtlas()->clearMappingOperation();

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_1_after_loop_correction") /
              "data");
  }

  // Mark this iteration
  loop_closure_iteration_ = true;

  // Prioritize chunks from loop closure detection area for optimization
  // if (false && keyframe_selection_strategy_ == 1) {
  //   std::unordered_set<int64_t> loop_detection_chunks;
  //   int loop_closure_kf_count = 0;

  //   for (auto& kf : associated_kfs) {
  //     bool is_loop_closure_kf = std::get<4>(kf);  // isLoopClosureKF flag
  //     if (is_loop_closure_kf) {
  //       loop_closure_kf_count++;
  //       auto kfid = std::get<0>(kf);
  //       std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
  //       if (pkf) {
  //         std::vector<ChunkCoord> visible_chunks =
  //             frustumCullChunks(pkf, chunk_size_, nullptr);
  //         torch::Tensor visible_chunk_coords_tensor =
  //             chunkCoordVectorToTensor(visible_chunks);
  //         torch::Tensor visible_chunk_ids =
  //             encodeChunkCoordsTensor(visible_chunk_coords_tensor);
  //         for (int i = 0; i < visible_chunk_ids.size(0); ++i) {
  //           loop_detection_chunks.insert(visible_chunk_ids[i].item<int64_t>());
  //         }
  //       }
  //     }
  //   }

  //   if (!loop_detection_chunks.empty()) {
  //     keyframe_queue_->addLoopClosurePriorityChunks(loop_detection_chunks);
  //     std::cout << "[Loop Closure] Found " << loop_closure_kf_count
  //               << " loop closure keyframes, enqueued "
  //               << loop_detection_chunks.size()
  //               << " unique chunks for priority optimization" << std::endl;
  //   }
  // }

  auto time_end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start)
          .count();
  std::cout << duration << " s" << std::endl;

  std::cout << "[Loop Closure] Completed - Total gaussians transformed: "
            << total_transformed << std::endl;
}

int GaussianMapper::processBatchedLoopClosure(
    std::vector<std::tuple<unsigned long,
                           unsigned long,
                           Sophus::SE3f,
                           cv::Mat,
                           bool,
                           cv::Mat,
                           std::vector<float>,
                           std::vector<float>,
                           std::string>>& associated_kfs,
    const std::vector<std::pair<std::shared_ptr<GaussianKeyframe>,
                                torch::Tensor>>& kf_chunk_pairs,
    const std::unordered_set<int64_t>& all_unique_chunks,
    float loop_kf_scale) {
  int total_transformed = 0;

  // Step 1: Batch load ALL required chunks at once
  if (!all_unique_chunks.empty()) {
    // Create tensor on CPU first, then move to device
    torch::Tensor all_chunk_ids = torch::empty(
        {static_cast<int64_t>(all_unique_chunks.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto accessor = all_chunk_ids.accessor<int64_t, 1>();
    int idx = 0;
    for (int64_t chunk_id : all_unique_chunks) {
      accessor[idx++] = chunk_id;
    }

    // Move to target device
    all_chunk_ids = all_chunk_ids.to(gaussians_->device_type_);

    std::cout << "[Batched Loop] Loading " << all_unique_chunks.size()
              << " unique chunks in single batch" << std::endl;
    gaussians_->loadChunks(all_chunk_ids);
  }

  // Step 2: Use vectorized transformation tracking
  torch::Tensor global_transform_mask = torch::zeros(
      {gaussians_->xyz_.size(0)}, torch::TensorOptions()
                                      .dtype(torch::kBool)
                                      .device(gaussians_->device_type_));

  // Step 3: Process all keyframes with pre-loaded chunks
  std::vector<torch::Tensor> chunks_to_redistribute;

  for (auto& kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

    if (pkf) {
      auto& pose = std::get<2>(kf);
      Sophus::SE3f original_pose = pkf->getPosef();
      Sophus::SE3f inv_pose = pose.inverse();
      Sophus::SE3f diff_pose = inv_pose * original_pose;

      bool large_rot = !diff_pose.rotationMatrix().isApprox(
          Eigen::Matrix3f::Identity(), large_rot_th_);
      bool large_trans =
          !diff_pose.translation().isMuchSmallerThan(1.0, large_trans_th_);

      // Check if this is a loop closure keyframe (where two reconstructions
      // merge)
      bool is_loop_closure_kf = std::get<4>(kf);
      if (is_loop_closure_kf) {
        std::cout << "[Batched Loop] Loop closure keyframe detected: " << kfid
                  << std::endl;
        // Reset opacity immediately for visible gaussians
        torch::Tensor visible_gaussians =
            gaussians_->cullVisibleGaussians(pkf, false);
        if (torch::any(visible_gaussians).item<bool>()) {
          // gaussians_->resetOpacityForMask(visible_gaussians);
          gaussians_->resetPositionLRAndOptimizerState(visible_gaussians);
        }
        // Reset depth loss weight for keyframe that underwent large
        // transformation
        // pkf->resetDepthLossWeight();
      }

      if (large_rot || large_trans) {
        // Find the pre-computed chunks for this keyframe
        auto it =
            std::find_if(kf_chunk_pairs.begin(), kf_chunk_pairs.end(),
                         [&](const auto& pair) { return pair.first == pkf; });

        if (it != kf_chunk_pairs.end()) {
          torch::Tensor relevant_chunk_ids = it->second;

          std::cout << "[Batched Loop] Large loop correction detected for kf"
                    << kfid << std::endl;

          // Prepare transformation tensor
          torch::Tensor diff_pose_tensor =
              tensor_utils::EigenMatrix2TorchTensor(diff_pose.matrix(),
                                                    gaussians_->device_type_)
                  .transpose(0, 1);

          int gaussians_transformed_by_this_kf = 0;
          loop_kf_scale = 1.0;

          // All chunks are already loaded, so this should be fast
          gaussians_->scaledTransformVisiblePointsOfKeyframe(
              global_transform_mask, diff_pose_tensor,
              pkf->world_view_transform_, pkf->full_proj_transform_,
              pkf->creation_iter_, stableNumIterExistence(),
              gaussians_transformed_by_this_kf, loop_kf_scale);

          total_transformed += gaussians_transformed_by_this_kf;
          std::cout << "[Batched Loop] Keyframe " << kfid << " transformed "
                    << gaussians_transformed_by_this_kf << " points"
                    << std::endl;

          // Collect chunks for batch redistribution
          chunks_to_redistribute.push_back(relevant_chunk_ids);

          // Give loop keyframes times of use
          increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
        }
      }

      // Update keyframe pose
      pkf->setPose(pose.unit_quaternion().cast<double>(),
                   pose.translation().cast<double>());
      pkf->computeTransformTensors();
    }
  }

  // Step 4: Batch redistribution at the end
  if (!chunks_to_redistribute.empty()) {
    torch::Tensor all_redistrib_chunks = torch::cat(chunks_to_redistribute, 0);
    torch::Tensor unique_redistrib_chunks =
        std::get<0>(torch::_unique2(all_redistrib_chunks));

    std::cout << "[Batched Loop] Batch redistributing "
              << unique_redistrib_chunks.size(0) << " chunks" << std::endl;
    gaussians_->handleBatchChunkRedistribution(unique_redistrib_chunks);
  }

  return total_transformed;
}

int GaussianMapper::processSequentialLoopClosure(
    const std::vector<std::tuple<unsigned long,
                                 unsigned long,
                                 Sophus::SE3f,
                                 cv::Mat,
                                 bool,
                                 cv::Mat,
                                 std::vector<float>,
                                 std::vector<float>,
                                 std::string>>& associated_kfs,
    float loop_kf_scale) {
  int total_transformed = 0;

  // Track gaussians visible from loop closure keyframes (for opacity reset)
  torch::Tensor loop_closure_opacity_mask = torch::zeros(
      {gaussians_->xyz_.size(0)}, torch::TensorOptions()
                                      .dtype(torch::kBool)
                                      .device(gaussians_->device_type_));
  int loop_closure_kf_count = 0;

  // Use the original implementation with individual tensor tracking
  torch::Tensor transformed_gaussian_ids =
      torch::full({static_cast<long>(gaussians_->countAllGaussians() * 1.1)},
                  -1,  // -1 = empty slot
                  torch::TensorOptions()
                      .dtype(torch::kInt64)
                      .device(gaussians_->device_type_));
  int next_slot = 0;

  auto updateTransformTracking = [&](const torch::Tensor& old_flags,
                                     const torch::Tensor& new_flags) {
    torch::Tensor newly_transformed_mask = new_flags & (~old_flags);
    torch::Tensor new_indices = torch::where(newly_transformed_mask)[0];

    if (new_indices.size(0) > 0) {
      torch::Tensor new_ids = gaussians_->gaussian_ids_.index({new_indices});
      int end_slot = next_slot + new_ids.size(0);
      transformed_gaussian_ids.slice(0, next_slot, end_slot).copy_(new_ids);
      next_slot = end_slot;
    }
  };

  auto getCurrentTransformFlags = [&]() -> torch::Tensor {
    torch::Tensor valid_ids = transformed_gaussian_ids.slice(0, 0, next_slot);
    return torch::isin(gaussians_->gaussian_ids_, valid_ids);
  };

  // Process keyframes sequentially (original implementation)
  for (auto& kf : associated_kfs) {
    auto kfid = std::get<0>(kf);
    std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

    if (pkf) {
      auto& pose = std::get<2>(kf);
      Sophus::SE3f original_pose = pkf->getPosef();
      Sophus::SE3f inv_pose = pose.inverse();
      Sophus::SE3f diff_pose = inv_pose * original_pose;

      bool large_rot = !diff_pose.rotationMatrix().isApprox(
          Eigen::Matrix3f::Identity(), large_rot_th_);
      bool large_trans =
          !diff_pose.translation().isMuchSmallerThan(1.0, large_trans_th_);

      // Check if this is a loop closure keyframe (where two reconstructions
      // merge)
      bool is_loop_closure_kf = std::get<4>(kf);
      if (is_loop_closure_kf) {
        loop_closure_kf_count++;
        std::cout << "[Sequential Loop] Loop closure keyframe detected: "
                  << kfid << std::endl;
        // Mark visible gaussians for opacity reset
        torch::Tensor visible_gaussians = gaussians_->cullVisibleGaussians(pkf);
        loop_closure_opacity_mask =
            loop_closure_opacity_mask | visible_gaussians;
      }

      if (large_rot || large_trans) {
        // Reset depth loss weight for keyframe that underwent large
        // transformation
        pkf->resetDepthLossWeight();

        std::cout << "[Sequential Loop] Large loop correction detected for kf"
                  << kfid << std::endl;

        torch::Tensor diff_pose_tensor =
            tensor_utils::EigenMatrix2TorchTensor(diff_pose.matrix(),
                                                  gaussians_->device_type_)
                .transpose(0, 1);

        std::vector<ChunkCoord> visible_chunk_coords =
            gaussians_->frustumCullChunks(pkf, /*use_cache=*/true);

        torch::Tensor visible_chunk_coords_tensor = chunkCoordVectorToTensor(
            visible_chunk_coords, gaussians_->device_type_);
        torch::Tensor visible_chunk_ids =
            encodeChunkCoordsTensor(visible_chunk_coords_tensor);

        torch::Tensor loaded_mask = torch::isin(
            visible_chunk_ids, gaussians_->chunks_loaded_from_disk_);
        torch::Tensor on_disk_mask =
            torch::isin(visible_chunk_ids, gaussians_->chunks_on_disk_);
        torch::Tensor spatial_chunks =
            std::get<0>(torch::_unique2(gaussians_->gaussian_chunk_ids_));
        torch::Tensor has_gaussians_mask =
            torch::isin(visible_chunk_ids, spatial_chunks);

        torch::Tensor relevant_mask =
            loaded_mask | on_disk_mask | has_gaussians_mask;
        torch::Tensor relevant_chunk_ids =
            visible_chunk_ids.index({relevant_mask});

        if (relevant_chunk_ids.size(0) <= 0) {
          continue;
        }

        gaussians_->loadChunks(relevant_chunk_ids);

        torch::Tensor old_transform_flags = getCurrentTransformFlags();
        torch::Tensor current_transform_flags = old_transform_flags.clone();

        int gaussians_transformed_by_this_kf = 0;

        gaussians_->scaledTransformVisiblePointsOfKeyframe(
            current_transform_flags, diff_pose_tensor,
            pkf->world_view_transform_, pkf->full_proj_transform_,
            pkf->creation_iter_, stableNumIterExistence(),
            gaussians_transformed_by_this_kf, loop_kf_scale);

        updateTransformTracking(old_transform_flags, current_transform_flags);

        total_transformed += gaussians_transformed_by_this_kf;
        std::cout << "[Sequential Loop] Keyframe " << kfid << " transformed "
                  << gaussians_transformed_by_this_kf << " points" << std::endl;

        gaussians_->handleBatchChunkRedistribution(relevant_chunk_ids);

        increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
      }

      pkf->setPose(pose.unit_quaternion().cast<double>(),
                   pose.translation().cast<double>());
      pkf->computeTransformTensors();
    }
  }

  // Reset optimizer state for ALL transformed gaussians
  torch::Tensor final_transform_mask = getCurrentTransformFlags();
  if (torch::any(final_transform_mask).item<bool>()) {
    std::cout << "[Sequential Loop] Resetting optimizer state for transformed "
                 "gaussians"
              << std::endl;
    gaussians_->resetPositionLRAndOptimizerState(final_transform_mask);
  }

  // Reset opacity ONLY for gaussians visible from loop closure keyframes
  if (torch::any(loop_closure_opacity_mask).item<bool>()) {
    std::cout
        << "[Sequential Loop] Resetting opacity for gaussians visible from "
        << loop_closure_kf_count << " loop closure keyframes" << std::endl;
    gaussians_->resetOpacityForMask(loop_closure_opacity_mask);
  }

  return total_transformed;
}

void GaussianMapper::processScaleRefinement(ORB_SLAM3::MappingOperation& opr) {
  throw std::runtime_error("Scale refinement not implemented!");
  // Existing scale refinement code...
  // std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming "
  //              "all kfs and points..."
  //           << std::endl;

  // float s = opr.mfScale;
  // Sophus::SE3f& T = opr.mT;
  // if (initial_mapped_) {
  //   // Apply the scaled transformation on ALL gaussian model points,
  //   // including those on disk
  //   {
  //     std::unique_lock<std::mutex> lock_render(mutex_render_);

  //     // Get all existing chunk coordinates (both in memory and on disk)
  //     std::vector<ChunkCoord> all_chunks =
  //         chunk_manager_->getExistingChunkCoords();

  //     std::cout << "Applying scale transformation to " << all_chunks.size()
  //               << " chunks" << std::endl;

  //     // Process chunks in batches to manage memory
  //     const int batch_size = 5;  // Adjust based on memory constraints
  //     for (size_t i = 0; i < all_chunks.size(); i += batch_size) {
  //       size_t end = std::min(i + batch_size, all_chunks.size());

  //       // Process current batch
  //       for (size_t j = i; j < end; j++) {
  //         const auto& coord = all_chunks[j];
  //         if (chunk_manager_->loadChunkSync(coord, true)) {
  //           {
  //             std::shared_ptr<Chunk> chunk =
  //             chunk_manager_->getChunkAt(coord); ChunkOptimizationGuard
  //             guard(chunk_manager_.get(), {chunk});
  //             chunk->getGaussians()->applyScaledTransformation(s, T);
  //           }  // Guard automatically releases here
  //           chunk_manager_->saveChunkAsync(coord);
  //         }
  //       }
  //     }
  //   }
  //   // Apply the scaled transformation to the scene
  //   scene_->applyScaledTransformation(s, T);
  // } else {  // TODO: the workflow should not come here, delete this
  //           // branch
  //   // Apply the scaled transformation to the cached points
  //   for (auto& pt : scene_->cached_point_cloud_) {
  //     // pt <- (s * Ryw * pt + tyw)
  //     auto& pt_xyz = pt.second.xyz_;
  //     pt_xyz *= s;
  //     pt_xyz = T.cast<double>() * pt_xyz;
  //   }

  //   // Apply the scaled transformation on gaussian keyframes
  //   for (auto& kfit : scene_->keyframes()) {
  //     std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
  //     Sophus::SE3f Twc = pkf->getPosef().inverse();
  //     Twc.translation() *= s;
  //     Sophus::SE3f Tyc = T * Twc;
  //     Sophus::SE3f Tcy = Tyc.inverse();
  //     pkf->setPose(Tcy.unit_quaternion().cast<double>(),
  //                  Tcy.translation().cast<double>());
  //     pkf->computeTransformTensors();
  //   }
  // }

  // Gaussians will be all over the place, transfer them to their
  // respective chunks
  // chunk_manager_->transferGaussiansAcrossChunks();
}

// Common keyframe initialization logic used by both ORB-SLAM and external modes
void GaussianMapper::createAndInitializeKeyframe(
    std::shared_ptr<GaussianKeyframe>& pkf,
    cv::Mat& rgb_image,
    cv::Mat& aux_image,
    const Camera& camera,
    const std::string& filename) {
  // Set z clipping planes
  pkf->zfar_ = z_far_ * scene_->cameras_extent_;
  pkf->znear_ = z_near_ * scene_->cameras_extent_;

  // Set camera parameters
  pkf->setCameraParams(camera);
  pkf->img_filename_ = filename;
  pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
  pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
  pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

  // Add the new keyframe to the scene
  pkf->computeTransformTensors();
  scene_->addKeyframe(pkf);
  kfid_shuffled_ = false;

  // Update chunk-keyframe mapping if using strategy 1
  if (keyframe_selection_strategy_ == 1) {
    keyframe_queue_->updateChunkKeyframeMapping(pkf, true);
  }

  // Give new keyframes times of use and add it to the training sliding window
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

  // Sample gaussians (requires render lock)
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  sampleGaussians(pkf);

  pkf->allow_eviction_ = true;
}

void GaussianMapper::handleNewKeyframeFromORBSLAM(
    std::tuple<unsigned long /*Id*/,
               unsigned long /*CameraId*/,
               Sophus::SE3f /*pose*/,
               cv::Mat /*image*/,
               bool /*isLoopClosure*/,
               cv::Mat /*auxiliaryImage*/,
               std::vector<float>,
               std::vector<float>,
               std::string>& kf) {
  // Create keyframe
  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(
      std::get<0>(kf), getIteration(), keyframe_save_dir_);

  // Set pose from ORB-SLAM data
  auto& pose = std::get<2>(kf);
  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());

  // Extract images and camera from ORB-SLAM tuple
  cv::Mat imgRGB_undistorted = std::get<3>(kf);
  cv::Mat imgAux_undistorted = std::get<5>(kf);

  try {
    Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    std::string filename = std::get<8>(kf);

    // Store ORB-SLAM specific keypoints
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));

    // Call common initialization logic
    createAndInitializeKeyframe(pkf, imgRGB_undistorted, imgAux_undistorted,
                                camera, filename);

    // Release ORB-SLAM resources
    pSLAM_->getAtlas()->ReleaseKeyFrameImages(pkf->fid_);

  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[GaussianMapper::handleNewKeyframeFromORBSLAM] KeyFrame Camera not "
        "found!");
  }
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times) {
  pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes() {
  std::unordered_set<unsigned long> kfids =
      pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
  std::vector<unsigned long> kfids_to_erase;
  std::size_t nkfs = scene_->keyframes().size();
  kfids_to_erase.reserve(nkfs);
  for (auto& kfit : scene_->keyframes()) {
    unsigned long kfid = kfit.first;
    if (kfids.find(kfid) == kfids.end()) {
      kfids_to_erase.emplace_back(kfid);
    }
  }

  for (auto& kfid : kfids_to_erase) {
    scene_->keyframes().erase(kfid);
  }
}

void GaussianMapper::sampleGaussians(std::shared_ptr<GaussianKeyframe> pkf) {
  torch::NoGradGuard no_grad;
  auto start_time = std::chrono::steady_clock::now();

  std::vector<std::shared_ptr<GaussianKeyframe>> newly_loaded_keyframes;
  if (!pkf->loaded_) {
    pkf->loadDataFromDisk();
    newly_loaded_keyframes.push_back(pkf);
  }

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  // Step 1: Get RGB image and depth data
  torch::Tensor rgb = pkf->gaus_pyramid_original_image_[0];

  if (downsample_for_sampling_) {
    // Step 1: Downsample by factor of 2 using average pooling
    // avg_pool2d expects [N, C, H, W], so add batch dimension
    rgb = rgb.unsqueeze(0);           // [1, 3, H, W]
    rgb = torch::avg_pool2d(rgb, 2);  // [1, 3, H/2, W/2]

    // // Step 2: Upsample back to original resolution using bilinear
    // interpolation
    rgb = torch::nn::functional::interpolate(
        rgb,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{pkf->image_height_, pkf->image_width_})
            .mode(torch::kBilinear)
            .align_corners(true));

    // Remove batch dimension: [1, 3, H, W] -> [3, H, W]
    rgb = rgb.squeeze(0);
  }

  torch::Tensor depth_confidence = pkf->depth_confidence_;

  // Step 2: Compute initial probability based on image gradients (like
  // Python)
  torch::Tensor init_proba = computeLoGProbability(rgb);

  // Step 3: Render current view and compute penalty (if scene is initialized)
  torch::Tensor penalty = torch::zeros_like(init_proba);
  torch::Tensor rendered_depth;
  torch::Tensor main_gaussian_ids;
  torch::Tensor visible_gaussian_mask;
  bool has_rendered_depth = false;

  if (initial_mapped_) {
    // std::unique_lock<std::mutex> lock_render(mutex_render_);
    visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf);

    torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
    auto render_pkg = GaussianRenderer::render(
        gaussians_, visible_gaussian_mask, pkf, pkf->image_height_,
        pkf->image_width_, pipe_params_, background_, override_color_, 1.0f,
        false, pkf->FoVx_, pkf->FoVy_, view_matrix, pkf->projection_matrix_);

    torch::Tensor rendered_image = std::get<1>(render_pkg);
    rendered_depth = 1 / std::get<0>(render_pkg).clamp_min(1e-8);
    has_rendered_depth = true;
    main_gaussian_ids = std::get<3>(render_pkg)[0];
    penalty = computeLoGProbability(rendered_image);
  }

  // Step 4: Apply scaling factor and compute sampling probability
  init_proba *= init_proba_scaler_;
  penalty *= init_proba_scaler_;
  // std::cout << "Penalty mean: " << penalty.mean().item<float>() <<
  // std::endl;

  // Step 5: Generate initial sample mask based on probability
  torch::Tensor sample_mask =
      torch::rand_like(init_proba) < init_proba - penalty;
  torch::Tensor flat_sample_mask = sample_mask.flatten();

  // std::cout << "Sample mask count: " << sample_mask.sum().item<int>()
  //           << std::endl;

  // Pre-compute UV grid (similar to Python's self.uv)
  torch::Tensor uv_;
  {
    auto x_coords = torch::arange(0, pkf->image_width_, torch::kFloat32).cuda();
    auto y_coords =
        torch::arange(0, pkf->image_height_, torch::kFloat32).cuda();
    auto meshgrid = torch::meshgrid({x_coords, y_coords}, "xy");
    uv_ = torch::stack({meshgrid[0], meshgrid[1]}, -1);
    // std::cout << "UV grid size: " << uv_.sizes() << std::endl;
  }

  // std::cout << "Sample mask size: " << sample_mask.sizes() << std::endl;

  // Get UV coordinates of initially sampled points
  torch::Tensor sampled_uv = uv_.view({-1, 2}).index({sample_mask.flatten()});
  // std::cout << "Sampled UV size: " << sampled_uv.sizes() << std::endl;

  // Get closest keyframes for MVS
  std::vector<std::shared_ptr<GaussianKeyframe>> prev_keyframes =
      getClosestKeyframes(pkf, guided_mvs_->getNumCams(), 6);

  for (const auto& kf : prev_keyframes) {
    if (!kf->loaded_) {
      // std::cout << "Loading keyframe " << std::to_string(kf->fid_)
      //           << " from CPU for sampling" << std::endl;
      kf->loadDataFromDisk();
      newly_loaded_keyframes.push_back(kf);
    } else {
      // std::cout << "Keyframe " << std::to_string(kf->fid_)
      //           << " already marked as loaded for sampling" << std::endl;
    }
  }

  torch::Tensor accurate_mask, depth;

  if (prev_keyframes.size() != guided_mvs_->getNumCams()) {
    std::cout << "Not enough previous keyframes found for MVS." << std::endl;
    assert(pkf->gaus_pyramid_inv_depth_image_[0].defined());

    torch::Tensor depth_map =
        1 / pkf->gaus_pyramid_inv_depth_image_[0].clamp_min(1e-8);
    // std::cout << "Depth map size: " << depth_map.sizes() << std::endl;
    torch::Tensor sample_indices = torch::nonzero(flat_sample_mask).squeeze(-1);
    // std::cout << "sample_indices size: " << sample_indices.sizes() <<
    // std::endl;
    torch::Tensor depth_map_flat = depth_map.flatten();
    // std::cout << "depth_map_flat size: " << depth_map_flat.sizes() <<
    // std::endl;
    depth = depth_map_flat.index({sample_indices});
    // std::cout << "depth size: " << depth.sizes() << std::endl;
    // Set accurate mask to all ones (since we're not using MVS)
    accurate_mask = torch::ones_like(depth, torch::kBool);
    // std::cout << "accurate_mask size: " << accurate_mask.sizes() <<
    // std::endl;

  } else {
    // Apply guided MVS - returns depth and accurate mask for sampled points
    auto start_time_mvs = std::chrono::steady_clock::now();

    std::tie(depth, accurate_mask) =
        (*guided_mvs_)(sampled_uv, pkf, prev_keyframes);
    auto end_time_mvs = std::chrono::steady_clock::now();
    auto duration_mvs = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time_mvs - start_time_mvs);
  }

  // std::cout << "MVS completed in " << duration_mvs.count() << "ms" <<
  // std::endl;

  // auto [depth, accurate_mask, debug_stats] = guided_mvs_->operator_debug(
  //     sampled_uv, pkf, prev_keyframes, true, "kitti_scene_10");

  // guided_mvs_->debug_specific_point(sampled_uv, pkf, prev_keyframes, 420);

  // Apply confidence filtering exactly like Python
  torch::Tensor sampled_confidence = sampleConf(
      depth_confidence, sampled_uv, pkf->image_width_, pkf->image_height_);
  // torch::Tensor valid_mask =
  //     (depth > 1e-6) & (sampled_confidence > 0.5) & (depth < 100.0f);
  torch::Tensor valid_mask = (depth > 1e-6) & (sampled_confidence > 0.5);

  // std::cout << "Valid mask count: " << valid_mask.sum().item<int>()
  //           << std::endl;

  // Update the sample_mask correctly
  torch::Tensor original_sample_indices =
      torch::nonzero(flat_sample_mask).squeeze(-1);
  torch::Tensor valid_sample_indices =
      original_sample_indices.index({valid_mask});

  // Reset sample_mask and set only valid positions
  sample_mask.fill_(false);
  sample_mask.view(-1).index_put_({valid_sample_indices}, true);

  // Filter other tensors
  depth = depth.index({valid_mask});
  sampled_uv = sampled_uv.index({valid_mask});
  accurate_mask = accurate_mask.index({valid_mask});

  // Gaussian replacement and occlusion checks (same as before)
  // std::cout << "=== Starting Gaussian Replacement and Occlusion Checks ==="
  //           << std::endl;
  // std::cout << "Initial valid samples: " << depth.size(0) << std::endl;

  // Handle Gaussian removal (only if we have existing Gaussians and rendered
  // depth)
  if (has_rendered_depth) {
    // std::cout << "Processing Gaussian removal for coarser Gaussians..."
    //           << std::endl;

    torch::Tensor accurate_sample_mask = torch::zeros_like(sample_mask);
    torch::Tensor current_flat_indices =
        sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
    torch::Tensor accurate_positions =
        current_flat_indices.index({accurate_mask});
    accurate_sample_mask.view(-1).index_put_(
        {accurate_positions.to(torch::kLong)}, true);

    // std::cout << "Accurate samples for Gaussian removal: "
    //           << accurate_sample_mask.sum().item<int>() << std::endl;

    if (accurate_sample_mask.any().item<bool>()) {
      torch::Tensor selected_main_gaussians =
          main_gaussian_ids.index({accurate_sample_mask});
      torch::Tensor valid_ids_mask = selected_main_gaussians >= 0;
      // std::cout << "Valid Gaussian IDs found: "
      //           << valid_ids_mask.sum().item<int>() << std::endl;

      if (valid_ids_mask.any().item<bool>()) {
        selected_main_gaussians =
            selected_main_gaussians.index({valid_ids_mask});

        auto unique_result = torch::_unique2(selected_main_gaussians,
                                             /*sorted=*/false,
                                             /*return_inverse=*/false,
                                             /*return_counts=*/true);
        torch::Tensor unique_ids = std::get<0>(unique_result);
        torch::Tensor counts = std::get<2>(unique_result);

        // std::cout << "Found " << unique_ids.size(0) << " unique Gaussians"
        //           << std::endl;

        torch::Tensor removal_mask = counts >= 10;

        if (removal_mask.any().item<bool>()) {
          torch::Tensor gaussians_to_remove_subset =
              unique_ids.index({removal_mask});
          torch::Tensor visible_indices =
              torch::where(visible_gaussian_mask)[0];
          torch::Tensor gaussians_to_remove_full =
              visible_indices.index({gaussians_to_remove_subset});

          torch::Tensor full_model_prune_mask = torch::zeros(
              {gaussians_->getXYZ().size(0)},
              torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

          full_model_prune_mask.index_put_({gaussians_to_remove_full}, true);
          gaussians_->prunePoints(full_model_prune_mask);

          visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf);

          torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
          auto updated_render_pkg = GaussianRenderer::render(
              gaussians_, visible_gaussian_mask, pkf, pkf->image_height_,
              pkf->image_width_, pipe_params_, background_, override_color_,
              1.0f, false, pkf->FoVx_, pkf->FoVy_, view_matrix,
              pkf->projection_matrix_);

          rendered_depth = 1 / std::get<0>(updated_render_pkg).clamp_min(1e-8);
          // std::cout << "Re-rendered scene after Gaussian removal"
          //           << std::endl;

        } else {
          // std::cout << "No Gaussians need removal (all counts < 10)"
          //           << std::endl;
        }
      }
    }
  }

  // Check for occlusions
  if (has_rendered_depth) {
    // std::cout << "Checking for occlusions..." << std::endl;

    torch::Tensor current_flat_indices =
        sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
    torch::Tensor rendered_depth_flat = rendered_depth.flatten();
    torch::Tensor rendered_depth_sampled =
        rendered_depth_flat.index({current_flat_indices.to(torch::kLong)});

    torch::Tensor occlusion_mask = depth < rendered_depth_sampled;

    // std::cout << "Samples passing occlusion check: "
    //           << occlusion_mask.sum().item<int>() << " / " << depth.size(0)
    //           << std::endl;

    // Filter all our data by occlusion mask
    depth = depth.index({occlusion_mask});
    sampled_uv = sampled_uv.index({occlusion_mask});
    accurate_mask = accurate_mask.index({occlusion_mask});

    // Update the global sample_mask to reflect final surviving samples
    sample_mask.fill_(false);
    if (depth.size(0) > 0) {
      torch::Tensor final_flat_indices =
          sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
      sample_mask.view(-1).index_put_({final_flat_indices.to(torch::kLong)},
                                      true);
    }

    // std::cout << "Final samples after all filtering: " << depth.size(0)
    //           << std::endl;
  } else {
    // std::cout << "No rendered depth available, skipping occlusion check"
    //           << std::endl;
  }

  // Early exit if no samples remain
  if (depth.size(0) == 0) {
    // std::cout << "No samples remain after filtering, exiting" << std::endl;
    return;
  }

  // Step 8: Get matched keypoints and their 3D positions BEFORE flattening
  // RGB
  torch::Tensor match_pts_3d;
  torch::Tensor match_colors;
  torch::Tensor match_init_proba;
  int num_matched_points = 0;

  // Check if we have valid keypoints with 3D coordinates
  if (!pkf->kps_pixel_.empty() && !pkf->kps_point_local_.empty()) {
    int num_keypoints = pkf->kps_pixel_.size() / 2;
    // std::cout << "Processing " << num_keypoints << " keypoints" <<
    // std::endl;

    // Convert vectors to tensors directly on GPU for vectorized operations
    torch::Tensor kps_pixel_tensor =
        torch::from_blob(pkf->kps_pixel_.data(), {num_keypoints, 2},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);

    torch::Tensor kps_point_local_tensor =
        torch::from_blob(pkf->kps_point_local_.data(), {num_keypoints, 3},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);

    // Create validity mask using vectorized operations
    torch::Tensor u_coords = kps_pixel_tensor.select(1, 0);
    torch::Tensor v_coords = kps_pixel_tensor.select(1, 1);
    torch::Tensor z_coords = kps_point_local_tensor.select(1, 2);

    torch::Tensor valid_mask =
        (z_coords > 1e-6) & (u_coords >= 0) & (u_coords < pkf->image_width_) &
        (v_coords >= 0) & (v_coords < pkf->image_height_) &
        torch::isfinite(kps_point_local_tensor.select(1, 0)) &
        torch::isfinite(kps_point_local_tensor.select(1, 1)) &
        torch::isfinite(z_coords);

    num_matched_points = valid_mask.sum().item<int>();
    // std::cout << "Valid matched points: " << num_matched_points <<
    // std::endl;

    if (num_matched_points > 0) {
      // Extract valid keypoints using mask indexing
      torch::Tensor valid_kps_pixel = kps_pixel_tensor.index({valid_mask});
      match_pts_3d = kps_point_local_tensor.index({valid_mask});

      // Convert pixel coordinates to normalized grid coordinates [-1, 1] for
      // grid_sample
      torch::Tensor normalized_coords = torch::zeros(
          {num_matched_points, 2},
          torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
      normalized_coords.select(1, 0) =
          2.0f * valid_kps_pixel.select(1, 0) / (pkf->image_width_ - 1) -
          1.0f;  // x
      normalized_coords.select(1, 1) =
          2.0f * valid_kps_pixel.select(1, 1) / (pkf->image_height_ - 1) -
          1.0f;  // y

      // Reshape for grid_sample: [1, 1, N, 2]
      torch::Tensor grid =
          normalized_coords.view({1, 1, num_matched_points, 2});

      // std::cout << "Grid size: " << grid.sizes() << std::endl;
      // std::cout << "RGB size before sampling: " << rgb.sizes() <<
      // std::endl;

      // Use grid_sample for RGB (expects [N, C, H, W] format)
      torch::Tensor rgb_for_sampling = rgb.unsqueeze(0);  // [1, 3, H, W]
      torch::Tensor sampled_colors_raw = torch::nn::functional::grid_sample(
          rgb_for_sampling, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kBilinear)
              .align_corners(true));  // [1, 3, 1, N]

      match_colors =
          sampled_colors_raw.squeeze(0).squeeze(1).transpose(0, 1);  // [N, 3]

      // Use grid_sample for init_proba
      torch::Tensor init_proba_for_sampling =
          init_proba.unsqueeze(0).unsqueeze(0);  // [1, 1, H, W]
      torch::Tensor sampled_proba_raw = torch::nn::functional::grid_sample(
          init_proba_for_sampling, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kBilinear)
              .align_corners(true));  // [1, 1, 1, N]

      match_init_proba = sampled_proba_raw.squeeze();  // [N]

      // std::cout << "match_colors size: " << match_colors.sizes() <<
      // std::endl; std::cout << "match_init_proba size: " <<
      // match_init_proba.sizes()
      //           << std::endl;
      // std::cout << "Found " << num_matched_points << " valid matched
      // keypoints"
      //           << std::endl;
    }
  }

  // Step 9: NOW flatten RGB for the regular sampled points extraction
  // Convert RGB to flat format for efficient sampling of regular points
  rgb = rgb.permute({1, 2, 0}).flatten(0, 1);  // [H*W, 3]
  init_proba = init_proba.flatten();           // [H*W]

  // Extract sampled data using the filtered results from MVS
  torch::Tensor flat_indices =
      sampled_uv.select(1, 1) * pkf->image_width_ + sampled_uv.select(1, 0);
  torch::Tensor sampled_colors = rgb.index({flat_indices.to(torch::kLong)});
  torch::Tensor sampled_init_proba =
      init_proba.index({flat_indices.to(torch::kLong)});

  // std::cout << "Sampled colors size: " << sampled_colors.sizes() <<
  // std::endl; std::cout << "Sampled init proba size: " <<
  // sampled_init_proba.sizes()
  //           << std::endl;

  // Step 10: Reproject sampled points to 3D
  float fx = pkf->intr_[0];
  float fy = pkf->intr_[1];
  float cx = pkf->intr_[2];
  float cy = pkf->intr_[3];

  torch::Tensor u_coords = sampled_uv.select(1, 0);  // [N]
  torch::Tensor v_coords = sampled_uv.select(1, 1);  // [N]

  torch::Tensor sampled_points3D =
      torch::zeros({depth.size(0), 3}, depth.options());
  sampled_points3D.select(1, 0) = (u_coords - cx) * depth / fx;  // X
  sampled_points3D.select(1, 1) = (v_coords - cy) * depth / fy;  // Y
  sampled_points3D.select(1, 2) = depth;                         // Z

  // Step 11: Combine sampled points and matched points
  torch::Tensor all_points3D;
  torch::Tensor all_colors;
  torch::Tensor all_init_proba;

  if (num_matched_points > 0) {
    // Concatenate sampled and matched points
    all_points3D = torch::cat({sampled_points3D, match_pts_3d}, 0);
    all_colors = torch::cat({sampled_colors, match_colors}, 0);
    all_init_proba = torch::cat({sampled_init_proba, match_init_proba}, 0);

    // std::cout << "Combined " << sampled_points3D.size(0)
    //           << " sampled points with " << num_matched_points
    //           << " matched points" << std::endl;
  } else {
    // Only sampled points
    all_points3D = sampled_points3D;
    all_colors = sampled_colors;
    all_init_proba = sampled_init_proba;

    // std::cout << "Using only " << sampled_points3D.size(0) << " sampled
    // points "
    //           << std::endl;
  }

  // Transform all points to world coordinates
  torch::Tensor Twc_tensor =
      tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(all_points3D, Twc_tensor);

  // std::cout << "All Points3D size: " << all_points3D.sizes() << std::endl;

  // Step 12: Compute scales for all points (following Python implementation)
  torch::Tensor scales = 1.0f / torch::sqrt(all_init_proba + 1e-8f);
  scales =
      torch::clamp(scales, 1.0f, static_cast<float>(pkf->image_width_) / 10.0f);
  scales *= (1.0f / pkf->intr_[0]);  // fx

  // Scale by distance to camera center
  torch::Tensor diff = all_points3D - pkf->getCenter().unsqueeze(0);
  torch::Tensor distances = torch::norm(diff, 2, 1);
  scales *= distances;
  scales = torch::log(torch::clamp(scales, 1e-6f, 1e6f));
  torch::Tensor all_scales = scales.unsqueeze(1).repeat({1, 3});

  // std::cout << "All scales size: " << all_scales.sizes() << std::endl;

  // Step 13: Set opacities based on point type (sampled vs matched)
  torch::Tensor all_opacities = torch::zeros(
      {all_points3D.size(0), 1}, torch::TensorOptions().device(device_type_));

  int num_sampled = sampled_points3D.size(0);

  if (num_sampled > 0) {
    // Set opacities for sampled points based on accuracy (like Python)
    torch::Tensor sampled_accurate_opacity = torch::full(
        {num_sampled, 1}, 0.07f, torch::TensorOptions().device(device_type_));
    torch::Tensor sampled_inaccurate_opacity = torch::full(
        {num_sampled, 1}, 0.02f, torch::TensorOptions().device(device_type_));

    torch::Tensor sampled_opacities =
        torch::where(accurate_mask.unsqueeze(-1), sampled_accurate_opacity,
                     sampled_inaccurate_opacity);

    // Fill in sampled point opacities
    all_opacities.slice(0, 0, num_sampled) = sampled_opacities;
  }

  if (num_matched_points > 0) {
    // Set higher opacity for matched keypoints (like Python: 0.2)
    torch::Tensor matched_opacities =
        torch::full({num_matched_points, 1}, 0.2f,
                    torch::TensorOptions().device(device_type_));

    // Fill in matched point opacities
    all_opacities.slice(0, num_sampled, num_sampled + num_matched_points) =
        matched_opacities;
  }

  // std::cout << "All opacities size: " << all_opacities.sizes() <<
  // std::endl;

  // Step 14: Add all points to the scene in a single call
  // std::unique_lock lock_render(mutex_render_);

  auto start_time_prune = std::chrono::steady_clock::now();

  if (initial_mapped_) {
    // std::cout << "Pruning low opacity gaussians" << std::endl;
    gaussians_->pruneLowOpacityGaussians(pkf, visible_gaussian_mask);
  }

  auto end_time_prune = std::chrono::steady_clock::now();
  auto duration_prune = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time_prune - start_time_prune);
  // std::cout << "pruneLowOpacityGaussians completed in "
  //           << duration_prune.count() << "ms" << std::endl;

  // std::cout << "Adding " << all_points3D.size(0) << " total points to
  // scene("
  //           << num_sampled << " sampled + " << num_matched_points <<
  //           "matched) "
  //           << std::endl;

  // Convert opacities using inverse sigmoid (like
  // Python)
  torch::Tensor final_opacities = general_utils::inverse_sigmoid(all_opacities);

  gaussians_->addPoints(all_points3D, all_colors, all_scales, final_opacities,
                        getIteration(), scene_->cameras_extent_);

  // Track which chunks received new gaussians and add optimization budget
  // if (keyframe_selection_strategy_ == 1 && all_points3D.size(0) > 0) {
  //   keyframe_queue_->addBudgetFromGaussianPositions(all_points3D);
  // }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // std::cout << "addPoints complete" << std::endl;

  // Later, save only the keyframes that were loaded
  for (const auto& kf : newly_loaded_keyframes) {
    // std::cout << "Saving keyframe " << std::to_string(kf->fid_)
    //           << " back to CPU" << std::endl;
    kf->saveDataToDisk();
  }
  // std::cout << "sampleGaussians completed in " << duration.count() << "ms"
  //           << std::endl;
}