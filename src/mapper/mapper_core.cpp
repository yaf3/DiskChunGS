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
#include "rendering/triangle_rasterizer.h"
#include "rendering/triangle_renderer.h"
#include "utils/loss_utils.h"
#include "utils/profiling.h"
#include "utils/trajectory_viewer.h"

// Helper function to query current RAM usage from /proc/self/status
float getCurrentRAMUsageMB() {
  std::ifstream status_file("/proc/self/status");
  std::string line;
  while (std::getline(status_file, line)) {
    if (line.substr(0, 6) == "VmRSS:") {
      std::istringstream iss(line);
      std::string name, value, unit;
      iss >> name >> value >> unit;
      return std::stof(value) / 1024.0f;  // Convert from KB to MB
    }
  }
  return 0.0f;  // Return 0 if unable to read
}

void TriangleMapper::run() {
  std::chrono::steady_clock::time_point training_start =
      std::chrono::steady_clock::now();
  training_start_time_ = training_start;

  // Could be used to normalize scene size (e.g. for learning rates, not used
  // now)
  scene_->cameras_extent_ = 1.0f;

  // Delete existing chunks since training
  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  std::filesystem::remove_all(keyframe_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

  // First loop: Initial triangle mapping
  while (!isStopped()) {
    // Check conditions for initial mapping
    if (hasMetInitialMappingConditions()) {
      pSLAM_->getAtlas()->clearMappingOperation();

      // Get initial sparse map
      auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
      std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
      std::vector<ORB_SLAM3::MapPoint*> vpMPs;
      {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        vpKFs = pMap->GetAllKeyFrames();

        for (const auto& pKF : vpKFs) {
          // Get keypoint info
          std::vector<float> pixels;
          std::vector<float> pointsLocal;
          pKF->GetKeypointInfo(pixels, pointsLocal);

          // Create tuple to pass into handleNewKeyframeFromORBSLAM
          KeyframeTuple kf_tuple = std::make_tuple(
                  pKF->mnId,               // Id
                  pKF->mpCamera->GetId(),  // CameraId
                  pKF->GetPose(),          // pose
                  pKF->imgLeftRGB,         // image
                  false,                   // isLoopClosure
                  pKF->imgAuxiliary,       // auxiliaryImage
                  std::move(pixels),       // keypoint pixels
                  std::move(pointsLocal),  // keypoint points local
                  pKF->mNameFile           // filename
              );

          handleNewKeyframeFromORBSLAM(kf_tuple);

          // Setup training on first keyframe
          if (!initial_mapped_) {
            triangles_->trainingSetup(opt_params_);
            initial_mapped_ = true;
          }
        }
      }

      // Invoke training once
      trainForOneIteration();

      // Finish initial mapping loop
      break;
    } else if (pSLAM_->isShutDown()) {
      break;
    } else {
      // Initial conditions not satisfied
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Second loop: Incremental triangle mapping
  int SLAM_stop_iter = 0;
  while (!isStopped()) {
    auto timer_TotalLoop = ProfilingUtils::Timer("TotalLoop");

    // Check conditions for incremental mapping
    if (hasMetIncrementalMappingConditions()) {
      combineMappingOperations();
    }

    // Invoke training once
    trainForOneIteration();
    timer_TotalLoop.stop();

    if (pSLAM_->isShutDown()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
    }

    if (SLAM_ended_) {
      break;
    }
  }

  std::chrono::steady_clock::time_point training_end =
      std::chrono::steady_clock::now();
  double total_time_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(training_end -
                                                                training_start)
          .count();
  std::ofstream out((result_dir_ / "training_time.txt").string());
  if (out.is_open()) {
    out << std::fixed << std::setprecision(4) << total_time_seconds
        << std::endl;
    out.close();
    std::cout << "Saved training time: " << std::fixed << std::setprecision(4)
              << total_time_seconds << " seconds to "
              << (result_dir_ / "training_time.txt").string() << std::endl;
  } else {
    std::cerr << "Warning: Could not save training time to "
              << (result_dir_ / "training_time.txt").string() << std::endl;
  }

  // Finalization: save outputs and clean up
  saveTotalTriangles("_shutdown");
  renderAndRecordAllKeyframes("_shutdown");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
  writeTrainingMetricsCSV(result_dir_);

  std::filesystem::remove_all(chunk_save_dir_);
  std::filesystem::remove_all(keyframe_save_dir_);

  signalStop();

  if (completion_callback_) {
    completion_callback_();
  }
}

void TriangleMapper::trainForOneIteration() {
  increaseIteration(1);

  // Collect training metrics at regular intervals
  int current_iteration = getIteration();
  if (current_iteration % metrics_collection_interval_ == 0) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - training_start_time_);
    double elapsed_seconds = elapsed.count() / 1000.0;

    int totalTriangles = triangles_->countAllTriangles();
    int activeTriangles = int(triangles_->getXYZ().size(0));

    // Get VRAM usage
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes =
        mem_stats
            .reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float reserved_MB = reserved_bytes.current / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float alloc_MB = alloc_bytes.current / (1024.0 * 1024.0);

    // Store metrics
    TrainingMetrics metrics;
    metrics.iteration = current_iteration;
    metrics.elapsed_time_seconds = elapsed_seconds;
    metrics.active_triangle_count = activeTriangles;
    metrics.total_triangle_count = totalTriangles;
    metrics.reserved_memory_mb = reserved_MB;
    metrics.allocated_memory_mb = alloc_MB;
    metrics.ram_usage_mb = getCurrentRAMUsageMB();
    metrics.queue_keyframes = keyframe_selector_->getQueueSize();

    training_metrics_.push_back(metrics);
  }

  auto iter_start_timing = std::chrono::steady_clock::now();

  // Select keyframe for training
  std::shared_ptr<TriangleKeyframe> viewpoint_cam =
      keyframe_selector_->getNextKeyframe();

  if (!viewpoint_cam) {
    increaseIteration(-1);
    throw std::runtime_error(
        "[TriangleMapper] Keyframe not found for training");
    return;
  }

  // Record keyframe selection for trajectory viewer
  if (trajectory_viewer_) {
    trajectory_viewer_->recordKeyframeSelection(
        static_cast<int>(viewpoint_cam->fid_));
  }

  // Load keyframe from disk if necessary
  bool had_to_load = false;
  if (!viewpoint_cam->loaded_) {
    std::cout << "Loading keyframe " << std::to_string(viewpoint_cam->fid_)
              << " from disk to GPU" << std::endl;
    viewpoint_cam->loadDataFromDisk();
    had_to_load = true;
  }

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  auto [gt_image, gt_inv_depth, mask, image_height, image_width] =
      viewpoint_cam->getTrainingData(
          undistort_mask_[viewpoint_cam->camera_id_],
          scene_->cameras_.at(viewpoint_cam->camera_id_)
              .gaus_pyramid_undistort_mask_);

  // Sigma schedule: linear anneal from sigma_init to sigma_final.
  {
    int iter = current_iteration;
    int start = opt_params_.sigma_start_iter_;
    int until = opt_params_.sigma_until_iter_;
    float sigma;
    if (iter < start) {
      sigma = opt_params_.sigma_init_;
    } else {
      float progress = static_cast<float>(iter - start) /
                       static_cast<float>(until - start);
      progress = std::min(progress, 1.0f);
      sigma = opt_params_.sigma_init_ -
              (opt_params_.sigma_init_ - opt_params_.sigma_final_) * progress;
    }
    triangles_->sigma_value_ = sigma;
  }

  // Mutex lock for usage of the triangle model (Since we allow rendering at the
  // same time from e.g. GUI)
  std::unique_lock<std::mutex> lock_render(mutex_render_);

  // Get mask for triangles in visible chunks, also load/evict chunks as
  // necessary
  torch::Tensor visible_triangle_mask =
      triangles_->cullVisibleTriangles(viewpoint_cam);

  // Get view matrix of keyframe
  torch::Tensor view_matrix = viewpoint_cam->getRT().transpose(0, 1);

  // Rasterize triangles for selected keyframe
  auto render_pkg =
      TriangleRenderer::render(triangles_, visible_triangle_mask, viewpoint_cam,
                               image_height, image_width, pipe_params_,
                               background_, override_color_, 1.0f, false,
                               viewpoint_cam->FoVx_, viewpoint_cam->FoVy_,
                               view_matrix, viewpoint_cam->full_proj_transform_);

  torch::Tensor rendered_image = std::get<1>(render_pkg);
  torch::Tensor radii = std::get<2>(render_pkg);
  torch::Tensor full_model_scaling = std::get<4>(render_pkg);
  torch::Tensor rend_normal = std::get<5>(render_pkg);  // [3, H, W]

  // Loss calculation
  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto Ll1 = l1_loss(rendered_image, gt_image, 1.0f);
  auto Lssim = loss_utils::fast_ssim(rendered_image, gt_image);
  float lambda_dssim = lambdaDssim();
  auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);

  // Calculate depth loss if gt depth exists
  if (gt_inv_depth.defined()) {
    float lambda_depth = lambdaDepth();
    torch::Tensor rendered_inv_depth = std::get<0>(render_pkg);
    torch::Tensor depth_loss = (rendered_inv_depth - gt_inv_depth).abs().mean();
    loss += lambda_depth * depth_loss;
  }

  // Normal loss: penalise triangle plane normals diverging from surface normals.
  // Mode 1 (self-consistency): rend_normal vs surf_normal from rendered depth.
  // Mode 2 (GT-anchored):      rend_normal vs normals from GT sensor depth.
  int normal_mode = normalLossMode();
  float lambda_normal = lambdaNormal();
  if (normal_mode > 0 && lambda_normal > 0.0f) {
    float fx = viewpoint_cam->intr_[0], fy = viewpoint_cam->intr_[1];
    float cx = viewpoint_cam->intr_[2], cy = viewpoint_cam->intr_[3];
    torch::Tensor ref_normal;
    if (normal_mode == 1) {
      ref_normal = std::get<6>(render_pkg);  // surf_normal from rendered depth
    } else {
      // Mode 2: GT sensor depth (only available for RGBD)
      if (gt_inv_depth.defined()) {
        ref_normal = loss_utils::computeNormalsFromDepth(gt_inv_depth, fx, fy, cx, cy);
      }
    }
    if (ref_normal.defined()) {
      torch::Tensor normal_loss =
          lambda_normal * (1.0f - (rend_normal * ref_normal).sum(0)).mean();
      loss += normal_loss;
    }
  }

  // Vertex weight regularization: penalizes high average weight to encourage pruning.
  float lambda_weight = lambdaWeight();
  if (lambda_weight > 0.0f) {
    loss += lambda_weight * triangles_->getVertexWeightActivation().mean();
  }

  // Backwards pass
  loss.backward();

  // Update keyframe exposure and depth bias/scale parameters. Currently
  // disabled
  // viewpoint_cam->step();

  // Optimizer step
  if (getIteration() < opt_params_.iterations_ ||
      opt_params_.iterations_ == -1) {
    // visibility_filter from renderer is only for visible triangles
    torch::Tensor subset_contributed = (radii > 0);

    // Map back to full model indices
    torch::Tensor full_model_contributed = torch::zeros(
        {triangles_->getXYZ().size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // Compute indices only when needed for mapping radii back to full model
    torch::Tensor visible_triangle_indices =
        torch::nonzero(visible_triangle_mask).squeeze(1);

    // Set true for triangles that were both visible AND had radii > 0
    full_model_contributed.index_put_({visible_triangle_indices},
                                      subset_contributed);

    triangles_->optimizerStep(full_model_contributed);
  }

  // Zero out gradients
  triangles_->optimizer_->zero_grad(true);

  // Occasionally, prune low opacity triangles
  if (getIteration() % 10 == 0) {
    triangles_->pruneLowWeightTriangles(viewpoint_cam, visible_triangle_mask, full_model_scaling);
  }

  // Training statistics
  {
    torch::NoGradGuard no_grad;
    float current_loss = loss.item().toFloat();
    kfs_loss_[viewpoint_cam->fid_] = current_loss;
    ema_loss_for_log_ = 0.4f * current_loss + 0.6 * ema_loss_for_log_;

    if (keyframe_record_interval_ &&
        getIteration() % keyframe_record_interval_ == 0)
      recordKeyframeRendered(rendered_image, gt_image, viewpoint_cam->fid_,
                             result_dir_, result_dir_, result_dir_);
  }

  auto iter_end_timing = std::chrono::steady_clock::now();
  auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                       iter_end_timing - iter_start_timing)
                       .count();

  // Reporting and periodic saves
  if (training_report_interval_ &&
      (getIteration() % training_report_interval_ == 0)) {
    std::cout << std::fixed << std::setprecision(8) << "Training iteration "
              << getIteration() << "/" << opt_params_.iterations_
              << ", time elapsed:" << iter_time / 1000.0 << "s"
              << ", ema_loss:" << ema_loss_for_log_ << std::endl;
  }

  if ((all_keyframes_record_interval_ &&
       getIteration() % all_keyframes_record_interval_ == 0)) {
    renderAndRecordAllKeyframes();
    saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
              "data");
  }

  if (loop_closure_iteration_) loop_closure_iteration_ = false;

  // If keyframe had to be loaded, save back to disk
  if (had_to_load) {
    viewpoint_cam->saveDataToDisk();
  }
}

bool TriangleMapper::isStopped() {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  return this->stopped_;
}

void TriangleMapper::signalStop(const bool going_to_stop) {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  this->stopped_ = going_to_stop;
}

bool TriangleMapper::hasMetInitialMappingConditions() {
  if (!pSLAM_->isShutDown() &&
      pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
      pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}

bool TriangleMapper::hasMetIncrementalMappingConditions() {
  if (!pSLAM_->isShutDown() && pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}
