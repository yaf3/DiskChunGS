#include "gaussian_mapper.h"
#include "rendering/gaussian_rasterizer.h"
#include "rendering/gaussian_renderer.h"
#include "utils/loss_utils.h"
#include "utils/profiling.h"
#include "utils/trajectory_viewer.h"

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

void trainingReport(int iteration,
                    int num_iterations,
                    torch::Tensor& Ll1,
                    torch::Tensor& loss,
                    float ema_loss_for_log,
                    int64_t elapsed_time,
                    GaussianModel& gaussians,
                    GaussianScene& scene,
                    GaussianPipelineParams& pipe,
                    torch::Tensor& background) {
  std::cout << std::fixed << std::setprecision(8) << "Training iteration "
            << iteration << "/" << num_iterations
            << ", time elapsed:" << elapsed_time / 1000.0 << "s"
            << ", ema_loss:" << ema_loss_for_log
            << ", num_points:" << gaussians.xyz_.size(0) << std::endl;
}

void GaussianMapper::run() {
  std::cout << "[MAPPER DEBUG] GaussianMapper::run() started" << std::endl;

  std::chrono::steady_clock::time_point training_start =
      std::chrono::steady_clock::now();
  training_start_time_ = training_start;

  // Set cameras extent early so it's available for all keyframe processing
  scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
  scene_->cameras_extent_ = 1.0;  // For debugging, maybe its better without;
  std::cout << "Extent: " << scene_->cameras_extent_ << std::endl;

  // Delete existing chunks since training
  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  std::filesystem::remove_all(keyframe_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

  std::cout << "[MAPPER DEBUG] Starting initial mapping phase" << std::endl;
  // First loop: Initial gaussian mapping
  while (!isStopped()) {
    // Check conditions for initial mapping
    if (hasMetInitialMappingConditions()) {
      std::cout << "[MAPPER DEBUG] Initial mapping completed, breaking to "
                   "next phase"
                << std::endl;
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

          // Create tuple matching handleNewKeyframeFromORBSLAM signature
          std::tuple<unsigned long, unsigned long, Sophus::SE3f, cv::Mat, bool,
                     cv::Mat, std::vector<float>, std::vector<float>,
                     std::string>
              kf_tuple = std::make_tuple(
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

          // Use the common keyframe handling function
          handleNewKeyframeFromORBSLAM(kf_tuple);

          // Setup training on first keyframe
          if (!initial_mapped_) {
            gaussians_->trainingSetup(opt_params_);
            std::cout << "Inital mapped!\n";
            initial_mapped_ = true;
          }
        }
      }

      // Invoke training once
      trainForOneIteration();

      // Finish initial mapping loop
      break;
    } else if (pSLAM_->isShutDown()) {
      std::cout << "[MAPPER DEBUG] SLAM shutdown during initial mapping"
                << std::endl;
      break;
    } else {
      // Initial conditions not satisfied
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "[MAPPER DEBUG] Exited initial mapping phase" << std::endl;

  std::cout << "[MAPPER DEBUG] Starting incremental mapping phase" << std::endl;
  // Second loop: Incremental gaussian mapping
  int SLAM_stop_iter = 0;
  while (!isStopped()) {
    auto timer_TotalLoop = ProfilingUtils::Timer("TotalLoop");

    // Check conditions for incremental mapping
    if (hasMetIncrementalMappingConditions()) {
      combineMappingOperations();
      if (cull_keyframes_) cullKeyframes();
    }

    // Invoke training once
    trainForOneIteration();
    timer_TotalLoop.stop();

    if (pSLAM_->isShutDown()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
      std::cout << "[MAPPER DEBUG] SLAM shutdown at iteration "
                << SLAM_stop_iter << std::endl;
    }

    if (SLAM_ended_) {
      std::cout << "[MAPPER DEBUG] Breaking from incremental mapping"
                << std::endl;
      break;
    }

    // if (getIteration() == 3000) {
    //   testTransferGaussiansAcrossChunks();
    // }

    // if (getIteration() % 1000 == 0) {
    //   gaussians_->testSaveLoadEvictCycle();
    // }
  }

  std::cout << "[MAPPER DEBUG] Exited incremental mapping phase" << std::endl;

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

  // while (getIteration() < 30000) {
  //   trainForOneIteration();
  // }

  std::cout << "[MAPPER DEBUG] ===== STARTING CLEANUP SECTION ====="
            << std::endl;
  std::cout << "[MAPPER DEBUG] Saving total gaussians" << std::endl;
  saveTotalGaussians("_shutdown");

  std::cout << "[MAPPER DEBUG] Rendering and recording all keyframes"
            << std::endl;
  renderAndRecordAllKeyframes("_shutdown");

  std::cout << "[MAPPER DEBUG] Saving scene" << std::endl;
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");

  std::cout << "[MAPPER DEBUG] Writing keyframe used times" << std::endl;
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  std::cout << "[MAPPER DEBUG] Writing training metrics CSV" << std::endl;
  writeTrainingMetricsCSV(result_dir_);

  std::cout << "[MAPPER DEBUG] Cleaning up temporary directories" << std::endl;
  std::filesystem::remove_all(chunk_save_dir_);
  std::filesystem::remove_all(keyframe_save_dir_);

  std::cout << "[MAPPER DEBUG] Signaling stop" << std::endl;
  signalStop();

  if (completion_callback_) {
    std::cout << "[MAPPER DEBUG] Calling completion callback" << std::endl;
    completion_callback_();
  }

  std::cout << "[MAPPER DEBUG] ===== GaussianMapper::run() COMPLETED ====="
            << std::endl;
}

// Modified version of trainForOneIteration that uses the chunk manager
void GaussianMapper::trainForOneIteration(
    std::shared_ptr<GaussianKeyframe> selected_keyframe) {
  // gaussians_->runFullConsistencyCheck("trainForOneIteration_START");
  // std::cout << "[GaussianMapper] Starting Optimization Iteration" <<
  // std::endl;
  auto timer_trainForOneIteration =
      ProfilingUtils::Timer("trainForOneIteration");

  increaseIteration(1);

  // Collect training metrics at regular intervals
  int current_iteration = getIteration();
  if (current_iteration % metrics_collection_interval_ == 0) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - training_start_time_);
    double elapsed_seconds = elapsed.count() / 1000.0;

    int totalGaussians = gaussians_->countAllGaussians();
    int activeGaussians = int(gaussians_->getXYZ().size(0));

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
    metrics.active_gaussian_count = activeGaussians;
    metrics.total_gaussian_count = totalGaussians;
    metrics.reserved_memory_mb = reserved_MB;
    metrics.allocated_memory_mb = alloc_MB;
    metrics.ram_usage_mb = getCurrentRAMUsageMB();
    if (keyframe_selection_strategy_ == 1)
      metrics.queue_keyframes = keyframe_queue_->getQueueSize();
    else {
      metrics.queue_keyframes = 0;
    }
    training_metrics_.push_back(metrics);
  }

  auto iter_start_timing = std::chrono::steady_clock::now();

  auto timer_pickKeyframe = ProfilingUtils::Timer("pickKeyframe");

  std::shared_ptr<GaussianKeyframe> viewpoint_cam;
  if (selected_keyframe) {
    viewpoint_cam = selected_keyframe;
  } else {
    switch (keyframe_selection_strategy_) {
      // Random sliding window keyframe
      case 0: {
        viewpoint_cam = useOneRandomSlidingWindowKeyframe();
      } break;
      // Grid based KF selection
      case 1: {
        viewpoint_cam = keyframe_queue_->getNextKeyframe();
      } break;
      default: {
        throw std::runtime_error(
            "[GaussianMapper] Invalid keyframe selection strategy");
      }
    }
  }

  timer_pickKeyframe.stop();
  if (!viewpoint_cam) {
    increaseIteration(-1);
    throw std::runtime_error(
        "[GaussianMapper] Keyframe not found for training");
    return;
  }

  // Record keyframe selection for trajectory viewer
  if (trajectory_viewer_) {
    trajectory_viewer_->recordKeyframeSelection(
        static_cast<int>(viewpoint_cam->fid_));
  }

  auto timer_loadKeyframe = ProfilingUtils::Timer("LoadKeyframe");
  bool had_to_load = false;
  if (!viewpoint_cam->loaded_) {
    std::cout << "Loading keyframe " << std::to_string(viewpoint_cam->fid_)
              << " to GPU for training" << std::endl;
    viewpoint_cam->loadDataFromDisk();
    had_to_load = true;
  }
  timer_loadKeyframe.stop();

  // std::cout << "Using keyframe id: " << viewpoint_cam->fid_ << std::endl;

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  auto [gt_image, gt_inv_depth, mask, image_height, image_width] =
      viewpoint_cam->getTrainingData(
          undistort_mask_[viewpoint_cam->camera_id_],
          scene_->cameras_.at(viewpoint_cam->camera_id_)
              .gaus_pyramid_undistort_mask_);

  auto timer_waitForMutex = ProfilingUtils::Timer("waitForMutex");
  // Mutex lock for usage of the gaussian model
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  timer_waitForMutex.stop();

  // auto timer_deleteSparseChunks =
  // ProfilingUtils::Timer("deleteSparseChunks");
  // int min_gaussians_per_chunk = 100;
  // if (getIteration() % 100 == 0) {
  //   gaussians_->deleteSparseChunks(min_gaussians_per_chunk);
  // }
  // timer_deleteSparseChunks.stop();

  if (getIteration() % 1000 == 0) {
    int loaded_cout = 0;
    for (const auto& [index, keyframe] : scene_->keyframes_) {
      if (keyframe->loaded_) loaded_cout++;
    }
    std::cout << "Loaded keyframes: " << loaded_cout << " / "
              << scene_->keyframes().size() << std::endl;
  }

  auto timer_loadVisibleChunks = ProfilingUtils::Timer("loadVisibleChunks");

  torch::Tensor visible_gaussian_mask =
      gaussians_->cullVisibleGaussians(viewpoint_cam);

  timer_loadVisibleChunks.stop();

  // std::cout << "Rendering " << visible_gaussian_mask.sum().item<int>()
  //           << " visible gaussians from " << visible_gaussian_mask.size(0)
  //           << " total gaussians." << std::endl;

  auto timer_misc_updates = ProfilingUtils::Timer("ITER/LR/SH Updates");

  timer_misc_updates.stop();

  torch::Tensor view_matrix = viewpoint_cam->getRT().transpose(0, 1);

  auto timer_render = ProfilingUtils::Timer("render");
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg =
      GaussianRenderer::render(gaussians_, visible_gaussian_mask, viewpoint_cam,
                               image_height, image_width, pipe_params_,
                               background_, override_color_, 1.0f, false,
                               viewpoint_cam->FoVx_, viewpoint_cam->FoVy_,
                               view_matrix, viewpoint_cam->projection_matrix_);

  timer_render.stop();
  torch::Tensor rendered_image = std::get<1>(render_pkg);
  torch::Tensor radii = std::get<2>(render_pkg);

  auto timer_loss_calculation = ProfilingUtils::Timer("loss_calculation");
  // Loss calculation (same as before)
  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto Ll1 = l1_loss(rendered_image, gt_image, 1.0f);
  auto Lssim = loss_utils::fast_ssim(rendered_image, gt_image);
  float lambda_dssim = lambdaDssim();
  auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);

  if (gt_inv_depth.defined()) {
    float lambda_depth = lambdaDepth();
    torch::Tensor rendered_inv_depth = std::get<0>(render_pkg);
    torch::Tensor depth_loss = (rendered_inv_depth - gt_inv_depth).abs().mean();
    loss += lambda_depth * depth_loss;
  }

  timer_loss_calculation.stop();

  auto timer_backwards = ProfilingUtils::Timer("backwards");
  loss.backward();

  timer_backwards.stop();

  auto timer_synchronize = ProfilingUtils::Timer("synchronize");
  // torch::cuda::synchronize();
  timer_synchronize.stop();

  auto timer_pose_exposure_step = ProfilingUtils::Timer("pose&exposure_step");
  viewpoint_cam->step();
  timer_pose_exposure_step.stop();

  auto timer_optimizer_step = ProfilingUtils::Timer("optimizer_step");
  // Optimizer step
  if (getIteration() < opt_params_.iterations_ ||
      opt_params_.iterations_ == -1) {
    // visibility_filter from renderer is only for visible gaussians
    torch::Tensor subset_contributed = (radii > 0);

    // Map back to full model indices
    torch::Tensor full_model_contributed = torch::zeros(
        {gaussians_->getXYZ().size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // Compute indices only when needed for mapping radii back to full model
    torch::Tensor visible_gaussian_indices =
        torch::nonzero(visible_gaussian_mask).squeeze(1);

    // Set true for gaussians that were both visible AND had radii > 0
    full_model_contributed.index_put_({visible_gaussian_indices},
                                      subset_contributed);

    gaussians_->optimizerStep(full_model_contributed,
                              gaussians_->getXYZ().size(0));
  }
  gaussians_->optimizer_->zero_grad(true);
  timer_optimizer_step.stop();

  auto timer_prune = ProfilingUtils::Timer("prune");
  if (getIteration() % 10 == 0) {
    gaussians_->pruneLowOpacityGaussians(viewpoint_cam, visible_gaussian_mask);
  }
  timer_prune.stop();

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

  auto timer_saveKeyframe = ProfilingUtils::Timer("SaveKeyframe");
  if (had_to_load) {
    viewpoint_cam->saveDataToDisk();
  }
  timer_saveKeyframe.stop();

  // gaussians_->runFullConsistencyCheck("trainForOneIteration_END");

  // Chunks automatically released by ChunkOptimizationGuard destructor
  timer_trainForOneIteration.stop();
  if (getIteration() % 500 == 0) {
    ProfilingUtils::getInstance().printStats();
    ProfilingUtils::getInstance().reset();
  }
}

bool GaussianMapper::isStopped() {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop) {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  this->stopped_ = going_to_stop;
  std::cout << "Signal stop received" << std::endl;
}

bool GaussianMapper::hasMetInitialMappingConditions() {
  if (!pSLAM_->isShutDown() &&
      pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
      pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions() {
  if (!pSLAM_->isShutDown() && pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}

void GaussianMapper::generateKfidRandomShuffle() {
  if (scene_->keyframes().empty()) return;

  std::size_t nkfs = scene_->keyframes().size();
  kfid_shuffle_.resize(nkfs);
  std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
  std::mt19937 g(rd_());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe() {
  // auto t1 = std::chrono::steady_clock::now();
  if (scene_->keyframes().empty()) return nullptr;

  if (!kfid_shuffled_) generateKfidRandomShuffle();

  std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
  int random_cam_idx;

  if (kfid_shuffled_) {
    int start_shuffle_idx = kfid_shuffle_idx_;
    do {
      // Next shuffled idx
      ++kfid_shuffle_idx_;
      if (kfid_shuffle_idx_ >= kfid_shuffle_.size()) kfid_shuffle_idx_ = 0;
      // Add 1 time of use to all kfs if they are all unavalible
      if (kfid_shuffle_idx_ == start_shuffle_idx) {
        for (auto& kfit : scene_->keyframes()) {
          increaseKeyframeTimesOfUse(kfit.second, 1);
        }
        if (opt_params_.auto_distribute_) {
          std::vector<std::pair<std::size_t, float>> vec(kfs_loss_.begin(),
                                                         kfs_loss_.end());
          // std::vector<std::pair<std::size_t, int>>
          // vec(kfs_used_times_.begin(), kfs_used_times_.end());
          int k = std::max(
              1, static_cast<int>(vec.size() / opt_params_.auto_distribute_));
          std::nth_element(vec.begin(), vec.begin() + k, vec.end(),
                           [](const std::pair<std::size_t, float>& a,
                              const std::pair<std::size_t, float>& b) {
                             return a.second > b.second;
                           });
          for (int i = 0; i < k; ++i) {
            increaseKeyframeTimesOfUse(scene_->keyframes()[vec[i].first], 1);
          }
        }
      }
      // Get viewpoint kf
      random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
      auto random_cam_it = scene_->keyframes().begin();
      for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
      viewpoint_cam = (*random_cam_it).second;
    } while (viewpoint_cam->remaining_times_of_use_ <= 0);
  }

  // Count used times
  auto viewpoint_fid = viewpoint_cam->fid_;
  if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
    kfs_used_times_[viewpoint_fid] = 1;
  else
    ++kfs_used_times_[viewpoint_fid];

  // Handle times of use
  --(viewpoint_cam->remaining_times_of_use_);

  // Efficient GPU memory management
  if (!viewpoint_cam->loaded_) {
    viewpoint_cam->loadDataFromDisk();
  }

  // Remove keyframe if already in queue to avoid duplicates
  auto queue_it = std::find(gpu_queue.begin(), gpu_queue.end(), viewpoint_cam);
  if (queue_it != gpu_queue.end()) {
    gpu_queue.erase(queue_it);
  }

  // Add to front (most recently used)
  gpu_queue.push_front(viewpoint_cam);

  // Clean up oldest keyframes
  while (gpu_queue.size() > max_gpu_keyframes_) {
    std::shared_ptr<GaussianKeyframe> oldest = gpu_queue.back();
    gpu_queue.pop_back();

    // Only transfer to CPU if it's loaded and not the selected one
    if (oldest->loaded_ && oldest != viewpoint_cam) {
      oldest->saveDataToDisk();
    }
  }

  // auto t2 = std::chrono::steady_clock::now();
  // auto t21 =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  // std::cout<<t21 <<" ns"<<std::endl;
  return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe> GaussianMapper::useOneRandomKeyframe() {
  if (scene_->keyframes().empty()) return nullptr;

  // Get randomly
  int nkfs = static_cast<int>(scene_->keyframes().size());
  int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
  auto random_cam_it = scene_->keyframes().begin();
  for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx) ++random_cam_it;
  std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

  // Count used times
  auto viewpoint_fid = viewpoint_cam->fid_;
  if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
    kfs_used_times_[viewpoint_fid] = 1;
  else
    ++kfs_used_times_[viewpoint_fid];

  return viewpoint_cam;
}

std::vector<std::shared_ptr<GaussianKeyframe>>
GaussianMapper::getClosestKeyframes(
    std::shared_ptr<GaussianKeyframe> current_kf,
    int n,
    int k) {
  std::vector<std::shared_ptr<GaussianKeyframe>> closest_keyframes;
  if (n <= 0 || k <= 0) return closest_keyframes;

  auto all_keyframes = scene_->getAllKeyframes();
  if (all_keyframes.empty()) return closest_keyframes;

  // Get current keyframe's camera center position
  Eigen::Vector3f current_center = current_kf->getTranslationf();

  // Create a vector of keyframes sorted by spatial distance to current
  // keyframe
  std::vector<std::pair<float, std::shared_ptr<GaussianKeyframe>>> candidates;
  for (const auto& kf_pair : all_keyframes) {
    if (kf_pair.second != current_kf) {  // Exclude current keyframe
      // Get candidate keyframe's camera center position
      Eigen::Vector3f candidate_center = kf_pair.second->getTranslationf();
      // Calculate Euclidean distance between camera centers
      float spatial_distance = (current_center - candidate_center).norm();
      candidates.push_back({spatial_distance, kf_pair.second});
    }
  }

  // Sort by spatial distance (closest first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // First, try to take every k-th keyframe from the sorted list
  int selected_count = 0;
  for (int i = 0; i < static_cast<int>(candidates.size()) && selected_count < n;
       i += k) {
    closest_keyframes.push_back(candidates[i].second);
    selected_count++;

    // Optional debug output
    // std::cout << "Chosen Keyframe (k-spaced): "
    //           << std::to_string(candidates[i].second->fid_)
    //           << " Dist: " << candidates[i].first << std::endl;
  }

  // If we still need more keyframes and haven't used all candidates,
  // fill the remaining slots with the closest unused keyframes
  if (selected_count < n) {
    // Create a set of already selected keyframes for quick lookup
    std::set<std::shared_ptr<GaussianKeyframe>> selected_set;
    for (const auto& kf : closest_keyframes) {
      selected_set.insert(kf);
    }

    // Add remaining closest keyframes that weren't selected
    for (int i = 0;
         i < static_cast<int>(candidates.size()) && selected_count < n; ++i) {
      if (selected_set.find(candidates[i].second) == selected_set.end()) {
        closest_keyframes.push_back(candidates[i].second);
        selected_count++;

        // Optional debug output
        // std::cout << "Chosen Keyframe (fill): "
        //           << std::to_string(candidates[i].second->fid_)
        //           << " Dist: " << candidates[i].first << std::endl;
      }
    }
  }

  return closest_keyframes;
}

std::shared_ptr<GaussianKeyframe> GaussianMapper::useRecentKeyframe() {
  if (scene_->keyframes().empty()) return nullptr;

  // Find keyframe with the highest ID (most recent)
  unsigned long max_id = 0;
  std::shared_ptr<GaussianKeyframe> most_recent_kf = nullptr;

  for (const auto& kf_pair : scene_->keyframes()) {
    if (kf_pair.first > max_id) {
      max_id = kf_pair.first;
      most_recent_kf = kf_pair.second;
    }
  }

  // Check if keyframe has remaining uses
  if (most_recent_kf && most_recent_kf->remaining_times_of_use_ <= 0) {
    // Increase it to allow usage
    increaseKeyframeTimesOfUse(most_recent_kf, 1);
  }

  // Track usage for statistics
  if (most_recent_kf) {
    auto viewpoint_fid = most_recent_kf->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
      kfs_used_times_[viewpoint_fid] = 1;
    else
      ++kfs_used_times_[viewpoint_fid];

    // Decrease remaining times of use
    --(most_recent_kf->remaining_times_of_use_);
  }

  return most_recent_kf;
}