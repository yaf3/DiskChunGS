/**
 * This file is part of CaRtGS, modified from Photo-SLAM under GPL v3 license.
 *
 * Copyright (C) 2024 Dapeng Feng, Sun Yat-sen University.
 *
 * CaRtGS is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * CaRtGS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CaRtGS. If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"

#include "include/chunk_manager.h"
#include "include/gaussian_renderer.h"
#include "include/keyframe_selector.h"
#include "include/loss_utils.h"
#include "include/profiling.h"

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

GaussianMapper::GaussianMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                               std::filesystem::path gaussian_config_file_path,
                               std::filesystem::path result_dir,
                               int seed,
                               torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0) {
  // Random seed
  std::srand(seed);
  torch::manual_seed(seed);

  // Device
  if (device_type == torch::kCUDA && torch::cuda::is_available()) {
    std::cout << "[Gaussian Mapper]CUDA available! Training on GPU."
              << std::endl;
    device_type_ = torch::kCUDA;
    model_params_.data_device_ = "cuda";
  } else {
    std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
    device_type_ = torch::kCPU;
    model_params_.data_device_ = "cpu";
  }

  result_dir_ = result_dir;
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  chunk_save_dir_ = result_dir / "chunks";
  if (!chunk_save_dir_.empty() && std::filesystem::exists(chunk_save_dir_)) {
    for (const auto& entry :
         std::filesystem::directory_iterator(chunk_save_dir_)) {
      std::filesystem::remove_all(entry.path());
    }
  }
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  config_file_path_ = gaussian_config_file_path;
  readConfigFromFile(gaussian_config_file_path);

  std::vector<float> bg_color;
  if (model_params_.white_background_)
    bg_color = {1.0f, 1.0f, 1.0f};
  else
    bg_color = {0.0f, 0.0f, 0.0f};
  background_ = torch::tensor(
      bg_color,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  override_color_ =
      torch::empty(0, torch::TensorOptions().device(device_type_));

  // Initialize scene
  scene_ = std::make_shared<GaussianScene>(model_params_);

  // Mode
  if (!pSLAM) {
    // NO SLAM
    return;
  }

  // Sensors
  switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR: {
      this->sensor_type_ = MONOCULAR;
    } break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO: {
      this->sensor_type_ = STEREO;
      this->stereo_baseline_length_ = pSLAM->getSettings()->b();
      this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
          this->stereo_min_disparity_, this->stereo_num_disparity_);
      this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
      stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    } break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD: {
      this->sensor_type_ = RGBD;
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }

  // Cameras
  // TODO: not only monocular
  auto settings = pSLAM->getSettings();
  cv::Size SLAM_im_size = settings->newImSize();
  UndistortParams undistort_params(SLAM_im_size,
                                   settings->camera1DistortionCoef());

  auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
  std::cout << "Num. of Camera is " << vpCameras.size() << std::endl;
  for (auto& SLAM_camera : vpCameras) {
    Camera camera;
    camera.camera_id_ = SLAM_camera->GetId();
    if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
      camera.setModelId(Camera::CameraModelType::PINHOLE);
      float SLAM_fx = SLAM_camera->getParameter(0);
      float SLAM_fy = SLAM_camera->getParameter(1);
      float SLAM_cx = SLAM_camera->getParameter(2);
      float SLAM_cy = SLAM_camera->getParameter(3);

      // Old K, i.e. K in SLAM
      cv::Mat K = (cv::Mat_<float>(3, 3) << SLAM_fx, 0.f, SLAM_cx, 0.f, SLAM_fy,
                   SLAM_cy, 0.f, 0.f, 1.f);

      // camera.width_ = this->sensor_type_ == STEREO ?
      // undistort_params.old_size_.width
      //                                              :
      //                                              graphics_utils::roundToIntegerMultipleOf16(
      //                                                    undistort_params.old_size_.width);
      camera.width_ = undistort_params.old_size_.width;
      float x_ratio =
          static_cast<float>(camera.width_) / undistort_params.old_size_.width;

      // camera.height_ = this->sensor_type_ == STEREO ?
      // undistort_params.old_size_.height
      //                                               :
      //                                               graphics_utils::roundToIntegerMultipleOf16(
      //                                                     undistort_params.old_size_.height);
      camera.height_ = undistort_params.old_size_.height;
      float y_ratio = static_cast<float>(camera.height_) /
                      undistort_params.old_size_.height;

      camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
      camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
      camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        camera.gaus_pyramid_width_[l] =
            camera.width_ * this->kf_gaus_pyramid_factors_[l];
        camera.gaus_pyramid_height_[l] =
            camera.height_ * this->kf_gaus_pyramid_factors_[l];
      }

      camera.params_[0] /*new fx*/ = SLAM_fx * x_ratio;
      camera.params_[1] /*new fy*/ = SLAM_fy * y_ratio;
      camera.params_[2] /*new cx*/ = SLAM_cx * x_ratio;
      camera.params_[3] /*new cy*/ = SLAM_cy * y_ratio;

      cv::Mat K_new =
          (cv::Mat_<float>(3, 3) << camera.params_[0], 0.f, camera.params_[2],
           0.f, camera.params_[1], camera.params_[3], 0.f, 0.f, 1.f);

      // Undistortion
      if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
        undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

      camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

      undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(camera.undistort_mask,
                                                  device_type_);

      cv::Mat viewer_sub_undistort_mask;
      int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
      int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
      cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                 cv::Size(viewer_image_width_, viewer_image_height_));
      viewer_sub_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_sub_undistort_mask,
                                                  device_type_);

      cv::Mat viewer_main_undistort_mask;
      int viewer_image_height_main_ =
          camera.height_ * rendered_image_viewer_scale_main_;
      int viewer_image_width_main_ =
          camera.width_ * rendered_image_viewer_scale_main_;
      cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                 cv::Size(viewer_image_width_main_, viewer_image_height_main_));
      viewer_main_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask,
                                                  device_type_);

      if (this->sensor_type_ == STEREO) {
        camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
        if (this->stereo_Q_.cols != 4) {
          this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
          this->stereo_Q_.setTo(0.0f);
          this->stereo_Q_.at<float>(0, 0) = 1.0f;
          this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
          this->stereo_Q_.at<float>(1, 1) = 1.0f;
          this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
          this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
          this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
        }
      }
    } else if (SLAM_camera->GetType() ==
               ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
      camera.setModelId(Camera::CameraModelType::FISHEYE);
    } else {
      camera.setModelId(Camera::CameraModelType::INVALID);
    }

    if (!viewer_camera_id_set_) {
      viewer_camera_id_ = camera.camera_id_;
      viewer_camera_id_set_ = true;
    }
    this->scene_->addCamera(camera);
  }

  initializeChunkManagement();
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path) {
  cv::FileStorage settings_file(cfg_path.string().c_str(),
                                cv::FileStorage::READ);
  if (!settings_file.isOpened()) {
    std::cerr << "[Gaussian Mapper]Failed to open settings file at: "
              << cfg_path << std::endl;
    exit(-1);
  }

  std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path
            << std::endl;
  std::unique_lock<std::mutex> lock(mutex_settings_);

  // Model parameters
  model_params_.sh_degree_ = settings_file["Model.sh_degree"].operator int();
  model_params_.resolution_ =
      settings_file["Model.resolution"].operator float();
  model_params_.white_background_ =
      (settings_file["Model.white_background"].operator int()) != 0;
  model_params_.eval_ = (settings_file["Model.eval"].operator int()) != 0;

  // Pipeline Parameters
  z_near_ = settings_file["Camera.z_near"].operator float();
  z_far_ = settings_file["Camera.z_far"].operator float();

  monocular_inactive_geo_densify_max_pixel_dist_ =
      settings_file["Monocular.inactive_geo_densify_max_pixel_dist"]
          .operator float();
  stereo_min_disparity_ = settings_file["Stereo.min_disparity"].operator int();
  stereo_num_disparity_ = settings_file["Stereo.num_disparity"].operator int();
  RGBD_min_depth_ = settings_file["RGBD.min_depth"].operator float();
  RGBD_max_depth_ = settings_file["RGBD.max_depth"].operator float();

  inactive_geo_densify_ =
      (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
  max_depth_cached_ = settings_file["Mapper.depth_cache"].operator int();
  min_num_initial_map_kfs_ = static_cast<unsigned long>(
      settings_file["Mapper.min_num_initial_map_kfs"].operator int());
  new_keyframe_times_of_use_ =
      settings_file["Mapper.new_keyframe_times_of_use"].operator int();
  local_BA_increased_times_of_use_ =
      settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
  loop_closure_increased_times_of_use_ =
      settings_file["Mapper.loop_closure_increased_times_of_use_"]
          .operator int();
  cull_keyframes_ =
      (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
  large_rot_th_ =
      settings_file["Mapper.large_rotation_threshold"].operator float();
  large_trans_th_ =
      settings_file["Mapper.large_translation_threshold"].operator float();
  stable_num_iter_existence_ =
      settings_file["Mapper.stable_num_iter_existence"].operator int();

  pipe_params_.convert_SHs_ =
      (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
  pipe_params_.compute_cov3D_ =
      (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

  do_gaus_pyramid_training_ =
      (settings_file["GausPyramid.do"].operator int()) != 0;
  num_gaus_pyramid_sub_levels_ =
      settings_file["GausPyramid.num_sub_levels"].operator int();
  int sub_level_times_of_use =
      settings_file["GausPyramid.sub_level_times_of_use"].operator int();
  kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
  kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    kf_gaus_pyramid_factors_[l] =
        std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
  }

  keyframe_record_interval_ =
      settings_file["Record.keyframe_record_interval"].operator int();
  all_keyframes_record_interval_ =
      settings_file["Record.all_keyframes_record_interval"].operator int();
  record_rendered_image_ =
      (settings_file["Record.record_rendered_image"].operator int()) != 0;
  record_ground_truth_image_ =
      (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
  record_loss_image_ =
      (settings_file["Record.record_loss_image"].operator int()) != 0;
  training_report_interval_ =
      settings_file["Record.training_report_interval"].operator int();
  record_loop_ply_ =
      (settings_file["Record.record_loop_ply"].operator int()) != 0;

  // Optimization Parameters
  opt_params_.iterations_ =
      settings_file["Optimization.max_num_iterations"].operator int();
  opt_params_.position_lr_init_ =
      settings_file["Optimization.position_lr_init"].operator float();
  opt_params_.position_lr_final_ =
      settings_file["Optimization.position_lr_final"].operator float();
  opt_params_.position_lr_delay_mult_ =
      settings_file["Optimization.position_lr_delay_mult"].operator float();
  opt_params_.position_lr_max_steps_ =
      settings_file["Optimization.position_lr_max_steps"].operator int();
  opt_params_.feature_lr_ =
      settings_file["Optimization.feature_lr"].operator float();
  opt_params_.opacity_lr_ =
      settings_file["Optimization.opacity_lr"].operator float();
  opt_params_.scaling_lr_ =
      settings_file["Optimization.scaling_lr"].operator float();
  opt_params_.rotation_lr_ =
      settings_file["Optimization.rotation_lr"].operator float();
  opt_params_.smooth_l1_ =
      (settings_file["Optimization.smooth_l1"].operator int()) != 0;
  opt_params_.opacity_reg_ =
      settings_file["Optimization.opacity_reg"].operator float();

  opt_params_.percent_dense_ =
      settings_file["Optimization.percent_dense"].operator float();
  opt_params_.lambda_dssim_ =
      settings_file["Optimization.lambda_dssim"].operator float();
  opt_params_.densification_interval_ =
      settings_file["Optimization.densification_interval"].operator int();
  opt_params_.opacity_reset_interval_ =
      settings_file["Optimization.opacity_reset_interval"].operator int();
  opt_params_.densify_from_iter_ =
      settings_file["Optimization.densify_from_iter"].operator int();
  opt_params_.densify_until_iter_ =
      settings_file["Optimization.densify_until_iter"].operator int();
  opt_params_.densify_grad_threshold_ =
      settings_file["Optimization.densify_grad_threshold"].operator float();
  opt_params_.auto_distribute_ =
      settings_file["Optimization.auto_distribute"].operator int();

  prune_big_point_after_iter_ =
      settings_file["Optimization.prune_big_point_after_iter"].operator int();
  densify_min_opacity_ =
      settings_file["Optimization.densify_min_opacity"].operator float();

  // Viewer Parameters
  rendered_image_viewer_scale_ =
      settings_file["GaussianViewer.image_scale"].operator float();
  rendered_image_viewer_scale_main_ =
      settings_file["GaussianViewer.image_scale_main"].operator float();

  chunk_size_ = settings_file["Chunking.chunk_size"].operator float();
  overlap_margin_ = settings_file["Chunking.overlap_margin"].operator float();
  max_chunks_in_memory_ = settings_file["Chunking.max_chunks"].operator int();
}

void GaussianMapper::run() {
  // First loop: Initial gaussian mapping
  while (!isStopped()) {
    // Check conditions for initial mapping
    if (hasMetInitialMappingConditions()) {
      pSLAM_->getAtlas()->clearMappingOperation();

      // Get initial sparse map
      auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
      std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
      std::vector<ORB_SLAM3::MapPoint*> vpMPs;
      torch::Tensor initialSparsePoints;
      torch::Tensor initialSparseColors;
      {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        vpKFs = pMap->GetAllKeyFrames();
        vpMPs = pMap->GetAllMapPoints();

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        initialSparsePoints =
            torch::zeros({static_cast<int64_t>(vpMPs.size()), 3}, options);
        initialSparseColors =
            torch::zeros({static_cast<int64_t>(vpMPs.size()), 3}, options);

        // Get accessor for direct memory access
        auto accessor_points = initialSparsePoints.accessor<float, 2>();
        auto accessor_colors = initialSparseColors.accessor<float, 2>();

        for (size_t i = 0; i < vpMPs.size(); ++i) {
          const auto& pos = vpMPs[i]->GetWorldPos();
          const auto& color = vpMPs[i]->GetColorRGB();
          accessor_points[i][0] = pos.x();
          accessor_points[i][1] = pos.y();
          accessor_points[i][2] = pos.z();

          accessor_colors[i][0] = color(0);
          accessor_colors[i][1] = color(1);
          accessor_colors[i][2] = color(2);
        }
        for (const auto& pKF : vpKFs) {
          std::shared_ptr<GaussianKeyframe> new_kf =
              std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
          new_kf->zfar_ = z_far_;
          new_kf->znear_ = z_near_;
          // Pose
          auto pose = pKF->GetPose();
          new_kf->setPose(pose.unit_quaternion().cast<double>(),
                          pose.translation().cast<double>());
          cv::Mat imgRGB_undistorted, imgAux_undistorted;
          try {
            // Camera
            Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
            new_kf->setCameraParams(camera);

            imgRGB_undistorted = pKF->imgLeftRGB;
            imgAux_undistorted = pKF->imgAuxiliary;

            new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
                imgRGB_undistorted, device_type_);
            new_kf->img_filename_ = pKF->mNameFile;
            new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
          } catch (std::out_of_range) {
            throw std::runtime_error(
                "[GaussianMapper::run]KeyFrame Camera not found!");
          }
          new_kf->computeTransformTensors();
          scene_->addKeyframe(new_kf, &kfid_shuffled_);

          increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

          // Features
          std::vector<float> pixels;
          std::vector<float> pointsLocal;
          pKF->GetKeypointInfo(pixels, pointsLocal);
          new_kf->kps_pixel_ = std::move(pixels);
          new_kf->kps_point_local_ = std::move(pointsLocal);
          new_kf->img_undist_ = imgRGB_undistorted;
          new_kf->img_auxiliary_undist_ = imgAux_undistorted;
        }
      }

      // Prepare multi resolution images for training
      for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        if (device_type_ == torch::kCUDA) {
          cv::cuda::GpuMat img_gpu;
          img_gpu.upload(pkf->img_undist_);
          pkf->gaus_pyramid_original_image_.resize(
              num_gaus_pyramid_sub_levels_);
          for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                             cv::Size(pkf->gaus_pyramid_width_[l],
                                      pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
          }
        } else {
          pkf->gaus_pyramid_original_image_.resize(
              num_gaus_pyramid_sub_levels_);
          for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                       cv::Size(pkf->gaus_pyramid_width_[l],
                                pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized,
                                                        device_type_);
          }
        }
      }

      // Prepare for training
      {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        std::cout << "Adding initial points\n";
        addPoints(initialSparsePoints, initialSparseColors,
                  scene_->keyframes());
      }

      // Invoke training once
      trainForOneIteration();

      // Finish initial mapping loop
      initial_mapped_ = true;
      break;
    } else if (pSLAM_->isShutDown()) {
      break;
    } else {
      // Initial conditions not satisfied
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Second loop: Incremental gaussian mapping
  int SLAM_stop_iter = 0;
  while (!isStopped()) {
    // Check conditions for incremental mapping
    if (hasMetIncrementalMappingConditions()) {
      combineMappingOperations();
      if (cull_keyframes_) cullKeyframes();
    }

    // Invoke training once
    trainForOneIteration();

    if (pSLAM_->isShutDown()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
    }

    if (SLAM_ended_) break;
  }

  // Third loop: After SLAM stopped, keep training
  while (!isStopped()) {
    // Invoke training once
    trainForOneIteration();

    if (getIteration() >= opt_params_.iterations_) break;
  }

  // Fourth loop: Tail gaussian optimization
  int densify_interval = densifyInterval();
  int n_delay_iters = densify_interval * 0.8;
  while (getIteration() - SLAM_stop_iter < n_delay_iters ||
         getIteration() % densify_interval < n_delay_iters ||
         isKeepingTraining()) {
    trainForOneIteration();
  }

  auto video_dir = result_dir_ / "flythrough";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
  renderFlyThroughVideo(video_dir / "output_video", 1920, 1080, 30, 30.0f);

  // For debug: basically viewer now
  while (getIteration() < 100000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  // Save and clear
  renderAndRecordAllKeyframes("_shutdown");
  savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  signalStop();
}

void GaussianMapper::trainColmap() {
  // Prepare multi resolution images for training
  for (auto& kfit : scene_->keyframes()) {
    auto pkf = kfit.second;
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
    if (device_type_ == torch::kCUDA) {
      cv::cuda::GpuMat img_gpu;
      img_gpu.upload(pkf->img_undist_);
      pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::cuda::GpuMat img_resized;
        cv::cuda::resize(img_gpu, img_resized,
                         cv::Size(pkf->gaus_pyramid_width_[l],
                                  pkf->gaus_pyramid_height_[l]));
        pkf->gaus_pyramid_original_image_[l] =
            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
      }
    } else {
      pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::Mat img_resized;
        cv::resize(pkf->img_undist_, img_resized,
                   cv::Size(pkf->gaus_pyramid_width_[l],
                            pkf->gaus_pyramid_height_[l]));
        pkf->gaus_pyramid_original_image_[l] =
            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
      }
    }
  }

  // Todo: Fix for chunking
  // // Prepare for training
  // {
  //   std::unique_lock<std::mutex> lock_render(mutex_render_);
  //   scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
  //   gaussians_->createFromPcd(scene_->cached_point_cloud_,
  //                             scene_->cameras_extent_);
  //   std::unique_lock<std::mutex> lock(mutex_settings_);
  //   gaussians_->trainingSetup(opt_params_);
  //   this->initial_mapped_ = true;
  // }

  // Main loop: gaussian splatting training
  while (!isStopped()) {
    // Invoke training once
    trainForOneIteration();

    if (getIteration() >= opt_params_.iterations_) break;
  }

  // Tail gaussian optimization
  int densify_interval = densifyInterval();
  int n_delay_iters = densify_interval * 0.8;
  while (getIteration() % densify_interval <= n_delay_iters ||
         isKeepingTraining()) {
    trainForOneIteration();
    densify_interval = densifyInterval();
    n_delay_iters = densify_interval * 0.8;
  }

  // Save and clear
  renderAndRecordAllKeyframes("_shutdown");
  savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  signalStop();
}

// Modified version of trainForOneIteration that uses the chunk manager
void GaussianMapper::trainForOneIteration() {
  // std::cout << "[GaussianMapper] Starting Optimization Iteration" <<
  // std::endl;
  auto timer_trainForOneIteration =
      ProfilingUtils::Timer("trainForOneIteration");
  increaseIteration(1);
  chunk_manager_->setCurrentIteration(getIteration());
  int min_points_chunk_threshold = 10;
  chunk_manager_->cullSparseChunks(min_points_chunk_threshold);
  auto iter_start_timing = std::chrono::steady_clock::now();

  auto timer_selectLocalityAwareKeyframe =
      ProfilingUtils::Timer("selectLocalityAwareKeyframe");
  // Pick a keyframe using our locality-aware strategy
  // std::shared_ptr<GaussianKeyframe> viewpoint_cam =
  //     selectLocalityAwareKeyframe();
  std::shared_ptr<GaussianKeyframe> viewpoint_cam =
      useOneRandomSlidingWindowKeyframe();
  timer_selectLocalityAwareKeyframe.stop();
  if (!viewpoint_cam) {
    increaseIteration(-1);
    return;
  }

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

  int training_level = num_gaus_pyramid_sub_levels_;
  int image_height, image_width;
  torch::Tensor gt_image, mask;
  if (isdoingGausPyramidTraining())
    training_level = viewpoint_cam->getCurrentGausPyramidLevel();
  if (training_level == num_gaus_pyramid_sub_levels_) {
    image_height = viewpoint_cam->image_height_;
    image_width = viewpoint_cam->image_width_;
    gt_image = viewpoint_cam->original_image_.cuda();
    mask = undistort_mask_[viewpoint_cam->camera_id_];
  } else {
    image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
    image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
    gt_image =
        viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
    mask = scene_->cameras_.at(viewpoint_cam->camera_id_)
               .gaus_pyramid_undistort_mask_[training_level];
  }

  auto timer_waitForMutex = ProfilingUtils::Timer("waitForMutex");
  // Mutex lock for usage of the gaussian model
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  timer_waitForMutex.stop();

  auto timer_getVisibleChunks = ProfilingUtils::Timer("getVisibleChunks");
  // Get visible chunks using ChunkManager instead of updateActiveChunks
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->getVisibleChunks(viewpoint_cam);
  // auto active_chunks = chunk_manager_->getActiveChunks();
  // std::vector<std::shared_ptr<GaussianModel>> models;
  // models.reserve(active_chunks.size());
  // for (const auto& [coord, chunk] : active_chunks) {
  //   if (chunk && chunk->gaussians_) {
  //     models.push_back(chunk->gaussians_);
  //   } else {
  //     throw std::runtime_error("[renderFromPose] Chunk/Gaussian not valid");
  //   }
  // }
  timer_getVisibleChunks.stop();

  std::cout << "[Optimization] Num visible chunks: " << visible_chunks.size()
            << std::endl;

  // Extract models from chunks
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    } else {
      throw std::runtime_error("Chunk/Gaussians are null");
    }
  }

  if (models.empty()) {
    std::cout << "[Optimization] No valid models to render" << std::endl;
    // throw std::runtime_error("[Optimization] No valid models to render");
    return;  // Early return if no valid models
  }

  // Update learning rates and SH degrees
  for (const auto& gaussians : models) {
    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
      default_sh_ += 1;
    gaussians->setShDegree(default_sh_);

    // Update learning rate
    gaussians->updateLearningRate(getIteration());
    gaussians->setFeatureLearningRate(featureLearningRate());
    gaussians->setOpacityLearningRate(opacityLearningRate());
    gaussians->setScalingLearningRate(scalingLearningRate());
    gaussians->setRotationLearningRate(rotationLearningRate());
    // std::cout << "[0]: "
    //           << gaussians->optimizer_->param_groups()[0].options().get_lr()
    //           << std::endl;
    // std::cout << "[1]: "
    //           << gaussians->optimizer_->param_groups()[1].options().get_lr()
    //           << std::endl;
    // std::cout << "[2]: "
    //           << gaussians->optimizer_->param_groups()[2].options().get_lr()
    //           << std::endl;
    // std::cout << "[3]: "
    //           << gaussians->optimizer_->param_groups()[3].options().get_lr()
    //           << std::endl;
    // std::cout << "[4]: "
    //           << gaussians->optimizer_->param_groups()[4].options().get_lr()
    //           << std::endl;
    // std::cout << "[5]: "
    //           << gaussians->optimizer_->param_groups()[5].options().get_lr()
    //           << std::endl;
  }

  // Render
  auto timer_render = ProfilingUtils::Timer("render");
  auto render_pkg =
      GaussianRenderer::render(models, viewpoint_cam, image_height, image_width,
                               pipe_params_, background_, override_color_);
  timer_render.stop();
  auto rendered_image = std::get<0>(render_pkg);

  std::vector<torch::Tensor> screenspace_points_vec = std::get<1>(render_pkg);
  std::vector<torch::Tensor> radii_vec = std::get<2>(render_pkg);

  // Loss calculation (same as before)
  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto Ll1 = l1_loss(rendered_image, gt_image, 1.0f);
  auto Lssim = loss_utils::fast_ssim(rendered_image, gt_image);
  float lambda_dssim = lambdaDssim();
  auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);

  if (opt_params_.opacity_reg_) {
    for (const auto& gaussians : models) {
      loss += opt_params_.opacity_reg_ *
              gaussians->getOpacityActivation().abs().mean();
    }
  }

  auto timer_backwards = ProfilingUtils::Timer("backwards");
  loss.backward();
  timer_backwards.stop();

  torch::cuda::synchronize();
  auto timer_densification = ProfilingUtils::Timer("densification");
  {
    torch::NoGradGuard no_grad;
    kfs_loss_[viewpoint_cam->fid_] = loss.item().toFloat();
    ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

    if (keyframe_record_interval_ &&
        getIteration() % keyframe_record_interval_ == 0)
      recordKeyframeRendered(rendered_image, gt_image, viewpoint_cam->fid_,
                             result_dir_, result_dir_, result_dir_);

    int num_models = models.size();
    for (int model_idx = 0; model_idx < num_models; model_idx++) {
      const auto& gaussians = models[model_idx];
      // Get radii for this model
      const auto& radii = radii_vec[model_idx];

      // Calculate visibility filter for this specific model
      auto visibility_filter = (radii > 0).nonzero().reshape({-1});

      // Densification
      if (getIteration() < opt_params_.densify_until_iter_ ||
          opt_params_.densify_until_iter_ == -1) {
        // Keep track of max radii in image-space for pruning
        gaussians->max_radii2D_.index_put_(
            {visibility_filter},
            torch::max(gaussians->max_radii2D_.index({visibility_filter}),
                       radii.index({visibility_filter})));

        gaussians->addDensificationStats(screenspace_points_vec[model_idx],
                                         visibility_filter);

        if ((getIteration() > opt_params_.densify_from_iter_) &&
            (getIteration() % densifyInterval() == 0)) {
          int size_threshold = (getIteration() < prune_big_point_after_iter_ ||
                                prune_big_point_after_iter_ == -1)
                                   ? 0
                                   : 20;
          gaussians->densifyAndPrune(densifyGradThreshold(),
                                     densify_min_opacity_,
                                     scene_->cameras_extent_, size_threshold);
        }

        if (opacityResetInterval() &&
            (getIteration() % opacityResetInterval() == 0 ||
             (model_params_.white_background_ &&
              getIteration() == opt_params_.densify_from_iter_)))
          gaussians->resetOpacity();
      }
    }

    timer_densification.stop();

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
                << ", ema_loss:" << ema_loss_for_log_ << ", active_chunks:"
                << chunk_manager_->getStats().active_chunks << std::endl;
    }

    if ((all_keyframes_record_interval_ &&
         getIteration() % all_keyframes_record_interval_ == 0)) {
      renderAndRecordAllKeyframes();
      savePly(result_dir_ / std::to_string(getIteration()) / "ply");
    }

    if (loop_closure_iteration_) loop_closure_iteration_ = false;

    // Prefetch chunks for upcoming keyframes
    // auto timer_predictUpcomingKeyframes =
    //     ProfilingUtils::Timer("predictUpcomingKeyframes");
    // auto upcoming_keyframes = predictUpcomingKeyframes();
    // timer_predictUpcomingKeyframes.stop();

    // auto timer_preloadChunksForKeyframes =
    //     ProfilingUtils::Timer("preloadChunksForKeyframes");
    // chunk_manager_->preloadChunksForKeyframes(upcoming_keyframes);
    // timer_preloadChunksForKeyframes.stop();

    // if (getIteration() % 1000 == 0) {
    //   auto active_chunks = chunk_manager_->getActiveChunks();
    //   for (const auto& [coord, chunk] : active_chunks) {
    //     if (chunk && chunk->gaussians_) {
    //       chunk_manager_->saveChunk(coord, true);
    //       sleep(1);
    //       chunk_manager_->loadChunk(coord, true);
    //     } else {
    //       throw std::runtime_error("Chunk/Gaussian not valid");
    //     }
    //   }
    // }

    // Optimizer step
    for (const auto& gaussians : models) {
      if (getIteration() < opt_params_.iterations_ ||
          opt_params_.iterations_ == -1) {
        gaussians->optimizer_->step();
        gaussians->optimizer_->zero_grad(true);
      }
    }
  }

  // Periodically evict unused chunks (every 50 iterations)
  if (getIteration() % 50 == 0) {
    chunk_manager_->evictUnusedChunks();
  }

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
  if (chunk_manager_) {
    chunk_manager_->shutdown();
  }
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

void GaussianMapper::combineMappingOperations() {
  // Get Mapping Operations
  while (pSLAM_->getAtlas()->hasMappingOperation()) {
    ORB_SLAM3::MappingOperation opr =
        pSLAM_->getAtlas()->getAndPopMappingOperation();

    switch (opr.meOperationType) {
      case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA: {
        // std::cout << "[Gaussian Mapper]Local BA Detected."
        //           << std::endl;

        // Get new keyframes
        auto& associated_kfs = opr.associatedKeyFrames();
        std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
            associated_keyframe_map;

        // Add keyframes to the scene
        for (auto& kf : associated_kfs) {
          // Keyframe Id
          auto kfid = std::get<0>(kf);
          std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
          // If the keyframe is already in the scene, only update the pose.
          // Otherwise create a new one
          if (pkf) {
            auto& pose = std::get<2>(kf);
            pkf->setPose(pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
            pkf->computeTransformTensors();

            // Give local BA keyframes times of use
            increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
          } else {
            handleNewKeyframe(kf);
          }
          associated_keyframe_map[kfid] = scene_->getKeyframe(kfid);
        }

        // Get new points
        auto& associated_points = opr.associatedMapPoints();
        auto& points = std::get<0>(associated_points);
        auto& colors = std::get<1>(associated_points);

        int num_new_points = static_cast<int>(points.size() / 3);
        torch::Tensor points_tensor =
            torch::from_blob(points.data(), {num_new_points, 3},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .to(device_type_);
        torch::Tensor colors_tensor =
            torch::from_blob(colors.data(), {num_new_points, 3},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .to(device_type_);

        // Add new points to the model
        if (initial_mapped_ && points.size() >= 30) {
          torch::NoGradGuard no_grad;
          std::unique_lock<std::mutex> lock_render(mutex_render_);
          addPoints(points_tensor, colors_tensor, associated_keyframe_map);
        }
      } break;

        // case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA: {
        //   std::cout << "[Gaussian Mapper]Loop Closure Detected." <<
        //   std::endl;

        //   // Get the loop keyframe scale modification factor
        //   float loop_kf_scale = opr.mfScale;

        //   // Get new keyframes (scaled transformation applied in ORB-SLAM3)
        //   auto& associated_kfs = opr.associatedKeyFrames();
        //   // Mark the transformed points to avoid transforming more than once
        //   torch::Tensor point_not_transformed_flags = torch::full(
        //       {gaussians_->xyz_.size(0)}, true,
        //       torch::TensorOptions().device(device_type_).dtype(torch::kBool));
        //   if (record_loop_ply_)
        //     savePly(result_dir_ / (std::to_string(getIteration()) +
        //                            "_0_before_loop_correction"));
        //   int num_transformed = 0;
        //   // Add keyframes to the scene
        //   for (auto& kf : associated_kfs) {
        //     // Keyframe Id
        //     auto kfid = std::get<0>(kf);
        //     std::shared_ptr<GaussianKeyframe> pkf =
        //     scene_->getKeyframe(kfid);
        //     // In case new points are added in handleNewKeyframe()
        //     int64_t num_new_points =
        //         gaussians_->xyz_.size(0) -
        //         point_not_transformed_flags.size(0);
        //     if (num_new_points > 0)
        //       point_not_transformed_flags =
        //           torch::cat({point_not_transformed_flags,
        //                       torch::full({num_new_points}, true,
        //                                   point_not_transformed_flags.options())},
        //                      /*dim=*/0);
        //     // If kf is already in the scene, evaluate the change in pose,
        //     // if too large we perform loop correction on its visible model
        //     // points. If not in the scene, create a new one.
        //     if (pkf) {
        //       auto& pose = std::get<2>(kf);
        //       Sophus::SE3f original_pose =
        //           pkf->getPosef();  // original_pose = old, inv_pose = new
        //       Sophus::SE3f inv_pose = pose.inverse();
        //       Sophus::SE3f diff_pose = inv_pose * original_pose;
        //       bool large_rot = !diff_pose.rotationMatrix().isApprox(
        //           Eigen::Matrix3f::Identity(), large_rot_th_);
        //       bool large_trans = !diff_pose.translation().isMuchSmallerThan(
        //           1.0, large_trans_th_);
        //       if (large_rot || large_trans) {
        //         std::cout << "[Gaussian Mapper]Large loop correction
        //         detected, "
        //                      "transforming visible points of kf "
        //                   << kfid << std::endl;
        //         diff_pose.translation() -=
        //             inv_pose
        //                 .translation();  // t = (R_new * t_old + t_new) -
        //                 t_new
        //         diff_pose.translation() *=
        //             loop_kf_scale;  // t = s * (R_new * t_old)
        //         diff_pose.translation() +=
        //             inv_pose.translation();  // t = (s * R_new * t_old) +
        //             t_new
        //         torch::Tensor diff_pose_tensor =
        //             tensor_utils::EigenMatrix2TorchTensor(diff_pose.matrix(),
        //                                                   device_type_)
        //                 .transpose(0, 1);
        //         {
        //           std::unique_lock<std::mutex> lock_render(mutex_render_);
        //           gaussians_->scaledTransformVisiblePointsOfKeyframe(
        //               point_not_transformed_flags, diff_pose_tensor,
        //               pkf->world_view_transform_, pkf->full_proj_transform_,
        //               pkf->creation_iter_, stableNumIterExistence(),
        //               num_transformed,
        //               loop_kf_scale);  // selected xyz *= s
        //         }
        //         // Give loop keyframes times of use
        //         increaseKeyframeTimesOfUse(pkf,
        //                                    loop_closure_increased_times_of_use_);
        //       }
        //       // }
        //       pkf->setPose(pose.unit_quaternion().cast<double>(),
        //                    pose.translation().cast<double>());
        //       pkf->computeTransformTensors();
        //     } else {
        //       handleNewKeyframe(kf);
        //     }
        //   }
        //   if (record_loop_ply_)
        //     savePly(result_dir_ / (std::to_string(getIteration()) +
        //                            "_1_after_loop_correction"));
        //   // Get new points (scaled transformation applied in ORB-SLAM3, so
        //   this
        //   // step is performed at last to avoid scaling twice)
        //   auto& associated_points = opr.associatedMapPoints();
        //   auto& points = std::get<0>(associated_points);
        //   auto& colors = std::get<1>(associated_points);

        //   // Add new points to the model
        //   if (initial_mapped_ && points.size() >= 30) {
        //     torch::NoGradGuard no_grad;
        //     std::unique_lock<std::mutex> lock_render(mutex_render_);
        //     gaussians_->increasePcd(points, colors, getIteration());
        //   }

        //   // Mark this iteration
        //   loop_closure_iteration_ = true;
        // } break;

        // case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement: {
        //   std::cout << "[Gaussian Mapper]Scale refinement Detected.
        //   Transforming "
        //                "all kfs and points..."
        //             << std::endl;

        //   float s = opr.mfScale;
        //   Sophus::SE3f& T = opr.mT;
        //   if (initial_mapped_) {
        //     // Apply the scaled transformation on gaussian model points
        //     {
        //       std::unique_lock<std::mutex> lock_render(mutex_render_);
        //       gaussians_->applyScaledTransformation(s, T);
        //     }
        //     // Apply the scaled transformation to the scene
        //     scene_->applyScaledTransformation(s, T);
        //   } else {  // TODO: the workflow should not come here, delete this
        //   branch
        //     // Apply the scaled transformation to the cached points
        //     for (auto& pt : scene_->cached_point_cloud_) {
        //       // pt <- (s * Ryw * pt + tyw)
        //       auto& pt_xyz = pt.second.xyz_;
        //       pt_xyz *= s;
        //       pt_xyz = T.cast<double>() * pt_xyz;
        //     }

        //     // Apply the scaled transformation on gaussian keyframes
        //     for (auto& kfit : scene_->keyframes()) {
        //       std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
        //       Sophus::SE3f Twc = pkf->getPosef().inverse();
        //       Twc.translation() *= s;
        //       Sophus::SE3f Tyc = T * Twc;
        //       Sophus::SE3f Tcy = Tyc.inverse();
        //       pkf->setPose(Tcy.unit_quaternion().cast<double>(),
        //                    Tcy.translation().cast<double>());
        //       pkf->computeTransformTensors();
        //     }
        //   }
        // } break;

      default: {
        throw std::runtime_error("MappingOperation type not supported!");
      } break;
    }
  }
}

void GaussianMapper::handleNewKeyframe(std::tuple<unsigned long /*Id*/,
                                                  unsigned long /*CameraId*/,
                                                  Sophus::SE3f /*pose*/,
                                                  cv::Mat /*image*/,
                                                  bool /*isLoopClosure*/,
                                                  cv::Mat /*auxiliaryImage*/,
                                                  std::vector<float>,
                                                  std::vector<float>,
                                                  std::string>& kf) {
  std::shared_ptr<GaussianKeyframe> pkf =
      std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
  pkf->zfar_ = z_far_;
  pkf->znear_ = z_near_;
  // Pose
  auto& pose = std::get<2>(kf);
  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());
  cv::Mat imgRGB_undistorted, imgAux_undistorted;
  try {
    // Camera
    Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    imgRGB_undistorted = std::get<3>(kf);
    imgAux_undistorted = std::get<5>(kf);

    pkf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
        imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
  }
  // Add the new keyframe to the scene
  pkf->computeTransformTensors();
  scene_->addKeyframe(pkf, &kfid_shuffled_);

  // Give new keyframes times of use and add it to the training sliding window
  increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

  // Get dense point cloud from the new keyframe to accelerate training
  pkf->img_undist_ = imgRGB_undistorted;
  pkf->img_auxiliary_undist_ = imgAux_undistorted;
  pkf->kps_pixel_ = std::move(std::get<6>(kf));
  pkf->kps_point_local_ = std::move(std::get<7>(kf));
  if (isdoingInactiveGeoDensify()) increasePcdByKeyframeInactiveGeoDensify(pkf);

  // increasePcdByStereoReprojection(pkf);

  // Prepare multi resolution images for training
  if (device_type_ == torch::kCUDA) {
    cv::cuda::GpuMat img_gpu;
    img_gpu.upload(pkf->img_undist_);
    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
      cv::cuda::GpuMat img_resized;
      cv::cuda::resize(
          img_gpu, img_resized,
          cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
      pkf->gaus_pyramid_original_image_[l] =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
    }
  } else {
    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
      cv::Mat img_resized;
      cv::resize(
          pkf->img_undist_, img_resized,
          cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
      pkf->gaus_pyramid_original_image_[l] =
          tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
    }
  }
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

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf) {
  // auto start_timing = std::chrono::steady_clock::now();
  torch::NoGradGuard no_grad;

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  switch (this->sensor_type_) {
    case MONOCULAR: {
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
      assert(pkf->kps_pixel_.size() % 2 == 0);
      int N = pkf->kps_pixel_.size() / 2;
      torch::Tensor kps_pixel_tensor =
          torch::from_blob(pkf->kps_pixel_.data(), {N, 2},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device_type_);
      torch::Tensor kps_point_local_tensor =
          torch::from_blob(pkf->kps_point_local_.data(), {N, 3},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device_type_);
      torch::Tensor kps_has3D_tensor = torch::where(
          kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
          true, false);

      cv::cuda::GpuMat rgb_gpu;
      rgb_gpu.upload(pkf->img_undist_);
      torch::Tensor colors =
          tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
      colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

      auto result =
          monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
              kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor,
              colors, monocular_inactive_geo_densify_max_pixel_dist_,
              pkf->intr_, pkf->image_width_);
      torch::Tensor& points3D_valid = std::get<0>(result);
      torch::Tensor& colors_valid = std::get<1>(result);
      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);
      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    } break;
    case STEREO: {
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
      cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
      cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

      rgb_left_gpu.upload(pkf->img_undist_);
      rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

      // From CV_32FC3 to CV_32FC1
      cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
      cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

      // From CV_32FC1 to CV_8UC1
      gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
      gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

      // Compute disparity
      cv::cuda::GpuMat cv_disp;
      stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
      cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

      // Reproject to get 3D points
      cv::cuda::GpuMat cv_points3D;
      cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

      // From cv::cuda::GpuMat to torch::Tensor
      torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
      disp = disp.flatten(0, 1).contiguous();
      torch::Tensor points3D =
          tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
      points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
      torch::Tensor colors =
          tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
      colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

      // Clear undisired and unreliable stereo points
      torch::Tensor point_valid_flags = torch::full(
          {disp.size(0)}, false,
          torch::TensorOptions().dtype(torch::kBool).device(device_type_));
      int nkps_twice = pkf->kps_pixel_.size();
      int width = pkf->image_width_;
      for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
        int idx = static_cast<int>(/*u*/ pkf->kps_pixel_[kpidx]) +
                  static_cast<int>(/*v*/ pkf->kps_pixel_[kpidx + 1]) * width;
        // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
        // if (u < 0.3 * width || u > 0.7 * width)
        point_valid_flags[idx] = true;
      }
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(
              disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
              true, false));
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(
              disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()),
              true, false));

      torch::Tensor points3D_valid = points3D.index({point_valid_flags});
      torch::Tensor colors_valid = colors.index({point_valid_flags});

      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);

      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
    } break;
    case RGBD: {
      cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
      img_rgb_gpu.upload(pkf->img_undist_);
      img_depth_gpu.upload(pkf->img_auxiliary_undist_);

      // From cv::cuda::GpuMat to torch::Tensor
      torch::Tensor rgb =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
      rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
      torch::Tensor depth =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
      depth = depth.flatten(0, 1).contiguous();

      // To clear undisired and unreliable depth
      torch::Tensor point_valid_flags = torch::full(
          {depth.size(0)}, false /*true*/,
          torch::TensorOptions().dtype(torch::kBool).device(device_type_));
      int nkps_twice = pkf->kps_pixel_.size();
      int width = pkf->image_width_;
      for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
        int idx = static_cast<int>(/*u*/ pkf->kps_pixel_[kpidx]) +
                  static_cast<int>(/*v*/ pkf->kps_pixel_[kpidx + 1]) * width;
        point_valid_flags[idx] = true;
      }
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(depth > RGBD_min_depth_, true, false));
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(depth < RGBD_max_depth_, true, false));

      torch::Tensor colors_valid = rgb.index({point_valid_flags});

      // Reproject to get 3D points
      torch::Tensor points3D_valid;
      Camera& camera = scene_->cameras_.at(pkf->camera_id_);
      switch (camera.model_id_) {
        case Camera::PINHOLE: {
          points3D_valid = reprojectDepthPinhole(depth, point_valid_flags,
                                                 pkf->intr_, pkf->image_width_);
        } break;
        case Camera::FISHEYE: {
          // TODO: support fisheye camera?
          throw std::runtime_error(
              "[Gaussian Mapper]Fisheye cameras are not supported currently!");
        } break;
        default: {
          throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        } break;
      }
      points3D_valid = points3D_valid.index({point_valid_flags});

      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);

      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }

  pkf->done_inactive_geo_densify_ = true;
  ++depth_cached_;
  depth_cache_keyframes_[pkf->fid_] = pkf;

  if (depth_cached_ >= max_depth_cached_) {
    depth_cached_ = 0;
    // Add new points to the model
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    addPoints(depth_cache_points_, depth_cache_colors_, depth_cache_keyframes_);
    depth_cache_keyframes_.clear();
  }

  // auto end_timing = std::chrono::steady_clock::now();
  // auto completion_time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(
  //                 end_timing - start_timing).count();
  // std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify()
  // takes "
  //             << completion_time
  //             << " ms"
  //             << std::endl;
}

void GaussianMapper::increasePcdByStereoReprojection(
    std::shared_ptr<GaussianKeyframe> pkf) {
  // auto start_timing = std::chrono::steady_clock::now();
  torch::NoGradGuard no_grad;

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  switch (this->sensor_type_) {
    case MONOCULAR: {
      throw std::runtime_error("Can't densify with mono");
    } break;
    case STEREO: {
      // Get original image dimensions
      int orig_height = pkf->img_undist_.rows;
      int orig_width = pkf->img_undist_.cols;

      // Compute downsampling factor (adjust these values based on your memory
      // constraints) For example, scale = 4 means we use 1/16th of the pixels
      const int scale = 1;
      int new_height = orig_height / scale;
      int new_width = orig_width / scale;

      // Resize images before stereo matching
      cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
      cv::cuda::GpuMat rgb_left_small, rgb_right_small;

      // Upload and resize left image
      rgb_left_gpu.upload(pkf->img_undist_);
      cv::cuda::resize(rgb_left_gpu, rgb_left_small,
                       cv::Size(new_width, new_height));

      // Upload and resize right image
      rgb_right_gpu.upload(pkf->img_auxiliary_undist_);
      cv::cuda::resize(rgb_right_gpu, rgb_right_small,
                       cv::Size(new_width, new_height));

      // Convert to grayscale for disparity computation
      cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;
      cv::cuda::cvtColor(rgb_left_small, gray_left_gpu, cv::COLOR_RGB2GRAY);
      cv::cuda::cvtColor(rgb_right_small, gray_right_gpu, cv::COLOR_RGB2GRAY);

      // Convert to uint8 format required by stereo matching
      gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
      gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

      // Scale stereo parameters for the downsampled images
      cv::Mat Q_scaled = stereo_Q_.clone();
      Q_scaled.at<float>(0, 0) /= scale;  // fx
      Q_scaled.at<float>(1, 1) /= scale;  // fy
      Q_scaled.at<float>(0, 3) /= scale;  // cx
      Q_scaled.at<float>(1, 3) /= scale;  // cy

      // Compute disparity
      cv::cuda::GpuMat disparity_gpu;
      stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, disparity_gpu);
      disparity_gpu.convertTo(disparity_gpu, CV_32F, 1.0 / 16.0);

      // Reproject to 3D using scaled Q matrix
      cv::cuda::GpuMat points3D_gpu;
      cv::cuda::reprojectImageTo3D(disparity_gpu, points3D_gpu, Q_scaled, 3);

      // Convert to torch tensors
      torch::Tensor disparity =
          tensor_utils::cvGpuMat2TorchTensor_Float32(disparity_gpu);
      disparity = disparity.flatten(0, 1).contiguous();

      torch::Tensor points3D =
          tensor_utils::cvGpuMat2TorchTensor_Float32(points3D_gpu);
      points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();

      torch::Tensor colors =
          tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_small);
      colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

      // Filter points
      torch::Tensor valid_points = torch::logical_and(
          disparity > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
          disparity < static_cast<float>(stereo_cv_sgm_->getNumDisparities()));

      // Depth range filtering (adjust these thresholds as needed)
      valid_points = torch::logical_and(
          valid_points, points3D.index({torch::indexing::Slice(), 2}) > 4.0f);
      valid_points = torch::logical_and(
          valid_points, points3D.index({torch::indexing::Slice(), 2}) < 50.0f);

      // Further random subsampling if needed
      // Keep only 25% of the valid points randomly
      const float keep_probability = 0.5f;
      torch::Tensor random_mask =
          torch::rand_like(valid_points.to(torch::kFloat)) < keep_probability;
      valid_points = torch::logical_and(valid_points, random_mask);

      // Keep only valid points
      points3D = points3D.index({valid_points});
      colors = colors.index({valid_points});

      // Transform to world coordinates
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D, Twc_tensor);

      // Cache the points
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D;
        depth_cache_colors_ = colors;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors}, /*dim=*/0);
      }

      ++depth_cached_;
      depth_cache_keyframes_[pkf->fid_] = pkf;

      // Add to g{aussian model when cache is full
      if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        addPoints(depth_cache_points_, depth_cache_colors_,
                  depth_cache_keyframes_);
        depth_cache_keyframes_.clear();
      }

    } break;
    case RGBD: {
      throw std::runtime_error("Not implemented yet");
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }

  ++depth_cached_;
  depth_cache_keyframes_[pkf->fid_] = pkf;

  if (depth_cached_ >= max_depth_cached_) {
    depth_cached_ = 0;
    // Add new points to the model
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    addPoints(depth_cache_points_, depth_cache_colors_, depth_cache_keyframes_);
    depth_cache_keyframes_.clear();
  }

  // auto end_timing = std::chrono::steady_clock::now();
  // auto completion_time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(
  //                 end_timing - start_timing).count();
  // std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify()
  // takes "
  //             << completion_time
  //             << " ms"
  //             << std::endl;
}

void GaussianMapper::recordKeyframeRendered(
    torch::Tensor& rendered,
    torch::Tensor& ground_truth,
    unsigned long kfid,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix) {
  if (record_rendered_image_) {
    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
    cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" +
                                  std::to_string(kfid) + name_suffix + ".jpg"),
                image_cv);
  }

  if (record_ground_truth_image_) {
    auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
    cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
    gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
    cv::imwrite(
        result_gt_dir / (std::to_string(getIteration()) + "_" +
                         std::to_string(kfid) + name_suffix + "_gt.jpg"),
        gt_image_cv);
  }

  if (record_loss_image_) {
    torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
    auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
    cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
    loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
    cv::imwrite(
        result_loss_dir / (std::to_string(getIteration()) + "_" +
                           std::to_string(kfid) + name_suffix + "_loss.jpg"),
        loss_image_cv);
  }
}

cv::Mat GaussianMapper::renderFromPose(const Sophus::SE3f& Tcw,
                                       const int width,
                                       const int height,
                                       const bool main_vision) {
  if (!initial_mapped_ || getIteration() <= 0)
    return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
  pkf->zfar_ = z_far_;
  pkf->znear_ = z_near_;
  // Pose
  pkf->setPose(Tcw.unit_quaternion().cast<double>(),
               Tcw.translation().cast<double>());
  try {
    // Camera
    Camera& camera = scene_->cameras_.at(viewer_camera_id_);
    pkf->setCameraParams(camera);
    // Transformations
    pkf->computeTransformTensors();
  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
  }

  std::unique_lock lock_render(mutex_render_);

  // Get visible chunks using ChunkManager instead of updateActiveChunks
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->getVisibleChunks(pkf);
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    std::cout << "[" << chunk->getCoord().x << " " << chunk->getCoord().y << " "
              << chunk->getCoord().z << "], ";
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    } else {
      throw "[renderFromPose] Chunk/Gaussian not valid";
    }
  }
  std::cout << std::endl;

  // auto active_chunks = chunk_manager_->getActiveChunks();
  // std::vector<std::shared_ptr<GaussianModel>> models;
  // models.reserve(active_chunks.size());
  // for (const auto& [coord, chunk] : active_chunks) {
  //   if (chunk && chunk->gaussians_) {
  //     models.push_back(chunk->gaussians_);
  //   } else {
  //     throw std::runtime_error("[renderFromPose] Chunk/Gaussian not valid");
  //   }
  // }

  // Check if we have any valid models to render
  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    cv::Mat black_image = cv::Mat::zeros(height, width, CV_32FC3);
    return black_image;  // Early return if no valid models
  }

  // Render
  auto render_pkg = GaussianRenderer::render(
      models, pkf, height, width, pipe_params_, background_, override_color_);

  // Return rendered image
  return tensor_utils::torchTensor2CvMat_Float32(std::get<0>(render_pkg));
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float& dssim,
    float& psnr,
    float& psnr_gs,
    double& render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix) {
  auto start_timing = std::chrono::steady_clock::now();

  // Get visible chunks using ChunkManager instead of updateActiveChunks
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->getVisibleChunks(pkf);

  // Extract models from chunks
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    }
  }

  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    return;  // Early return if no valid models
  }

  auto render_pkg = GaussianRenderer::render(models, pkf, pkf->image_height_,
                                             pkf->image_width_, pipe_params_,
                                             background_, override_color_);
  auto rendered_image = std::get<0>(render_pkg);
  torch::cuda::synchronize();
  auto end_timing = std::chrono::steady_clock::now();
  auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_timing - start_timing)
                            .count();
  render_time = 1e-6 * render_time_ns;
  auto gt_image = pkf->original_image_;

  dssim = loss_utils::fast_ssim(rendered_image, gt_image).item().toFloat();
  psnr = loss_utils::psnr(rendered_image, gt_image).item().toFloat();
  psnr_gs = loss_utils::psnr_gaussian_splatting(rendered_image, gt_image)
                .item()
                .toFloat();

  recordKeyframeRendered(rendered_image, gt_image, pkf->fid_, result_img_dir,
                         result_gt_dir, result_loss_dir, name_suffix);
}

void GaussianMapper::renderAndRecordAllKeyframes(std::string name_suffix) {
  std::filesystem::path result_dir =
      result_dir_ / (std::to_string(getIteration()) + name_suffix);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  std::filesystem::path image_dir = result_dir / "image";
  if (record_rendered_image_)
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

  std::filesystem::path image_gt_dir = result_dir / "image_gt";
  if (record_ground_truth_image_)
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

  std::filesystem::path image_loss_dir = result_dir / "image_loss";
  if (record_loss_image_) {
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
  }

  std::filesystem::path render_time_path = result_dir / "render_time.txt";
  std::ofstream out_time(render_time_path);
  out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, "
              "time(milliseconds)"
           << std::endl;

  std::filesystem::path dssim_path = result_dir / "dssim.txt";
  std::ofstream out_dssim(dssim_path);
  out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

  std::filesystem::path psnr_path = result_dir / "psnr.txt";
  std::ofstream out_psnr(psnr_path);
  out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

  std::filesystem::path psnr_gs_path =
      result_dir / "psnr_gaussian_splatting.txt";
  std::ofstream out_psnr_gs(psnr_gs_path);
  out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting"
              << std::endl;

  std::size_t nkfs = scene_->keyframes().size();
  auto kfit = scene_->keyframes().begin();
  float dssim, psnr, psnr_gs;
  double render_time;
  for (std::size_t i = 0; i < nkfs; ++i) {
    renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time,
                            image_dir, image_gt_dir, image_loss_dir);
    out_time << (*kfit).first << " " << std::fixed << std::setprecision(8)
             << render_time << std::endl;

    out_dssim << (*kfit).first << " " << std::fixed << std::setprecision(10)
              << dssim << std::endl;
    out_psnr << (*kfit).first << " " << std::fixed << std::setprecision(10)
             << psnr << std::endl;
    out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10)
                << psnr_gs << std::endl;

    ++kfit;
  }
}

void GaussianMapper::savePly(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  keyframesToJson(result_dir);
  saveModelParams(result_dir);

  std::filesystem::path ply_dir = result_dir / "point_cloud";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  // Todo fix
  // gaussians_->savePly(ply_dir / "point_cloud.ply");
  // gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  std::filesystem::path result_path = result_dir / "cameras.json";
  std::ofstream out_stream;
  out_stream.open(result_path);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open json file at " +
                             result_path.string());

  Json::Value json_root;
  Json::StreamWriterBuilder builder;
  const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  int i = 0;
  for (const auto& kfit : scene_->keyframes()) {
    const auto pkf = kfit.second;
    Eigen::Matrix4f Rt;
    Rt.setZero();
    Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = pkf->t_.cast<float>();
    Rt.topRightCorner<3, 1>() = t;
    Rt(3, 3) = 1.0f;

    Eigen::Matrix4f Twc = Rt.inverse();
    Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
    Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

    Json::Value json_kf;
    json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
    json_kf["img_name"] =
        pkf->img_filename_;  // (std::to_string(getIteration()) + "_" +
                             // std::to_string(pkf->fid_));
    json_kf["width"] = pkf->image_width_;
    json_kf["height"] = pkf->image_height_;

    json_kf["position"][0] = pos.x();
    json_kf["position"][1] = pos.y();
    json_kf["position"][2] = pos.z();

    json_kf["rotation"][0][0] = rot(0, 0);
    json_kf["rotation"][0][1] = rot(0, 1);
    json_kf["rotation"][0][2] = rot(0, 2);
    json_kf["rotation"][1][0] = rot(1, 0);
    json_kf["rotation"][1][1] = rot(1, 1);
    json_kf["rotation"][1][2] = rot(1, 2);
    json_kf["rotation"][2][0] = rot(2, 0);
    json_kf["rotation"][2][1] = rot(2, 1);
    json_kf["rotation"][2][2] = rot(2, 2);

    json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
    json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

    json_root[i] = Json::Value(json_kf);
    ++i;
  }

  writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  std::filesystem::path result_path = result_dir / "cfg_args";
  std::ofstream out_stream;
  out_stream.open(result_path);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open file at " + result_path.string());

  out_stream << "Namespace("
             << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
             << "images=" << "\'" << model_params_.images_ << "\', "
             << "model_path=" << "\'" << model_params_.model_path_.string()
             << "\', "
             << "resolution=" << model_params_.resolution_ << ", "
             << "sh_degree=" << model_params_.sh_degree_ << ", "
             << "source_path=" << "\'" << model_params_.source_path_.string()
             << "\', "
             << "white_background="
             << (model_params_.white_background_ ? "True" : "False") << ", "
             << ")";

  out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir,
                                            std::string name_suffix) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  std::filesystem::path result_path =
      result_dir / ("keyframe_used_times" + name_suffix + ".txt");
  std::ofstream out_stream;
  out_stream.open(result_path, std::ios::app);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open json at " + result_path.string());

  out_stream << "##[Gaussian Mapper]Iteration " << getIteration()
             << " keyframe id, used times, remaining times:\n";
  for (const auto& used_times_it : kfs_used_times_)
    out_stream
        << used_times_it.first << " " << used_times_it.second << " "
        << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
        << "\n";
  out_stream << "##=========================================" << std::endl;

  out_stream.close();
}

int GaussianMapper::getIteration() {
  std::unique_lock<std::mutex> lock(mutex_status_);
  return iteration_;
}
void GaussianMapper::increaseIteration(const int inc) {
  std::unique_lock<std::mutex> lock(mutex_status_);
  iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.percent_dense_ = percent_dense;
  // gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  VariableParameters params;
  params.position_lr_init = opt_params_.position_lr_init_;
  params.feature_lr = opt_params_.feature_lr_;
  params.opacity_lr = opt_params_.opacity_lr_;
  params.scaling_lr = opt_params_.scaling_lr_;
  params.rotation_lr = opt_params_.rotation_lr_;
  params.percent_dense = opt_params_.percent_dense_;
  params.lambda_dssim = opt_params_.lambda_dssim_;
  params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
  params.densify_grad_th = opt_params_.densify_grad_threshold_;
  params.densify_interval = opt_params_.densification_interval_;
  params.new_kf_times_of_use = new_keyframe_times_of_use_;
  params.stable_num_iter_existence = stable_num_iter_existence_;
  params.keep_training = keep_training_;
  params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
  params.do_inactive_geo_densify = inactive_geo_densify_;
  return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters& params) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.position_lr_init_ = params.position_lr_init;
  opt_params_.feature_lr_ = params.feature_lr;
  opt_params_.opacity_lr_ = params.opacity_lr;
  opt_params_.scaling_lr_ = params.scaling_lr;
  opt_params_.rotation_lr_ = params.rotation_lr;
  opt_params_.percent_dense_ = params.percent_dense;
  // gaussians_->setPercentDense(params.percent_dense);
  opt_params_.lambda_dssim_ = params.lambda_dssim;
  opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
  opt_params_.densify_grad_threshold_ = params.densify_grad_th;
  opt_params_.densification_interval_ = params.densify_interval;
  new_keyframe_times_of_use_ = params.new_kf_times_of_use;
  stable_num_iter_existence_ = params.stable_num_iter_existence;
  keep_training_ = params.keep_training;
  do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
  inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path,
                             std::filesystem::path camera_path) {
  // this->gaussians_->loadPly(ply_path);

  // Camera
  if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
    cv::FileStorage camera_file(camera_path.string().c_str(),
                                cv::FileStorage::READ);
    if (!camera_file.isOpened())
      throw std::runtime_error(
          "[Gaussian Mapper]Failed to open settings file at: " +
          camera_path.string());

    Camera camera;
    camera.camera_id_ = 0;
    camera.width_ = camera_file["Camera.w"].operator int();
    camera.height_ = camera_file["Camera.h"].operator int();

    std::string camera_type = camera_file["Camera.type"].string();
    if (camera_type == "Pinhole") {
      camera.setModelId(Camera::CameraModelType::PINHOLE);

      float fx = camera_file["Camera.fx"].operator float();
      float fy = camera_file["Camera.fy"].operator float();
      float cx = camera_file["Camera.cx"].operator float();
      float cy = camera_file["Camera.cy"].operator float();

      float k1 = camera_file["Camera.k1"].operator float();
      float k2 = camera_file["Camera.k2"].operator float();
      float p1 = camera_file["Camera.p1"].operator float();
      float p2 = camera_file["Camera.p2"].operator float();
      float k3 = camera_file["Camera.k3"].operator float();

      cv::Mat K =
          (cv::Mat_<float>(3, 3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f);

      camera.params_[0] = fx;
      camera.params_[1] = fy;
      camera.params_[2] = cx;
      camera.params_[3] = cy;

      std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
      camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
      camera.initUndistortRectifyMapAndMask(
          K, cv::Size(camera.width_, camera.height_), K, false);

      undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(camera.undistort_mask,
                                                  device_type_);

      cv::Mat viewer_main_undistort_mask;
      int viewer_image_height_main_ =
          camera.height_ * rendered_image_viewer_scale_main_;
      int viewer_image_width_main_ =
          camera.width_ * rendered_image_viewer_scale_main_;
      cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                 cv::Size(viewer_image_width_main_, viewer_image_height_main_));
      viewer_main_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask,
                                                  device_type_);

    } else {
      throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " +
                               camera_path.string());
    }

    if (!viewer_camera_id_set_) {
      viewer_camera_id_ = camera.camera_id_;
      viewer_camera_id_set_ = true;
    }
    this->scene_->addCamera(camera);
  }

  // Ready
  this->initial_mapped_ = true;
  increaseIteration();
}

std::tuple<torch::Tensor, torch::Tensor> GaussianMapper::filterPointsByDepth(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  const int num_points = points.size(0);
  auto device = points.device();
  auto options = torch::TensorOptions().device(device).dtype(points.dtype());

  // Initialize validity mask for all points (start with all false)
  torch::Tensor valid_mask = torch::zeros(
      {num_points}, torch::TensorOptions().device(device).dtype(torch::kBool));

  // Process each keyframe
  for (const auto& [kfid, keyframe] : keyframes) {
    if (!keyframe->set_pose_) continue;

    // Get the rotation and translation from Sophus SE3
    Eigen::Matrix3d R = keyframe->Tcw_.rotationMatrix();
    Eigen::Vector3d t = keyframe->Tcw_.translation();

    // Convert to tensors and ensure same dtype as points
    torch::Tensor R_tensor =
        torch::from_blob(const_cast<double*>(R.data()), {3, 3},
                         torch::TensorOptions().dtype(torch::kDouble))
            .to(device)
            .to(points.dtype());

    torch::Tensor t_tensor =
        torch::from_blob(const_cast<double*>(t.data()), {3},
                         torch::TensorOptions().dtype(torch::kDouble))
            .to(device)
            .to(points.dtype());

    // Transform points: R * points + t
    torch::Tensor points_cam = torch::matmul(points, R_tensor.t());
    points_cam += t_tensor.unsqueeze(0);

    // Extract depths (z-coordinates)
    torch::Tensor depths = points_cam.select(1, 2);

    // Check depth constraints
    torch::Tensor valid_in_frame =
        (depths >= keyframe->znear_) & (depths <= keyframe->zfar_);

    // Update global validity mask
    valid_mask = valid_mask | valid_in_frame;
  }

  // Count valid points
  int64_t valid_count = valid_mask.sum().item<int64_t>();

  // Use boolean indexing to filter points and colors
  torch::Tensor filtered_points = points.index({valid_mask});
  torch::Tensor filtered_colors = colors.index({valid_mask});

  return std::make_tuple(filtered_points, filtered_colors);
}

void GaussianMapper::handleNewFrameExternal(const cv::Mat& rgb_image,
                                            const cv::Mat& depth_or_right_image,
                                            const Sophus::SE3f& pose,
                                            const double timestamp) {
  return;
  // std::cout << "New external frame" << std::endl;
  // frame_queue_.push(Frame(rgb_image, depth_or_right_image, pose,
  // timestamp));
}

void GaussianMapper::run_external_poses() { return; }

// External mode initialization
GaussianMapper::GaussianMapper(const SystemSensorType sensor_type,
                               const string& orb_settings_path,
                               std::filesystem::path gaussian_config_file_path,
                               std::filesystem::path result_dir,
                               int seed,
                               torch::DeviceType device_type)
    : pSLAM_(nullptr),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      sensor_type_(sensor_type) {
  return;
}

std::vector<std::shared_ptr<GaussianModel>>
GaussianMapper::selectRandomModelSubset(
    const std::vector<std::shared_ptr<GaussianModel>>& allModels,
    size_t subset_size) {
  // torch::cuda::synchronize();
  // Create a vector for the result
  std::vector<std::shared_ptr<GaussianModel>> subset;

  // Make sure we don't try to select more elements than available
  subset_size = std::min(subset_size, allModels.size());
  if (subset_size == 0) return subset;

  // Create a vector of indices
  std::vector<size_t> indices(allModels.size());
  std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, 2, ...

  // Shuffle the indices
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Select the first subset_size elements
  subset.reserve(subset_size);
  for (size_t i = 0; i < subset_size; ++i) {
    subset.push_back(allModels[indices[i]]);
  }
  // torch::cuda::synchronize();

  return subset;
}

void GaussianMapper::initializeChunkManagement() {
  // Create the chunk manager with direct model parameters
  chunk_manager_ = std::make_shared<ChunkManager>(
      model_params_, opt_params_, chunk_save_dir_, chunk_size_, overlap_margin_,
      max_chunks_in_memory_);

  // Create the keyframe selector with just the chunk manager
  keyframe_selector_ = std::make_shared<KeyframeSelector>(chunk_manager_);
}

void GaussianMapper::addPoints(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes) {
  std::cout << "addPoints called in GaussianMapper" << std::endl;
  // Make sure chunk manager has current iteration
  chunk_manager_->setCurrentIteration(getIteration());

  // Delegate to chunk manager
  chunk_manager_->addPointsToChunks(points, colors, keyframes,
                                    scene_->cameras_extent_);
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::selectLocalityAwareKeyframe() {
  if (!keyframe_selector_) {
    // Fallback to old method
    std::cout << "Smart Keyframe selection unavailable, fallback to OG "
                 "method."
              << std::endl;
    return useOneRandomSlidingWindowKeyframe();
  }

  auto keyframe = keyframe_selector_->selectKeyframe(
      scene_->keyframes(), kfs_loss_, kfs_used_times_, getIteration());

  if (keyframe) {
    // Handle keyframe usage internally inside GaussianMapper
    increaseKeyframeTimesOfUse(keyframe, -1);
  }

  return keyframe;
}

// Predict upcoming keyframes for prefetching
std::vector<std::shared_ptr<GaussianKeyframe>>
GaussianMapper::predictUpcomingKeyframes(int count) {
  if (!keyframe_selector_) {
    return {};
  }

  return keyframe_selector_->predictUpcomingKeyframes(scene_->keyframes(),
                                                      kfs_loss_, count);
}

/**
 * Generates a smooth fly-through video along keyframe path with constant speed
 *
 * @param output_path Directory where frames and video will be saved
 * @param width Width of the output video
 * @param height Height of the output video
 * @param fps Frames per second
 * @param duration_seconds Total duration of the video
 */
void GaussianMapper::renderFlyThroughVideo(const std::string& output_path,
                                           int width,
                                           int height,
                                           int fps,
                                           float duration_seconds) {
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_path);

  // 1. Get all keyframes sorted by fid
  auto keyframes_map = scene_->getAllKeyframes();
  std::vector<std::size_t> keyframe_ids;
  for (const auto& [fid, kf] : keyframes_map) {
    keyframe_ids.push_back(fid);
  }
  std::sort(keyframe_ids.begin(), keyframe_ids.end());

  // Check if we have enough keyframes
  if (keyframe_ids.size() < 2) {
    std::cerr << "Need at least 2 keyframes to create a fly-through video"
              << std::endl;
    return;
  }

  // 2. Extract keyframe positions and orientations
  std::vector<Eigen::Vector3d> keyframe_positions;
  std::vector<Eigen::Quaterniond> keyframe_orientations;
  for (const auto& fid : keyframe_ids) {
    auto kf = keyframes_map[fid];
    // Get camera-to-world transform and invert to get world-to-camera
    Sophus::SE3d Tcw = kf->getPose();
    Sophus::SE3d Twc = Tcw.inverse();
    keyframe_positions.push_back(Twc.translation());
    keyframe_orientations.push_back(Twc.unit_quaternion());
  }

  // 3. Create a smooth path through the keyframe positions
  std::vector<Eigen::Vector3d> path_points =
      createSmoothPath(keyframe_positions, 20);

  // 4. Sample the path at equal distances to ensure constant speed
  std::vector<Eigen::Vector3d> sampled_positions;
  std::vector<Eigen::Quaterniond> sampled_orientations;
  samplePathConstantSpeed(path_points, keyframe_positions,
                          keyframe_orientations,
                          static_cast<int>(duration_seconds * fps),
                          sampled_positions, sampled_orientations);

  // 5. Render each frame
  int total_frames = sampled_positions.size();
  std::vector<std::string> frame_paths;  // Store frame paths for cleanup later

  for (int i = 0; i < total_frames; i++) {
    // Create world-to-camera transform
    Sophus::SE3d Twc(sampled_orientations[i], sampled_positions[i]);
    Sophus::SE3f Tcw = Twc.inverse().cast<float>();

    // Render frame
    cv::Mat frame = renderFromPose(Tcw, width, height, true);

    // Convert if needed (assuming renderFromPose returns float image)
    cv::Mat output_frame;
    frame.convertTo(output_frame, CV_8UC3, 255.0);

    // Convert from BGR to RGB color space
    cv::cvtColor(output_frame, output_frame, cv::COLOR_BGR2RGB);

    // Save frame
    std::string frame_path =
        output_path + "/frame_" + std::to_string(i + 1) + ".png";
    cv::imwrite(frame_path, output_frame);
    frame_paths.push_back(frame_path);  // Store path for later cleanup

    {
      std::cout << "Rendered frame " << i + 1 << "/" << total_frames
                << std::endl;
    }
  }

  // 6. Combine frames into video using ffmpeg
  std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) + " -i " +
                    output_path + "/frame_%d.png" +
                    " -c:v libx264 -crf 18 -pix_fmt yuv420p " + output_path +
                    "/flythrough.mp4";

  std::cout << "Creating video with command: " << cmd << std::endl;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr
        << "Failed to create video using ffmpeg. Check if ffmpeg is installed."
        << std::endl;
  } else {
    std::cout << "Video created successfully at " << output_path
              << "/flythrough.mp4" << std::endl;

    // 7. Delete all frame files
    std::cout << "Cleaning up frame files..." << std::endl;
    int deleted_frames = 0;
    for (const auto& frame_path : frame_paths) {
      if (std::filesystem::remove(frame_path)) {
        deleted_frames++;
      }
    }
    std::cout << "Deleted " << deleted_frames << "/" << frame_paths.size()
              << " frames." << std::endl;
  }
}

/**
 * Creates a smooth path through the given points using Catmull-Rom splines
 */
std::vector<Eigen::Vector3d> GaussianMapper::createSmoothPath(
    const std::vector<Eigen::Vector3d>& keypoints,
    int points_per_segment) {
  std::vector<Eigen::Vector3d> path;
  if (keypoints.size() < 2) return keypoints;

  // For only 2 points, do linear interpolation
  if (keypoints.size() == 2) {
    for (int i = 0; i <= points_per_segment; i++) {
      double t = static_cast<double>(i) / points_per_segment;
      path.push_back(keypoints[0] * (1 - t) + keypoints[1] * t);
    }
    return path;
  }

  // Create extended points array with extrapolated endpoints
  // This handles boundary conditions for Catmull-Rom
  std::vector<Eigen::Vector3d> extended;
  extended.push_back(keypoints[0] * 2 - keypoints[1]);  // Extrapolate start
  extended.insert(extended.end(), keypoints.begin(), keypoints.end());
  extended.push_back(keypoints.back() * 2 -
                     keypoints[keypoints.size() - 2]);  // Extrapolate end

  // Interpolate each segment
  for (size_t i = 1; i < extended.size() - 2; i++) {
    for (int j = 0; j < points_per_segment; j++) {
      double t = static_cast<double>(j) / points_per_segment;
      path.push_back(catmullRomInterpolate(
          extended[i - 1], extended[i], extended[i + 1], extended[i + 2], t));
    }
  }

  // Add the final point
  path.push_back(keypoints.back());

  return path;
}

/**
 * Catmull-Rom spline interpolation for a single point
 */
Eigen::Vector3d GaussianMapper::catmullRomInterpolate(const Eigen::Vector3d& p0,
                                                      const Eigen::Vector3d& p1,
                                                      const Eigen::Vector3d& p2,
                                                      const Eigen::Vector3d& p3,
                                                      double t) {
  double t2 = t * t;
  double t3 = t2 * t;

  // Catmull-Rom basis functions
  double h1 = -0.5 * t3 + t2 - 0.5 * t;
  double h2 = 1.5 * t3 - 2.5 * t2 + 1.0;
  double h3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  double h4 = 0.5 * t3 - 0.5 * t2;

  return h1 * p0 + h2 * p1 + h3 * p2 + h4 * p3;
}

/**
 * Compute arc lengths along a path
 */
std::vector<double> GaussianMapper::computeArcLengths(
    const std::vector<Eigen::Vector3d>& path) {
  std::vector<double> arc_lengths(path.size(), 0.0);
  for (size_t i = 1; i < path.size(); i++) {
    double segment_length = (path[i] - path[i - 1]).norm();
    arc_lengths[i] = arc_lengths[i - 1] + segment_length;
  }
  return arc_lengths;
}

/**
 * Sample the path at equal distances and interpolate orientations
 */
void GaussianMapper::samplePathConstantSpeed(
    const std::vector<Eigen::Vector3d>& path,
    const std::vector<Eigen::Vector3d>& keyframe_positions,
    const std::vector<Eigen::Quaterniond>& keyframe_orientations,
    int num_samples,
    std::vector<Eigen::Vector3d>& sampled_positions,
    std::vector<Eigen::Quaterniond>& sampled_orientations) {
  sampled_positions.clear();
  sampled_orientations.clear();

  if (path.empty() || keyframe_positions.empty() ||
      keyframe_orientations.empty()) {
    return;
  }

  // 1. Compute arc lengths
  std::vector<double> arc_lengths = computeArcLengths(path);
  double total_length = arc_lengths.back();

  // 2. Map keyframes to path parameters
  std::vector<double> keyframe_parameters;
  mapKeyframesToPath(keyframe_positions, path, arc_lengths,
                     keyframe_parameters);

  // 3. Sample at equal distances
  for (int i = 0; i < num_samples; i++) {
    double t = static_cast<double>(i) / (num_samples - 1);  // Normalized [0,1]
    double target_length = total_length * t;

    // Position at this arc length
    Eigen::Vector3d position =
        samplePositionAtArcLength(path, arc_lengths, target_length);

    // Path parameter
    double path_param = target_length / total_length;

    // Interpolate orientation
    Eigen::Quaterniond orientation = interpolateOrientation(
        path_param, keyframe_parameters, keyframe_orientations);

    sampled_positions.push_back(position);
    sampled_orientations.push_back(orientation);
  }
}

/**
 * Map keyframe positions to their closest corresponding points on the path
 */
void GaussianMapper::mapKeyframesToPath(
    const std::vector<Eigen::Vector3d>& keyframe_positions,
    const std::vector<Eigen::Vector3d>& path,
    const std::vector<double>& arc_lengths,
    std::vector<double>& keyframe_parameters) {
  keyframe_parameters.clear();
  double total_length = arc_lengths.back();

  for (const auto& kf_pos : keyframe_positions) {
    // Find closest point on path
    size_t closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < path.size(); i++) {
      double dist = (kf_pos - path[i]).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_idx = i;
      }
    }

    // Parameter is normalized arc length
    double param = arc_lengths[closest_idx] / total_length;
    keyframe_parameters.push_back(param);
  }

  // Ensure parameters are strictly increasing (required for interpolation)
  for (size_t i = 1; i < keyframe_parameters.size(); i++) {
    if (keyframe_parameters[i] <= keyframe_parameters[i - 1]) {
      keyframe_parameters[i] = keyframe_parameters[i - 1] + 0.001;
    }
  }
}

/**
 * Sample a position at a specific arc length along the path
 */
Eigen::Vector3d GaussianMapper::samplePositionAtArcLength(
    const std::vector<Eigen::Vector3d>& path,
    const std::vector<double>& arc_lengths,
    double target_length) {
  // Find segment containing this arc length
  auto it =
      std::lower_bound(arc_lengths.begin(), arc_lengths.end(), target_length);
  int idx = std::distance(arc_lengths.begin(), it);

  if (idx >= path.size()) {
    return path.back();  // Beyond the end
  } else if (idx == 0) {
    return path.front();  // Before the start
  } else {
    // Interpolate within segment
    double segment_start = arc_lengths[idx - 1];
    double segment_length = arc_lengths[idx] - segment_start;
    double t = segment_length > 0
                   ? (target_length - segment_start) / segment_length
                   : 0;

    return path[idx - 1] * (1 - t) + path[idx] * t;
  }
}

/**
 * Interpolate orientation using SLERP based on path parameter
 */
Eigen::Quaterniond GaussianMapper::interpolateOrientation(
    double param,
    const std::vector<double>& keyframe_parameters,
    const std::vector<Eigen::Quaterniond>& keyframe_orientations) {
  // Handle boundary cases
  if (param <= keyframe_parameters.front()) {
    return keyframe_orientations.front();
  }
  if (param >= keyframe_parameters.back()) {
    return keyframe_orientations.back();
  }

  // Find the keyframes before and after this parameter
  size_t idx = 0;
  while (idx < keyframe_parameters.size() - 1 &&
         keyframe_parameters[idx + 1] < param) {
    idx++;
  }

  // SLERP between these orientations
  double segment_length =
      keyframe_parameters[idx + 1] - keyframe_parameters[idx];
  double t = segment_length > 0
                 ? (param - keyframe_parameters[idx]) / segment_length
                 : 0;
  t = std::max(0.0, std::min(1.0, t));  // Clamp to [0,1]

  return keyframe_orientations[idx].slerp(t, keyframe_orientations[idx + 1]);
}