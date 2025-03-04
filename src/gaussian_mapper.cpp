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

#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"

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

  // Initialize scene and model
  gaussians_ = std::make_shared<GaussianModel>(model_params_);
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

  // Third loop: After SLAM training
  while (getIteration() < 20000) {
    trainForOneIteration();
  }

  // Fourth loop: Tail gaussian optimization
  int densify_interval = densifyInterval();
  int n_delay_iters = densify_interval * 0.8;
  while (getIteration() - SLAM_stop_iter < n_delay_iters ||
         getIteration() % densify_interval < n_delay_iters ||
         isKeepingTraining()) {
    trainForOneIteration();
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

/**
 * @brief The training iteration body
 *
 */
void GaussianMapper::trainForOneIteration() {
  increaseIteration(1);
  auto iter_start_timing = std::chrono::steady_clock::now();

  // Pick a random Camera
  std::shared_ptr<GaussianKeyframe> viewpoint_cam =
      useOneRandomSlidingWindowKeyframe();
  if (!viewpoint_cam) {
    increaseIteration(-1);
    return;
  }

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  // if (isdoingInactiveGeoDensify() &&
  // !viewpoint_cam->done_inactive_geo_densify_)
  //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

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

  // Mutex lock for usage of the gaussian model
  std::unique_lock<std::mutex> lock_render(mutex_render_);

  if (active_chunks_.empty()) {
    std::cout << "[Optimization] No active chunks to optimize" << std::endl;
    return;  // Early return if there are no chunks
  }

  // Collect valid models for rendering
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(active_chunks_.size());
  for (const auto& [coord, chunk] : active_chunks_) {
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    } else {
      std::cerr << "[Optimization] Chunk/Model " << coord.x << " " << coord.y
                << " " << coord.z << " invalid" << std::endl;
      exit(1);
      // Skip this chunk - don't add it to models
    }
  }

  // Check if we have any valid models to render
  if (models.empty()) {
    std::cout << "[Optimization] No valid models to render" << std::endl;
    return;  // Early return if no valid models
  }

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
  }

  // Render
  std::cout << "\r[Gaussian Mapper] Rendering "
            << gaussians_->getXYZ().sizes()[0] << "..." << std::flush;
  auto render_pkg = GaussianRenderer::render(
      models, viewpoint_cam, image_height, image_width, gaussians_,
      pipe_params_, background_, override_color_);
  auto rendered_image = std::get<0>(render_pkg);
  std::vector<torch::Tensor> screenspace_points_vec = std::get<1>(render_pkg);
  std::vector<torch::Tensor> radii_vec = std::get<2>(render_pkg);

  // Loss
  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto Ll1 = l1_loss(rendered_image, gt_image, 1.0f);
  auto Lssim = loss_utils::fast_ssim(rendered_image, gt_image);
  float lambda_dssim = lambdaDssim();
  auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);

  // Todo
  // if (opt_params_.opacity_reg_) {
  //   loss += opt_params_.opacity_reg_ *
  //           gaussians_->getOpacityActivation().abs().mean();
  // }
  loss.backward();

  torch::cuda::synchronize();

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
                                     densify_min_opacity_,  // 0.005,//
                                     scene_->cameras_extent_, size_threshold);
        }

        if (opacityResetInterval() &&
            (getIteration() % opacityResetInterval() == 0 ||
             (model_params_.white_background_ &&
              getIteration() == opt_params_.densify_from_iter_)))
          gaussians->resetOpacity();
      }
    }

    auto iter_end_timing = std::chrono::steady_clock::now();
    auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         iter_end_timing - iter_start_timing)
                         .count();

    // Log and save
    if (training_report_interval_ &&
        (getIteration() % training_report_interval_ == 0))
      trainingReport(getIteration(), opt_params_.iterations_, Ll1, loss,
                     ema_loss_for_log_, iter_time, *gaussians_, *scene_,
                     pipe_params_, background_);
    if ((all_keyframes_record_interval_ &&
         getIteration() % all_keyframes_record_interval_ == 0)) {
      renderAndRecordAllKeyframes();
      savePly(result_dir_ / std::to_string(getIteration()) / "ply");
    }

    if (loop_closure_iteration_) loop_closure_iteration_ = false;

    for (const auto& gaussians : models) {
      // Optimizer step
      if (getIteration() < opt_params_.iterations_ ||
          opt_params_.iterations_ == -1) {
        gaussians->optimizer_->step();
        gaussians->optimizer_->zero_grad(true);
      }
    }
  }
}

bool GaussianMapper::isStopped() {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop) {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  this->stopped_ = going_to_stop;
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

  // Collect valid models for rendering
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(active_chunks_.size());
  for (const auto& [coord, chunk] : active_chunks_) {
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    } else {
      std::cerr << "[renderFromPose] Chunk/Model " << coord.x << " " << coord.y
                << " " << coord.z << " invalid" << std::endl;
      exit(1);
      // Skip this chunk - don't add it to models
    }
  }

  // Check if we have any valid models to render
  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    cv::Mat black_image = cv::Mat::zeros(height, width, CV_32FC3);
    return black_image;  // Early return if no valid models
  }

  std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>>
      render_pkg;
  {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    // Render
    render_pkg =
        GaussianRenderer::render(models, pkf, height, width, gaussians_,
                                 pipe_params_, background_, override_color_);
  }

  // Result
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

  // Collect valid models for rendering
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(active_chunks_.size());
  for (const auto& [coord, chunk] : active_chunks_) {
    if (chunk && chunk->gaussians_) {
      models.push_back(chunk->gaussians_);
    } else {
      std::cerr << "[renderFromPose] Chunk/Model " << coord.x << " " << coord.y
                << " " << coord.z << " invalid" << std::endl;
      exit(1);
      // Skip this chunk - don't add it to models
    }
  }

  // Check if we have any valid models to render
  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    return;  // Early return if no valid models
  }

  auto render_pkg = GaussianRenderer::render(
      models, pkf, pkf->image_height_, pkf->image_width_, gaussians_,
      pipe_params_, background_, override_color_);
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

  gaussians_->savePly(ply_dir / "point_cloud.ply");
  gaussians_->saveSparsePointsPly(result_dir / "input.ply");
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
  gaussians_->setPercentDense(percent_dense);
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
  gaussians_->setPercentDense(params.percent_dense);
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
  this->gaussians_->loadPly(ply_path);

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

ChunkCoord GaussianMapper::getChunkCoord(const Eigen::Vector3f& position) {
  float effective_size = chunk_size_ - overlap_margin_;  // Account for overlap
  return ChunkCoord{
      static_cast<int>(std::floor(position.x() / effective_size)),
      static_cast<int>(std::floor(position.y() / effective_size)),
      static_cast<int>(std::floor(position.z() / effective_size))};
}

bool GaussianMapper::isInViewFrustum(const ChunkCoord& coord,
                                     const Eigen::Vector3f& camera_pos,
                                     const Eigen::Vector3f& view_dir) {
  // Called by updateActiveChunks
  // Simple frustum check for now
  // Should be replaced with proper frustum culling
  Eigen::Vector3f chunk_center((coord.x + 0.5f) * chunk_size_,
                               (coord.y + 0.5f) * chunk_size_,
                               (coord.z + 0.5f) * chunk_size_);

  Eigen::Vector3f to_chunk = chunk_center - camera_pos;
  float dist = to_chunk.norm();

  // Simple distance and angle check
  if (dist > 5.0f * chunk_size_) return false;  // Too far

  float angle = std::acos(to_chunk.normalized().dot(view_dir));
  return angle < M_PI / 3.0f;  // Within 60 degree cone
}

void GaussianMapper::updateActiveChunks(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  // Called by trainForOneIteration, renderFromPose & renderAndRecordKeyframe
  if (!keyframe) {
    std::cerr << "Error: Null keyframe passed to updateActiveChunks"
              << std::endl;
    return;
  }
  // Extract camera pose
  Sophus::SE3d camera_pose = keyframe->getPose();

  // The camera position is the translation part of the inverse transform
  // Since Tcw_ is camera-to-world, we need its inverse to get world-to-camera
  Sophus::SE3d Twc = camera_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();

  // The view direction is the negative z-axis of the camera frame transformed
  // to world frame We use the rotation matrix from camera to world transform
  Eigen::Vector3f view_direction =
      -(camera_pose.rotationMatrix() * Eigen::Vector3d(0, 0, 1)).cast<float>();
  view_direction.normalize();

  // Find which chunks should be active based on view frustum
  std::unordered_set<ChunkCoord, ChunkCoordHash> needed_chunks;

  // Get chunk coordinate for camera position
  ChunkCoord camera_chunk = getChunkCoord(camera_position);

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                               camera_chunk.z + dz};
        if (isInViewFrustum(check_coord, camera_position, view_direction)) {
          needed_chunks.insert(check_coord);
        }
      }
    }
  }

  // Unload chunks that are no longer needed
  std::vector<ChunkCoord> to_unload;
  for (const auto& [coord, chunk] : active_chunks_) {
    if (needed_chunks.find(coord) == needed_chunks.end()) {
      to_unload.push_back(coord);
    }
  }

  for (const auto& coord : to_unload) {
    saveChunk(coord);
    // Remove from active chunks

    auto chunk_to_delete = active_chunks_[coord];
    std::cout << "Current reference count: " << chunk_to_delete.use_count()
              << std::endl;
    active_chunks_.erase(coord);
    std::cout << "Current reference count: " << chunk_to_delete.use_count()
              << std::endl;
  }
  // Clear CUDA cache after removing chunks
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!active_chunks_.empty()) {
    for (const auto& [coord, chunk] : active_chunks_) {
      std::cout << coord.x << " " << coord.y << " " << coord.z << std::endl;
    }
  }

  // // Load newly needed chunks
  // for (const auto& coord : needed_chunks) {
  //   if (active_chunks_.find(coord) == active_chunks_.end()) {
  //     if (!loadChunk(coord)) {
  //       // std::cout << "Chunk visible but can't load: " << coord.x << " "
  //       //           << coord.y << " " << coord.z << " " << std::endl;
  //     }
  //   }
  // }
}

std::filesystem::path GaussianMapper::getChunkFilename(
    const ChunkCoord& coord) {
  // Using 'p' for positive and 'n' for negative prefixes
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));

  return chunk_save_dir_ / (x_str + "_" + y_str + "_" + z_str);
}

void GaussianMapper::saveChunk(const ChunkCoord& coord) {
  auto chunk_it = active_chunks_.find(coord);
  if (chunk_it == active_chunks_.end()) {
    std::cout << "Chunk to save not found in active_chunks_" << std::endl;
    return;
  }

  auto chunk_filename = getChunkFilename(coord);

  // Save the gaussian model
  // Todo implement
  // chunk_it->second->gaussians_->save_checkpoint(chunk_filename.string());
}

bool GaussianMapper::loadChunk(const ChunkCoord& coord) {
  // Called by updateActiveChunks & addPoints
  auto chunk_filename = getChunkFilename(coord);
  if (active_chunks_.find(coord) != active_chunks_.end()) {
    std::cout << "Chunk already in active memory: " << coord.x << " " << coord.y
              << " " << coord.z << std::endl;
    return true;
  }

  if (!chunkExistsOnDisk(coord)) {
    std::cout << "Chunk doesn't exist on disk: " << coord.x << " " << coord.y
              << " " << coord.z << std::endl;
    return false;
  }

  std::cout << "Loading chunk from disk: " << coord.x << " " << coord.y << " "
            << coord.z << std::endl;

  try {
    auto chunk = std::make_shared<Chunk>(model_params_);
    if (!chunk) {
      std::cerr << "Failed to create chunk object" << std::endl;
      return false;
    }

    if (!chunk->gaussians_) {
      std::cerr << "Chunk created but gaussians_ is null" << std::endl;
      return false;
    }

    // Todo implement
    // chunk->gaussians_->load_checkpoint(chunk_filename.string(), opt_params_);
    std::cout << "Setting train mode for just loaded chunk" << std::endl;
    active_chunks_[coord] = chunk;
    std::cout << "Chunk loaded successfully: " << coord.x << " " << coord.y
              << " " << coord.z << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception loading chunk: " << e.what() << std::endl;
    exit(1);
    return false;
  } catch (...) {
    std::cerr << "Unknown exception loading chunk" << std::endl;
    exit(1);
    return false;
  }
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianMapper::groupPointsByChunk(const torch::Tensor& positions) {
  // Called by addPoints
  // Convert positions to chunk coordinates
  torch::Tensor chunk_coords = torch::floor(positions / chunk_size_);

  // Convert to int64 for bit operations
  chunk_coords = chunk_coords.to(torch::kInt64);

  std::cout << "Chunk coordinates range: "
            << torch::min(chunk_coords).item<int64_t>() << " to "
            << torch::max(chunk_coords).item<int64_t>() << std::endl;

  // Offset coordinates to ensure they're positive
  auto x = chunk_coords.index({torch::indexing::Slice(), 0}) + 2048;
  auto y = chunk_coords.index({torch::indexing::Slice(), 1}) + 2048;
  auto z = chunk_coords.index({torch::indexing::Slice(), 2}) + 2048;

  // Use 13 bits per dimension (8192 range per dimension)
  auto x_shifted =
      torch::bitwise_left_shift(x, 26);  // 26 bits for first component
  auto y_shifted =
      torch::bitwise_left_shift(y, 13);  // 13 bits for second component
  // z uses remaining 13 bits, no shift needed

  // Combine using bitwise OR
  torch::Tensor chunk_ids =
      torch::bitwise_or(torch::bitwise_or(x_shifted, y_shifted), z);

  // Get unique chunks, inverse indices, and counts
  // Using unique_dim with proper parameters
  auto [unique_chunk_ids, inverse_indices, points_per_chunk] =
      torch::unique_dim(chunk_ids, 0, true, true, true);

  std::cout << "Number of unique chunks: " << unique_chunk_ids.size(0)
            << std::endl;

  // Convert unique chunk IDs back to 3D coordinates
  torch::Tensor unique_coords = torch::zeros(
      {unique_chunk_ids.size(0), 3},
      torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));

  // Extract coordinates using masks
  unique_coords.index({torch::indexing::Slice(), 0}) =
      torch::bitwise_and(torch::bitwise_right_shift(unique_chunk_ids, 26),
                         8191) -
      2048;

  unique_coords.index({torch::indexing::Slice(), 1}) =
      torch::bitwise_and(torch::bitwise_right_shift(unique_chunk_ids, 13),
                         8191) -
      2048;

  unique_coords.index({torch::indexing::Slice(), 2}) =
      torch::bitwise_and(unique_chunk_ids, 8191) - 2048;

  std::cout << "Reconstructed coordinate range: "
            << torch::min(unique_coords).item<int64_t>() << " to "
            << torch::max(unique_coords).item<int64_t>() << std::endl;

  return std::make_tuple(unique_coords, inverse_indices, points_per_chunk);
}

void GaussianMapper::addPoints(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes) {
  // All instances already mutex_render_ write lock
  torch::Tensor points_cuda = points.to(torch::kCUDA);
  torch::Tensor colors_cuda = colors.to(torch::kCUDA);

  // Todo: No logic for allocating keyframes here yet, currently all for all
  // chunks

  // Group points by chunk
  auto [unique_chunks, inverse_indices, points_per_chunk] =
      groupPointsByChunk(points_cuda);

  // Process each unique chunk
  for (int64_t i = 0; i < unique_chunks.size(0); i++) {
    // Get chunk coordinate
    ChunkCoord coord{unique_chunks[i][0].item<int64_t>(),
                     unique_chunks[i][1].item<int64_t>(),
                     unique_chunks[i][2].item<int64_t>()};

    std::cout << "Adding point for chunk: " << coord.x << " " << coord.y << " "
              << coord.z << " " << std::endl;

    // Create mask for points in this chunk
    torch::Tensor chunk_mask = (inverse_indices == i);

    // Extract points and features for this chunk
    torch::Tensor chunk_points = points_cuda.index({chunk_mask});
    torch::Tensor chunk_colors = colors_cuda.index({chunk_mask});
    std::cout << "Num points in this chunk: " << chunk_points.sizes()[0]
              << std::endl;

    auto [filtered_points, filtered_colors] =
        filterPointsByDepth(chunk_points, chunk_colors, keyframes);

    std::cout << "Num points after depth filter: " << chunk_points.sizes()[0]
              << std::endl;

    int min_new_points_threshold = 10;
    if (chunk_points.sizes()[0] < min_new_points_threshold) {
      std::cout << "Too little points, skipping" << std::endl;
      continue;
    }

    // Initialize or get existing chunk
    if (active_chunks_.find(coord) == active_chunks_.end()) {
      // Chunk not in active chunks. This will either load existing chunk from
      // disk or create new chunk at the coords
      std::cout << "Chunk " << coord.x << " " << coord.y << " " << coord.z
                << " not found in memory." << std::endl;
      if (!loadChunk(coord)) {
        // Initialize new empty chunk
        // For new chunks, we need the full setup

        auto chunk = std::make_shared<Chunk>(model_params_);
        active_chunks_[coord] = chunk;

        auto chunk_gaussians = chunk->gaussians_;

        // Prepare for training
        {
          scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
          chunk_gaussians->createFromPcd(filtered_points, filtered_colors,
                                         scene_->cameras_extent_);
          std::unique_lock<std::mutex> lock(mutex_settings_);
          gaussians_->trainingSetup(opt_params_);
        }

      } else {
        // Chunk sucessfully loaded
        auto& chunk = active_chunks_[coord];
        auto& chunk_gaussians = chunk->gaussians_;
        if (!chunk || !chunk_gaussians) {
          throw "Chunks/Gaussians not correctly loaded";
        }
        chunk_gaussians->increasePcd(filtered_points, filtered_colors,
                                     getIteration());
      }
    } else {
      // Chunk already in memory
      std::cout << "Chunk " << coord.x << " " << coord.y << " " << coord.z
                << " found in memory, adding points to "
                   "model"
                << std::endl;
      auto& chunk_gaussians = active_chunks_[coord]->gaussians_;
      if (!chunk_gaussians) {
        throw "Gaussians not correctly loaded";
      }
      chunk_gaussians->increasePcd(filtered_points, filtered_colors,
                                   getIteration());
    }
  }

  std::cout << "End of addPoints function reached" << std::endl;
}

bool GaussianMapper::chunkExistsOnDisk(const ChunkCoord& coord) {
  // Check if we've already determined if this chunk exists
  auto it = chunk_exists_cache_.find(coord);
  if (it != chunk_exists_cache_.end()) {
    return it->second;
  }

  // Check for existence on disk
  auto chunk_filename = getChunkFilename(coord);
  bool exists = std::filesystem::exists(chunk_filename);

  // Cache the result
  chunk_exists_cache_[coord] = exists;
  return exists;
}

// void GaussianMapper::update_render_chunks() {
//   std::unique_lock<std::shared_mutex> lock(mutex_render_);
//   render_chunks_.clear();

//   // Create deep copies of each active chunk using the clone method
//   for (const auto& [coord, chunk_ptr] : active_chunks_) {
//     render_chunks_[coord] = chunk_ptr->clone();
//     render_chunks_[coord]->setEvalMode();
//   }
// }