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

#include <torch/csrc/cuda/memory_snapshot.h>

#include <fstream>
#include <sstream>

#include "include/debugging_utils.h"
#include "include/gaussian_rasterizer.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"
#include "include/profiling.h"
#include "include/render_flythrough.h"
#include "include/trajectory_viewer.h"

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

// Main GaussianMapper initialiation. Used both by examples and ROS wrapper.
// ORB-SLAM ptr is optional e.g. omited in external mode
GaussianMapper::GaussianMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                               std::filesystem::path gaussian_config_file_path,
                               std::filesystem::path result_dir,
                               int seed,
                               torch::DeviceType device_type,
                               ORB_SLAM3::System::eSensor sensor_type,
                               const string& orb_settings_path)
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
  chunk_save_dir_ = result_dir_ / "chunks";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)
  keyframe_save_dir_ = result_dir_ / "keyframes";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

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

  // Initialize Gaussian model
  gaussians_ = std::make_shared<GaussianModel>(
      model_params_, chunk_save_dir_.string(), chunk_size_);

  // Initialize scene
  scene_ = std::make_shared<GaussianScene>(model_params_);

  // keyframe_queue_ = std::make_shared<KeyframeQueue>(
  //     scene_, gaussians_, chunk_size_, &kfs_loss_, &kfs_used_times_);
  keyframe_queue_ = std::make_shared<KeyframeQueue>(scene_, 50.0, &kfs_loss_,
                                                    &kfs_used_times_);

  // Initialize Laplacian of Gaussian kernel
  initializeLaplacianOfGaussianKernel();

  ORB_SLAM3::Settings* orb_settings;

  std::vector<ORB_SLAM3::GeometricCamera*> vpCameras;
  cv::Size SLAM_im_size;
  UndistortParams undistort_params;
  if (pSLAM) {
    // Cameras
    // TODO: not only monocular
    orb_settings = pSLAM->getSettings();
    SLAM_im_size = orb_settings->newImSize();
    undistort_params =
        UndistortParams(SLAM_im_size, orb_settings->camera1DistortionCoef());

    vpCameras = pSLAM->getAtlas()->GetAllCameras();
    std::cout << "Num. of Camera is " << vpCameras.size() << std::endl;
  } else {
    // Check settings file
    cv::FileStorage fsSettings(orb_settings_path.c_str(),
                               cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      cerr << "Failed to open settings file at: " << orb_settings_path << endl;
      exit(-1);
    }

    cv::FileNode node = fsSettings["File.version"];
    if (!node.empty() && node.isString() && node.string() == "1.0") {
      orb_settings = new ORB_SLAM3::Settings(orb_settings_path, sensor_type);
    }

    SLAM_im_size = orb_settings->newImSize();
    undistort_params =
        UndistortParams(SLAM_im_size, orb_settings->camera1DistortionCoef());

    vpCameras.push_back(orb_settings->camera1());
    vpCameras.push_back(orb_settings->camera2());
  }

  if (pSLAM) {
    sensor_type = pSLAM->getSensorType();
  }

  switch (sensor_type) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR: {
      this->sensor_type_ = MONOCULAR;
    } break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO: {
      this->sensor_type_ = STEREO;
      this->stereo_baseline_length_ = orb_settings->b();
      this->stereo_Q_ = orb_settings->Q().clone();
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

  float mvs_inverse_depth_range;
  if (sensor_type_ == STEREO) {
    initializeStereoDepthEstimator();
    mvs_inverse_depth_range = 0.01f;
  } else if (sensor_type_ == RGBD) {
    mvs_inverse_depth_range = 0.05f;
  } else if (sensor_type_ == MONOCULAR) {
    initializeMonocularDepthEstimator();
    mvs_inverse_depth_range = 0.2f;
  }

  int num_prev_keyframes = 6;
  int num_depth_candidates = 16;

  guided_mvs_ = std::make_unique<GuidedMVS>(
      num_prev_keyframes, num_depth_candidates, mvs_inverse_depth_range);

  feat_extractor_ = std::make_unique<XFeat::XFDetector>(4096, 0.05, true);

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
        std::cout << "For level: " << l
                  << " width: " << camera.gaus_pyramid_width_[l]
                  << " and height: " << camera.gaus_pyramid_height_[l]
                  << std::endl;
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

// Initialization for viewer example (also used for evaluation). Don't need to
// set up training infrastructure
GaussianMapper::GaussianMapper(std::filesystem::path gaussian_config_file_path,
                               std::filesystem::path result_dir,
                               int seed,
                               torch::DeviceType device_type)
    : initial_mapped_(false),
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
  chunk_save_dir_ = result_dir_ / "chunks";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)
  keyframe_save_dir_ = result_dir_ / "keyframes";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

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

  // Initialize Laplacian of Gaussian kernel
  initializeLaplacianOfGaussianKernel();

  loadScene(result_dir);

  // Initialize keyframe queue after gaussians_ is loaded
  // keyframe_queue_ = std::make_shared<KeyframeQueue>(
  //     scene_, gaussians_, chunk_size_, &kfs_loss_, &kfs_used_times_);
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
  model_params_.white_background_ =
      (settings_file["Model.white_background"].operator int()) != 0;
  model_params_.enable_lod_ =
      (settings_file["Model.enable_lod"].operator int()) != 0;
  model_params_.lod_distance_multiplier_ =
      (settings_file["Model.lod_distance_multiplier"].operator int());
  model_params_.max_gaussians_in_memory_ =
      settings_file["Model.max_gaussians_in_memory"].operator int();
  init_proba_scaler_ =
      settings_file["Model.init_proba_scaler"].operator float();
  downsample_for_sampling_ =
      (settings_file["Model.downsample_for_sampling"].operator int()) != 0;

  // Pipeline Parameters
  z_near_ = settings_file["Camera.z_near"].operator float();
  z_far_ = settings_file["Camera.z_far"].operator float();

  min_depth_ = settings_file["Mapper.min_depth_"].operator float();
  max_depth_ = settings_file["Mapper.max_depth_"].operator float();

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
  keyframe_selection_strategy_ =
      settings_file["Mapper.keyframe_selection_strategy"].operator float();

  min_keyframe_translation_ =
      settings_file["External.min_keyframe_translation"].operator float();
  min_keyframe_rotation_ =
      settings_file["External.min_keyframe_rotation"].operator float();
  min_keyframe_time_ =
      settings_file["External.min_keyframe_time"].operator float();

  pipe_params_.convert_SHs_ =
      (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
  pipe_params_.compute_cov3D_ =
      (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;
  num_gaus_pyramid_sub_levels_ =
      settings_file["GausPyramid.num_levels"].operator int();
  int sub_level_times_of_use =
      settings_file["GausPyramid.level_times_of_use"].operator int();
  kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
  kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    kf_gaus_pyramid_factors_[l] = std::pow(0.5f, l);
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
  render_fly_through_ =
      (settings_file["Record.render_fly_through"].operator int()) != 0;
  render_fly_through_speed_ =
      settings_file["Record.render_fly_through_speed"].operator float();

  // Optimization Parameters
  opt_params_.iterations_ =
      settings_file["Optimization.max_num_iterations"].operator int();
  opt_params_.position_lr_init_ =
      settings_file["Optimization.position_lr_init"].operator float();
  opt_params_.position_lr_decay_ =
      settings_file["Optimization.position_lr_decay"].operator float();
  opt_params_.feature_lr_ =
      settings_file["Optimization.feature_lr"].operator float();
  opt_params_.opacity_lr_ =
      settings_file["Optimization.opacity_lr"].operator float();
  opt_params_.scaling_lr_ =
      settings_file["Optimization.scaling_lr"].operator float();
  opt_params_.rotation_lr_ =
      settings_file["Optimization.rotation_lr"].operator float();
  opt_params_.pose_lr_ = settings_file["Optimization.pose_lr"].operator float();
  opt_params_.exposure_lr_ =
      settings_file["Optimization.exposure_lr"].operator float();
  opt_params_.depth_scale_bias_lr_ =
      settings_file["Optimization.depth_scale_bias_lr"].operator float();
  opt_params_.smooth_l1_ =
      (settings_file["Optimization.smooth_l1"].operator int()) != 0;
  opt_params_.opacity_reg_ =
      settings_file["Optimization.opacity_reg"].operator float();

  opt_params_.lambda_dssim_ =
      settings_file["Optimization.lambda_dssim"].operator float();
  opt_params_.lambda_depth_ =
      settings_file["Optimization.lambda_depth"].operator float();
  opt_params_.auto_distribute_ =
      settings_file["Optimization.auto_distribute"].operator int();
  exposure_optimization_ =
      settings_file["Optimization.exposure_optimization"].operator int();

  // Viewer Parameters
  rendered_image_viewer_scale_ =
      settings_file["GaussianViewer.image_scale"].operator float();
  rendered_image_viewer_scale_main_ =
      settings_file["GaussianViewer.image_scale_main"].operator float();

  chunk_size_ = settings_file["Chunking.chunk_size"].operator float();
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

  if (render_fly_through_) {
    std::cout << "[MAPPER DEBUG] Rendering fly-through video" << std::endl;
    auto video_dir = result_dir_ / "flythrough";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
    renderFlyThroughVideo(video_dir / "output_video", 2452, 740, 30,
                          render_fly_through_speed_, 0.8f, 2);
  }

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
            gaussians_->cullVisibleGaussians(pkf, false, false);
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

void GaussianMapper::generateKfidRandomShuffle() {
  if (scene_->keyframes().empty()) return;

  std::size_t nkfs = scene_->keyframes().size();
  kfid_shuffle_.resize(nkfs);
  std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
  std::mt19937 g(rd_());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
}

// std::shared_ptr<GaussianKeyframe>
// GaussianMapper::useOneRandomSlidingWindowKeyframe() {
//   return keyframe_queue_->getNextKeyframe();
// }

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
    visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf, false);

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

          visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf, false);

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

std::tuple<cv::Mat, cv::Mat> GaussianMapper::renderFromPose(
    const Sophus::SE3f& Tcw,
    const int width,
    const int height,
    const bool main_vision) {
  torch::NoGradGuard no_grad;
  if (!initial_mapped_ || getIteration() <= 0) {
    cv::Mat empty_rgb(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    cv::Mat empty_depth(height, width, CV_32FC1, cv::Scalar(0.0f));
    return std::make_tuple(empty_rgb, empty_depth);
  }
  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
  pkf->zfar_ = z_far_ * scene_->cameras_extent_;
  pkf->znear_ = z_near_ * scene_->cameras_extent_;
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
  // Eigen::Vector3f cam_position;
  // for (int i = 0; i < 3; ++i) {
  //   cam_position[i] = pkf->camera_center_[i].item<float>();
  // }
  // auto cam_chunk_coord = chunk_manager_->getChunkCoord(cam_position);
  // std::cout << "Cam chunk: " << cam_chunk_coord.x << " " <<
  // cam_chunk_coord.y
  //           << " " << cam_chunk_coord.z << std::endl;

  // std::cout << "Tcw matrix:\n" << Tcw.matrix() << std::endl;

  std::unique_lock lock_render(mutex_render_);

  // std::cout << width << " " << height << std::endl;

  // Get visible chunks using ChunkManager instead of updateActiveChunks
  torch::Tensor visible_gaussian_mask =
      gaussians_->cullVisibleGaussians(pkf, true);

  // Render
  torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
  auto render_pkg = GaussianRenderer::render(
      gaussians_, visible_gaussian_mask, pkf, height, width, pipe_params_,
      background_, override_color_, 1.0f, false, pkf->FoVx_, pkf->FoVy_,
      view_matrix, pkf->projection_matrix_);

  // Return rendered image and depth
  cv::Mat rendered_rgb =
      tensor_utils::torchTensor2CvMat_Float32(std::get<1>(render_pkg));

  torch::Tensor rendered_depth_tensor = std::get<0>(render_pkg);
  // Convert tensor to OpenCV Mat
  cv::Mat rendered_depth;
  if (rendered_depth_tensor.dim() == 3) {
    // If tensor is [H, W, 1] or [1, H, W], squeeze to [H, W]
    rendered_depth_tensor = rendered_depth_tensor.squeeze();
  }

  // Ensure tensor is contiguous and on CPU
  rendered_depth_tensor = rendered_depth_tensor.contiguous().cpu();

  // Convert to OpenCV Mat
  rendered_depth =
      cv::Mat(rendered_depth_tensor.size(0), rendered_depth_tensor.size(1),
              CV_32F, rendered_depth_tensor.data_ptr<float>())
          .clone();
  return std::make_tuple(rendered_rgb, rendered_depth);
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

  bool had_to_load = false;
  if (!pkf->loaded_) {
    pkf->loadDataFromDisk();
    had_to_load = true;
  }

  torch::Tensor visible_gaussian_mask =
      gaussians_->cullVisibleGaussians(pkf, true);

  torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
  auto render_pkg = GaussianRenderer::render(
      gaussians_, visible_gaussian_mask, pkf, pkf->image_height_,
      pkf->image_width_, pipe_params_, background_, override_color_, 1.0f,
      false, pkf->FoVx_, pkf->FoVy_, view_matrix, pkf->projection_matrix_);

  // Chunks automatically released by ChunkOptimizationGuard destructor
  auto rendered_image = std::get<1>(render_pkg);
  // torch::cuda::synchronize();
  auto end_timing = std::chrono::steady_clock::now();
  auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_timing - start_timing)
                            .count();
  render_time = 1e-6 * render_time_ns;
  auto gt_image = pkf->gaus_pyramid_original_image_[0];

  dssim = loss_utils::fast_ssim(rendered_image, gt_image).item().toFloat();
  psnr = loss_utils::psnr(rendered_image, gt_image).item().toFloat();
  psnr_gs = loss_utils::psnr_gaussian_splatting(rendered_image, gt_image)
                .item()
                .toFloat();

  recordKeyframeRendered(rendered_image, gt_image, pkf->fid_, result_img_dir,
                         result_gt_dir, result_loss_dir, name_suffix);

  if (had_to_load) {
    pkf->saveDataToDisk();
  }
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
    Eigen::Matrix3f R = pkf->getRotationMatrixf();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = pkf->getTranslationf();
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

    if (pkf->intr_.size() >= 4) {
      json_kf["cx"] = pkf->intr_[2];  // cx is at index 2
      json_kf["cy"] = pkf->intr_[3];  // cy is at index 3
    }

    auto& keyframe_cam = scene_->getCamera(pkf->camera_id_);

    if (!keyframe_cam.dist_coeff_.empty() &&
        keyframe_cam.dist_coeff_.total() >= 5) {
      json_kf["k1"] = keyframe_cam.dist_coeff_.at<float>(0);
      json_kf["k2"] = keyframe_cam.dist_coeff_.at<float>(1);
      json_kf["p1"] = keyframe_cam.dist_coeff_.at<float>(2);
      json_kf["p2"] = keyframe_cam.dist_coeff_.at<float>(3);
      json_kf["k3"] = keyframe_cam.dist_coeff_.at<float>(4);
    } else {
      // For rectified images, set distortion to zero
      json_kf["k1"] = 0.0f;
      json_kf["k2"] = 0.0f;
      json_kf["p1"] = 0.0f;
      json_kf["p2"] = 0.0f;
      json_kf["k3"] = 0.0f;
    }

    json_root[i] = Json::Value(json_kf);
    ++i;
  }

  writer->write(json_root, &out_stream);
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir,
                                            std::string name_suffix) {
  return;
  //   CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  //   std::filesystem::path result_path =
  //       result_dir / ("keyframe_used_times" + name_suffix + ".txt");
  //   std::ofstream out_stream;
  //   out_stream.open(result_path, std::ios::app);
  //   if (!out_stream.is_open())
  //     throw std::runtime_error("Cannot open json at " +
  //     result_path.string());

  //   out_stream << "##[Gaussian Mapper]Iteration " << getIteration()
  //              << " keyframe id, used times, remaining times:\n";
  //   for (const auto& used_times_it : keyframe_queue_->getKfsUsedTimes()) {
  //     out_stream
  //         << used_times_it.first << " " << used_times_it.second << " "
  //         <<
  //         scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
  //         << "\n";
  //   }
  //   out_stream << "##=========================================" <<
  //   std::endl;

  //   out_stream.close();
}

void GaussianMapper::writeTrainingMetricsCSV(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  std::filesystem::path result_path = result_dir / "training_metrics.csv";
  std::ofstream out_stream;
  out_stream.open(result_path);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open CSV at " + result_path.string());

  // Write CSV header
  out_stream << "iteration,elapsed_time_seconds,active_gaussian_count,total_"
                "gaussian_count,reserved_memory_"
                "mb,allocated_memory_mb,ram_usage_mb,queue_keyframes\n";

  // Write data
  for (const auto& metrics : training_metrics_) {
    out_stream << metrics.iteration << "," << metrics.elapsed_time_seconds
               << "," << metrics.active_gaussian_count << ","
               << metrics.total_gaussian_count << ","
               << metrics.reserved_memory_mb << ","
               << metrics.allocated_memory_mb << "," << metrics.ram_usage_mb
               << "," << metrics.queue_keyframes << "\n";
  }

  out_stream.close();
  std::cout << "[GaussianMapper] Training metrics saved to " << result_path
            << std::endl;
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
float GaussianMapper::lambdaDssim() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_dssim_;
}
float GaussianMapper::lambdaDepth() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_depth_;
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
void GaussianMapper::setLambdaDssim(const float lambda_dssim) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.lambda_dssim_ = lambda_dssim;
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

VariableParameters GaussianMapper::getVaribleParameters() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  VariableParameters params;
  params.position_lr_init = opt_params_.position_lr_init_;
  params.feature_lr = opt_params_.feature_lr_;
  params.opacity_lr = opt_params_.opacity_lr_;
  params.scaling_lr = opt_params_.scaling_lr_;
  params.rotation_lr = opt_params_.rotation_lr_;
  params.lambda_dssim = opt_params_.lambda_dssim_;
  params.new_kf_times_of_use = new_keyframe_times_of_use_;
  params.stable_num_iter_existence = stable_num_iter_existence_;
  params.keep_training = keep_training_;
  return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters& params) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.position_lr_init_ = params.position_lr_init;
  opt_params_.feature_lr_ = params.feature_lr;
  opt_params_.opacity_lr_ = params.opacity_lr;
  opt_params_.scaling_lr_ = params.scaling_lr;
  opt_params_.rotation_lr_ = params.rotation_lr;
  opt_params_.lambda_dssim_ = params.lambda_dssim;
  new_keyframe_times_of_use_ = params.new_kf_times_of_use;
  stable_num_iter_existence_ = params.stable_num_iter_existence;
  keep_training_ = params.keep_training;
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

/**
 * Generates a smooth fly-through video along keyframe path with constant
 * speed
 *
 * @param output_path Directory where frames and video will be saved
 * @param width Width of the output video
 * @param height Height of the output video
 * @param fps Frames per second
 * @param speed Speed of the camera in units per second
 * @param smoothness_factor Controls path smoothness (0.0-1.0, higher =
 * smoother but deviates more from keyframes)
 * @param keyframe_subsample Use only every Nth keyframe (1 = use all, 2 = use
 * every other, etc.)
 */
void GaussianMapper::renderFlyThroughVideo(const std::string& output_path,
                                           int width,
                                           int height,
                                           int fps,
                                           float speed,
                                           float smoothness_factor,
                                           int keyframe_subsample) {
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_path);

  // 1. Get all keyframes sorted by fid
  auto keyframes_map = scene_->getAllKeyframes();
  std::vector<std::size_t> keyframe_ids;
  for (const auto& [fid, kf] : keyframes_map) {
    keyframe_ids.push_back(fid);
  }
  std::sort(keyframe_ids.begin(), keyframe_ids.end());

  // Subsample keyframes if requested (for smoother overall path)
  if (keyframe_subsample > 1 && keyframe_ids.size() > keyframe_subsample * 2) {
    std::vector<std::size_t> subsampled_ids;
    // Always keep first and last keyframe
    subsampled_ids.push_back(keyframe_ids.front());

    for (size_t i = keyframe_subsample; i < keyframe_ids.size() - 1;
         i += keyframe_subsample) {
      subsampled_ids.push_back(keyframe_ids[i]);
    }

    subsampled_ids.push_back(keyframe_ids.back());

    std::cout << "Reduced keyframes from " << keyframe_ids.size() << " to "
              << subsampled_ids.size() << " for smoother path" << std::endl;

    keyframe_ids = subsampled_ids;
  }

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
  // Using 20 points per segment for smoother interpolation
  std::vector<Eigen::Vector3d> path_points =
      createSmoothPath(keyframe_positions, 20);

  // Apply additional smoothing if requested
  if (smoothness_factor > 0.0f) {
    smoothPath(path_points, smoothness_factor);
  }

  // 4. Calculate total path length using arc lengths
  std::vector<double> arc_lengths = computeArcLengths(path_points);
  double total_path_length = arc_lengths.back();

  // 5. Calculate duration based on path length and speed
  float duration_seconds = total_path_length / speed;
  int total_frames = static_cast<int>(duration_seconds * fps);

  // Ensure we have at least 2 frames
  total_frames = std::max(2, total_frames);

  std::cout << "Path length: " << total_path_length << " units" << std::endl;
  std::cout << "Speed: " << speed << " units/second" << std::endl;
  std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;
  std::cout << "Total frames: " << total_frames << std::endl;

  // 6. Sample the path at equal distances to ensure constant speed
  std::vector<Eigen::Vector3d> sampled_positions;
  std::vector<Eigen::Quaterniond> sampled_orientations;
  samplePathConstantSpeed(path_points, keyframe_positions,
                          keyframe_orientations,
                          static_cast<int>(duration_seconds * fps),
                          sampled_positions, sampled_orientations);

  // 7. Render each frame
  std::vector<std::string> frame_paths;

  for (int i = 0; i < total_frames; i++) {
    // Create world-to-camera transform
    Sophus::SE3d Twc(sampled_orientations[i], sampled_positions[i]);
    Sophus::SE3f Tcw = Twc.inverse().cast<float>();

    // Render frame
    auto render_result = renderFromPose(Tcw, width, height, true);
    cv::Mat frame = std::get<0>(render_result);  // Get RGB image, ignore depth

    // Convert if needed (assuming renderFromPose returns float image)
    cv::Mat output_frame;
    frame.convertTo(output_frame, CV_8UC3, 255.0);

    cv::cvtColor(output_frame, output_frame, cv::COLOR_BGR2RGB);

    // Save frame
    std::string frame_path =
        output_path + "/frame_" + std::to_string(i + 1) + ".png";
    cv::imwrite(frame_path, output_frame);
    frame_paths.push_back(frame_path);

    {
      std::cout << "Rendered frame " << i + 1 << "/" << total_frames
                << std::endl;
    }
  }

  // 8. Combine frames into video using ffmpeg
  std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) + " -i " +
                    output_path + "/frame_%d.png" +
                    " -c:v libx264 -crf 18 -pix_fmt yuv420p " + output_path +
                    "/flythrough.mp4";

  std::cout << "Creating video with command: " << cmd << std::endl;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Failed to create video using ffmpeg. Check if ffmpeg is "
                 "installed."
              << std::endl;
  } else {
    std::cout << "Video created successfully at " << output_path
              << "/flythrough.mp4" << std::endl;

    // 9. Delete all frame files
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
 * Generates a 3D exploration video that showcases the depth and
 * dimensionality of the scene by adding camera movements that deviate from
 * the main path
 *
 * @param output_path Directory where frames and video will be saved
 * @param width Width of the output video
 * @param height Height of the output video
 * @param fps Frames per second
 * @param duration_seconds Total duration of the video
 * @param deviation_scale Scale factor for path deviation (default: 0.15)
 * @param look_around Enable camera rotation to look around (default: true)
 */
void GaussianMapper::render3DExplorationVideo(const std::string& output_path,
                                              int width,
                                              int height,
                                              int fps,
                                              float duration_seconds,
                                              float deviation_scale,
                                              bool look_around) {
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
    std::cerr << "Need at least 2 keyframes to create a 3D exploration video"
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

  // 4. Compute arc lengths and camera orientations along the base path
  std::vector<double> arc_lengths = computeArcLengths(path_points);
  std::vector<double> keyframe_parameters;
  mapKeyframesToPath(keyframe_positions, path_points, arc_lengths,
                     keyframe_parameters);

  // 5. Sample path and generate interesting camera movements
  int total_frames = static_cast<int>(duration_seconds * fps);
  double total_length = arc_lengths.back();
  std::vector<Eigen::Vector3d> exploration_positions;
  std::vector<Eigen::Quaterniond> exploration_orientations;

  // Compute local coordinate frames along the path (for controlled deviation)
  std::vector<Eigen::Matrix3d> path_frames = computePathFrames(path_points);

  // Calculate scene scale to properly scale deviations
  double scene_scale = calculateSceneScale(keyframe_positions);
  double deviation_amount = scene_scale * deviation_scale;

  for (int i = 0; i < total_frames; i++) {
    double t = static_cast<double>(i) / (total_frames - 1);  // Normalized [0,1]
    double target_length = total_length * t;

    // Get base position and orientation at this path location
    Eigen::Vector3d base_position =
        samplePositionAtArcLength(path_points, arc_lengths, target_length);
    double path_param = target_length / total_length;
    Eigen::Quaterniond base_orientation = interpolateOrientation(
        path_param, keyframe_parameters, keyframe_orientations);

    // Find closest point on path for local frame
    size_t closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t j = 0; j < path_points.size(); j++) {
      double dist = (base_position - path_points[j]).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_idx = j;
      }
    }

    // Apply sinusoidal deviations in local frame coordinates
    Eigen::Matrix3d local_frame = path_frames[closest_idx];
    Eigen::Vector3d deviation = Eigen::Vector3d::Zero();

    // Vertical deviation (up/down)
    deviation +=
        local_frame.col(1) * sin(t * 2.0 * M_PI * 2.5) * deviation_amount;

    // Horizontal deviation (left/right)
    deviation += local_frame.col(0) * sin(t * 2.0 * M_PI * 1.7 + 1.0) *
                 deviation_amount * 5.0;

    // Forward/backward small deviation for dynamic feel
    // deviation += local_frame.col(2) * sin(t * 2.0 * M_PI * 3.2 + 0.5) *
    //              deviation_amount * 0.3;

    // Apply deviation to base position
    Eigen::Vector3d explorer_position = base_position + deviation;

    // Create a modified orientation that occasionally looks toward
    // interesting features
    Eigen::Quaterniond explorer_orientation = base_orientation;

    if (look_around) {
      // Look slightly up/down and left/right based on a different frequency
      double look_factor = 0.15;  // How much to look around (in radians)

      // Look up/down
      Eigen::Vector3d look_up_axis = local_frame.col(0);
      double look_up_angle = sin(t * 2.0 * M_PI * 1.2 + 0.8) * look_factor;

      // Look left/right
      Eigen::Vector3d look_side_axis = local_frame.col(1);
      double look_side_angle = sin(t * 2.0 * M_PI * 0.9 + 2.1) * look_factor;

      // Apply these rotations to base orientation
      Eigen::Quaterniond look_up =
          Eigen::Quaterniond(Eigen::AngleAxisd(look_up_angle, look_up_axis));
      Eigen::Quaterniond look_side = Eigen::Quaterniond(
          Eigen::AngleAxisd(look_side_angle, look_side_axis));

      explorer_orientation = base_orientation * look_up * look_side;
      explorer_orientation.normalize();
    }

    exploration_positions.push_back(explorer_position);
    exploration_orientations.push_back(explorer_orientation);
  }

  // 6. Render each frame with the exploration camera path
  std::vector<std::string> frame_paths;
  for (int i = 0; i < total_frames; i++) {
    // Create world-to-camera transform
    Sophus::SE3d Twc(exploration_orientations[i], exploration_positions[i]);
    Sophus::SE3f Tcw = Twc.inverse().cast<float>();

    // Render frame
    auto render_result = renderFromPose(Tcw, width, height, true);
    cv::Mat frame = std::get<0>(render_result);  // Get RGB image, ignore depth

    // Convert if needed (assuming renderFromPose returns float image)
    cv::Mat output_frame;
    frame.convertTo(output_frame, CV_8UC3, 255.0);

    cv::cvtColor(output_frame, output_frame, cv::COLOR_BGR2RGB);

    // Save frame
    std::string frame_path =
        output_path + "/frame_" + std::to_string(i + 1) + ".png";
    cv::imwrite(frame_path, output_frame);
    frame_paths.push_back(frame_path);

    {
      std::cout << "Rendered frame " << i + 1 << "/" << total_frames
                << std::endl;
    }
  }

  // 7. Combine frames into video using ffmpeg
  std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) + " -i " +
                    output_path + "/frame_%d.png" +
                    " -c:v libx264 -crf 18 -pix_fmt yuv420p " + output_path +
                    "/3d_exploration.mp4";

  std::cout << "Creating video with command: " << cmd << std::endl;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Failed to create video using ffmpeg. Check if ffmpeg is "
                 "installed."
              << std::endl;
  } else {
    std::cout << "Video created successfully at " << output_path
              << "/3d_exploration.mp4" << std::endl;

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

bool GaussianMapper::saveScene(std::filesystem::path scene_dir) {
  std::cout << "saveScene called" << std::endl;
  // Create directory if it doesn't exist
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(scene_dir);

  // Save camera params and scene metadata
  keyframesToJson(scene_dir);

  // Need to save chunks before saving manifest since we need to update chunks
  // in memory map
  gaussians_->saveAllChunks();

  // Save a manifest of all chunks on disk
  saveChunkManifest(scene_dir);

  std::filesystem::path cameras_exent_path = scene_dir / "cameras_extent.json";

  Json::Value json_root;
  Json::StreamWriterBuilder builder;
  const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  json_root[0] = Json::Value(scene_->cameras_extent_);

  // Write to file
  std::ofstream out_stream(cameras_exent_path);
  if (!out_stream.is_open()) {
    throw std::runtime_error("Cannot open cameras_extent file at " +
                             cameras_exent_path.string());
  }
  writer->write(json_root, &out_stream);
  out_stream.close();

  // Save config used to train the model
  try {
    std::filesystem::copy_file(
        config_file_path_, scene_dir / "gaussian_mapper_cfg.yaml",
        std::filesystem::copy_options::overwrite_existing);
    std::cout << "Config saved successfully" << std::endl;
  } catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  std::cout << "Copying chunk data to save dir" << std::endl;
  // Copy chunks over to scene dir
  std::filesystem::path scene_chunk_dir = scene_dir / "chunks";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(scene_chunk_dir);
  copyFolder(chunk_save_dir_, scene_dir / "chunks");

  std::cout << "Done copying chunk data to save dir" << std::endl;

  std::cout << "Scene saved to " << scene_dir << std::endl;
  return true;
}

// Implementation for loadScene in gaussian_mapper.cpp
bool GaussianMapper::loadScene(std::filesystem::path scene_dir,
                               std::filesystem::path optional_camera_path) {
  std::filesystem::path cameras_exent_path = scene_dir / "cameras_extent.json";

  if (!std::filesystem::exists(cameras_exent_path)) {
    throw std::runtime_error("cameras_extent JSON not found at " +
                             cameras_exent_path.string());
  }

  // Parse the cameras_extent JSON file
  std::ifstream file(cameras_exent_path);
  Json::Value root;
  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  if (!Json::parseFromStream(builder, file, &root, &errs)) {
    throw std::runtime_error("Error parsing cameras_extent JSON: " + errs);
  }
  scene_->cameras_extent_ = root[0].asFloat();
  std::cout << "Loaded cameras extent: " << scene_->cameras_extent_
            << std::endl;

  gaussians_ =
      std::make_shared<GaussianModel>(model_params_, chunk_save_dir_.string(),
                                      chunk_size_ * scene_->cameras_extent_);

  if (!std::filesystem::exists(scene_dir)) {
    throw std::runtime_error("Scene directory does not exist: " +
                             scene_dir.string());
  }

  // // Load camera parameters
  loadCamerasFromJson(scene_dir / "cameras.json");

  if (!gaussians_->is_initialized_) {
    gaussians_->initializeEmpty(scene_->cameras_extent_);
    gaussians_->trainingSetup(opt_params_);
    std::cout << "Initialized empty Gaussian model for loading" << std::endl;
  }

  // Load chunk information from the manifest
  loadChunkManifest(scene_dir);

  std::cout << "Loaded " << gaussians_->chunks_on_disk_.size(0)
            << " chunks from manifest" << std::endl;

  // // Load a few chunks for initial visualization if desired
  // if (load_initial_chunks_ && !chunk_coords.empty()) {
  //   int max_to_load =
  //       std::min(static_cast<int>(chunk_coords.size()),
  //       max_initial_chunks_);

  //   for (int i = 0; i < max_to_load; i++) {
  //     chunk_manager_->loadChunk(chunk_coords[i]);
  //   }
  // }

  // Optinal new Camera configs
  if (!optional_camera_path.empty() &&
      std::filesystem::exists(optional_camera_path)) {
    cv::FileStorage camera_file(optional_camera_path.string().c_str(),
                                cv::FileStorage::READ);
    if (!camera_file.isOpened())
      throw std::runtime_error(
          "[Gaussian Mapper]Failed to open settings file at: " +
          optional_camera_path.string());

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
                               optional_camera_path.string());
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

  std::cout << "Scene loaded from " << scene_dir << std::endl;
  return true;
}

void GaussianMapper::saveChunkManifest(std::filesystem::path scene_dir) {
  std::filesystem::path manifest_path = scene_dir / "chunk_manifest.json";
  Json::Value json_root;
  Json::StreamWriterBuilder builder;
  const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  // Save chunks_in_memory_ (std::unordered_set<int64_t>)
  // Json::Value chunks_in_memory_array(Json::arrayValue);
  // for (const auto& chunk_id : gaussians_->chunks_in_memory_) {
  //   chunks_in_memory_array.append(
  //       Json::Value(static_cast<Json::Int64>(chunk_id)));
  // }
  // json_root["chunks_in_memory"] = chunks_in_memory_array;

  // Save chunks_on_disk_ (std::unordered_set<int64_t>)
  Json::Value chunks_on_disk_array(Json::arrayValue);
  Json::Value chunk_gaussian_counts_array(Json::arrayValue);
  auto chunks_cpu = gaussians_->chunks_on_disk_.cpu();
  auto chunk_gaussian_counts_cpu = gaussians_->chunk_gaussian_counts_.cpu();
  auto accessor_id = chunks_cpu.accessor<int64_t, 1>();
  auto accessor_count = chunk_gaussian_counts_cpu.accessor<int64_t, 1>();

  for (int i = 0; i < chunks_cpu.size(0); ++i) {
    int64_t chunk_id = accessor_id[i];
    int64_t count = accessor_count[i];
    chunks_on_disk_array.append(
        Json::Value(static_cast<Json::Int64>(chunk_id)));
    chunk_gaussian_counts_array.append(
        Json::Value(static_cast<Json::Int64>(count)));
  }
  json_root["chunks_on_disk"] = chunks_on_disk_array;
  json_root["chunk_gaussian_counts"] = chunk_gaussian_counts_array;

  // Write to file
  std::ofstream out_stream(manifest_path);
  if (!out_stream.is_open()) {
    throw std::runtime_error("Cannot open manifest file at " +
                             manifest_path.string());
  }
  writer->write(json_root, &out_stream);
  out_stream.close();
}

// Implementation for loadChunkManifest
void GaussianMapper::loadChunkManifest(std::filesystem::path scene_dir) {
  std::filesystem::path manifest_path = scene_dir / "chunk_manifest.json";
  if (!std::filesystem::exists(manifest_path)) {
    std::cerr << "Warning: Chunk manifest not found at " << manifest_path
              << std::endl;
    return;
  }

  // Parse the JSON file
  std::ifstream file(manifest_path);
  Json::Value root;
  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;
  if (!Json::parseFromStream(builder, file, &root, &errs)) {
    throw std::runtime_error("Error parsing chunk manifest: " + errs);
  }

  // Clear existing sets before loading
  gaussians_->chunks_on_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  gaussians_->chunks_loaded_from_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  // // Load chunks_in_memory_ (std::unordered_set<int64_t>)
  // if (root.isMember("chunks_in_memory") &&
  // root["chunks_in_memory"].isArray()) {
  //   const Json::Value& chunks_in_memory_array = root["chunks_in_memory"];
  //   for (const auto& chunk_value : chunks_in_memory_array) {
  //     if (chunk_value.isInt64()) {
  //       gaussians_->chunks_in_memory_.insert(chunk_value.asInt64());
  //     }
  //   }
  // }

  // Load chunks_on_disk_ (std::unordered_set<int64_t>)
  if (root.isMember("chunks_on_disk") && root["chunks_on_disk"].isArray()) {
    const Json::Value& chunks_on_disk_array = root["chunks_on_disk"];

    // Collect into vector first
    std::vector<int64_t> chunks_on_disk_vec;
    chunks_on_disk_vec.reserve(chunks_on_disk_array.size());

    for (const auto& chunk_value : chunks_on_disk_array) {
      if (chunk_value.isInt64()) {
        chunks_on_disk_vec.push_back(chunk_value.asInt64());
      }
    }

    // Convert to tensor
    if (!chunks_on_disk_vec.empty()) {
      gaussians_->chunks_on_disk_ =
          torch::from_blob(chunks_on_disk_vec.data(),
                           {static_cast<int64_t>(chunks_on_disk_vec.size())},
                           torch::TensorOptions().dtype(torch::kInt64))
              .clone()
              .to(gaussians_->device_type_);
    } else {
      gaussians_->chunks_on_disk_ =
          torch::empty({0}, torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(gaussians_->device_type_));
    }
  }

  if (root.isMember("chunk_gaussian_counts") &&
      root["chunk_gaussian_counts"].isArray()) {
    const Json::Value& chunk_gaussian_counts_array =
        root["chunk_gaussian_counts"];

    // Collect into vector first
    std::vector<int64_t> chunk_gaussian_counts_vec;
    chunk_gaussian_counts_vec.reserve(chunk_gaussian_counts_array.size());

    for (const auto& chunk_value : chunk_gaussian_counts_array) {
      if (chunk_value.isInt64()) {
        chunk_gaussian_counts_vec.push_back(chunk_value.asInt64());
      }
    }

    // Convert to tensor
    if (!chunk_gaussian_counts_vec.empty()) {
      gaussians_->chunk_gaussian_counts_ =
          torch::from_blob(
              chunk_gaussian_counts_vec.data(),
              {static_cast<int64_t>(chunk_gaussian_counts_vec.size())},
              torch::TensorOptions().dtype(torch::kInt64))
              .clone()
              .to(gaussians_->device_type_);
    } else {
      gaussians_->chunk_gaussian_counts_ =
          torch::empty({0}, torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(gaussians_->device_type_));
    }
  }
}

void GaussianMapper::loadCamerasFromJson(std::filesystem::path json_path) {
  if (!std::filesystem::exists(json_path)) {
    throw std::runtime_error("Camera JSON not found at " + json_path.string());
  }

  // Parse the JSON file
  std::ifstream file(json_path);
  Json::Value root;
  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  if (!Json::parseFromStream(builder, file, &root, &errs)) {
    throw std::runtime_error("Error parsing camera JSON: " + errs);
  }

  // Clear existing keyframes
  scene_->keyframes().clear();

  // Process each camera entry
  for (const auto& camera_entry : root) {
    // Extract camera ID
    unsigned long fid = camera_entry["id"].asUInt64();

    // Create a new keyframe
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(
        fid, getIteration(), keyframe_save_dir_);

    // Set image dimensions
    pkf->image_width_ = camera_entry["width"].asInt();
    pkf->image_height_ = camera_entry["height"].asInt();

    // Set focal lengths and field of view
    float fx = camera_entry["fx"].asFloat();
    float fy = camera_entry["fy"].asFloat();

    float cx = camera_entry["cx"].asFloat();
    float cy = camera_entry["cy"].asFloat();

    float k1 = camera_entry.get("k1", 0.0f).asFloat();
    float k2 = camera_entry.get("k2", 0.0f).asFloat();
    float p1 = camera_entry.get("p1", 0.0f).asFloat();
    float p2 = camera_entry.get("p2", 0.0f).asFloat();
    float k3 = camera_entry.get("k3", 0.0f).asFloat();

    // Convert focal length to FoV if needed
    pkf->FoVy_ = 2.0f * std::atan(pkf->image_height_ / (2.0f * fy));
    pkf->FoVx_ = 2.0f * std::atan(pkf->image_width_ / (2.0f * fx));

    // Set near and far planes
    pkf->znear_ = z_near_;
    pkf->zfar_ = z_far_;

    // Set position and rotation
    Eigen::Vector3d pos;
    pos.x() = camera_entry["position"][0].asDouble();
    pos.y() = camera_entry["position"][1].asDouble();
    pos.z() = camera_entry["position"][2].asDouble();

    Eigen::Matrix3d rot;
    rot(0, 0) = camera_entry["rotation"][0][0].asDouble();
    rot(0, 1) = camera_entry["rotation"][0][1].asDouble();
    rot(0, 2) = camera_entry["rotation"][0][2].asDouble();
    rot(1, 0) = camera_entry["rotation"][1][0].asDouble();
    rot(1, 1) = camera_entry["rotation"][1][1].asDouble();
    rot(1, 2) = camera_entry["rotation"][1][2].asDouble();
    rot(2, 0) = camera_entry["rotation"][2][0].asDouble();
    rot(2, 1) = camera_entry["rotation"][2][1].asDouble();
    rot(2, 2) = camera_entry["rotation"][2][2].asDouble();

    // Convert rotation matrix to quaternion
    Eigen::Quaterniond quat(rot);

    // Set the pose (Tcw is inverse of position and rotation)
    Sophus::SE3d Twc(quat, pos);
    Sophus::SE3d Tcw = Twc.inverse();
    pkf->setPose(Tcw.unit_quaternion(), Tcw.translation());

    Camera camera;
    camera.camera_id_ = 0;
    camera.width_ = pkf->image_width_;
    camera.height_ = pkf->image_height_;
    camera.setModelId(Camera::CameraModelType::PINHOLE);

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
    if (!viewer_camera_id_set_) {
      viewer_camera_id_ = camera.camera_id_;
      viewer_camera_id_set_ = true;
    }
    this->scene_->addCamera(camera);

    // Set camera parameters
    if (scene_->cameras_.find(viewer_camera_id_) != scene_->cameras_.end()) {
      pkf->setCameraParams(scene_->cameras_.at(viewer_camera_id_));
    } else {
      std::cerr << "Warning: No camera found with ID " << viewer_camera_id_
                << std::endl;
    }

    // Compute transform tensors
    pkf->computeTransformTensors();

    // Add keyframe to scene
    scene_->addKeyframe(pkf);
    pkf->initOptimizer(device_type_, opt_params_.pose_lr_,
                       opt_params_.exposure_lr_,
                       opt_params_.depth_scale_bias_lr_);
    kfid_shuffled_ = false;

    break;
  }
}

void copyFolder(const std::filesystem::path& source,
                const std::filesystem::path& destination) {
  // Create the destination directory if it doesn't exist
  std::filesystem::create_directories(destination);

  // Iterate through the source directory
  for (const auto& entry : std::filesystem::directory_iterator(source)) {
    const auto& path = entry.path();
    const auto destPath = destination / path.filename();

    if (std::filesystem::is_directory(path)) {
      // Recursively copy subdirectories
      copyFolder(path, destPath);
    } else {
      // Copy files
      std::filesystem::copy_file(
          path, destPath, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

void GaussianMapper::saveTotalGaussians(std::string name_suffix) {
  int totalGaussians = gaussians_->countAllGaussians();

  std::filesystem::path result_dir =
      result_dir_ / (std::to_string(getIteration()) + name_suffix);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  std::filesystem::path file_path = result_dir / "gaussianCount.txt";

  std::ofstream outFile(file_path);

  // Check if the file was opened successfully
  if (!outFile) {
    std::cerr << "Error: Could not open the file." << std::endl;
    return;
  }
  outFile << totalGaussians;

  // Close the file
  outFile.close();
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

void GaussianMapper::handleNewFrameExternal(const cv::Mat& rgb_image,
                                            const cv::Mat& depth_or_right_image,
                                            const Sophus::SE3f& pose,
                                            const double timestamp) {
  static int frame_count = 0;
  // std::cout << "External frame #" << frame_count++ << " with timestamp "
  //           << timestamp << " and position " <<
  //           pose.translation().transpose()
  //           << std::endl;
  frame_queue_.push(Frame(rgb_image, depth_or_right_image, pose, timestamp));
}

void GaussianMapper::run_external_poses() {
  std::cout << "[MAPPER DEBUG] GaussianMapper::run_external_poses() started"
            << std::endl;

  std::chrono::steady_clock::time_point training_start =
      std::chrono::steady_clock::now();
  training_start_time_ = training_start;

  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  std::filesystem::remove_all(keyframe_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

  scene_->cameras_extent_ = 1.0f;

  // Process frames until we have enough keyframes
  while (!initial_mapped_ && !isStopped() && !isExternalDataStopped()) {
    auto maybe_frame = frame_queue_.pop(true);
    if (!maybe_frame) continue;

    auto& frame = *maybe_frame;
    handleNewKeyframeFromExternal(frame.rgb_image, frame.depth_image,
                                  frame.pose, frame.timestamp);
    std::cout << "Num keyframes: " << scene_->keyframes().size() << std::endl;

    std::unique_lock<std::mutex> lock_render(mutex_render_);
    std::cout << "Calling training setup!" << std::endl;
    gaussians_->trainingSetup(opt_params_);
    initial_mapped_ = true;
  }

  int SLAM_stop_iter = 0;
  // Start training loop while still processing new frames
  std::cout << "Starting training loop" << std::endl;
  while (!isExternalDataStopped() && !isStopped()) {
    // Process any pending frames
    while (auto maybe_frame = frame_queue_.pop(false)) {
      handleNewKeyframeFromExternal(maybe_frame->rgb_image,
                                    maybe_frame->depth_image, maybe_frame->pose,
                                    maybe_frame->timestamp);
    }

    trainForOneIteration();
    SLAM_stop_iter = getIteration();

    // std::cout << "Loop check: isExternalDataStopped()="
    //           << (isExternalDataStopped() ? "true" : "false")
    //           << ", isStopped()=" << (isStopped() ? "true" : "false")
    //           << std::endl;
  }

  std::cout << "Training finished" << std::endl;

  frame_queue_.stop();

  if (render_fly_through_) {
    auto video_dir = result_dir_ / "flythrough";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
    renderFlyThroughVideo(video_dir / "output_video", 2452, 740, 30,
                          render_fly_through_speed_, 0.8f, 2);
    renderFlyThroughVideo(video_dir / "output_video_2", 1226, 360, 30,
                          render_fly_through_speed_, 0.8f, 2);
    // render3DExplorationVideo(video_dir / "3d_exploration", 1920, 1080, 30,
    //                          20.0f, 0.05f, false);
  }

  // For debug: basically viewer now
  // while (getIteration() < 100000) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // }

  saveTotalGaussians("_shutdown");
  // Save and clear
  renderAndRecordAllKeyframes("_shutdown");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
  writeTrainingMetricsCSV(result_dir_);

  signalStop();
  if (completion_callback_) {
    completion_callback_();
  }
}

bool GaussianMapper::isKeyframe(const Sophus::SE3f& current_pose,
                                double current_time) {
  if (scene_->keyframes().empty()) {
    return true;
  }

  if (current_time - last_keyframe_timestamp_ < min_keyframe_time_) {
    return false;
    std::cout << "[isKeyframe] Not enough time since last keyframe"
              << std::endl;
  }

  // Check motion
  Sophus::SE3f relative_motion = current_pose.inverse() * last_keyframe_pose_;

  float translation = relative_motion.translation().norm();
  float rotation = Eigen::AngleAxisf(relative_motion.rotationMatrix()).angle();

  if (translation > min_keyframe_translation_ ||
      rotation > min_keyframe_rotation_) {
    // std::cout << "[isKeyframe] Suitable keyframe" << std::endl;
    last_keyframe_pose_ = current_pose;
    return true;
  } else {
    return false;
  }
}

// Frame implementation
GaussianMapper::Frame::Frame(const cv::Mat& rgb,
                             const cv::Mat& depth,
                             const Sophus::SE3f& p,
                             double ts)
    : rgb_image(rgb.clone()),
      depth_image(depth.clone()),
      pose(p),
      timestamp(ts) {}

// LeakyFrameQueue implementation
GaussianMapper::LeakyFrameQueue::LeakyFrameQueue(size_t max_size)
    : max_size_(max_size) {}

void GaussianMapper::LeakyFrameQueue::push(Frame&& frame) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (queue_.size() >= max_size_) {
    // Drop newest frame when full
    // std::cout << "Queue full, dropped newest frame" << std::endl;
    return;
  }

  queue_.push_back(std::move(frame));
  cv_.notify_one();
}

std::optional<GaussianMapper::Frame> GaussianMapper::LeakyFrameQueue::pop(
    bool wait) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (wait) {
    cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
  }

  if (queue_.empty() || stopped_) {
    return std::nullopt;
  }

  Frame frame = std::move(queue_.front());
  queue_.pop_front();
  return frame;
}

void GaussianMapper::LeakyFrameQueue::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  cv_.notify_all();
}

bool GaussianMapper::LeakyFrameQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

size_t GaussianMapper::LeakyFrameQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

void GaussianMapper::handleNewKeyframeFromExternal(
    cv::Mat& rgb_image,
    cv::Mat& depth_or_right_image,
    const Sophus::SE3f& pose,
    const double timestamp) {
  std::cout << "[External Mode] Updating external data..." << std::endl;
  setRecentExternalData(rgb_image, pose);

  // Check if this frame should be a keyframe
  if (!isKeyframe(pose, timestamp)) {
    std::cout << "[External Mode] Not a keyframe, returning" << std::endl;
    return;
  }

  // Update tracking info
  last_keyframe_pose_ = pose;
  last_keyframe_timestamp_ = timestamp;

  // Create keyframe
  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(
      scene_->keyframes().size(), getIteration(), keyframe_save_dir_);
  std::cout << "New kf. fid: " << pkf->fid_ << std::endl;

  // Set pose from external data
  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());

  // External mode always uses camera 0
  Camera& camera = scene_->cameras_.at(0);

  // Call common initialization logic
  createAndInitializeKeyframe(pkf, rgb_image, depth_or_right_image, camera);

  std::cout << "[External Mode] Successfully completed" << std::endl;
}

void GaussianMapper::setRecentExternalData(const cv::Mat& rgb_image,
                                           const Sophus::SE3f& pose) {
  std::unique_lock<std::mutex> lock(mutex_external_data_);

  external_image_ = rgb_image.clone();
  external_pose_ = pose;
}

std::tuple<const cv::Mat, const Sophus::SE3f>
GaussianMapper::getRecentExternalData() {
  std::unique_lock<std::mutex> lock(mutex_external_data_);

  return std::make_tuple(external_image_, external_pose_);
}

void GaussianMapper::setCompletionCallback(std::function<void()> callback) {
  completion_callback_ = callback;
}

torch::Tensor GaussianMapper::computeLoGProbability(
    const torch::Tensor& image) {
  // Step 1: Create Laplacian kernel (same as Python)
  torch::Tensor laplacian_kernel = torch::tensor(
      {{{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));

  // Step 2: Repeat kernel for each channel of input image
  // laplacian_kernel shape: [1, input_channels, 3, 3]
  laplacian_kernel = laplacian_kernel.repeat({1, image.size(0), 1, 1});

  // Step 3: Apply Laplacian convolution with "same" padding
  // For 3x3 kernel, "same" padding = 1 on all sides
  torch::Tensor laplacian = torch::nn::functional::conv2d(
      image.unsqueeze(0),  // Add batch dimension: [1, C, H, W]
      laplacian_kernel,
      torch::nn::functional::Conv2dFuncOptions().padding(
          1)  // "same" padding for 3x3 kernel
  );

  // Step 4: Compute L1 norm across channels (dim=1), keep dimension
  torch::Tensor laplacian_norm =
      torch::linalg_vector_norm(laplacian, 1, /*dim=*/1, /*keepdim=*/true);

  // Step 5: Zero out the borders (exactly like Python)
  // laplacian_norm shape: [1, 1, H, W]
  int H = laplacian_norm.size(-2);  // Second to last dimension
  int W = laplacian_norm.size(-1);  // Last dimension

  // Zero out borders using proper LibTorch indexing
  using namespace torch::indexing;

  // Zero out top and bottom rows
  laplacian_norm.index_put_({Ellipsis, 0, Slice()}, 0);      // Top row
  laplacian_norm.index_put_({Ellipsis, H - 1, Slice()}, 0);  // Bottom row

  // Zero out left and right columns
  laplacian_norm.index_put_({Ellipsis, Slice(), 0}, 0);      // Left column
  laplacian_norm.index_put_({Ellipsis, Slice(), W - 1}, 0);  // Right column

  // Step 6: Convolve with disc kernel and clamp
  // For disc_kernel_ size calculation: if radius=3, kernel is 7x7, so
  // padding=3
  int pad_h = disc_kernel_.size(2) / 2;
  int pad_w = disc_kernel_.size(3) / 2;

  torch::Tensor result = torch::nn::functional::conv2d(
      laplacian_norm, disc_kernel_,
      torch::nn::functional::Conv2dFuncOptions().padding({pad_h, pad_w}));

  // Step 7: Extract result and clamp to [0, 1]
  result = result[0][0];  // Remove batch and channel dimensions -> [H, W]
  return torch::clamp(result, 0.0f, 1.0f);
}

void GaussianMapper::initializeLaplacianOfGaussianKernel() {
  int radius = 3;  // Match Python version
  int kernel_size = 2 * radius + 1;

  // Create coordinate grids
  torch::Tensor y = torch::arange(
      -radius, radius + 1, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor x = y.clone();

  // Create 2D grids
  auto meshgrid = torch::meshgrid({x, y}, "ij");
  torch::Tensor X = meshgrid[1];  // x coordinates
  torch::Tensor Y = meshgrid[0];  // y coordinates

  // Create disc kernel: 1 where distance <= radius + 0.5, 0 elsewhere
  torch::Tensor distances = torch::sqrt(X * X + Y * Y);
  torch::Tensor disc_mask = distances <= (radius + 0.5);

  // Initialize kernel with zeros and set disc region to 1
  disc_kernel_ = torch::zeros({1, 1, kernel_size, kernel_size},
                              torch::TensorOptions().dtype(torch::kFloat32));
  disc_kernel_[0][0] = disc_mask.to(torch::kFloat32);

  // Normalize kernel (divide by sum)
  disc_kernel_ = disc_kernel_ / disc_kernel_.sum();
  disc_kernel_ = disc_kernel_.to(device_type_);
}

void GaussianMapper::initializeStereoDepthEstimator() {
  cv::Size model_resolution(1280, 384);

  // ONNX model in Docker image (rebuilt on each Docker build)
  std::string model_path =
      "/workspace/models/fast_acvnet_plus_onnx_gridsample/"
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(model_resolution.height) + "x" +
      std::to_string(model_resolution.width) + ".onnx";

  // Engine will be saved to /workspace/repo/engines/ (persistent)
  // by StereoDepth::initialize_model()

  // std::string model_path =
  //     "/workspace/models/crestereo/"
  //     "crestereo_init_iter20_720x1280.onnx";
  // std::string model_path =
  //     "/workspace/models/IGEV-plusplus/"
  //     "IGEVplusplusRT_fp32_iter6_kitti.onnx";
  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(model_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  // std::string model_path =
  // "/workspace/repo/models/metric3dv2/metric3d-vit-large.onnx";
  // std::string model_path =
  // "/workspace/repo/models/depth_anything/"
  // "depth_anything_v2_vitl.onnx";

  // ONNX model in Docker image (rebuilt on each Docker build)
  std::string onnx_path =
      "/workspace/models/"
      "depth_anything_v2_vitl.onnx";

  // Engine in persistent volume mount (survives Docker rebuilds)
  std::string persistent_engine_path =
      "/workspace/repo/engines/depth_anything_v2_vitl.engine";

  // Temporary engine path (where DepthAnything initially saves it)
  std::string temp_engine_path =
      "/workspace/models/depth_anything_v2_vitl.engine";

  std::string model_path;

  // Check if persistent engine file exists
  if (std::filesystem::exists(persistent_engine_path)) {
    std::cout << "Using cached TensorRT engine: " << persistent_engine_path
              << std::endl;
    model_path = persistent_engine_path;
  } else {
    std::cout << "Persistent engine not found. Building from ONNX: "
              << onnx_path << std::endl;

    // Check if temporary engine exists from previous run
    if (std::filesystem::exists(temp_engine_path)) {
      std::cout << "Found temporary engine, moving to persistent location..."
                << std::endl;
      // Create engines directory if it doesn't exist
      std::filesystem::create_directories("/workspace/repo/engines");
      std::filesystem::copy_file(
          temp_engine_path, persistent_engine_path,
          std::filesystem::copy_options::overwrite_existing);
      model_path = persistent_engine_path;
    } else {
      // Build from ONNX (DepthAnything will save to temp location)
      std::cout << "Building TensorRT engine from ONNX (this may take a few "
                   "minutes)..."
                << std::endl;
      model_path = onnx_path;

      // Initialize with ONNX to trigger build
      this->monocular_depth_estimator_ =
          std::make_shared<MonoDepth>(model_path);

      // Copy the built engine to persistent location
      if (std::filesystem::exists(temp_engine_path)) {
        std::cout << "Saving engine to persistent location: "
                  << persistent_engine_path << std::endl;
        std::filesystem::create_directories("/workspace/repo/engines");
        std::filesystem::copy_file(
            temp_engine_path, persistent_engine_path,
            std::filesystem::copy_options::overwrite_existing);
        std::cout << "Engine saved! Future runs will use the cached engine."
                  << std::endl;
      }
      return;  // Already initialized
    }
  }

  this->monocular_depth_estimator_ = std::make_shared<MonoDepth>(model_path);
}

void GaussianMapper::projectRgbDepthToPointCloud(
    torch::Tensor& rgb_tensor,
    torch::Tensor& depth_tensor,
    std::vector<float>& camera_intrinsics,
    float min_depth,
    float max_depth,
    Sophus::SE3f& pose,
    std::string& output_path,
    int subsample_factor) {
  int height = rgb_tensor.size(1);
  int width = rgb_tensor.size(2);

  std::cout << "Projecting " << width << "x" << height
            << " image to point cloud..." << std::endl;

  std::cout << "RGB tensor size: " << rgb_tensor.sizes() << std::endl;
  std::cout << "Depth tensor size: " << depth_tensor.sizes() << std::endl;

  // Create validity mask for depth
  // torch::Tensor valid_depth =
  //     (depth_tensor >= min_depth) & (depth_tensor <= max_depth);
  torch::Tensor valid_depth = torch::ones_like(depth_tensor, torch::kBool);

  // Optional: Add subsampling for performance
  if (subsample_factor > 1) {
    torch::Tensor subsample_mask = torch::zeros_like(valid_depth);
    for (int v = 0; v < height; v += subsample_factor) {
      for (int u = 0; u < width; u += subsample_factor) {
        if (v < height && u < width) {
          subsample_mask[v][u] = true;
        }
      }
    }
    valid_depth = valid_depth & subsample_mask;
  }

  // Flatten for processing (following your existing pattern)
  torch::Tensor sample_mask = valid_depth.flatten();
  torch::Tensor depth_flat = depth_tensor.flatten();
  torch::Tensor rgb_flat =
      rgb_tensor.permute({1, 2, 0}).flatten(0, 1);  // HWC -> (H*W)C

  // Get valid data
  torch::Tensor sampled_colors = rgb_flat.index({sample_mask});
  torch::Tensor sampled_depths = depth_flat.index({sample_mask});

  std::cout << "Valid points after filtering: " << sampled_depths.size(0)
            << std::endl;

  if (sampled_depths.size(0) == 0) {
    std::cerr << "No valid depth points found!" << std::endl;
    return;
  }

  // Reproject to 3D using your existing function
  torch::Tensor points3D =
      reprojectDepthPinhole(depth_flat, sample_mask, camera_intrinsics, width);
  points3D = points3D.index({sample_mask});

  // Transform to world coordinates if pose is provided
  if (!pose.matrix().isIdentity()) {
    Sophus::SE3f Twc = pose.inverse();  // Convert camera-to-world
    torch::Tensor Twc_tensor =
        tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
            .transpose(0, 1);
    transformPoints(points3D, Twc_tensor);
  }

  // Visualize using your existing function
  visualizePointCloud(points3D, sampled_colors, output_path);

  // Print some statistics
  auto points_cpu = points3D.cpu();
  auto points_accessor = points_cpu.accessor<float, 2>();

  float min_x = points_accessor[0][0], max_x = points_accessor[0][0];
  float min_y = points_accessor[0][1], max_y = points_accessor[0][1];
  float min_z = points_accessor[0][2], max_z = points_accessor[0][2];

  int num_points = points_cpu.size(0);
  for (int i = 0; i < num_points; i++) {
    min_x = std::min(min_x, points_accessor[i][0]);
    max_x = std::max(max_x, points_accessor[i][0]);
    min_y = std::min(min_y, points_accessor[i][1]);
    max_y = std::max(max_y, points_accessor[i][1]);
    min_z = std::min(min_z, points_accessor[i][2]);
    max_z = std::max(max_z, points_accessor[i][2]);
  }

  std::cout << "Point cloud bounds:" << std::endl;
  std::cout << "  X: [" << min_x << ", " << max_x << "]" << std::endl;
  std::cout << "  Y: [" << min_y << ", " << max_y << "]" << std::endl;
  std::cout << "  Z: [" << min_z << ", " << max_z << "]" << std::endl;
}

void GaussianMapper::projectKeypointsToPointCloud(
    std::shared_ptr<GaussianKeyframe> pkf,
    const std::string& output_path) {
  std::vector<float> valid_points_3d;  // Will store [x1,y1,z1, x2,y2,z2, ...]
  std::vector<float> valid_colors;     // Will store [r1,g1,b1, r2,g2,b2, ...]

  int num_keypoints = pkf->kps_pixel_.size() / 2;

  for (int i = 0; i < num_keypoints; i++) {
    float u = pkf->kps_pixel_[2 * i];
    float v = pkf->kps_pixel_[2 * i + 1];
    float x = pkf->kps_point_local_[3 * i];
    float y = pkf->kps_point_local_[3 * i + 1];
    float z = pkf->kps_point_local_[3 * i + 2];

    bool has_valid_3d =
        (z > 0.1f && z < 100.0f) && (u >= 0 && u < pkf->image_width_) &&
        (v >= 0 && v < pkf->image_height_) && std::isfinite(x) &&
        std::isfinite(y) && std::isfinite(z);

    if (has_valid_3d) {
      // Add 3D point
      valid_points_3d.push_back(x);
      valid_points_3d.push_back(y);
      valid_points_3d.push_back(z);

      // Add red color
      valid_colors.push_back(1.0f);  // R
      valid_colors.push_back(0.0f);  // G
      valid_colors.push_back(0.0f);  // B
    }
  }

  if (valid_points_3d.empty()) {
    std::cerr << "No valid keypoints found!" << std::endl;
    return;
  }

  int num_valid = valid_points_3d.size() / 3;
  std::cout << "Found " << num_valid << " valid keypoints out of "
            << num_keypoints << " total" << std::endl;

  // Create tensors from vectors
  torch::Tensor points3D =
      torch::from_blob(valid_points_3d.data(), {num_valid, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .clone();  // Clone to own the memory

  torch::Tensor colors =
      torch::from_blob(valid_colors.data(), {num_valid, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .clone();  // Clone to own the memory

  // Transform to world coordinates if needed
  Sophus::SE3f pose = pkf->getPosef();
  if (!pose.matrix().isIdentity()) {
    Sophus::SE3f Twc = pose.inverse();
    torch::Tensor Twc_tensor =
        tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
            .transpose(0, 1);
    transformPoints(points3D, Twc_tensor);
  }

  // Use your existing visualization function
  visualizePointCloud(points3D, colors, output_path);

  // Print statistics
  // std::cout << "Keypoint cloud statistics:" << std::endl;
  // auto points_cpu = points3D.cpu();
  // auto mins = points_cpu.min(0).values;
  // auto maxs = points_cpu.max(0).values;
  // std::cout << " X: [" << mins[0].item<float>() << ", " <<
  // maxs[0].item<float>()
  //           << "]" << std::endl;
  // std::cout << " Y: [" << mins[1].item<float>() << ", " <<
  // maxs[1].item<float>()
  //           << "]" << std::endl;
  // std::cout << " Z: [" << mins[2].item<float>() << ", " <<
  // maxs[2].item<float>()
  //           << "]" << std::endl;
}

void GaussianMapper::updateORBSLAMPoses() {
  if (!pSLAM_) return;

  auto* atlas = pSLAM_->getAtlas();
  auto* map = atlas->GetCurrentMap();

  // Get all ORB-SLAM keyframes
  std::vector<ORB_SLAM3::KeyFrame*> orb_keyframes;
  {
    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
    orb_keyframes = map->GetAllKeyFrames();
  }

  // Update each ORB-SLAM keyframe with optimized pose
  for (auto* orb_kf : orb_keyframes) {
    unsigned long kf_id = orb_kf->mnId;

    // Find corresponding Gaussian keyframe
    auto gaussian_kf_it = scene_->keyframes().find(kf_id);
    if (gaussian_kf_it != scene_->keyframes().end()) {
      auto gaussian_kf = gaussian_kf_it->second;

      // Get optimized pose from Gaussian keyframe
      Sophus::SE3f optimized_pose = gaussian_kf->getPosef();

      // Convert to ORB-SLAM format and update
      orb_kf->SetPose(optimized_pose);
    }
  }
}

torch::Tensor GaussianMapper::sampleConf(const torch::Tensor& mono_depth_conf,
                                         const torch::Tensor& uv,
                                         int width,
                                         int height) {
  // mono_depth_conf shape: [1, 1, H, W]
  // uv shape: [N, 2] where N is number of points
  // Returns: [N] confidence values

  // Reshape uv to [1, 1, N, 2] for grid_sample
  torch::Tensor uv_reshaped = uv.view({1, 1, -1, 2});

  // Convert UV coordinates to normalized coordinates [-1, 1]
  // grid_sample expects coordinates in [-1, 1] range
  torch::Tensor normalized_uv = uv_reshaped.clone();
  normalized_uv.select(-1, 0) =
      (normalized_uv.select(-1, 0) / (width - 1)) * 2.0f - 1.0f;  // x
  normalized_uv.select(-1, 1) =
      (normalized_uv.select(-1, 1) / (height - 1)) * 2.0f - 1.0f;  // y

  // Use grid_sample for bilinear interpolation
  torch::Tensor sampled = torch::nn::functional::grid_sample(
      mono_depth_conf, normalized_uv,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .padding_mode(torch::kZeros)
          .align_corners(true));

  // Return flattened result [N]
  return sampled[0][0][0];  // Remove batch and channel dimensions
}