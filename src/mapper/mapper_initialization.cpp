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

  // Create keyframe
  keyframe_selector_ = std::make_shared<KeyframeSelection>(
      scene_, 50.0, &kfs_loss_, &kfs_used_times_);

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
}