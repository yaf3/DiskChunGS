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
#include "include/debugging_utils.h"
#include "include/gaussian_rasterizer.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"
#include "include/profiling.h"
#include "include/render_flythrough.h"

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

  chunk_save_dir_ = result_dir_ / "chunks";

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

  // Initialize chunk manager
  initializeChunkManagement();

  keyframe_queue_ = std::make_shared<KeyframeQueue>(
      scene_, 40, keyframe_similarity_threshold_, opt_params_.auto_distribute_,
      &kfs_loss_);
  // keyframe_queue_->setChunkManager(chunk_manager_);

  // Initialize Laplacian of Gaussian kernel
  initializeLaplacianOfGaussianKernel();

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
      initializeMonocularDepthEstimator();
    } break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO: {
      this->sensor_type_ = STEREO;
      this->stereo_baseline_length_ = pSLAM->getSettings()->b();
      this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
          this->stereo_min_disparity_, this->stereo_num_disparity_);
      this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
      stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);

      initializeStereoDepthEstimator();
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

  // Initialize chunk manager
  initializeChunkManagement();

  keyframe_queue_ = std::make_shared<KeyframeQueue>(
      scene_, 40, keyframe_similarity_threshold_, opt_params_.auto_distribute_,
      &kfs_loss_);
  // keyframe_queue_->setChunkManager(chunk_manager_);

  // Initialize Laplacian of Gaussian kernel
  initializeLaplacianOfGaussianKernel();

  ORB_SLAM3::System::eSensor system_mode;
  if (sensor_type == STEREO) {
    system_mode = ORB_SLAM3::System::STEREO;
  } else if (sensor_type == RGBD) {
    system_mode = ORB_SLAM3::System::RGBD;
  } else {
    system_mode = ORB_SLAM3::System::MONOCULAR;
  }

  // Check settings file
  cv::FileStorage fsSettings(orb_settings_path.c_str(), cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "Failed to open settings file at: " << orb_settings_path << endl;
    exit(-1);
  }

  ORB_SLAM3::Settings* orb_settings;

  cv::FileNode node = fsSettings["File.version"];
  if (!node.empty() && node.isString() && node.string() == "1.0") {
    orb_settings = new ORB_SLAM3::Settings(orb_settings_path, system_mode);
  }

  cv::Size SLAM_im_size = orb_settings->newImSize();
  UndistortParams undistort_params(SLAM_im_size,
                                   orb_settings->camera1DistortionCoef());

  vector<ORB_SLAM3::GeometricCamera*> SLAM_cameras;
  SLAM_cameras.push_back(orb_settings->camera1());
  SLAM_cameras.push_back(orb_settings->camera2());

  for (auto& SLAM_camera : SLAM_cameras) {
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

  if (sensor_type == STEREO) {
    initializeStereoDepthEstimator();
  } else if (sensor_type == RGBD) {
    // initializeMonocularDepthEstimator();
  } else {
    initializeMonocularDepthEstimator();
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
  do_stereo_loss_ = settings_file["Stereo.do_stereo_loss"].operator int();
  min_depth_ = settings_file["Mapper.min_depth_"].operator float();
  max_depth_ = settings_file["Mapper.max_depth_"].operator float();

  inactive_geo_densify_ =
      (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
  depth_densify_subsample_ratio_ =
      settings_file["Mapper.depth_densify_subsample_ratio"].operator float();
  depth_densify_ = (settings_file["Mapper.depth_densify"].operator int()) != 0;
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
  keyframe_similarity_threshold_ =
      settings_file["Mapper.keyframe_similarity_threshold"].operator float();
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
  render_fly_through_ =
      (settings_file["Record.render_fly_through"].operator int()) != 0;
  render_fly_through_speed_ =
      (settings_file["Record.render_fly_through_speed"].operator float()) > 0;

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

  opt_params_.percent_dense_ =
      settings_file["Optimization.percent_dense"].operator float();
  opt_params_.lambda_dssim_ =
      settings_file["Optimization.lambda_dssim"].operator float();
  opt_params_.lambda_depth_ =
      settings_file["Optimization.lambda_depth"].operator float();
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
  appearance_embedding_ =
      settings_file["Optimization.appearance_embedding"].operator int();
  init_proba_scaler_ =
      settings_file["Optimization.init_proba_scaler"].operator float();

  // Viewer Parameters
  rendered_image_viewer_scale_ =
      settings_file["GaussianViewer.image_scale"].operator float();
  rendered_image_viewer_scale_main_ =
      settings_file["GaussianViewer.image_scale_main"].operator float();

  chunk_size_ = settings_file["Chunking.chunk_size"].operator float();
  max_chunks_in_memory_ = settings_file["Chunking.max_chunks"].operator int();
}

void GaussianMapper::run() {
  // Delete existing chunks since training
  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  // First loop: Initial gaussian mapping
  while (!isStopped()) {
    // Check conditions for initial mapping
    if (hasMetInitialMappingConditions()) {
      pSLAM_->getAtlas()->clearMappingOperation();

      // Get initial sparse map
      auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
      std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
      std::vector<ORB_SLAM3::MapPoint*> vpMPs;
      torch::Tensor initialSparsePoints, initialSparseColors, initialOpacities;
      {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        vpKFs = pMap->GetAllKeyFrames();
        vpMPs = pMap->GetAllMapPoints();

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        initialSparsePoints =
            torch::zeros({static_cast<int64_t>(vpMPs.size()), 3}, options);
        initialSparseColors =
            torch::zeros({static_cast<int64_t>(vpMPs.size()), 3}, options);
        initialOpacities = general_utils::inverse_sigmoid(
            0.2f * torch::ones({initialSparsePoints.size(0), 1},
                               torch::TensorOptions()
                                   .dtype(torch::kFloat)
                                   .device(device_type_)));

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
          scene_->addKeyframe(new_kf);
          new_kf->initOptimizer(device_type_, opt_params_.pose_lr_,
                                opt_params_.exposure_lr_,
                                opt_params_.depth_scale_bias_lr_);
          kfid_shuffled_ = false;
          keyframe_queue_->notifyNewKeyframeAdded(new_kf);

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
        pkf->generatePyramidImages(device_type_);

        if (sensor_type_ == MONOCULAR) {
          pkf->setupMonoData(device_type_, monocular_depth_estimator_,
                             min_depth_, max_depth_);
        } else if (sensor_type_ == STEREO &&
                   !pkf->img_auxiliary_undist_.empty()) {
          pkf->setupStereoData(stereo_baseline_length_, device_type_,
                               stereo_depth_estimator_, min_depth_, max_depth_);
        } else if (sensor_type_ == RGBD &&
                   !pkf->img_auxiliary_undist_.empty()) {
          // Preprocess and store depth image tensor
          if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat depth_gpu;
            depth_gpu.upload(pkf->img_auxiliary_undist_);
            pkf->depth_image_ =
                tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
          } else {
            // CPU version
            pkf->depth_image_ = tensor_utils::cvMat2TorchTensor_Float32(
                pkf->img_auxiliary_undist_, device_type_);
          }
          // Create depth pyramid for initial keyframes
          if (do_gaus_pyramid_training_) {
            pkf->generatePyramidDepth(device_type_, pkf->img_auxiliary_undist_);
          }
        }

        if (!initial_mapped_) {
          std::unique_lock<std::mutex> lock_render(mutex_render_);
          scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
          std::cout << "Adding initial points\n";
          addPoints(initialSparsePoints, initialSparseColors, torch::Tensor(),
                    initialOpacities);
          initial_mapped_ = true;
        }
        if (isdoingDepthDensify()) increasePcdByDepthReconstruction(pkf);
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

    // if (getIteration() % 2000 == 0) {
    //   keyframe_queue_->visualizeClusters(
    //       "/workspaces/large_scale_gaussian_slam/cluster_visualization.svg");
    // }

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

    // if (getIteration() % 2000 == 0) {
    //   keyframe_queue_->visualizeClusters(
    //       "/workspaces/large_scale_gaussian_slam/cluster_visualization.svg");
    // }

    if (getIteration() >= opt_params_.iterations_) break;
  }

  // while (getIteration() < 10000000) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // }

  // Fourth loop: Tail gaussian optimization
  int densify_interval = densifyInterval();
  int n_delay_iters = densify_interval * 0.8;
  while (getIteration() - SLAM_stop_iter < n_delay_iters ||
         getIteration() % densify_interval < n_delay_iters ||
         isKeepingTraining()) {
    trainForOneIteration();
  }

  if (render_fly_through_) {
    auto video_dir = result_dir_ / "flythrough";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
    renderFlyThroughVideo(video_dir / "output_video", 1920, 1080, 30,
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
  // savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
  // "ply");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  signalStop();

  if (completion_callback_) {
    completion_callback_();
  }
}

void GaussianMapper::trainColmap() {
  // Delete existing chunks since training
  if (!chunk_save_dir_.empty() && std::filesystem::exists(chunk_save_dir_)) {
    for (const auto& entry :
         std::filesystem::directory_iterator(chunk_save_dir_)) {
      std::filesystem::remove_all(entry.path());
    }
  }

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

  // Prepare for training
  {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
    int num_points = static_cast<int>(scene_->cached_point_cloud_.size());
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    auto pcd_it = scene_->cached_point_cloud_.begin();
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
      auto& point = (*pcd_it).second;
      fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
      fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
      fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
      color.index({point_idx, 0}) = point.color_(0);
      color.index({point_idx, 1}) = point.color_(1);
      color.index({point_idx, 2}) = point.color_(2);
      ++pcd_it;
    }
    std::cout << "Adding initial points\n";
    // TODO UPDATE
    // addPoints(fused_point_cloud, color, torch::Tensor(),
    // scene_->keyframes());
    this->initial_mapped_ = true;
  }

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

  if (render_fly_through_) {
    auto video_dir = result_dir_ / "flythrough";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
    renderFlyThroughVideo(video_dir / "output_video", 1920, 1080, 30,
                          render_fly_through_speed_, 0.8f, 2);
    // render3DExplorationVideo(video_dir / "3d_exploration", 1920, 1080, 30,
    //                          20.0f, 0.05f, false);
  }

  // Save and clear
  renderAndRecordAllKeyframes("_shutdown");
  saveTotalGaussians("_shutdown");
  // savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
  // "ply");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
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

  // if (getIteration() % 100 == 0) {
  //   updateORBSLAMPoses();
  // }

  auto timer_cullSparseChunks = ProfilingUtils::Timer("cullSparseChunks");
  int min_points_chunk_threshold = 1000;
  int min_chunk_iterations = 200;
  chunk_manager_->cullSparseChunks(min_points_chunk_threshold,
                                   min_chunk_iterations);
  timer_cullSparseChunks.stop();

  auto iter_start_timing = std::chrono::steady_clock::now();

  // size_t keyframe_lookahead = 3;
  // std::vector<std::shared_ptr<GaussianKeyframe>> upcoming_keyframes =
  //     getUpcomingKeyframes(keyframe_lookahead);

  // std::cout << "Keyframes lookahead: ";
  // for (auto& keyframe : upcoming_keyframes) {
  //   std::cout << keyframe->fid_ << " ";
  // }
  // std::cout << std::endl;

  auto timer_pickKeyframe = ProfilingUtils::Timer("pickKeyframe");

  // if (pSLAM_) {
  //   ORB_SLAM3::MapDrawer* pSlamMapDrawer = pSLAM_->getMapDrawer();
  //   Sophus::SE3f slam_Twc = pSlamMapDrawer->GetCurrentCameraPose();
  //   keyframe_queue_->setCurrentPose(slam_Twc);
  // } else {
  //   auto [_, external_Twc] = getRecentExternalData();
  //   keyframe_queue_->setCurrentPose(external_Twc);
  // }

  std::shared_ptr<GaussianKeyframe> viewpoint_cam;
  switch (keyframe_selection_strategy_) {
    // Random sliding window keyframe
    case 0: {
      viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    } break;
    // Recent k
    case 1: {
      viewpoint_cam = keyframe_queue_->getNextKeyframe();
    } break;
    default: {
      throw std::runtime_error(
          "[GaussianMapper] Invalid keyframe selection strategy");
    }
  }
  timer_pickKeyframe.stop();
  if (!viewpoint_cam) {
    increaseIteration(-1);
    throw std::runtime_error(
        "[GaussianMapper] Keyframe not found for training");
    return;
  }

  // std::cout << "Using keyframe id: " << viewpoint_cam->fid_ << std::endl;

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  // if (isdoingInactiveGeoDensify() &&
  // !viewpoint_cam->done_inactive_geo_densify_)
  //   increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

  auto [gt_image, gt_depth, mask, image_height, image_width] =
      viewpoint_cam->getTrainingData(
          undistort_mask_[viewpoint_cam->camera_id_],
          scene_->cameras_.at(viewpoint_cam->camera_id_)
              .gaus_pyramid_undistort_mask_,
          isdoingGausPyramidTraining());

  auto timer_waitForMutex = ProfilingUtils::Timer("waitForMutex");
  // Mutex lock for usage of the gaussian model
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  timer_waitForMutex.stop();

  size_t keyframe_lookahead = 3;
  // std::vector<std::shared_ptr<GaussianKeyframe>> upcoming_keyframes =
  //     getUpcomingKeyframes(keyframe_lookahead);
  // std::cout << "Keyframes lookahead: ";
  // auto timer_preload = ProfilingUtils::Timer("preloadUpcomingKeyframes");
  // Only load keyframe after next (so basically get ready for the next
  // iteration)
  // chunk_manager_->preloadVisibleChunks(upcoming_keyframes[1], true);
  // for (auto& keyframe : upcoming_keyframes) {
  //   chunk_manager_->preloadVisibleChunks(keyframe, true);
  //   // std::cout << keyframe->fid_ << " ";
  // }
  // std::cout << std::endl;
  // timer_preload.stop();

  auto timer_loadVisibleChunks = ProfilingUtils::Timer("loadVisibleChunks");
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->loadVisibleChunks(viewpoint_cam, true);
  timer_loadVisibleChunks.stop();

  auto timer_misc_updates = ProfilingUtils::Timer("ITER/LR/SH Updates");

  // std::cout << "[Optimization] Num visible chunks: " <<
  // visible_chunks.size()
  //           << std::endl;

  // Extract models from chunks
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    // std::cout << chunk->getCoord().x << " " << chunk->getCoord().y << " "
    //           << chunk->getCoord().z << std::endl;
    if (chunk && chunk->getGaussians() &&
        chunk->getGaussians()->getXYZ().sizes()[0] > 0) {
      models.push_back(chunk->getGaussians());
    } else {
      throw std::runtime_error("Chunk/Gaussians are null");
    }
  }

  if (models.empty()) {
    std::cout << "[Optimization] No valid models to render" << std::endl;
    // throw std::runtime_error("[Optimization] No valid models to render");
    return;  // Early return if no valid models
  }

  for (const auto& gaussians : models) {
    gaussians->incrementLocalIteration();

    // Update learning rate based on the model's local iteration count
    // gaussians->updateLearningRate();

    // Set feature, opacity, scaling, and rotation learning rates
    gaussians->setFeatureLearningRate(featureLearningRate());
    gaussians->setOpacityLearningRate(opacityLearningRate());
    gaussians->setScalingLearningRate(scalingLearningRate());
    gaussians->setRotationLearningRate(rotationLearningRate());
  }

  timer_misc_updates.stop();

  // int num_gaussians = 0;
  // for (const auto& model : models) {
  //   num_gaussians += model->getXYZ().size(0);
  // }
  // std::cout << "[Optimization] Num visible chunks: " << visible_chunks.size()
  //           << ", Num Gaussians: " << num_gaussians << std::endl;
  // torch::Tensor identity_view_matrix = torch::eye(
  //     4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  // Render

  torch::Tensor view_matrix = viewpoint_cam->getRT().transpose(0, 1);

  auto timer_render = ProfilingUtils::Timer("render");
  auto render_pkg = GaussianRenderer::render(
      models, viewpoint_cam, image_height, image_width, pipe_params_,
      background_, override_color_, 1.0f, false, viewpoint_cam->FoVx_,
      viewpoint_cam->FoVy_, view_matrix, viewpoint_cam->projection_matrix_);

  timer_render.stop();
  auto rendered_image = std::get<1>(render_pkg);

  std ::vector<torch::Tensor> screenspace_points_vec = std::get<2>(render_pkg);
  std::vector<torch::Tensor> radii_vec = std::get<3>(render_pkg);

  auto timer_loss_calculation = ProfilingUtils::Timer("loss_calculation");
  // Loss calculation (same as before)
  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto Ll1 = l1_loss(rendered_image, gt_image, 1.0f);
  auto Lssim = loss_utils::fast_ssim(rendered_image, gt_image);
  float lambda_dssim = lambdaDssim();
  float lambda_depth = lambdaDepth();
  auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);

  if (gt_depth.defined()) {
    torch::Tensor rendered_depth, depth_loss;
    rendered_depth = std::get<0>(render_pkg);
    depth_loss = loss_utils::smooth_l1_depth_loss(rendered_depth, gt_depth);
    loss += viewpoint_cam->depth_loss_weight * depth_loss;

    // if (getIteration() % 100 == 0) {
    //   std::filesystem::create_directories("./debug_mono");
    //   std::string rgb_render_filename = "./debug_mono/rendered_rgb_" +
    //                                     std::to_string(viewpoint_cam->fid_) +
    //                                     ".png";
    //   cv::Mat output_image =
    //       tensor_utils::torchTensor2CvMat_Float32(rendered_image);
    //   output_image.convertTo(output_image, CV_8UC1, 255.0, 0.0);
    //   cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
    //   cv::imwrite(rgb_render_filename, output_image);
    //   std::string render_filename = "./debug_mono/rendered_depth_" +
    //                                 std::to_string(viewpoint_cam->fid_) +
    //                                 ".png";
    //   colorize_and_save_depth(rendered_depth.detach().cpu(), render_filename,
    //                           min_depth_, max_depth_);
    //   std::string gt_filename = "./debug_mono/gt_depth_" +
    //                             std::to_string(viewpoint_cam->fid_) + ".png";
    //   colorize_and_save_depth(gt_depth.detach().cpu(), gt_filename,
    //   min_depth_,
    //                           max_depth_);
    // }
  }

  // std::cout << "Ll1: " << Ll1.item<float>() << std::endl;
  // std::cout << "Lssim: " << Lssim.item<float>() << std::endl;
  // std::cout << "Ll1_depth: " << Ll1_depth.item<float>() << std::endl;

  if (opt_params_.opacity_reg_) {
    for (const auto& gaussians : models) {
      loss += opt_params_.opacity_reg_ *
              gaussians->getOpacityActivation().abs().mean();
      // std::cout << "opacity_reg: "
      //           << (opt_params_.opacity_reg_ *
      //               gaussians->getOpacityActivation().abs().mean())
      //                  .item<float>()
      //           << std::endl;
    }
  }

  // float iso_reg_weight = 0.1f;  // Adjust as needed
  // for (const auto& gaussians : models) {
  //   // Get scaling
  //   torch::Tensor scaling = gaussians->getScalingActivation();

  //   // Calculate mean scaling for each Gaussian
  //   torch::Tensor mean_scale = scaling.mean(1, /*keepdim=*/true);

  //   // Calculate L1 distance from each scaling component to the mean
  //   // This penalizes primitives with high aspect ratio as in Eq. (9)
  //   torch::Tensor iso_penalty = (scaling - mean_scale).abs().sum(1).mean();

  //   // Add weighted regularization term to loss
  //   loss += iso_reg_weight * iso_penalty;
  //   // std::cout << "iso_penalty: " << iso_penalty.item<float>() <<
  //   std::endl;
  // }

  timer_loss_calculation.stop();

  auto timer_backwards = ProfilingUtils::Timer("backwards");
  loss.backward();
  timer_backwards.stop();

  // Debug: Show pose optimization status
  // if (getIteration() % 100 == 0) {
  //   std::cout << "=== Training Iteration " << getIteration()
  //             << " ===" << std::endl;
  //   std::cout << "Keyframe ID: " << viewpoint_cam->fid_ << std::endl;
  //   std::cout << "Loss before step: " << loss.item<float>() << std::endl;
  // }

  auto timer_pose_exposure_step = ProfilingUtils::Timer("pose&exposure_step");
  viewpoint_cam->step();
  timer_pose_exposure_step.stop();
  // if (true) {
  //   if (getIteration() % 100 == 0) {
  //     auto transform = viewpoint_cam->exposure_transform_.detach().cpu();

  //     // Extract the 3x3 scaling/rotation part and bias part
  //     auto scale_rot = transform.slice(1, 0, 3);        // First 3 columns
  //     (3x3) auto bias = transform.slice(1, 3, 4).squeeze(1);  // Last column
  //     (3x1)

  //     std::cout << "Keyframe " << viewpoint_cam->fid_ << " appearance at iter
  //     "
  //               << getIteration() << std::endl;

  //     // Print the full 3x4 matrix for complete visibility
  //     std::cout << "  Transform matrix (3x4):" << std::endl;
  //     for (int i = 0; i < 3; i++) {
  //       std::cout << "    [";
  //       for (int j = 0; j < 4; j++) {
  //         std::cout << std::setprecision(4) << std::fixed
  //                   << transform[i][j].item<float>();
  //         if (j < 3) std::cout << ", ";
  //       }
  //       std::cout << "]" << std::endl;
  //     }

  //     // Also show diagonal values (main scaling factors) and bias for quick
  //     // reference
  //     std::cout << "  Diagonal scaling: [" << std::setprecision(4) <<
  //     std::fixed
  //               << scale_rot[0][0].item<float>() << ", "
  //               << scale_rot[1][1].item<float>() << ", "
  //               << scale_rot[2][2].item<float>() << "]" << std::endl;
  //     std::cout << "  Bias: [" << bias[0].item<float>() << ", "
  //               << bias[1].item<float>() << ", " << bias[2].item<float>() <<
  //               "]"
  //               << std::endl;

  //     // Compute and display the magnitude of change from identity
  //     auto identity_3x4 = torch::zeros_like(transform);
  //     identity_3x4.slice(1, 0, 3) = torch::eye(3);
  //     auto deviation = torch::norm(transform - identity_3x4).item<float>();
  //     std::cout << "  Deviation from identity: " << std::setprecision(6)
  //               << deviation << std::endl;

  //     std::cout << std::endl;
  //   }
  // }

  auto timer_cuda_sync = ProfilingUtils::Timer("cuda_sync");
  torch::cuda::synchronize();
  timer_cuda_sync.stop();

  auto timer_optimizer_step = ProfilingUtils::Timer("optimizer_step");
  // Optimizer step
  if (getIteration() < opt_params_.iterations_ ||
      opt_params_.iterations_ == -1) {
    for (int model_idx = 0; model_idx < models.size(); model_idx++) {
      const auto& gaussians = models[model_idx];

      // // Check if gaussians and optimizer are valid
      // if (!gaussians) {
      //   std::cerr << "Warning: Gaussians is null for model " << model_idx
      //             << std::endl;
      //   continue;
      // }

      // if (!gaussians->optimizer_) {
      //   std::cerr << "Warning: Optimizer is null for model " << model_idx
      //             << std::endl;
      //   continue;
      // }

      // Get radii for this model to determine visibility
      const auto& radii = radii_vec[model_idx];

      // Calculate visibility filter - gaussians with radii > 0 are visible
      auto visibility_filter = (radii > 0);

      // // Debug the inputs before calling step
      // std::cout << "Model " << model_idx << " debug info:" << std::endl;
      // std::cout << "  - Gaussians XYZ size: " << gaussians->getXYZ().size(0)
      //           << std::endl;
      // std::cout << "  - Radii size: " << radii.size(0) << std::endl;
      // std::cout << "  - Visibility filter size: " <<
      // visibility_filter.size(0)
      //           << std::endl;
      // std::cout << "  - Visibility filter device: "
      //           << visibility_filter.device() << std::endl;
      // std::cout << "  - Visibility filter dtype: " <<
      // visibility_filter.dtype()
      //           << std::endl;

      // // Check if sizes match
      // if (radii.size(0) != gaussians->getXYZ().size(0)) {
      //   std::cerr << "ERROR: Radii size doesn't match gaussians count!"
      //             << std::endl;
      //   continue;
      // }

      // if (visibility_filter.size(0) != gaussians->getXYZ().size(0)) {
      //   std::cerr
      //       << "ERROR: Visibility filter size doesn't match gaussians count!"
      //       << std::endl;
      //   continue;
      // }

      // Use the sparse optimizer with visibility information
      // gaussians->optimizer_->step(visibility_filter,
      //                             gaussians->getXYZ().size(0));
      gaussians->optimizerStep(visibility_filter, gaussians->getXYZ().size(0));
      gaussians->optimizer_->zero_grad(true);
    }
  }
  timer_optimizer_step.stop();

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

      int local_iter = gaussians->getLocalIteration();

      // Densification
      if (local_iter < opt_params_.densify_until_iter_ ||
          opt_params_.densify_until_iter_ == -1) {
        // Keep track of max radii in image-space for pruning
        gaussians->max_radii2D_.index_put_(
            {visibility_filter},
            torch::max(gaussians->max_radii2D_.index({visibility_filter}),
                       radii.index({visibility_filter})));

        // gaussians->addDensificationStats(screenspace_points_vec[model_idx],
        //                                  visibility_filter);

        if ((local_iter > opt_params_.densify_from_iter_) &&
            (local_iter % densifyInterval() == 0)) {
          // int size_threshold = (local_iter < prune_big_point_after_iter_ ||
          //                       prune_big_point_after_iter_ == -1)
          //                          ? 0
          //                          : 20;
          int size_threshold = viewpoint_cam->image_width_ / 2.0f;
          // int size_threshold = 5000;
          gaussians->prune(densify_min_opacity_, size_threshold);
          // gaussians->densifyAndPrune(densifyGradThreshold(),
          //                            densify_min_opacity_,
          //                            scene_->cameras_extent_,
          //                            size_threshold);
        }

        if (opacityResetInterval() &&
            (local_iter % opacityResetInterval() == 0 ||
             (model_params_.white_background_ &&
              local_iter == opt_params_.densify_from_iter_)))
          gaussians->resetOpacity();
      }
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
              << ", ema_loss:" << ema_loss_for_log_
              << ", active_chunks:" << chunk_manager_->getStats().active_chunks
              << std::endl;
  }

  if ((all_keyframes_record_interval_ &&
       getIteration() % all_keyframes_record_interval_ == 0)) {
    renderAndRecordAllKeyframes();
    // savePly(result_dir_ / std::to_string(getIteration()) / "ply");
    saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
              "data");
  }

  if (loop_closure_iteration_) loop_closure_iteration_ = false;

  // auto timer_evictUnusedChunks =
  // ProfilingUtils::Timer("evictUnusedChunks"); Periodically cull gaussians
  // outside of borders & evict unused chunks if (getIteration() % 200 == 0) {
  // chunk_manager_->transferGaussiansAcrossChunks();
  // chunk_manager_->cullGaussiansOutsideChunkBorders();
  // chunk_manager_->evictUnusedChunks();
  // }
  // timer_evictUnusedChunks.stop();

  auto timer_releaseChunksFromOptimization =
      ProfilingUtils::Timer("releaseChunksFromOptimization");
  chunk_manager_->releaseChunksFromOptimization(visible_chunks);
  timer_releaseChunksFromOptimization.stop();
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
    processLoopClosureBA(opr);
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

      // If the keyframe is already in the scene, only update the pose
      if (pkf) {
        auto& pose = std::get<2>(kf);
        pkf->setPose(pose.unit_quaternion().cast<double>(),
                     pose.translation().cast<double>());
        pkf->computeTransformTensors();

        // Give local BA keyframes times of use
        increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
      } else {
        // Create a new keyframe
        handleNewKeyframe(kf);
      }

      // Add to the collection of keyframes for point processing
      associated_keyframe_map[kfid] = scene_->getKeyframe(kfid);
    }

    // Collect all points
    auto& associated_points = opr.associatedMapPoints();
    auto& points = std::get<0>(associated_points);
    auto& colors = std::get<1>(associated_points);

    // Append to our batched points/colors vectors
    all_points.insert(all_points.end(), points.begin(), points.end());
    all_colors.insert(all_colors.end(), colors.begin(), colors.end());
  }

  // timer_LocalMapping_before_addPoints.stop();

  // auto timer_addPoints = ProfilingUtils::Timer("addPoints");
  // Add all collected points to the model in a single call
  if (initial_mapped_ && all_points.size() >= 30) {
    int num_new_points = static_cast<int>(all_points.size() / 3);

    // Use pinned memory for faster GPU transfer if using CUDA
    torch::TensorOptions tensor_options =
        torch::TensorOptions().dtype(torch::kFloat32);
    if (device_type_ == torch::kCUDA) {
      tensor_options = tensor_options.pinned_memory(true);
    }

    // Create tensors optimized for transfer
    torch::Tensor points_tensor =
        torch::from_blob(all_points.data(), {num_new_points, 3}, tensor_options)
            .to(device_type_);
    torch::Tensor colors_tensor =
        torch::from_blob(all_colors.data(), {num_new_points, 3}, tensor_options)
            .to(device_type_);
    torch::Tensor opacities_tensor = general_utils::inverse_sigmoid(
        0.2f *
        torch::ones(
            {points_tensor.size(0), 1},
            torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    // Process all points at once
    torch::NoGradGuard no_grad;
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    addPoints(points_tensor, colors_tensor, torch::Tensor(), opacities_tensor);
  }
  // timer_addPoints.stop();
}

void GaussianMapper::processLoopClosureBA(ORB_SLAM3::MappingOperation& opr) {
  // Existing loop closure code...
  std::cout << "[Gaussian Mapper]Loop Closure Detected." << std::endl;
  std::cout << "[DEBUG] Starting loop closure with scale factor: "
            << opr.mfScale << std::endl;

  // Get the loop keyframe scale modification factor
  float loop_kf_scale = opr.mfScale;

  // Get new keyframes (scaled transformation applied in ORB-SLAM3)
  auto& associated_kfs = opr.associatedKeyFrames();

  std::cout << "[DEBUG] Processing " << associated_kfs.size() << " keyframes"
            << std::endl;

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_0_before_loop_correction") /
              "data");
  }

  int num_transformed = 0;

  // Instead of tracking processed chunks as a set
  std::unordered_map<ChunkCoord, torch::Tensor, ChunkCoordHash>
      chunk_transformed_flags;

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

      if (large_rot || large_trans) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::cout << "[Gaussian Mapper]Large loop correction detected for kf "
                  << kfid << std::endl;

        while (chunk_manager_->getActiveChunks().size() >
               max_chunks_in_memory_) {
          std::cout << "[Gaussian Mapper]Too many chunks in memory: "
                    << chunk_manager_->getActiveChunks().size()
                    << ", waiting for chunks to be released" << std::endl;
          chunk_manager_->triggerLruCheck();

          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // Get chunks visible from this keyframe
        std::vector<std::shared_ptr<Chunk>> visible_chunks =
            chunk_manager_->loadVisibleChunks(pkf);

        for (const auto& chunk : visible_chunks) {
          std::cout << "Now processing: " << chunk->getCoord().x << " "
                    << chunk->getCoord().y << " " << chunk->getCoord().z
                    << std::endl;
        }

        // Calculate the transformation
        diff_pose.translation() -= inv_pose.translation();
        diff_pose.translation() *= loop_kf_scale;
        diff_pose.translation() += inv_pose.translation();

        torch::Tensor diff_pose_tensor = tensor_utils::EigenMatrix2TorchTensor(
                                             diff_pose.matrix(), device_type_)
                                             .transpose(0, 1);

        for (auto& chunk : visible_chunks) {
          if (!chunk || !chunk->getGaussians()) {
            throw std::runtime_error("Chunk or gaussians are null");
          }

          auto gaussians = chunk->getGaussians();
          auto chunk_coord = chunk->getCoord();

          // Get or create the transformed flags for this chunk
          auto it = chunk_transformed_flags.find(chunk_coord);
          torch::Tensor chunk_point_flags;

          if (it == chunk_transformed_flags.end()) {
            // First time seeing this chunk, initialize all flags to
            // "not transformed"
            std::cout << "First time seeing this chunk, set all flags to "
                         "not transformed"
                      << std::endl;
            chunk_point_flags = torch::full({gaussians->xyz_.size(0)}, true,
                                            torch::TensorOptions()
                                                .device(device_type_)
                                                .dtype(torch::kBool));
            chunk_transformed_flags[chunk_coord] = chunk_point_flags;
          } else {
            chunk_point_flags = it->second;
            // If chunk size has changed (new points added), resize the
            // flags tensor
            if (chunk_point_flags.size(0) != gaussians->xyz_.size(0)) {
              // std::cout << "[DEBUG] Resizing flags tensor from "
              //           << chunk_point_flags.size(0) << " to "
              //           << gaussians->xyz_.size(0) << std::endl;
              int64_t num_new_points =
                  gaussians->xyz_.size(0) - chunk_point_flags.size(0);
              if (num_new_points > 0) {
                chunk_point_flags =
                    torch::cat({chunk_point_flags,
                                torch::full({num_new_points}, true,
                                            chunk_point_flags.options())},
                               /*dim=*/0);
                chunk_transformed_flags[chunk_coord] = chunk_point_flags;
              }
            }
          }

          // std::cout << "[DEBUG] Before transform - xyz: "
          //           << gaussians->xyz_.sizes()
          //           << ", flags: " << chunk_point_flags.sizes() <<
          //           std::endl;

          int chunk_transformed = 0;
          // std::cout << "Calling scaledTransformVisiblePointsOfKeyframe"
          //           << std::endl;
          assert(chunk_manager_->getChunkState(chunk_coord) ==
                     ChunkState::OPTIMIZING &&
                 "Chunk should be in optimizing state");

          std::cout << "[DEBUG] Chunk flags - true count: "
                    << chunk_point_flags.sum().item<int>() << " out of "
                    << chunk_point_flags.size(0) << std::endl;
          gaussians->scaledTransformVisiblePointsOfKeyframe(
              chunk_point_flags, diff_pose_tensor, pkf->world_view_transform_,
              pkf->full_proj_transform_, pkf->creation_iter_,
              stableNumIterExistence(), chunk_transformed, loop_kf_scale);

          chunk_transformed_flags[chunk_coord] = chunk_point_flags;

          std::cout << "[DEBUG] Transformed " << chunk_transformed
                    << " points successfully" << std::endl;

          num_transformed += chunk_transformed;
        }

        // Give loop keyframes times of use
        increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);

        chunk_manager_->releaseChunksFromOptimization(visible_chunks);
        chunk_manager_->triggerLruCheck();
      }

      // Update keyframe pose
      pkf->setPose(pose.unit_quaternion().cast<double>(),
                   pose.translation().cast<double>());
      pkf->computeTransformTensors();
    } else {
      std::cout << "Actually new keyframe in LC" << std::endl;
      handleNewKeyframe(kf);
    }
  }

  if (record_loop_ply_) {
    saveScene(result_dir_ /
              (std::to_string(getIteration()) + "_1_after_loop_correction") /
              "data");
  }

  std::cout << "All points transformed, now time to add new points"
            << std::endl;
  // Get new points (scaled transformation applied in ORB-SLAM3, so this
  // step is performed at last to avoid scaling twice)
  auto& associated_points = opr.associatedMapPoints();
  auto& points = std::get<0>(associated_points);
  auto& colors = std::get<1>(associated_points);

  // Add new points to the appropriate chunks
  if (initial_mapped_ && points.size() >= 30) {
    torch::NoGradGuard no_grad;
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Convert to tensors
    int num_new_points = static_cast<int>(points.size() / 3);
    torch::Tensor points_tensor =
        torch::from_blob(points.data(), {num_new_points, 3},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);
    torch::Tensor colors_tensor =
        torch::from_blob(colors.data(), {num_new_points, 3},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_type_);
    torch::Tensor opacities_tensor = general_utils::inverse_sigmoid(
        0.2f *
        torch::ones(
            {points_tensor.size(0), 1},
            torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    // Create a map of keyframes for the chunk manager
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> loop_keyframes;
    for (auto& kf : associated_kfs) {
      auto kfid = std::get<0>(kf);
      auto pkf = scene_->getKeyframe(kfid);
      if (pkf) {
        loop_keyframes[kfid] = pkf;
      }
    }

    std::cout << "Adding points to chunks" << std::endl;
    chunk_manager_->addPointsToChunks(points_tensor, colors_tensor,
                                      torch::Tensor(), opacities_tensor);
  }

  chunk_manager_->releaseAllChunksFromOptimization();

  // Gaussians will be all over the place, transfer them to their
  // respective chunks
  // std::cout << "Transferring gaussians across chunks" << std::endl;
  // chunk_manager_->transferGaussiansAcrossChunks();

  chunk_manager_->releaseAllChunksFromOptimization();

  // Mark this iteration
  loop_closure_iteration_ = true;
}

void GaussianMapper::processScaleRefinement(ORB_SLAM3::MappingOperation& opr) {
  // Existing scale refinement code...
  std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming "
               "all kfs and points..."
            << std::endl;

  float s = opr.mfScale;
  Sophus::SE3f& T = opr.mT;
  if (initial_mapped_) {
    // Apply the scaled transformation on ALL gaussian model points,
    // including those on disk
    {
      std::unique_lock<std::mutex> lock_render(mutex_render_);

      // Get all existing chunk coordinates (both in memory and on disk)
      std::vector<ChunkCoord> all_chunks =
          chunk_manager_->getExistingChunkCoords();

      std::cout << "Applying scale transformation to " << all_chunks.size()
                << " chunks" << std::endl;

      // Process chunks in batches to manage memory
      const int batch_size = 5;  // Adjust based on memory constraints
      for (size_t i = 0; i < all_chunks.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, all_chunks.size());

        // Process current batch
        for (size_t j = i; j < end; j++) {
          const auto& coord = all_chunks[j];
          if (chunk_manager_->loadChunkSync(coord, true)) {
            std::shared_ptr<Chunk> chunk = chunk_manager_->getChunkAt(coord);
            chunk->getGaussians()->applyScaledTransformation(s, T);
            chunk_manager_->releaseChunksFromOptimization({chunk});
            chunk_manager_->saveChunkAsync(coord);
          }
        }
      }
    }
    // Apply the scaled transformation to the scene
    scene_->applyScaledTransformation(s, T);
  } else {  // TODO: the workflow should not come here, delete this
            // branch
    // Apply the scaled transformation to the cached points
    for (auto& pt : scene_->cached_point_cloud_) {
      // pt <- (s * Ryw * pt + tyw)
      auto& pt_xyz = pt.second.xyz_;
      pt_xyz *= s;
      pt_xyz = T.cast<double>() * pt_xyz;
    }

    // Apply the scaled transformation on gaussian keyframes
    for (auto& kfit : scene_->keyframes()) {
      std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
      Sophus::SE3f Twc = pkf->getPosef().inverse();
      Twc.translation() *= s;
      Sophus::SE3f Tyc = T * Twc;
      Sophus::SE3f Tcy = Tyc.inverse();
      pkf->setPose(Tcy.unit_quaternion().cast<double>(),
                   Tcy.translation().cast<double>());
      pkf->computeTransformTensors();
    }
  }

  // Gaussians will be all over the place, transfer them to their
  // respective chunks
  chunk_manager_->transferGaussiansAcrossChunks();
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
        "[GaussianMapper::combineMappingOperations]KeyFrame Camera not "
        "found!");
  }
  // Add the new keyframe to the scene
  pkf->computeTransformTensors();
  scene_->addKeyframe(pkf);
  kfid_shuffled_ = false;
  keyframe_queue_->notifyNewKeyframeAdded(pkf);

  // Give new keyframes times of use and add it to the training sliding window
  increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

  // Get dense point cloud from the new keyframe to accelerate training
  pkf->img_undist_ = imgRGB_undistorted;
  pkf->img_auxiliary_undist_ = imgAux_undistorted;
  pkf->kps_pixel_ = std::move(std::get<6>(kf));
  pkf->kps_point_local_ = std::move(std::get<7>(kf));
  // if (isdoingInactiveGeoDensify())
  // increasePcdByKeyframeInactiveGeoDensify(pkf);

  pkf->initOptimizer(device_type_, opt_params_.pose_lr_,
                     opt_params_.exposure_lr_,
                     opt_params_.depth_scale_bias_lr_);

  // Prepare multi resolution images for training
  pkf->generatePyramidImages(device_type_);

  if (sensor_type_ == MONOCULAR) {
    pkf->setupMonoData(device_type_, monocular_depth_estimator_, min_depth_,
                       max_depth_);
  } else if (sensor_type_ == STEREO && !pkf->img_auxiliary_undist_.empty()) {
    pkf->setupStereoData(stereo_baseline_length_, device_type_,
                         stereo_depth_estimator_, min_depth_, max_depth_);
  } else if (sensor_type_ == RGBD && !pkf->img_auxiliary_undist_.empty()) {
    // Preprocess and store depth image tensor
    if (device_type_ == torch::kCUDA) {
      cv::cuda::GpuMat depth_gpu;
      depth_gpu.upload(pkf->img_auxiliary_undist_);
      pkf->depth_image_ = tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
    } else {
      pkf->depth_image_ = tensor_utils::cvMat2TorchTensor_Float32(
          pkf->img_auxiliary_undist_, device_type_);
    }
    if (do_gaus_pyramid_training_) {
      pkf->generatePyramidDepth(device_type_, pkf->img_auxiliary_undist_);
    }
  }

  // Convert tensors to cv::Mat for processing
  torch::Tensor rgb_image =
      tensor_utils::cvMat2TorchTensor_Float32(pkf->img_undist_, device_type_);

  // Get camera pose (world-to-camera)
  Sophus::SE3f Tcw = pkf->getPosef();

  // Project to point cloud
  // std::string pcd_path = "depth_pcd_kf.ply";
  // projectRgbDepthToPointCloud(rgb_image, pkf->depth_image_, pkf->intr_,
  //                             min_depth_, max_depth_, Tcw, pcd_path, 2);

  if (isdoingDepthDensify()) increasePcdByDepthReconstruction(pkf);
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

      torch::Tensor opacities_tensor = general_utils::inverse_sigmoid(
          0.02f *
          torch::ones({points3D_valid.size(0), 1}, torch::TensorOptions()
                                                       .dtype(torch::kFloat)
                                                       .device(device_type_)));

      std::unique_lock<std::mutex> lock_render(mutex_render_);
      addPoints(points3D_valid, colors_valid, torch::Tensor(),
                opacities_tensor);
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

      torch::Tensor opacities_tensor = general_utils::inverse_sigmoid(
          0.02f *
          torch::ones({points3D_valid.size(0), 1}, torch::TensorOptions()
                                                       .dtype(torch::kFloat)
                                                       .device(device_type_)));

      std::unique_lock<std::mutex> lock_render(mutex_render_);
      addPoints(points3D_valid, colors_valid, torch::Tensor(),
                opacities_tensor);
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
          point_valid_flags, torch::where(depth > min_depth_, true, false));
      point_valid_flags = torch::logical_and(
          point_valid_flags, torch::where(depth < max_depth_, true, false));

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
              "[Gaussian Mapper]Fisheye cameras are not supported "
              "currently!");
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

      torch::Tensor opacities_tensor = general_utils::inverse_sigmoid(
          0.02f *
          torch::ones({points3D_valid.size(0), 1}, torch::TensorOptions()
                                                       .dtype(torch::kFloat)
                                                       .device(device_type_)));

      std::unique_lock<std::mutex> lock_render(mutex_render_);
      addPoints(points3D_valid, colors_valid, torch::Tensor(),
                opacities_tensor);
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }
}

void GaussianMapper::increasePcdByDepthReconstruction(
    std::shared_ptr<GaussianKeyframe> pkf) {
  // auto start_timing = std::chrono::steady_clock::now();
  torch::NoGradGuard no_grad;

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  // Step 1: Get RGB image and depth data
  cv::cuda::GpuMat rgb_gpu;
  rgb_gpu.upload(pkf->img_undist_);
  torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);

  torch::Tensor depth;
  bool has_depth = false;

  switch (this->sensor_type_) {
    case MONOCULAR:
    case STEREO: {
      if (pkf->depth_image_.defined()) {
        depth = pkf->depth_image_;
        has_depth = true;
      }
    } break;
    case RGBD: {
      if (!pkf->img_auxiliary_undist_.empty()) {
        cv::Mat depth_cleaned = pkf->img_auxiliary_undist_.clone();
        cv::patchNaNs(depth_cleaned, 0.0);
        cv::threshold(depth_cleaned, depth_cleaned, max_depth_, max_depth_,
                      cv::THRESH_TRUNC);

        if (device_type_ == torch::kCUDA) {
          cv::cuda::GpuMat depth_gpu;
          depth_gpu.upload(depth_cleaned);
          depth = tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
          has_depth = true;
        }
      }
    } break;
  }

  // Step 2: Compute initial probability based on image gradients (like Python)
  torch::Tensor prob_L = computeLoGProbability(rgb);

  // Step 3: Render current view and compute penalty (if scene is initialized)
  torch::Tensor prob_penalty = torch::zeros_like(prob_L);
  torch::Tensor rendered_depth;
  torch::Tensor main_gaussian_ids;
  std::vector<std::shared_ptr<GaussianModel>> models;
  std::vector<std::shared_ptr<Chunk>> visible_chunks;
  std::vector<int> model_sizes;
  bool has_rendered_depth = false;

  if (initial_mapped_) {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    visible_chunks = chunk_manager_->loadVisibleChunks(pkf, true);

    if (!visible_chunks.empty()) {
      for (const auto& chunk : visible_chunks) {
        if (chunk && chunk->getGaussians() &&
            chunk->getGaussians()->getXYZ().sizes()[0] > 0) {
          models.push_back(chunk->getGaussians());
          model_sizes.push_back(chunk->getGaussians()->getXYZ().sizes()[0]);
        }
      }

      if (!models.empty()) {
        torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
        auto render_pkg = GaussianRenderer::render(
            models, pkf, pkf->image_height_, pkf->image_width_, pipe_params_,
            background_, override_color_, 1.0f, false, pkf->FoVx_, pkf->FoVy_,
            view_matrix, pkf->projection_matrix_);

        torch::Tensor rendered_image = std::get<1>(render_pkg);
        rendered_depth = std::get<0>(render_pkg);
        has_rendered_depth = true;
        main_gaussian_ids = std::get<4>(render_pkg)[0];
        prob_penalty = computeLoGProbability(rendered_image);
      }
    }
  }

  // Step 4: Apply scaling factor and compute final probability
  prob_L *= init_proba_scaler_;
  prob_penalty *= init_proba_scaler_;
  torch::Tensor prob_s = torch::clamp(prob_L - prob_penalty, 0.0f, 1.0f);

  // Debug: Save probability visualizations
  std::filesystem::create_directories("/workspace/repo/debug_prob");
  auto save_tensor = [](const torch::Tensor& t, const std::string& name) {
    torch::Tensor cpu_t = t.detach().cpu().to(torch::kFloat);
    if (cpu_t.dim() == 4)
      cpu_t = cpu_t[0][0];
    else if (cpu_t.dim() == 3 && cpu_t.size(0) == 1)
      cpu_t = cpu_t[0];
    cpu_t = torch::clamp(cpu_t, 0.0f, 1.0f);

    int h = cpu_t.size(0), w = cpu_t.size(1);
    cv::Mat mat(h, w, CV_32F, cpu_t.data_ptr<float>());
    cv::Mat img_8bit, colored;
    mat.convertTo(img_8bit, CV_8U, 255.0);
    cv::applyColorMap(img_8bit, colored, cv::COLORMAP_JET);
    cv::imwrite("/workspace/repo/debug_prob/" + name + ".png", colored);
  };

  // save_tensor(prob_L, "prob_L");
  // save_tensor(prob_penalty, "prob_penalty");
  // save_tensor(prob_s, "prob_s");

  torch::Tensor sample_mask = torch::rand_like(prob_s) < prob_s;

  // 5a: DEPTH-BASED POINTS
  if (has_depth) {
    torch::Tensor valid_depth = (depth >= min_depth_) & (depth <= max_depth_);
    sample_mask = sample_mask & valid_depth;

    // Handle occlusions
    if (has_rendered_depth && !models.empty()) {
      sample_mask = sample_mask.flatten();
      torch::Tensor depth_flat = depth.flatten();
      torch::Tensor rendered_depth_flat = rendered_depth.flatten();

      torch::Tensor accurate_sample_mask = sample_mask.clone();

      // Remove coarser gaussians based on accurate samples (like Python)
      if (accurate_sample_mask.any().item<bool>()) {
        torch::Tensor selected_main_gaussians =
            main_gaussian_ids.flatten().index({accurate_sample_mask});
        torch::Tensor valid_ids_mask = selected_main_gaussians >= 0;

        if (valid_ids_mask.any().item<bool>()) {
          selected_main_gaussians =
              selected_main_gaussians.index({valid_ids_mask});

          auto unique_result = torch::_unique2(selected_main_gaussians,
                                               /*sorted=*/false,
                                               /*return_inverse=*/false,
                                               /*return_counts=*/true);
          torch::Tensor unique_ids = std::get<0>(unique_result);
          torch::Tensor counts = std::get<2>(unique_result);

          const int max_replacement_count = 10;
          torch::Tensor removal_mask = counts >= max_replacement_count;

          if (removal_mask.any().item<bool>()) {
            torch::Tensor gaussians_to_remove =
                unique_ids.index({removal_mask});
            createAndApplyGlobalRemovalMask(gaussians_to_remove, models,
                                            model_sizes);

            // Recompute visible chunks after removal
            std::vector<std::shared_ptr<GaussianModel>> pruned_models;
            if (!visible_chunks.empty()) {
              for (const auto& chunk : visible_chunks) {
                if (chunk && chunk->getGaussians() &&
                    chunk->getGaussians()->getXYZ().sizes()[0] > 0) {
                  pruned_models.push_back(chunk->getGaussians());
                }
              }

              if (!pruned_models.empty()) {
                // Re-render after gaussian removal
                torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
                auto updated_render_pkg = GaussianRenderer::render(
                    pruned_models, pkf, pkf->image_height_, pkf->image_width_,
                    pipe_params_, background_, override_color_, 1.0f, false,
                    pkf->FoVx_, pkf->FoVy_, view_matrix,
                    pkf->projection_matrix_);

                rendered_depth = std::get<0>(updated_render_pkg);
                rendered_depth_flat = rendered_depth.flatten();
              }
            }
          }
        }
      }

      // Check for occlusions
      torch::Tensor occlusion_mask = depth_flat < rendered_depth_flat;
      sample_mask = sample_mask & occlusion_mask;
    } else {
      sample_mask = sample_mask.flatten();
    }
  } else {
    sample_mask = sample_mask.flatten();
  }

  chunk_manager_->releaseChunksFromOptimization(visible_chunks);

  if (!sample_mask.any().item<bool>()) return;

  depth = depth.flatten();
  rgb = rgb.permute({1, 2, 0}).flatten(0, 1);
  prob_L = prob_L.flatten();

  // Get sampled data
  torch::Tensor sampled_colors = rgb.index({sample_mask});
  torch::Tensor sampled_depths = depth.index({sample_mask});
  torch::Tensor sampled_init_proba = prob_L.index({sample_mask});

  // Reproject to 3D
  torch::Tensor points3D =
      reprojectDepthPinhole(depth, sample_mask, pkf->intr_, pkf->image_width_);
  points3D = points3D.index({sample_mask});

  // Transform to world coordinates
  torch::Tensor Twc_tensor =
      tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(points3D, Twc_tensor);

  // Compute scales (like Python implementation)
  torch::Tensor scales = 1.0f / torch::sqrt(sampled_init_proba + 1e-8f);
  scales =
      torch::clamp(scales, 1.0f, static_cast<float>(pkf->image_width_) / 10.0f);
  scales *= (1.0f / pkf->intr_[0]);  // fx

  torch::Tensor diff = points3D - pkf->camera_center_.unsqueeze(0);
  torch::Tensor distances = torch::norm(diff, 2, 1);
  scales *= distances;
  scales = torch::log(torch::clamp(scales, 1e-6f, 1e6f));
  torch::Tensor sampled_scales = scales.unsqueeze(1).repeat({1, 3});

  // // Compute opacities based on depth (closer = higher opacity)
  // float min_opacity = 0.02f;  // For furthest points
  // float max_opacity = 0.2f;   // For closest points

  // // Map depth range [min_depth_, max_depth_] to opacity range [max_opacity,
  // // min_opacity]
  // torch::Tensor normalized_depths =
  //     (sampled_depths - min_depth_) / (max_depth_ - min_depth_);
  // normalized_depths = torch::clamp(normalized_depths, 0.0f, 1.0f);

  // // Linear interpolation: opacity = max_opacity - (max_opacity -
  // min_opacity) *
  // // normalized_depth This gives max_opacity for min_depth_ (closest) and
  // // min_opacity for max_depth_ (furthest)
  // const float opacity_range = max_opacity - min_opacity;
  // torch::Tensor opacities = max_opacity - opacity_range * normalized_depths;
  // opacities =
  //     opacities.unsqueeze(1);  // Add dimension to match expected shape [N,
  //     1]

  // Compute opacities (like Python - lower for inaccurate points)
  torch::Tensor opacities =
      torch::full({points3D.size(0), 1}, 0.07f,
                  torch::TensorOptions().device(device_type_));

  std::cout << "New points from depth size: " << points3D.sizes() << std::endl;

  std::unique_lock lock_render(mutex_render_);
  addPoints(points3D, sampled_colors, sampled_scales, opacities);
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
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->loadVisibleChunks(pkf, false);
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    // std::cout << "[" << chunk->getCoord().x << " " << chunk->getCoord().y
    // << " "
    //           << chunk->getCoord().z << "], ";
    if (chunk && chunk->getGaussians() &&
        chunk->getGaussians()->getXYZ().sizes()[0] > 0) {
      models.push_back(chunk->getGaussians());
    } else {
      throw "[renderFromPose] Chunk/Gaussian not valid";
    }
  }
  // std::cout << std::endl;

  // auto active_chunks = chunk_manager_->getActiveChunks();
  // std::vector<std::shared_ptr<GaussianModel>> models;
  // models.reserve(active_chunks.size());
  // for (const auto& [coord, chunk] : active_chunks) {
  //   if (chunk && chunk->getGaussians()) {
  //     models.push_back(chunk->getGaussians());
  //   } else {
  //     throw std::runtime_error("[renderFromPose] Chunk/Gaussian not
  //     valid");
  //   }
  // }

  // Check if we have any valid models to render
  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    chunk_manager_->releaseChunksFromOptimization(visible_chunks);
    cv::Mat black_image = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat empty_depth = cv::Mat::zeros(height, width, CV_32FC1);
    return std::make_tuple(black_image,
                           empty_depth);  // Early return if no valid models
  }

  // Render
  torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
  auto render_pkg = GaussianRenderer::render(
      models, pkf, height, width, pipe_params_, background_, override_color_,
      1.0f, false, pkf->FoVx_, pkf->FoVy_, view_matrix,
      pkf->projection_matrix_);

  // Return rendered image and depth
  chunk_manager_->releaseChunksFromOptimization(visible_chunks);
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

  // Get visible chunks using ChunkManager instead of updateActiveChunks
  std::vector<std::shared_ptr<Chunk>> visible_chunks =
      chunk_manager_->loadVisibleChunks(pkf, false);

  // Extract models from chunks
  std::vector<std::shared_ptr<GaussianModel>> models;
  models.reserve(visible_chunks.size());
  for (const auto& chunk : visible_chunks) {
    if (chunk && chunk->getGaussians() &&
        chunk->getGaussians()->getXYZ().sizes()[0] > 0) {
      models.push_back(chunk->getGaussians());
    }
  }

  if (models.empty()) {
    std::cout << "[renderFromPose] No valid models to render" << std::endl;
    chunk_manager_->releaseChunksFromOptimization(visible_chunks);
    return;  // Early return if no valid models
  }

  torch::Tensor view_matrix = pkf->getRT().transpose(0, 1);
  auto render_pkg = GaussianRenderer::render(
      models, pkf, pkf->image_height_, pkf->image_width_, pipe_params_,
      background_, override_color_, 1.0f, false, pkf->FoVx_, pkf->FoVy_,
      view_matrix, pkf->projection_matrix_);

  chunk_manager_->releaseChunksFromOptimization(visible_chunks);
  auto rendered_image = std::get<1>(render_pkg);
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
  for (const auto& used_times_it : keyframe_queue_->getKfsUsedTimes()) {
    out_stream
        << used_times_it.first << " " << used_times_it.second << " "
        << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
        << "\n";
  }
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
float GaussianMapper::lambdaDepth() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_depth_;
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
bool GaussianMapper::isdoingDepthDensify() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return depth_densify_;
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
  // this->getGaussians()->loadPly(ply_path);

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
  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  // Process frames until we have enough keyframes
  while (!initial_mapped_ && !isStopped() && !isExternalDataStopped()) {
    auto maybe_frame = frame_queue_.pop(true);
    if (!maybe_frame) continue;

    auto& frame = *maybe_frame;
    processNewFrame(frame.rgb_image, frame.depth_image, frame.pose,
                    frame.timestamp);

    if (scene_->keyframes().size() >= min_num_initial_map_kfs_) {
      std::cout << "Initializing with " << scene_->keyframes().size()
                << " keyframes" << std::endl;
      initial_mapped_ = true;
      scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
      break;
    }
  }

  if (!initial_mapped_) {
    std::cout << "Failed to initialize, stopping" << std::endl;
    return;
  }

  int SLAM_stop_iter = 0;
  // Start training loop while still processing new frames
  std::cout << "Starting training loop" << std::endl;
  while (!isExternalDataStopped() && !isStopped()) {
    // Process any pending frames
    while (auto maybe_frame = frame_queue_.pop(false)) {
      processNewFrame(maybe_frame->rgb_image, maybe_frame->depth_image,
                      maybe_frame->pose, maybe_frame->timestamp);
    }

    trainForOneIteration();
    SLAM_stop_iter = getIteration();

    // std::cout << "Loop check: isExternalDataStopped()="
    //           << (isExternalDataStopped() ? "true" : "false")
    //           << ", isStopped()=" << (isStopped() ? "true" : "false")
    //           << std::endl;
  }

  std::cout << "Starting post-training loop" << std::endl;
  while (!isExternalDataStopped() && !isStopped()) {
    // std::cout << "Loop check: isExternalDataStopped()="
    //           << (isExternalDataStopped() ? "true" : "false")
    //           << ", isStopped()=" << (isStopped() ? "true" : "false")
    //           << std::endl;
    trainForOneIteration();
    if (getIteration() >= opt_params_.iterations_) break;
  }

  // std::cout << "Starting Tail gaussian optimization" << std::endl;
  // Fourth loop: Tail gaussian optimization
  int densify_interval = densifyInterval();
  int n_delay_iters = densify_interval * 0.8;
  std::cout << "Starting tail optimization loop" << std::endl;
  while (getIteration() - SLAM_stop_iter < n_delay_iters ||
         getIteration() % densify_interval < n_delay_iters) {
    trainForOneIteration();
  }

  // std::cout << "Training finished" << std::endl;

  frame_queue_.stop();

  if (render_fly_through_) {
    auto video_dir = result_dir_ / "flythrough";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(video_dir)
    renderFlyThroughVideo(video_dir / "output_video", 1920, 1080, 30,
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
  // savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
  // "ply");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  signalStop();
  if (completion_callback_) {
    completion_callback_();
  }
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
  chunk_manager_ = std::make_shared<ChunkManager>(model_params_, opt_params_,
                                                  chunk_save_dir_, chunk_size_,
                                                  max_chunks_in_memory_);
}

void GaussianMapper::addPoints(const torch::Tensor& points,
                               const torch::Tensor& colors,
                               const torch::Tensor& scales,
                               const torch::Tensor& opacities) {
  // std::cout << "addPoints called in GaussianMapper" << std::endl;
  // Make sure chunk manager has current iteration
  if (!chunk_manager_) {
    throw std::runtime_error("chunk_manager_ is null");
  }
  chunk_manager_->setCurrentIteration(getIteration());

  // std::cout << "Scene cameras extent: " << scene_->cameras_extent_ <<
  // std::endl;

  // Delegate to chunk manager
  chunk_manager_->addPointsToChunks(points, colors, scales, opacities);
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
  saveModelParams(scene_dir);

  // Save a manifest of all chunks on disk
  saveChunkManifest(scene_dir);

  // Save config used to train the model
  try {
    std::filesystem::copy_file(
        config_file_path_, scene_dir / "gaussian_mapper_cfg.yaml",
        std::filesystem::copy_options::overwrite_existing);
    std::cout << "Config saved successfully" << std::endl;
  } catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // Save all active chunks
  chunk_manager_->releaseAllChunksFromOptimization();
  auto active_chunks = chunk_manager_->getActiveChunks();
  bool all_saved = true;

  std::cout << active_chunks.size() << " active chunks to save" << std::endl;

  for (const auto& [coord, chunk] : active_chunks) {
    if (chunk_manager_->getChunkState(coord) == ChunkState::ACTIVE) {
      if (!chunk_manager_->saveChunkSync(coord)) {
        throw std::runtime_error(
            "Failed to save chunk: " + std::to_string(coord.x) + "," +
            std::to_string(coord.y) + "," + std::to_string(coord.z));
        all_saved = false;
      }
    } else {
      throw std::runtime_error(
          "Chunk state is not ACTIVE, cannot save chunk: " +
          std::to_string(coord.x) + "," + std::to_string(coord.y) + "," +
          std::to_string(coord.z));
    }
  }

  // Copy chunks over to scene dir
  std::filesystem::path scene_chunk_dir = scene_dir / "chunks";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(scene_chunk_dir);
  copyFolder(chunk_save_dir_, scene_dir / "chunks");

  std::cout << "Scene saved to " << scene_dir << std::endl;
  return all_saved;
}

// Implementation for loadScene in gaussian_mapper.cpp
bool GaussianMapper::loadScene(std::filesystem::path scene_dir,
                               std::filesystem::path optional_camera_path) {
  if (!std::filesystem::exists(scene_dir)) {
    throw std::runtime_error("Scene directory does not exist: " +
                             scene_dir.string());
  }

  // // Load camera parameters
  loadCamerasFromJson(scene_dir / "cameras.json");

  // Load chunk information from the manifest
  loadChunkManifest(scene_dir);

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

  // Get chunks from the chunk manager's cache
  std::vector<ChunkCoord> chunk_coords =
      chunk_manager_->getExistingChunkCoords();

  // Write chunk coordinates to JSON
  for (size_t i = 0; i < chunk_coords.size(); i++) {
    Json::Value chunk_entry;
    // Use explicit casts to Json::Value::Int64
    chunk_entry["x"] = Json::Value::Int64(chunk_coords[i].x);
    chunk_entry["y"] = Json::Value::Int64(chunk_coords[i].y);
    chunk_entry["z"] = Json::Value::Int64(chunk_coords[i].z);

    std::vector<ChunkCoord> all_chunks =
        chunk_manager_->getExistingChunkCoords();
    // std::cout << all_chunks.size() << " chunks during saveChunkManifest"
    //           << std::endl;
    bool success = chunk_manager_->loadChunkSync(chunk_coords[i], true, false);
    if (!success) continue;

    auto chunk = chunk_manager_->getChunkAt(chunk_coords[i]);
    if (!chunk || !chunk->getGaussians()) continue;

    chunk_entry["num_gaussians"] =
        Json::Value::Int64(chunk->getGaussians()->getXYZ().size(0));
    chunk_entry["local_iteration"] =
        Json::Value::Int64(chunk->getGaussians()->getLocalIteration());
    chunk_entry["sh_degree"] =
        Json::Value::Int64(chunk->getGaussians()->sh_degree_);

    chunk_manager_->releaseChunksFromOptimization({chunk_coords[i]});
    chunk_manager_->saveChunkSync(chunk_coords[i]);

    json_root[static_cast<int>(i)] = chunk_entry;
  }

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

  // Process each chunk entry
  for (const auto& chunk_entry : root) {
    int64_t x = chunk_entry["x"].asInt64();
    int64_t y = chunk_entry["y"].asInt64();
    int64_t z = chunk_entry["z"].asInt64();

    ChunkCoord coord{x, y, z};
    chunk_manager_->updateChunkExistenceCache({coord}, true);
    chunk_manager_->initializeMetaData(coord);
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
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(fid, getIteration());

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
    keyframe_queue_->notifyNewKeyframeAdded(pkf);

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
  int totalGaussians = 0;

  // Get all existing chunk coordinates (both in memory and on disk)
  std::vector<ChunkCoord> allChunkCoords =
      chunk_manager_->getExistingChunkCoords();

  for (size_t i = 0; i < allChunkCoords.size(); i++) {
    ChunkCoord coord = allChunkCoords[i];
    if (!chunk_manager_->loadChunkSync(coord, true, false)) {
      std::cout << "Skipping chunk, can't load" << std::endl;
      chunk_manager_->releaseChunksFromOptimization({coord});
      chunk_manager_->triggerLruCheck();
      continue;
    }

    std::shared_ptr<Chunk> chunk = chunk_manager_->getChunkAt(coord);

    if (!chunk || !chunk->getGaussians()) {
      throw std::runtime_error("Gaussians/Chunk invalid");
      chunk_manager_->releaseChunksFromOptimization({coord});
      chunk_manager_->triggerLruCheck();
      continue;
    }

    auto gaussians = chunk->getGaussians();
    auto num_points = gaussians->getXYZ().size(0);
    totalGaussians += num_points;

    chunk_manager_->releaseChunksFromOptimization({coord});
    chunk_manager_->triggerLruCheck();
  }

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

void GaussianMapper::processNewFrame(const cv::Mat& rgb_image,
                                     const cv::Mat& depth_or_right_image,
                                     const Sophus::SE3f& pose,
                                     const double timestamp) {
  // std::cout << "\n[ProcessFrame] Starting..." << std::endl;
  // size_t free_mem, total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "[ProcessFrame] Starting CUDA Memory - Free: "
  //           << free_mem / 1024 / 1024
  //           << "MB, Total: " << total_mem / 1024 / 1024 << "MB" <<
  //           std::endl;

  try {
    // First, update external data without holding the main lock
    // std::cout << "[ProcessFrame] Updating external data..." << std::endl;
    setRecentExternalData(rgb_image, pose);

    // Check if this should be a keyframe - use a separate short lock
    bool should_create_keyframe = false;
    {
      // std::cout << "[ProcessFrame] Acquiring lock for keyframe check..."
      //           << std::endl;
      std::unique_lock<std::mutex> lock(mutex_new_frame_);
      should_create_keyframe = isKeyframe(pose, timestamp);
      // std::cout << "[ProcessFrame] Should create keyframe: "
      //           << should_create_keyframe << std::endl;
    }

    if (!should_create_keyframe) {
      // std::cout << "[ProcessFrame] Not a keyframe, returning" << std::endl;
      return;
    }

    // Prepare the new keyframe without holding the lock
    // std::cout << "[ProcessFrame] Creating new keyframe..." << std::endl;
    std::shared_ptr<GaussianKeyframe> new_kf =
        std::make_shared<GaussianKeyframe>(scene_->keyframes().size(),
                                           getIteration());
    // std::cout << "New kf. fid: " << new_kf->fid_ << std::endl;

    new_kf->zfar_ = z_far_;
    new_kf->znear_ = z_near_;

    // std::cout << "new_kf->zfar_" << new_kf->zfar_ << std::endl;
    // std::cout << "new_kf->znear_" << new_kf->znear_ << std::endl;

    // Set pose
    // std::cout << "[ProcessFrame] Setting pose..." << std::endl;
    new_kf->setPose(pose.unit_quaternion().cast<double>(),
                    pose.translation().cast<double>());

    // Get camera parameters - brief lock
    Camera camera;
    {
      // std::cout << "[ProcessFrame] Getting camera parameters..." <<
      // std::endl;
      std::unique_lock<std::mutex> lock(mutex_new_frame_);
      camera = scene_->cameras_.at(0);
    }
    new_kf->setCameraParams(camera);

    // Process images - no lock needed
    // std::cout << "[ProcessFrame] Processing images..." << std::endl;

    // std::cout << "[ProcessFrame] Image check - RGB size: " <<
    // rgb_image.size()
    //           << ", type: " << rgb_image.type()
    //           << ", empty: " << rgb_image.empty() << std::endl;

    // {
    //   // Save input RGB image
    //   cv::Mat save_rgb = rgb_image.clone();
    //   if (save_rgb.type() == CV_32FC3) {
    //     save_rgb.convertTo(save_rgb, CV_8UC3, 255.0);
    //   }
    //   cv::cvtColor(save_rgb, save_rgb, cv::COLOR_RGB2BGR);
    //   cv::imwrite("/workspaces/large_scale_gaussian_slam/debug_input_rgb.png",
    //               save_rgb);
    //   std::cout << "Saved input RGB image to "
    //                "/workspaces/large_scale_gaussian_slam/debug_input_rgb.png"
    //             << std::endl;
    // }

    cv::Mat rgb_undistorted = rgb_image;
    // std::cout << "[ProcessFrame] Undistorted copy created" << std::endl;

    try {
      // std::cout << "[ProcessFrame] Starting tensor conversion..." <<
      // std::endl; std::cout << "[ProcessFrame] Device type: " <<
      // device_type_
      // << std::endl;

      new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
          rgb_undistorted, device_type_);
      new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
      new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
      new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

      // {
      //   // Save PyTorch tensor image
      //   torch::Tensor cpu_tensor = new_kf->original_image_.cpu().clone();
      //   if (cpu_tensor.dim() == 3 && cpu_tensor.size(0) == 3) {
      //     cpu_tensor = cpu_tensor.permute({1, 2, 0}).contiguous();
      //   }
      //   cv::Mat tensor_img(cpu_tensor.size(0), cpu_tensor.size(1),
      //   CV_32FC3); std::memcpy(tensor_img.data,
      //   cpu_tensor.data_ptr<float>(),
      //               sizeof(float) * tensor_img.rows * tensor_img.cols * 3);

      //   tensor_img.convertTo(tensor_img, CV_8UC3, 255.0);
      //   cv::cvtColor(tensor_img, tensor_img, cv::COLOR_RGB2BGR);
      //   cv::imwrite(
      //       "/workspaces/large_scale_gaussian_slam/debug_tensor_image.png",
      //       tensor_img);
      //   std::cout
      //       << "Saved tensor image to "
      //          "/workspaces/large_scale_gaussian_slam/debug_tensor_image.png"
      //       << std::endl;
      // }

    } catch (const std::exception& e) {
      std::cerr << "[ProcessFrame] Exception in tensor conversion: " << e.what()
                << std::endl;
      throw;
    } catch (...) {
      std::cerr << "[ProcessFrame] Unknown exception in tensor conversion"
                << std::endl;
      throw;
    }

    // std::cout << "[ProcessFrame] Setting undistorted image" << std::endl;
    new_kf->img_undist_ = rgb_undistorted;

    // std::cout << "[ProcessFrame] Setting auxiliary image..." << std::endl;
    if (sensor_type_ == STEREO) {
      new_kf->img_auxiliary_undist_ = depth_or_right_image;
    } else if (sensor_type_ == RGBD) {
      new_kf->img_auxiliary_undist_ = depth_or_right_image;
    } else {
      throw std::runtime_error(
          "[GaussianMapper] Unsupported sensor type for auxiliary image");
    }

    // Compute transforms - no lock needed
    // std::cout << "[ProcessFrame] Computing transforms..." << std::endl;
    try {
      new_kf->computeTransformTensors();
    } catch (const std::exception& e) {
      std::cerr << "[ProcessFrame] Exception in transform computation: "
                << e.what() << std::endl;
      throw;
    }

    // Now take the lock only for the critical section of adding to scene
    {
      // std::cout << "[ProcessFrame] Adding keyframe to scene..." <<
      // std::endl;
      std::unique_lock<std::mutex> lock(mutex_new_frame_);

      // // Double-check in case another thread created a keyframe while we
      // were
      // // preparing
      // if (!isKeyframe(pose, timestamp)) {
      //   std::cout << "[ProcessFrame] Keyframe no longer needed after "
      //                "preparation, returning"
      //             << std::endl;
      //   return;
      // }

      increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

      scene_->addKeyframe(new_kf);
      new_kf->initOptimizer(device_type_, opt_params_.pose_lr_,
                            opt_params_.exposure_lr_,
                            opt_params_.depth_scale_bias_lr_);
      kfid_shuffled_ = false;

      // Update tracking info
      last_keyframe_pose_ = pose;
      last_keyframe_timestamp_ = timestamp;
    }

    // Generate point cloud after releasing the lock
    // std::cout << "[ProcessFrame] Generating point cloud..." << std::endl;

    // Prepare multi resolution images for training
    new_kf->generatePyramidImages(device_type_);

    if (sensor_type_ == MONOCULAR) {
      new_kf->setupMonoData(device_type_, monocular_depth_estimator_,
                            min_depth_, max_depth_);
    } else if (sensor_type_ == STEREO && !depth_or_right_image.empty()) {
      new_kf->setupStereoData(stereo_baseline_length_, device_type_,
                              stereo_depth_estimator_, min_depth_, max_depth_);
    }

    if (sensor_type_ == RGBD && !new_kf->img_auxiliary_undist_.empty()) {
      // Create a clean version of the depth map
      cv::Mat depth = new_kf->img_auxiliary_undist_.clone();

      // First replace NaN values with 0 (or some invalid depth marker)
      cv::patchNaNs(depth, 0.0);

      cv::Mat min_depth_mask, max_depth_mask;
      cv::threshold(depth, min_depth_mask, min_depth_, 1.0, cv::THRESH_BINARY);
      cv::threshold(depth, max_depth_mask, max_depth_, 1.0,
                    cv::THRESH_BINARY_INV);
      cv::Mat valid_mask = (depth > min_depth_);

      // cv::Mat combined_mask;
      // cv::multiply(max_depth_mask, min_depth_mask, combined_mask);
      // cv::multiply(depth, combined_mask, depth);

      // Now threshold to handle infinity values
      // cv::threshold(depth_cleaned, depth_cleaned, max_depth_, max_depth_,
      //               cv::THRESH_TRUNC);

      // // Finally, create a valid mask to exclude zeros from later
      // computations cv::Mat valid_mask = (depth_cleaned > min_depth_);

      // double min_val, max_val;
      // cv::minMaxLoc(depth_cleaned, &min_val, &max_val);
      // cv::Scalar mean = cv::mean(depth_cleaned, valid_mask);

      // // Get camera pose (world-to-camera)
      // Sophus::SE3f Tcw = new_kf->getPosef();

      // std::string render_filename =
      // "/workspace/repo/rgbd_predicted_depth.png";
      // colorize_and_save_depth(merged_depth.detach().cpu(), render_filename,
      //                         merged_depth.min().item<float>(),
      //                         merged_depth.max().item<float>());

      // // Project to point cloud
      // std::string pcd_path = "/workspace/repo/depth_pcd.ply";
      // torch::Tensor rgb_torch =
      //     tensor_utils::cvMat2TorchTensor_Float32(rgb_image, torch::kCUDA);
      // projectRgbDepthToPointCloud(rgb_torch, merged_depth, new_kf->intr_,
      //                             min_depth_, max_depth_, Tcw, pcd_path,
      //                             2);

      // std::cout << "Cleaned depth matrix - type: " << depth_cleaned.type()
      //           << ", min: " << min_val << ", max: " << max_val
      //           << ", mean: " << mean[0] << std::endl;

      // Preprocess and store right image tensor
      if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat depth_gpu;
        depth_gpu.upload(depth);
        new_kf->depth_image_ =
            tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
        // std::cout << "GT Depth statistics: min "
        //           << new_kf->depth_image_.min().item<float>() << " max "
        //           << new_kf->depth_image_.max().item<float>() << " mean"
        //           << new_kf->depth_image_.mean().item<float>() << " median
        //           "
        //           << new_kf->depth_image_.median().item<float>() <<
        //           std::endl;
      } else {
        throw std::runtime_error(
            "[GaussianMapper] RGBD mode only supported on CUDA for now");
      }

      if (do_gaus_pyramid_training_) {
        new_kf->generatePyramidDepth(device_type_, depth);
      }
    }

    assert(isdoingDepthDensify() &&
           "Depth densification needs to be enabled for external mode!");
    increasePcdByDepthReconstruction(new_kf);

    keyframe_queue_->notifyNewKeyframeAdded(new_kf);

    // std::cout << "[ProcessFrame] Successfully completed" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[ProcessFrame] Critical exception in process frame: "
              << e.what() << std::endl;
    throw;  // Re-throw after logging
  }
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
void GaussianMapper::visualizeDepthReconstruction(
    std::shared_ptr<GaussianKeyframe> pkf,
    const torch::Tensor& points3D,
    const torch::Tensor& valid_points,
    const std::string& save_path) {
  // Step 1: Get the original image dimensions
  int height = pkf->img_undist_.rows;
  int width = pkf->img_undist_.cols;

  // Step 2: Create empty depth map tensor (initialize with zero values)
  torch::Tensor depth_map = torch::zeros(
      {height, width},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  // Step 3: Project 3D points back to 2D using camera parameters
  torch::Tensor camera_points = points3D.clone();

  // Get camera intrinsics
  float fx = pkf->intr_[0];
  float fy = pkf->intr_[1];
  float cx = pkf->intr_[2];
  float cy = pkf->intr_[3];

  // Calculate pixel coordinates and depths
  torch::Tensor z = camera_points.index({torch::indexing::Slice(), 2});
  torch::Tensor x = camera_points.index({torch::indexing::Slice(), 0});
  torch::Tensor y = camera_points.index({torch::indexing::Slice(), 1});

  // Project to pixel coordinates: u = fx * x / z + cx, v = fy * y / z + cy
  torch::Tensor u = fx * x.div(z) + cx;
  torch::Tensor v = fy * y.div(z) + cy;

  // Convert to integer pixel coordinates
  torch::Tensor u_int = u.round().to(torch::kInt64);
  torch::Tensor v_int = v.round().to(torch::kInt64);

  // Step 4: Filter out points outside the image boundaries
  torch::Tensor in_bounds = (u_int >= 0) & (u_int < width) & (v_int >= 0) &
                            (v_int < height) & (z > 0);

  // Apply the valid_points mask (from our filtering steps)
  in_bounds = in_bounds & valid_points;

  // Get points that are in bounds
  torch::Tensor valid_u = u_int.index({in_bounds});
  torch::Tensor valid_v = v_int.index({in_bounds});
  torch::Tensor valid_z = z.index({in_bounds});

  // Step 5: Create depth map by filling in valid depth values
  // Note: This might have conflicts where multiple 3D points project to same
  // pixel In that case, we take the closest point (minimum z value)

  // Convert to CPU for processing
  torch::Tensor valid_u_cpu = valid_u.to(torch::kCPU);
  torch::Tensor valid_v_cpu = valid_v.to(torch::kCPU);
  torch::Tensor valid_z_cpu = valid_z.to(torch::kCPU);

  // Get size as int for loop
  int num_valid_points = valid_u_cpu.size(0);

  // Create depth map on CPU and fill
  cv::Mat depth_map_cv(height, width, CV_32FC1, 0.0f);

  for (int i = 0; i < num_valid_points; i++) {
    int u = valid_u_cpu[i].item<int64_t>();
    int v = valid_v_cpu[i].item<int64_t>();
    float depth = valid_z_cpu[i].item<float>();

    // If pixel is empty or new depth is closer
    if (depth_map_cv.at<float>(v, u) == 0.0f ||
        depth < depth_map_cv.at<float>(v, u)) {
      depth_map_cv.at<float>(v, u) = depth;
    }
  }

  // Step 6: Create a colorized visualization using JET colormap
  cv::Mat depth_colored;
  double min_depth = min_depth_;
  double max_depth = max_depth_;

  // Normalize the depth map to 0-1 range for visualization
  cv::Mat depth_normalized;
  cv::Mat valid_mask = (depth_map_cv > 0);

  // Find actual min/max in the valid depth values
  double actual_min, actual_max;
  cv::minMaxLoc(depth_map_cv, &actual_min, &actual_max, nullptr, nullptr,
                valid_mask);

  // Use actual min/max values with clamping
  min_depth = std::max(min_depth_, static_cast<float>(actual_min));
  max_depth = std::min(max_depth_, static_cast<float>(actual_max));

  // Normalize between the clamped min/max values
  depth_map_cv = (depth_map_cv - min_depth) / (max_depth - min_depth);
  depth_map_cv.setTo(0, ~valid_mask);  // Set invalid regions to 0

  // Apply JET colormap for better visualization
  // Normalize depth_map_cv to the range [0, 255] and convert to 8-bit
  cv::Mat depth_norm255;
  depth_map_cv.convertTo(depth_norm255, CV_8UC1, 255.0);

  // Apply the JET colormap
  cv::applyColorMap(depth_norm255, depth_colored, cv::COLORMAP_JET);

  // Add original image as background where depth is not available
  cv::Mat rgb_display;
  pkf->img_undist_.convertTo(rgb_display, CV_8UC3, 255.0);
  cv::cvtColor(rgb_display, rgb_display, cv::COLOR_RGB2BGR);

  cv::Mat mask_8uc1 = (depth_map_cv > 0) * 255;
  mask_8uc1.convertTo(mask_8uc1, CV_8UC1);

  // Blend the colored depth map with the original image
  cv::Mat blended = rgb_display.clone();
  depth_colored.copyTo(blended, mask_8uc1);

  // Step 7: Add information overlay
  float coverage =
      100.0f * cv::countNonZero(valid_mask) / (float)(width * height);

  std::stringstream ss;
  ss << "Depth " << std::fixed << std::setprecision(1) << min_depth << "m - "
     << max_depth << "m | Coverage: " << std::setprecision(1) << coverage
     << "%";

  cv::putText(blended, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.7, cv::Scalar(255, 255, 255), 2);

  // Save the visualization
  cv::imwrite(save_path, blended);

  // Create a pure depth map visualization as well
  cv::Mat depth_only;
  cv::Mat depth_normalized_8u;
  depth_map_cv.convertTo(depth_normalized_8u, CV_8UC1, 255.0);
  cv::applyColorMap(depth_normalized_8u, depth_only, cv::COLORMAP_JET);
  cv::imwrite(
      save_path.substr(0, save_path.find_last_of('.')) + "_depth_only.png",
      depth_only);

  std::cout << "Saved depth visualization to " << save_path << std::endl;
}

torch::Tensor GaussianMapper::computeLoGProbability(
    const torch::Tensor& image) {
  torch::Tensor gray_image;
  if (image.size(0) == 3) {
    // Convert RGB to grayscale using standard weights
    torch::Tensor weights =
        torch::tensor({0.299, 0.587, 0.114}, image.options());
    gray_image =
        torch::sum(image * weights.view({3, 1, 1}), 0, true);  // [1, H, W]
  } else {
    // Already single channel
    gray_image = image;
  }

  // Calculate padding for "same" behavior
  int pad_h = disc_kernel_.size(2) / 2;  // kernel height / 2
  int pad_w = disc_kernel_.size(3) / 2;  // kernel width / 2

  torch::Tensor result = torch::nn::functional::conv2d(
      gray_image.unsqueeze(0),  // Add batch dimension [1, 1, H, W]
      disc_kernel_,             // [1, 1, kernel_h, kernel_w]
      torch::nn::functional::Conv2dFuncOptions().padding({pad_h, pad_w}));
  // Result is [1, 1, H, W]

  // Extract the result and remove batch/channel dimensions
  result = result[0][0];  // -> [H, W]

  return result;
}

// Alternative fast implementation using morphological operations
torch::Tensor GaussianMapper::densify_depth_morphological(
    const torch::Tensor& depth_map,
    float invalid_threshold,
    int dilation_size) {
  auto device = depth_map.device();
  if (!device.is_cuda()) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  auto depth = depth_map.to(torch::kFloat32);
  auto valid_mask = depth > invalid_threshold;

  // Create structuring element for morphological operations
  auto kernel_size = dilation_size;
  auto kernel =
      torch::ones({1, 1, kernel_size, kernel_size},
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // Prepare input for convolution
  auto input_4d = depth;
  if (input_4d.dim() == 2) {
    input_4d = input_4d.unsqueeze(0).unsqueeze(0);
  } else if (input_4d.dim() == 3) {
    input_4d = input_4d.unsqueeze(0);
  }

  auto mask_4d = valid_mask.to(torch::kFloat32);
  if (mask_4d.dim() == 2) {
    mask_4d = mask_4d.unsqueeze(0).unsqueeze(0);
  } else if (mask_4d.dim() == 3) {
    mask_4d = mask_4d.unsqueeze(0);
  }

  // Dilate valid regions and interpolate
  auto dilated_mask =
      torch::conv2d(mask_4d, kernel, {}, 1, kernel_size / 2) > 0;
  auto sum_values =
      torch::conv2d(input_4d * mask_4d, kernel, {}, 1, kernel_size / 2);
  auto sum_weights = torch::conv2d(mask_4d, kernel, {}, 1, kernel_size / 2);

  auto interpolated = sum_values / (sum_weights + 1e-8f);

  // Remove extra dimensions
  if (depth.dim() == 2) {
    interpolated = interpolated.squeeze(0).squeeze(0);
    dilated_mask = dilated_mask.squeeze(0).squeeze(0);
  } else if (depth.dim() == 3) {
    interpolated = interpolated.squeeze(0);
    dilated_mask = dilated_mask.squeeze(0);
  }

  // Fill invalid regions with interpolated values
  auto result = torch::where(valid_mask, depth,
                             torch::where(dilated_mask, interpolated, depth));

  return result;
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
  std::string model_path =
      "/workspace/repo/models/fast_acvnet_plus_onnx_gridsample/"
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(model_resolution.height) + "x" +
      std::to_string(model_resolution.width) + ".onnx";

  // std::string model_path =
  //     "/workspace/repo/models/crestereo/"
  //     "crestereo_init_iter20_720x1280.onnx";
  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(model_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  std::string model_path =
      "/workspace/repo/models/metric3dv2/metric3d-vit-large.onnx";
  // std::string model_path =
  //     "/workspace/repo/models/depth_anything/depth_anything_v2_vitb_dynamic.onnx";
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

// MUCH MORE EFFICIENT: Create boolean masks instead of loops
void GaussianMapper::createAndApplyGlobalRemovalMask(
    const torch::Tensor& global_ids_to_remove,
    const std::vector<std::shared_ptr<GaussianModel>>& models,
    const std::vector<int>& model_sizes) {
  if (global_ids_to_remove.numel() == 0) return;

  // Calculate total gaussians across all models
  int total_gaussians = 0;
  for (int size : model_sizes) {
    total_gaussians += size;
  }

  // Create a global boolean mask for all gaussians (MUCH faster than loops)
  torch::Tensor global_removal_mask = torch::zeros(
      {total_gaussians}, torch::dtype(torch::kBool).device(device_type_));

  // Mark gaussians for removal using advanced indexing (GPU-accelerated)
  global_removal_mask.index_put_({global_ids_to_remove}, true);

  // Split the global mask back into per-model masks and apply
  int offset = 0;
  for (size_t model_idx = 0; model_idx < models.size(); model_idx++) {
    int model_size = model_sizes[model_idx];

    if (model_size == 0) continue;

    // Extract this model's portion of the removal mask
    torch::Tensor model_removal_mask =
        global_removal_mask.slice(0, offset, offset + model_size);

    // Only call prunePoints if there's actually something to remove
    if (model_removal_mask.any().item<bool>()) {
      std::unique_lock<std::mutex> lock_render(mutex_render_);
      models[model_idx]->prunePoints(model_removal_mask);

      int removed_count = model_removal_mask.sum().item<int>();
      // std::cout << "Model " << model_idx << ": Removed " << removed_count
      //           << " Gaussians" << std::endl;
    }

    offset += model_size;
  }
}