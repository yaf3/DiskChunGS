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
#include "utils/loss_utils.h"
#include "utils/profiling.h"

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

  torch::Tensor visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf);

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

  torch::Tensor visible_gaussian_mask = gaussians_->cullVisibleGaussians(pkf);

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

    break;
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