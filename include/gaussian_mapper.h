/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <jsoncpp/json/json.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "DepthAnything.h"
#include "FastACVNet.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "ORB-SLAM3/include/MapDrawer.h"
#include "ORB-SLAM3/include/System.h"
#include "chunk_manager.h"
#include "chunk_types.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"
#include "keyframe_selection.h"
#include "operate_points.h"
#include "stereo_vision.h"
#include "tensor_utils.h"

class ChunkManager;      // Forward declaration
class KeyframeSelector;  // Forward declaration

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                 \
  if (!dir.empty() && !std::filesystem::exists(dir))                  \
    if (!std::filesystem::create_directories(dir))                    \
      throw std::runtime_error("Cannot create result directory at " + \
                               dir.string());
struct UndistortParams {
  UndistortParams(
      const cv::Size &old_size,
      cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
      : old_size_(old_size) {
    dist_coeff.copyTo(dist_coeff_);
  }

  cv::Size old_size_;
  cv::Mat dist_coeff_;
};

enum SystemSensorType { INVALID = 0, MONOCULAR = 1, STEREO = 2, RGBD = 3 };

struct VariableParameters {
  float position_lr_init;
  float feature_lr;
  float opacity_lr;
  float scaling_lr;
  float rotation_lr;
  float percent_dense;
  float lambda_dssim;
  int opacity_reset_interval;
  float densify_grad_th;
  int densify_interval;
  int new_kf_times_of_use;
  int stable_num_iter_existence;  ///< loop closure correction

  bool keep_training;
  bool do_gaus_pyramid_training;
  bool do_inactive_geo_densify;
};

void copyFolder(const std::filesystem::path &source,
                const std::filesystem::path &destination);

class GaussianMapper {
 public:
  GaussianMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                 std::filesystem::path gaussian_config_file_path,
                 std::filesystem::path result_dir = "",
                 int seed = 0,
                 torch::DeviceType device_type = torch::kCUDA);

  // External mode initialization
  GaussianMapper(const SystemSensorType sensor_type,
                 const string &orb_settings_path,
                 std::filesystem::path gaussian_config_file_path,
                 std::filesystem::path result_dir,
                 int seed,
                 torch::DeviceType device_type);

  void readConfigFromFile(std::filesystem::path cfg_path);

  void run();
  void trainColmap();
  void trainForOneIteration();

  bool isStopped();
  void signalStop(const bool going_to_stop = true);

  cv::Mat renderFromPose(const Sophus::SE3f &Tcw,
                         const int width,
                         const int height,
                         const bool main_vision = false);

  int getIteration();
  void increaseIteration(const int inc = 1);

  // Gaussian management
  void addPoints(
      const torch::Tensor &points,
      const torch::Tensor &colors,
      const torch::Tensor &scales,
      std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes);

  std::tuple<torch::Tensor, torch::Tensor> filterPointsByDepth(
      const torch::Tensor &points,
      const torch::Tensor &colors,
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
          &keyframes);

  float positionLearningRateInit();
  float featureLearningRate();
  float opacityLearningRate();
  float scalingLearningRate();
  float rotationLearningRate();
  float percentDense();
  float lambdaDssim();
  float lambdaDepth();
  int opacityResetInterval();
  float densifyGradThreshold();
  int densifyInterval();
  int newKeyframeTimesOfUse();
  int stableNumIterExistence();
  bool isKeepingTraining();
  bool isdoingGausPyramidTraining();
  bool isdoingInactiveGeoDensify();
  bool isdoingDepthDensify();

  void setPositionLearningRateInit(const float lr);
  void setFeatureLearningRate(const float lr);
  void setOpacityLearningRate(const float lr);
  void setScalingLearningRate(const float lr);
  void setRotationLearningRate(const float lr);
  void setPercentDense(const float percent_dense);
  void setLambdaDssim(const float lambda_dssim);
  void setOpacityResetInterval(const int interval);
  void setDensifyGradThreshold(const float th);
  void setDensifyInterval(const int interval);
  void setNewKeyframeTimesOfUse(const int times);
  void setStableNumIterExistence(const int niter);
  void setKeepTraining(const bool keep);
  void setDoGausPyramidTraining(const bool gaus_pyramid);
  void setDoInactiveGeoDensify(const bool inactive_geo_densify);

  VariableParameters getVaribleParameters();
  void setVaribleParameters(const VariableParameters &params);

  GaussianModelParams &getGaussianModelParams() { return this->model_params_; }
  void setColmapDataPath(std::filesystem::path colmap_path) {
    this->model_params_.source_path_ = colmap_path;
  }
  void setSensorType(SystemSensorType sensor_type) {
    this->sensor_type_ = sensor_type;
  }

  void loadPly(std::filesystem::path ply_path,
               std::filesystem::path camera_path = "");

 protected:
  bool hasMetInitialMappingConditions();
  bool hasMetIncrementalMappingConditions();

  void combineMappingOperations();
  void processLocalMappingBABatch(
      std::vector<ORB_SLAM3::MappingOperation> &operations);
  void processLoopClosureBA(ORB_SLAM3::MappingOperation &opr);
  void processScaleRefinement(ORB_SLAM3::MappingOperation &opr);

  void handleNewKeyframe(std::tuple<unsigned long,
                                    unsigned long,
                                    Sophus::SE3f,
                                    cv::Mat,
                                    bool,
                                    cv::Mat,
                                    std::vector<float>,
                                    std::vector<float>,
                                    std::string> &kf);
  std::shared_ptr<GaussianKeyframe> useOneRandomSlidingWindowKeyframe();
  std::vector<std::shared_ptr<GaussianKeyframe>> getUpcomingKeyframes(
      size_t count);
  std::shared_ptr<GaussianKeyframe> useOneRandomKeyframe();
  std::shared_ptr<GaussianKeyframe> useRecentKeyframe();
  void generateKfidRandomShuffle();

 public:
  void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf,
                                  int times);
  bool saveScene(std::filesystem::path scene_dir);
  bool loadScene(std::filesystem::path scene_dir,
                 std::filesystem::path optional_camera_path = "");

  void signalStopEvalMode();

 protected:
  void cullKeyframes();

  void increasePcdByKeyframeInactiveGeoDensify(
      std::shared_ptr<GaussianKeyframe> pkf);

  void increasePcdByDepthReconstruction(std::shared_ptr<GaussianKeyframe> pkf);

  // bool needInterruptTraining();
  // void setInterruptTraining(const bool interrupt_training);

  void recordKeyframeRendered(torch::Tensor &rendered,
                              torch::Tensor &ground_truth,
                              unsigned long kfid,
                              std::filesystem::path result_img_dir,
                              std::filesystem::path result_gt_dir,
                              std::filesystem::path result_loss_dir,
                              std::string name_suffix = "");
  void renderAndRecordKeyframe(std::shared_ptr<GaussianKeyframe> pkf,
                               float &dssim,
                               float &psnr,
                               float &psnr_gs,
                               double &render_time,
                               std::filesystem::path result_img_dir,
                               std::filesystem::path result_gt_dir,
                               std::filesystem::path result_loss_dir,
                               std::string name_suffix = "");
  void renderAndRecordAllKeyframes(std::string name_suffix = "");

  void savePly(std::filesystem::path result_dir);
  void keyframesToJson(std::filesystem::path result_dir);
  void saveModelParams(std::filesystem::path result_dir);
  void writeKeyframeUsedTimes(std::filesystem::path result_dir,
                              std::string name_suffix = "");

  std::vector<std::shared_ptr<GaussianModel>> selectRandomModelSubset(
      const std::vector<std::shared_ptr<GaussianModel>> &allModels,
      size_t subset_size);

  void renderFlyThroughVideo(const std::string &output_path,
                             int width,
                             int height,
                             int fps,
                             float duration_seconds,
                             float smoothness_factor = 0.5,
                             int keyframe_subsample = 1);

  void render3DExplorationVideo(const std::string &output_path,
                                int width,
                                int height,
                                int fps,
                                float duration_seconds,
                                float deviation_scale = 0.15f,
                                bool look_around = true);

  void saveChunkManifest(std::filesystem::path scene_dir);
  void loadChunkManifest(std::filesystem::path scene_dir);
  void loadCamerasFromJson(std::filesystem::path json_path);

  void saveTotalGaussians(std::string name_suffix);

  torch::Tensor log_kernel_;
  float log_sigma_ = 3.0f;  // Sigma for LoG operator

  torch::Tensor computeLoGProbability(const torch::Tensor &image);
  void initializeLaplacianOfGaussianKernel();
  torch::Tensor densify_depth_morphological(const torch::Tensor &depth_map,
                                            float invalid_threshold = 0.0f,
                                            int dilation_size = 3);
  void initializeMonocularDepthEstimator();
  void initializeStereoDepthEstimator();

 private:
  // Updated function declarations:
  std::shared_ptr<GaussianKeyframe> selectLocalityAwareKeyframe();
  std::vector<std::shared_ptr<GaussianKeyframe>> predictUpcomingKeyframes(
      int count = 5);
  void initializeChunkManagement();

 private:
  // Frame structure for the queue
  struct Frame {
    cv::Mat rgb_image;
    cv::Mat depth_image;
    Sophus::SE3f pose;
    double timestamp;

    Frame(const cv::Mat &rgb,
          const cv::Mat &depth,
          const Sophus::SE3f &p,
          double ts);
  };

  class LeakyFrameQueue {
   public:
    explicit LeakyFrameQueue(size_t max_size = 20);
    void push(Frame &&frame);
    std::optional<Frame> pop(bool wait = true);
    void stop();
    bool empty() const;
    size_t size() const;

   private:
    std::deque<Frame> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_size_;
    bool stopped_{false};
  };

 public:
  // Parameters
  std::filesystem::path config_file_path_;

  // Chunk manager for efficient memory handling
  std::shared_ptr<ChunkManager> chunk_manager_;

  // Scene
  std::shared_ptr<GaussianScene> scene_;

  std::shared_ptr<KeyframeQueue> keyframe_queue_;

  // SLAM system
  std::shared_ptr<ORB_SLAM3::System> pSLAM_;

  float chunk_size_ = 50.0;
  int max_chunks_in_memory_ = 50;
  std::filesystem::path chunk_save_dir_;

  // Settings
  torch::DeviceType device_type_;
  int num_gaus_pyramid_sub_levels_ = 0;
  std::vector<int> kf_gaus_pyramid_times_of_use_;
  std::vector<float> kf_gaus_pyramid_factors_;

  bool viewer_camera_id_set_ = false;
  std::uint32_t viewer_camera_id_ = 0;
  float rendered_image_viewer_scale_ = 1.0f;
  float rendered_image_viewer_scale_main_ = 1.0f;

  float z_near_ = 0.01f;
  float z_far_ = 100.0f;

  // Data
  std::map<camera_id_t, torch::Tensor> undistort_mask_;
  std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
  std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;
  bool kfid_shuffled_ = false;

 protected:
  // Parameters
  GaussianModelParams model_params_;
  GaussianOptimizationParams opt_params_;
  GaussianPipelineParams pipe_params_;

  // Data
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
      viewpoint_sliding_window_;
  std::vector<std::size_t> kfid_shuffle_;
  std::size_t kfid_shuffle_idx_ = 0;

 public:
  std::map<std::size_t, float> kfs_loss_;
  std::map<std::size_t, int> kfs_used_times_;
  float keyframe_similarity_threshold_ = 0.30f;
  int keyframe_selection_strategy_ = 0;  // 0: all, 1: recent k

  // Status
  bool initial_mapped_;
  bool interrupt_training_;
  bool stopped_;
  int iteration_;
  float ema_loss_for_log_;
  bool SLAM_ended_;
  bool loop_closure_iteration_;
  bool keep_training_ = false;
  int default_sh_ = 0;

  // Settings
  SystemSensorType sensor_type_;

  float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;
  float depth_densify_subsample_ratio_ = 0.1;
  float stereo_baseline_length_ = 0.0f;
  int stereo_min_disparity_ = 0;
  int stereo_num_disparity_ = 128;
  bool do_stereo_loss_ = false;

  cv::Mat stereo_Q_;
  cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
  std::shared_ptr<FastACVNet> stereo_depth_estimator_;
  std::shared_ptr<DepthAnything> monocular_depth_estimator_;
  float min_depth_ = 0.0f;
  float max_depth_ = 100.0f;

  bool inactive_geo_densify_ = true;
  bool depth_densify_ = false;
  int depth_cached_ = 0;
  int max_depth_cached_ = 1;
  torch::Tensor depth_cache_points_;
  torch::Tensor depth_cache_colors_;
  torch::Tensor depth_cache_scales_;
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
      depth_cache_keyframes_;

  unsigned long min_num_initial_map_kfs_;
  torch::Tensor background_;
  float large_rot_th_;
  float large_trans_th_;
  torch::Tensor override_color_;

  int new_keyframe_times_of_use_;
  int local_BA_increased_times_of_use_;
  int loop_closure_increased_times_of_use_;

  bool cull_keyframes_;
  int stable_num_iter_existence_;

  bool do_gaus_pyramid_training_;

  std::filesystem::path result_dir_;
  int keyframe_record_interval_;
  int all_keyframes_record_interval_;
  bool record_rendered_image_;
  bool record_ground_truth_image_;
  bool record_loss_image_;
  bool render_fly_through_;
  float render_fly_through_speed_ = 5.0;

  int training_report_interval_;
  bool record_loop_ply_;

  int prune_big_point_after_iter_;
  float densify_min_opacity_ = 20;
  int appearance_embedding_ = 0;

  // Tools
  std::random_device rd_;

  cv::Mat external_image_;
  Sophus::SE3f external_pose_;
  LeakyFrameQueue frame_queue_;

  // Mutex
  std::mutex mutex_status_;
  std::mutex mutex_settings_;
  std::mutex
      mutex_render_;  ///< the model is suppose to be read-only from outside
  std::mutex mutex_external_data_;

 public:
  void initializeMapFromExternal();
  bool isKeyframe(const Sophus::SE3f &current_pose, double current_time);
  void processNewFrame(const cv::Mat &rgb_image,
                       const cv::Mat &depth_or_right_image,
                       const Sophus::SE3f &pose,
                       const double timestamp);
  void handleNewFrameExternal(const cv::Mat &rgb_image,
                              const cv::Mat &depth_or_right_image,
                              const Sophus::SE3f &pose,
                              const double timestamp);

  void setRecentExternalData(const cv::Mat &rgb_image,
                             const Sophus::SE3f &pose);

  std::tuple<const cv::Mat, const Sophus::SE3f> getRecentExternalData();
  void run_external_poses();
  void visualizeDepthReconstruction(std::shared_ptr<GaussianKeyframe> pkf,
                                    const torch::Tensor &points3D,
                                    const torch::Tensor &valid_points,
                                    const std::string &save_path);

  volatile bool isExternalDataStopped() {
    return external_data_stopped_.load(std::memory_order_acquire);
  }

  volatile void signalExternalDataStopped() {
    std::cout << "External data stopped" << std::endl;
    external_data_stopped_.store(true, std::memory_order_release);
  }

  std::atomic<bool> external_data_stopped_{false};
  std::function<void()> completion_callback_;
  void setCompletionCallback(std::function<void()> callback);

  // Member variables for external pose handling
  std::mutex mutex_new_frame_;
  Sophus::SE3f last_keyframe_pose_;
  float min_keyframe_translation_{
      0.25f};                           // Minimum translation for new keyframe
  float min_keyframe_rotation_{0.15f};  // Minimum rotation in radians
  double last_keyframe_timestamp_{0.0};
  float min_keyframe_time_{0.5f};  // Minimum time between keyframes
};
