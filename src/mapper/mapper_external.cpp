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

#include "gaussian_mapper_external.h"

#include <iostream>

#include "gaussian_mapper.h"

// Frame implementation
Frame::Frame(const cv::Mat& rgb,
             const cv::Mat& depth,
             const Sophus::SE3f& p,
             double ts)
    : rgb_image(rgb.clone()),
      depth_image(depth.clone()),
      pose(p),
      timestamp(ts) {}

// LeakyFrameQueue implementation
LeakyFrameQueue::LeakyFrameQueue(size_t max_size) : max_size_(max_size) {}

void LeakyFrameQueue::push(Frame&& frame) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (queue_.size() >= max_size_) {
    // Drop newest frame when full
    // std::cout << "Queue full, dropped newest frame" << std::endl;
    return;
  }

  queue_.push_back(std::move(frame));
  cv_.notify_one();
}

std::optional<Frame> LeakyFrameQueue::pop(bool wait) {
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

void LeakyFrameQueue::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  cv_.notify_all();
}

bool LeakyFrameQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

size_t LeakyFrameQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

// GaussianMapper external mode methods
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
