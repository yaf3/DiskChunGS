#include "include/keyframe_selection.h"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t max_active_keyframes,
                             const std::map<std::size_t, float>* loss_map,
                             float use_last_frame_proba,
                             size_t n_kept_frames)
    : scene_(scene),
      max_active_keyframes_(max_active_keyframes),
      n_kept_frames_(n_kept_frames),
      use_last_frame_proba_(use_last_frame_proba),
      cpu_offload_count_(0),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f) {
  std::cout << "Created KeyframeQueue with max_active_keyframes="
            << max_active_keyframes_ << ", n_kept_frames=" << n_kept_frames_
            << ", use_last_frame_proba=" << use_last_frame_proba_ << std::endl;

  // Start the save worker thread
  save_worker_ = std::thread(&KeyframeQueue::saveWorker, this);
}

// Destructor
KeyframeQueue::~KeyframeQueue() {
  // Signal worker to stop
  {
    std::unique_lock<std::mutex> lock(save_mutex_);
    save_worker_stop_ = true;
  }
  save_cv_.notify_all();

  // Wait for worker to finish
  if (save_worker_.joinable()) {
    save_worker_.join();
  }
}

// Move random keyframe from GPU to CPU (with protection)
void KeyframeQueue::moveRandomKeyframeToCPU() {
  if (active_frames_gpu_.size() <= n_kept_frames_) {
    return;  // Not enough frames to move any
  }

  // Select from all except the last n_kept_frames_ (protected frames)
  size_t selectable_count = active_frames_gpu_.size() - n_kept_frames_;
  std::uniform_int_distribution<size_t> dist(0, selectable_count - 1);
  size_t random_idx = dist(rng_);

  std::size_t frame_id = active_frames_gpu_[random_idx];

  // Move keyframe data to disk/CPU
  auto it = scene_->keyframes().find(frame_id);
  if (it != scene_->keyframes().end()) {
    // Save data to disk
    queueForSaving(it->second);
    std::cout << "Moved keyframe " << frame_id << " to CPU/disk" << std::endl;
  }

  // Update lists
  active_frames_cpu_.push_back(frame_id);
  active_frames_gpu_.erase(active_frames_gpu_.begin() + random_idx);

  cpu_offload_count_++;
}

// Move random keyframe from CPU back to GPU
void KeyframeQueue::moveRandomKeyframeToGPU() {
  if (active_frames_cpu_.empty()) {
    return;
  }

  std::uniform_int_distribution<size_t> dist(0, active_frames_cpu_.size() - 1);
  size_t random_idx = dist(rng_);
  std::size_t frame_id = active_frames_cpu_[random_idx];

  // Load keyframe data from disk
  auto it = scene_->keyframes().find(frame_id);
  if (it != scene_->keyframes().end()) {
    // Load data from disk
    it->second->loadDataFromDisk();
    std::cout << "Moved keyframe " << frame_id << " back to GPU" << std::endl;
  }

  // Insert at front
  active_frames_gpu_.push_front(frame_id);
  active_frames_cpu_.erase(active_frames_cpu_.begin() + random_idx);
}

// Perform reshuffling (every 5th CPU offload)
void KeyframeQueue::performReshuffling() {
  std::cout << "Performing memory reshuffling..." << std::endl;

  // Additional CPU offload
  moveRandomKeyframeToCPU();

  // Move one frame back from CPU to GPU
  moveRandomKeyframeToGPU();

  std::cout << "Memory reshuffling complete" << std::endl;
}

void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  std::unique_lock<std::mutex> lock(mutex_memory_mgmt_);

  // Add new keyframe to GPU list
  active_frames_gpu_.push_back(keyframe->fid_);

  // Check if we exceed the memory limit
  if (active_frames_gpu_.size() > max_active_keyframes_) {
    moveRandomKeyframeToCPU();

    // Reshuffling logic: every 5th CPU offload
    if (cpu_offload_count_ % 5 == 0) {
      performReshuffling();
    }
  }

  std::cout << "Added keyframe " << keyframe->fid_
            << " (GPU: " << active_frames_gpu_.size()
            << ", CPU: " << active_frames_cpu_.size() << ")" << std::endl;
}

// Updated keyframe selection using active_frames_gpu_
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  std::unique_lock<std::mutex> lock(mutex_memory_mgmt_);

  if (active_frames_gpu_.empty()) return nullptr;

  std::size_t kf_id;

  // Same probabilistic logic: 20% latest, 80% random from GPU frames
  if (uniform_dist_(rng_) <= use_last_frame_proba_) {
    // Use the most recent keyframe (last in GPU list)
    kf_id = active_frames_gpu_.back();
  } else {
    // Randomly select from all GPU keyframes
    std::uniform_int_distribution<size_t> distrib(
        0, active_frames_gpu_.size() - 1);
    size_t randomIndex = distrib(rng_);
    kf_id = active_frames_gpu_[randomIndex];
  }

  auto it = scene_->keyframes().find(kf_id);
  return it->second;
}

void KeyframeQueue::saveWorker() {
  while (true) {
    std::shared_ptr<GaussianKeyframe> keyframe_to_save;

    {
      std::unique_lock<std::mutex> lock(save_mutex_);
      save_cv_.wait(
          lock, [this] { return !save_queue_.empty() || save_worker_stop_; });

      if (save_worker_stop_ && save_queue_.empty()) {
        break;
      }

      if (!save_queue_.empty()) {
        keyframe_to_save = save_queue_.front();
        save_queue_.pop();
      }
    }

    if (keyframe_to_save) {
      keyframe_to_save->saving_ = true;
      try {
        keyframe_to_save->saveDataToDisk();
      } catch (const std::exception& e) {
        std::cerr << "Error saving keyframe " << keyframe_to_save->fid_ << ": "
                  << e.what() << std::endl;
      }
    }
    keyframe_to_save->saving_ = false;
  }
}

void KeyframeQueue::queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe) {
  {
    std::unique_lock<std::mutex> lock(save_mutex_);
    save_queue_.push(keyframe);
  }
  save_cv_.notify_one();
}
