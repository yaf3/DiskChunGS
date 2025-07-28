#include "include/keyframe_selection.h"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t queue_size,
                             float similarity_threshold,
                             int auto_distribute,
                             const std::map<std::size_t, float>* loss_map)
    : scene_(scene),
      queue_size_(queue_size),
      recent_keyframes_count_(queue_size),  // Use queue_size as default for k
      rng_(std::random_device{}()) {
  std::cout << "Created SimpleKeyframeQueue with k=" << recent_keyframes_count_
            << " most recent keyframes" << std::endl;

  // Start the save worker thread
  save_worker_ = std::thread(&KeyframeQueue::saveWorker, this);
}

// Destructor - make sure to add this to your header
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

// Worker thread function - add this to your header as private method
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
      try {
        keyframe_to_save->saveDataToDisk();
        // std::cout << "Async saved keyframe " << keyframe_to_save->fid_
        //           << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "Error saving keyframe " << keyframe_to_save->fid_ << ": "
                  << e.what() << std::endl;
      }
    }
  }
}

// Helper method to queue keyframe for async saving
void KeyframeQueue::queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe) {
  {
    std::unique_lock<std::mutex> lock(save_mutex_);
    save_queue_.push(keyframe);
  }
  save_cv_.notify_one();
}

// Get next keyframe (randomly selected from k most recent)
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (keyframe_ids_.empty()) return nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, keyframe_ids_.size() - 1);
  int randomIndex = distrib(gen);
  std::size_t kf_id = keyframe_ids_[randomIndex];

  auto it = scene_->keyframes().find(kf_id);
  return it->second;
}

void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  std::unique_lock<std::mutex> lock(mutex_new_keyframe_);

  // Add to the list of keyframe IDs
  if (keyframe_ids_.size() >= queue_size_) {
    // Remove the oldest keyframe ID
    std::size_t oldest_kf_id = keyframe_ids_.front();
    keyframe_ids_.erase(keyframe_ids_.begin());

    // Queue the oldest keyframe for async saving
    auto it = scene_->keyframes().find(oldest_kf_id);
    if (it != scene_->keyframes().end()) {
      queueForSaving(it->second);
    }
  }

  keyframe_ids_.push_back(keyframe->fid_);
}