#pragma once
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

class KeyframeQueue {
 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                size_t queue_size = 10,
                const std::map<std::size_t, float>* loss_map = nullptr);
  ~KeyframeQueue();

  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

 private:
  std::shared_ptr<GaussianScene> scene_;
  size_t queue_size_;
  size_t recent_keyframes_count_;  // k most recent keyframes to select from

  // Store the keyframe IDs in the order they were added
  std::vector<std::size_t> keyframe_ids_;

  // Keep track of usage counts
  std::unordered_map<std::size_t, int> kfs_used_times_;

  // Random number generator
  std::mt19937 rng_;

  std::mutex mutex_new_keyframe_;

  std::queue<std::shared_ptr<GaussianKeyframe>> save_queue_;
  std::thread save_worker_;
  std::mutex save_mutex_;
  std::condition_variable save_cv_;
  bool save_worker_stop_ = false;

  void saveWorker();
  void queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe);
};