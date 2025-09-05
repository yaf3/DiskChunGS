#pragma once

#include <deque>
#include <map>
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
                size_t max_active_keyframes = 200,
                const std::map<std::size_t, float>* loss_map = nullptr,
                float use_last_frame_proba = 0.2f,
                size_t n_kept_frames = 20);
  ~KeyframeQueue();

  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

 private:
  std::shared_ptr<GaussianScene> scene_;

  // Memory management parameters
  size_t max_active_keyframes_;  // Maximum keyframes in GPU memory
  size_t n_kept_frames_;         // Number of most recent frames to protect
  float use_last_frame_proba_;   // Probability of using latest frame

  // Two-tier memory system (equivalent to Python's lists)
  std::deque<std::size_t> active_frames_gpu_;  // Keyframes in GPU memory
  std::deque<std::size_t> active_frames_cpu_;  // Keyframes moved to CPU/disk

  // Counters for reshuffling logic
  size_t cpu_offload_count_;

  // Random number generation
  std::mt19937 rng_;
  std::uniform_real_distribution<float> uniform_dist_;

  // Thread safety
  std::mutex mutex_memory_mgmt_;

  // Async saving (existing)
  std::queue<std::shared_ptr<GaussianKeyframe>> save_queue_;
  std::thread save_worker_;
  std::mutex save_mutex_;
  std::condition_variable save_cv_;
  bool save_worker_stop_ = false;

  // Memory management methods
  void moveRandomKeyframeToCPU();
  void moveRandomKeyframeToGPU();
  void performReshuffling();

  // Existing methods
  void saveWorker();
  void queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe);
};