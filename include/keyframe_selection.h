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
  KeyframeQueue(std::shared_ptr<GaussianScene> scene, float chunk_size = 20.0f);

  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

 private:
  std::shared_ptr<GaussianScene> scene_;
  float chunk_size_;
  float chunk_sizes_[3];  // Different levels: FINE, MEDIUM, COARSE
  int64_t current_active_chunk_id_;

  std::shared_ptr<GaussianKeyframe> latest_keyframe_;

  // Random number generation
  std::mt19937 rng_;
  std::uniform_real_distribution<float> uniform_dist_;
  std::discrete_distribution<int> level_dist_;

  std::map<int64_t, std::vector<std::shared_ptr<GaussianKeyframe>>>
      chunk_to_keyframes_[3];

  std::deque<std::shared_ptr<GaussianKeyframe>> gpu_queue;
  size_t max_gpu_keyframes_ = 200;

  Eigen::Vector3f tensorToEigen(const torch::Tensor& tensor) const;
};