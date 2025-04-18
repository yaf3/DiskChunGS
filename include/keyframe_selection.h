#pragma once

#include <map>
#include <memory>
#include <queue>
#include <unordered_map>

#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

class KeyframeQueue {
 private:
  std::queue<std::shared_ptr<GaussianKeyframe>> keyframe_queue_;
  std::vector<size_t> kfid_shuffle_;
  size_t kfid_shuffle_idx_ = 0;
  bool& kfid_shuffled_;  // Reference to track shuffle status
  std::shared_ptr<GaussianScene> scene_;
  std::map<std::size_t, int>& kfs_used_times_;  // Reference to track usage
  const size_t queue_size_ = 10;  // Pre-fill queue with this many frames

 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                std::map<std::size_t, int>& used_times,
                bool& shuffled)
      : scene_(scene), kfs_used_times_(used_times), kfid_shuffled_(shuffled) {
    std::cout << "KeyframeQueue initialized" << std::endl;
  }

  void generateKfidRandomShuffle();
  void fillQueue();
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  std::vector<std::shared_ptr<GaussianKeyframe>> peekUpcomingKeyframes(
      size_t count);
};