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
                float similarity_threshold = 0.30f,
                int auto_distribute = 4,
                const std::map<std::size_t, float>* loss_map = nullptr);
  ~KeyframeQueue();

  // Interface methods to match the original
  void generateVisibilityBasedClusters();
  void generateKfidRandomShuffle();
  void fillQueue();
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  std::vector<std::shared_ptr<GaussianKeyframe>> peekUpcomingKeyframes(
      size_t count);
  void setIterationsPerCluster(int iterations);
  int getCurrentClusterIndex() const;
  int getClusterCount() const;
  void forceNextCluster();
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);
  void visualizeClusters(
      const std::string& output_file = "cluster_visualization.svg",
      int width = 800,
      int height = 600);

  std::unordered_map<std::size_t, int> getKfsUsedTimes() const {
    return kfs_used_times_;
  }

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