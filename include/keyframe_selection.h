#pragma once

#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "chunk_manager.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

class KeyframeQueue {
 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene, size_t queue_size = 10);

  // Generate random shuffle (original method)
  void generateKfidRandomShuffle();

  // New method for spatial coherence
  void generateSpatiallyCoherentBatches();

  // Add single keyframe to existing clusters
  void addKeyframeToExistingClusters(
      std::shared_ptr<GaussianKeyframe> keyframe);

  // Fill the keyframe queue
  void fillQueue();

  // Get next keyframe
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();

  // Look ahead without modifying queue
  std::vector<std::shared_ptr<GaussianKeyframe>> peekUpcomingKeyframes(
      size_t count);

  // Set iterations per cluster
  void setIterationsPerCluster(int iterations);

  // Get current cluster index
  int getCurrentClusterIndex() const;

  // Get number of clusters
  int getClusterCount() const;

  // Force switch to next cluster
  void forceNextCluster();

  // Set chunk manager reference
  void setChunkManager(std::shared_ptr<ChunkManager> chunk_manager);

  // Notify that a new keyframe was added
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

 private:
  std::shared_ptr<GaussianScene> scene_;
  size_t queue_size_;
  std::shared_ptr<ChunkManager> chunk_manager_;

  // Original shuffle-related members
  std::vector<std::size_t> kfid_shuffle_;
  int kfid_shuffle_idx_;
  bool kfid_shuffled_;
  std::queue<std::shared_ptr<GaussianKeyframe>> keyframe_queue_;
  std::map<std::size_t, int> kfs_used_times_;

  // New cluster-related members
  std::vector<std::vector<std::size_t>> clusters_;  // Clusters of keyframes
  std::vector<Eigen::Vector3f> cluster_centers_;    // Centers of each cluster
  int current_cluster_;                             // Current active cluster
  int cluster_iterations_;      // Iterations spent in current cluster
  int iterations_per_cluster_;  // Iterations to spend in each cluster
  int keyframes_since_last_full_clustering_;  // Counter for new keyframes

  // Number of new keyframes before full reclustering
  const int RECLUSTER_THRESHOLD = 15;

  // Helper to pre-warm cache when switching clusters
  void prewarmClusterCache();

  // Helper to find nearest cluster for a keyframe
  int findNearestCluster(const Eigen::Vector3f& position) const;
};