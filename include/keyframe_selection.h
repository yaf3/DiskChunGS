#pragma once
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chunk_manager.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

class KeyframeQueue {
 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene, size_t queue_size = 10);

  // Set chunk manager reference
  void setChunkManager(std::shared_ptr<ChunkManager> chunk_manager);

  // Generate new visibility-based clusters
  void generateVisibilityBasedClusters();

  // Legacy method for backwards compatibility
  void generateKfidRandomShuffle();

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

  // Notify that a new keyframe was added
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

  // Visualize clusters and their chunk overlaps
  void visualizeClusters(
      const std::string& output_file = "cluster_visualization.svg",
      int width = 800,
      int height = 600);

 private:
  std::shared_ptr<GaussianScene> scene_;
  size_t queue_size_;
  std::shared_ptr<ChunkManager> chunk_manager_;

  // Original shuffle-related members
  std::vector<std::size_t> kfid_shuffle_;
  int kfid_shuffle_idx_;
  bool kfid_shuffled_;
  std::queue<std::shared_ptr<GaussianKeyframe>> keyframe_queue_;
  std::unordered_map<std::size_t, int> kfs_used_times_;

  // Visibility-based clustering
  std::vector<std::vector<std::size_t>> clusters_;
  int current_cluster_;
  int cluster_iterations_;
  int iterations_per_cluster_;
  int keyframes_since_last_full_clustering_;

  // Thresholds and constants
  const int RECLUSTER_THRESHOLD = 15;
  const float SIMILARITY_THRESHOLD =
      0.3f;  // Minimum similarity to consider keyframes in same cluster
  const int MAX_PRELOAD_CHUNKS =
      10;  // Max chunks to preload when switching clusters
  const std::chrono::seconds VISIBILITY_CACHE_EXPIRY{
      30};  // How long to keep visibility data

  // Helper functions
  std::vector<ChunkCoord> getOrComputeVisibleChunks(
      std::shared_ptr<GaussianKeyframe> keyframe);
  float computeChunkOverlapSimilarity(std::size_t kf1_id, std::size_t kf2_id);
  void preloadClusterChunks();
  void addKeyframeToExistingClusters(
      std::shared_ptr<GaussianKeyframe> keyframe);
};