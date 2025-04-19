#pragma once
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "chunk_manager.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"
#include "nanoflann.hpp"  // Using nanoflann for KD-tree

// Define a point cloud adapter for nanoflann
struct ClusterCentersAdapter {
  const std::vector<Eigen::Vector3f>& centers;

  explicit ClusterCentersAdapter(const std::vector<Eigen::Vector3f>& centers)
      : centers(centers) {}

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return centers.size(); }

  // Returns the dim'th component of the idx'th point in the class
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    return centers[idx](dim);
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }
};

// Define the KD-tree type using the adapter
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, ClusterCentersAdapter>,
    ClusterCentersAdapter,
    3,      // dimension
    size_t  // index type
    >
    ClusterKDTree;

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

  void visualizeClusterCenters(
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
  std::unordered_map<std::size_t, int>
      kfs_used_times_;  // Changed to unordered_map for O(1) lookup

  // New cluster-related members
  std::vector<std::vector<std::size_t>> clusters_;  // Clusters of keyframes
  std::vector<Eigen::Vector3f> cluster_centers_;    // Centers of each cluster
  int current_cluster_;                             // Current active cluster
  int cluster_iterations_;      // Iterations spent in current cluster
  int iterations_per_cluster_;  // Iterations to spend in each cluster
  int keyframes_since_last_full_clustering_;  // Counter for new keyframes
  // Cache for keyframe positions
  std::unordered_map<std::size_t, Eigen::Vector3f> keyframe_positions_cache_;

  // Number of new keyframes before full reclustering
  const int RECLUSTER_THRESHOLD = 15;

  // Helper to pre-warm cache when switching clusters
  void prewarmClusterCache();

  // Helper to find nearest cluster for a keyframe
  int findNearestCluster(const Eigen::Vector3f& position) const;

  // Nanoflann KD-tree for efficient nearest neighbor searches
  std::unique_ptr<ClusterCentersAdapter> cluster_centers_adapter_;
  std::unique_ptr<ClusterKDTree> cluster_kdtree_;

  // For batch processing
  static constexpr int BATCH_SIZE = 100;
};