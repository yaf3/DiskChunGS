#pragma once

#include <Eigen/Dense>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chunk_types.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

/**
 * Spatial grid-based keyframe selection queue.
 * Uses a 3D spatial grid to efficiently find keyframes within a radius
 * of the most recent keyframe position.
 *
 * Key Features:
 * - Maintains ALL keyframes in spatial grid permanently
 * - Automatic memory management based on distance from latest keyframe
 * - Two-tier radius system: selection radius and memory management radius
 * - Efficient O(1) spatial queries even with 1000+ keyframes
 */
class KeyframeQueue {
 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                size_t queue_size = 10,
                const std::map<std::size_t, float>* loss_map = nullptr,
                std::map<std::size_t, int>* used_times_map = nullptr);

  ~KeyframeQueue();

  // Main interface methods
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();
  void notifyNewKeyframeAdded(std::shared_ptr<GaussianKeyframe> keyframe);

  // Configuration methods
  void setSelectionRadius(float radius) { selection_radius_ = radius; }
  void setMemoryManagementRadius(float radius) {
    memory_management_radius_ = radius;
  }
  void setGridCellSize(float cell_size) { grid_cell_size_ = cell_size; }
  float getSelectionRadius() const { return selection_radius_; }
  float getMemoryManagementRadius() const { return memory_management_radius_; }
  float getGridCellSize() const { return grid_cell_size_; }

  // Memory management
  void manageKeyframeMemory();  // Save distant keyframes, load nearby ones

  void printGridStatistics() const;

  // Legacy interface methods (for compatibility)
  void generateVisibilityBasedClusters() {}
  void generateKfidRandomShuffle() {}
  void fillQueue() {}
  std::vector<std::shared_ptr<GaussianKeyframe>> peekUpcomingKeyframes(
      size_t count) {
    return std::vector<std::shared_ptr<GaussianKeyframe>>();
  }
  void setIterationsPerCluster(int iterations) {}
  int getCurrentClusterIndex() const { return 0; }
  int getClusterCount() const { return 1; }
  void forceNextCluster() {}
  void visualizeClusters(const std::string& output_file = "",
                         int width = 800,
                         int height = 600) {}
  void addKeyframeToGrid(std::shared_ptr<GaussianKeyframe> keyframe);
  void updateKeyframeInGrid(std::shared_ptr<GaussianKeyframe> keyframe);

  void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> keyframe,
                                  int additional_uses = 1);

  int getNumberOfLoadedKeyframes() const { return loaded_keyframes_.size(); }
  void unloadAllKeyframes();

 private:
  // Core data structures
  std::shared_ptr<GaussianScene> scene_;
  size_t queue_size_;
  float selection_radius_;  // Radius around latest keyframe to select from
  float grid_cell_size_;    // Size of each grid cell

  // Spatial grid: maps grid coordinates to sets of keyframe IDs (ALL keyframes,
  // loaded or not)
  std::
      unordered_map<ChunkCoord, std::unordered_set<std::size_t>, ChunkCoordHash>
          spatial_grid_;

  // Track keyframe positions for efficient updates (ALL keyframes, loaded or
  // not)
  std::unordered_map<std::size_t, Eigen::Vector3f> keyframe_positions_;

  // Track which keyframes are currently loaded in memory
  std::unordered_set<std::size_t> loaded_keyframes_;

  // Track the most recent keyframe
  std::shared_ptr<GaussianKeyframe> latest_keyframe_;

  // Memory management radius - keyframes outside this radius get saved to disk
  float memory_management_radius_;

  // Usage tracking
  const std::map<std::size_t, float>* loss_map_;
  std::map<std::size_t, int>* used_times_map_;

  // Random number generation
  std::mt19937 rng_;

  // Thread synchronization
  mutable std::mutex grid_mutex_;

  // Async saving infrastructure
  std::queue<std::shared_ptr<GaussianKeyframe>> save_queue_;
  std::thread save_worker_;
  std::mutex save_mutex_;
  std::condition_variable save_cv_;
  bool save_worker_stop_ = false;

  // Helper methods
  ChunkCoord getGridCoord(const Eigen::Vector3f& position) const;
  std::vector<ChunkCoord> getNeighborGridCoords(const ChunkCoord& center_coord,
                                                float radius) const;
  std::vector<std::shared_ptr<GaussianKeyframe>> getKeyframesInRadius(
      const Eigen::Vector3f& center_position,
      float radius) const;

  // Memory management helpers
  void saveKeyframeToDisk(std::size_t keyframe_id);
  void loadKeyframeFromDisk(std::size_t keyframe_id);
  bool isKeyframeLoaded(std::size_t keyframe_id) const;

  // Async saving methods
  void saveWorker();
  void queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe);

  // Utility methods
  Eigen::Vector3f tensorToEigen(const torch::Tensor& tensor) const;
};