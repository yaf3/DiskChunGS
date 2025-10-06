#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "include/chunk_types.h"

// Forward declarations
class GaussianKeyframe;

enum FrustumTestResult { OUTSIDE = 0, INTERSECT = 1, INSIDE = 2 };

class FrustumCuller {
 public:
  Eigen::Vector4f planes_[6];

 public:
  explicit FrustumCuller(const Eigen::Matrix4f& MVP);
  FrustumTestResult test_AABB(const Eigen::Matrix4f& MVP, const AABB& aabb);
};

// Standalone frustum culling function
std::vector<ChunkCoord> cullChunksHierarchical(
    const Eigen::Matrix4f& view_projection_matrix,
    const Eigen::Vector3f& camera_position,
    const ChunkCoord& camera_chunk,
    int search_radius,
    float chunk_size,
    float max_distance);

// Reusable frustum culling cache
class FrustumCullingCache {
 public:
  struct CacheEntry {
    Sophus::SE3d pose;  // Keyframe pose when visibility was calculated
    std::vector<ChunkCoord> visible_chunks;  // Visible chunk coordinates
    std::chrono::steady_clock::time_point
        timestamp;  // When this cache entry was created/updated
  };

  FrustumCullingCache(float chunk_size,
                      std::chrono::seconds expiry_time = std::chrono::seconds(10),
                      size_t max_entries = 100)
      : chunk_size_(chunk_size),
        cache_expiry_time_(expiry_time),
        max_cache_entries_(max_entries) {}

  // Try to get cached result
  bool getCached(size_t keyframe_id,
                 const Sophus::SE3d& current_pose,
                 std::vector<ChunkCoord>& out_chunks) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto cache_it = cache_.find(keyframe_id);

    if (cache_it != cache_.end()) {
      auto& entry = cache_it->second;
      if ((now - entry.timestamp) < cache_expiry_time_ &&
          poseNearlyEqual(current_pose, entry.pose)) {
        entry.timestamp = now;  // Update timestamp
        out_chunks = entry.visible_chunks;
        return true;
      }
    }
    return false;
  }

  // Update cache with new result
  void updateCache(size_t keyframe_id,
                   const Sophus::SE3d& pose,
                   const std::vector<ChunkCoord>& visible_chunks) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    CacheEntry entry;
    entry.pose = pose;
    entry.visible_chunks = visible_chunks;
    entry.timestamp = std::chrono::steady_clock::now();
    cache_[keyframe_id] = entry;

    // Limit cache size
    if (cache_.size() > max_cache_entries_) {
      // Remove oldest entry
      auto oldest = cache_.begin();
      for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->second.timestamp < oldest->second.timestamp) {
          oldest = it;
        }
      }
      cache_.erase(oldest);
    }
  }

  void clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
  }

 private:
  float chunk_size_;
  std::chrono::seconds cache_expiry_time_;
  size_t max_cache_entries_;
  std::unordered_map<size_t, CacheEntry> cache_;
  std::mutex cache_mutex_;

  // Helper to compare poses for cache validity
  bool poseNearlyEqual(const Sophus::SE3d& a, const Sophus::SE3d& b) {
    // Translation tolerance: small fraction of chunk size
    const double translation_tol = chunk_size_ * 0.05;  // 5% of chunk size

    // Rotation tolerance: a few degrees
    const double rotation_tol = 0.05;  // ~3 degrees in radians

    return (a.translation() - b.translation()).norm() < translation_tol &&
           a.unit_quaternion().angularDistance(b.unit_quaternion()) <
               rotation_tol;
  }
};

// Standalone frustumCullChunks function
std::vector<ChunkCoord> frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    float chunk_size,
    FrustumCullingCache* cache = nullptr);