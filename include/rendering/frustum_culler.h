/**
 * This file is part of DiskChunGS.
 *
 * Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "chunk_types.h"

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
  };

  explicit FrustumCullingCache(float chunk_size) : chunk_size_(chunk_size) {}

  // Try to get cached result
  bool getCached(size_t keyframe_id,
                 const Sophus::SE3d& current_pose,
                 std::vector<ChunkCoord>& out_chunks) {
    auto cache_it = cache_.find(keyframe_id);

    if (cache_it != cache_.end()) {
      auto& entry = cache_it->second;
      if (poseNearlyEqual(current_pose, entry.pose)) {
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
    CacheEntry entry;
    entry.pose = pose;
    entry.visible_chunks = visible_chunks;
    cache_[keyframe_id] = entry;
  }

  void clearCache() { cache_.clear(); }

 private:
  float chunk_size_;
  std::unordered_map<size_t, CacheEntry> cache_;

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