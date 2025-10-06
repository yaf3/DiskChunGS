#include "include/frustum_culler.h"

#include <omp.h>
#include <torch/torch.h>

#include <cmath>
#include <mutex>

#include "include/gaussian_keyframe.h"

FrustumCuller::FrustumCuller(const Eigen::Matrix4f& MVP) {
  // Extract frustum planes from MVP matrix
  // Left: MVP[3] + MVP[0]
  planes_[0] = MVP.row(3) + MVP.row(0);
  // Right: MVP[3] - MVP[0]
  planes_[1] = MVP.row(3) - MVP.row(0);
  // Bottom: MVP[3] + MVP[1]
  planes_[2] = MVP.row(3) + MVP.row(1);
  // Top: MVP[3] - MVP[1]
  planes_[3] = MVP.row(3) - MVP.row(1);
  // Near: MVP[3] + MVP[2]
  planes_[4] = MVP.row(3) + MVP.row(2);
  // Far: MVP[3] - MVP[2]
  planes_[5] = MVP.row(3) - MVP.row(2);

  // Normalize planes for accurate distance calculations
  for (int i = 0; i < 6; ++i) {
    float length = planes_[i].head<3>().norm();
    if (length > 0) {
      planes_[i] /= length;
    }
  }
}

FrustumTestResult FrustumCuller::test_AABB(const Eigen::Matrix4f& MVP,
                                           const AABB& aabb) {
  // Use the pre-extracted and normalized planes from constructor
  bool intersects = false;

  for (int i = 0; i < 6; ++i) {
    Eigen::Vector3f normal = planes_[i].head<3>();
    float d = planes_[i].w();

    // Get the "positive vertex" (farthest in direction of plane normal)
    Eigen::Vector3f positive_vertex = aabb.min;
    if (normal.x() > 0) positive_vertex.x() = aabb.max.x();
    if (normal.y() > 0) positive_vertex.y() = aabb.max.y();
    if (normal.z() > 0) positive_vertex.z() = aabb.max.z();

    // Get the "negative vertex" (nearest in direction of plane normal)
    Eigen::Vector3f negative_vertex = aabb.max;
    if (normal.x() > 0) negative_vertex.x() = aabb.min.x();
    if (normal.y() > 0) negative_vertex.y() = aabb.min.y();
    if (normal.z() > 0) negative_vertex.z() = aabb.min.z();

    // Distance from plane to vertices
    float pos_distance = normal.dot(positive_vertex) + d;
    float neg_distance = normal.dot(negative_vertex) + d;

    // If both vertices are behind plane, AABB is outside
    if (pos_distance < 0 && neg_distance < 0) {
      return FrustumTestResult::OUTSIDE;
    }

    // If one vertex is behind plane, AABB intersects
    if (pos_distance < 0 || neg_distance < 0) {
      intersects = true;
    }
  }

  return intersects ? FrustumTestResult::INTERSECT : FrustumTestResult::INSIDE;
}

std::vector<ChunkCoord> cullChunksHierarchical(
    const Eigen::Matrix4f& view_projection_matrix,
    const Eigen::Vector3f& camera_position,
    const ChunkCoord& camera_chunk,
    int search_radius,
    float chunk_size,
    float max_distance) {
  FrustumCuller culler(view_projection_matrix);

  std::vector<ChunkCoord> visible_chunks;
  visible_chunks.reserve(1000);  // Reasonable estimate

  // Mutex for thread-safe result collection
  std::mutex result_mutex;

  // Hierarchical culling function
  std::function<void(ChunkCoord, ChunkCoord, int)> cullRegion =
      [&](ChunkCoord min_coord, ChunkCoord max_coord, int depth) {
        // Create AABB for this region
        AABB region_aabb = getRegionAABB(min_coord, max_coord, chunk_size);

        // Test against frustum
        FrustumTestResult result =
            culler.test_AABB(view_projection_matrix, region_aabb);

        if (result == OUTSIDE) {
          // Entire region is outside - skip
          return;
        }

        // Calculate region size
        int64_t dx = max_coord.x - min_coord.x + 1;
        int64_t dy = max_coord.y - min_coord.y + 1;
        int64_t dz = max_coord.z - min_coord.z + 1;

        // If region is small or completely inside, process all chunks
        if (result == INSIDE || (dx <= 2 && dy <= 2 && dz <= 2) || depth > 4) {
          // For larger regions, use parallel processing
          bool use_parallel = (dx * dy * dz > 64) && (depth <= 2);

          if (use_parallel) {
            // Thread-local storage for results
            std::vector<ChunkCoord> local_chunks;
            local_chunks.reserve(dx * dy * dz);

// Parallel chunk processing
#pragma omp parallel
            {
              std::vector<ChunkCoord> thread_chunks;
              thread_chunks.reserve(64);

#pragma omp for collapse(3) schedule(dynamic, 8)
              for (int64_t x = min_coord.x; x <= max_coord.x; ++x) {
                for (int64_t y = min_coord.y; y <= max_coord.y; ++y) {
                  for (int64_t z = min_coord.z; z <= max_coord.z; ++z) {
                    ChunkCoord coord{x, y, z};

                    // Distance cull individual chunks
                    Eigen::Vector3f chunk_center =
                        getChunkCenter(coord, chunk_size);
                    float distance = (chunk_center - camera_position).norm();
                    if (distance <= max_distance) {
                      // Final per-chunk test if region was INTERSECT
                      if (result == INSIDE) {
                        thread_chunks.push_back(coord);
                      } else {
                        AABB chunk_aabb = getChunkAABB(coord, chunk_size);
                        if (culler.test_AABB(view_projection_matrix,
                                             chunk_aabb) != OUTSIDE) {
                          thread_chunks.push_back(coord);
                        }
                      }
                    }
                  }
                }
              }

              // Merge thread results
              if (!thread_chunks.empty()) {
                std::lock_guard<std::mutex> lock(result_mutex);
                visible_chunks.insert(visible_chunks.end(),
                                      thread_chunks.begin(),
                                      thread_chunks.end());
              }
            }
          } else {
            // Sequential processing for small regions
            std::vector<ChunkCoord> local_chunks;
            for (int64_t x = min_coord.x; x <= max_coord.x; ++x) {
              for (int64_t y = min_coord.y; y <= max_coord.y; ++y) {
                for (int64_t z = min_coord.z; z <= max_coord.z; ++z) {
                  ChunkCoord coord{x, y, z};

                  // Distance cull individual chunks
                  Eigen::Vector3f chunk_center =
                      getChunkCenter(coord, chunk_size);
                  float distance = (chunk_center - camera_position).norm();
                  if (distance <= max_distance) {
                    // Final per-chunk test if region was INTERSECT
                    if (result == INSIDE) {
                      local_chunks.push_back(coord);
                    } else {
                      AABB chunk_aabb = getChunkAABB(coord, chunk_size);
                      if (culler.test_AABB(view_projection_matrix,
                                           chunk_aabb) != OUTSIDE) {
                        local_chunks.push_back(coord);
                      }
                    }
                  }
                }
              }
            }

            // Add to main result
            if (!local_chunks.empty()) {
              std::lock_guard<std::mutex> lock(result_mutex);
              visible_chunks.insert(visible_chunks.end(), local_chunks.begin(),
                                    local_chunks.end());
            }
          }
          return;
        }

        // Subdivide region - use parallel sections for large regions at shallow
        // depth
        int64_t mid_x = (min_coord.x + max_coord.x) / 2;
        int64_t mid_y = (min_coord.y + max_coord.y) / 2;
        int64_t mid_z = (min_coord.z + max_coord.z) / 2;

        // Use parallel sections for subdivision at shallow depths
        if (depth <= 1 && (dx * dy * dz > 512)) {
#pragma omp parallel sections
          {
#pragma omp section
            cullRegion({min_coord.x, min_coord.y, min_coord.z},
                       {mid_x, mid_y, mid_z}, depth + 1);
#pragma omp section
            cullRegion({mid_x + 1, min_coord.y, min_coord.z},
                       {max_coord.x, mid_y, mid_z}, depth + 1);
#pragma omp section
            cullRegion({min_coord.x, mid_y + 1, min_coord.z},
                       {mid_x, max_coord.y, mid_z}, depth + 1);
#pragma omp section
            cullRegion({mid_x + 1, mid_y + 1, min_coord.z},
                       {max_coord.x, max_coord.y, mid_z}, depth + 1);
#pragma omp section
            cullRegion({min_coord.x, min_coord.y, mid_z + 1},
                       {mid_x, mid_y, max_coord.z}, depth + 1);
#pragma omp section
            cullRegion({mid_x + 1, min_coord.y, mid_z + 1},
                       {max_coord.x, mid_y, max_coord.z}, depth + 1);
#pragma omp section
            cullRegion({min_coord.x, mid_y + 1, mid_z + 1},
                       {mid_x, max_coord.y, max_coord.z}, depth + 1);
#pragma omp section
            cullRegion({mid_x + 1, mid_y + 1, mid_z + 1},
                       {max_coord.x, max_coord.y, max_coord.z}, depth + 1);
          }
        } else {
          // Sequential subdivision for deeper levels or smaller regions
          cullRegion({min_coord.x, min_coord.y, min_coord.z},
                     {mid_x, mid_y, mid_z}, depth + 1);
          cullRegion({mid_x + 1, min_coord.y, min_coord.z},
                     {max_coord.x, mid_y, mid_z}, depth + 1);
          cullRegion({min_coord.x, mid_y + 1, min_coord.z},
                     {mid_x, max_coord.y, mid_z}, depth + 1);
          cullRegion({mid_x + 1, mid_y + 1, min_coord.z},
                     {max_coord.x, max_coord.y, mid_z}, depth + 1);
          cullRegion({min_coord.x, min_coord.y, mid_z + 1},
                     {mid_x, mid_y, max_coord.z}, depth + 1);
          cullRegion({mid_x + 1, min_coord.y, mid_z + 1},
                     {max_coord.x, mid_y, max_coord.z}, depth + 1);
          cullRegion({min_coord.x, mid_y + 1, mid_z + 1},
                     {mid_x, max_coord.y, max_coord.z}, depth + 1);
          cullRegion({mid_x + 1, mid_y + 1, mid_z + 1},
                     {max_coord.x, max_coord.y, max_coord.z}, depth + 1);
        }
      };

  // Start hierarchical culling
  ChunkCoord min_coord{camera_chunk.x - search_radius,
                       camera_chunk.y - search_radius,
                       camera_chunk.z - search_radius};
  ChunkCoord max_coord{camera_chunk.x + search_radius,
                       camera_chunk.y + search_radius,
                       camera_chunk.z + search_radius};

  cullRegion(min_coord, max_coord, 0);

  return visible_chunks;
}

// Standalone frustumCullChunks function
std::vector<ChunkCoord> frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    float chunk_size,
    FrustumCullingCache* cache) {
  if (!keyframe) {
    return {};
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  // Check cache if provided
  if (cache) {
    std::vector<ChunkCoord> cached_chunks;
    if (cache->getCached(keyframe_id, current_pose, cached_chunks)) {
      return cached_chunks;
    }
  }

  // Compute frustum culling
  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);
  torch::Tensor tensor_matrix = keyframe->projection_matrix_;

  // Ensure tensor is on CPU and contiguous
  tensor_matrix = tensor_matrix.cpu().contiguous();

  // Get data pointer and create Eigen matrix
  float* data_ptr = tensor_matrix.data_ptr<float>();
  Eigen::Matrix4f proj_matrix = Eigen::Map<Eigen::Matrix4f>(data_ptr);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Get camera position for chunk search
  Sophus::SE3d Twc = current_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position, chunk_size);

  // Calculate parameters
  int search_radius =
      std::ceil(keyframe->zfar_ / chunk_size * std::sqrt(3.0f)) + 2;
  float max_distance = keyframe->zfar_ + chunk_size * 1.732f;

  // Call hierarchical culling
  std::vector<ChunkCoord> visible_chunks =
      cullChunksHierarchical(vp_matrix, camera_position, camera_chunk,
                             search_radius, chunk_size, max_distance);

  // Update cache if provided
  if (cache) {
    cache->updateCache(keyframe_id, current_pose, visible_chunks);
  }

  return visible_chunks;
}