#include <Eigen/Dense>
#include <functional>
#include <vector>

#include "include/chunk_types.h"

struct AABB {
  Eigen::Vector3f min;
  Eigen::Vector3f max;

  AABB() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
  AABB(const Eigen::Vector3f& min_val, const Eigen::Vector3f& max_val)
      : min(min_val), max(max_val) {}
};

enum FrustumTestResult { OUTSIDE = 0, INTERSECT = 1, INSIDE = 2 };

class FrustumCuller {
 private:
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