#pragma once
// Segment / triangle-soup any-hit query via LBVH traversal.

#include <Eigen/Core>
#include <vector>

#include "lbvh.h"

namespace restricted_delaunay {

// For each segment (P0[i], P1[i]) return 1 if it hits any triangle in the
// soup, else 0. verts_f / faces describe the same soup the BVH was built on.
std::vector<uint8_t> segments_any_hit(
    const lbvh::BVH& bvh,
    const std::vector<lbvh::Vec3f>& verts_f,
    const std::vector<std::array<int, 3>>& faces,
    const Eigen::MatrixXf& P0,
    const Eigen::MatrixXf& P1);

}  // namespace restricted_delaunay
