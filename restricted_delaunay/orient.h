#pragma once
// BFS-based face orientation pass (igl::bfs_orient analogue).
// Flips faces so adjacent triangles share oriented half-edges consistently.

#include <Eigen/Core>

namespace restricted_delaunay {

// faces: (F,3) int — vertex indices. Returns (F,3) with consistent winding
// per connected component. Topology (vertex set, face count) is preserved.
Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>
bfs_orient(const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& faces);

}  // namespace restricted_delaunay
