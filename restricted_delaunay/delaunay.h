#pragma once
// Tetgen-backed 3D Delaunay tetrahedralization of a point cloud.

#include <Eigen/Core>

namespace restricted_delaunay {

struct DelaunayOut {
    // tets: (T,4) int — indices into the original input vertex array.
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> tets;
    // neighbors: (T,4) int — neighbor tet index opposite each vertex of each
    // tet, or -1 if the face is on the convex hull boundary.
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> neighbors;
};

DelaunayOut tetrahedralize_delaunay(const Eigen::MatrixXd& verts);

}  // namespace restricted_delaunay
