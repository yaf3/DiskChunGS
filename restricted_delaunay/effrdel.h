#pragma once

// Restricted Delaunay surface reconstruction (C++ port of effrdel).
//
// Given a triangle soup (vertices + faces), produce an almost-watertight
// surface mesh by:
//   1. Building a 3D Delaunay tetrahedralization of the input vertices
//      (TetGen backend).
//   2. Intersecting each tet edge against the soup via an LBVH; edges that
//      cross the surface select dual triangles of the Delaunay complex.
//   3. Orienting the resulting face set with a BFS pass over the dual graph
//      so normals are consistent (required for downstream physics/sim use).
//
// Not bit-identical to the original gstaichi-based implementation; outputs
// are equivalent up to Delaunay tie-breaking and floating-point intersection
// tangents in general position.

#include <Eigen/Core>
#include <tuple>

namespace restricted_delaunay {

// verts: (V,3) double, faces: (F,3) int32. Returns (out_verts, out_faces).
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
run(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& faces);

}  // namespace restricted_delaunay
