#include "effrdel.h"

#include <Eigen/Dense>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "delaunay.h"
#include "intersect.h"
#include "lbvh.h"
#include "orient.h"

namespace restricted_delaunay {

namespace {

// Circumcenter of a tetrahedron via the standard linear system
//   2 (p_i - p_0) . c  =  |p_i|^2 - |p_0|^2,   i = 1,2,3.
inline Eigen::Vector3d tet_circumcenter(const Eigen::Vector3d& p0,
                                        const Eigen::Vector3d& p1,
                                        const Eigen::Vector3d& p2,
                                        const Eigen::Vector3d& p3) {
    Eigen::Matrix3d A;
    A.row(0) = 2.0 * (p1 - p0).transpose();
    A.row(1) = 2.0 * (p2 - p0).transpose();
    A.row(2) = 2.0 * (p3 - p0).transpose();
    Eigen::Vector3d b(p1.squaredNorm() - p0.squaredNorm(),
                      p2.squaredNorm() - p0.squaredNorm(),
                      p3.squaredNorm() - p0.squaredNorm());
    // Near-degenerate tets fall back to the centroid; the segment-vs-soup
    // intersect test still answers "miss" sensibly when both endpoints
    // coincide.
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) {
        return (p0 + p1 + p2 + p3) * 0.25;
    }
    return lu.solve(b);
}

// Pack the three vertices of `tet` that are not the k-th, preserving the
// cyclic order around the opposite vertex. Sign of the parity isn't relevant
// here — restricted-Delaunay output is later re-oriented by BFS.
inline std::array<int, 3> face_opposite(const int* tet, int k) {
    static constexpr int kOpp[4][3] = {
        {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    return {tet[kOpp[k][0]], tet[kOpp[k][1]], tet[kOpp[k][2]]};
}

}  // namespace

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
run(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& faces) {
    if (verts.cols() != 3 || faces.cols() != 3) {
        throw std::invalid_argument(
            "restricted_delaunay::run: verts must be (V,3), faces must be (F,3)");
    }
    const int V = static_cast<int>(verts.rows());
    const int F = static_cast<int>(faces.rows());
    if (V < 4 || F < 1) {
        throw std::invalid_argument(
            "restricted_delaunay::run: need at least 4 verts and 1 face");
    }

    // 1) Build LBVH over the input triangle soup (float32).
    std::vector<lbvh::Vec3f> verts_f((size_t)V);
    for (int i = 0; i < V; ++i) {
        verts_f[(size_t)i] = lbvh::Vec3f(
            static_cast<float>(verts(i, 0)), static_cast<float>(verts(i, 1)),
            static_cast<float>(verts(i, 2)));
    }
    std::vector<std::array<int, 3>> faces_v((size_t)F);
    for (int i = 0; i < F; ++i) {
        faces_v[(size_t)i] = {faces(i, 0), faces(i, 1), faces(i, 2)};
    }
    lbvh::BuildInput bin{verts_f, faces_v};
    lbvh::BVH bvh = lbvh::buildLBVH(bin);

    // 2) Delaunay tetrahedralization of the vertex set (Tetgen).
    DelaunayOut dt = tetrahedralize_delaunay(verts);
    const int T = static_cast<int>(dt.tets.rows());
    if (T == 0) {
        return {verts, Eigen::MatrixXi(0, 3)};
    }

    // 3) Compute tet circumcenters.
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> tet_cc(T, 3);
    for (int t = 0; t < T; ++t) {
        Eigen::Vector3d p0 = verts.row(dt.tets(t, 0)).transpose();
        Eigen::Vector3d p1 = verts.row(dt.tets(t, 1)).transpose();
        Eigen::Vector3d p2 = verts.row(dt.tets(t, 2)).transpose();
        Eigen::Vector3d p3 = verts.row(dt.tets(t, 3)).transpose();
        tet_cc.row(t) = tet_circumcenter(p0, p1, p2, p3).transpose();
    }

    // 4) Enumerate interior Delaunay faces (those shared by two tets) and
    //    the Voronoi segments between their tet circumcenters.
    std::vector<std::array<int, 3>> cand_faces;
    cand_faces.reserve((size_t)T * 2);
    std::vector<std::pair<int, int>> cand_tets;
    cand_tets.reserve((size_t)T * 2);
    for (int t = 0; t < T; ++t) {
        const int* tv = &dt.tets(t, 0);
        for (int k = 0; k < 4; ++k) {
            int n = dt.neighbors(t, k);
            if (n < 0 || n <= t) continue;  // skip hull + dedupe
            cand_faces.push_back(face_opposite(tv, k));
            cand_tets.emplace_back(t, n);
        }
    }
    const int M = static_cast<int>(cand_faces.size());
    if (M == 0) {
        return {verts, Eigen::MatrixXi(0, 3)};
    }

    Eigen::MatrixXf P0(M, 3), P1(M, 3);
    for (int i = 0; i < M; ++i) {
        P0(i, 0) = static_cast<float>(tet_cc(cand_tets[(size_t)i].first, 0));
        P0(i, 1) = static_cast<float>(tet_cc(cand_tets[(size_t)i].first, 1));
        P0(i, 2) = static_cast<float>(tet_cc(cand_tets[(size_t)i].first, 2));
        P1(i, 0) = static_cast<float>(tet_cc(cand_tets[(size_t)i].second, 0));
        P1(i, 1) = static_cast<float>(tet_cc(cand_tets[(size_t)i].second, 1));
        P1(i, 2) = static_cast<float>(tet_cc(cand_tets[(size_t)i].second, 2));
    }

    // 5) BVH any-hit test on the Voronoi edges.
    std::vector<uint8_t> hits =
        segments_any_hit(bvh, verts_f, faces_v, P0, P1);

    // 6) Collect surviving faces.
    int n_hit = 0;
    for (auto h : hits) n_hit += (h != 0);
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> out_faces(n_hit, 3);
    int w = 0;
    for (int i = 0; i < M; ++i) {
        if (!hits[(size_t)i]) continue;
        out_faces(w, 0) = cand_faces[(size_t)i][0];
        out_faces(w, 1) = cand_faces[(size_t)i][1];
        out_faces(w, 2) = cand_faces[(size_t)i][2];
        ++w;
    }

    // 7) Orient face normals consistently across each connected component
    //    (required for downstream physics / sim consumption).
    out_faces = bfs_orient(out_faces);

    // The output references the same vertex array as the input; downstream
    // code that wants only used vertices can compact externally.
    return {verts, Eigen::MatrixXi(out_faces)};
}

}  // namespace restricted_delaunay
