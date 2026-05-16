#include "intersect.h"

#include <cmath>
#include <stdexcept>

namespace restricted_delaunay {

namespace {

inline lbvh::Vec3f sub(const lbvh::Vec3f& a, const lbvh::Vec3f& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline lbvh::Vec3f cross(const lbvh::Vec3f& a, const lbvh::Vec3f& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline float dot(const lbvh::Vec3f& a, const lbvh::Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float norm_sqr(const lbvh::Vec3f& a) { return dot(a, a); }

// Slab test: segment p0->p1 vs AABB.
inline bool seg_aabb_hit(const lbvh::Vec3f& p0, const lbvh::Vec3f& p1,
                         const lbvh::Vec3f& bmin, const lbvh::Vec3f& bmax) {
    const float eps = 1e-9f;
    const float dx = p1.x - p0.x, dy = p1.y - p0.y, dz = p1.z - p0.z;
    float tmin = 0.0f, tmax = 1.0f;
    const float p[3] = {p0.x, p0.y, p0.z};
    const float d[3] = {dx, dy, dz};
    const float lo[3] = {bmin.x, bmin.y, bmin.z};
    const float hi[3] = {bmax.x, bmax.y, bmax.z};
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(d[k]) < eps) {
            if (p[k] < lo[k] || p[k] > hi[k]) return false;
        } else {
            float invd = 1.0f / d[k];
            float t0 = (lo[k] - p[k]) * invd;
            float t1 = (hi[k] - p[k]) * invd;
            float a = std::min(t0, t1), b = std::max(t0, t1);
            tmin = std::max(tmin, a);
            tmax = std::min(tmax, b);
            if (tmin > tmax) return false;
        }
    }
    return true;
}

// Möller–Trumbore segment / triangle intersection, boundary-inclusive.
inline bool seg_tri_hit(const lbvh::Vec3f& p0, const lbvh::Vec3f& p1,
                        const lbvh::Vec3f& v0, const lbvh::Vec3f& v1,
                        const lbvh::Vec3f& v2) {
    const float eps = 1e-9f;
    lbvh::Vec3f dir = sub(p1, p0);
    lbvh::Vec3f e1 = sub(v1, v0);
    lbvh::Vec3f e2 = sub(v2, v0);
    lbvh::Vec3f pvec = cross(dir, e2);
    float det = dot(e1, pvec);
    if (std::fabs(det) <= eps) return false;
    float inv_det = 1.0f / det;
    lbvh::Vec3f tvec = sub(p0, v0);
    float u = dot(tvec, pvec) * inv_det;
    lbvh::Vec3f qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * inv_det;
    float t = dot(e2, qvec) * inv_det;
    if (u < -eps || v < -eps || (u + v) > 1.0f + eps) return false;
    if (t < -eps || t > 1.0f + eps) return false;
    // Degenerate filter
    float area2 = norm_sqr(cross(e1, e2));
    float seg2 = norm_sqr(dir);
    return area2 > eps * eps && seg2 > eps * eps;
}

constexpr int kMaxStack = 96;

}  // namespace

std::vector<uint8_t> segments_any_hit(
    const lbvh::BVH& bvh,
    const std::vector<lbvh::Vec3f>& verts_f,
    const std::vector<std::array<int, 3>>& faces,
    const Eigen::MatrixXf& P0,
    const Eigen::MatrixXf& P1) {
    if (P0.rows() != P1.rows() || P0.cols() != 3 || P1.cols() != 3) {
        throw std::invalid_argument(
            "segments_any_hit: P0/P1 must be (E,3) with matching E");
    }
    const int E = static_cast<int>(P0.rows());
    std::vector<uint8_t> hits(E, 0);
    if (E == 0 || bvh.nodeCount() == 0) return hits;

    const int root = bvh.root;

#pragma omp parallel for schedule(dynamic, 256)
    for (int e = 0; e < E; ++e) {
        const lbvh::Vec3f p0(P0(e, 0), P0(e, 1), P0(e, 2));
        const lbvh::Vec3f p1(P1(e, 0), P1(e, 1), P1(e, 2));

        if (!seg_aabb_hit(p0, p1, bvh.node_min[root], bvh.node_max[root]))
            continue;

        int stack[kMaxStack];
        int sp = 0;
        stack[sp++] = root;
        bool hit = false;

        while (sp > 0 && !hit) {
            int n = stack[--sp];
            if (!seg_aabb_hit(p0, p1, bvh.node_min[n], bvh.node_max[n]))
                continue;
            if (bvh.isLeaf(n)) {
                int first = bvh.leaf_first[n];
                int count = bvh.leaf_count[n];
                for (int i = 0; i < count; ++i) {
                    int tri = bvh.tri_idx_sorted[first + i];
                    const auto& f = faces[(size_t)tri];
                    if (seg_tri_hit(p0, p1, verts_f[(size_t)f[0]],
                                    verts_f[(size_t)f[1]],
                                    verts_f[(size_t)f[2]])) {
                        hit = true;
                        break;
                    }
                }
            } else {
                int L = bvh.left[n], R = bvh.right[n];
                if (sp + 2 <= kMaxStack) {
                    stack[sp++] = L;
                    stack[sp++] = R;
                } else {
                    stack[sp++] = L;  // truncation (extremely rare)
                }
            }
        }
        hits[(size_t)e] = hit ? 1 : 0;
    }
    return hits;
}

}  // namespace restricted_delaunay
