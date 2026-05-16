#pragma once
// LBVH builder (Karras 2012) over triangles. CPU-only, single-threaded.
// Vendored from thirdparty/effrdel/src/lbvh.hpp with light cleanups.

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lbvh {

struct Vec3f {
    float x, y, z;
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3f operator+(const Vec3f& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3f operator-(const Vec3f& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline Vec3f minv(const Vec3f& a, const Vec3f& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3f maxv(const Vec3f& a, const Vec3f& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

struct AABB {
    Vec3f bmin, bmax;
    AABB() {
        const float inf = std::numeric_limits<float>::infinity();
        bmin = {inf, inf, inf};
        bmax = {-inf, -inf, -inf};
    }
    AABB(const Vec3f& mn, const Vec3f& mx) : bmin(mn), bmax(mx) {}
    void expand(const Vec3f& p) {
        bmin = minv(bmin, p);
        bmax = maxv(bmax, p);
    }
    void expand(const AABB& a) {
        bmin = minv(bmin, a.bmin);
        bmax = maxv(bmax, a.bmax);
    }
};

inline uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
inline uint32_t morton3D_10bit(float x, float y, float z) {
    auto clamp01 = [](float a) {
        return std::min(std::max(a, 0.0f), std::nextafter(1.0f, 0.0f));
    };
    x = clamp01(x);
    y = clamp01(y);
    z = clamp01(z);
    const uint32_t xx = (uint32_t)(x * 1024.0f);
    const uint32_t yy = (uint32_t)(y * 1024.0f);
    const uint32_t zz = (uint32_t)(z * 1024.0f);
    return (expandBits(xx) << 2) | (expandBits(yy) << 1) | (expandBits(zz));
}

inline int clz32(uint32_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    int n = 0;
    while ((x & 0x80000000u) == 0 && n < 32) {
        x <<= 1;
        ++n;
    }
    return x ? n : 32;
#endif
}

inline int commonPrefix(const std::vector<uint32_t>& code, int i, int j) {
    const int n = (int)code.size();
    if (j < 0 || j >= n) return -1;
    uint32_t ci = code[i];
    uint32_t cj = code[j];
    if (ci == cj) {
        uint32_t x = (uint32_t)(i ^ j);
        return 32 + clz32(x);
    }
    return clz32(ci ^ cj);
}

struct BVH {
    std::vector<Vec3f> node_min;
    std::vector<Vec3f> node_max;
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> parent;
    std::vector<int> leaf_first;
    std::vector<int> leaf_count;
    std::vector<int> tri_idx_sorted;

    int root = 0;
    int n_internal = 0;
    int n_leaves = 0;

    int nodeCount() const { return (int)node_min.size(); }
    bool isLeaf(int n) const { return left[n] < 0 && right[n] < 0; }
};

struct BuildInput {
    const std::vector<Vec3f>& verts;
    const std::vector<std::array<int, 3>>& faces;
};

inline BVH buildLBVH(const BuildInput& in) {
    const auto& V = in.verts;
    const auto& F = in.faces;
    const int N = (int)F.size();
    assert(N > 0);

    std::vector<AABB> triBBox(N);
    std::vector<Vec3f> centroid(N);
    AABB scene;
    for (int i = 0; i < N; ++i) {
        const auto& f = F[i];
        const Vec3f a = V[(size_t)f[0]];
        const Vec3f b = V[(size_t)f[1]];
        const Vec3f c = V[(size_t)f[2]];
        AABB bbi;
        bbi.expand(a);
        bbi.expand(b);
        bbi.expand(c);
        triBBox[i] = bbi;
        centroid[i] = (a + b + c) * (1.0f / 3.0f);
        scene.expand(bbi);
    }

    Vec3f smin = scene.bmin, smax = scene.bmax;
    Vec3f ext = smax - smin;
    ext.x = (std::fabs(ext.x) < 1e-20f) ? 1.0f : ext.x;
    ext.y = (std::fabs(ext.y) < 1e-20f) ? 1.0f : ext.y;
    ext.z = (std::fabs(ext.z) < 1e-20f) ? 1.0f : ext.z;

    std::vector<uint32_t> morton(N);
    std::vector<int> primIdx(N);
    for (int i = 0; i < N; ++i) {
        Vec3f q = {(centroid[i].x - smin.x) / ext.x,
                   (centroid[i].y - smin.y) / ext.y,
                   (centroid[i].z - smin.z) / ext.z};
        morton[i] = morton3D_10bit(q.x, q.y, q.z);
        primIdx[i] = i;
    }

    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (morton[a] < morton[b]) return true;
        if (morton[a] > morton[b]) return false;
        return primIdx[a] < primIdx[b];
    });

    std::vector<uint32_t> code(N);
    std::vector<int> primSorted(N);
    for (int i = 0; i < N; ++i) {
        code[i] = morton[order[i]];
        primSorted[i] = primIdx[order[i]];
    }

    const int nInternal = (N > 1) ? (N - 1) : 0;
    const int nLeaves = N;
    const int nNodes = nInternal + nLeaves;
    const int leafBase = nInternal;

    BVH bvh;
    bvh.node_min.resize(nNodes);
    bvh.node_max.resize(nNodes);
    bvh.left.assign(nNodes, -1);
    bvh.right.assign(nNodes, -1);
    bvh.parent.assign(nNodes, -1);
    bvh.leaf_first.assign(nNodes, -1);
    bvh.leaf_count.assign(nNodes, 0);
    bvh.tri_idx_sorted = primSorted;
    bvh.n_internal = nInternal;
    bvh.n_leaves = nLeaves;

    auto lcp = [&](int i, int j) { return commonPrefix(code, i, j); };

    auto determineRange = [&](int i, int n, int& first, int& last) {
        int lcpL = lcp(i, i - 1);
        int lcpR = lcp(i, i + 1);
        int d = (lcpR > lcpL) ? +1 : -1;

        int lcpMin = lcp(i, i - d);
        int lmax = 2;
        while ((i + lmax * d) >= 0 && (i + lmax * d) < n &&
               lcp(i, i + lmax * d) > lcpMin) {
            lmax *= 2;
        }
        int l = 0;
        int t = lmax;
        do {
            int step = (t + 1) >> 1;
            int idx = i + (l + step) * d;
            if (idx >= 0 && idx < n && lcp(i, idx) > lcpMin) l += step;
            t -= step;
        } while (t > 0);
        last = i + l * d;
        first = std::min(i, last);
        last = std::max(i, last);
    };

    auto findSplit = [&](int first, int last) -> int {
        int lcpFirstLast = lcp(first, last);
        if (lcpFirstLast >= 64) return (first + last) >> 1;
        int split = first;
        int step = last - first;
        do {
            step = (step + 1) >> 1;
            int newSplit = split + step;
            if (newSplit < last) {
                int lcpFirstNew = lcp(first, newSplit);
                if (lcpFirstNew > lcpFirstLast) split = newSplit;
            }
        } while (step > 1);
        return split;
    };

    if (N == 1) {
        bvh.leaf_first[0] = 0;
        bvh.leaf_count[0] = 1;
    } else {
        for (int i = 0; i < nInternal; ++i) {
            int first, last;
            determineRange(i, N, first, last);
            int split = findSplit(first, last);

            int leftChild = (split == first) ? (leafBase + split) : split;
            int rightChild = (split + 1 == last) ? (leafBase + split + 1) : (split + 1);

            bvh.left[i] = leftChild;
            bvh.right[i] = rightChild;
            bvh.parent[leftChild] = i;
            bvh.parent[rightChild] = i;
        }
        for (int i = 0; i < nLeaves; ++i) {
            int leaf = leafBase + i;
            bvh.leaf_first[leaf] = i;
            bvh.leaf_count[leaf] = 1;
        }
    }

    if (N == 1) {
        int tri = bvh.tri_idx_sorted[0];
        bvh.node_min[0] = triBBox[tri].bmin;
        bvh.node_max[0] = triBBox[tri].bmax;
    } else {
        for (int i = 0; i < nLeaves; ++i) {
            int leaf = leafBase + i;
            int tri = bvh.tri_idx_sorted[i];
            bvh.node_min[leaf] = triBBox[tri].bmin;
            bvh.node_max[leaf] = triBBox[tri].bmax;
        }
        std::vector<int> pending(nInternal, 2);
        std::vector<int> stack;
        stack.reserve(nInternal);
        for (int i = 0; i < nLeaves; ++i) {
            int p = bvh.parent[leafBase + i];
            if (p >= 0 && --pending[p] == 0) stack.push_back(p);
        }
        while (!stack.empty()) {
            int n = stack.back();
            stack.pop_back();
            int L = bvh.left[n];
            int R = bvh.right[n];
            AABB a(bvh.node_min[L], bvh.node_max[L]);
            a.expand(AABB(bvh.node_min[R], bvh.node_max[R]));
            bvh.node_min[n] = a.bmin;
            bvh.node_max[n] = a.bmax;

            int p = bvh.parent[n];
            if (p >= 0 && --pending[p] == 0) stack.push_back(p);
        }
    }

    bvh.root = 0;
    return bvh;
}

}  // namespace lbvh
