#include "orient.h"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace restricted_delaunay {

namespace {

// Hash an undirected edge (a,b) into a single 64-bit key. Vertex indices are
// non-negative and well below 2^31 in practice, so packing min/max into the
// high/low halves is collision-free.
inline uint64_t edge_key(int a, int b) {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    return (static_cast<uint64_t>(static_cast<uint32_t>(lo)) << 32) |
           static_cast<uint32_t>(hi);
}

}  // namespace

Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>
bfs_orient(const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& faces_in) {
    const int F = static_cast<int>(faces_in.rows());
    auto faces = faces_in;
    if (F == 0) return faces;

    // For each undirected edge, list the (face_idx, edge_local_idx) entries
    // that touch it. edge_local 0 = (v0,v1), 1 = (v1,v2), 2 = (v2,v0).
    struct EdgeRef {
        int face;
        int local;
    };
    std::unordered_map<uint64_t, std::vector<EdgeRef>> edge_map;
    edge_map.reserve((size_t)F * 3);
    for (int f = 0; f < F; ++f) {
        for (int k = 0; k < 3; ++k) {
            int a = faces(f, k);
            int b = faces(f, (k + 1) % 3);
            edge_map[edge_key(a, b)].push_back({f, k});
        }
    }

    std::vector<uint8_t> visited((size_t)F, 0);
    for (int seed = 0; seed < F; ++seed) {
        if (visited[(size_t)seed]) continue;
        visited[(size_t)seed] = 1;
        std::queue<int> q;
        q.push(seed);
        while (!q.empty()) {
            int f = q.front();
            q.pop();
            for (int k = 0; k < 3; ++k) {
                int a = faces(f, k);
                int b = faces(f, (k + 1) % 3);
                auto it = edge_map.find(edge_key(a, b));
                if (it == edge_map.end()) continue;
                for (const auto& ref : it->second) {
                    if (ref.face == f) continue;
                    int g = ref.face;
                    if (visited[(size_t)g]) continue;
                    // Edge (a,b) appears in face f. For consistent winding,
                    // face g must see the same edge in reverse order (b,a).
                    int ga = faces(g, ref.local);
                    int gb = faces(g, (ref.local + 1) % 3);
                    bool reversed = (ga == b && gb == a);
                    if (!reversed) {
                        // Flip g: swap v1 and v2.
                        std::swap(faces(g, 1), faces(g, 2));
                    }
                    visited[(size_t)g] = 1;
                    q.push(g);
                }
            }
        }
    }
    return faces;
}

}  // namespace restricted_delaunay
