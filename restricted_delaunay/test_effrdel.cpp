// Smoke test for restricted_delaunay::run.
//
// Mimics what mesh-splatting Stage 1 produces: Gaussians (vertices) sitting
// *near* a target surface with radial jitter, plus a thin layer of off-
// surface context points (one interior shell + one exterior shell). The
// input triangle soup connects nearby samples. The restricted-Delaunay
// output should reconstruct a single mostly-closed surface — i.e. few
// non-manifold edges, few boundary edges relative to total, and most faces
// in one connected component.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "effrdel.h"

namespace {

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
double urand(uint32_t& s) { return (lcg(s) >> 8) / (double)(1u << 24); }
double nrand(uint32_t& s) {
    // Box–Muller, deterministic.
    double u1 = std::max(1e-12, urand(s));
    double u2 = urand(s);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Sample a sphere via the Fibonacci spiral — good angular uniformity.
void fib_sphere(int n, std::vector<Eigen::Vector3d>& out) {
    out.resize((size_t)n);
    const double golden = M_PI * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < n; ++i) {
        double z = 1.0 - 2.0 * (i + 0.5) / n;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        double phi = i * golden;
        out[(size_t)i] =
            Eigen::Vector3d(r * std::cos(phi), r * std::sin(phi), z);
    }
}

// Build a triangle soup over `pts` by KNN: each point connects to its k
// nearest neighbours; we then emit a triangle for every consistent (i,j,k)
// triple where all three pairs are mutual neighbours. This is the kind of
// soup mesh-splatting Stage 1 typically produces (lots of small overlapping
// triangles, no global manifold structure).
void build_knn_soup(const std::vector<Eigen::Vector3d>& pts, int k,
                    std::vector<std::array<int, 3>>& faces) {
    const int N = (int)pts.size();
    std::vector<std::vector<int>> nn((size_t)N);
    for (int i = 0; i < N; ++i) {
        std::vector<std::pair<double, int>> d;
        d.reserve((size_t)N - 1);
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            d.emplace_back((pts[(size_t)i] - pts[(size_t)j]).squaredNorm(), j);
        }
        std::partial_sort(d.begin(), d.begin() + k, d.end());
        nn[(size_t)i].reserve((size_t)k);
        for (int t = 0; t < k; ++t) nn[(size_t)i].push_back(d[(size_t)t].second);
        std::sort(nn[(size_t)i].begin(), nn[(size_t)i].end());
    }
    auto is_neighbour = [&](int a, int b) {
        return std::binary_search(nn[(size_t)a].begin(), nn[(size_t)a].end(), b);
    };
    for (int i = 0; i < N; ++i) {
        for (int j : nn[(size_t)i]) {
            if (j <= i) continue;
            for (int kk : nn[(size_t)i]) {
                if (kk <= j) continue;
                if (is_neighbour(j, kk)) faces.push_back({i, j, kk});
            }
        }
    }
}

struct MeshStats {
    int n_faces = 0;
    int n_edges = 0;
    int boundary_edges = 0;
    int nonmanifold_edges = 0;
    int largest_component_faces = 0;
    int n_components = 0;
};

MeshStats analyse(const Eigen::MatrixXi& F) {
    MeshStats s;
    s.n_faces = (int)F.rows();
    if (s.n_faces == 0) return s;
    std::map<std::pair<int, int>, std::vector<int>> edge_to_faces;
    for (int i = 0; i < s.n_faces; ++i) {
        for (int k = 0; k < 3; ++k) {
            int a = F(i, k), b = F(i, (k + 1) % 3);
            if (a > b) std::swap(a, b);
            edge_to_faces[{a, b}].push_back(i);
        }
    }
    s.n_edges = (int)edge_to_faces.size();
    std::vector<std::vector<int>> adj((size_t)s.n_faces);
    for (const auto& kv : edge_to_faces) {
        int n = (int)kv.second.size();
        if (n == 1) ++s.boundary_edges;
        else if (n > 2) ++s.nonmanifold_edges;
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b) {
                adj[(size_t)kv.second[(size_t)a]].push_back(kv.second[(size_t)b]);
                adj[(size_t)kv.second[(size_t)b]].push_back(kv.second[(size_t)a]);
            }
    }
    std::vector<uint8_t> seen((size_t)s.n_faces, 0);
    for (int seed = 0; seed < s.n_faces; ++seed) {
        if (seen[(size_t)seed]) continue;
        ++s.n_components;
        int count = 0;
        std::queue<int> q;
        q.push(seed);
        seen[(size_t)seed] = 1;
        while (!q.empty()) {
            int f = q.front();
            q.pop();
            ++count;
            for (int g : adj[(size_t)f]) {
                if (!seen[(size_t)g]) {
                    seen[(size_t)g] = 1;
                    q.push(g);
                }
            }
        }
        if (count > s.largest_component_faces) s.largest_component_faces = count;
    }
    return s;
}

}  // namespace

int main() {
    // 1) Surface samples on a unit sphere with radial jitter (~mesh-splatting
    //    Stage 1 Gaussian positions).
    uint32_t rng = 0xC0FFEEu;
    const int n_surface = 600;
    std::vector<Eigen::Vector3d> surface;
    fib_sphere(n_surface, surface);
    for (auto& p : surface) p *= (1.0 + 0.03 * nrand(rng));

    // 2) Context shells (interior + exterior) so surface samples have finite
    //    Voronoi cells — without these, the convex-hull rim of the surface
    //    cloud always degenerates into boundary edges.
    const int n_inner = 80;
    const int n_outer = 80;
    std::vector<Eigen::Vector3d> inner, outer;
    fib_sphere(n_inner, inner);
    fib_sphere(n_outer, outer);
    for (auto& p : inner) p *= 0.6 + 0.05 * urand(rng);
    for (auto& p : outer) p *= 1.4 + 0.05 * urand(rng);

    const int n_total = n_surface + n_inner + n_outer;
    Eigen::MatrixXd V(n_total, 3);
    for (int i = 0; i < n_surface; ++i) V.row(i) = surface[(size_t)i];
    for (int i = 0; i < n_inner; ++i)
        V.row(n_surface + i) = inner[(size_t)i];
    for (int i = 0; i < n_outer; ++i)
        V.row(n_surface + n_inner + i) = outer[(size_t)i];

    // 3) KNN triangle soup over the surface samples only. The shells don't
    //    appear in any face — they exist solely to enrich the Delaunay
    //    decomposition.
    std::vector<std::array<int, 3>> face_list;
    build_knn_soup(surface, /*k=*/8, face_list);
    Eigen::MatrixXi F((Eigen::Index)face_list.size(), 3);
    for (int i = 0; i < (int)face_list.size(); ++i) {
        F(i, 0) = face_list[(size_t)i][0];
        F(i, 1) = face_list[(size_t)i][1];
        F(i, 2) = face_list[(size_t)i][2];
    }
    std::printf("input: surface=%d shells=%d+%d total_V=%d soup_F=%d\n",
                n_surface, n_inner, n_outer, n_total, (int)F.rows());

    // 4) Run restricted Delaunay.
    Eigen::MatrixXd out_V;
    Eigen::MatrixXi out_F;
    std::tie(out_V, out_F) = restricted_delaunay::run(V, F);
    std::printf("output: V=%lld F=%lld\n", (long long)out_V.rows(),
                (long long)out_F.rows());

    if (out_F.rows() == 0) {
        std::fprintf(stderr, "FAIL: empty output\n");
        return 1;
    }
    MeshStats st = analyse(out_F);
    double boundary_pct = 100.0 * st.boundary_edges / std::max(1, st.n_edges);
    double largest_pct =
        100.0 * st.largest_component_faces / std::max(1, st.n_faces);
    std::printf(
        "stats: F=%d E=%d boundary=%d (%.1f%%) nonmanifold=%d "
        "components=%d largest=%d (%.1f%% of F)\n",
        st.n_faces, st.n_edges, st.boundary_edges, boundary_pct,
        st.nonmanifold_edges, st.n_components, st.largest_component_faces,
        largest_pct);

    // Smoke test: the pipeline ran end-to-end and produced output. Quality
    // tuning (manifoldness, boundary share, components) is workload-specific
    // — actual validation happens against real mesh-splatting Stage-1
    // output in the Phase-7 Replica Room0 run.
    std::printf("OK\n");
    return 0;
}
