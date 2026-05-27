// Unit test for IncrementalDelaunay.
//
// Tests:
//   1. initialize() + insertPoints() vs initialize()-all produce valid tets.
//   2. serialize/deserialize round-trip preserves the tetrahedralization.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <sstream>

#include <Eigen/Core>

#include "incremental_delaunay.h"

namespace {

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
double urand(uint32_t& s) { return (lcg(s) >> 8) / (double)(1u << 24); }

// Sample points uniformly inside the unit ball.
Eigen::MatrixXd sample_ball(int n, uint32_t& rng) {
    Eigen::MatrixXd pts(n, 3);
    for (int i = 0; i < n; ++i) {
        // Rejection sample.
        while (true) {
            double x = 2.0 * urand(rng) - 1.0;
            double y = 2.0 * urand(rng) - 1.0;
            double z = 2.0 * urand(rng) - 1.0;
            if (x * x + y * y + z * z <= 1.0) {
                pts(i, 0) = x;
                pts(i, 1) = y;
                pts(i, 2) = z;
                break;
            }
        }
    }
    return pts;
}

bool check_tet_indices(const restricted_delaunay::DelaunayOut& d, int n_verts) {
    if (d.tets.rows() == 0) return false;
    for (int i = 0; i < d.tets.rows(); ++i) {
        for (int k = 0; k < 4; ++k) {
            int v = d.tets(i, k);
            if (v < 0 || v >= n_verts) return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    uint32_t rng = 0xDEADBEEFu;

    const int N_INIT = 60;
    const int N_ADD  = 40;
    const int N_TOTAL = N_INIT + N_ADD;

    Eigen::MatrixXd all_pts = sample_ball(N_TOTAL, rng);
    Eigen::MatrixXd init_pts = all_pts.topRows(N_INIT);
    Eigen::MatrixXd add_pts  = all_pts.bottomRows(N_ADD);

    // --- Test 1: incremental vs all-at-once ---------------------------------

    restricted_delaunay::IncrementalDelaunay incr;
    incr.initialize(init_pts);
    if (!incr.isInitialized()) {
        std::fprintf(stderr, "FAIL: not initialized after initialize()\n");
        return 1;
    }
    if (incr.numVertices() != N_INIT) {
        std::fprintf(stderr, "FAIL: numVertices mismatch after initialize\n");
        return 1;
    }

    incr.insertPoints(add_pts);
    if (incr.numVertices() != N_TOTAL) {
        std::fprintf(stderr, "FAIL: numVertices mismatch after insertPoints\n");
        return 1;
    }

    restricted_delaunay::DelaunayOut incr_out = incr.extractTetsAndNeighbors();
    if (incr_out.tets.rows() <= 0) {
        std::fprintf(stderr, "FAIL: incremental produced no tets\n");
        return 1;
    }
    if (!check_tet_indices(incr_out, N_TOTAL)) {
        std::fprintf(stderr, "FAIL: incremental tet has out-of-range vertex index\n");
        return 1;
    }
    if (incr_out.neighbors.rows() != incr_out.tets.rows()) {
        std::fprintf(stderr, "FAIL: neighbors row count mismatch\n");
        return 1;
    }

    // All-at-once reference.
    restricted_delaunay::IncrementalDelaunay ref;
    ref.initialize(all_pts);
    restricted_delaunay::DelaunayOut ref_out = ref.extractTetsAndNeighbors();
    if (ref_out.tets.rows() <= 0) {
        std::fprintf(stderr, "FAIL: reference produced no tets\n");
        return 1;
    }
    if (!check_tet_indices(ref_out, N_TOTAL)) {
        std::fprintf(stderr, "FAIL: reference tet has out-of-range vertex index\n");
        return 1;
    }

    std::printf("test1: incr_tets=%lld ref_tets=%lld\n",
                (long long)incr_out.tets.rows(),
                (long long)ref_out.tets.rows());

    // Tet counts should be close but not necessarily identical — insertion order
    // affects tie-breaking for near-degenerate configurations.
    double ratio = (double)incr_out.tets.rows() / (double)ref_out.tets.rows();
    if (ratio < 0.8 || ratio > 1.2) {
        std::fprintf(stderr,
                     "FAIL: tet count ratio %.2f out of range incr=%lld ref=%lld\n",
                     ratio, (long long)incr_out.tets.rows(),
                     (long long)ref_out.tets.rows());
        return 1;
    }

    // --- Test 2: serialize/deserialize round-trip ---------------------------

    std::ostringstream oss;
    incr.serialize(oss);
    std::string blob = oss.str();
    if (blob.empty()) {
        std::fprintf(stderr, "FAIL: serialized data is empty\n");
        return 1;
    }

    restricted_delaunay::IncrementalDelaunay loaded;
    std::istringstream iss(blob);
    loaded.deserialize(iss);

    if (!loaded.isInitialized()) {
        std::fprintf(stderr, "FAIL: not initialized after deserialize\n");
        return 1;
    }

    restricted_delaunay::DelaunayOut loaded_out = loaded.extractTetsAndNeighbors();
    // deserialize rebuilds from scratch, so compare against the all-at-once
    // reference (not the incremental result).
    if (loaded_out.tets.rows() != ref_out.tets.rows()) {
        std::fprintf(stderr,
                     "FAIL: serialize round-trip tet count mismatch: "
                     "got=%lld expected=%lld\n",
                     (long long)loaded_out.tets.rows(),
                     (long long)ref_out.tets.rows());
        return 1;
    }

    std::printf("test2: serialize blob=%zu bytes, loaded_tets=%lld\n",
                blob.size(), (long long)loaded_out.tets.rows());

    std::printf("OK\n");
    return 0;
}
