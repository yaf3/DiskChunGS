#include "effrdel.h"

#include <stdexcept>

namespace restricted_delaunay {

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
run(const Eigen::MatrixXd& /*verts*/, const Eigen::MatrixXi& /*faces*/) {
    // Phase 1.1 scaffold: implementation lands in Phase 1.2.
    throw std::logic_error(
        "restricted_delaunay::run() not implemented yet (Phase 1.2)");
}

}  // namespace restricted_delaunay
