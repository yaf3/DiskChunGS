#include "delaunay.h"

#include <stdexcept>
#include <vector>

#include "tetgen.h"

namespace restricted_delaunay {

DelaunayOut tetrahedralize_delaunay(const Eigen::MatrixXd& verts) {
    if (verts.cols() != 3 || verts.rows() < 4) {
        throw std::invalid_argument(
            "tetrahedralize_delaunay: verts must be (V>=4, 3)");
    }

    tetgenio in, out;
    in.firstnumber = 0;
    in.numberofpoints = static_cast<int>(verts.rows());
    in.pointlist = new REAL[in.numberofpoints * 3];
    for (int i = 0; i < in.numberofpoints; ++i) {
        in.pointlist[3 * i + 0] = verts(i, 0);
        in.pointlist[3 * i + 1] = verts(i, 1);
        in.pointlist[3 * i + 2] = verts(i, 2);
    }

    // Switches: Q = quiet, n = produce neighbor list. Default mode (no -p,
    // no -r) produces a Delaunay tetrahedralization of the input point set.
    char switches[] = "Qn";
    try {
        tetrahedralize(switches, &in, &out);
    } catch (int err) {
        throw std::runtime_error(
            "tetgen::tetrahedralize failed (error code " +
            std::to_string(err) + ")");
    }

    if (out.numberoftetrahedra <= 0 || out.numberofcorners != 4) {
        throw std::runtime_error(
            "tetgen produced no tets or wrong cell type");
    }

    DelaunayOut result;
    const int T = out.numberoftetrahedra;
    result.tets.resize(T, 4);
    result.neighbors.resize(T, 4);
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < 4; ++k) {
            result.tets(i, k) = out.tetrahedronlist[4 * i + k];
            result.neighbors(i, k) =
                out.neighborlist ? out.neighborlist[4 * i + k] : -1;
        }
    }
    return result;
}

}  // namespace restricted_delaunay
