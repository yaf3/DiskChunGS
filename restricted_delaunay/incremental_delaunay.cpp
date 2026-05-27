#include "incremental_delaunay.h"

#include <stdexcept>
#include <vector>

#include "tetgen.h"

namespace restricted_delaunay {

IncrementalDelaunay::IncrementalDelaunay() = default;
IncrementalDelaunay::~IncrementalDelaunay() = default;
IncrementalDelaunay::IncrementalDelaunay(IncrementalDelaunay&&) noexcept = default;
IncrementalDelaunay& IncrementalDelaunay::operator=(IncrementalDelaunay&&) noexcept = default;

void IncrementalDelaunay::initialize(const Eigen::MatrixXd& vertices) {
    if (vertices.cols() != 3 || vertices.rows() < 4) {
        throw std::invalid_argument(
            "IncrementalDelaunay::initialize: need >= 4 points in R^3");
    }
    const int V = static_cast<int>(vertices.rows());

    behavior_ = std::make_unique<tetgenbehavior>();
    behavior_->quiet = 1;
    behavior_->neighout = 1;
    behavior_->zeroindex = 1;

    in_ = std::make_unique<tetgenio>();
    in_->firstnumber = 0;
    in_->numberofpoints = V;
    in_->pointlist = new REAL[V * 3];
    for (int i = 0; i < V; ++i) {
        in_->pointlist[3 * i + 0] = vertices(i, 0);
        in_->pointlist[3 * i + 1] = vertices(i, 1);
        in_->pointlist[3 * i + 2] = vertices(i, 2);
    }

    mesh_ = std::make_unique<tetgenmesh>();
    mesh_->b = behavior_.get();
    mesh_->in = in_.get();

    mesh_->initializepools();
    mesh_->transfernodes();

    exactinit(behavior_->verbose, behavior_->noexact, behavior_->nostaticfilter,
              mesh_->xmax - mesh_->xmin, mesh_->ymax - mesh_->ymin,
              mesh_->zmax - mesh_->zmin);

    clock_t tv;
    mesh_->incrementaldelaunay(tv);

    // Allocate cave pools that insertpoint/insertpoint_abort use unconditionally
    // but initializepools only creates when b->plc || b->refine.
    if (!mesh_->cavetetshlist) {
        mesh_->caveshlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->caveshbdlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->cavesegshlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 4);
        mesh_->cavetetshlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->cavetetseglist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->caveencshlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->caveencseglist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->encseglist = new tetgenmesh::arraypool(sizeof(tetgenmesh::face), 8);
        mesh_->encshlist = new tetgenmesh::arraypool(sizeof(tetgenmesh::badface), 8);
    }

    num_vertices_ = V;
}

void IncrementalDelaunay::insertPoints(const Eigen::MatrixXd& new_verts) {
    if (!mesh_) throw std::runtime_error("IncrementalDelaunay: not initialized");
    const int N = static_cast<int>(new_verts.rows());
    if (N == 0) return;

    // Update bounding box.
    for (int i = 0; i < N; ++i) {
        double x = new_verts(i, 0), y = new_verts(i, 1), z = new_verts(i, 2);
        if (x < mesh_->xmin) mesh_->xmin = x;
        if (x > mesh_->xmax) mesh_->xmax = x;
        if (y < mesh_->ymin) mesh_->ymin = y;
        if (y > mesh_->ymax) mesh_->ymax = y;
        if (z < mesh_->zmin) mesh_->zmin = z;
        if (z > mesh_->zmax) mesh_->zmax = z;
    }

    // Allocate point objects via tetgen's pool and set coordinates.
    // makepoint auto-assigns pointmark from points->items and in->firstnumber.
    tetgenmesh::point* insertarray = new tetgenmesh::point[N];
    for (int i = 0; i < N; ++i) {
        mesh_->makepoint(&insertarray[i], tetgenmesh::UNUSEDVERTEX);
        insertarray[i][0] = new_verts(i, 0);
        insertarray[i][1] = new_verts(i, 1);
        insertarray[i][2] = new_verts(i, 2);
    }

    tetgenmesh::insertvertexflags ivf;
    tetgenmesh::flipconstraints fc;
    ivf.bowywat = 1;
    ivf.lawson = 0;
    ivf.validflag = 1;
    ivf.rejflag = 0;
    ivf.chkencflag = 0;
    ivf.sloc = static_cast<int>(tetgenmesh::INSTAR);
    ivf.sbowywat = 3;
    ivf.splitbdflag = 0;
    ivf.respectbdflag = 0;
    ivf.assignmeshsize = 0;

    long bak_samples = mesh_->samples;
    mesh_->samples = 3l;

    for (int i = 0; i < N; ++i) {
        tetgenmesh::triface searchtet;
        searchtet.tet = NULL;
        ivf.iloc = mesh_->scoutpoint(insertarray[i], &searchtet, 0);

        mesh_->setpointtype(insertarray[i], tetgenmesh::FREEVOLVERTEX);

        tetgenmesh::face splitsh, splitseg;
        splitsh.sh = NULL;
        splitseg.sh = NULL;

        if (mesh_->insertpoint(insertarray[i], &searchtet, &splitsh, &splitseg,
                               &ivf)) {
            if (mesh_->flipstack != NULL) {
                fc.enqflag = 2;
                mesh_->lawsonflip3d(&fc);
                mesh_->unflipqueue->restart();
            }
        } else {
            mesh_->setpointtype(insertarray[i], tetgenmesh::UNUSEDVERTEX);
            mesh_->unuverts++;
        }
    }

    mesh_->samples = bak_samples;

    delete[] insertarray;
    num_vertices_ += N;
}

DelaunayOut IncrementalDelaunay::extractTetsAndNeighbors() {
    if (!mesh_) throw std::runtime_error("IncrementalDelaunay: not initialized");

    DelaunayOut result;
    const long ntets = mesh_->tetrahedrons->items - mesh_->hullsize;
    if (ntets <= 0) return result;

    result.tets.resize(static_cast<int>(ntets), 4);
    result.neighbors.resize(static_cast<int>(ntets), 4);

    // First pass: assign element indices and extract vertex marks.
    mesh_->tetrahedrons->traversalinit();
    tetgenmesh::tetrahedron* tptr = mesh_->tetrahedrontraverse();
    int idx = 0;
    while (tptr != NULL) {
        mesh_->setelemindex(tptr, idx);
        tetgenmesh::point p1, p2, p3, p4;
        if (!mesh_->b->reversetetori) {
            p1 = (tetgenmesh::point) tptr[4];
            p2 = (tetgenmesh::point) tptr[5];
        } else {
            p1 = (tetgenmesh::point) tptr[5];
            p2 = (tetgenmesh::point) tptr[4];
        }
        p3 = (tetgenmesh::point) tptr[6];
        p4 = (tetgenmesh::point) tptr[7];
        result.tets(idx, 0) = mesh_->pointmark(p1);
        result.tets(idx, 1) = mesh_->pointmark(p2);
        result.tets(idx, 2) = mesh_->pointmark(p3);
        result.tets(idx, 3) = mesh_->pointmark(p4);
        ++idx;
        tptr = mesh_->tetrahedrontraverse();
    }

    // Second pass: extract neighbor indices using the assigned element indices.
    mesh_->tetrahedrons->traversalinit();
    tetgenmesh::triface tetloop, tetsym;
    tetloop.tet = mesh_->tetrahedrontraverse();
    idx = 0;
    while (tetloop.tet != NULL) {
        for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++) {
            mesh_->fsym(tetloop, tetsym);
            if (!mesh_->ishulltet(tetsym)) {
                result.neighbors(idx, tetloop.ver) =
                    mesh_->elemindex(tetsym.tet);
            } else {
                result.neighbors(idx, tetloop.ver) = -1;
            }
        }
        tetloop.tet = mesh_->tetrahedrontraverse();
        ++idx;
    }

    return result;
}

void IncrementalDelaunay::serialize(std::ostream& out) {
    if (!mesh_) {
        int32_t nv = 0;
        out.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
        return;
    }

    // Collect coordinates first so we can write the accurate count.
    std::vector<double> coords;
    coords.reserve(static_cast<size_t>(mesh_->points->items) * 3);
    mesh_->points->traversalinit();
    tetgenmesh::point pt = mesh_->pointtraverse();
    while (pt != NULL) {
        coords.push_back(pt[0]);
        coords.push_back(pt[1]);
        coords.push_back(pt[2]);
        pt = mesh_->pointtraverse();
    }

    int32_t nv = static_cast<int32_t>(coords.size() / 3);
    out.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
    out.write(reinterpret_cast<const char*>(coords.data()),
              static_cast<std::streamsize>(coords.size() * sizeof(double)));
}

void IncrementalDelaunay::deserialize(std::istream& in) {
    int32_t nv;
    in.read(reinterpret_cast<char*>(&nv), sizeof(nv));
    if (nv == 0) {
        mesh_.reset();
        behavior_.reset();
        in_.reset();
        num_vertices_ = 0;
        return;
    }
    Eigen::MatrixXd verts(nv, 3);
    for (int i = 0; i < nv; ++i) {
        double coords[3];
        in.read(reinterpret_cast<char*>(coords), sizeof(coords));
        verts(i, 0) = coords[0];
        verts(i, 1) = coords[1];
        verts(i, 2) = coords[2];
    }
    initialize(verts);
}

}  // namespace restricted_delaunay
