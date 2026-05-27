#pragma once

#include <Eigen/Core>
#include <iostream>
#include <memory>

#include "delaunay.h"  // for DelaunayOut

// Forward-declare tetgen types to avoid pulling the massive header into ours.
class tetgenmesh;
class tetgenbehavior;
class tetgenio;

namespace restricted_delaunay {

// Persistent wrapper around tetgen's tetgenmesh that supports incremental
// point insertion without rebuilding from scratch each time.
class IncrementalDelaunay {
public:
    IncrementalDelaunay();
    ~IncrementalDelaunay();

    // Non-copyable, movable.
    IncrementalDelaunay(const IncrementalDelaunay&) = delete;
    IncrementalDelaunay& operator=(const IncrementalDelaunay&) = delete;
    IncrementalDelaunay(IncrementalDelaunay&&) noexcept;
    IncrementalDelaunay& operator=(IncrementalDelaunay&&) noexcept;

    // Build initial Delaunay from a point set. Must have >= 4 non-coplanar points.
    void initialize(const Eigen::MatrixXd& vertices);

    // Insert new points into the existing tetrahedralization.
    void insertPoints(const Eigen::MatrixXd& new_vertices);

    // Extract current tetrahedra + neighbor adjacency.
    // Not const: traversal and setelemindex mutate internal state.
    DelaunayOut extractTetsAndNeighbors();

    // Serialize/deserialize for chunk eviction.
    void serialize(std::ostream& out);
    void deserialize(std::istream& in);

    int numVertices() const { return num_vertices_; }
    bool isInitialized() const { return mesh_ != nullptr; }

private:
    std::unique_ptr<tetgenmesh> mesh_;
    std::unique_ptr<tetgenbehavior> behavior_;
    std::unique_ptr<tetgenio> in_;
    int num_vertices_ = 0;
};

}  // namespace restricted_delaunay
