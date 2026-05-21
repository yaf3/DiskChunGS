# restricted_delaunay: C++ port of effrdel (mesh-splatting restricted Delaunay)
#
# Public API: restricted_delaunay/effrdel.h
#   std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
#     run(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& faces);
#
# TetGen 1.5.1 is vendored in-tree under slam_deps/tetgen/ (AGPL-3, GPL-3 §13).

set(TETGEN_SRC_DIR ${PROJECT_SOURCE_DIR}/slam_deps/tetgen)

# --- Vendored TetGen ----------------------------------------------------------
# Built as an OBJECT library so it links into restricted_delaunay without
# producing a separately-installed .so. Tetgen requires TETLIBRARY when used
# as a library (rather than as the standalone tetgen binary).
add_library(
    tetgen_vendored
    OBJECT
    ${TETGEN_SRC_DIR}/tetgen.cxx
    ${TETGEN_SRC_DIR}/predicates.cxx
)
target_include_directories(tetgen_vendored PUBLIC ${TETGEN_SRC_DIR})
target_compile_definitions(tetgen_vendored PUBLIC TETLIBRARY)
# Tetgen is legacy C++; silence the noisiest warnings so it doesn't drown
# our own build log. -fPIC so it can link into a SHARED library.
target_compile_options(tetgen_vendored PRIVATE -w -fPIC)

# --- restricted_delaunay ------------------------------------------------------
# NOTE: Phase 1.1 scaffold only carries effrdel.{h,cpp}; the lbvh/delaunay/
# tetgrid/intersect/orient translation units land in Phase 1.2.
add_library(
    restricted_delaunay
    SHARED
    effrdel.h
    effrdel.cpp
    $<TARGET_OBJECTS:tetgen_vendored>
)

target_include_directories(restricted_delaunay PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TETGEN_SRC_DIR}
)

target_link_libraries(restricted_delaunay
    Eigen3::Eigen
)

target_compile_definitions(restricted_delaunay PRIVATE TETLIBRARY)
