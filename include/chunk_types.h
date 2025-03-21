#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "gaussian_model.h"

struct ChunkCoord {
  int64_t x, y, z;

  bool operator==(const ChunkCoord &other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator<(const ChunkCoord &other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

// Custom hash function for ChunkCoord
struct ChunkCoordHash {
  std::size_t operator()(const ChunkCoord &coord) const {
    // Simple hash combining function
    std::size_t h1 = std::hash<int>{}(coord.x);
    std::size_t h2 = std::hash<int>{}(coord.y);
    std::size_t h3 = std::hash<int>{}(coord.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

class Chunk {
 public:
  // Original constructor
  Chunk(const GaussianModelParams &model_params,
        const ChunkCoord &coord = ChunkCoord{0, 0, 0})
      : coord_(coord) {
    gaussians_ = std::make_shared<GaussianModel>(model_params);
  }

  // Add a default constructor to support cloning
  Chunk() = default;

  // Getter for coordinates
  const ChunkCoord &getCoord() const { return coord_; }

  // Setter for coordinates
  void setCoord(const ChunkCoord &coord) { coord_ = coord; }

  // Getter for Gaussians
  const std::shared_ptr<GaussianModel> &getGaussians() const {
    return gaussians_;
  }

 private:
  ChunkCoord coord_{0, 0, 0};
  std::shared_ptr<GaussianModel> gaussians_;
};