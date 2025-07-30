#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

struct AABB {
  Eigen::Vector3f min;
  Eigen::Vector3f max;

  AABB() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
  AABB(const Eigen::Vector3f &min_val, const Eigen::Vector3f &max_val)
      : min(min_val), max(max_val) {}
};

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

// Get chunk coordinate from 3D position
inline ChunkCoord getChunkCoord(const Eigen::Vector3f &position,
                                float chunk_size) {
  float half_chunk = chunk_size * 0.5f;
  return ChunkCoord{static_cast<int64_t>(
                        std::floor((position.x() + half_chunk) / chunk_size)),
                    static_cast<int64_t>(
                        std::floor((position.y() + half_chunk) / chunk_size)),
                    static_cast<int64_t>(
                        std::floor((position.z() + half_chunk) / chunk_size))};
}

// Get chunk center
inline Eigen::Vector3f getChunkCenter(const ChunkCoord &coord,
                                      float chunk_size) {
  return Eigen::Vector3f(coord.x * chunk_size, coord.y * chunk_size,
                         coord.z * chunk_size);
}

// Calculate AABB for a chunk
inline AABB getChunkAABB(const ChunkCoord &coord, float chunk_size) {
  float half_chunk = chunk_size * 0.5f;
  Eigen::Vector3f center(coord.x * chunk_size, coord.y * chunk_size,
                         coord.z * chunk_size);
  Eigen::Vector3f min_corner = center - Eigen::Vector3f::Constant(half_chunk);
  Eigen::Vector3f max_corner = center + Eigen::Vector3f::Constant(half_chunk);
  return AABB(min_corner, max_corner);
}

inline AABB getRegionAABB(const ChunkCoord &min_coord,
                          const ChunkCoord &max_coord,
                          float chunk_size) {
  float half_chunk = chunk_size * 0.5f;

  Eigen::Vector3f min_pos(min_coord.x * chunk_size - half_chunk,
                          min_coord.y * chunk_size - half_chunk,
                          min_coord.z * chunk_size - half_chunk);

  Eigen::Vector3f max_pos((max_coord.x + 1) * chunk_size - half_chunk,
                          (max_coord.y + 1) * chunk_size - half_chunk,
                          (max_coord.z + 1) * chunk_size - half_chunk);

  return AABB(min_pos, max_pos);
}

// Encode/decode single chunk coordinates
inline int64_t encodeChunkCoord(const ChunkCoord& coord) {
  const int32_t OFFSET = 2048;
  
  int64_t x = coord.x + OFFSET;
  int64_t y = coord.y + OFFSET;
  int64_t z = coord.z + OFFSET;
  
  // 12 bits per coordinate = 36 total bits
  return x * (1 << 24) + y * (1 << 12) + z;
}

inline ChunkCoord decodeChunkCoord(int64_t chunk_id) {
  const int32_t OFFSET = 2048;
  
  int64_t z = (chunk_id % (1 << 12)) - OFFSET;
  int64_t y = ((chunk_id / (1 << 12)) % (1 << 12)) - OFFSET;
  int64_t x = (chunk_id / (1 << 24)) - OFFSET;
  
  return ChunkCoord{x, y, z};
}