#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chunk_types.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"

// Removed forward declaration of GaussianMapper

// Type definitions
using u32 = uint32_t;

// Axis-aligned bounding box
struct AABB {
  Eigen::Vector3f min;
  Eigen::Vector3f max;

  AABB() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
  AABB(const Eigen::Vector3f& min_val, const Eigen::Vector3f& max_val)
      : min(min_val), max(max_val) {}
};

// Camera structure
struct Cam {
  Eigen::Matrix4f view;
  Eigen::Matrix4f projection;
  // Additional camera properties can be added here
};

// Pure Eigen implementation without explicit SIMD (relies on Eigen's
// optimizations)
bool test_AABB_against_frustum_eigen(const Eigen::Matrix4f& MVP,
                                     const AABB& aabb);

// Main culling function using Eigen types
void cull_AABBs_against_frustum(const Cam& camera,
                                const std::vector<Eigen::Matrix4f>& transforms,
                                const std::vector<AABB>& aabb_list,
                                std::vector<u32>& out_visible_list,
                                bool use_simd = true);

// Metadata for managing chunks lifecycle
struct ChunkMetadata {
  std::chrono::time_point<std::chrono::steady_clock> last_used;
  std::chrono::time_point<std::chrono::steady_clock> load_time;
  int usage_count;
  bool dirty;  // Has been modified since last save

  ChunkMetadata()
      : last_used(std::chrono::steady_clock::now()),
        load_time(std::chrono::steady_clock::now()),
        usage_count(0),
        dirty(false) {}
};

class ChunkManager {
 public:
  ChunkManager(const GaussianModelParams& model_params,
               const GaussianOptimizationParams& opt_params,
               std::filesystem::path chunk_save_dir,
               float chunk_size = 50.0f,
               float overlap_margin = 0.0f,
               int max_chunks = 50);

  ~ChunkManager();

  // Main interface methods
  void evictUnusedChunks(int keep_count = -1);

  // Mark chunks as used (update metadata)
  void markChunkUsed(const ChunkCoord& coord);

  std::vector<ChunkCoord> findChunksToEvict(int count);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> groupPointsByChunk(
      const torch::Tensor& positions);

  // Add points to appropriate chunks (simplified interface)
  void addPointsToChunks(
      const torch::Tensor& points,
      const torch::Tensor& colors,
      std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes,
      float cameras_extent);

  // Moved filterPointsByDepth from GaussianMapper to ChunkManager
  std::tuple<torch::Tensor, torch::Tensor> filterPointsByDepth(
      const torch::Tensor& points,
      const torch::Tensor& colors,
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
          keyframes);

  // Get chunk at specific coordinate
  std::shared_ptr<Chunk> getChunkAt(const ChunkCoord& coord);

  // Check if chunk exists on disk
  bool chunkExistsOnDisk(const ChunkCoord& coord);

  // Get chunk coordinate from 3D position
  ChunkCoord getChunkCoord(const Eigen::Vector3f& position);

  std::vector<std::shared_ptr<Chunk>> getVisibleChunks(
      std::shared_ptr<GaussianKeyframe> keyframe,
      bool use_cache = true);

  Eigen::Matrix4f createProjectionMatrix(
      std::shared_ptr<GaussianKeyframe> keyframe);

  std::pair<std::vector<std::shared_ptr<Chunk>>, std::vector<ChunkCoord>>
  findVisibleChunks(const ChunkCoord& camera_chunk,
                    int search_radius,
                    const Eigen::Vector3f& camera_position,
                    float zfar,
                    const Eigen::Matrix4f& vp_matrix);

  void manageMemoryForNewChunks(size_t chunks_to_load_count);

  void loadVisibleChunks(const std::vector<ChunkCoord>& chunks_to_load,
                         std::vector<std::shared_ptr<Chunk>>& visible_chunks);

  // std::vector<std::shared_ptr<Chunk>> getChunksInFrustumWithMargin(
  //     std::shared_ptr<GaussianKeyframe> keyframe,
  //     float margin_factor = 1.2);

  // Check if a chunk is inside or intersects with a view frustum
  AABB getChunkAABB(const ChunkCoord& coord);

  // Load a chunk
  bool loadChunk(const ChunkCoord& coord);

  // Save a chunk
  bool saveChunk(const ChunkCoord& coord);

  // Shutdown the manager (stops background threads)
  void shutdown();

  std::unordered_map<ChunkCoord, std::shared_ptr<Chunk>, ChunkCoordHash>
  getActiveChunks() const {
    return active_chunks_;
  }

  bool cullSparseChunks(int min_points_threshold);

  // Cull gaussians that are outside of chunk borders
  void cullGaussiansOutsideChunkBorders();

  void updateChunkExistenceCache(const std::vector<ChunkCoord>& coords,
                                 bool exists);

  std::vector<ChunkCoord> getExistingChunkCoords();

  void transferGaussiansAcrossChunks(float spatial_lr_scale);

  int getChunkLocalIteration(const ChunkCoord& coord) {
    auto chunk = getChunkAt(coord);
    if (chunk && chunk->getGaussians()) {
      return chunk->getGaussians()->getLocalIteration();
    }
    return 0;
  }

  // Stats for debugging/monitoring
  struct Stats {
    int active_chunks;
    int disk_loads;
    int disk_saves;
    int cache_hits;
    int prefetched;
  };

  Stats getStats() const;

  // Access to model parameters
  const GaussianModelParams& getModelParams() const { return model_params_; }
  const GaussianOptimizationParams& getOptParams() const { return opt_params_; }

  // Update current iteration
  void setCurrentIteration(int iteration) { current_iteration_ = iteration; }
  int getCurrentIteration() const { return current_iteration_; }

 private:
  // Store model parameters directly
  GaussianModelParams model_params_;
  GaussianOptimizationParams opt_params_;
  int current_iteration_ = 0;

  // Core data
  std::unordered_map<ChunkCoord, std::shared_ptr<Chunk>, ChunkCoordHash>
      active_chunks_;
  std::unordered_map<ChunkCoord, ChunkMetadata, ChunkCoordHash> chunk_metadata_;

  // Cache of chunk existence to avoid repeated disk checks
  std::unordered_map<ChunkCoord, bool, ChunkCoordHash> chunk_exists_cache_;

  // I/O thread and synchronization
  std::atomic<bool> should_terminate_;

  // Settings
  std::filesystem::path chunk_save_dir_;
  float chunk_size_;
  float overlap_margin_;
  int max_chunks_in_memory_;
  std::chrono::seconds min_retention_time_{
      5};  // Minimum time to keep a chunk after loading

  // Statistics
  mutable std::mutex stats_mutex_;
  Stats stats_{0, 0, 0, 0, 0};

  // Private helper methods
  std::filesystem::path getChunkFilename(const ChunkCoord& coord);

  Eigen::Vector3f getChunkCenter(const ChunkCoord& coord);

  // Update statistics
  void incrementStat(int& stat);
  void decrementStat(int& stat);

  // Cache for keyframe visibility results
  struct VisibilityCacheEntry {
    Sophus::SE3d pose;  // Keyframe pose when visibility was calculated
    std::vector<ChunkCoord> visible_chunks;  // Visible chunk coordinates
    std::chrono::steady_clock::time_point
        timestamp;  // When this cache entry was created/updated
  };

  // Cache mapping keyframe ID to visibility information
  std::unordered_map<size_t, VisibilityCacheEntry> visibility_cache_;
  std::mutex cache_mutex_;  // Protect the cache during concurrent access

  // Cache expiration time (in seconds)
  const std::chrono::seconds cache_expiry_time_{
      10};  // Can be adjusted based on your needs

  // Maximum number of entries in the cache
  const size_t max_cache_entries_{
      100};  // Adjust based on expected number of keyframes

  // Helper to compare poses for cache validity
  bool pose_nearly_equal(const Sophus::SE3d& a,
                         const Sophus::SE3d& b,
                         double tol = 1e-6) {
    return (a.translation() - b.translation()).norm() < tol &&
           a.unit_quaternion().angularDistance(b.unit_quaternion()) < tol;
  }

 public:
  void clearVisibilityCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    visibility_cache_.clear();
  }

  bool deleteChunk(const ChunkCoord& coord);

  void shutdownWithoutSaving() {
    // Signal thread to terminate
    should_terminate_ = true;

    // Just clear memory
    active_chunks_.clear();

    // Clear CUDA cache
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
};