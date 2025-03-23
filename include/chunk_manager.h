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
  bool dirty;    // Has been modified since last save
  bool loading;  // Currently being loaded
  bool saving;   // Currently being saved

  ChunkMetadata()
      : last_used(std::chrono::steady_clock::now()),
        load_time(std::chrono::steady_clock::now()),
        usage_count(0),
        dirty(false),
        loading(false),
        saving(false) {}
};

// Chunk I/O operation
enum class ChunkOperation { LOAD, SAVE, DELETE, NONE };

// Chunk I/O request
struct ChunkIORequest {
  ChunkCoord coord;
  ChunkOperation operation;
  int priority;  // Higher number means higher priority

  ChunkIORequest(const ChunkCoord& c, ChunkOperation op, int p = 0)
      : coord(c), operation(op), priority(p) {}

  // Compare for priority queue (higher priority comes first)
  bool operator<(const ChunkIORequest& other) const {
    return priority < other.priority;
  }
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

  // Schedule chunk save
  void scheduleChunkSave(const ChunkCoord& coord, int priority = 0);
  void scheduleChunkLoad(const ChunkCoord& coord, int priority = 0);

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
  std::shared_ptr<Chunk> getChunkAtNoLock(const ChunkCoord& coord);
  std::shared_ptr<Chunk> getChunkAt(const ChunkCoord& coord);

  // Check if chunk exists on disk
  bool chunkExistsOnDisk(const ChunkCoord& coord);

  // Get chunk coordinate from 3D position
  ChunkCoord getChunkCoord(const Eigen::Vector3f& position);

  std::vector<std::shared_ptr<Chunk>> getVisibleChunks(
      std::shared_ptr<GaussianKeyframe> keyframe);

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

  // Find chunks to evict based on LRU policy
  std::vector<ChunkCoord> findChunksToEvict(int count);

  // Shutdown the manager (stops background threads)
  void shutdown();

  std::unordered_map<ChunkCoord, std::shared_ptr<Chunk>, ChunkCoordHash>
  getActiveChunks() const {
    return active_chunks_;
  }

  bool cullSparseChunks(int min_points_threshold);

  // Cull gaussians that are outside of chunk borders
  void cullGaussiansOutsideChunkBorders();

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
  // No-lock versions of methods that are called within locked sections
  void markChunkUsedNoLock(const ChunkCoord& coord);
  void scheduleChunkSaveNoLock(const ChunkCoord& coord, int priority = 0);
  void scheduleChunkLoadNoLock(const ChunkCoord& coord, int priority = 0);
  bool chunkExistsOnDiskNoLock(const ChunkCoord& coord);
  bool loadChunkNoLock(const ChunkCoord& coord);
  bool saveChunkNoLock(const ChunkCoord& coord);
  std::vector<ChunkCoord> findChunksToEvictNoLock(int count);

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
  std::thread io_thread_;
  std::priority_queue<ChunkIORequest> io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;
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

  // Background I/O thread function
  void ioThreadFunc();

  // Private helper methods
  std::filesystem::path getChunkFilename(const ChunkCoord& coord);

  Eigen::Vector3f getChunkCenter(const ChunkCoord& coord);

  // Update statistics
  void incrementStat(int& stat);
  void decrementStat(int& stat);
};