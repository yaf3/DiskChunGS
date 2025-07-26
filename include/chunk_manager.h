#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chunk_types.h"
#include "cuda.h"
#include "frustum_culler.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"

// Pure Eigen implementation without explicit SIMD (relies on Eigen's
// optimizations)
bool test_AABB_against_frustum_eigen(const Eigen::Matrix4f& MVP,
                                     const AABB& aabb);

// Define chunk state enum for tracking lifecycle
enum class ChunkState {
  INACTIVE,    // Not in memory
  LOADING,     // Being loaded from disk
  ACTIVE,      // In memory and usable
  OPTIMIZING,  // In memory and currently being optimized
  SAVING,      // Being saved to disk
  DELETING,    // Being deleted
  UNKNOWN      // State not known (e.g., metadata missing)
};

// Definition for chunk operations
struct ChunkOperation {
  ChunkCoord coord;
  enum Type { LOAD, SAVE, DELETE } type;
  std::promise<bool> completion_promise;
  int priority;
  std::chrono::steady_clock::time_point timestamp;
  bool load_for_optimization;
};

// Comparator for priority queue
struct ChunkOperationComparator {
  bool operator()(const std::shared_ptr<ChunkOperation>& a,
                  const std::shared_ptr<ChunkOperation>& b) {
    // Higher priority first, then older operations first
    if (a->priority != b->priority) return a->priority < b->priority;
    return a->timestamp > b->timestamp;
  }
};

// Enhanced metadata for chunks
struct ChunkMetadata {
  std::atomic<ChunkState> state{ChunkState::INACTIVE};
  std::shared_ptr<Chunk> chunk;
  std::chrono::steady_clock::time_point load_time;
  std::chrono::steady_clock::time_point last_used;
  int usage_count = 0;
  std::mutex operation_mutex;  // Fine-grained lock for this chunk
  std::condition_variable operation_cv;
};

class ChunkManager {
 public:
  ChunkManager(const GaussianModelParams& model_params,
               const GaussianOptimizationParams& opt_params,
               std::filesystem::path chunk_save_dir,
               float chunk_size = 50.0f,
               float cameras_extent = 1.0f,
               int max_chunks = 50,
               int num_io_threads = 4,
               size_t max_vram_budget_mb = 8192);

  ~ChunkManager();

  // New async methods
  std::future<bool> loadChunkAsync(const ChunkCoord& coord,
                                   int priority = 0,
                                   bool load_for_optimization = true,
                                   bool skip_busy_chunks = true);
  std::future<bool> saveChunkAsync(const ChunkCoord& coord, int priority = 0);
  std::future<bool> deleteChunkAsync(const ChunkCoord& coord, int priority = 0);

  // Synchronous wrappers
  bool loadChunkSync(const ChunkCoord& coord,
                     bool load_for_optimization = true,
                     bool skip_busy_chunks = true);
  bool saveChunkSync(const ChunkCoord& coord);
  bool deleteChunkSync(const ChunkCoord& coord);

  void releaseChunksFromOptimization(
      const std::vector<ChunkCoord>& chunk_coords);
  void releaseChunksFromOptimization(
      const std::vector<std::shared_ptr<Chunk>>& chunks);
  void releaseAllChunksFromOptimization();

  void initializeMetaData(ChunkCoord& coord);

 private:
  std::atomic<bool> is_shutting_down_{false};

  // Thread pool and task queue
  std::vector<std::thread> io_threads_;
  std::atomic<bool> shutdown_threads_{false};

  // Priority-based task queue
  std::priority_queue<std::shared_ptr<ChunkOperation>,
                      std::vector<std::shared_ptr<ChunkOperation>>,
                      ChunkOperationComparator>
      operation_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  // Enhanced tracking with thread-safety
  std::unordered_map<ChunkCoord, ChunkMetadata, ChunkCoordHash> chunk_metadata_;
  std::mutex chunk_metadata_mutex_;

  // Thread pool methods
  void initializeThreadPool(int num_threads);
  void shutdownThreadPool();
  void ioThreadFunction();

  // Operation methods
  void enqueueOperation(std::shared_ptr<ChunkOperation> operation);
  void processOperation(std::shared_ptr<ChunkOperation> operation);
  bool processLoadOperation(const ChunkCoord& coord, bool load_for_opt);
  bool processSaveOperation(const ChunkCoord& coord);
  bool processDeleteOperation(const ChunkCoord& coord);

  // State management helpers
  bool transitionChunkState(const ChunkCoord& coord,
                            ChunkState expected,
                            ChunkState new_state);
  bool waitForChunkState(const ChunkCoord& coord,
                         ChunkState target_state,
                         std::chrono::milliseconds timeout);
  std::future<bool> createWaitFuture(const ChunkCoord& coord,
                                     ChunkState target_state);

 public:
  ChunkState getChunkState(const ChunkCoord& coord);
  // Main interface methods
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> groupPointsByChunk(
      const torch::Tensor& positions);

  // Add points to appropriate chunks (simplified interface)
  void addPointsToChunks(const torch::Tensor& points,
                         const torch::Tensor& colors,
                         const torch::Tensor& scales,
                         const torch::Tensor& opacities);

  // Get chunk at specific coordinate
  std::shared_ptr<Chunk> getChunkAt(const ChunkCoord& coord);

  // Check if chunk exists on disk
  bool chunkExists(const ChunkCoord& coord);

  // Get chunk coordinate from 3D position
  ChunkCoord getChunkCoord(const Eigen::Vector3f& position);

  Eigen::Matrix4f createProjectionMatrix(
      std::shared_ptr<GaussianKeyframe> keyframe);

  std::vector<ChunkCoord> frustumCullChunks(
      std::shared_ptr<GaussianKeyframe> keyframe,
      bool use_cache = true);

  // Check if a chunk is inside or intersects with a view frustum
  AABB getChunkAABB(const ChunkCoord& coord);

  std::vector<std::shared_ptr<Chunk>> loadVisibleChunks(
      std::shared_ptr<GaussianKeyframe> keyframe,
      bool use_cache = true);
  void preloadVisibleChunks(std::shared_ptr<GaussianKeyframe> keyframe,
                            bool use_cache = true);

  bool cullSparseChunks(int min_points_threshold, int min_chunk_iterations);

  std::vector<ChunkCoord> getExistingChunkCoords();

  void transferGaussiansAcrossChunks();
  void processBatchWithStates(
      const std::vector<std::tuple<ChunkCoord, GaussianTransferData>>& batch);
  GaussianTransferData combineTransferData(
      const std::vector<GaussianTransferData>& transfers);
  void applyTransferToDestinationWithStates(
      const ChunkCoord& dest_coord,
      const GaussianTransferData& transfer_data);

  int getChunkLocalIteration(const ChunkCoord& coord) {
    auto chunk = getChunkAt(coord);
    if (chunk && chunk->getGaussians()) {
      return chunk->getGaussians()->getLocalIteration();
    }
    return 0;
  }

  void updateVramEstimateFromCurrentState();
  bool saveAllChunksSync();

  // ChunkStats for debugging/monitoring
  struct ChunkStats {
    int active_chunks;
    int disk_loads;
    int disk_saves;
    int cache_hits;
    int prefetched;
    int existing_chunks;
  };

  ChunkStats getStats() const;

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

  // I/O thread and synchronization
  std::atomic<bool> should_terminate_;

  // Settings
  std::filesystem::path chunk_save_dir_;
  float chunk_size_;
  float cameras_extent_;
  int max_chunks_in_memory_;
  std::chrono::milliseconds min_retention_time_{
      0};  // Minimum time to keep a chunk after loading

  // VRAM-based chunk management
  size_t max_vram_budget_mb_;
  float estimated_chunk_vram_mb_;
  const float vram_estimate_alpha_ = 0.1f;  // Exponential moving average factor
  const int min_chunks_limit_ = 10;         // Minimum chunks to always allow
  const int max_chunks_limit_ = 200;        // Maximum chunks to ever allow
  std::chrono::steady_clock::time_point last_vram_check_ =
      std::chrono::steady_clock::now();
  size_t last_measured_vram_ = 0;
  int last_chunk_count_ = 0;
  static constexpr auto vram_check_interval_ = std::chrono::seconds(5);
  std::chrono::steady_clock::time_point last_eviction_time_;
  static constexpr auto MIN_EVICTION_INTERVAL = std::chrono::seconds(5);

  // VRAM thresholds (as percentages of budget)
  static constexpr float VRAM_NORMAL_THRESHOLD = 0.70f;  // Below this: all good
  static constexpr float VRAM_EVICTION_THRESHOLD =
      0.80f;  // Above this: start evicting
  static constexpr float VRAM_AGGRESSIVE_THRESHOLD =
      0.90f;  // Above this: aggressive eviction
  static constexpr float VRAM_EMERGENCY_THRESHOLD =
      0.95f;  // Above this: block new loads

  // Eviction targets (what we aim for after eviction)
  static constexpr float VRAM_TARGET_AFTER_EVICTION = 0.65f;
  static constexpr float VRAM_TARGET_AFTER_AGGRESSIVE = 0.60f;

  // Statistics
  mutable std::mutex stats_mutex_;
  ChunkStats stats_{0, 0, 0, 0, 0, 0};

  // Private helper methods
  std::filesystem::path getChunkFilename(const ChunkCoord& coord);

  Eigen::Vector3f getChunkCenter(const ChunkCoord& coord);

  // Update statistics
  void incrementStat(int& stat);
  void decrementStat(int& stat);

  // Memory monitoring
  size_t getGPUMemoryUsage() const;
  void logMemoryUsage(const std::string& operation) const;

  // Cache for keyframe visibility results
  struct VisibilityCacheEntry {
    Sophus::SE3d pose;  // Keyframe pose when visibility was calculated
    std::vector<ChunkCoord> visible_chunks;  // Visible chunk coordinates
    std::chrono::steady_clock::time_point
        timestamp;  // When this cache entry was created/updated
  };

  // Cache mapping keyframe ID to visibility information
  std::unordered_map<size_t, VisibilityCacheEntry> visibility_cache_;
  std::mutex
      visibility_cache_mutex_;  // Protect the cache during concurrent access

  // Cache expiration time (in seconds)
  const std::chrono::seconds cache_expiry_time_{
      10};  // Can be adjusted based on your needs

  // Maximum number of entries in the cache
  const size_t max_cache_entries_{
      100};  // Adjust based on expected number of keyframes

  // Helper to compare poses for cache validity
  bool pose_nearly_equal(const Sophus::SE3d& a, const Sophus::SE3d& b) {
    // Translation tolerance: small fraction of chunk size
    const double translation_tol = chunk_size_ * 0.05;  // 5% of chunk size

    // Rotation tolerance: a few degrees
    const double rotation_tol = 0.05;  // ~3 degrees in radians

    return (a.translation() - b.translation()).norm() < translation_tol &&
           a.unit_quaternion().angularDistance(b.unit_quaternion()) <
               rotation_tol;
  }

 public:
  void clearVisibilityCache() {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
    visibility_cache_.clear();
  }

  bool waitForVramAvailable(
      float max_usage_ratio = 0.80f,
      std::chrono::milliseconds timeout = std::chrono::seconds(10),
      const std::string& operation_name = "operation");

 private:
  std::thread lru_eviction_thread_;
  std::atomic<bool> stop_lru_thread_{false};
  std::mutex lru_mutex_;
  std::condition_variable lru_cv_;
  std::chrono::milliseconds lru_check_interval_{200};

  void lruEvictionThreadFunction();
  void updateLastUsedTime(const ChunkCoord& coord);
  void updateLastUsedTimeForChunks(const std::vector<ChunkCoord>& coords);

  bool shouldEvictChunks();
  bool shouldBlockNewLoads();
  int calculateEvictionCount();
  float getCurrentVramUsageRatio() const;
  std::string getVramStatus();

 public:
  void triggerLruCheck();
  int getOperationQueueSize() const {
    // std::lock_guard<std::mutex> lock(queue_mutex_);
    return operation_queue_.size();
  }
  void testMemoryUsagePattern();
};

// RAII guard for automatic chunk optimization release
// Ensures chunks are released from optimization state on any function exit
class ChunkOptimizationGuard {
 private:
  ChunkManager* chunk_manager_;
  std::vector<std::shared_ptr<Chunk>> chunks_;
  static std::atomic<int> guard_counter_;
  int guard_id_;  // Declare this AFTER the static counter

 public:
  // Constructor taking const reference to chunks
  ChunkOptimizationGuard(ChunkManager* manager,
                         const std::vector<std::shared_ptr<Chunk>>& chunks)
      : chunk_manager_(manager), chunks_(chunks), guard_id_(++guard_counter_) {
    // std::cout << "[Guard " << guard_id_ << "] CREATED for chunks: ";
    // for (const auto& chunk : chunks_) {
    //   if (chunk) {
    //     std::cout << "(" << chunk->getCoord().x << "," << chunk->getCoord().y
    //               << "," << chunk->getCoord().z << ") ";
    //   }
    // }
    // std::cout << std::endl;

    if (!manager) return;
    for (const auto& chunk : chunks) {
      if (chunk &&
          manager->getChunkState(chunk->getCoord()) != ChunkState::OPTIMIZING) {
        std::cout << "[Guard " << guard_id_ << "] ERROR: Chunk ("
                  << chunk->getCoord().x << "," << chunk->getCoord().y << ","
                  << chunk->getCoord().z << ") is in state "
                  << static_cast<int>(manager->getChunkState(chunk->getCoord()))
                  << " instead of OPTIMIZING" << std::endl;
        throw std::runtime_error("Chunk is not in OPTIMIZING state");
      }
    }
  }

  // Constructor taking chunks by move
  ChunkOptimizationGuard(ChunkManager* manager,
                         std::vector<std::shared_ptr<Chunk>>&& chunks)
      : chunk_manager_(manager),
        chunks_(std::move(chunks)),
        guard_id_(++guard_counter_) {
    // std::cout << "[Guard " << guard_id_ << "] CREATED for chunks: ";
    // for (const auto& chunk : chunks_) {
    //   if (chunk) {
    //     std::cout << "(" << chunk->getCoord().x << "," << chunk->getCoord().y
    //               << "," << chunk->getCoord().z << ") ";
    //   }
    // }
    // std::cout << std::endl;

    if (!manager) return;
    for (const auto& chunk : chunks_) {  // Use chunks_ here since we moved
      if (chunk &&
          manager->getChunkState(chunk->getCoord()) != ChunkState::OPTIMIZING) {
        std::cout << "[Guard " << guard_id_ << "] ERROR: Chunk ("
                  << chunk->getCoord().x << "," << chunk->getCoord().y << ","
                  << chunk->getCoord().z << ") is in state "
                  << static_cast<int>(manager->getChunkState(chunk->getCoord()))
                  << " instead of OPTIMIZING" << std::endl;
        throw std::runtime_error("Chunk is not in OPTIMIZING state");
      }
    }
  }

  // Constructor taking non-const reference (for compatibility)
  ChunkOptimizationGuard(ChunkManager* manager,
                         std::vector<std::shared_ptr<Chunk>>& chunks)
      : chunk_manager_(manager), chunks_(chunks), guard_id_(++guard_counter_) {
    // std::cout << "[Guard " << guard_id_ << "] CREATED for chunks: ";
    // for (const auto& chunk : chunks_) {
    //   if (chunk) {
    //     std::cout << "(" << chunk->getCoord().x << "," << chunk->getCoord().y
    //               << "," << chunk->getCoord().z << ") ";
    //   }
    // }
    // std::cout << std::endl;

    if (!manager) return;
    for (const auto& chunk : chunks) {
      if (chunk &&
          manager->getChunkState(chunk->getCoord()) != ChunkState::OPTIMIZING) {
        std::cout << "[Guard " << guard_id_ << "] ERROR: Chunk ("
                  << chunk->getCoord().x << "," << chunk->getCoord().y << ","
                  << chunk->getCoord().z << ") is in state "
                  << static_cast<int>(manager->getChunkState(chunk->getCoord()))
                  << " instead of OPTIMIZING" << std::endl;
        throw std::runtime_error("Chunk is not in OPTIMIZING state");
      }
    }
  }

  // Non-copyable, movable
  ChunkOptimizationGuard(const ChunkOptimizationGuard&) = delete;
  ChunkOptimizationGuard& operator=(const ChunkOptimizationGuard&) = delete;

  ChunkOptimizationGuard(ChunkOptimizationGuard&& other) noexcept
      : chunk_manager_(other.chunk_manager_),
        chunks_(std::move(other.chunks_)),
        guard_id_(other.guard_id_) {  // <-- COPY the guard_id from other

    std::cout << "[Guard " << guard_id_ << "] MOVED from Guard "
              << other.guard_id_ << std::endl;
    other.chunk_manager_ = nullptr;
    other.guard_id_ = -1;  // Mark the moved-from object
  }

  ChunkOptimizationGuard& operator=(ChunkOptimizationGuard&& other) noexcept {
    if (this != &other) {
      release();  // Release current chunks if any
      chunk_manager_ = other.chunk_manager_;
      chunks_ = std::move(other.chunks_);
      guard_id_ = other.guard_id_;  // <-- COPY the guard_id

      std::cout << "[Guard " << guard_id_ << "] MOVE-ASSIGNED from Guard "
                << other.guard_id_ << std::endl;
      other.chunk_manager_ = nullptr;
      other.guard_id_ = -1;  // Mark the moved-from object
    }
    return *this;
  }

  ~ChunkOptimizationGuard() {
    // std::cout << "[Guard " << guard_id_ << "] DESTROYING for chunks: ";
    // for (const auto& chunk : chunks_) {
    //   if (chunk) {
    //     std::cout << "(" << chunk->getCoord().x << "," << chunk->getCoord().y
    //               << "," << chunk->getCoord().z << ") ";
    //   }
    // }
    // std::cout << std::endl;
    release();
  }

  void release() {
    if (chunk_manager_ && !chunks_.empty()) {
      // std::cout << "[Guard " << guard_id_ << "] RELEASING " << chunks_.size()
      //           << " chunks from optimization" << std::endl;
      chunk_manager_->releaseChunksFromOptimization(chunks_);
      chunks_.clear();
    }
  }

  // Allow manual early release
  void releaseEarly() {
    release();
    chunk_manager_ = nullptr;
  }
};