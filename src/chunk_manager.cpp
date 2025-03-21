#include "include/chunk_manager.h"

#include <torch/cuda.h>

#include <algorithm>
#include <iostream>

#include "include/profiling.h"

// Non-SIMD implementation of AABB frustum culling using Eigen
bool test_AABB_against_frustum(const Eigen::Matrix4f& MVP, const AABB& aabb) {
  // Define the 8 corners of the AABB
  Eigen::Vector4f corners[8];
  corners[0] =
      Eigen::Vector4f(aabb.min.x(), aabb.min.y(), aabb.min.z(), 1.0f);  // x y z
  corners[1] =
      Eigen::Vector4f(aabb.max.x(), aabb.min.y(), aabb.min.z(), 1.0f);  // X y z
  corners[2] =
      Eigen::Vector4f(aabb.min.x(), aabb.max.y(), aabb.min.z(), 1.0f);  // x Y z
  corners[3] =
      Eigen::Vector4f(aabb.max.x(), aabb.max.y(), aabb.min.z(), 1.0f);  // X Y z
  corners[4] =
      Eigen::Vector4f(aabb.min.x(), aabb.min.y(), aabb.max.z(), 1.0f);  // x y Z
  corners[5] =
      Eigen::Vector4f(aabb.max.x(), aabb.min.y(), aabb.max.z(), 1.0f);  // X y Z
  corners[6] =
      Eigen::Vector4f(aabb.min.x(), aabb.max.y(), aabb.max.z(), 1.0f);  // x Y Z
  corners[7] =
      Eigen::Vector4f(aabb.max.x(), aabb.max.y(), aabb.max.z(), 1.0f);  // X Y Z

  bool inside = false;

  for (int i = 0; i < 8; ++i) {
    // Transform vertex to clip space
    Eigen::Vector4f transformed = MVP * corners[i];

    // Check vertex against clip space bounds
    inside =
        inside || (within(-transformed.w(), transformed.x(), transformed.w()) &&
                   within(-transformed.w(), transformed.y(), transformed.w()) &&
                   within(0.0f, transformed.z(), transformed.w()));
  }

  return inside;
}

// SIMD-optimized implementation of AABB frustum culling
// bool test_AABB_against_frustum_256(const Eigen::Matrix4f& transform,
//                                    const AABB& aabb) {
//   // Prepare AABB corners for SIMD processing
//   Eigen::Vector4f min(aabb.min.x(), aabb.min.y(), aabb.min.z(), 1.0f);
//   Eigen::Vector4f max(aabb.max.x(), aabb.max.y(), aabb.max.z(), 1.0f);

//   // Load AABB min and max into SIMD registers
//   // Note: We're assuming Eigen uses column-major order by default
//   const __m128 aabb_min = _mm_load_ps(min.data());
//   const __m128 aabb_max = _mm_load_ps(max.data());

//   // Shuffle components to prepare for corner calculations
//   __m128 x_minmax =
//       _mm_shuffle_ps(aabb_min, aabb_max, _MM_SHUFFLE(0, 0, 0, 0));  // x x X
//       X
//   x_minmax = _mm_permute_ps(x_minmax, _MM_SHUFFLE(2, 0, 2, 0));     // x X x
//   X const __m128 y_minmax =
//       _mm_shuffle_ps(aabb_min, aabb_max, _MM_SHUFFLE(1, 1, 1, 1));  // y y Y
//       Y
//   const __m128 z_min = SPLAT(aabb_min, 2);                          // z z z
//   z const __m128 z_max = SPLAT(aabb_max, 2);                          // Z Z
//   Z Z

//   // Combine into 256-bit registers for 8 corners
//   const __m256 x = _mm256_set_m128(x_minmax, x_minmax);
//   const __m256 y = _mm256_set_m128(y_minmax, y_minmax);
//   const __m256 z = _mm256_set_m128(z_min, z_max);

//   // Storage for transformed corner components
//   __m256 corner_comps[4];

//   // Transform all 8 corners at once using SIMD
//   for (int i = 0; i < 4; ++i) {
//     // Load matrix row from Eigen matrix
//     __m256 res = _mm256_broadcast_ss(&transform(i, 3));  // w component
//     res = _mm256_add_ps(
//         res, _mm256_mul_ps(_mm256_broadcast_ss(&transform(i, 0)), x));
//     res = _mm256_add_ps(
//         res, _mm256_mul_ps(_mm256_broadcast_ss(&transform(i, 1)), y));
//     res = _mm256_add_ps(
//         res, _mm256_mul_ps(_mm256_broadcast_ss(&transform(i, 2)), z));
//     corner_comps[i] = res;
//   }

//   // Prepare for clip space tests
//   const __m256 neg_ws = _mm256_sub_ps(_mm256_setzero_ps(), corner_comps[3]);

//   // Test whether -w < x < w
//   __m256 inside = _mm256_and_ps(
//       _mm256_cmp_ps(neg_ws, corner_comps[0], _CMP_LE_OQ),
//       _mm256_cmp_ps(corner_comps[0], corner_comps[3], _CMP_LE_OQ));
//   // inside && -w < y < w
//   inside = _mm256_and_ps(
//       inside, _mm256_and_ps(
//                   _mm256_cmp_ps(neg_ws, corner_comps[1], _CMP_LE_OQ),
//                   _mm256_cmp_ps(corner_comps[1], corner_comps[3],
//                   _CMP_LE_OQ)));
//   // inside && 0 < z < w
//   inside = _mm256_and_ps(
//       inside,
//       _mm256_and_ps(
//           _mm256_cmp_ps(_mm256_setzero_ps(), corner_comps[2], _CMP_LE_OQ),
//           _mm256_cmp_ps(corner_comps[2], corner_comps[3], _CMP_LE_OQ)));

//   // Reduce our 8 different in/out lanes to a single boolean
//   __m128 reduction = _mm_or_ps(_mm256_extractf128_ps(inside, 0),
//                                _mm256_extractf128_ps(inside, 1));
//   reduction =
//       _mm_or_ps(reduction, _mm_permute_ps(reduction, _MM_SHUFFLE(2, 3, 0,
//       1)));
//   reduction =
//       _mm_or_ps(reduction, _mm_permute_ps(reduction, _MM_SHUFFLE(1, 0, 3,
//       2)));

//   // Store our reduction
//   u32 res = 0u;
//   _mm_store_ss(reinterpret_cast<float*>(&res), reduction);
//   return res != 0;
// }

// Pure Eigen implementation without explicit SIMD (relies on Eigen's
// optimizations)
bool test_AABB_against_frustum_eigen(const Eigen::Matrix4f& MVP,
                                     const AABB& aabb) {
  // Define the 8 corners of the AABB
  std::array<Eigen::Vector3f, 8> corners;
  corners[0] = aabb.min;
  corners[1] = Eigen::Vector3f(aabb.max.x(), aabb.min.y(), aabb.min.z());
  corners[2] = Eigen::Vector3f(aabb.min.x(), aabb.max.y(), aabb.min.z());
  corners[3] = Eigen::Vector3f(aabb.max.x(), aabb.max.y(), aabb.min.z());
  corners[4] = Eigen::Vector3f(aabb.min.x(), aabb.min.y(), aabb.max.z());
  corners[5] = Eigen::Vector3f(aabb.max.x(), aabb.min.y(), aabb.max.z());
  corners[6] = Eigen::Vector3f(aabb.min.x(), aabb.max.y(), aabb.max.z());
  corners[7] = aabb.max;

  for (const auto& corner : corners) {
    // Transform to clip space
    Eigen::Vector4f clipSpace =
        MVP * Eigen::Vector4f(corner.x(), corner.y(), corner.z(), 1.0f);

    // Check if this corner is inside the view frustum
    if (clipSpace.x() >= -clipSpace.w() && clipSpace.x() <= clipSpace.w() &&
        clipSpace.y() >= -clipSpace.w() && clipSpace.y() <= clipSpace.w() &&
        clipSpace.z() >= 0.0f && clipSpace.z() <= clipSpace.w()) {
      return true;
    }
  }

  return false;
}

// Main culling function using Eigen types
void cull_AABBs_against_frustum(const Cam& camera,
                                const std::vector<Eigen::Matrix4f>& transforms,
                                const std::vector<AABB>& aabb_list,
                                std::vector<u32>& out_visible_list,
                                bool use_simd) {
  // Compute view-projection matrix
  Eigen::Matrix4f VP = camera.projection * camera.view;

  // Reserve space for visible objects
  out_visible_list.reserve(aabb_list.size());
  out_visible_list.clear();

  for (size_t i = 0; i < aabb_list.size(); i++) {
    // Compute model-view-projection matrix
    Eigen::Matrix4f MVP = VP * transforms[i];

    // Test using appropriate method
    bool visible;
    if (use_simd) {
      // visible = test_AABB_against_frustum_256(MVP, aabb_list[i]);
    } else {
      visible = test_AABB_against_frustum_eigen(MVP, aabb_list[i]);
    }

    if (visible) {
      out_visible_list.push_back(static_cast<u32>(i));
    }
  }
}

// Constructor
ChunkManager::ChunkManager(const GaussianModelParams& model_params,
                           const GaussianOptimizationParams& opt_params,
                           std::filesystem::path chunk_save_dir,
                           float chunk_size,
                           float overlap_margin,
                           int max_chunks)
    : model_params_(model_params),
      opt_params_(opt_params),
      chunk_save_dir_(chunk_save_dir),
      chunk_size_(chunk_size),
      overlap_margin_(overlap_margin),
      max_chunks_in_memory_(max_chunks),
      should_terminate_(false) {
  std::cout << "Creating ChunkManager" << std::endl;
  // Create save directory if it doesn't exist
  if (!chunk_save_dir_.empty() && !std::filesystem::exists(chunk_save_dir_)) {
    std::filesystem::create_directories(chunk_save_dir_);
  }

  // Start I/O thread
  std::cout << "Creating I/O Thread" << std::endl;
  io_thread_ = std::thread(&ChunkManager::ioThreadFunc, this);
}

// Destructor
ChunkManager::~ChunkManager() { shutdown(); }

std::tuple<torch::Tensor, torch::Tensor> ChunkManager::filterPointsByDepth(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  std::cout << "Filtering points, starting with " << points.size(0)
            << std::endl;
  const int num_points = points.size(0);
  auto device = points.device();
  auto options = torch::TensorOptions().device(device).dtype(points.dtype());

  // Initialize validity mask for all points (start with all false)
  torch::Tensor valid_mask = torch::zeros(
      {num_points}, torch::TensorOptions().device(device).dtype(torch::kBool));

  // Process each keyframe
  for (const auto& [kfid, keyframe] : keyframes) {
    if (!keyframe->set_pose_) continue;

    // Get the rotation and translation from Sophus SE3
    Eigen::Matrix3d R = keyframe->Tcw_.rotationMatrix();
    Eigen::Vector3d t = keyframe->Tcw_.translation();

    // Convert to tensors and ensure same dtype as points
    torch::Tensor R_tensor =
        torch::from_blob(const_cast<double*>(R.data()), {3, 3},
                         torch::TensorOptions().dtype(torch::kDouble))
            .to(device)
            .to(points.dtype());

    torch::Tensor t_tensor =
        torch::from_blob(const_cast<double*>(t.data()), {3},
                         torch::TensorOptions().dtype(torch::kDouble))
            .to(device)
            .to(points.dtype());

    // Transform points: R * points + t
    torch::Tensor points_cam = torch::matmul(points, R_tensor.t());
    points_cam += t_tensor.unsqueeze(0);

    // Extract depths (z-coordinates)
    torch::Tensor depths = points_cam.select(1, 2);

    // Check depth constraints
    torch::Tensor valid_in_frame =
        (depths >= keyframe->znear_) & (depths <= keyframe->zfar_);

    // Update global validity mask
    valid_mask = valid_mask | valid_in_frame;
  }

  // Use boolean indexing to filter points and colors
  torch::Tensor filtered_points = points.index({valid_mask});
  torch::Tensor filtered_colors = colors.index({valid_mask});
  std::cout << "After filter " << filtered_points.size(0) << std::endl;

  return std::make_tuple(filtered_points, filtered_colors);
}

// Private version that assumes lock is already held
void ChunkManager::markChunkUsedNoLock(const ChunkCoord& coord) {
  auto it = chunk_metadata_.find(coord);
  if (it != chunk_metadata_.end()) {
    it->second.last_used = std::chrono::steady_clock::now();
    it->second.usage_count++;
    it->second.dirty =
        true;  // Mark as dirty since it will be used for optimization
  }
}

// Public version that acquires the lock
void ChunkManager::markChunkUsed(const ChunkCoord& coord) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  markChunkUsedNoLock(coord);
}

// Private version that assumes lock is already held
void ChunkManager::scheduleChunkSaveNoLock(const ChunkCoord& coord,
                                           int priority) {
  // Only schedule if chunk exists and is dirty
  auto meta_it = chunk_metadata_.find(coord);
  if (meta_it != chunk_metadata_.end() && meta_it->second.dirty &&
      !meta_it->second.saving) {
    meta_it->second.saving = true;
    io_queue_.push(ChunkIORequest(coord, ChunkOperation::SAVE, priority));
    io_cv_.notify_one();
  }
}

// Public version that acquires the lock
void ChunkManager::scheduleChunkSave(const ChunkCoord& coord, int priority) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  scheduleChunkSaveNoLock(coord, priority);
}

// Evict least recently used chunks
void ChunkManager::evictUnusedChunks(int keep_count) {
  auto timer = ProfilingUtils::Timer("ChunkManager::evictUnusedChunks");
  // std::cout << "evictUnusedChunks called with keep_count " << keep_count
  //           << std::endl;

  if (keep_count < 0) {
    // Default: keep enough space for some new chunks
    keep_count = std::max(5, max_chunks_in_memory_ / 10);
  }

  std::vector<ChunkCoord> to_evict;
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    to_evict = findChunksToEvictNoLock(keep_count);
  }

  // Schedule saves for evicted chunks
  for (const auto& coord : to_evict) {
    scheduleChunkSave(coord, 10);  // High priority for eviction
  }
}

// Private version that assumes lock is already held
std::vector<ChunkCoord> ChunkManager::findChunksToEvictNoLock(int count) {
  // If we have space, don't evict
  if (active_chunks_.size() <= max_chunks_in_memory_ - count) {
    return {};
  }

  // How many chunks to evict
  int to_evict = active_chunks_.size() - (max_chunks_in_memory_ - count);

  // Gather candidates (not loading or saving)
  std::vector<
      std::pair<ChunkCoord, std::chrono::time_point<std::chrono::steady_clock>>>
      candidates;
  auto now = std::chrono::steady_clock::now();

  for (const auto& [coord, chunk] : active_chunks_) {
    auto meta_it = chunk_metadata_.find(coord);
    if (meta_it != chunk_metadata_.end() && !meta_it->second.loading &&
        !meta_it->second.saving) {
      // Skip recently loaded chunks
      auto time_since_load = std::chrono::duration_cast<std::chrono::seconds>(
          now - meta_it->second.load_time);

      if (time_since_load > min_retention_time_) {
        candidates.push_back({coord, meta_it->second.last_used});
      }
    }
  }

  // Sort by last used time (oldest first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Take up to to_evict chunks
  std::vector<ChunkCoord> result;
  for (size_t i = 0;
       i < std::min(static_cast<size_t>(to_evict), candidates.size()); ++i) {
    result.push_back(candidates[i].first);
  }

  return result;
}

// Public version that acquires the lock
std::vector<ChunkCoord> ChunkManager::findChunksToEvict(int count) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return findChunksToEvictNoLock(count);
}

// Background I/O thread function
void ChunkManager::ioThreadFunc() {
  while (!should_terminate_) {
    ChunkIORequest request{
        ChunkCoord{0, 0, 0},
        ChunkOperation::NONE
    };

    // Get next request
    {
      std::unique_lock<std::mutex> lock(io_mutex_);

      if (io_queue_.empty()) {
        // Wait for new requests or termination signal
        io_cv_.wait(lock,
                    [this] { return !io_queue_.empty() || should_terminate_; });

        if (should_terminate_ && io_queue_.empty()) {
          break;
        }
      }

      if (!io_queue_.empty()) {
        request = io_queue_.top();
        io_queue_.pop();
      }
    }

    // Process request
    if (request.operation == ChunkOperation::LOAD) {
      std::cout << "[IO Thread] Processing request to load chunk: "
                << request.coord.x << " " << request.coord.y << " "
                << request.coord.z << " " << std::endl;
      if (!loadChunk(request.coord, true)) {
        throw std::runtime_error("Failed to load chunk");
      }
    } else if (request.operation == ChunkOperation::SAVE) {
      std::cout << "[IO Thread] Processing request to save chunk: "
                << request.coord.x << " " << request.coord.y << " "
                << request.coord.z << " " << std::endl;
      if (!saveChunk(request.coord, true)) {
        throw std::runtime_error("Failed to save chunk");
      }
    } else if (request.operation == ChunkOperation::DELETE) {
      std::cout << "[IO Thread] Processing request to delete chunk: "
                << request.coord.x << " " << request.coord.y << " "
                << request.coord.z << " " << std::endl;

      // Delete the file if it exists
      auto chunk_filename = getChunkFilename(request.coord);
      std::lock_guard<std::mutex> lock(io_mutex_);

      if (std::filesystem::exists(chunk_filename)) {
        try {
          std::filesystem::remove(chunk_filename);
        } catch (const std::exception& e) {
          std::cerr << "Error deleting chunk file: " << e.what() << std::endl;
        }
      }

      // Update metadata
      auto meta_it = chunk_metadata_.find(request.coord);
      if (meta_it != chunk_metadata_.end()) {
        // Clear the metadata or mark as not dirty/saving
        meta_it->second.dirty = false;
        meta_it->second.saving = false;
      }

      // Update the disk cache to reflect the deletion
      chunk_exists_cache_[request.coord] = false;
    }
  }

  // Save any remaining dirty chunks on shutdown
  std::vector<ChunkCoord> dirty_chunks;

  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    for (const auto& [coord, meta] : chunk_metadata_) {
      if (meta.dirty && !meta.saving &&
          active_chunks_.find(coord) != active_chunks_.end()) {
        dirty_chunks.push_back(coord);
      }
    }
  }

  for (const auto& coord : dirty_chunks) {
    saveChunk(coord, false);
  }
}

// Get chunk filename
std::filesystem::path ChunkManager::getChunkFilename(const ChunkCoord& coord) {
  // Using 'p' for positive and 'n' for negative prefixes
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));

  return chunk_save_dir_ / (x_str + "_" + y_str + "_" + z_str);
}

// Private version that assumes lock is already held
bool ChunkManager::loadChunkNoLock(const ChunkCoord& coord, bool background) {
  auto timer = ProfilingUtils::Timer("ChunkManager::loadChunk");

  // Check if already loaded
  auto it = active_chunks_.find(coord);
  if (it != active_chunks_.end()) {
    if (background) {
      // Update metadata
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end()) {
        meta_it->second.loading = false;
      }
    }
    incrementStat(stats_.cache_hits);
    return true;
  }

  auto chunk_filename = getChunkFilename(coord);

  if (!chunkExistsOnDiskNoLock(coord)) {
    if (background) {
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end()) {
        meta_it->second.loading = false;
      }
    }
    return false;
  }

  try {
    // Create new chunk with mapper's GaussianModelParams
    auto chunk = std::make_shared<Chunk>(getModelParams(), coord);
    if (!chunk || !chunk->getGaussians()) {
      std::cerr << "Failed to create chunk object" << std::endl;
      return false;
    }

    // Load from file
    // chunk->getGaussians()->load_checkpoint(chunk_filename.string(),
    // getOptParams());
    chunk->getGaussians()->load_checkpoint_incremental(
        chunk_filename.string(), getOptParams(), true, true, true);

    // Update in-memory structures
    active_chunks_[coord] = chunk;

    // Update metadata
    auto& meta = chunk_metadata_[coord];
    meta.load_time = std::chrono::steady_clock::now();
    meta.last_used = meta.load_time;
    meta.loading = false;
    meta.dirty = false;
    meta.usage_count = 0;

    incrementStat(stats_.active_chunks);
    incrementStat(stats_.disk_loads);

    // std::cout << "Successfully loaded chunk from disk: " << coord.x << "
    // "
    //           << coord.y << " " << coord.z << " " << std::endl;

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception loading chunk: " << e.what() << std::endl;
    throw std::runtime_error("Chunk could not be loaded");

    if (background) {
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end()) {
        meta_it->second.loading = false;
      }
    }

    return false;
  }
}

// Public version that acquires the lock
bool ChunkManager::loadChunk(const ChunkCoord& coord, bool background) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return loadChunkNoLock(coord, background);
}

// Private version that assumes lock is already held
bool ChunkManager::saveChunkNoLock(const ChunkCoord& coord, bool background) {
  auto timer = ProfilingUtils::Timer("ChunkManager::saveChunk");

  // First check if chunk exists and get it
  auto it = active_chunks_.find(coord);
  if (it == active_chunks_.end() || !it->second ||
      !it->second->getGaussians()) {
    // Update metadata
    auto meta_it = chunk_metadata_.find(coord);
    if (meta_it != chunk_metadata_.end()) {
      meta_it->second.saving = false;
    }
    return false;
  }
  std::shared_ptr<Chunk> chunk = it->second;

  auto chunk_filename = getChunkFilename(coord);

  try {
    // Save to file
    chunk->getGaussians()->save_checkpoint(chunk_filename.string());

    // Update metadata
    auto meta_it = chunk_metadata_.find(coord);
    if (meta_it != chunk_metadata_.end()) {
      meta_it->second.dirty = false;
      meta_it->second.saving = false;
    }

    chunk_exists_cache_[coord] = true;

    incrementStat(stats_.disk_saves);

    // Remove from memory if was a background save for eviction
    if (background) {
      active_chunks_.erase(coord);
      decrementStat(stats_.active_chunks);
    }

    // Clear CUDA cache after saving to free memory
    c10::cuda::CUDACachingAllocator::emptyCache();

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception saving chunk: " << e.what() << std::endl;
    throw std::runtime_error("Exception saving chunk");

    // Update metadata
    auto meta_it = chunk_metadata_.find(coord);
    if (meta_it != chunk_metadata_.end()) {
      meta_it->second.saving = false;
    }

    return false;
  }
}

// Public version that acquires the lock
bool ChunkManager::saveChunk(const ChunkCoord& coord, bool background) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return saveChunkNoLock(coord, background);
}

// Get chunk at specific coordinate
std::shared_ptr<Chunk> ChunkManager::getChunkAtNoLock(const ChunkCoord& coord) {
  auto it = active_chunks_.find(coord);
  if (it != active_chunks_.end()) {
    return it->second;
  }
  return nullptr;
}
// Get chunk at specific coordinate
std::shared_ptr<Chunk> ChunkManager::getChunkAt(const ChunkCoord& coord) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return getChunkAtNoLock(coord);
}

// Private version that assumes lock is already held
bool ChunkManager::chunkExistsOnDiskNoLock(const ChunkCoord& coord) {
  // Check cache first
  auto it = chunk_exists_cache_.find(coord);
  if (it != chunk_exists_cache_.end()) {
    // std::cout << "Chunk exists in disk cache with status: " << it->second
    //           << std::endl;
    return it->second;
  }

  // Check filesystem
  auto chunk_filename = getChunkFilename(coord);
  bool exists = std::filesystem::exists(chunk_filename);

  // Update cache
  chunk_exists_cache_[coord] = exists;

  return exists;
}

// Public version that acquires the lock
bool ChunkManager::chunkExistsOnDisk(const ChunkCoord& coord) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return chunkExistsOnDiskNoLock(coord);
}

// Get chunk coordinate from 3D position
ChunkCoord ChunkManager::getChunkCoord(const Eigen::Vector3f& position) {
  float effective_size = chunk_size_ - overlap_margin_;  // Account for overlap
  return ChunkCoord{
      static_cast<int64_t>(std::floor(position.x() / effective_size)),
      static_cast<int64_t>(std::floor(position.y() / effective_size)),
      static_cast<int64_t>(std::floor(position.z() / effective_size))};
}

// Get chunk center
Eigen::Vector3f ChunkManager::getChunkCenter(const ChunkCoord& coord) {
  float effective_size = chunk_size_ - overlap_margin_;
  return Eigen::Vector3f((coord.x + 0.5f) * effective_size,
                         (coord.y + 0.5f) * effective_size,
                         (coord.z + 0.5f) * effective_size);
}

// Calculate AABB for a chunk
AABB ChunkManager::getChunkAABB(const ChunkCoord& coord) {
  float effective_size = chunk_size_ - overlap_margin_;

  // Calculate minimum corner of the chunk
  Eigen::Vector3f min_corner(coord.x * effective_size, coord.y * effective_size,
                             coord.z * effective_size);

  // Calculate maximum corner of the chunk (including overlap margin)
  Eigen::Vector3f max_corner =
      min_corner + Eigen::Vector3f::Constant(chunk_size_);

  return AABB(min_corner, max_corner);
}

// std::vector<std::shared_ptr<Chunk>> ChunkManager::getVisibleChunks(
//     std::shared_ptr<GaussianKeyframe> keyframe) {
//   std::vector<std::shared_ptr<Chunk>> visible_chunks;

//   if (active_chunks_.empty()) {
//     return visible_chunks;
//   }

//   bool use_simd = false;

//   // Calculate view-projection matrix from the keyframe
//   Eigen::Matrix4f view_matrix =
//       keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);

//   // Create projection matrix using Eigen (based on the keyframe's
//   // getProjectionMatrix method)
//   Eigen::Matrix4f proj_matrix = Eigen::Matrix4f::Zero();
//   float fovX = keyframe->FoVx_;
//   float fovY = keyframe->FoVy_;
//   float znear = keyframe->znear_;
//   float zfar = keyframe->zfar_;

//   float tanHalfFovY = std::tan(fovY / 2);
//   float tanHalfFovX = std::tan(fovX / 2);
//   float top = tanHalfFovY * znear;
//   float bottom = -top;
//   float right = tanHalfFovX * znear;
//   float left = -right;

//   proj_matrix(0, 0) = 2.0f * znear / (right - left);
//   proj_matrix(1, 1) = 2.0f * znear / (top - bottom);
//   proj_matrix(0, 2) = (right + left) / (right - left);
//   proj_matrix(1, 2) = (top + bottom) / (top - bottom);
//   proj_matrix(3, 2) = 1.0f;  // z_sign
//   proj_matrix(2, 2) = zfar / (zfar - znear);
//   proj_matrix(2, 3) = -(zfar * znear) / (zfar - znear);

//   // Calculate the view-projection matrix
//   Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

//   // Test each active chunk against the frustum
//   for (const auto& [chunk_coord, chunk] : active_chunks_) {
//     AABB chunk_aabb = getChunkAABB(chunk_coord);

//     bool visible;
//     if (use_simd) {
//       // visible = test_AABB_against_frustum_256(vp_matrix, chunk_aabb);
//     } else {
//       visible = test_AABB_against_frustum_eigen(vp_matrix, chunk_aabb);
//     }

//     if (visible) {
//       visible_chunks.push_back(chunk);
//     }
//   }

//   return visible_chunks;
// }

// Main function that returns visible chunks, handling both active and on-disk
// chunks
std::vector<std::shared_ptr<Chunk>> ChunkManager::getVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  auto timer = ProfilingUtils::Timer("ChunkManager::getVisibleChunks");
  std::cout << "Called getVisibleChunks" << std::endl;

  if (!keyframe) {
    std::cerr << "Error: Null keyframe passed to getVisibleChunks" << std::endl;
    return {};
  }

  // Get camera parameters and calculate view-projection matrix
  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);
  Eigen::Matrix4f proj_matrix = createProjectionMatrix(keyframe);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Get camera position to calculate chunk search radius
  Sophus::SE3d camera_pose = keyframe->getPose();
  Sophus::SE3d Twc = camera_pose.inverse();  // World to camera transform
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position);

  // Determine search radius based on far plane distance (with limit)
  int search_radius = std::min(std::ceil(keyframe->zfar_ / chunk_size_), 10.0f);

  // Results to be returned
  std::vector<std::shared_ptr<Chunk>> all_visible_chunks;

  // ACQUIRE LOCK ONCE AND HOLD IT FOR THE ENTIRE OPERATION
  {
    std::lock_guard<std::mutex> lock(io_mutex_);

    // Find visible chunks (both active and on-disk)
    auto [visible_active_chunks, chunks_to_load] =
        findVisibleChunks(camera_chunk, search_radius, camera_position,
                          keyframe->zfar_, vp_matrix);

    // Manage memory if needed before loading new chunks
    // Instead of calling the function, inline the memory management logic
    if (active_chunks_.size() + chunks_to_load.size() > max_chunks_in_memory_) {
      int to_evict = std::min(
          static_cast<int>(chunks_to_load.size()),
          static_cast<int>(active_chunks_.size() + chunks_to_load.size() -
                           max_chunks_in_memory_));

      // Instead of calling evictUnusedChunks, inline the eviction logic
      std::vector<ChunkCoord> to_evict_chunks =
          findChunksToEvictNoLock(to_evict);

      // Schedule saves for chunks to evict
      for (const auto& coord : to_evict_chunks) {
        // Use scheduleChunkSaveNoLock to avoid locking the mutex again
        scheduleChunkSaveNoLock(coord, 10);  // High priority for eviction
      }
    }

    // Start with active chunks
    all_visible_chunks = visible_active_chunks;

    // Load necessary chunks from disk and add to visible chunks
    for (const auto& coord : chunks_to_load) {
      if (loadChunkNoLock(
              coord)) {  // Use no-lock version since we already have the lock
        auto chunk = getChunkAtNoLock(coord);
        if (chunk && chunk->getGaussians()) {
          all_visible_chunks.push_back(chunk);
          markChunkUsedNoLock(coord);
        }
      } else {
        // Log the error instead of throwing exception
        std::cerr << "Warning: Unable to load chunk from disk: " << coord.x
                  << "," << coord.y << "," << coord.z << std::endl;
      }
    }
  }
  // LOCK RELEASED HERE

  return all_visible_chunks;
}

// Helper function to create the projection matrix from keyframe parameters
Eigen::Matrix4f ChunkManager::createProjectionMatrix(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  Eigen::Matrix4f proj_matrix = Eigen::Matrix4f::Zero();
  float fovX = keyframe->FoVx_;
  float fovY = keyframe->FoVy_;
  float znear = keyframe->znear_;
  float zfar = keyframe->zfar_;

  float tanHalfFovY = std::tan(fovY / 2);
  float tanHalfFovX = std::tan(fovX / 2);
  float top = tanHalfFovY * znear;
  float bottom = -top;
  float right = tanHalfFovX * znear;
  float left = -right;

  proj_matrix(0, 0) = 2.0f * znear / (right - left);
  proj_matrix(1, 1) = 2.0f * znear / (top - bottom);
  proj_matrix(0, 2) = (right + left) / (right - left);
  proj_matrix(1, 2) = (top + bottom) / (top - bottom);
  proj_matrix(3, 2) = 1.0f;  // z_sign
  proj_matrix(2, 2) = zfar / (zfar - znear);
  proj_matrix(2, 3) = -(zfar * znear) / (zfar - znear);

  return proj_matrix;
}

// Helper function to find visible chunks within search radius
std::pair<std::vector<std::shared_ptr<Chunk>>, std::vector<ChunkCoord>>
ChunkManager::findVisibleChunks(const ChunkCoord& camera_chunk,
                                int search_radius,
                                const Eigen::Vector3f& camera_position,
                                float zfar,
                                const Eigen::Matrix4f& vp_matrix) {
  std::vector<std::shared_ptr<Chunk>> visible_active_chunks;
  std::vector<ChunkCoord> chunks_to_load;

  for (int dx = -search_radius; dx <= search_radius; dx++) {
    for (int dy = -search_radius; dy <= search_radius; dy++) {
      for (int dz = -search_radius; dz <= search_radius; dz++) {
        ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                               camera_chunk.z + dz};

        // Skip chunks that are too far from camera (rough distance check)
        Eigen::Vector3f chunk_center = getChunkCenter(check_coord);
        float dist_to_camera = (chunk_center - camera_position).norm();
        if (dist_to_camera >
            zfar + chunk_size_ * 1.732f) {  // sqrt(3) for diagonal
          continue;
        }

        // Get AABB for the chunk and test against frustum
        AABB chunk_aabb = getChunkAABB(check_coord);
        bool visible = test_AABB_against_frustum_eigen(vp_matrix, chunk_aabb);

        if (visible) {
          auto it = active_chunks_.find(check_coord);

          // If chunk is active, add to visible chunks
          if (it != active_chunks_.end() && it->second &&
              it->second->getGaussians()) {
            visible_active_chunks.push_back(it->second);
            markChunkUsedNoLock(check_coord);
          }
          // If chunk exists on disk but not loaded, queue for loading
          else if (chunkExistsOnDiskNoLock(check_coord)) {
            chunks_to_load.push_back(check_coord);
          }
        }
      }
    }
  }

  return {visible_active_chunks, chunks_to_load};
}

// Helper function to manage memory before loading new chunks
void ChunkManager::manageMemoryForNewChunks(size_t chunks_to_load_count) {
  if (active_chunks_.size() + chunks_to_load_count > max_chunks_in_memory_) {
    int to_evict =
        std::min(static_cast<int>(chunks_to_load_count),
                 static_cast<int>(active_chunks_.size() + chunks_to_load_count -
                                  max_chunks_in_memory_));
    evictUnusedChunks(to_evict);
  }
}

// Helper function to load visible chunks from disk
void ChunkManager::loadVisibleChunks(
    const std::vector<ChunkCoord>& chunks_to_load,
    std::vector<std::shared_ptr<Chunk>>& visible_chunks) {
  for (const auto& coord : chunks_to_load) {
    if (loadChunkNoLock(
            coord)) {  // Use no-lock version since we already have the lock
      auto chunk = getChunkAtNoLock(coord);
      if (chunk && chunk->getGaussians()) {
        visible_chunks.push_back(chunk);
        markChunkUsedNoLock(coord);
      }
    } else {
      // Consider logging the error instead of throwing exception
      std::cerr << "Warning: Unable to load chunk from disk: " << coord.x << ","
                << coord.y << "," << coord.z << std::endl;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ChunkManager::groupPointsByChunk(const torch::Tensor& positions) {
  // Called by addPoints
  // Convert positions to chunk coordinates
  torch::Tensor chunk_coords = torch::floor(positions / chunk_size_);
  std::cout << "New ungrouped points " << positions.size(0) << std::endl;

  // Convert to int64 for bit operations
  chunk_coords = chunk_coords.to(torch::kInt64);

  std::cout << "Chunk coordinates range: "
            << torch::min(chunk_coords).item<int64_t>() << " to "
            << torch::max(chunk_coords).item<int64_t>() << std::endl;

  // Offset coordinates to ensure they're positive
  auto x = chunk_coords.index({torch::indexing::Slice(), 0}) + 2048;
  auto y = chunk_coords.index({torch::indexing::Slice(), 1}) + 2048;
  auto z = chunk_coords.index({torch::indexing::Slice(), 2}) + 2048;

  // Use 13 bits per dimension (8192 range per dimension)
  auto x_shifted =
      torch::bitwise_left_shift(x, 26);  // 26 bits for first component
  auto y_shifted =
      torch::bitwise_left_shift(y, 13);  // 13 bits for second component
  // z uses remaining 13 bits, no shift needed

  // Combine using bitwise OR
  torch::Tensor chunk_ids =
      torch::bitwise_or(torch::bitwise_or(x_shifted, y_shifted), z);

  // Get unique chunks, inverse indices, and counts
  // Using unique_dim with proper parameters
  auto [unique_chunk_ids, inverse_indices, points_per_chunk] =
      torch::unique_dim(chunk_ids, 0, true, true, true);

  std::cout << "Number of unique chunks: " << unique_chunk_ids.size(0)
            << std::endl;

  // Convert unique chunk IDs back to 3D coordinates
  torch::Tensor unique_coords = torch::zeros(
      {unique_chunk_ids.size(0), 3},
      torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));

  // Extract coordinates using masks
  unique_coords.index({torch::indexing::Slice(), 0}) =
      torch::bitwise_and(torch::bitwise_right_shift(unique_chunk_ids, 26),
                         8191) -
      2048;

  unique_coords.index({torch::indexing::Slice(), 1}) =
      torch::bitwise_and(torch::bitwise_right_shift(unique_chunk_ids, 13),
                         8191) -
      2048;

  unique_coords.index({torch::indexing::Slice(), 2}) =
      torch::bitwise_and(unique_chunk_ids, 8191) - 2048;

  std::cout << "Reconstructed coordinate range: "
            << torch::min(unique_coords).item<int64_t>() << " to "
            << torch::max(unique_coords).item<int64_t>() << std::endl;

  return std::make_tuple(unique_coords, inverse_indices, points_per_chunk);
}

// Add points to appropriate chunks
void ChunkManager::addPointsToChunks(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes,
    float cameras_extent) {
  int min_new_points_threshold = 10;
  std::cout << "addPoints called in ChunkManager" << std::endl;
  // Filter points by depth first
  auto [filtered_points, filtered_colors] =
      filterPointsByDepth(points, colors, keyframes);

  if (filtered_points.sizes()[0] < min_new_points_threshold) {
    std::cout << "Too little points, exiting" << std::endl;
    return;
  }

  // Convert to CUDA for processing
  torch::Tensor points_cuda = filtered_points.to(torch::kCUDA);
  torch::Tensor colors_cuda = filtered_colors.to(torch::kCUDA);

  // Group points by chunk
  auto [unique_chunks, inverse_indices, points_per_chunk] =
      groupPointsByChunk(points_cuda);

  // Process each unique chunk
  for (int64_t i = 0; i < unique_chunks.size(0); i++) {
    // Get chunk coordinate
    ChunkCoord coord{unique_chunks[i][0].item<int64_t>(),
                     unique_chunks[i][1].item<int64_t>(),
                     unique_chunks[i][2].item<int64_t>()};

    std::cout << "Adding points for chunk: " << coord.x << " " << coord.y << " "
              << coord.z << " " << std::endl;

    // Create mask for points in this chunk
    torch::Tensor chunk_mask = (inverse_indices == i);

    // Extract points and features for this chunk
    torch::Tensor chunk_points = points_cuda.index({chunk_mask});
    torch::Tensor chunk_colors = colors_cuda.index({chunk_mask});

    // Skip if not enough points
    if (chunk_points.sizes()[0] < min_new_points_threshold) {
      std::cout << "Too little points, skipping" << std::endl;
      continue;
    }

    // Get or initialize chunk
    std::shared_ptr<Chunk> chunk;
    bool is_new_chunk = false;

    {
      std::lock_guard<std::mutex> lock(io_mutex_);

      // Check if already in memory
      auto it = active_chunks_.find(coord);
      if (it != active_chunks_.end()) {
        chunk = it->second;
        std::cout << "Chunk found in memory" << std::endl;
      }
      // Try to load from disk
      else if (chunkExistsOnDiskNoLock(coord)) {  // Use no-lock version
        if (loadChunkNoLock(coord, false)) {      // Use no-lock version
          chunk = active_chunks_[coord];
          std::cout << "Chunk loaded from disk" << std::endl;
        } else {
          // Loading failed, create new
          chunk = std::make_shared<Chunk>(model_params_, coord);
          active_chunks_[coord] = chunk;
          is_new_chunk = true;
        }
      }
      // Create new chunk
      else {
        std::cout << "Creating new chunk" << std::endl;
        chunk = std::make_shared<Chunk>(model_params_, coord);
        active_chunks_[coord] = chunk;

        // Initialize metadata
        chunk_metadata_[coord] = ChunkMetadata();
        is_new_chunk = true;

        incrementStat(stats_.active_chunks);
      }

      // Mark as used
      markChunkUsedNoLock(coord);  // Use no-lock version
    }

    // Initialize or add points to the chunk
    if (is_new_chunk) {
      std::cout << "Since new chunk, calling setup" << std::endl;
      // For new chunks, initialize with points
      chunk->getGaussians()->createFromPcd(chunk_points, chunk_colors,
                                           cameras_extent);

      chunk->getGaussians()->trainingSetup(opt_params_);
    } else {
      // For existing chunks, add new points
      std::cout << "Chunk already exists, adding points" << std::endl;
      chunk->getGaussians()->increasePcd(chunk_points, chunk_colors,
                                         getCurrentIteration());
    }
  }
}

// Shutdown the manager
void ChunkManager::shutdown() {
  // Signal thread to terminate
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    should_terminate_ = true;
  }

  // Notify waiting thread
  io_cv_.notify_all();

  // Wait for thread to finish
  if (io_thread_.joinable()) {
    io_thread_.join();
  }
}

// Get stats
ChunkManager::Stats ChunkManager::getStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

// Update statistics helper
void ChunkManager::incrementStat(int& stat) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stat++;
}

// Update statistics helper
void ChunkManager::decrementStat(int& stat) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  if (stat > 0) stat--;
}

// Cull chunks with too few points to ensure rendering stability
bool ChunkManager::cullSparseChunks(int min_points_threshold) {
  std::vector<ChunkCoord> chunks_to_cull;
  bool any_culled = false;

  {
    std::lock_guard<std::mutex> lock(io_mutex_);

    // Examine all active chunks
    for (const auto& [coord, chunk] : active_chunks_) {
      if (!chunk || !chunk->getGaussians()) continue;

      // Get number of active points in chunk
      int num_points = chunk->getGaussians()->getXYZ().size(0);

      // If below threshold, mark for culling
      if (num_points < min_points_threshold) {
        chunks_to_cull.push_back(coord);
      }
    }
  }

  // Remove culled chunks
  for (const auto& coord : chunks_to_cull) {
    // Just remove from active chunks
    {
      std::lock_guard<std::mutex> lock(io_mutex_);
      active_chunks_.erase(coord);
      decrementStat(stats_.active_chunks);

      // Update disk cache to prevent reloading
      chunk_exists_cache_[coord] = false;

      // Mark for background deletion if needed
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end()) {
        if (meta_it->second.dirty) {
          // Queue for background deletion rather than handling now
          io_queue_.push(ChunkIORequest(coord, ChunkOperation::DELETE, 5));
          io_cv_.notify_one();
        }
      }
    }

    any_culled = true;
  }

  return any_culled;
}