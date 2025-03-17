#include "include/chunk_manager.h"

#include <torch/cuda.h>

#include <algorithm>
#include <iostream>

#include "include/profiling.h"

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

// Get chunks visible from a keyframe
std::vector<std::shared_ptr<Chunk>> ChunkManager::getVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  auto timer = ProfilingUtils::Timer("ChunkManager::getVisibleChunks");

  if (!keyframe) {
    std::cerr << "Error: Null keyframe passed to getVisibleChunks" << std::endl;
    return {};
  }

  // Extract camera parameters
  Sophus::SE3d camera_pose = keyframe->getPose();
  Sophus::SE3d Twc = camera_pose.inverse();  // World to camera transform
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();

  // Create camera frustum planes in world space
  std::array<Eigen::Vector4f, 6> frustum_planes =
      computeFrustumPlanes(keyframe);

  // Find visible chunks
  std::vector<std::shared_ptr<Chunk>> visible_chunks;
  std::vector<ChunkCoord> chunks_to_load;

  // Determine search radius based on far plane distance
  int search_radius = std::ceil(keyframe->zfar_ / chunk_size_);
  search_radius = std::min(search_radius, 10);  // Limit search radius

  // Get chunk coordinate for camera position
  ChunkCoord camera_chunk = getChunkCoord(camera_position);
  std::cout << "Keyframe camera in chunk: " << camera_chunk.x << " "
            << camera_chunk.y << " " << camera_chunk.z << " " << std::endl;

  // Search chunks in the vicinity
  {
    std::lock_guard<std::mutex> lock(io_mutex_);

    for (int dx = -search_radius; dx <= search_radius; dx++) {
      for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dz = -search_radius; dz <= search_radius; dz++) {
          ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                                 camera_chunk.z + dz};

          std::cout << "Checking chunk for visibility: " << check_coord.x << " "
                    << check_coord.y << " " << check_coord.z << " "
                    << std::endl;

          // Skip chunks that are too far from camera (rough distance check)
          Eigen::Vector3f chunk_center = getChunkCenter(check_coord);
          float dist_to_camera = (chunk_center - camera_position).norm();
          if (dist_to_camera >
              keyframe->zfar_ + chunk_size_ * 1.732f) {  // sqrt(3) for diagonal
            std::cout << "Skipping as " << dist_to_camera << "is too far away"
                      << std::endl;
            continue;
          }

          // Check if chunk is inside or intersects view frustum
          if (isChunkInFrustum(check_coord, frustum_planes)) {
            auto it = active_chunks_.find(check_coord);
            std::cout << "Chunk is in viewing frustum" << std::endl;

            // If chunk is active, add to visible chunks
            if (it != active_chunks_.end() && it->second &&
                it->second->gaussians_) {
              std::cout << "Chunk already in memory" << std::endl;
              visible_chunks.push_back(it->second);
              markChunkUsedNoLock(check_coord);  // Use no-lock version
            }
            // If chunk exists on disk but not loaded, queue for loading
            else if (chunkExistsOnDiskNoLock(
                         check_coord)) {  // Use no-lock version
              std::cout << "Chunk exists on disk, load soon" << std::endl;
              chunks_to_load.push_back(check_coord);
            }
          }
        }
      }
    }
  }

  // If we're near memory limit, evict some chunks before loading new ones
  if (active_chunks_.size() + chunks_to_load.size() > max_chunks_in_memory_) {
    std::cout << "Need to evict chunks. Have " << active_chunks_.size()
              << std::endl;
    std::cout << "Want to load " << chunks_to_load.size() << std::endl;
    std::cout << "Which goes over " << max_chunks_in_memory_ << " limit"
              << std::endl;
    int to_evict = std::min(
        static_cast<int>(chunks_to_load.size()),
        static_cast<int>(active_chunks_.size() + chunks_to_load.size() -
                         max_chunks_in_memory_));
    evictUnusedChunks(to_evict);
  }

  // Load necessary chunks (synchronously for now, as we need them for
  // rendering)
  for (const auto& coord : chunks_to_load) {
    std::cout << "[Synchronous Request] Processing request to load chunk: "
              << coord.x << " " << coord.y << " " << coord.z << " "
              << std::endl;
    if (loadChunk(coord)) {  // Use the public version which handles locking
      auto chunk = getChunkAt(coord);  // Use accessor method
      if (chunk && chunk->gaussians_) {
        visible_chunks.push_back(chunk);
        markChunkUsed(coord);  // Use public version
      }
    } else {
      throw std::runtime_error("Not able to load chunk from disk");
    }
  }

  return visible_chunks;
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

// Preload chunks for upcoming keyframes
void ChunkManager::preloadChunksForKeyframes(
    const std::vector<std::shared_ptr<GaussianKeyframe>>& upcoming) {
  if (upcoming.empty()) return;

  std::unordered_set<ChunkCoord, ChunkCoordHash> chunks_to_preload;

  // Find chunks needed for upcoming keyframes
  for (const auto& keyframe : upcoming) {
    if (!keyframe) continue;

    Sophus::SE3d camera_pose = keyframe->getPose();
    Sophus::SE3d Twc = camera_pose.inverse();
    Eigen::Vector3f camera_position = Twc.translation().cast<float>();

    std::array<Eigen::Vector4f, 6> frustum_planes =
        computeFrustumPlanes(keyframe);

    // Determine search radius
    int search_radius =
        std::min(2, static_cast<int>(std::ceil(keyframe->zfar_ / chunk_size_)));

    // Get chunk coordinate for camera position
    ChunkCoord camera_chunk = getChunkCoord(camera_position);

    // Find chunks in view frustum (smaller radius for preloading)
    for (int dx = -search_radius; dx <= search_radius; dx++) {
      for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dz = -search_radius; dz <= search_radius; dz++) {
          ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                                 camera_chunk.z + dz};

          // Skip chunks that are too far
          Eigen::Vector3f chunk_center = getChunkCenter(check_coord);
          float dist_to_camera = (chunk_center - camera_position).norm();
          if (dist_to_camera > keyframe->zfar_ + chunk_size_) {
            continue;
          }

          // Check if in frustum and not already loaded
          {
            std::lock_guard<std::mutex> lock(io_mutex_);
            if (isChunkInFrustum(check_coord, frustum_planes) &&
                active_chunks_.find(check_coord) == active_chunks_.end() &&
                chunkExistsOnDiskNoLock(check_coord)) {  // Use no-lock version
              chunks_to_preload.insert(check_coord);
            }
          }
        }
      }
    }
  }

  // If we'd exceed memory limit, don't preload
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (active_chunks_.size() + chunks_to_preload.size() >
        max_chunks_in_memory_) {
      return;
    }
  }

  // Queue preloading with low priority
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    for (const auto& coord : chunks_to_preload) {
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it == chunk_metadata_.end() ||
          (!meta_it->second.loading && !meta_it->second.saving)) {
        if (meta_it == chunk_metadata_.end()) {
          chunk_metadata_[coord] = ChunkMetadata();
        }
        chunk_metadata_[coord].loading = true;
        io_queue_.push(
            ChunkIORequest(coord, ChunkOperation::LOAD, -10));  // Low priority
        incrementStat(stats_.prefetched);
      }
    }

    // Notify I/O thread
    io_cv_.notify_one();
  }
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
    auto chunk = std::make_shared<Chunk>(getModelParams());
    if (!chunk || !chunk->gaussians_) {
      std::cerr << "Failed to create chunk object" << std::endl;
      return false;
    }

    // Load from file
    // chunk->gaussians_->load_checkpoint(chunk_filename.string(),
    // getOptParams());
    chunk->gaussians_->load_checkpoint_incremental(
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

    std::cout << "Successfully loaded chunk from disk: " << coord.x << " "
              << coord.y << " " << coord.z << " " << std::endl;

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
  if (it == active_chunks_.end() || !it->second || !it->second->gaussians_) {
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
    chunk->gaussians_->save_checkpoint(chunk_filename.string());

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
std::shared_ptr<Chunk> ChunkManager::getChunkAt(const ChunkCoord& coord) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  auto it = active_chunks_.find(coord);
  if (it != active_chunks_.end()) {
    return it->second;
  }
  return nullptr;
}

// Private version that assumes lock is already held
bool ChunkManager::chunkExistsOnDiskNoLock(const ChunkCoord& coord) {
  // Check cache first
  auto it = chunk_exists_cache_.find(coord);
  if (it != chunk_exists_cache_.end()) {
    std::cout << "Chunk exists in disk cache with status: " << it->second
              << std::endl;
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

// Get chunk corners (for frustum culling)
std::array<Eigen::Vector3f, 8> ChunkManager::getChunkCorners(
    const ChunkCoord& coord) {
  float effective_size = chunk_size_ - overlap_margin_;
  float x = coord.x * effective_size;
  float y = coord.y * effective_size;
  float z = coord.z * effective_size;

  std::array<Eigen::Vector3f, 8> corners;
  corners[0] = Eigen::Vector3f(x, y, z);
  corners[1] = Eigen::Vector3f(x + chunk_size_, y, z);
  corners[2] = Eigen::Vector3f(x, y + chunk_size_, z);
  corners[3] = Eigen::Vector3f(x + chunk_size_, y + chunk_size_, z);
  corners[4] = Eigen::Vector3f(x, y, z + chunk_size_);
  corners[5] = Eigen::Vector3f(x + chunk_size_, y, z + chunk_size_);
  corners[6] = Eigen::Vector3f(x, y + chunk_size_, z + chunk_size_);
  corners[7] =
      Eigen::Vector3f(x + chunk_size_, y + chunk_size_, z + chunk_size_);

  return corners;
}

// Get chunk center
Eigen::Vector3f ChunkManager::getChunkCenter(const ChunkCoord& coord) {
  float effective_size = chunk_size_ - overlap_margin_;
  return Eigen::Vector3f((coord.x + 0.5f) * effective_size,
                         (coord.y + 0.5f) * effective_size,
                         (coord.z + 0.5f) * effective_size);
}

// Create a plane from 3 points
Eigen::Vector4f ChunkManager::planeFromPoints(const Eigen::Vector3f& p1,
                                              const Eigen::Vector3f& p2,
                                              const Eigen::Vector3f& p3) {
  Eigen::Vector3f v1 = p2 - p1;
  Eigen::Vector3f v2 = p3 - p1;
  Eigen::Vector3f normal = v1.cross(v2).normalized();
  float d = -normal.dot(p1);
  return Eigen::Vector4f(normal.x(), normal.y(), normal.z(), d);
}

// Compute frustum planes for a keyframe
std::array<Eigen::Vector4f, 6> ChunkManager::computeFrustumPlanes(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  // Frustum planes: left, right, bottom, top, near, far
  std::array<Eigen::Vector4f, 6> planes;

  // Get camera parameters
  Sophus::SE3d camera_pose = keyframe->getPose();
  Sophus::SE3d Twc = camera_pose.inverse();  // World to camera transform
  Eigen::Vector3f camera_pos = Twc.translation().cast<float>();
  Eigen::Matrix3f R_wc = Twc.rotationMatrix().cast<float>();

  // Camera basis vectors in world space
  Eigen::Vector3f cam_right = R_wc.col(0);  // x-axis
  Eigen::Vector3f cam_up = R_wc.col(1);     // y-axis
  Eigen::Vector3f cam_forward =
      -R_wc.col(2);  // Camera looks down the negative z-axis

  // Compute frustum corners using FOV
  float near_z = keyframe->znear_;
  float far_z = keyframe->zfar_;

  // Calculate frustum dimensions at near and far planes
  float near_height = 2.0f * near_z * std::tan(keyframe->FoVy_ * 0.5f);
  float near_width = 2.0f * near_z * std::tan(keyframe->FoVx_ * 0.5f);
  float far_height = 2.0f * far_z * std::tan(keyframe->FoVy_ * 0.5f);
  float far_width = 2.0f * far_z * std::tan(keyframe->FoVx_ * 0.5f);

  // Compute frustum corners in world space
  Eigen::Vector3f near_center = camera_pos + cam_forward * near_z;
  Eigen::Vector3f far_center = camera_pos + cam_forward * far_z;

  // Near plane corners
  Eigen::Vector3f ntl = near_center + (cam_up * near_height * 0.5f) -
                        (cam_right * near_width * 0.5f);
  Eigen::Vector3f ntr = near_center + (cam_up * near_height * 0.5f) +
                        (cam_right * near_width * 0.5f);
  Eigen::Vector3f nbl = near_center - (cam_up * near_height * 0.5f) -
                        (cam_right * near_width * 0.5f);
  Eigen::Vector3f nbr = near_center - (cam_up * near_height * 0.5f) +
                        (cam_right * near_width * 0.5f);

  // Far plane corners
  Eigen::Vector3f ftl = far_center + (cam_up * far_height * 0.5f) -
                        (cam_right * far_width * 0.5f);
  Eigen::Vector3f ftr = far_center + (cam_up * far_height * 0.5f) +
                        (cam_right * far_width * 0.5f);
  Eigen::Vector3f fbl = far_center - (cam_up * far_height * 0.5f) -
                        (cam_right * far_width * 0.5f);
  Eigen::Vector3f fbr = far_center - (cam_up * far_height * 0.5f) +
                        (cam_right * far_width * 0.5f);

  // Compute frustum planes (normal points inward)
  // Left plane
  planes[0] = planeFromPoints(camera_pos, ntl, ftl);

  // Right plane
  planes[1] = planeFromPoints(camera_pos, ftr, ntr);

  // Bottom plane
  planes[2] = planeFromPoints(camera_pos, nbr, fbr);

  // Top plane
  planes[3] = planeFromPoints(camera_pos, ftl, ntl);

  // Near plane
  planes[4] = planeFromPoints(ntl, ntr, nbl);

  // Far plane
  planes[5] = planeFromPoints(ftr, ftl, fbr);

  return planes;
}

// Check if a chunk is inside or intersects the frustum
bool ChunkManager::isChunkInFrustum(
    const ChunkCoord& coord,
    const std::array<Eigen::Vector4f, 6>& frustum_planes) {
  // Get chunk corners (AABB)
  std::array<Eigen::Vector3f, 8> corners = getChunkCorners(coord);

  // Check each plane
  for (const auto& plane : frustum_planes) {
    bool all_outside = true;

    // If all corners are on the negative side of a plane, the chunk is outside
    // the frustum
    for (const auto& corner : corners) {
      float dist = plane.x() * corner.x() + plane.y() * corner.y() +
                   plane.z() * corner.z() + plane.w();
      if (dist >= -chunk_size_ *
                      0.1f) {  // Add a small margin to prevent culling at edges
        all_outside = false;
        break;
      }
    }

    if (all_outside) {
      return false;  // Completely outside this plane, thus outside frustum
    }
  }

  return true;  // Inside or intersects the frustum
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

    std::cout << "Adding point for chunk: " << coord.x << " " << coord.y << " "
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
          chunk = std::make_shared<Chunk>(model_params_);
          active_chunks_[coord] = chunk;
          is_new_chunk = true;
        }
      }
      // Create new chunk
      else {
        std::cout << "Creating new chunk" << std::endl;
        chunk = std::make_shared<Chunk>(model_params_);
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
      chunk->gaussians_->createFromPcd(chunk_points, chunk_colors,
                                       cameras_extent);

      chunk->gaussians_->trainingSetup(opt_params_);
    } else {
      // For existing chunks, add new points
      std::cout << "Chunk already exists, adding points" << std::endl;
      chunk->gaussians_->increasePcd(chunk_points, chunk_colors,
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