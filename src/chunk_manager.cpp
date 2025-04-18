#include "include/chunk_manager.h"

#include <torch/cuda.h>

#include <algorithm>
#include <iostream>

#include "include/profiling.h"

// Pure Eigen implementation without explicit SIMD (relies on Eigen's
// optimizations)
bool test_AABB_against_frustum_eigen(const Eigen::Matrix4f& MVP,
                                     const AABB& aabb) {
  // Define the 8 corners of the AABB
  std::array<Eigen::Vector4f, 8> corners;
  corners[0] = Eigen::Vector4f(aabb.min.x(), aabb.min.y(), aabb.min.z(), 1.0f);
  corners[1] = Eigen::Vector4f(aabb.max.x(), aabb.min.y(), aabb.min.z(), 1.0f);
  corners[2] = Eigen::Vector4f(aabb.min.x(), aabb.max.y(), aabb.min.z(), 1.0f);
  corners[3] = Eigen::Vector4f(aabb.max.x(), aabb.max.y(), aabb.min.z(), 1.0f);
  corners[4] = Eigen::Vector4f(aabb.min.x(), aabb.min.y(), aabb.max.z(), 1.0f);
  corners[5] = Eigen::Vector4f(aabb.max.x(), aabb.min.y(), aabb.max.z(), 1.0f);
  corners[6] = Eigen::Vector4f(aabb.min.x(), aabb.max.y(), aabb.max.z(), 1.0f);
  corners[7] = Eigen::Vector4f(aabb.max.x(), aabb.max.y(), aabb.max.z(), 1.0f);

  // Test 1: Check if any corner is inside the view frustum
  bool any_corner_inside = false;
  bool all_corners_outside_same_plane = true;

  // Arrays to track which side of each frustum plane each corner is on
  bool outside_left[8] = {false};
  bool outside_right[8] = {false};
  bool outside_bottom[8] = {false};
  bool outside_top[8] = {false};
  bool outside_near[8] = {false};
  bool outside_far[8] = {false};

  // Transform and test all corners
  for (int i = 0; i < 8; ++i) {
    // Transform to clip space
    Eigen::Vector4f clipSpace = MVP * corners[i];

    // To handle perspective division properly
    float w = clipSpace.w();
    float x = clipSpace.x();
    float y = clipSpace.y();
    float z = clipSpace.z();

    // Check which side of each plane this corner is on
    outside_left[i] = x < -w;
    outside_right[i] = x > w;
    outside_bottom[i] = y < -w;
    outside_top[i] = y > w;
    outside_near[i] = z < 0;
    outside_far[i] = z > w;

    // If any corner is inside, we're done
    if (!outside_left[i] && !outside_right[i] && !outside_bottom[i] &&
        !outside_top[i] && !outside_near[i] && !outside_far[i]) {
      any_corner_inside = true;
    }
  }

  if (any_corner_inside) {
    return true;
  }

  // Test 2: If all corners are outside the same frustum plane, the AABB is
  // outside
  bool all_outside_left = true;
  bool all_outside_right = true;
  bool all_outside_bottom = true;
  bool all_outside_top = true;
  bool all_outside_near = true;
  bool all_outside_far = true;

  for (int i = 0; i < 8; ++i) {
    all_outside_left &= outside_left[i];
    all_outside_right &= outside_right[i];
    all_outside_bottom &= outside_bottom[i];
    all_outside_top &= outside_top[i];
    all_outside_near &= outside_near[i];
    all_outside_far &= outside_far[i];
  }

  // If all corners are outside any single plane, the AABB is outside the
  // frustum
  if (all_outside_left || all_outside_right || all_outside_bottom ||
      all_outside_top || all_outside_near || all_outside_far) {
    return false;
  }

  // Test 3: If we reach here, the AABB and frustum intersect
  // (No corner is inside, but the AABB isn't completely outside any plane)
  return true;
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

    bool visible = test_AABB_against_frustum_eigen(MVP, aabb_list[i]);

    if (visible) {
      out_visible_list.push_back(static_cast<u32>(i));
    }
  }
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

// Constructor
ChunkManager::ChunkManager(const GaussianModelParams& model_params,
                           const GaussianOptimizationParams& opt_params,
                           std::filesystem::path chunk_save_dir,
                           float chunk_size,
                           float overlap_margin,
                           int max_chunks,
                           int num_io_threads)
    : model_params_(model_params),
      opt_params_(opt_params),
      chunk_save_dir_(chunk_save_dir),
      chunk_size_(chunk_size),
      overlap_margin_(overlap_margin),
      max_chunks_in_memory_(max_chunks),
      should_terminate_(false) {
  std::cout << "Creating ChunkManager with async I/O threads: "
            << num_io_threads << std::endl;
  // Create save directory if it doesn't exist
  if (!chunk_save_dir_.empty() && !std::filesystem::exists(chunk_save_dir_)) {
    std::filesystem::create_directories(chunk_save_dir_);
  }

  // Initialize thread pool
  initializeThreadPool(num_io_threads);
}

// Destructor
ChunkManager::~ChunkManager() { shutdown(); }

// Initialize thread pool
void ChunkManager::initializeThreadPool(int num_threads) {
  // Create worker threads that process the operation queue
  for (int i = 0; i < num_threads; i++) {
    io_threads_.emplace_back([this]() { this->ioThreadFunction(); });
  }

  std::cout << "Initialized " << num_threads << " I/O worker threads"
            << std::endl;
}

// Shutdown thread pool
void ChunkManager::shutdownThreadPool() {
  std::cout << "Shutting down I/O thread pool..." << std::endl;

  // Signal threads to terminate
  shutdown_threads_ = true;

  // Wake up all threads
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.notify_all();
  }

  // Join all threads
  for (auto& thread : io_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  io_threads_.clear();
  std::cout << "I/O thread pool shutdown complete" << std::endl;
}

// Worker thread function
void ChunkManager::ioThreadFunction() {
  while (!shutdown_threads_) {
    std::shared_ptr<ChunkOperation> operation = nullptr;

    // Get next operation from queue
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() {
        return !operation_queue_.empty() || shutdown_threads_;
      });

      if (shutdown_threads_) {
        if (operation_queue_.empty()) {
          return;
        }
      }

      if (!operation_queue_.empty()) {
        operation = operation_queue_.top();
        operation_queue_.pop();
      }
    }

    if (operation) {
      // std::cout << "IO Thread: New operation popped" << std::endl;
      processOperation(operation);
    }
  }
}

// Enqueue an operation
void ChunkManager::enqueueOperation(std::shared_ptr<ChunkOperation> operation) {
  // std::cout << "Called enqueueOperation" << std::endl;
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    operation_queue_.push(operation);
  }

  queue_cv_.notify_one();
}

// State management methods
ChunkState ChunkManager::getChunkState(const ChunkCoord& coord) {
  // std::cout << "Called getChunkState" << std::endl;
  std::unique_lock<std::mutex> lock(metadata_mutex_);
  auto it = chunk_metadata_.find(coord);
  if (it != chunk_metadata_.end()) {
    return it->second.state.load();
  }
  std::cout << "Warning: No metadata found for this coord!" << std::endl;
  return ChunkState::INACTIVE;
}

bool ChunkManager::transitionChunkState(const ChunkCoord& coord,
                                        ChunkState expected,
                                        ChunkState new_state) {
  // std::cout << "Called transitionChunkState" << std::endl;
  std::unique_lock<std::mutex> lock(metadata_mutex_);
  auto& metadata = chunk_metadata_[coord];
  ChunkState current = metadata.state.load();

  if (current != expected) {
    std::cout << "Failed state transition for " << coord.x << "," << coord.y
              << "," << coord.z << ": expected=" << static_cast<int>(expected)
              << ", actual=" << static_cast<int>(current)
              << ", target=" << static_cast<int>(new_state) << std::endl;
    return false;  // State already changed
  }

  metadata.state.store(new_state);
  // std::cout << "Successful state transition for " << coord.x << "," <<
  // coord.y
  //           << "," << coord.z << ": " << static_cast<int>(expected) << " -> "
  //           << static_cast<int>(new_state) << std::endl;
  return true;
}

// Wait for a chunk to reach a specific state
bool ChunkManager::waitForChunkState(const ChunkCoord& coord,
                                     ChunkState target_state,
                                     std::chrono::milliseconds timeout) {
  // std::cout << "Called waitForChunkState" << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = start_time + timeout;

  while (std::chrono::steady_clock::now() < end_time) {
    {
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      auto it = chunk_metadata_.find(coord);
      if (it != chunk_metadata_.end()) {
        if (it->second.state.load() == target_state) {
          return true;
        }

        // Wait on the condition variable
        std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
        if (it->second.operation_cv.wait_until(op_lock, end_time) ==
            std::cv_status::timeout) {
          return false;
        }
      } else {
        if (target_state == ChunkState::INACTIVE) {
          std::cout << "Warning: No metadata found for this coord!"
                    << std::endl;
          return true;
        }
        // If the chunk doesn't exist in metadata and we're waiting for any
        // other state
        std::cout << "Warning: No metadata found for this coord!" << std::endl;
        return false;
      }
    }
  }

  return false;
}

// Asynchronous load with priority
std::future<bool> ChunkManager::loadChunkAsync(const ChunkCoord& coord,
                                               int priority,
                                               bool load_for_optimization,
                                               bool skip_busy_chunks) {
  // std::cout << "Called loadChunkAsync" << std::endl;
  ChunkState current_state = getChunkState(coord);
  // std::cout << "loadChunkAsync: " << static_cast<int>(current_state)
  //           << std::endl;

  // Return quickly if already in target state
  if ((current_state == ChunkState::ACTIVE && !load_for_optimization) ||
      (current_state == ChunkState::OPTIMIZING && load_for_optimization)) {
    std::promise<bool> promise;
    promise.set_value(true);
    return promise.get_future();
  }

  // Skip if chunk is busy and we're allowed to skip
  if (skip_busy_chunks && (current_state == ChunkState::SAVING ||
                           current_state == ChunkState::DELETING)) {
    std::cout << "Skipping busy chunk in state "
              << static_cast<int>(current_state) << ": " << coord.x << ","
              << coord.y << "," << coord.z << std::endl;
    std::promise<bool> promise;
    promise.set_value(false);  // Return false to indicate chunk was skipped
    return promise.get_future();
  }

  // Handle ACTIVE → OPTIMIZING transition
  if (current_state == ChunkState::ACTIVE && load_for_optimization) {
    // std::cout << "Attempting to transition chunk " << coord.x << "," <<
    // coord.y
    //           << "," << coord.z << " from ACTIVE to OPTIMIZING" << std::endl;

    if (transitionChunkState(coord, ChunkState::ACTIVE,
                             ChunkState::OPTIMIZING)) {
      // std::cout << "Successfully transitioned to OPTIMIZING" << std::endl;
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      optimizing_chunks_.insert(coord);

      std::promise<bool> promise;
      promise.set_value(true);
      return promise.get_future();
    } else {
      std::cout << "Failed to transition from ACTIVE to OPTIMIZING, current "
                   "state is: "
                << static_cast<int>(getChunkState(coord)) << std::endl;
      std::promise<bool> promise;
      promise.set_exception(std::make_exception_ptr(std::runtime_error(
          "Failed to transition from ACTIVE to OPTIMIZING")));
      return promise.get_future();
    }
  }

  // If already loading, return a future that waits for that operation
  if (current_state == ChunkState::LOADING) {
    return createWaitFuture(coord, load_for_optimization
                                       ? ChunkState::OPTIMIZING
                                       : ChunkState::ACTIVE);
  }

  // NEW: If saving, wait for it to complete before loading
  if (current_state == ChunkState::SAVING) {
    std::cout << "Chunk is being saved, waiting for completion before loading: "
              << coord.x << "," << coord.y << "," << coord.z << std::endl;

    // First wait for INACTIVE state (after save completes)
    auto inactive_future = createWaitFuture(coord, ChunkState::INACTIVE);

    // Create a shared state to continue after waiting
    auto shared_promise = std::make_shared<std::promise<bool>>();
    auto result_future = shared_promise->get_future();

    // Launch a continuation task
    std::thread([this, future = std::move(inactive_future), coord, priority,
                 load_for_optimization, shared_promise]() mutable {
      try {
        // Wait for save to complete
        bool saved = future.get();
        if (!saved) {
          shared_promise->set_value(false);
          return;
        }

        // Now try to load
        auto load_future =
            loadChunkAsync(coord, priority, load_for_optimization);
        shared_promise->set_value(load_future.get());
      } catch (const std::exception& e) {
        try {
          shared_promise->set_exception(std::current_exception());
        } catch (...) {
        }
      }
    }).detach();

    return result_future;
  }

  // Create new load operation if we can transition to LOADING state
  if (transitionChunkState(coord, ChunkState::INACTIVE, ChunkState::LOADING)) {
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::LOAD;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();
    operation->load_for_optimization = load_for_optimization;

    std::future<bool> future = operation->completion_promise.get_future();
    enqueueOperation(operation);

    return future;
  } else {
    // Failed to transition state - show current state
    ChunkState actual_state = getChunkState(coord);
    std::cout
        << "Failed to transition from INACTIVE to LOADING. Current state is: "
        << static_cast<int>(actual_state) << " for chunk " << coord.x << ","
        << coord.y << "," << coord.z << std::endl;

    std::promise<bool> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Failed to queue load operation")));
    return promise.get_future();
  }
}

// Asynchronous save with priority
std::future<bool> ChunkManager::saveChunkAsync(const ChunkCoord& coord,
                                               int priority) {
  std::cout << "Called saveChunkAsync" << std::endl;
  ChunkState current_state = getChunkState(coord);

  // Can't save if not active
  if (current_state != ChunkState::ACTIVE) {
    std::promise<bool> promise;
    if (current_state == ChunkState::SAVING) {
      // If already saving, return a future that waits for that operation
      return createWaitFuture(coord, ChunkState::INACTIVE);
    } else if (current_state == ChunkState::OPTIMIZING) {
      // If being optimized, reject the request or queue it for later
      promise.set_value(false);  // Can't save chunks being optimized
      return promise.get_future();
    } else {
      promise.set_value(false);  // Can't save non-active chunks
      return promise.get_future();
    }
  }

  // Create new save operation if we can transition to SAVING state
  if (transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::SAVING)) {
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::SAVE;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();

    std::future<bool> future = operation->completion_promise.get_future();
    enqueueOperation(operation);

    return future;
  } else {
    // Failed to transition state
    std::promise<bool> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Failed to queue save operation")));
    return promise.get_future();
  }
}

// Process a load operation
bool ChunkManager::processLoadOperation(const ChunkCoord& coord,
                                        bool load_for_optimization) {
  // std::cout << "Called processLoadOperation" << std::endl;

  bool is_active = false;
  {
    std::unique_lock<std::mutex> lock(active_chunks_mutex_);
    auto it = active_chunks_.find(coord);
    if (it != active_chunks_.end()) {
      is_active = true;
    }
  }

  if (!is_active) {
    try {
      if (!chunkExists(coord)) {
        std::cout << "Warning: Tried to load coord that doesn't exist!"
                  << std::endl;
        transitionChunkState(coord, ChunkState::LOADING, ChunkState::INACTIVE);
        return false;
      }

      auto chunk_filename = getChunkFilename(coord);

      // Create new chunk with model parameters
      auto chunk = std::make_shared<Chunk>(model_params_, coord);
      if (!chunk || !chunk->getGaussians()) {
        std::cerr << "Failed to create chunk object" << std::endl;
        transitionChunkState(coord, ChunkState::LOADING, ChunkState::INACTIVE);
        return false;
      }

      // Load from file
      chunk->getGaussians()->load_checkpoint_incremental(
          chunk_filename.string(), opt_params_, true, true, true);

      // Update in-memory structures
      {
        std::unique_lock<std::mutex> lock(active_chunks_mutex_);
        active_chunks_[coord] = chunk;
      }

      // Update metadata
      {
        std::unique_lock<std::mutex> lock(metadata_mutex_);
        auto& meta = chunk_metadata_[coord];
        meta.load_time = std::chrono::steady_clock::now();
        meta.last_used = meta.load_time;
        meta.usage_count = 0;
      }

      // Update cache
      {
        std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
        chunk_exists_cache_[coord] = true;
      }
    }

    catch (const std::exception& e) {
      std::cerr << "Exception in load operation: " << e.what() << std::endl;
      // Handle failure, revert to INACTIVE state
      transitionChunkState(coord, ChunkState::LOADING, ChunkState::INACTIVE);
      return false;
    }
  }

  // Transition to ACTIVE state
  if (load_for_optimization) {
    transitionChunkState(coord, ChunkState::LOADING, ChunkState::OPTIMIZING);
    std::unique_lock<std::mutex> lock(metadata_mutex_);
    optimizing_chunks_.insert(coord);
  } else {
    transitionChunkState(coord, ChunkState::LOADING, ChunkState::ACTIVE);
  }

  // Notify any waiting threads
  {
    std::unique_lock<std::mutex> lock(metadata_mutex_);
    auto it = chunk_metadata_.find(coord);
    if (it != chunk_metadata_.end()) {
      std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
      it->second.operation_cv.notify_all();
    }
  }

  incrementStat(stats_.active_chunks);
  incrementStat(stats_.disk_loads);

  // std::cout << "IO Thread: Load operation successful for: " << coord.x << " "
  //           << coord.y << " " << coord.z << " " << std::endl;
  return true;
}

// Process a save operation
bool ChunkManager::processSaveOperation(const ChunkCoord& coord) {
  // std::cout << "Called processSaveOperation" << std::endl;
  try {
    std::shared_ptr<Chunk> chunk;

    // Copy chunk pointer to work with (minimize lock time)
    {
      std::unique_lock<std::mutex> lock(active_chunks_mutex_);
      auto it = active_chunks_.find(coord);
      if (it == active_chunks_.end() || !it->second) {
        std::cout << "Warning: Tried to save null chunk!" << std::endl;
        transitionChunkState(coord, ChunkState::SAVING, ChunkState::INACTIVE);
        return false;
      }
      chunk = it->second;
    }

    if (!chunk->getGaussians()) {
      std::cerr << "Null gaussians in save operation" << std::endl;
      transitionChunkState(coord, ChunkState::SAVING, ChunkState::INACTIVE);
      return false;
    }

    auto chunk_filename = getChunkFilename(coord);

    // Save to file
    chunk->getGaussians()->save_checkpoint(chunk_filename.string());

    // Update cache
    {
      std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
      chunk_exists_cache_[coord] = true;
    }

    // Remove from active chunks
    {
      std::unique_lock<std::mutex> lock(active_chunks_mutex_);
      active_chunks_.erase(coord);
      decrementStat(stats_.active_chunks);
    }

    // Increment save counter
    incrementStat(stats_.disk_saves);

    // Transition to INACTIVE state
    transitionChunkState(coord, ChunkState::SAVING, ChunkState::INACTIVE);

    // Notify any waiting threads
    {
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      auto it = chunk_metadata_.find(coord);
      if (it != chunk_metadata_.end()) {
        std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
        it->second.operation_cv.notify_all();
      }
    }

    // Clear CUDA cache after saving to free memory
    c10::cuda::CUDACachingAllocator::emptyCache();

    // std::cout << "IO Thread: Save operation successful for: " << coord.x << "
    // "
    //           << coord.y << " " << coord.z << " " << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception in save operation: " << e.what() << std::endl;
    transitionChunkState(coord, ChunkState::SAVING, ChunkState::INACTIVE);
    return false;
  }
}

// Synchronous wrapper for loadChunkAsync
bool ChunkManager::loadChunkSync(const ChunkCoord& coord,
                                 bool load_for_optimization,
                                 bool skip_busy_chunks) {
  // std::cout << "Called loadChunkSync" << std::endl;
  auto future = loadChunkAsync(coord, 10, load_for_optimization,
                               skip_busy_chunks);  // High priority
  try {
    return future.get();  // Wait for completion
  } catch (const std::exception& e) {
    std::cerr << "Sync load failed: " << e.what() << std::endl;
    return false;
  }
}

// Synchronous wrapper for saveChunkAsync
bool ChunkManager::saveChunkSync(const ChunkCoord& coord) {
  // std::cout << "Called saveChunkSync" << std::endl;
  auto future = saveChunkAsync(coord, 10);  // High priority
  try {
    return future.get();  // Wait for completion
  } catch (const std::exception& e) {
    std::cerr << "Sync save failed: " << e.what() << std::endl;
    return false;
  }
}

// Process operation dispatcher
void ChunkManager::processOperation(std::shared_ptr<ChunkOperation> operation) {
  // std::cout << "Called processOperation" << std::endl;
  bool success = false;

  try {
    switch (operation->type) {
      case ChunkOperation::LOAD:
        success = processLoadOperation(operation->coord,
                                       operation->load_for_optimization);
        break;

      case ChunkOperation::SAVE:
        success = processSaveOperation(operation->coord);
        break;

      case ChunkOperation::DELETE:
        success = processDeleteOperation(operation->coord);
        break;
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception in chunk operation: " << e.what() << std::endl;
  }

  try {
    operation->completion_promise.set_value(success);
  } catch (const std::exception& e) {
    std::cerr << "Error setting promise value: " << e.what() << std::endl;
  }
}

// Create a future that resolves when a chunk reaches a specific state
std::future<bool> ChunkManager::createWaitFuture(const ChunkCoord& coord,
                                                 ChunkState target_state) {
  // std::cout << "Called createWaitFuture" << std::endl;
  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  // Launch a wait task on a separate thread
  std::thread([this, coord, target_state, promise]() {
    const int MAX_RETRIES = 100;
    const auto RETRY_INTERVAL = std::chrono::milliseconds(50);

    bool success = false;
    int attempts = 0;

    while (attempts < MAX_RETRIES) {
      ChunkState current = getChunkState(coord);

      if (current == target_state) {
        success = true;
        break;
      }

      // Wait with timeout on the condition variable
      {
        std::unique_lock<std::mutex> lock(metadata_mutex_);
        auto it = chunk_metadata_.find(coord);
        if (it != chunk_metadata_.end()) {
          std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
          it->second.operation_cv.wait_for(op_lock, RETRY_INTERVAL);
        } else {
          // If no metadata, just sleep
          std::this_thread::sleep_for(RETRY_INTERVAL);
        }
      }

      attempts++;
    }

    promise->set_value(success);
  }).detach();

  return future;
}

// Asynchronous delete with priority
std::future<bool> ChunkManager::deleteChunkAsync(const ChunkCoord& coord,
                                                 int priority) {
  // std::cout << "Called deleteChunkAsync" << std::endl;
  ChunkState current_state = getChunkState(coord);

  // If already deleting, return a future that waits for completion
  if (current_state == ChunkState::DELETING) {
    return createWaitFuture(coord, ChunkState::INACTIVE);
  }

  // Can't delete if it's being saved or loaded or optimized
  if (current_state == ChunkState::SAVING ||
      current_state == ChunkState::LOADING ||
      current_state == ChunkState::OPTIMIZING) {
    std::promise<bool> promise;
    promise.set_value(false);
    return promise.get_future();
  }

  if (transitionChunkState(coord, ChunkState::INACTIVE, ChunkState::DELETING) ||
      transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::DELETING)) {
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::DELETE;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();

    std::future<bool> future = operation->completion_promise.get_future();
    enqueueOperation(operation);

    return future;
  } else {
    // Failed to transition state
    std::promise<bool> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Failed to queue delete operation")));
    return promise.get_future();
  }
}

// Process a delete operation
bool ChunkManager::processDeleteOperation(const ChunkCoord& coord) {
  // Delete from active_chunks_ if exists
  {
    std::unique_lock<std::mutex> lock(active_chunks_mutex_);
    auto it = active_chunks_.find(coord);
    if (it != active_chunks_.end()) {
      active_chunks_.erase(it);
      decrementStat(stats_.active_chunks);
    }
  }

  // Update the disk cache
  {
    std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
    chunk_exists_cache_[coord] = false;
  }

  // NEW: Invalidate visibility cache entries containing this chunk
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // Loop through visibility cache and remove entries containing this chunk
    for (auto it = visibility_cache_.begin(); it != visibility_cache_.end();
         it++) {
      auto& visible_chunks = it->second.visible_chunks;

      // Check if this chunk is in the list of visible chunks
      auto chunk_it =
          std::find(visible_chunks.begin(), visible_chunks.end(), coord);
      if (chunk_it != visible_chunks.end()) {
        // Remove the chunk from the visible_chunks list
        visible_chunks.erase(chunk_it);
      }
    }
  }

  // Delete from disk if exists
  try {
    auto chunk_filename = getChunkFilename(coord);

    // Delete the file if it exists
    bool success = false;
    if (std::filesystem::exists(chunk_filename)) {
      if (std::filesystem::is_directory(chunk_filename)) {
        success = std::filesystem::remove_all(chunk_filename) > 0;
      } else {
        success = std::filesystem::remove(chunk_filename);
      }
    } else {
      // If file doesn't exist, consider it a success
      success = true;
    }

    // First transition to INACTIVE and notify waiters
    {
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      auto it = chunk_metadata_.find(coord);
      if (it != chunk_metadata_.end()) {
        // Set state to INACTIVE before notifying
        it->second.state.store(ChunkState::INACTIVE);

        // Notify any waiting threads
        {
          std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
          it->second.operation_cv.notify_all();
        }

        // After notification, erase the metadata
        chunk_metadata_.erase(coord);
      }
    }

    return success;
  } catch (const std::exception& e) {
    std::cerr << "Exception in delete operation: " << e.what() << std::endl;

    // Handle error case
    {
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      auto it = chunk_metadata_.find(coord);
      if (it != chunk_metadata_.end()) {
        it->second.state.store(ChunkState::INACTIVE);
        std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
        it->second.operation_cv.notify_all();
      }
    }

    return false;
  }
}

// Synchronous wrapper for deleteChunkAsync
bool ChunkManager::deleteChunkSync(const ChunkCoord& coord) {
  // std::cout << "Called deleteChunkSync" << std::endl;
  auto future = deleteChunkAsync(coord, 10);  // High priority
  try {
    return future.get();  // Wait for completion
  } catch (const std::exception& e) {
    std::cerr << "Sync delete failed: " << e.what() << std::endl;
    return false;
  }
}

std::tuple<torch::Tensor, torch::Tensor> ChunkManager::filterPointsByDepth(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  // std::cout << "Filtering points, starting with " << points.size(0)
  //           << std::endl;
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
  // std::cout << "After filter " << filtered_points.size(0) << std::endl;

  return std::make_tuple(filtered_points, filtered_colors);
}

std::vector<ChunkCoord> ChunkManager::findChunksToEvict(int count) {
  // std::cout << "Called findChunksToEvict" << std::endl;
  int active_chunk_count = 0;
  {
    std::unique_lock<std::mutex> chunks_lock(active_chunks_mutex_);
    // std::cout << "Locked active_chunks_mutex_" << std::endl;
    active_chunk_count = active_chunks_.size();

    // If we have space, don't evict
    if (active_chunk_count <= max_chunks_in_memory_ - count) {
      return {};
    }
  }

  // Calculate how many to evict - no locks needed
  int to_evict = active_chunk_count - (max_chunks_in_memory_ - count);

  // Gather candidates with proper locking
  std::vector<
      std::pair<ChunkCoord, std::chrono::time_point<std::chrono::steady_clock>>>
      candidates;
  auto now = std::chrono::steady_clock::now();

  {
    // Lock both mutexes in the correct order when we need both
    std::unique_lock<std::mutex> meta_lock(metadata_mutex_);
    // std::cout << "Locked metadata_mutex_" << std::endl;
    std::unique_lock<std::mutex> chunks_lock(active_chunks_mutex_);
    // std::cout << "Locked active_chunks_mutex_" << std::endl;

    for (const auto& [coord, chunk] : active_chunks_) {
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end()) {
        // Only consider ACTIVE chunks (not ones being loaded/saved)
        if (meta_it->second.state.load() == ChunkState::ACTIVE) {
          // Skip recently loaded chunks
          auto time_since_load =
              std::chrono::duration_cast<std::chrono::seconds>(
                  now - meta_it->second.load_time);

          if (time_since_load > min_retention_time_) {
            candidates.push_back({coord, meta_it->second.last_used});
          }
        }
      }
    }
  }

  // Sort by last used time - no locks needed
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Take up to to_evict chunks - no locks needed
  std::vector<ChunkCoord> result;
  for (size_t i = 0;
       i < std::min(static_cast<size_t>(to_evict), candidates.size()); ++i) {
    result.push_back(candidates[i].first);
  }

  return result;
}

// Get chunk filename
std::filesystem::path ChunkManager::getChunkFilename(const ChunkCoord& coord) {
  // Using 'p' for positive and 'n' for negative prefixes
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));

  return chunk_save_dir_ / (x_str + "_" + y_str + "_" + z_str);
}

// Get chunk at specific coordinate
std::shared_ptr<Chunk> ChunkManager::getChunkAt(const ChunkCoord& coord) {
  // std::cout << "Called getChunkAt" << std::endl;
  std::unique_lock<std::mutex> lock(active_chunks_mutex_);
  auto it = active_chunks_.find(coord);
  if (it != active_chunks_.end()) {
    return it->second;
  }
  return nullptr;
}

// Private version that assumes lock is already held
bool ChunkManager::chunkExists(const ChunkCoord& coord) {
  // std::cout << "Called chunkExists" << std::endl;
  // Check cache first (thread-safe read)
  {
    std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
    auto it = chunk_exists_cache_.find(coord);
    if (it != chunk_exists_cache_.end()) {
      return it->second;
    }
  }

  // Check filesystem (no locks needed)
  auto chunk_filename = getChunkFilename(coord);
  bool exists = std::filesystem::exists(chunk_filename);

  // Update cache (thread-safe write)
  {
    std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
    chunk_exists_cache_[coord] = exists;
  }

  return exists;
}

// Main function that returns visible chunks, handling both active and on-disk
// chunks
std::vector<std::shared_ptr<Chunk>> ChunkManager::loadVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  // std::cout << "Called loadVisibleChunks" << std::endl;
  auto timer = ProfilingUtils::Timer("ChunkManager::loadVisibleChunks");

  if (!keyframe) {
    std::cerr << "Error: Null keyframe passed to loadVisibleChunks"
              << std::endl;
    return {};
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  std::vector<ChunkCoord> visible_chunk_coords;

  bool cache_exists = false;
  // Check if we can use cached visibility results
  if (use_cache) {
    cache_exists = false;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto now = std::chrono::steady_clock::now();

    // Check cache entry exists and is valid
    auto cache_it = visibility_cache_.find(keyframe_id);
    if (cache_it != visibility_cache_.end()) {
      auto& entry = cache_it->second;

      // Check if cache entry is recent enough and pose hasn't changed
      // significantly
      if ((now - entry.timestamp) < cache_expiry_time_ &&
          pose_nearly_equal(current_pose, entry.pose)) {
        // std::cout << "Using chunk visibility cache!" << std::endl;
        visible_chunk_coords = entry.visible_chunks;
        cache_exists = true;

        // Update timestamp to keep this entry fresh
        entry.timestamp = now;
      }
    }

    // Clean up old cache entries periodically
    if (visibility_cache_.size() > max_cache_entries_) {
      // Find and remove oldest entries
      std::vector<std::size_t> to_remove;
      for (const auto& [id, entry] : visibility_cache_) {
        if ((now - entry.timestamp) > cache_expiry_time_) {
          to_remove.push_back(id);
        }
      }

      for (auto id : to_remove) {
        visibility_cache_.erase(id);
      }
    }
  }

  // If we can't use the cache, perform frustum culling
  if (!use_cache || !cache_exists) {
    visible_chunk_coords = frustumCullChunks(keyframe);

    // Update the cache
    if (use_cache && !cache_exists) {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      VisibilityCacheEntry entry;
      entry.pose = current_pose;
      entry.visible_chunks = visible_chunk_coords;
      entry.timestamp = std::chrono::steady_clock::now();
      visibility_cache_[keyframe_id] = entry;
    }
  }

  // Now we have the list of visible chunk coordinates
  // Start asynchronous loading of chunks
  std::vector<std::shared_ptr<Chunk>> result_chunks;
  std::vector<ChunkCoord> chunks_to_load;
  std::vector<std::future<bool>> load_futures;
  std::vector<ChunkCoord> load_coords;

  for (const auto& coord : visible_chunk_coords) {
    if (!chunkExists(coord)) continue;

    ChunkState state = getChunkState(coord);

    if (state == ChunkState::ACTIVE) {
      // Directly transition to OPTIMIZING
      if (transitionChunkState(coord, ChunkState::ACTIVE,
                               ChunkState::OPTIMIZING)) {
        {
          std::unique_lock<std::mutex> lock(metadata_mutex_);
          optimizing_chunks_.insert(coord);
        }

        auto chunk = getChunkAt(coord);
        if (chunk) {
          result_chunks.push_back(chunk);
        }
      }
    } else if (state == ChunkState::OPTIMIZING) {
      // Already in the right state
      auto chunk = getChunkAt(coord);
      if (chunk) {
        result_chunks.push_back(chunk);
      }

    } else {
      // Need to load from disk or in an incompatible state
      load_futures.push_back(loadChunkAsync(coord, 10, true, true));
      load_coords.push_back(coord);
    }
  }

  // Wait for critical chunks to load (with timeout)
  const auto timeout = std::chrono::milliseconds(100);
  for (size_t i = 0; i < load_futures.size(); ++i) {
    if (load_futures[i].wait_for(timeout) == std::future_status::ready) {
      if (load_futures[i].get()) {
        ChunkCoord coord = load_coords[i];  // Use the correct array

        // Also add proper locking and validation
        std::shared_ptr<Chunk> chunk;
        {
          std::unique_lock<std::mutex> lock(active_chunks_mutex_);
          auto it = active_chunks_.find(coord);
          if (it != active_chunks_.end()) {
            chunk = it->second;
          }
        }

        if (chunk && chunk->getGaussians()) {
          result_chunks.push_back(chunk);
        }
      }
    }
  }

  return result_chunks;
}

void ChunkManager::preloadVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  // std::cout << "Called loadVisibleChunks" << std::endl;
  auto timer = ProfilingUtils::Timer("ChunkManager::loadVisibleChunks");

  if (!keyframe) {
    std::cerr << "Error: Null keyframe passed to loadVisibleChunks"
              << std::endl;
    return;
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  std::vector<ChunkCoord> visible_chunk_coords;

  bool cache_exists = false;
  // Check if we can use cached visibility results
  if (use_cache) {
    cache_exists = false;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto now = std::chrono::steady_clock::now();

    // Check cache entry exists and is valid
    auto cache_it = visibility_cache_.find(keyframe_id);
    if (cache_it != visibility_cache_.end()) {
      auto& entry = cache_it->second;

      // Check if cache entry is recent enough and pose hasn't changed
      // significantly
      if ((now - entry.timestamp) < cache_expiry_time_ &&
          pose_nearly_equal(current_pose, entry.pose)) {
        // std::cout << "Using chunk visibility cache!" << std::endl;
        visible_chunk_coords = entry.visible_chunks;
        cache_exists = true;

        // Update timestamp to keep this entry fresh
        entry.timestamp = now;
      }
    }

    // Clean up old cache entries periodically
    if (visibility_cache_.size() > max_cache_entries_) {
      // Find and remove oldest entries
      std::vector<std::size_t> to_remove;
      for (const auto& [id, entry] : visibility_cache_) {
        if ((now - entry.timestamp) > cache_expiry_time_) {
          to_remove.push_back(id);
        }
      }

      for (auto id : to_remove) {
        visibility_cache_.erase(id);
      }
    }
  }

  // If we can't use the cache, perform frustum culling
  if (!use_cache || !cache_exists) {
    visible_chunk_coords = frustumCullChunks(keyframe);

    // Update the cache
    if (use_cache && !cache_exists) {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      VisibilityCacheEntry entry;
      entry.pose = current_pose;
      entry.visible_chunks = visible_chunk_coords;
      entry.timestamp = std::chrono::steady_clock::now();
      visibility_cache_[keyframe_id] = entry;
    }
  }

  for (const auto& coord : visible_chunk_coords) {
    // Check if chunk has been seen before
    if (!chunkExists(coord)) continue;

    ChunkState state = getChunkState(coord);

    if (state == ChunkState::ACTIVE) {
      // Make sure it is not culled by LRU anytime soon

    } else if (state == ChunkState::OPTIMIZING) {
      // Allowed to be in this state

    } else {
      std::cout << "Preloading (chunk state: " << static_cast<int>(state)
                << "): " << coord.x << " " << coord.y << " " << coord.z
                << std::endl;
      // Need to load from disk or in an incompatible state
      loadChunkAsync(coord, 3, false, true);
    }
  }
}

// Evict least recently used chunks asynchronously
void ChunkManager::evictUnusedChunks(int keep_count) {
  std::cout << "Called evictUnusedChunks" << std::endl;
  auto timer = ProfilingUtils::Timer("ChunkManager::evictUnusedChunks");

  if (keep_count < 0) {
    // Default: keep enough space for some new chunks
    keep_count = std::max(5, max_chunks_in_memory_ / 10);
  }

  std::vector<ChunkCoord> to_evict;
  {
    to_evict = findChunksToEvict(keep_count);
  }

  // Schedule async saves for evicted chunks
  for (const auto& coord : to_evict) {
    // Use medium priority (5) for eviction operations
    saveChunkAsync(coord, 5);
  }
}

// Helper function to find visible chunks within search radius
std::vector<ChunkCoord> ChunkManager::frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  // std::cout << "Called frustumCullChunks" << std::endl;

  std::vector<ChunkCoord> visible_coords;

  if (!keyframe) {
    return visible_coords;
  }

  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);
  Eigen::Matrix4f proj_matrix = createProjectionMatrix(keyframe);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Get camera position for chunk search
  Sophus::SE3d current_pose = keyframe->getPose();
  Sophus::SE3d Twc = current_pose.inverse();  // World to camera transform
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position);

  // Determine search radius based on far plane distance
  int search_radius = std::min(std::ceil(keyframe->zfar_ / chunk_size_), 10.0f);

  for (int dx = -search_radius; dx <= search_radius; dx++) {
    for (int dy = -search_radius; dy <= search_radius; dy++) {
      for (int dz = -search_radius; dz <= search_radius; dz++) {
        ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                               camera_chunk.z + dz};

        // Skip chunks that are too far from camera (rough distance check)
        Eigen::Vector3f chunk_center = getChunkCenter(check_coord);
        float dist_to_camera = (chunk_center - camera_position).norm();
        if (dist_to_camera >
            keyframe->zfar_ + chunk_size_ * 1.732f) {  // sqrt(3) for diagonal
          continue;
        }

        // Get AABB for the chunk and test against frustum
        AABB chunk_aabb = getChunkAABB(check_coord);
        bool visible = test_AABB_against_frustum_eigen(vp_matrix, chunk_aabb);
        if (visible) {
          visible_coords.push_back(check_coord);
        }
      }
    }
  }
  return visible_coords;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ChunkManager::groupPointsByChunk(const torch::Tensor& positions) {
  // std::cout << "Called groupPointsByChunk" << std::endl;
  // Called by addPoints
  // Convert positions to chunk coordinates
  torch::Tensor chunk_coords = torch::floor(positions / chunk_size_);
  // std::cout << "New ungrouped points " << positions.size(0) << std::endl;

  // Convert to int64 for bit operations
  chunk_coords = chunk_coords.to(torch::kInt64);

  // std::cout << "Chunk coordinates range: "
  //           << torch::min(chunk_coords).item<int64_t>() << " to "
  //           << torch::max(chunk_coords).item<int64_t>() << std::endl;

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

  // std::cout << "Number of unique chunks: " << unique_chunk_ids.size(0)
  //           << std::endl;

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

  // std::cout << "Reconstructed coordinate range: "
  //           << torch::min(unique_coords).item<int64_t>() << " to "
  //           << torch::max(unique_coords).item<int64_t>() << std::endl;

  return std::make_tuple(unique_coords, inverse_indices, points_per_chunk);
}

// Add points to appropriate chunks
void ChunkManager::addPointsToChunks(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes,
    float cameras_extent) {
  int min_new_points_threshold = 10;
  // std::cout << "addPointsToChunks called with " << points.size(0) << "
  // points"
  //           << std::endl;

  // Filter points by depth first (no locks needed)
  auto filtered_points = points;
  auto filtered_colors = colors;

  // Skip if not enough points
  if (filtered_points.sizes()[0] < min_new_points_threshold) {
    // std::cout << "Too few points to add, skipping" << std::endl;
    return;
  }

  // Convert to CUDA for processing
  torch::Tensor points_cuda = filtered_points.to(torch::kCUDA);
  torch::Tensor colors_cuda = filtered_colors.to(torch::kCUDA);

  // Group points by chunk (no locks needed)
  auto [unique_chunks, inverse_indices, points_per_chunk] =
      groupPointsByChunk(points_cuda);

  // std::cout << "Adding points to " << unique_chunks.size(0) << " chunks"
  //           << std::endl;

  // Track which chunks we're modifying to ensure proper synchronization
  std::vector<ChunkCoord> modified_chunks;
  std::vector<std::future<bool>> pending_loads;
  std::unordered_map<ChunkCoord, std::future<bool>, ChunkCoordHash>
      load_futures;

  // First pass: start loading all chunks we'll need
  for (int64_t i = 0; i < unique_chunks.size(0); i++) {
    // Get chunk coordinate
    ChunkCoord coord{unique_chunks[i][0].item<int64_t>(),
                     unique_chunks[i][1].item<int64_t>(),
                     unique_chunks[i][2].item<int64_t>()};

    // Create mask for points in this chunk
    torch::Tensor chunk_mask = (inverse_indices == i);

    // Extract points for this chunk
    torch::Tensor chunk_points = points_cuda.index({chunk_mask}).clone();

    // Skip if not enough points
    if (chunk_points.sizes()[0] < min_new_points_threshold) {
      continue;
    }

    if (chunkExists(coord)) {
      ChunkState state = getChunkState(coord);
      if (state == ChunkState::INACTIVE) {
        auto future = loadChunkAsync(coord, 10, true);  // High priority load
        load_futures[coord] = std::move(future);
      } else if (state == ChunkState::ACTIVE) {
        transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::OPTIMIZING);
      }
    }
    modified_chunks.push_back(coord);
  }

  // Process each unique chunk with gathered points
  for (int64_t i = 0; i < unique_chunks.size(0); i++) {
    // Get chunk coordinate
    ChunkCoord coord{unique_chunks[i][0].item<int64_t>(),
                     unique_chunks[i][1].item<int64_t>(),
                     unique_chunks[i][2].item<int64_t>()};

    // Create mask for points in this chunk
    torch::Tensor chunk_mask = (inverse_indices == i);

    // Extract points and features for this chunk
    torch::Tensor chunk_points = points_cuda.index({chunk_mask}).clone();
    torch::Tensor chunk_colors = colors_cuda.index({chunk_mask}).clone();

    // std::cout << "Adding " << chunk_points.sizes()[0]
    //           << " points to chunk: " << coord.x << "," << coord.y << ","
    //           << coord.z << std::endl;

    // Skip if not enough points
    if (chunk_points.sizes()[0] < min_new_points_threshold) {
      // std::cout << "Too few points for this chunk, skipping" << std::endl;
      continue;
    }

    // Wait if we have a pending load for this chunk
    auto future_it = load_futures.find(coord);
    bool loaded_from_disk = false;

    if (future_it != load_futures.end()) {
      try {
        // Wait with timeout
        auto status = future_it->second.wait_for(std::chrono::seconds(5));
        if (status == std::future_status::ready) {
          loaded_from_disk = future_it->second.get();
        } else {
          std::cerr << "Timeout waiting for chunk load, skipping: " << coord.x
                    << "," << coord.y << "," << coord.z << std::endl;
          continue;
        }
      } catch (const std::exception& e) {
        std::cerr << "Error waiting for chunk load: " << e.what() << std::endl;
        continue;
      }
    }

    // Get or create the chunk
    std::shared_ptr<Chunk> chunk;
    bool is_new_chunk = false;

    if (chunkExists(coord)) {
      // Check current state again
      ChunkState state = getChunkState(coord);
      if (state == ChunkState::OPTIMIZING) {
        // Chunk is active, get it
        chunk = getChunkAt(coord);
        if (!chunk || !chunk->getGaussians()) {
          std::cerr << "Null chunk or gaussians after active check"
                    << std::endl;
          continue;
        }

        // Add points to existing chunk
        chunk->getGaussians()->increasePcd(chunk_points, chunk_colors,
                                           getCurrentIteration());

      } else {
        std::cerr << "Chunk in transitional state, cannot add points: "
                  << static_cast<int>(state) << std::endl;
        continue;
      }
    } else {
      // Create new chunk
      // std::cout << "Creating new chunk" << std::endl;
      chunk = std::make_shared<Chunk>(model_params_, coord);

      // Add to active chunks map
      {
        std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
        active_chunks_[coord] = chunk;
      }

      // Update metadata
      {
        std::unique_lock<std::mutex> lock(metadata_mutex_);
        auto& meta = chunk_metadata_[coord];
        meta.load_time = std::chrono::steady_clock::now();
        meta.last_used = meta.load_time;
        meta.usage_count = 0;
        meta.state.store(ChunkState::OPTIMIZING);
      }

      // Update cache
      {
        std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
        chunk_exists_cache_[coord] = true;
      }

      is_new_chunk = true;
      incrementStat(stats_.active_chunks);

      // Initialize the Gaussian model with these points
      chunk->getGaussians()->createFromPcd(chunk_points, chunk_colors,
                                           cameras_extent);

      // Set up training
      chunk->getGaussians()->trainingSetup(opt_params_);
    }

    releaseChunksFromOptimization({coord});
  }
}

// Updated shutdown to properly clean up threads
void ChunkManager::shutdown() {
  std::cout << "ChunkManager shutting down..." << std::endl;
  should_terminate_ = true;

  // Save all active chunks first
  std::vector<std::future<bool>> pending_saves;

  {
    std::unique_lock<std::mutex> lock(active_chunks_mutex_);
    for (const auto& [coord, chunk] : active_chunks_) {
      // Queue high-priority save
      pending_saves.push_back(saveChunkAsync(coord, 20));
    }
  }

  // Wait for all saves to complete (with timeout)
  const auto timeout = std::chrono::seconds(30);
  for (auto& future : pending_saves) {
    future.wait_for(timeout);
  }

  // Now shutdown the thread pool
  shutdownThreadPool();

  std::cout << "ChunkManager shutdown complete" << std::endl;
}

// Emergency shutdown without saving anything
void ChunkManager::shutdownWithoutSaving() {
  std::cout << "ChunkManager emergency shutdown..." << std::endl;
  should_terminate_ = true;

  // Just shutdown the thread pool
  shutdownThreadPool();

  // Clear active chunks directly
  std::unique_lock<std::mutex> lock(active_chunks_mutex_);
  active_chunks_.clear();

  std::cout << "ChunkManager emergency shutdown complete" << std::endl;
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
  // std::cout << "Called cullSparseChunks" << std::endl;
  std::vector<ChunkCoord> chunks_to_cull;
  bool any_culled = false;

  // First pass: identify chunks to cull (with minimal locking time)
  {
    std::unique_lock<std::mutex> lock(active_chunks_mutex_);

    // Examine all active chunks
    for (const auto& [coord, chunk] : active_chunks_) {
      // Skip chunks that are in transitional states
      ChunkState state;
      {
        std::unique_lock<std::mutex> meta_lock(metadata_mutex_);
        auto meta_it = chunk_metadata_.find(coord);
        if (meta_it != chunk_metadata_.end()) {
          state = meta_it->second.state.load();
        } else {
          continue;
        }
      }

      // Only consider ACTIVE chunks for culling
      if (state != ChunkState::ACTIVE) {
        continue;
      }

      if (!chunk || !chunk->getGaussians()) {
        // Mark invalid chunks for removal
        chunks_to_cull.push_back(coord);
        continue;
      }

      // Get number of active points in chunk
      int num_points = chunk->getGaussians()->getXYZ().size(0);

      // If below threshold, mark for culling
      if (num_points < min_points_threshold) {
        chunks_to_cull.push_back(coord);
      }
    }
  }

  for (const auto& coord : chunks_to_cull) {
    deleteChunkAsync(coord, 3);
    any_culled = true;
  }

  return any_culled;
}

// void ChunkManager::cullGaussiansOutsideChunkBorders() {
//   for (const auto& [coord, chunk] : active_chunks_) {
//     if (!chunk || !chunk->getGaussians()) {
//       continue;  // No chunk or no gaussians
//     }

//     auto meta_it = chunk_metadata_.find(coord);

//     // Get the AABB for the chunk
//     AABB aabb = getChunkAABB(coord);

//     // Get the gaussians from the chunk
//     auto gaussians = chunk->getGaussians();
//     auto points = gaussians->getXYZ();

//     int num_points = points.size(0);
//     if (num_points == 0) {
//       continue;  // No points to cull
//     }

//     // Create a mask for points that are outside the AABB
//     torch::Tensor outside_mask =
//         ((points.index({torch::indexing::Slice(), 0}) < aabb.min.x()) |
//          (points.index({torch::indexing::Slice(), 0}) > aabb.max.x()) |
//          (points.index({torch::indexing::Slice(), 1}) < aabb.min.y()) |
//          (points.index({torch::indexing::Slice(), 1}) > aabb.max.y()) |
//          (points.index({torch::indexing::Slice(), 2}) < aabb.min.z()) |
//          (points.index({torch::indexing::Slice(), 2}) > aabb.max.z()));

//     int num_outside = outside_mask.sum().item<int>();

//     // If no points are outside, no culling needed
//     if (num_outside == 0) {
//       continue;
//     }

//     // If all points are outside, remove the chunk entirely
//     if (num_outside == num_points) {
//       // Remove from active chunks
//       active_chunks_.erase(coord);
//       decrementStat(stats_.active_chunks);

//       // Update disk cache to prevent reloading
//       chunk_exists_cache_[coord] = false;

//       deleteChunk(coord);

//     } else {
//       // Otherwise, prune the outside points
//       try {
//         gaussians->prunePoints(outside_mask);

//         // Mark the chunk as dirty
//         if (meta_it != chunk_metadata_.end()) {
//           meta_it->second.dirty = true;
//         }
//       } catch (const std::exception& e) {
//         std::cerr << "Error pruning gaussians for chunk " << coord.x <<
//         ","
//                   << coord.y << "," << coord.z << ": " << e.what() <<
//                   std::endl;
//         continue;
//       }
//     }
//   }
// }

void ChunkManager::updateChunkExistenceCache(
    const std::vector<ChunkCoord>& coords,
    bool exists) {
  // std::cout << "Called updateChunkExistenceCache" << std::endl;
  std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
  for (const auto& coord : coords) {
    chunk_exists_cache_[coord] = exists;
  }
}

std::vector<ChunkCoord> ChunkManager::getExistingChunkCoords() {
  // std::cout << "Called getExistingChunkCoords" << std::endl;
  std::vector<ChunkCoord> result;
  std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
  // Iterate through the cache and collect all chunks that exist
  for (const auto& [coord, exists] : chunk_exists_cache_) {
    if (exists) {
      result.push_back(coord);
    }
  }

  return result;
}

void ChunkManager::transferGaussiansAcrossChunks(float spatial_lr_scale) {
  std::cout << "Called transferGaussiansAcrossChunks" << std::endl;
  torch::NoGradGuard no_grad;

  // Get all existing chunk coordinates
  std::vector<ChunkCoord> all_chunks = getExistingChunkCoords();

  std::cout << "Transferring Gaussians across " << all_chunks.size()
            << std::endl;

  // Process chunks in batches to manage memory
  const int batch_size = 5;  // Adjust based on memory constraints

  // First pass: Load chunks, identify and extract Gaussians that need
  // transfer
  std::vector<
      std::tuple<ChunkCoord, torch::Tensor, torch::Tensor, torch::Tensor,
                 torch::Tensor, torch::Tensor, torch::Tensor,
                 torch::Tensor>>
      transfers;  // Source chunk, points, features_dc, features_rest,
                  // opacity, scaling, rotation, exist_since

  // Track which chunks need to be loaded
  std::unordered_map<ChunkCoord, std::future<bool>, ChunkCoordHash>
      load_futures;

  // Process all chunks to find points that need transfer
  for (size_t i = 0; i < all_chunks.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_chunks.size());

    // Process current batch
    for (size_t j = i; j < end; j++) {
      const ChunkCoord& coord = all_chunks[j];
      // Get chunk (either already active or freshly loaded)
      std::shared_ptr<Chunk> chunk;

      if (!loadChunkSync(coord, true, false)) {
        std::cout << "Skipping chunk, can't load" << std::endl;
        continue;
      }

      if (!chunk || !chunk->getGaussians()) {
        continue;
      }

      auto gaussians = chunk->getGaussians();
      auto points = gaussians->getXYZ();

      // Skip if no points
      if (points.size(0) == 0) {
        releaseChunksFromOptimization({coord});
        saveChunkAsync(coord);  // Schedule async save
        continue;
      }

      // Get AABB for this chunk
      AABB chunk_aabb = getChunkAABB(coord);

      // Create mask for points outside this chunk
      torch::Tensor outside_mask =
          ((points.index({torch::indexing::Slice(), 0}) < chunk_aabb.min.x()) |
           (points.index({torch::indexing::Slice(), 0}) > chunk_aabb.max.x()) |
           (points.index({torch::indexing::Slice(), 1}) < chunk_aabb.min.y()) |
           (points.index({torch::indexing::Slice(), 1}) > chunk_aabb.max.y()) |
           (points.index({torch::indexing::Slice(), 2}) < chunk_aabb.min.z()) |
           (points.index({torch::indexing::Slice(), 2}) > chunk_aabb.max.z()));

      // If no points outside, skip
      int num_outside = outside_mask.sum().item<int>();
      if (num_outside == 0) {
        releaseChunksFromOptimization({coord});
        saveChunkAsync(coord);  // Schedule async save
        continue;
      }

      // Extract properties of outside points with explicit cloning
      torch::Tensor outside_points =
          points.index({outside_mask}).detach().clone();
      torch::Tensor outside_features_dc =
          gaussians->features_dc_.index({outside_mask}).detach().clone();
      torch::Tensor outside_features_rest =
          gaussians->features_rest_.index({outside_mask}).detach().clone();
      torch::Tensor outside_opacities =
          gaussians->opacity_.index({outside_mask}).detach().clone();
      torch::Tensor outside_scaling =
          gaussians->scaling_.index({outside_mask}).detach().clone();
      torch::Tensor outside_rotation =
          gaussians->rotation_.index({outside_mask}).detach().clone();
      torch::Tensor outside_exist_since =
          gaussians->exist_since_iter_.index({outside_mask}).detach().clone();

      // Remove migrated points from the source chunk
      gaussians->prunePoints(outside_mask);

      // Save properties and target chunk info for second pass
      auto [unique_dest_chunks, inverse_indices, points_per_chunk] =
          groupPointsByChunk(outside_points);

      // For each destination chunk, prepare the transfer data
      for (int k = 0; k < unique_dest_chunks.size(0); k++) {
        ChunkCoord dest_coord{unique_dest_chunks[k][0].item<int64_t>(),
                              unique_dest_chunks[k][1].item<int64_t>(),
                              unique_dest_chunks[k][2].item<int64_t>()};

        // Skip if destination is the same as source (shouldn't happen)
        if (dest_coord == coord) {
          continue;
        }

        // Mask for points going to this chunk
        torch::Tensor chunk_mask = (inverse_indices == k);

        // Minimum number of points to bother transferring
        const int MIN_TRANSFER_THRESHOLD = 30;
        if (chunk_mask.sum().item<int>() < MIN_TRANSFER_THRESHOLD) {
          continue;
        }

        // Extract properties of points going to this chunk with explicit
        // cloning
        torch::Tensor chunk_points = outside_points.index({chunk_mask}).clone();
        torch::Tensor chunk_features_dc =
            outside_features_dc.index({chunk_mask}).clone();
        torch::Tensor chunk_features_rest =
            outside_features_rest.index({chunk_mask}).clone();
        torch::Tensor chunk_opacities =
            outside_opacities.index({chunk_mask}).clone();
        torch::Tensor chunk_scaling =
            outside_scaling.index({chunk_mask}).clone();
        torch::Tensor chunk_rotation =
            outside_rotation.index({chunk_mask}).clone();
        torch::Tensor chunk_exist_since =
            outside_exist_since.index({chunk_mask}).clone();

        // Add to transfers list
        transfers.push_back(std::make_tuple(
            dest_coord, chunk_points, chunk_features_dc, chunk_features_rest,
            chunk_opacities, chunk_scaling, chunk_rotation, chunk_exist_since));
      }

      // Save and unload
      releaseChunksFromOptimization({coord});
      saveChunkAsync(coord);
    }
  }

  // Second pass: Apply the transfers to destination chunks
  std::unordered_set<ChunkCoord, ChunkCoordHash> dest_chunks;
  std::unordered_map<ChunkCoord, std::future<bool>, ChunkCoordHash>
      dest_load_futures;

  for (const auto& transfer : transfers) {
    ChunkCoord dest_coord = std::get<0>(transfer);

    // Get all transfers for this destination
    std::vector<
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor>>
        chunk_transfers;

    for (const auto& t : transfers) {
      ChunkCoord tc = std::get<0>(t);
      if (tc == dest_coord) {
        chunk_transfers.push_back(std::make_tuple(
            std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
            std::get<5>(t), std::get<6>(t), std::get<7>(t)));
      }
    }

    // Skip if no transfers to this chunk (shouldn't happen)
    if (chunk_transfers.empty()) {
      continue;
    }

    // Combine all transfers for this destination
    torch::Tensor all_points = std::get<0>(chunk_transfers[0]);
    torch::Tensor all_features_dc = std::get<1>(chunk_transfers[0]);
    torch::Tensor all_features_rest = std::get<2>(chunk_transfers[0]);
    torch::Tensor all_opacities = std::get<3>(chunk_transfers[0]);
    torch::Tensor all_scaling = std::get<4>(chunk_transfers[0]);
    torch::Tensor all_rotation = std::get<5>(chunk_transfers[0]);
    torch::Tensor all_exist_since = std::get<6>(chunk_transfers[0]);

    for (size_t i = 1; i < chunk_transfers.size(); i++) {
      all_points = torch::cat({all_points, std::get<0>(chunk_transfers[i])}, 0);
      all_features_dc =
          torch::cat({all_features_dc, std::get<1>(chunk_transfers[i])}, 0);
      all_features_rest =
          torch::cat({all_features_rest, std::get<2>(chunk_transfers[i])}, 0);
      all_opacities =
          torch::cat({all_opacities, std::get<3>(chunk_transfers[i])}, 0);
      all_scaling =
          torch::cat({all_scaling, std::get<4>(chunk_transfers[i])}, 0);
      all_rotation =
          torch::cat({all_rotation, std::get<5>(chunk_transfers[i])}, 0);
      all_exist_since =
          torch::cat({all_exist_since, std::get<6>(chunk_transfers[i])}, 0);
    }

    // Get or initialize chunk
    std::shared_ptr<Chunk> dest_chunk;
    bool is_new_chunk = false;

    {
      std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
      if (chunk_exists_cache_.find(dest_coord) != chunk_exists_cache_.end()) {
        is_new_chunk = chunk_exists_cache_[dest_coord];
      }
    }

    if (!is_new_chunk) {
      loadChunkSync(dest_coord, true, false);
      dest_chunk = getChunkAt(dest_coord);
    } else {
      dest_chunk = std::make_shared<Chunk>(model_params_, dest_coord);

      // Add to active chunks
      {
        std::unique_lock<std::mutex> lock(active_chunks_mutex_);
        active_chunks_[dest_coord] = dest_chunk;
      }

      // Update metadata
      {
        std::unique_lock<std::mutex> lock(metadata_mutex_);
        auto& meta = chunk_metadata_[dest_coord];
        meta.load_time = std::chrono::steady_clock::now();
        meta.last_used = meta.load_time;
        meta.usage_count = 0;
        meta.state.store(ChunkState::OPTIMIZING);
      }

      // Update cache
      {
        std::unique_lock<std::mutex> lock(chunk_exists_cache_mutex_);
        chunk_exists_cache_[dest_coord] = true;
      }

      incrementStat(stats_.active_chunks);
    }

    if (!dest_chunk || !dest_chunk->getGaussians()) {
      throw std::runtime_error("Null destination chunk or gaussians");
    }

    // Apply the transfer data
    auto gaussians = dest_chunk->getGaussians();

    try {
      // Initialize or add points to the chunk
      if (is_new_chunk) {
        // Initialize directly with the existing gaussians
        dest_chunk->getGaussians()->initializeFromExistingGaussians(
            all_points, all_features_dc, all_features_rest, all_opacities,
            all_scaling, all_rotation, all_exist_since, spatial_lr_scale,
            opt_params_);
      } else {
        // For existing chunks, add new points
        dest_chunk->getGaussians()->densificationPostfix(
            all_points, all_features_dc, all_features_rest, all_opacities,
            all_scaling, all_rotation, all_exist_since);
      }
    } catch (const std::exception& e) {
      std::cerr << "Error applying transfer data: " << e.what() << std::endl;
      continue;
    }

    releaseChunksFromOptimization({dest_coord});
    saveChunkAsync(dest_coord);
  }
  // Clear CUDA cache after processing
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void ChunkManager::releaseChunksFromOptimization(
    const std::vector<ChunkCoord>& chunk_coords) {
  std::unique_lock<std::mutex> lock(metadata_mutex_);
  for (const auto& coord : chunk_coords) {
    auto meta_it = chunk_metadata_.find(coord);
    if (meta_it != chunk_metadata_.end() &&
        meta_it->second.state.load() == ChunkState::OPTIMIZING) {
      // Transition back to ACTIVE
      meta_it->second.state.store(ChunkState::ACTIVE);
      optimizing_chunks_.erase(coord);

      // Notify any waiting threads
      std::unique_lock<std::mutex> op_lock(meta_it->second.operation_mutex);
      meta_it->second.operation_cv.notify_all();
    }
  }
}

void ChunkManager::releaseChunksFromOptimization(
    const std::vector<std::shared_ptr<Chunk>>& chunks) {
  for (const auto& chunk : chunks) {
    if (chunk && chunk->getGaussians()) {
      const ChunkCoord& coord = chunk->getCoord();
      std::unique_lock<std::mutex> lock(metadata_mutex_);
      auto meta_it = chunk_metadata_.find(coord);
      if (meta_it != chunk_metadata_.end() &&
          meta_it->second.state.load() == ChunkState::OPTIMIZING) {
        // Transition back to ACTIVE
        meta_it->second.state.store(ChunkState::ACTIVE);
        optimizing_chunks_.erase(coord);

        // Notify any waiting threads
        std::unique_lock<std::mutex> op_lock(meta_it->second.operation_mutex);
        meta_it->second.operation_cv.notify_all();
      }
    }
  }
}

void ChunkManager::releaseAllChunksFromOptimization() {
  std::unique_lock<std::mutex> lock(metadata_mutex_);
  std::vector<ChunkCoord> optimizing_chunks;
  optimizing_chunks.reserve(optimizing_chunks_.size());
  for (auto it = optimizing_chunks_.begin(); it != optimizing_chunks_.end();) {
    optimizing_chunks.push_back(
        std::move(optimizing_chunks_.extract(it++).value()));
  }
  releaseChunksFromOptimization(optimizing_chunks);
}