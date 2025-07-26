#include "include/chunk_manager.h"

#include <torch/cuda.h>

#include <algorithm>
#include <iostream>

#include "include/profiling.h"

std::atomic<int> ChunkOptimizationGuard::guard_counter_{0};

// Get chunk coordinate from 3D position
ChunkCoord ChunkManager::getChunkCoord(const Eigen::Vector3f& position) {
  float half_chunk = chunk_size_ * 0.5f;
  return ChunkCoord{static_cast<int64_t>(
                        std::floor((position.x() + half_chunk) / chunk_size_)),
                    static_cast<int64_t>(
                        std::floor((position.y() + half_chunk) / chunk_size_)),
                    static_cast<int64_t>(
                        std::floor((position.z() + half_chunk) / chunk_size_))};
}

// Get chunk center
Eigen::Vector3f ChunkManager::getChunkCenter(const ChunkCoord& coord) {
  return Eigen::Vector3f(coord.x * chunk_size_, coord.y * chunk_size_,
                         coord.z * chunk_size_);
}

// Calculate AABB for a chunk
AABB ChunkManager::getChunkAABB(const ChunkCoord& coord) {
  float half_chunk = chunk_size_ * 0.5f;
  Eigen::Vector3f center(coord.x * chunk_size_, coord.y * chunk_size_,
                         coord.z * chunk_size_);
  Eigen::Vector3f min_corner = center - Eigen::Vector3f::Constant(half_chunk);
  Eigen::Vector3f max_corner = center + Eigen::Vector3f::Constant(half_chunk);
  return AABB(min_corner, max_corner);
}

// Constructor
ChunkManager::ChunkManager(const GaussianModelParams& model_params,
                           const GaussianOptimizationParams& opt_params,
                           std::filesystem::path chunk_save_dir,
                           float chunk_size,
                           float cameras_extent,
                           int max_chunks,
                           int num_io_threads,
                           size_t max_vram_budget_mb)
    : model_params_(model_params),
      opt_params_(opt_params),
      chunk_save_dir_(chunk_save_dir),
      chunk_size_(chunk_size),
      cameras_extent_(cameras_extent),
      max_chunks_in_memory_(max_chunks),
      should_terminate_(false),
      max_vram_budget_mb_(max_vram_budget_mb),
      estimated_chunk_vram_mb_(100.0f) {  // Initial estimate of 100MB per chunk
  // Create save directory if it doesn't exist
  if (!chunk_save_dir_.empty() && !std::filesystem::exists(chunk_save_dir_)) {
    std::filesystem::create_directories(chunk_save_dir_);
  }

  std::cout << "Creating ChunkManager with async I/O threads: "
            << num_io_threads << std::endl;
  // Initialize thread pool
  initializeThreadPool(num_io_threads);

  // Start the LRU eviction thread
  lru_eviction_thread_ =
      std::thread(&ChunkManager::lruEvictionThreadFunction, this);
  std::cout << "Started LRU eviction thread" << std::endl;
}

// Destructor
ChunkManager::~ChunkManager() {
  if (!is_shutting_down_.exchange(true)) {  // Atomic exchange for thread safety
    // Full shutdown logic here - same code that was in shutdown()
    should_terminate_ = true;
    releaseAllChunksFromOptimization();

    // Stop the LRU thread
    {
      std::unique_lock<std::mutex> lock(lru_mutex_);
      stop_lru_thread_ = true;
      lru_cv_.notify_all();
    }

    // Join the LRU thread if it's running
    if (lru_eviction_thread_.joinable()) {
      lru_eviction_thread_.join();
    }

    // Shutdown the thread pool
    shutdownThreadPool();
  }
}

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
      // std::cout << "Operation_queue_size: " << (operation_queue_.size())
      //           << std::endl;
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
  auto start_time = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto it = chunk_metadata_.find(coord);
  if (it != chunk_metadata_.end()) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time);
    // std::cout << "getChunkState completed in " << duration.count() << "ns"
    //           << std::endl;
    return it->second.state.load();
  }
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);
  // std::cout << "getChunkState failed in " << duration.count() << "ns"
  //           << std::endl;
  return ChunkState::UNKNOWN;  // Default state if not found
}

bool ChunkManager::transitionChunkState(const ChunkCoord& coord,
                                        ChunkState expected,
                                        ChunkState new_state) {
  // std::cout << "Called transitionChunkState" << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto& metadata = chunk_metadata_[coord];
  ChunkState current = metadata.state.load();

  if (current == new_state) {
    return true;  // Already in the desired state
  }

  if (current != expected) {
    std::ostringstream oss;
    oss << "Failed state transition for " << coord.x << "," << coord.y << ","
        << coord.z << ": expected=" << static_cast<int>(expected)
        << ", actual=" << static_cast<int>(current)
        << ", target=" << static_cast<int>(new_state);
    throw std::runtime_error(oss.str());
  }

  metadata.state.store(new_state);
  // std::cout << "Successful state transition for " << coord.x << "," <<
  // coord.y
  //           << "," << coord.z << ": " << static_cast<int>(expected) << " -> "
  //           << static_cast<int>(new_state) << std::endl;
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);
  // std::cout << "Successful state transition in " << duration.count() << "ns"
  //           << std::endl;
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
      std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
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
  ChunkState state = getChunkState(coord);

  if (state == ChunkState::ACTIVE) {
    if (load_for_optimization) {
      transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::OPTIMIZING);
    }
    updateLastUsedTime(coord);
    std::promise<bool> promise;
    promise.set_value(true);
    return promise.get_future();
  }

  if (state == ChunkState::OPTIMIZING) {
    updateLastUsedTime(coord);
    std::promise<bool> promise;
    promise.set_value(true);
    return promise.get_future();
  }

  if (state == ChunkState::DELETING) {
    std::promise<bool> promise;
    promise.set_value(false);
    return promise.get_future();
  }

  // If already loading, return a future that waits for that operation
  if (state == ChunkState::LOADING) {
    return createWaitFuture(coord, load_for_optimization
                                       ? ChunkState::OPTIMIZING
                                       : ChunkState::ACTIVE);
  }

  // Skip if chunk is busy and we're allowed to skip
  if (skip_busy_chunks && (state == ChunkState::SAVING)) {
    std::cout << "Skipping busy chunk in state " << static_cast<int>(state)
              << ": " << coord.x << "," << coord.y << "," << coord.z
              << std::endl;
    std::promise<bool> promise;
    promise.set_value(false);  // Return false to indicate chunk was skipped
    return promise.get_future();
  }

  // If saving, wait for it to complete before loading
  if (state == ChunkState::SAVING) {
    std::cout << "Chunk is being saved, waiting for completion before loading: "
              << coord.x << "," << coord.y << "," << coord.z << std::endl;

    // First wait for INACTIVE state (after save completes)
    auto inactive_future = createWaitFuture(coord, ChunkState::INACTIVE);

    // Create a shared state to continue after waiting
    auto shared_promise = std::make_shared<std::promise<bool>>();
    auto result_future = shared_promise->get_future();

    // Launch a continuation task
    std::thread([this, future = std::move(inactive_future), coord, priority,
                 load_for_optimization, skip_busy_chunks,
                 shared_promise]() mutable {
      try {
        // Wait for save to complete
        bool saved = future.get();
        if (!saved) {
          shared_promise->set_value(false);
          return;
        }

        // Now try to load
        auto load_future = loadChunkAsync(
            coord, priority, load_for_optimization, skip_busy_chunks);
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

  if (state == ChunkState::INACTIVE) {
    // Create new load operation if we can transition to LOADING state
    transitionChunkState(coord, ChunkState::INACTIVE, ChunkState::LOADING);
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::LOAD;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();
    operation->load_for_optimization = load_for_optimization;

    std::future<bool> operation_future =
        operation->completion_promise.get_future();
    enqueueOperation(operation);

    return operation_future;
  } else {
    std::ostringstream oss;
    oss << "Warning: Can't load chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }
}

// Asynchronous save with priority
std::future<bool> ChunkManager::saveChunkAsync(const ChunkCoord& coord,
                                               int priority) {
  // std::cout << "Called saveChunkAsync" << std::endl;

  ChunkState state = getChunkState(coord);

  if (state == ChunkState::SAVING) {
    return createWaitFuture(coord, ChunkState::INACTIVE);
  }

  if (state == ChunkState::OPTIMIZING) {
    std::ostringstream oss;
    oss << "Warning: Can't save chunk in OPTIMIZING state: " << coord.x << ","
        << coord.y << "," << coord.z;
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  // If loading, wait for it to complete before saving
  if (state == ChunkState::LOADING) {
    std::cout << "Chunk is being loaded, waiting for completion before saving: "
              << coord.x << "," << coord.y << "," << coord.z << std::endl;

    // First wait for ACTIVE state (after save completes)
    auto inactive_future = createWaitFuture(coord, ChunkState::ACTIVE);

    // Create a shared state to continue after waiting
    auto shared_promise = std::make_shared<std::promise<bool>>();
    auto result_future = shared_promise->get_future();

    // Launch a continuation task
    std::thread([this, future = std::move(inactive_future), coord, priority,
                 shared_promise]() mutable {
      try {
        // Wait for save to complete
        bool saved = future.get();
        if (!saved) {
          shared_promise->set_value(false);
          return;
        }

        // Now try to load
        auto load_future = saveChunkAsync(coord, priority);
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

  if (state == ChunkState::ACTIVE) {
    transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::SAVING);
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::SAVE;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();

    std::future<bool> operation_future =
        operation->completion_promise.get_future();
    enqueueOperation(operation);

    return operation_future;
  } else {
    std::ostringstream oss;
    oss << "Warning: Can't save chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }
}

// Process a load operation
bool ChunkManager::processLoadOperation(const ChunkCoord& coord,
                                        bool load_for_opt) {
  // logMemoryUsage("Before Load chunk " + std::to_string(coord.x) + "," +
  //                std::to_string(coord.y) + "," + std::to_string(coord.z));
  // std::cout << "Called processLoadOperation" << std::endl;
  auto start_time = std::chrono::steady_clock::now();

  ChunkState state = getChunkState(coord);

  if (state != ChunkState::LOADING) {
    std::ostringstream oss;
    oss << "Warning: Can't load chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }

  auto chunk_filename = getChunkFilename(coord);
  auto chunk = std::make_shared<Chunk>(model_params_, coord);

  // Load from file
  chunk->getGaussians()->load_checkpoint_incremental(
      chunk_filename.string(), opt_params_, true, true, true);

  assert(chunk->getGaussians()->getXYZ().numel() >= 0 &&
         "Failed to load chunk with no gaussians!");
  assert(chunk->getGaussians()->optimizer_ &&
         "Failed to load chunk with no optimizer!");

  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    auto& meta = chunk_metadata_[coord];
    meta.chunk = chunk;  // Store the chunk in metadata
    meta.load_time = std::chrono::steady_clock::now();
    meta.last_used = meta.load_time;
    meta.usage_count = 0;
    if (load_for_opt) {
      meta.state.store(ChunkState::OPTIMIZING);
    } else {
      meta.state.store(ChunkState::ACTIVE);
    }
  }

  // Notify any waiting threads
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    auto& meta = chunk_metadata_[coord];
    std::unique_lock<std::mutex> op_lock(meta.operation_mutex);
    meta.operation_cv.notify_all();
  }

  incrementStat(stats_.active_chunks);
  incrementStat(stats_.disk_loads);

  // Track VRAM usage after loading and update estimate
  // size_t vram_after = getGPUMemoryUsage();
  // if (!is_active) {  // Only update estimate if we actually loaded a new
  // chunk
  //   updateChunkVramEstimate(vram_before, vram_after, 1);
  // }

  // logMemoryUsage("After Load chunk " + std::to_string(coord.x) + "," +
  //                std::to_string(coord.y) + "," + std::to_string(coord.z));

  // std::cout << "IO Thread: Load operation successful for: " << coord.x << "
  // "
  //           << coord.y << " " << coord.z << " " << std::endl;
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Chunk [" << coord.x << "," << coord.y << "," << coord.z
            << "]: " << "Load completed in " << duration.count() << "ms"
            << std::endl;
  return true;
}

// Process a save operation
bool ChunkManager::processSaveOperation(const ChunkCoord& coord) {
  logMemoryUsage("Before Save chunk " + std::to_string(coord.x) + "," +
                 std::to_string(coord.y) + "," + std::to_string(coord.z));
  // std::cout << "Called processSaveOperation" << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  // std::chrono::milliseconds time_spend_waiting_for_mutex(0);

  // Track VRAM usage before saving
  // size_t vram_before = getGPUMemoryUsage();

  ChunkState state = getChunkState(coord);

  if (state != ChunkState::SAVING) {
    std::ostringstream oss;
    oss << "Warning: Can't save chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }

  // Copy chunk pointer to work with (minimize lock time)
  std::shared_ptr<Chunk> chunk;
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    auto& meta = chunk_metadata_[coord];
    if (!meta.chunk) {
      std::ostringstream oss;
      oss << "Warning: Chunk ptr null for save operation at " << coord.x << ","
          << coord.y << "," << coord.z;
      throw std::runtime_error(oss.str());
    } else {
      chunk = meta.chunk;
    }
  }

  auto chunk_filename = getChunkFilename(coord);

  // Save to file
  chunk->getGaussians()->save_checkpoint(chunk_filename.string());

  // Increment save counter
  incrementStat(stats_.disk_saves);

  // torch::cuda::synchronize();

  // // Clear CUDA cache after saving to free memory
  // c10::cuda::CUDACachingAllocator::emptyCache();

  // Track VRAM usage after saving (should be lower) and update estimate
  // size_t vram_after = getGPUMemoryUsage();
  // updateChunkVramEstimate(vram_before, vram_after,
  //                         -1);  // Negative because chunk was removed

  // logMemoryUsage("After Save chunk " + std::to_string(coord.x) + "," +
  //                std::to_string(coord.y) + "," +
  //                std::to_string(coord.z));

  // std::cout << "IO Thread: Save operation successful for: " << coord.x <<
  // "
  // "
  //           << coord.y << " " << coord.z << " " << std::endl;

  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    auto it = chunk_metadata_.find(coord);
    if (it != chunk_metadata_.end()) {
      it->second.chunk.reset();  // Free memory immediately
      it->second.state.store(ChunkState::INACTIVE);
      // Notify waiters
      std::unique_lock<std::mutex> op_lock(it->second.operation_mutex);
      it->second.operation_cv.notify_all();
    }
  }
  decrementStat(stats_.active_chunks);

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Chunk [" << coord.x << "," << coord.y << "," << coord.z
            << "]: " << "Save completed in " << duration.count() << "ms"
            << std::endl;

  return true;
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
    std::ostringstream oss;
    oss << "Sync load failed for chunk " << coord.x << "," << coord.y << ","
        << coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
  }
}

// Synchronous wrapper for saveChunkAsync
bool ChunkManager::saveChunkSync(const ChunkCoord& coord) {
  // std::cout << "Called saveChunkSync" << std::endl;
  auto future = saveChunkAsync(coord, 10);  // High priority
  try {
    return future.get();  // Wait for completion
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Sync save failed for chunk " << coord.x << "," << coord.y << ","
        << coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
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
    std::ostringstream oss;
    oss << "Error processing operation on chunk " << operation->coord.x << ","
        << operation->coord.y << "," << operation->coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
  }

  try {
    operation->completion_promise.set_value(success);
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Error setting promise value for operation on chunk "
        << operation->coord.x << "," << operation->coord.y << ","
        << operation->coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
  }
}

// Create a future that resolves when a chunk reaches a specific state
std::future<bool> ChunkManager::createWaitFuture(const ChunkCoord& coord,
                                                 ChunkState target_state) {
  auto promise = std::make_shared<std::promise<bool>>();

  std::thread([this, coord, target_state, promise]() {
    const auto TOTAL_TIMEOUT = std::chrono::seconds(5);
    auto end_time = std::chrono::steady_clock::now() + TOTAL_TIMEOUT;
    bool success = false;

    while (std::chrono::steady_clock::now() < end_time) {
      // Step 1: Check current state with a scoped lock
      ChunkState current_state;
      std::condition_variable* cv_ptr = nullptr;
      std::mutex* mutex_ptr = nullptr;

      {
        // Only lock metadata briefly to check state and get pointers
        std::unique_lock<std::mutex> meta_lock(chunk_metadata_mutex_);
        auto it = chunk_metadata_.find(coord);

        if (it == chunk_metadata_.end()) {
          // No metadata exists
          success = (target_state == ChunkState::INACTIVE);
          break;
        }

        // Step 2: Check if already in target state
        current_state = it->second.state.load();
        if (current_state == target_state) {
          success = true;
          break;
        }

        // Step 3: Get pointers to the mutex and cv for this chunk
        mutex_ptr = &(it->second.operation_mutex);
        cv_ptr = &(it->second.operation_cv);
      }  // metadata_mutex is released here

      // Step 4: Now wait on the chunk's own cv with its own mutex
      if (mutex_ptr && cv_ptr) {
        std::unique_lock<std::mutex> op_lock(*mutex_ptr);
        auto wait_status =
            cv_ptr->wait_for(op_lock, std::chrono::milliseconds(100));
        // Continue to next loop iteration regardless of wait result
      } else {
        // If we couldn't get the pointers, sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    promise->set_value(success);
  }).detach();

  return promise->get_future();
}

// Asynchronous delete with priority
std::future<bool> ChunkManager::deleteChunkAsync(const ChunkCoord& coord,
                                                 int priority) {
  // std::cout << "Called deleteChunkAsync" << std::endl;

  ChunkState state = getChunkState(coord);

  // If already deleting, return a future that waits for completion
  if (state == ChunkState::DELETING) {
    return createWaitFuture(coord, ChunkState::INACTIVE);
  }

  // Can't delete if it's being saved or loaded or optimized
  if (state == ChunkState::SAVING || state == ChunkState::LOADING ||
      state == ChunkState::OPTIMIZING) {
    std::promise<bool> promise;
    promise.set_value(false);
    return promise.get_future();
  }

  if (state == ChunkState::INACTIVE || state == ChunkState::ACTIVE) {
    transitionChunkState(coord, state, ChunkState::DELETING);
    auto operation = std::make_shared<ChunkOperation>();
    operation->coord = coord;
    operation->type = ChunkOperation::DELETE;
    operation->priority = priority;
    operation->timestamp = std::chrono::steady_clock::now();

    std::future<bool> future = operation->completion_promise.get_future();
    enqueueOperation(operation);

    return future;
  } else {
    std::ostringstream oss;
    oss << "Warning: Can't delete chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }
}

// Process a delete operation
bool ChunkManager::processDeleteOperation(const ChunkCoord& coord) {
  // logMemoryUsage("Before Delete chunk " + std::to_string(coord.x) + "," +
  //                std::to_string(coord.y) + "," + std::to_string(coord.z));
  ChunkState state = getChunkState(coord);

  if (state != ChunkState::DELETING) {
    std::ostringstream oss;
    oss << "Warning: Can't delete chunk in state " << static_cast<int>(state)
        << ": " << coord.x << "," << coord.y << "," << coord.z;
    throw std::runtime_error(oss.str());
  }

  decrementStat(stats_.active_chunks);
  decrementStat(stats_.existing_chunks);

  // Invalidate visibility cache entries containing this chunk
  {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
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
      std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
      auto& meta = chunk_metadata_[coord];
      // Notify any waiting threads
      {
        std::unique_lock<std::mutex> op_lock(meta.operation_mutex);
        meta.operation_cv.notify_all();
      }
      chunk_metadata_.erase(coord);
    }

    // torch::cuda::synchronize();

    // // Force CUDA cache cleanup after deletion
    // c10::cuda::CUDACachingAllocator::emptyCache();

    // logMemoryUsage("After Delete chunk " + std::to_string(coord.x) + "," +
    //                std::to_string(coord.y) + "," +
    //                std::to_string(coord.z));

    return success;
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Failed to delete chunk " << coord.x << "," << coord.y << ","
        << coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
  }
}

// Synchronous wrapper for deleteChunkAsync
bool ChunkManager::deleteChunkSync(const ChunkCoord& coord) {
  // std::cout << "Called deleteChunkSync" << std::endl;
  auto future = deleteChunkAsync(coord, 10);  // High priority
  try {
    return future.get();  // Wait for completion
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Sync delete failed for chunk " << coord.x << "," << coord.y << ","
        << coord.z << ": " << e.what();
    throw std::runtime_error(oss.str());
    return false;
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

// Get chunk at specific coordinate
std::shared_ptr<Chunk> ChunkManager::getChunkAt(const ChunkCoord& coord) {
  // std::cout << "Called getChunkAt" << std::endl;
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto it = chunk_metadata_.find(coord);
  if (it != chunk_metadata_.end()) {
    return it->second.chunk;
  }
  return nullptr;
}

bool ChunkManager::chunkExists(const ChunkCoord& coord) {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto it = chunk_metadata_.find(coord);
  return (it != chunk_metadata_.end());
}

// Main function that returns visible chunks, handling both active and on-disk
// chunks
std::vector<std::shared_ptr<Chunk>> ChunkManager::loadVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  // std::cout << "Called loadVisibleChunks" << std::endl;
  // auto timer = ProfilingUtils::Timer("ChunkManager::loadVisibleChunks");
  auto start_time = std::chrono::steady_clock::now();

  if (!keyframe) {
    throw std::invalid_argument("Null keyframe passed to loadVisibleChunks");
  }

  triggerLruCheck();

  std::vector<ChunkCoord> visible_chunk_coords =
      frustumCullChunks(keyframe, use_cache);

  // Now we have the list of visible chunk coordinates
  // Start asynchronous loading of chunks
  std::vector<std::shared_ptr<Chunk>> result_chunks;
  std::vector<ChunkCoord> chunks_to_load;
  std::vector<std::future<bool>> load_futures;
  std::vector<ChunkCoord> load_coords;

  for (const auto& coord : visible_chunk_coords) {
    ChunkState state = getChunkState(coord);

    if (state == ChunkState::UNKNOWN) continue;

    if (state == ChunkState::ACTIVE || state == ChunkState::OPTIMIZING) {
      updateLastUsedTime(coord);

      if (state == ChunkState::ACTIVE) {
        transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::OPTIMIZING);
      }

      auto chunk = getChunkAt(coord);
      if (chunk) {
        result_chunks.push_back(chunk);
      } else {
        throw std::runtime_error(
            "Chunk in ACTIVE/OPTIMIZING state but not found in metadata");
      }

    } else {
      // Need to load from disk or in an incompatible state
      load_futures.push_back(loadChunkAsync(coord, 10, true, false));
      load_coords.push_back(coord);
    }
  }

  // Wait for critical chunks to load (with timeout)
  const auto timeout = std::chrono::milliseconds(2000);
  for (size_t i = 0; i < load_futures.size(); ++i) {
    if (load_futures[i].wait_for(timeout) == std::future_status::ready) {
      if (load_futures[i].get()) {
        ChunkCoord coord = load_coords[i];  // Use the correct array

        // Verify the chunk is in OPTIMIZING state after loading
        if (getChunkState(coord) == ChunkState::OPTIMIZING) {
          std::shared_ptr<Chunk> chunk;
          {
            std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
            auto it = chunk_metadata_.find(coord);
            if (it != chunk_metadata_.end()) {
              chunk = it->second.chunk;
            }
          }

          if (chunk && chunk->getGaussians()) {
            result_chunks.push_back(chunk);
          }
        } else {
          std::ostringstream oss;
          oss << "Chunk " << coord.x << "," << coord.y << "," << coord.z
              << " loaded but not in OPTIMIZING state after load";
          throw std::runtime_error(oss.str());
        }
      }
    } else {
      std::ostringstream oss;
      oss << "Timeout while loading chunk: " << load_coords[i].x << ","
          << load_coords[i].y << "," << load_coords[i].z;
      std::cerr << oss.str() << std::endl;  // Handle timeout case if needed
    }
  }

  // After loading chunks, trigger LRU check to potentially free up space
  triggerLruCheck();

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // std::cout << "loadVisibleChunks in " << duration.count() << "ms" <<
  // std::endl;

  return result_chunks;
}

// Unused right now
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

  triggerLruCheck();

  std::vector<ChunkCoord> visible_chunk_coords =
      frustumCullChunks(keyframe, use_cache);

  for (const auto& coord : visible_chunk_coords) {
    // Check if chunk has been seen before
    if (!chunkExists(coord)) continue;

    std::cout << "getChunkState called in preloadVisibleChunks" << std::endl;
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
      loadChunkAsync(coord, 3, false, false);
    }
  }
}

// Helper function to find visible chunks within search radius
std::vector<ChunkCoord> ChunkManager::frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  if (!keyframe) {
    return {};
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  // Check cache (keep original cache logic)
  if (use_cache) {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto cache_it = visibility_cache_.find(keyframe_id);
    if (cache_it != visibility_cache_.end()) {
      auto& entry = cache_it->second;
      if ((now - entry.timestamp) < cache_expiry_time_ &&
          pose_nearly_equal(current_pose, entry.pose)) {
        entry.timestamp = now;
        return entry.visible_chunks;
      }
    }
  }

  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);
  torch::Tensor tensor_matrix = keyframe->projection_matrix_;

  // Ensure tensor is on CPU and contiguous
  tensor_matrix = tensor_matrix.cpu().contiguous();

  // Get data pointer and create Eigen matrix
  float* data_ptr = tensor_matrix.data_ptr<float>();
  Eigen::Matrix4f proj_matrix = Eigen::Map<Eigen::Matrix4f>(data_ptr);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Get camera position for chunk search
  Sophus::SE3d Twc = current_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position);

  // Calculate parameters
  int search_radius =
      std::ceil(keyframe->zfar_ / chunk_size_ * std::sqrt(3.0f)) + 2;
  float max_distance = keyframe->zfar_ + chunk_size_ * 1.732f;

  // Call hierarchical culling
  std::vector<ChunkCoord> visible_chunks =
      cullChunksHierarchical(vp_matrix, camera_position, camera_chunk,
                             search_radius, chunk_size_, max_distance);

  // Update cache (keep original cache update logic)
  if (use_cache) {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
    VisibilityCacheEntry entry;
    entry.pose = current_pose;
    entry.visible_chunks = visible_chunks;
    entry.timestamp = std::chrono::steady_clock::now();
    visibility_cache_[keyframe_id] = entry;
  }

  return visible_chunks;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ChunkManager::groupPointsByChunk(const torch::Tensor& positions) {
  // std::cout << "Called groupPointsByChunk" << std::endl;
  // Called by addPoints
  // Convert positions to chunk coordinates
  float half_chunk = chunk_size_ * 0.5f;
  torch::Tensor shifted_positions = positions + half_chunk;
  torch::Tensor chunk_coords = torch::floor(shifted_positions / chunk_size_);
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
void ChunkManager::addPointsToChunks(const torch::Tensor& points,
                                     const torch::Tensor& colors,
                                     const torch::Tensor& scales,
                                     const torch::Tensor& opacities) {
  auto start_time = std::chrono::steady_clock::now();
  torch::NoGradGuard no_grad;
  const int min_new_points_threshold = 10;
  // std::cout << "addPointsToChunks called with " << points.size(0) << "
  // points
  // "
  //           << std::endl;

  // Skip if not enough points
  if (points.size(0) < min_new_points_threshold) {
    return;
  }

  // Convert to CUDA for processing - only once
  torch::Tensor points_cuda = points.to(torch::kCUDA);
  torch::Tensor colors_cuda = colors.to(torch::kCUDA);
  torch::Tensor opacities_cuda = opacities.to(torch::kCUDA);
  torch::Tensor scales_cuda;
  if (scales.defined() && scales.size(0) > 0)
    scales_cuda = scales.to(torch::kCUDA);

  // Group points by chunk
  auto [unique_chunks, inverse_indices, points_per_chunk] =
      groupPointsByChunk(points_cuda);

  // std::cout << "Adding points to " << unique_chunks.size(0) << " chunks"
  //           << std::endl;

  // Process each unique chunk sequentially
  for (int64_t i = 0; i < unique_chunks.size(0); i++) {
    // Get chunk coordinate
    ChunkCoord coord{unique_chunks[i][0].item<int64_t>(),
                     unique_chunks[i][1].item<int64_t>(),
                     unique_chunks[i][2].item<int64_t>()};

    // Create mask for points in this chunk
    torch::Tensor chunk_mask = (inverse_indices == i);

    // Extract points for this chunk
    torch::Tensor chunk_points = points_cuda.index({chunk_mask});
    torch::Tensor chunk_colors = colors_cuda.index({chunk_mask});
    torch::Tensor chunk_opacities = opacities_cuda.index({chunk_mask});

    torch::Tensor chunk_scales;
    if (scales.defined() && scales.size(0) > 0)
      chunk_scales = scales_cuda.index({chunk_mask});

    // std::cout << "Adding " << chunk_points.sizes()[0]
    //           << " points to chunk: " << coord.x << "," << coord.y << ","
    //           << coord.z << std::endl;

    // Skip if not enough points
    if (chunk_points.size(0) < min_new_points_threshold) {
      // std::cout << "Skip as not enough points" << std::endl;
      continue;  // Skip to next chunk
    }

    bool loaded = false;

    ChunkState state = getChunkState(coord);

    if (state == ChunkState::INACTIVE) {
      // std::cout << "Chunk is inactive, loading..." << std::endl;
      // Load the chunk synchronously
      loaded = loadChunkSync(coord, 10, true);

      if (!loaded) {
        std::ostringstream oss;
        oss << "Failed to load chunk: " << coord.x << "," << coord.y << ","
            << coord.z;
        throw std::runtime_error(oss.str());
      }

    } else if (state == ChunkState::ACTIVE) {
      // std::cout << "Chunk is active, no loading needed" << std::endl;
      // Transition to OPTIMIZING
      transitionChunkState(coord, ChunkState::ACTIVE, ChunkState::OPTIMIZING);

    } else if (state == ChunkState::UNKNOWN) {
      // If unknown, create a new chunk
      auto chunk = std::make_shared<Chunk>(model_params_, coord);

      // std::cout << "Chunk created, updating metadata" << std::endl;
      {
        std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
        auto& meta = chunk_metadata_[coord];
        meta.load_time = std::chrono::steady_clock::now();
        meta.last_used = meta.load_time;
        meta.usage_count = 0;
        meta.state.store(ChunkState::OPTIMIZING);
        meta.chunk = chunk;  // Store the chunk pointer
      }

      ChunkOptimizationGuard guard(this, {chunk});

      incrementStat(stats_.active_chunks);
      incrementStat(stats_.existing_chunks);

      // Initialize the Gaussian model
      if (chunk_scales.defined() && chunk_scales.size(0) > 0) {
        chunk->getGaussians()->createFromPcd(
            chunk_points, chunk_colors, chunk_scales, chunk_opacities,
            getCurrentIteration(), cameras_extent_);
      } else {
        chunk->getGaussians()->createFromPcd(
            chunk_points, chunk_colors, torch::Tensor(), chunk_opacities,
            getCurrentIteration(), cameras_extent_);
      }
      chunk->getGaussians()->trainingSetup(opt_params_);
      continue;
    } else if (state == ChunkState::LOADING || state == ChunkState::SAVING ||
               state == ChunkState::DELETING) {
      std::ostringstream oss;
      oss << "Can't add points to chunk in state: " << static_cast<int>(state)
          << " for chunk: " << coord.x << "," << coord.y << "," << coord.z;
      std::cerr << oss.str() << std::endl;
    }

    std::shared_ptr<Chunk> chunk = getChunkAt(coord);
    ChunkOptimizationGuard guard(this, {chunk});

    if (!chunk || !chunk->getGaussians()) {
      std::ostringstream oss;
      oss << "Chunk is null or has no Gaussians: " << coord.x << "," << coord.y
          << "," << coord.z;
      throw std::runtime_error(oss.str());
    }

    if (chunk_scales.defined() && chunk_scales.size(0) > 0) {
      chunk->getGaussians()->increasePcd(chunk_points, chunk_colors,
                                         chunk_scales, chunk_opacities,
                                         getCurrentIteration());
    } else {
      chunk->getGaussians()->increasePcd(chunk_points, chunk_colors,
                                         torch::Tensor(), chunk_opacities,
                                         getCurrentIteration());
    }
    triggerLruCheck();
  }

  // torch::cuda::synchronize();
  // triggerLruCheck();
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // std::cout << "addPointsToChunks completed in " << duration.count() << "ms"
  //           << std::endl;
}

// Get stats
ChunkManager::ChunkStats ChunkManager::getStats() const {
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

size_t ChunkManager::getGPUMemoryUsage() const {
  if (torch::cuda::is_available()) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    // Get current allocated bytes (this is what we want to track for chunks)
    c10Alloc::Stat alloc_bytes =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    return alloc_bytes.current;
  }
  return 0;
}

void ChunkManager::logMemoryUsage(const std::string& operation) const {
  if (torch::cuda::is_available()) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    // Get current allocated and reserved bytes
    c10Alloc::Stat alloc_bytes =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    c10Alloc::Stat reserved_bytes =
        mem_stats
            .reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];

    size_t allocated_mb = alloc_bytes.current / (1024 * 1024);
    size_t reserved_mb = reserved_bytes.current / (1024 * 1024);

    std::cout << "[GPU Memory] " << operation << ": " << allocated_mb
              << " MB allocated, " << reserved_mb << " MB reserved"
              << std::endl;
  }
}

// Cull chunks with too few points to ensure rendering stability
bool ChunkManager::cullSparseChunks(int min_points_threshold,
                                    int min_chunk_iterations) {
  // std::cout << "Called cullSparseChunks" << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  std::vector<ChunkCoord> chunks_to_cull;
  bool any_culled = false;

  // First pass: identify chunks to cull (with minimal locking time)
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);

    // Examine all active chunks
    for (const auto& [coord, metadata] : chunk_metadata_) {
      // Skip chunks that are in transitional states
      ChunkState state = metadata.state.load();

      // Only consider ACTIVE or OPTIMZING chunks for culling
      if (state != ChunkState::ACTIVE && state != ChunkState::OPTIMIZING) {
        continue;
      }

      auto chunk = metadata.chunk;

      if (!chunk || !chunk->getGaussians()) {
        // Mark invalid chunks for removal
        std::cout << "Invalid chunk found at " << coord.x << "," << coord.y
                  << "," << coord.z << ", marking for culling." << std::endl;
        chunks_to_cull.push_back(coord);
        continue;
      }

      // Get number of active points in chunk
      int num_points = chunk->getGaussians()->getXYZ().size(0);

      int iterations = chunk->getGaussians()->getLocalIteration();

      // Cull if has super few points. Else if old enough but still below
      // threshold, mark for culling
      if (num_points < 5 || (num_points < min_points_threshold &&
                             iterations > min_chunk_iterations)) {
        chunks_to_cull.push_back(coord);
      }
    }
  }

  for (const auto& coord : chunks_to_cull) {
    deleteChunkAsync(coord, 3);
    any_culled = true;
  }
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);
  // std::cout << "cullSparseChunks completed in " << duration.count() << "ns"
  //           << std::endl;

  return any_culled;
}

std::vector<ChunkCoord> ChunkManager::getExistingChunkCoords() {
  // std::cout << "Called getExistingChunkCoords" << std::endl;
  std::vector<ChunkCoord> result;
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  // Iterate through the cache and collect all chunks that exist
  for (const auto& [coord, metadata] : chunk_metadata_) {
    result.push_back(coord);
  }

  return result;
}

void ChunkManager::transferGaussiansAcrossChunks() {
  std::cout
      << "Called transferGaussiansAcrossChunks with optimizer state transfer"
      << std::endl;
  torch::NoGradGuard no_grad;

  // Configuration for batched processing
  const size_t MAX_BATCH_SIZE = 3000;  // Reduced due to additional state data
  const size_t MAX_POINTS_PER_BATCH = 30000;  // Reduced due to memory overhead

  std::vector<std::tuple<ChunkCoord, GaussianTransferData>> current_batch;

  size_t current_batch_points = 0;

  // Get all existing chunk coordinates
  std::vector<ChunkCoord> all_chunks = getExistingChunkCoords();
  std::cout << "Transferring Gaussians across " << all_chunks.size()
            << " chunks" << std::endl;

  // Process chunks and handle transfers in batches
  for (size_t i = 0; i < all_chunks.size(); i++) {
    const ChunkCoord& coord = all_chunks[i];

    std::cout << getVramStatus() << std::endl;
    waitForVramAvailable(0.80f, std::chrono::seconds(10), "gaussian transfer");

    if (!loadChunkSync(coord, true, false)) {
      throw std::runtime_error("Can't load chunk in gaussian transfer");
    }

    auto chunk = getChunkAt(coord);
    ChunkOptimizationGuard guard(this, {chunk});
    if (!chunk || !chunk->getGaussians()) {
      std::ostringstream oss;
      oss << "Gaussians/Chunk invalid for coord: " << coord.x << "," << coord.y
          << "," << coord.z;
      throw std::runtime_error(oss.str());
    }

    auto gaussians = chunk->getGaussians();
    auto points = gaussians->getXYZ();

    if (points.size(0) == 0) {
      std::cout << "Skipping chunk, no points in it" << std::endl;
      continue;  // Guard will automatically release
    }

    // Get AABB and find outside points
    AABB chunk_aabb = getChunkAABB(coord);
    torch::Tensor outside_mask =
        ((points.index({torch::indexing::Slice(), 0}) < chunk_aabb.min.x()) |
         (points.index({torch::indexing::Slice(), 0}) > chunk_aabb.max.x()) |
         (points.index({torch::indexing::Slice(), 1}) < chunk_aabb.min.y()) |
         (points.index({torch::indexing::Slice(), 1}) > chunk_aabb.max.y()) |
         (points.index({torch::indexing::Slice(), 2}) < chunk_aabb.min.z()) |
         (points.index({torch::indexing::Slice(), 2}) > chunk_aabb.max.z()));

    int num_outside = outside_mask.sum().item<int>();
    if (num_outside == 0) {
      continue;  // Guard will automatically release
    }

    std::cout << "Found " << num_outside << " points to transfer from chunk ("
              << coord.x << "," << coord.y << "," << coord.z << ")"
              << std::endl;

    // Extract outside points WITH their optimizer states
    GaussianTransferData outside_data =
        gaussians->extractGaussiansWithStates(outside_mask);

    // Remove points from source chunk
    gaussians->prunePoints(outside_mask);

    // Group points by destination chunk
    auto [unique_dest_chunks, inverse_indices, points_per_chunk] =
        groupPointsByChunk(outside_data.points);

    // Process each destination chunk for this source
    for (int k = 0; k < unique_dest_chunks.size(0); k++) {
      ChunkCoord dest_coord{unique_dest_chunks[k][0].item<int64_t>(),
                            unique_dest_chunks[k][1].item<int64_t>(),
                            unique_dest_chunks[k][2].item<int64_t>()};

      if (dest_coord == coord) continue;  // Skip self-transfer

      torch::Tensor chunk_mask = (inverse_indices == k);
      const int MIN_TRANSFER_THRESHOLD = 30;
      int points_to_transfer = chunk_mask.sum().item<int>();

      if (points_to_transfer < MIN_TRANSFER_THRESHOLD) continue;

      // Check if adding this transfer would exceed batch limits
      if ((current_batch.size() >= MAX_BATCH_SIZE) ||
          (current_batch_points + points_to_transfer > MAX_POINTS_PER_BATCH)) {
        // Process current batch before adding more
        std::cout << "Processing batch with " << current_batch.size()
                  << " transfers and " << current_batch_points << " points"
                  << std::endl;
        processBatchWithStates(current_batch);

        // Clear batch
        current_batch.clear();
        current_batch_points = 0;
      }

      // Extract data for this destination chunk
      GaussianTransferData chunk_data;
      chunk_data.points = outside_data.points.index({chunk_mask}).clone();
      chunk_data.features_dc =
          outside_data.features_dc.index({chunk_mask}).clone();
      chunk_data.features_rest =
          outside_data.features_rest.index({chunk_mask}).clone();
      chunk_data.opacities = outside_data.opacities.index({chunk_mask}).clone();
      chunk_data.scaling = outside_data.scaling.index({chunk_mask}).clone();
      chunk_data.rotation = outside_data.rotation.index({chunk_mask}).clone();
      chunk_data.exist_since =
          outside_data.exist_since.index({chunk_mask}).clone();

      // Extract auxiliary states
      chunk_data.position_lrs =
          outside_data.position_lrs.index({chunk_mask}).clone();
      chunk_data.xyz_gradient_accum =
          outside_data.xyz_gradient_accum.index({chunk_mask}).clone();
      chunk_data.denom = outside_data.denom.index({chunk_mask}).clone();
      chunk_data.max_radii2D =
          outside_data.max_radii2D.index({chunk_mask}).clone();

      // Extract optimizer states
      chunk_data.exp_avg_states.resize(6);
      chunk_data.exp_avg_sq_states.resize(6);
      chunk_data.step_states.resize(6);

      for (int group_idx = 0; group_idx < 6; ++group_idx) {
        if (outside_data.exp_avg_states[group_idx].defined()) {
          chunk_data.exp_avg_states[group_idx] =
              outside_data.exp_avg_states[group_idx]
                  .index({chunk_mask})
                  .clone();
          chunk_data.exp_avg_sq_states[group_idx] =
              outside_data.exp_avg_sq_states[group_idx]
                  .index({chunk_mask})
                  .clone();
          chunk_data.step_states[group_idx] =
              outside_data.step_states[group_idx].index({chunk_mask}).clone();
        }
      }

      current_batch.push_back(
          std::make_tuple(dest_coord, std::move(chunk_data)));

      current_batch_points += points_to_transfer;
    }
  }
  triggerLruCheck();

  // Process any remaining transfers in the final batch
  if (!current_batch.empty()) {
    std::cout << "Processing final batch with " << current_batch.size()
              << " transfers and " << current_batch_points << " points"
              << std::endl;
    processBatchWithStates(current_batch);
  }

  std::cout << "Gaussian transfer with optimizer states completed" << std::endl;
}

void ChunkManager::processBatchWithStates(
    const std::vector<std::tuple<ChunkCoord, GaussianTransferData>>& batch) {
  // Group by destination
  std::unordered_map<ChunkCoord, std::vector<size_t>, ChunkCoordHash>
      dest_to_transfers;

  for (size_t i = 0; i < batch.size(); i++) {
    ChunkCoord dest_coord = std::get<0>(batch[i]);
    dest_to_transfers[dest_coord].push_back(i);
  }

  // Process each destination
  for (const auto& [dest_coord, transfer_indices] : dest_to_transfers) {
    // Combine all transfers for this destination
    std::vector<GaussianTransferData> transfers_for_dest;
    for (size_t idx : transfer_indices) {
      transfers_for_dest.push_back(std::get<1>(batch[idx]));
    }

    GaussianTransferData combined_data =
        combineTransferData(transfers_for_dest);

    std::cout << "Applying " << combined_data.points.size(0)
              << " points with states to chunk (" << dest_coord.x << ","
              << dest_coord.y << "," << dest_coord.z << ")" << std::endl;

    try {
      applyTransferToDestinationWithStates(dest_coord, combined_data);
    } catch (const std::exception& e) {
      std::cerr << "Error applying batch transfer: " << e.what() << std::endl;
    }
  }
}

// Helper to combine multiple transfer data
GaussianTransferData ChunkManager::combineTransferData(
    const std::vector<GaussianTransferData>& transfers) {
  GaussianTransferData combined;

  std::vector<torch::Tensor> points_list, features_dc_list, features_rest_list;
  std::vector<torch::Tensor> opacities_list, scaling_list, rotation_list,
      exist_since_list;
  std::vector<torch::Tensor> position_lrs_list, xyz_grad_list, denom_list,
      radii_list;

  std::vector<std::vector<torch::Tensor>> exp_avg_lists(6), exp_avg_sq_lists(6),
      step_lists(6);

  for (const auto& transfer : transfers) {
    points_list.push_back(transfer.points);
    features_dc_list.push_back(transfer.features_dc);
    features_rest_list.push_back(transfer.features_rest);
    opacities_list.push_back(transfer.opacities);
    scaling_list.push_back(transfer.scaling);
    rotation_list.push_back(transfer.rotation);
    exist_since_list.push_back(transfer.exist_since);
    position_lrs_list.push_back(transfer.position_lrs);
    xyz_grad_list.push_back(transfer.xyz_gradient_accum);
    denom_list.push_back(transfer.denom);
    radii_list.push_back(transfer.max_radii2D);

    for (int i = 0; i < 6; ++i) {
      if (transfer.exp_avg_states[i].defined()) {
        exp_avg_lists[i].push_back(transfer.exp_avg_states[i]);
        exp_avg_sq_lists[i].push_back(transfer.exp_avg_sq_states[i]);
        step_lists[i].push_back(transfer.step_states[i]);
      }
    }
  }

  // Concatenate all tensors
  combined.points = torch::cat(points_list, 0);
  combined.features_dc = torch::cat(features_dc_list, 0);
  combined.features_rest = torch::cat(features_rest_list, 0);
  combined.opacities = torch::cat(opacities_list, 0);
  combined.scaling = torch::cat(scaling_list, 0);
  combined.rotation = torch::cat(rotation_list, 0);
  combined.exist_since = torch::cat(exist_since_list, 0);
  combined.position_lrs = torch::cat(position_lrs_list, 0);
  combined.xyz_gradient_accum = torch::cat(xyz_grad_list, 0);
  combined.denom = torch::cat(denom_list, 0);
  combined.max_radii2D = torch::cat(radii_list, 0);

  combined.exp_avg_states.resize(6);
  combined.exp_avg_sq_states.resize(6);
  combined.step_states.resize(6);

  for (int i = 0; i < 6; ++i) {
    if (!exp_avg_lists[i].empty()) {
      combined.exp_avg_states[i] = torch::cat(exp_avg_lists[i], 0);
      combined.exp_avg_sq_states[i] = torch::cat(exp_avg_sq_lists[i], 0);
      combined.step_states[i] = torch::cat(step_lists[i], 0);
    }
  }

  return combined;
}

void ChunkManager::applyTransferToDestinationWithStates(
    const ChunkCoord& dest_coord,
    const GaussianTransferData& transfer_data) {
  std::shared_ptr<Chunk> dest_chunk;
  bool chunk_exists = chunkExists(dest_coord);

  if (chunk_exists) {
    waitForVramAvailable(0.80f, std::chrono::seconds(10), "gaussian transfer");
    if (!loadChunkSync(dest_coord, true, false)) {
      throw std::runtime_error("Failed to load existing destination chunk");
    }

    dest_chunk = getChunkAt(dest_coord);
    ChunkOptimizationGuard guard(this, {dest_chunk});
    dest_chunk->getGaussians()->addGaussiansWithStates(transfer_data);
  } else {
    // Create new chunk
    dest_chunk = std::make_shared<Chunk>(model_params_, dest_coord);
    ChunkOptimizationGuard guard(this, {dest_chunk});

    {
      std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
      auto& meta = chunk_metadata_[dest_coord];
      meta.load_time = std::chrono::steady_clock::now();
      meta.last_used = meta.load_time;
      meta.usage_count = 0;
      meta.state.store(ChunkState::OPTIMIZING);
      meta.chunk = dest_chunk;  // Store the chunk pointer
    }

    incrementStat(stats_.active_chunks);
    incrementStat(stats_.existing_chunks);

    if (!dest_chunk || !dest_chunk->getGaussians()) {
      throw std::runtime_error("Null destination chunk or gaussians");
    }

    dest_chunk->getGaussians()->initializeFromTransferData(
        transfer_data, opt_params_, cameras_extent_);
  }
}

void ChunkManager::releaseChunksFromOptimization(
    const std::vector<ChunkCoord>& chunk_coords) {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  for (const auto& coord : chunk_coords) {
    auto& meta = chunk_metadata_[coord];

    if (meta.state.load() == ChunkState::OPTIMIZING) {
      // Transition back to ACTIVE
      meta.state.store(ChunkState::ACTIVE);

      // Notify any waiting threads
      std::unique_lock<std::mutex> op_lock(meta.operation_mutex);
      meta.operation_cv.notify_all();
    } else {
      std::cout << "[ChunkManager] WARNING: Chunk (" << coord.x << ","
                << coord.y << "," << coord.z << ") is in state "
                << static_cast<int>(meta.state.load())
                << " instead of OPTIMIZING" << std::endl;
    }
  }
}

void ChunkManager::releaseChunksFromOptimization(
    const std::vector<std::shared_ptr<Chunk>>& chunks) {
  // std::cout << "[ChunkManager] releaseChunksFromOptimization called for "
  //           << chunks.size() << " chunks" << std::endl;
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  for (const auto& chunk : chunks) {
    const ChunkCoord& coord = chunk->getCoord();
    auto& meta = chunk_metadata_[coord];

    if (meta.state.load() == ChunkState::OPTIMIZING) {
      // Transition back to ACTIVE
      meta.state.store(ChunkState::ACTIVE);

      // Notify any waiting threads
      std::unique_lock<std::mutex> op_lock(meta.operation_mutex);
      meta.operation_cv.notify_all();
    } else {
      std::cout << "[ChunkManager] WARNING: Chunk (" << coord.x << ","
                << coord.y << "," << coord.z << ") is in state "
                << static_cast<int>(meta.state.load())
                << " instead of OPTIMIZING" << std::endl;
    }
  }
}

void ChunkManager::releaseAllChunksFromOptimization() {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  for (auto& [coord, meta] : chunk_metadata_) {
    if (meta.state.load() == ChunkState::OPTIMIZING) {
      // Transition back to ACTIVE
      meta.state.store(ChunkState::ACTIVE);

      // Notify any waiting threads
      std::unique_lock<std::mutex> op_lock(meta.operation_mutex);
      meta.operation_cv.notify_all();
    } else {
      std::cout << "[ChunkManager] WARNING: Chunk (" << coord.x << ","
                << coord.y << "," << coord.z << ") is in state "
                << static_cast<int>(meta.state.load())
                << " instead of OPTIMIZING" << std::endl;
    }
  }
}

void ChunkManager::lruEvictionThreadFunction() {
  std::cout << "Starting VRAM-based LRU eviction thread" << std::endl;

  while (!stop_lru_thread_) {
    // Wait for trigger or periodic check
    {
      std::unique_lock<std::mutex> lock(lru_mutex_);
      lru_cv_.wait_for(lock, lru_check_interval_,
                       [this]() { return stop_lru_thread_.load(); });

      if (stop_lru_thread_) break;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Simple check: do we need to evict?
    if (!shouldEvictChunks()) {
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);
      // std::cout << "lruEvictionThreadFunction early exit in "
      //           << duration.count() << "ms" << std::endl;
      continue;  // VRAM usage is fine
    }

    // Calculate how many chunks to evict
    int to_evict = calculateEvictionCount();
    if (to_evict <= 0) {
      std::cout << "LRU Eviction: Need to evict " << to_evict << " chunks "
                << std::endl;
      continue;  // Nothing to evict
    }

    // Find and evict the least recently used chunks
    std::vector<std::pair<ChunkCoord, std::chrono::steady_clock::time_point>>
        candidates;

    int chunk_inactive_count = 0;
    int chunk_optimizing_count = 0;
    int chunk_loading_count = 0;
    int chunk_saving_count = 0;
    int chunks_skipped_since_too_recently_loaded = 0;

    {
      std::unique_lock<std::mutex> meta_lock(chunk_metadata_mutex_);

      // Calculate minimum retention time
      auto now = std::chrono::steady_clock::now();
      auto min_retention_cutoff = now - min_retention_time_;

      // Collect all active chunks with their last_used time
      for (const auto& [coord, meta] : chunk_metadata_) {
        auto state = meta.state.load();

        if (state == ChunkState::INACTIVE) {
          chunk_inactive_count++;
        } else if (state == ChunkState::OPTIMIZING) {
          chunk_optimizing_count++;
        } else if (state == ChunkState::LOADING) {
          chunk_loading_count++;
        } else if (state == ChunkState::SAVING) {
          chunk_saving_count++;
        } else if (state == ChunkState::UNKNOWN) {
          // Skip unknown chunks
          continue;
        } else if (state == ChunkState::ACTIVE) {
          if (meta.load_time < min_retention_cutoff) {
            candidates.push_back({coord, meta.last_used});
          } else {
            chunks_skipped_since_too_recently_loaded++;
          }
        }
      }
    }

    // If no valid candidates, try again later
    if (candidates.empty()) {
      std::cout << "No valid candidates for eviction" << std::endl;
      std::cout << "LRU Eviction: Skipped: Inactive: " << chunk_inactive_count
                << ", Optimizing: " << chunk_optimizing_count
                << ", Loading: " << chunk_loading_count
                << ", Saving: " << chunk_saving_count
                << ", Skipped due to recent load: "
                << chunks_skipped_since_too_recently_loaded << std::endl;
      continue;
    }

    // Sort by last_used time (oldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Schedule saves for the oldest chunks
    int evicted = 0;
    for (size_t i = 0; i < candidates.size() && evicted < to_evict; ++i) {
      const ChunkCoord& coord = candidates[i].first;
      saveChunkAsync(coord, 10);  // High priority save
      evicted++;
    }

    size_t current_mb = getGPUMemoryUsage() / (1024 * 1024);
    std::cout << "VRAM eviction: " << current_mb << "MB/" << max_vram_budget_mb_
              << "MB, evicted " << evicted << " chunks" << std::endl;

    std::cout << "LRU Eviction: Skipped: Inactive: " << chunk_inactive_count
              << ", Optimizing: " << chunk_optimizing_count
              << ", Loading: " << chunk_loading_count
              << ", Saving: " << chunk_saving_count
              << ", Skipped due to recent load: "
              << chunks_skipped_since_too_recently_loaded << std::endl;
  }

  std::cout << "VRAM-based LRU eviction thread terminating" << std::endl;
};

void ChunkManager::updateLastUsedTime(const ChunkCoord& coord) {
  auto start_time = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    auto& meta = chunk_metadata_[coord];
    meta.last_used = std::chrono::steady_clock::now();
    meta.usage_count++;
  }
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);
  // std::cout << "updateLastUsedTime completed in " << duration.count() << "ns"
  //           << std::endl;
}

void ChunkManager::updateLastUsedTimeForChunks(
    const std::vector<ChunkCoord>& coords) {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto now = std::chrono::steady_clock::now();
  for (const auto& coord : coords) {
    auto& meta = chunk_metadata_[coord];
    meta.last_used = std::chrono::steady_clock::now();
    meta.usage_count++;
  }
}

void ChunkManager::triggerLruCheck() {
  std::unique_lock<std::mutex> lock(lru_mutex_);
  lru_cv_.notify_one();
}

void ChunkManager::initializeMetaData(ChunkCoord& coord) {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  auto& meta = chunk_metadata_[coord];
  meta.load_time = std::chrono::steady_clock::now();
  meta.last_used = meta.load_time;
  meta.usage_count = 0;
  meta.state.store(ChunkState::INACTIVE);
}

// Memory test function - saves all active chunks, checks memory, then reloads
void ChunkManager::testMemoryUsagePattern() {
  std::cout << "=== MEMORY TEST STARTING ===" << std::endl;

  // Step 1: Log initial memory
  logMemoryUsage("Initial memory before test");

  // Step 2: Get all currently active chunks
  std::vector<ChunkCoord> active_chunk_coords;
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);

    // Examine all active chunks
    for (const auto& [coord, metadata] : chunk_metadata_) {
      // Skip chunks that are in transitional states
      ChunkState state = metadata.state.load();

      // Only consider ACTIVE or OPTIMZING chunks for culling
      if (state != ChunkState::ACTIVE && state != ChunkState::OPTIMIZING) {
        continue;
      }

      active_chunk_coords.push_back(coord);
    }
  }

  std::cout << "Found " << active_chunk_coords.size()
            << " active chunks to test" << std::endl;

  if (active_chunk_coords.empty()) {
    std::cout << "No active chunks to test, exiting memory test" << std::endl;
    return;
  }

  // Step 3: Release all chunks from optimization first
  releaseAllChunksFromOptimization();

  // Step 4: Save all active chunks
  std::cout << "Saving all active chunks..." << std::endl;
  std::vector<std::future<bool>> save_futures;
  for (const auto& coord : active_chunk_coords) {
    save_futures.push_back(saveChunkAsync(coord, 15));  // High priority
  }

  // Wait for all saves to complete
  int saved_count = 0;
  for (auto& future : save_futures) {
    try {
      if (future.get()) {
        saved_count++;
      }
    } catch (const std::exception& e) {
      std::cerr << "Save failed: " << e.what() << std::endl;
    }
  }

  std::cout << "Saved " << saved_count << " chunks successfully" << std::endl;

  // Step 5: Force memory cleanup
  std::cout << "Forcing memory cleanup..." << std::endl;
  torch::cuda::synchronize();
  // c10::cuda::CUDACachingAllocator::emptyCache();

  // Log memory after saves
  logMemoryUsage("After saving all chunks");

  // Step 6: Wait a moment for cleanup to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Step 7: Log memory after cleanup delay
  logMemoryUsage("After cleanup delay");

  // Step 8: Reload all chunks
  std::cout << "Reloading all chunks..." << std::endl;
  std::vector<std::future<bool>> load_futures;
  for (const auto& coord : active_chunk_coords) {
    load_futures.push_back(loadChunkAsync(
        coord, 15, true, false));  // High priority, for optimization
  }

  // Wait for all loads to complete
  int loaded_count = 0;
  for (auto& future : load_futures) {
    try {
      if (future.get()) {
        loaded_count++;
      }
    } catch (const std::exception& e) {
      std::cerr << "Load failed: " << e.what() << std::endl;
    }
  }

  std::cout << "Loaded " << loaded_count << " chunks successfully" << std::endl;

  // Step 9: Log final memory
  logMemoryUsage("After reloading all chunks");

  // Step 10: Force final cleanup and log again
  torch::cuda::synchronize();
  // c10::cuda::CUDACachingAllocator::emptyCache();
  logMemoryUsage("Final memory after cleanup");

  std::cout << "=== MEMORY TEST COMPLETE ===" << std::endl;
  std::cout << "Active chunks before: " << active_chunk_coords.size()
            << std::endl;
  std::cout << "Chunks saved: " << saved_count << std::endl;
  std::cout << "Chunks reloaded: " << loaded_count << std::endl;
}

bool ChunkManager::shouldEvictChunks() {
  auto now = std::chrono::steady_clock::now();
  if (now - last_eviction_time_ < MIN_EVICTION_INTERVAL) {
    return false;
  }

  size_t current_vram_bytes = getGPUMemoryUsage();
  size_t budget_bytes = max_vram_budget_mb_ * 1024 * 1024;
  float vram_ratio = static_cast<float>(current_vram_bytes) / budget_bytes;

  return vram_ratio > VRAM_EVICTION_THRESHOLD;
}

bool ChunkManager::shouldBlockNewLoads() {
  size_t current_vram_bytes = getGPUMemoryUsage();
  size_t budget_bytes = max_vram_budget_mb_ * 1024 * 1024;
  float vram_ratio = static_cast<float>(current_vram_bytes) / budget_bytes;

  return vram_ratio > VRAM_EMERGENCY_THRESHOLD;
}

// Calculate how many chunks to evict based on VRAM pressure int
int ChunkManager::calculateEvictionCount() {
  size_t current_vram_bytes = getGPUMemoryUsage();
  size_t budget_bytes = max_vram_budget_mb_ * 1024 * 1024;
  float vram_ratio = static_cast<float>(current_vram_bytes) / budget_bytes;

  int current_chunks = 0;
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    for (const auto& [coord, metadata] : chunk_metadata_) {
      if (metadata.state.load() == ChunkState::ACTIVE ||
          metadata.state.load() == ChunkState::OPTIMIZING) {
        current_chunks++;
      }
    }
  }

  if (current_chunks == 0) return 0;

  // Calculate target VRAM after eviction
  float target_ratio = (vram_ratio > VRAM_AGGRESSIVE_THRESHOLD)
                           ? VRAM_TARGET_AFTER_AGGRESSIVE
                           : VRAM_TARGET_AFTER_EVICTION;

  size_t target_vram_bytes = static_cast<size_t>(budget_bytes * target_ratio);

  // How much VRAM do we need to free?
  if (current_vram_bytes <= target_vram_bytes) {
    return 0;  // Shouldn't happen, but safety check
  }

  size_t vram_to_free = current_vram_bytes - target_vram_bytes;

  // Estimate VRAM per chunk from current state
  float current_vram_per_chunk =
      static_cast<float>(current_vram_bytes) / current_chunks;

  // Calculate chunks to evict
  int chunks_to_evict =
      static_cast<int>(std::ceil(vram_to_free / current_vram_per_chunk));

  // Safety bounds
  chunks_to_evict = std::max(1, chunks_to_evict);  // At least 1
  chunks_to_evict =
      std::min(chunks_to_evict, current_chunks / 2);  // At most half

  // Log the decision
  float current_vram_mb = current_vram_bytes / (1024.0f * 1024.0f);
  float target_vram_mb = target_vram_bytes / (1024.0f * 1024.0f);
  float vram_per_chunk_mb = current_vram_per_chunk / (1024.0f * 1024.0f);

  std::cout << "VRAM Eviction: " << current_vram_mb << "MB/"
            << max_vram_budget_mb_ << "MB (" << (vram_ratio * 100.0f)
            << "%) -> target " << target_vram_mb << "MB, ~" << vram_per_chunk_mb
            << "MB/chunk, evicting " << chunks_to_evict << "/" << current_chunks
            << " chunks" << std::endl;

  return chunks_to_evict;
}

float ChunkManager::getCurrentVramUsageRatio() const {
  size_t current_vram_bytes = getGPUMemoryUsage();
  size_t budget_bytes = max_vram_budget_mb_ * 1024 * 1024;
  return static_cast<float>(current_vram_bytes) / budget_bytes;
}

std::string ChunkManager::getVramStatus() {
  float ratio = getCurrentVramUsageRatio();
  size_t current_mb = getGPUMemoryUsage() / (1024 * 1024);

  int active_chunks = 0;
  {
    std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
    for (const auto& [coord, metadata] : chunk_metadata_) {
      if (metadata.state.load() == ChunkState::ACTIVE ||
          metadata.state.load() == ChunkState::OPTIMIZING) {
        active_chunks++;
      }
    }
  }

  return "VRAM: " + std::to_string(current_mb) + "MB/" +
         std::to_string(max_vram_budget_mb_) + "MB (" +
         std::to_string(ratio * 100.0f) + "%), " +
         std::to_string(active_chunks) + " chunks active";
}

bool ChunkManager::waitForVramAvailable(float max_usage_ratio,
                                        std::chrono::milliseconds timeout,
                                        const std::string& operation_name) {
  auto start_time = std::chrono::steady_clock::now();

  while (getCurrentVramUsageRatio() > max_usage_ratio) {
    // Check timeout
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      std::cout << "[Gaussian Mapper] VRAM wait timeout for " << operation_name
                << "! Current usage: " << (getCurrentVramUsageRatio() * 100.0f)
                << "%" << std::endl;
      return false;
    }

    // Log current status
    size_t current_vram_mb = getGPUMemoryUsage() / (1024 * 1024);
    std::cout << "[Gaussian Mapper] Waiting for VRAM for " << operation_name
              << ": " << current_vram_mb << "MB/" << max_vram_budget_mb_
              << "MB (" << (getCurrentVramUsageRatio() * 100.0f) << "%)"
              << std::endl;

    triggerLruCheck();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  return true;  // Success
}

bool ChunkManager::saveAllChunksSync() {
  std::unique_lock<std::mutex> lock(chunk_metadata_mutex_);
  for (const auto& [coord, meta] : chunk_metadata_) {
    auto state = meta.state.load();

    if (state == ChunkState::ACTIVE) {
      if (!saveChunkSync(coord)) {
        throw std::runtime_error(
            "Failed to save chunk: " + std::to_string(coord.x) + "," +
            std::to_string(coord.y) + "," + std::to_string(coord.z));
      }
    } else if (state == ChunkState::OPTIMIZING) {
      std::cout << "[ChunkManager] WARNING: Chunk (" << coord.x << ","
                << coord.y << "," << coord.z
                << ") is in state OPTIMIZING, have to set to ACTIVE first "
                   "before saving."
                << std::endl;
      transitionChunkState(coord, ChunkState::OPTIMIZING, ChunkState::ACTIVE);
      if (!saveChunkSync(coord)) {
        throw std::runtime_error(
            "Failed to save chunk: " + std::to_string(coord.x) + "," +
            std::to_string(coord.y) + "," + std::to_string(coord.z));
      }
    }
  }
  return true;
}