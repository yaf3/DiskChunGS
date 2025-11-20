#include "model/gaussian_model.h"

void GaussianModel::saveAndEvictChunks(const torch::Tensor& chunk_ids) {
  if (chunk_ids.size(0) == 0) return;

  // Mask of requested that are loaded
  torch::Tensor loaded_mask = torch::isin(chunk_ids, chunks_loaded_from_disk_);

  // IDs of requested that are loaded
  torch::Tensor chunks_to_save = chunk_ids.index({loaded_mask});

  // Save updated chunks that have been loaded before
  if (chunks_to_save.size(0) > 0) {
    saveChunks(chunks_to_save);
  }

  // IDs of all gaussian's chunks in memory
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  // Mask of requested chunks that have gaussians in memory
  torch::Tensor has_gaussians_mask = torch::isin(chunk_ids, spatial_chunks);

  // Requested IDs of chunks that aren't loaded and have gaussians in memory
  torch::Tensor non_loaded_with_gaussians =
      chunk_ids.index({has_gaussians_mask & (~loaded_mask)});

  if (non_loaded_with_gaussians.size(0) > 0) {
    // Distinguish spillover vs new chunks

    // Mask of requested chunks that aren't loaded and have gaussians in memory
    // and are saved on disk
    torch::Tensor is_spillover_mask =
        torch::isin(non_loaded_with_gaussians, chunks_on_disk_);

    // IDs of requested chunks that aren't loaded and have gaussians in memory
    // and are saved on disk
    torch::Tensor spillover_chunks =
        non_loaded_with_gaussians.index({is_spillover_mask});

    // IDs of requested chunks that aren't loaded and have gaussians in memory
    // and aren't saved on disk
    torch::Tensor new_chunks =
        non_loaded_with_gaussians.index({~is_spillover_mask});

    // Handle spillover chunks - discard without saving
    if (spillover_chunks.size(0) > 0) {
      // std::cout << "[Eviction] Discarding " << spillover_chunks.size(0)
      //           << " spillover-only chunks" << std::endl;
    }

    // Handle new chunks - save them!
    if (new_chunks.size(0) > 0) {
      // std::cout << "[Eviction] Saving " << new_chunks.size(0) << " new
      // chunks"
      //           << std::endl;
      saveChunks(new_chunks);  // Save the new chunks to disk
    }
  }

  // Remove all gaussians from evicted chunks. We have saved previously loaded
  // and new gaussians, only spillover gaussians remain. These are negligible
  torch::Tensor remove_mask = torch::isin(gaussian_chunk_ids_, chunk_ids);
  if (remove_mask.sum().item<int>() > 0) {
    prunePoints(remove_mask);
  }

  // Update tracking

  // Mask of loaded chunks not in the requested chunks to save
  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, chunk_ids);

  // Remove requested chunk IDs that were in chunks_loaded_from_disk_
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});
}

size_t GaussianModel::getCurrentGPUMemoryUsage() const {
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

void GaussianModel::checkMemoryPressure() {
  auto now = std::chrono::steady_clock::now();
  // if (now - last_memory_check_ < std::chrono::milliseconds(100)) {
  //   return;
  // }
  // last_memory_check_ = now;

  int current_gaussians = getXYZ().size(0);

  if (current_gaussians <= max_gaussians_in_memory_) {
    return;  // No pressure, exit early
  }

  // Keep evicting until we reach our memory goal or run out of chunks
  while (true) {
    current_gaussians = getXYZ().size(0);
    if (current_gaussians <= max_gaussians_in_memory_) {
      // Goal reached, exit the loop
      break;
    }

    // Get chunks that can be evicted (any chunk with gaussians in memory)
    torch::Tensor evictable_chunks =
        std::get<0>(torch::_unique2(gaussian_chunk_ids_));

    if (evictable_chunks.size(0) == 0) {
      // No chunks found to evict, break to avoid infinite loop
      std::cout << "[Memory] Warning: No evictable chunks found" << std::endl;
      break;
    }

    // Calculate how many gaussians to evict this iteration
    int64_t excess_gaussians = current_gaussians - max_gaussians_in_memory_;
    int64_t gaussians_to_evict =
        std::max(excess_gaussians,
                 static_cast<int64_t>(100000));  // Minimum 100k per iteration

    // Get LRU chunks that total at least gaussians_to_evict
    torch::Tensor lru_chunks =
        findLRUChunks(evictable_chunks, gaussians_to_evict);

    if (lru_chunks.size(0) == 0) {
      std::cout << "[Memory] Warning: No LRU chunks found to evict"
                << std::endl;
      break;
    }

    // std::cout << "[Memory] Evicting " << lru_chunks.size(0)
    //           << " LRU chunks (current Gaussians: " << current_gaussians
    //           << " target: " << max_gaussians_in_memory_ << std::endl;

    // Use updated saveAndEvictChunks
    saveAndEvictChunks(lru_chunks);
  }
}

torch::Tensor GaussianModel::findLRUChunks(
    const torch::Tensor& candidate_chunks,
    int64_t target_gaussian_count) {
  if (candidate_chunks.size(0) == 0 || target_gaussian_count <= 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  }

  auto chunks_cpu = candidate_chunks.cpu();
  auto chunks_accessor = chunks_cpu.accessor<int64_t, 1>();

  // Create vector of (chunk_id, access_time, gaussian_count) tuples
  std::vector<std::tuple<int64_t, float, int64_t>> chunk_data;
  for (int64_t i = 0; i < chunks_cpu.size(0); i++) {
    int64_t chunk_id = chunks_accessor[i];
    float access_time = chunk_access_times_.count(chunk_id)
                            ? chunk_access_times_[chunk_id]
                            : 0.0f;

    // Get gaussian count for this chunk
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    int64_t gaussian_count = chunk_mask.sum().item<int64_t>();

    chunk_data.emplace_back(chunk_id, access_time, gaussian_count);
  }

  // Sort by access time (oldest first)
  std::sort(chunk_data.begin(), chunk_data.end(),
            [](const auto& a, const auto& b) {
              return std::get<1>(a) < std::get<1>(b);
            });

  // Debug output
  if (!chunk_data.empty()) {
    float oldest_time = std::get<1>(chunk_data.front());
    float newest_time = std::get<1>(chunk_data.back());
    float time_delta = newest_time - oldest_time;
    // std::cout << "[LRU DEBUG] After sorting (oldest first):" << std::endl;
    // std::cout << "[LRU DEBUG] Time range: oldest=" << oldest_time
    //           << ", newest=" << newest_time << ", delta=" << time_delta <<
    //           "ms"
    //           << std::endl;
  }

  // Accumulate chunks until we reach target gaussian count
  std::vector<int64_t> selected_chunks;
  int64_t accumulated_gaussians = 0;

  for (const auto& [chunk_id, access_time, gaussian_count] : chunk_data) {
    selected_chunks.push_back(chunk_id);
    accumulated_gaussians += gaussian_count;

    if (accumulated_gaussians >= target_gaussian_count) {
      break;
    }
  }

  // std::cout << "[LRU DEBUG] Selected " << selected_chunks.size() << " chunks
  // ("
  //           << accumulated_gaussians << " gaussians) to reach target
  //           eviction"
  //           << target_gaussian_count << std::endl;

  // Convert to tensor
  torch::Tensor result = torch::empty(
      {static_cast<int64_t>(selected_chunks.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto result_accessor = result.accessor<int64_t, 1>();

  for (size_t i = 0; i < selected_chunks.size(); i++) {
    result_accessor[i] = selected_chunks[i];
  }

  return result.to(device_type_);
}

void GaussianModel::testSaveLoadEvictCycle() {
  std::cout << "\n=== STARTING VECTORIZED SAVE/LOAD/EVICT CYCLE TEST ==="
            << std::endl;

  if (!is_initialized_) {
    std::cout << "ERROR: Model not initialized, cannot run test" << std::endl;
    return;
  }

  auto test_start = std::chrono::steady_clock::now();

  // Step 1: Record initial state
  int initial_gaussians = xyz_.size(0);
  size_t initial_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

  std::cout << "INITIAL STATE:" << std::endl;
  std::cout << "  Gaussians: " << initial_gaussians << std::endl;
  std::cout << "  Memory: " << initial_memory_mb << "MB" << std::endl;
  std::cout << "  Loaded chunks: " << chunks_loaded_from_disk_.size(0)
            << std::endl;
  std::cout << "  Chunks on disk: " << chunks_on_disk_.size(0) << std::endl;

  // Step 2: Get all unique chunks currently in model (spatial chunks)
  torch::Tensor all_spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  std::cout << "  Spatial chunks: " << all_spatial_chunks.size(0) << std::endl;
  std::cout << "  Loaded chunks: " << chunks_loaded_from_disk_.size(0)
            << std::endl;

  if (all_spatial_chunks.size(0) == 0) {
    std::cout << "ERROR: No chunks found in model" << std::endl;
    return;
  }

  // Step 3: Evict all chunks from memory (vectorized)
  std::cout << "\nEVICTING ALL CHUNKS (VECTORIZED)..." << std::endl;
  auto evict_start = std::chrono::steady_clock::now();

  // Store the chunks we're testing so we only reload exactly these
  torch::Tensor chunks_to_test = all_spatial_chunks.clone();

  try {
    // Force evict all spatial chunks using vectorized operations
    saveAndEvictChunks(all_spatial_chunks);

    auto evict_end = std::chrono::steady_clock::now();
    auto evict_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(evict_end -
                                                              evict_start)
            .count();

    int remaining_gaussians = xyz_.size(0);
    size_t remaining_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Vectorized eviction completed in " << evict_duration_ms
              << "ms" << std::endl;
    std::cout << "  Remaining gaussians: " << remaining_gaussians << std::endl;
    std::cout << "  Remaining memory: " << remaining_memory_mb << "MB"
              << std::endl;
    std::cout << "  Loaded chunks after eviction: "
              << chunks_loaded_from_disk_.size(0) << std::endl;
    std::cout << "  Chunks on disk after eviction: " << chunks_on_disk_.size(0)
              << std::endl;
    std::cout << "  Memory freed: " << (initial_memory_mb - remaining_memory_mb)
              << "MB" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during vectorized eviction: " << e.what() << std::endl;
    return;
  }

  // Step 4: Reload all chunks from disk (vectorized)
  std::cout << "\nRELOADING ALL CHUNKS (VECTORIZED)..." << std::endl;
  auto load_start = std::chrono::steady_clock::now();

  try {
    // Load only the chunks we originally evicted, not all chunks on disk
    // (chunks_on_disk_ may contain additional chunks saved during memory
    // pressure)
    if (chunks_to_test.size(0) > 0) {
      loadChunks(chunks_to_test);
    } else {
      std::cout << "  WARNING: No chunks to reload!" << std::endl;
    }

    auto load_end = std::chrono::steady_clock::now();
    auto load_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                              load_start)
            .count();

    int final_gaussians = xyz_.size(0);
    size_t final_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Vectorized load completed in " << load_duration_ms << "ms"
              << std::endl;
    std::cout << "  Final gaussians: " << final_gaussians << std::endl;
    std::cout << "  Final memory: " << final_memory_mb << "MB" << std::endl;
    std::cout << "  Loaded chunks after reload: "
              << chunks_loaded_from_disk_.size(0) << std::endl;
    std::cout << "  Spatial chunks after reload: "
              << std::get<0>(torch::_unique2(gaussian_chunk_ids_)).size(0)
              << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during vectorized load: " << e.what() << std::endl;
    return;
  }

  // Step 5: Validation and summary
  auto test_end = std::chrono::steady_clock::now();
  auto total_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(test_end -
                                                            test_start)
          .count();

  // Check if we recovered all gaussians
  bool gaussians_match = (xyz_.size(0) == initial_gaussians);
  bool chunks_properly_loaded = (chunks_loaded_from_disk_.size(0) > 0);

  std::cout << "\n=== VECTORIZED TEST SUMMARY ===" << std::endl;
  std::cout << "  Total time: " << total_duration_ms << "ms" << std::endl;
  std::cout << "  Gaussians: " << initial_gaussians << " -> " << xyz_.size(0)
            << " (" << (gaussians_match ? "MATCH" : "MISMATCH") << ")"
            << std::endl;
  std::cout << "  Memory: " << initial_memory_mb << "MB -> "
            << (getCurrentGPUMemoryUsage() / (1024 * 1024)) << "MB"
            << std::endl;
  std::cout << "  Chunks properly loaded: "
            << (chunks_properly_loaded ? "YES" : "NO") << std::endl;

  // Additional validation
  if (!gaussians_match) {
    std::cout << "  WARNING: Gaussian count mismatch - possible data loss!"
              << std::endl;
  }

  if (!chunks_properly_loaded) {
    std::cout << "  WARNING: No chunks marked as loaded - state tracking issue!"
              << std::endl;
  }

  if (gaussians_match && chunks_properly_loaded) {
    std::cout << "  ✅ TEST PASSED: Save/Load/Evict cycle successful"
              << std::endl;
  } else {
    std::cout << "  ❌ TEST FAILED: Issues detected in save/load cycle"
              << std::endl;
  }

  std::cout << "=== VECTORIZED SAVE/LOAD/EVICT CYCLE TEST COMPLETE ===\n"
            << std::endl;
}

void GaussianModel::updateChunkAccess(const torch::Tensor& accessed_chunk_ids) {
  if (accessed_chunk_ids.size(0) == 0) return;

  float current_time = std::chrono::duration<float>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();

  auto chunk_ids_cpu = accessed_chunk_ids.cpu();
  auto chunk_ids_accessor = chunk_ids_cpu.accessor<int64_t, 1>();

  for (int64_t i = 0; i < chunk_ids_cpu.size(0); i++) {
    int64_t chunk_id = chunk_ids_accessor[i];
    chunk_access_times_[chunk_id] = current_time;
  }
}