#include "model/gaussian_model.h"
#include "rendering/gaussian_rasterizer.h"

void GaussianModel::loadChunks(const torch::Tensor& chunk_id_requests) {
  torch::NoGradGuard no_grad;
  if (chunk_id_requests.size(0) == 0) return;

  // Requested that are on disk
  torch::Tensor on_disk_mask = torch::isin(chunk_id_requests, chunks_on_disk_);

  // Requested that aren't loaded from disk
  torch::Tensor not_loaded_mask =
      ~torch::isin(chunk_id_requests, chunks_loaded_from_disk_);

  // Requested that are on disk and not loaded from disk
  torch::Tensor on_disk_not_loaded_mask = on_disk_mask & not_loaded_mask;

  // IDs of requested that are on disk but not loaded from disk
  torch::Tensor chunks_ids_needing_load =
      chunk_id_requests.index({on_disk_not_loaded_mask});

  // Filter out chunks that are already loaded
  if (chunks_ids_needing_load.size(0) == 0) {
    // std::cout << "[Load] All requested chunks already loaded" << std::endl;

    // Even if nothing needs loading, check if we're over the limit
    int64_t current_gaussians = xyz_.size(0);
    if (current_gaussians > max_gaussians_in_memory_) {
      int64_t excess = current_gaussians - max_gaussians_in_memory_;
      std::cout << "[Load] All chunks loaded but over limit by " << excess
                << " gaussians (have " << current_gaussians << ", max "
                << max_gaussians_in_memory_ << ")" << std::endl;

      // Get all spatial chunks
      torch::Tensor spatial_chunks =
          std::get<0>(torch::_unique2(gaussian_chunk_ids_));
      if (spatial_chunks.size(0) > 0) {
        // Evict chunks that are NOT in the requested (visible) set
        torch::Tensor evictable_mask =
            ~torch::isin(spatial_chunks, chunk_id_requests);
        torch::Tensor evictable_chunks = spatial_chunks.index({evictable_mask});

        if (evictable_chunks.size(0) > 0) {
          // Add hysteresis buffer to reduce eviction frequency
          int64_t buffer =
              static_cast<int64_t>(max_gaussians_in_memory_ * 0.05f);
          int64_t target_eviction = excess + buffer;
          std::cout << "[Load] Evicting with hysteresis: excess=" << excess
                    << ", buffer=" << buffer << ", target=" << target_eviction
                    << std::endl;

          torch::Tensor lru_chunks =
              findLRUChunks(evictable_chunks, target_eviction);
          if (lru_chunks.size(0) > 0) {
            saveAndEvictChunks(lru_chunks);
            std::cout << "[Load] Evicted non-visible chunks, new count: "
                      << xyz_.size(0) << std::endl;
          }
        }
      }
    }

    return;
  }

  // STEP 0: Pre-emptive eviction using exact gaussian counts
  if (chunks_ids_needing_load.any().item<bool>()) {
    // Find indices of loadable chunks in chunks_on_disk_
    int64_t exact_gaussians_to_load = 0;
    auto to_load_cpu = chunks_ids_needing_load.cpu();
    auto chunks_on_disk_cpu = chunks_on_disk_.cpu();
    auto counts_cpu = chunk_gaussian_counts_.cpu();

    auto to_load_accessor = to_load_cpu.accessor<int64_t, 1>();
    auto chunks_on_disk_accessor = chunks_on_disk_cpu.accessor<int64_t, 1>();
    auto counts_accessor = counts_cpu.accessor<int64_t, 1>();

    // Iterate through chunks needing load
    for (int64_t i = 0; i < to_load_accessor.size(0); i++) {
      int64_t to_load_chunk_id = to_load_accessor[i];

      // Find this chunk in chunks_on_disk_
      for (int64_t j = 0; j < chunks_on_disk_cpu.size(0); j++) {
        if (chunks_on_disk_accessor[j] == to_load_chunk_id) {
          exact_gaussians_to_load += counts_accessor[j];
          break;  // Found it, move to next requested chunk
        }
      }
    }

    int64_t current_gaussians = xyz_.size(0);
    int64_t projected_total = current_gaussians + exact_gaussians_to_load;

    // std::cout << "[Load] Planning to load " <<
    // chunks_ids_needing_load.size(0)
    //           << " chunks (exactly " << exact_gaussians_to_load << "
    //           gaussians)"
    //           << std::endl;

    if (projected_total > max_gaussians_in_memory_) {
      int64_t excess = projected_total - max_gaussians_in_memory_;
      // std::cout << "[Load] Pre-emptive eviction needed: current="
      //           << current_gaussians << ", incoming=" <<
      //           exact_gaussians_to_load
      //           << ", projected=" << projected_total << ", excess=" << excess
      //           << std::endl;

      // Get evictable chunks and find LRU ones to free up 'excess' gaussians
      torch::Tensor spatial_chunks =
          std::get<0>(torch::_unique2(gaussian_chunk_ids_));
      if (spatial_chunks.size(0) > 0) {
        // Exclude chunks we're trying to load from eviction candidates
        torch::Tensor evictable_mask =
            ~torch::isin(spatial_chunks, chunks_ids_needing_load);
        torch::Tensor evictable_chunks = spatial_chunks.index({evictable_mask});

        if (evictable_chunks.size(0) > 0) {
          // Add hysteresis buffer to reduce eviction frequency
          int64_t buffer =
              static_cast<int64_t>(max_gaussians_in_memory_ * 0.05f);
          int64_t target_eviction = excess + buffer;

          torch::Tensor lru_chunks =
              findLRUChunks(evictable_chunks, target_eviction);
          if (lru_chunks.size(0) > 0) {
            saveAndEvictChunks(lru_chunks);
          } else {
            throw std::runtime_error(
                "No evictable chunks found after excluding protected chunks");
          }
        } else {
          throw std::runtime_error(
              "No evictable chunks available - all chunks are protected");
        }
      } else {
        throw std::runtime_error(
            "Literally no chunks exist but we need to evict?");
      }
    }
  }

  // Parallel load from disk
  if (chunks_ids_needing_load.size(0) > 0) {
    // Remove spillover first
    // torch::Tensor spillover_mask =
    //     torch::isin(gaussian_chunk_ids_, loadable_chunks);
    // int spillover_count = spillover_mask.sum().item<int>();

    // if (spillover_count > 0) {
    //   std::cout << "[Load] Removing " << spillover_count
    //             << " spillover gaussians before loading "
    //             << loadable_chunks.size(0) << " chunks from disk" <<
    //             std::endl;
    //   prunePoints(spillover_mask);
    // }

    // PARALLEL LOADING
    auto chunks_ids_needing_load_cpu = chunks_ids_needing_load.cpu();
    auto accessor = chunks_ids_needing_load_cpu.accessor<int64_t, 1>();
    int num_chunks = chunks_ids_needing_load_cpu.size(0);

    std::vector<std::future<std::pair<int64_t, std::optional<ChunkData>>>>
        futures;

    // Launch parallel load tasks
    for (int i = 0; i < num_chunks; ++i) {
      int64_t chunk_id = accessor[i];

      auto future = std::async(std::launch::async, [this, chunk_id]() {
        return std::make_pair(chunk_id, loadSingleChunkFromDisk(chunk_id));
      });

      futures.push_back(std::move(future));
    }

    // Collect results
    std::vector<GaussianModel::ChunkData> chunks_to_append;
    std::vector<int64_t> loaded_chunk_ids;

    for (auto& future : futures) {
      auto [chunk_id, chunk_data] = future.get();
      if (chunk_data.has_value()) {
        chunks_to_append.push_back(chunk_data.value());
        loaded_chunk_ids.push_back(chunk_id);
      }
    }

    // Append results
    if (!chunks_to_append.empty()) {
      appendLoadedChunks(chunks_to_append, loaded_chunk_ids);
    }

    // Mark as loaded
    chunks_loaded_from_disk_ =
        torch::cat({chunks_loaded_from_disk_, chunks_ids_needing_load}, 0);

    // Clean up duplicates
    chunks_loaded_from_disk_ =
        std::get<0>(torch::_unique2(chunks_loaded_from_disk_));
  }
}

void GaussianModel::saveSingleChunkToDisk(int64_t chunk_id,
                                          const ChunkData& chunk_data) {
  std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

  // Create directory if it doesn't exist
  std::filesystem::path chunk_path(chunk_filename);
  std::filesystem::create_directories(chunk_path.parent_path());

  std::ofstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + chunk_filename);
  }

  try {
    // Write magic number and version for validation
    uint32_t magic = 0x43484E4B;  // "CHNK" in hex
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write chunk metadata
    file.write(reinterpret_cast<const char*>(&chunk_id), sizeof(chunk_id));
    uint32_t num_points = static_cast<uint32_t>(chunk_data.num_points);
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));

    // Save tensors in same order as loading
    saveTensorBinary(chunk_data.xyz, file);
    saveTensorBinary(chunk_data.features_dc, file);
    saveTensorBinary(chunk_data.features_rest, file);
    saveTensorBinary(chunk_data.scaling, file);
    saveTensorBinary(chunk_data.rotation, file);
    saveTensorBinary(chunk_data.opacity, file);
    saveTensorBinary(chunk_data.exist_since, file);
    saveTensorBinary(chunk_data.position_lrs, file);
    saveTensorBinary(chunk_data.gaussian_ids, file);

    // Save optimizer states
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      // Write step count
      file.write(
          reinterpret_cast<const char*>(&chunk_data.step_counts[group_idx]),
          sizeof(int64_t));

      // Save momentum tensors
      if (chunk_data.exp_avg_states[group_idx].defined()) {
        saveTensorBinary(chunk_data.exp_avg_states[group_idx], file);
        saveTensorBinary(chunk_data.exp_avg_sq_states[group_idx], file);
      } else {
        // Save empty tensors as placeholders
        torch::Tensor empty = torch::empty({0});
        saveTensorBinary(empty, file);
        saveTensorBinary(empty, file);
      }
    }

    file.close();
    // std::cout << "Saved chunk " << chunk_id << " with " << num_points
    //           << " points to " << chunk_filename << std::endl;

  } catch (const std::exception& e) {
    file.close();
    std::filesystem::remove(chunk_filename);  // Clean up partial file
    throw std::runtime_error("Failed to save chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

// Add this to your loadSingleChunkFromDisk function to isolate bottlenecks:

std::optional<GaussianModel::ChunkData> GaussianModel::loadSingleChunkFromDisk(
    int64_t chunk_id) {
  ChunkCoord chunk_coord = decodeChunkCoord(chunk_id);
  std::string chunk_filename = getChunkFilename(chunk_coord);

  if (!std::filesystem::exists(chunk_filename)) {
    throw std::runtime_error("Chunk file does not exist: " + chunk_filename);
    return std::nullopt;
  }

  // Use your existing memory-mapped loading
  std::ifstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + chunk_filename);
    return std::nullopt;
  }

  // Validate magic number and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));

  if (magic != 0x43484E4B) {
    throw std::runtime_error("Invalid checkpoint file format");
  }

  int64_t stored_chunk_id;
  uint32_t stored_num_points;

  file.read(reinterpret_cast<char*>(&stored_chunk_id), sizeof(stored_chunk_id));
  file.read(reinterpret_cast<char*>(&stored_num_points),
            sizeof(stored_num_points));

  if (stored_chunk_id != chunk_id) {
    throw std::runtime_error("Saved chunk ID does not match requested ID");
  }

  ChunkData data;
  try {
    // Load tensors in the same order as saved
    data.xyz = loadTensorBinary(file);
    data.features_dc = loadTensorBinary(file);
    data.features_rest = loadTensorBinary(file);
    data.scaling = loadTensorBinary(file);
    data.rotation = loadTensorBinary(file);
    data.opacity = loadTensorBinary(file);
    data.exist_since = loadTensorBinary(file);
    data.position_lrs = loadTensorBinary(file);
    data.gaussian_ids = loadTensorBinary(file);

    // Load optimizer states
    data.exp_avg_states.resize(6);
    data.exp_avg_sq_states.resize(6);
    data.step_counts.resize(6);

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      // Load step count
      file.read(reinterpret_cast<char*>(&data.step_counts[group_idx]),
                sizeof(int64_t));

      // Load momentum tensors
      data.exp_avg_states[group_idx] = loadTensorBinary(file);
      data.exp_avg_sq_states[group_idx] = loadTensorBinary(file);
    }

    data.num_points = data.xyz.size(0);

    file.close();

    // Validate loaded data
    if (data.num_points != static_cast<int>(stored_num_points)) {
      std::cerr << "Point count mismatch in chunk file: " << chunk_filename
                << std::endl;
      return std::nullopt;
    }

    // Create chunk IDs tensor (all points belong to this chunk)
    data.chunk_id = chunk_id;

    // std::cout << "Loaded chunk " << chunk_id << " with " << data.num_points
    //           << " points from " << chunk_filename << std::endl;

    return data;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load chunk " << chunk_id << ": " << e.what()
              << std::endl;
    throw std::runtime_error("Failed to load chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
    return std::nullopt;
  }
}

void GaussianModel::saveTensorBinary(const torch::Tensor& tensor,
                                     std::ofstream& file) {
  TensorHeader header = {};
  header.dims = tensor.dim();

  for (int i = 0; i < tensor.dim(); ++i) {
    header.sizes[i] = static_cast<uint32_t>(tensor.size(i));
  }
  header.dtype = static_cast<uint32_t>(tensor.scalar_type());
  header.data_size = tensor.nbytes();

  // Write header
  file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Move tensor to CPU if needed and write data
  torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
  file.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr()),
             header.data_size);
}

torch::Tensor GaussianModel::loadTensorBinary(std::ifstream& file) {
  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));

  // Reconstruct tensor sizes
  std::vector<int64_t> sizes(header.dims);
  for (uint32_t i = 0; i < header.dims; ++i) {
    sizes[i] = header.sizes[i];
  }

  // Create tensor with correct type and device
  torch::TensorOptions options =
      torch::TensorOptions()
          .dtype(static_cast<torch::ScalarType>(header.dtype))
          .device(device_type_);

  torch::Tensor tensor = torch::empty(sizes, options.device(torch::kCPU));

  // Read data
  file.read(reinterpret_cast<char*>(tensor.data_ptr()), header.data_size);

  // Move to target device if needed
  return tensor.to(device_type_);
}

void GaussianModel::appendLoadedChunks(
    const std::vector<ChunkData>& chunks_data,
    const std::vector<int64_t>& chunk_ids) {
  torch::NoGradGuard no_grad;
  if (chunks_data.empty()) return;

  // Concatenate all chunk data
  std::vector<torch::Tensor> all_xyz, all_features_dc, all_features_rest;
  std::vector<torch::Tensor> all_scaling, all_rotation, all_opacity;
  std::vector<torch::Tensor> all_exist_since, all_chunk_ids, all_position_lrs,
      all_gaussian_ids;

  // NEW: Concatenate optimizer states
  std::vector<std::vector<torch::Tensor>> all_exp_avg(6), all_exp_avg_sq(6);
  std::vector<int64_t> max_step_counts(6, 0);

  for (const auto& chunk : chunks_data) {
    all_xyz.push_back(chunk.xyz);
    all_features_dc.push_back(chunk.features_dc);
    all_features_rest.push_back(chunk.features_rest);
    all_scaling.push_back(chunk.scaling);
    all_rotation.push_back(chunk.rotation);
    all_opacity.push_back(chunk.opacity);
    all_exist_since.push_back(chunk.exist_since);
    all_position_lrs.push_back(chunk.position_lrs);
    all_gaussian_ids.push_back(chunk.gaussian_ids);

    torch::Tensor chunk_ids = torch::full(
        {chunk.num_points}, chunk.chunk_id,
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));
    all_chunk_ids.push_back(chunk_ids);

    // NEW: Collect optimizer states
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      if (chunk.exp_avg_states[group_idx].defined()) {
        all_exp_avg[group_idx].push_back(chunk.exp_avg_states[group_idx]);
        all_exp_avg_sq[group_idx].push_back(chunk.exp_avg_sq_states[group_idx]);
      }
      max_step_counts[group_idx] =
          std::max(max_step_counts[group_idx], chunk.step_counts[group_idx]);
    }
  }

  // Single concatenation operations
  torch::Tensor batch_xyz = torch::cat(all_xyz, 0);
  torch::Tensor batch_features_dc = torch::cat(all_features_dc, 0);
  torch::Tensor batch_features_rest = torch::cat(all_features_rest, 0);
  torch::Tensor batch_scaling = torch::cat(all_scaling, 0);
  torch::Tensor batch_rotation = torch::cat(all_rotation, 0);
  torch::Tensor batch_opacity = torch::cat(all_opacity, 0);
  torch::Tensor batch_exist_since = torch::cat(all_exist_since, 0);
  torch::Tensor batch_position_lrs = torch::cat(all_position_lrs, 0);
  torch::Tensor batch_chunk_ids = torch::cat(all_chunk_ids, 0);
  torch::Tensor batch_gaussian_ids = torch::cat(all_gaussian_ids, 0);

  // Concatenate optimizer states for each param group
  std::vector<torch::Tensor> concat_exp_avg(6), concat_exp_avg_sq(6);
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    if (!all_exp_avg[group_idx].empty()) {
      concat_exp_avg[group_idx] = torch::cat(all_exp_avg[group_idx], 0);
      concat_exp_avg_sq[group_idx] = torch::cat(all_exp_avg_sq[group_idx], 0);
    }
  }

  // Use densificationPostfix to append everything at once with optimizer states
  densificationPostfix(batch_xyz, batch_features_dc, batch_features_rest,
                       batch_opacity, batch_scaling, batch_rotation,
                       batch_exist_since, batch_chunk_ids, batch_position_lrs,
                       batch_gaussian_ids, concat_exp_avg, concat_exp_avg_sq,
                       max_step_counts);

  // std::cout << "Loaded " << batch_xyz.size(0) << " gaussians from "
  //           << chunks_data.size() << " chunks with full optimizer states"
  //           << std::endl;
}

void GaussianModel::saveChunks(const torch::Tensor& chunk_ids_to_save) {
  torch::NoGradGuard no_grad;
  if (chunk_ids_to_save.size(0) == 0) return;

  auto start_time = std::chrono::steady_clock::now();

  // std::cout << "[Chunk Save] Saving " << chunk_ids_to_save.size(0)
  //           << " chunks to disk" << std::endl;

  auto chunks_cpu = chunk_ids_to_save.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num_chunks = chunks_cpu.size(0);

  std::vector<std::pair<int64_t, ChunkData>> prepared_chunks;
  std::unordered_map<int64_t, int64_t> chunk_id_to_count;
  for (int i = 0; i < num_chunks; ++i) {
    int64_t chunk_id = accessor[i];
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    ChunkData chunk_data = extractChunkData(chunk_mask, chunk_id);
    chunk_id_to_count[chunk_id] = chunk_data.num_points;
    prepared_chunks.emplace_back(chunk_id, std::move(chunk_data));
  }

  // PARALLEL SAVING - NEW CODE
  std::vector<std::future<std::pair<int64_t, bool>>> futures;
  for (auto& [chunk_id, chunk_data] : prepared_chunks) {
    auto future = std::async(
        std::launch::async,
        [this](int64_t id, ChunkData data) {
          try {
            saveSingleChunkToDisk(id, data);  // Pure CPU/I/O work
            return std::make_pair(id, true);
          } catch (const std::exception& e) {
            std::cerr << "Failed to save chunk " << id << ": " << e.what()
                      << std::endl;
            return std::make_pair(id, false);
          }
        },
        chunk_id, std::move(chunk_data));

    futures.push_back(std::move(future));
  }

  // Collect results
  std::vector<int64_t> successfully_saved;
  for (auto& future : futures) {
    auto [chunk_id, success] = future.get();
    if (success) {
      successfully_saved.push_back(chunk_id);
    }
  }

  // Check if chunk_gaussian_counts_ and chunks_on_disk_ have gone out of sync
  if (chunk_gaussian_counts_.size(0) != chunks_on_disk_.size(0)) {
    throw std::runtime_error(
        "chunk_gaussian_counts_ doesn't match chunks_on_disk_ size!");
  }

  // Now add new chunk entries to chunks_on_disk_ and update
  // chunk_gaussian_counts_ for existing and new
  std::vector<int64_t> saved_counts;
  for (int64_t chunk_id : successfully_saved) {
    torch::Tensor chunk_id_tensor = torch::tensor(
        {chunk_id},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));
    torch::Tensor gaussian_count_tensor = torch::tensor(
        {chunk_id_to_count[chunk_id]},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));

    auto mask = torch::eq(chunks_on_disk_, chunk_id_tensor);
    bool found = torch::any(mask).item<bool>();

    // If we find existing entry, update the gaussian count
    if (found) {
      auto indices = torch::where(mask)[0];

      // If there are multiple entries with this chunk id, we messed up
      // somewhere
      if (indices.size(0) > 1) {
        throw std::runtime_error("chunks_on_disk_ has duplicates");
      }
      int64_t first_index = indices[0].item<int64_t>();
      chunk_gaussian_counts_[first_index] = chunk_id_to_count[chunk_id];

      // If first time observing this chunk id add it and the count
    } else {
      chunks_on_disk_ = torch::cat({chunks_on_disk_, chunk_id_tensor}, 0);
      chunk_gaussian_counts_ =
          torch::cat({chunk_gaussian_counts_, gaussian_count_tensor}, 0);
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // std::cout << "saveChunks completed in " << duration.count() << "ms"
  //           << std::endl;
}

GaussianModel::ChunkData GaussianModel::extractChunkData(
    const torch::Tensor& chunk_mask,
    int64_t chunk_id) {
  ChunkData data;

  // Extract main tensors
  data.xyz = xyz_.index({chunk_mask}).detach().clone();
  data.features_dc = features_dc_.index({chunk_mask}).detach().clone();
  data.features_rest = features_rest_.index({chunk_mask}).detach().clone();
  data.scaling = scaling_.index({chunk_mask}).detach().clone();
  data.rotation = rotation_.index({chunk_mask}).detach().clone();
  data.opacity = opacity_.index({chunk_mask}).detach().clone();
  data.exist_since = exist_since_iter_.index({chunk_mask}).detach().clone();
  data.position_lrs = position_lrs_.index({chunk_mask}).detach().clone();
  data.gaussian_ids = gaussian_ids_.index({chunk_mask}).detach().clone();
  data.num_points = data.xyz.size(0);
  data.chunk_id = chunk_id;

  // NEW: Extract optimizer states
  data.exp_avg_states.resize(6);
  data.exp_avg_sq_states.resize(6);
  data.step_counts.resize(6);

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& param_group = param_groups[group_idx];
    auto& param = param_group.params()[0];
    auto key = param.unsafeGetTensorImpl();

    if (state.find(key) != state.end()) {
      // Extract existing optimizer state
      auto& param_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);

      data.exp_avg_states[group_idx] =
          param_state.exp_avg().index({chunk_mask}).detach().clone();
      data.exp_avg_sq_states[group_idx] =
          param_state.exp_avg_sq().index({chunk_mask}).detach().clone();
      data.step_counts[group_idx] = param_state.step();
    } else {
      throw std::runtime_error("No param state found");
    }
  }

  return data;
}

void GaussianModel::saveAllChunks() {
  std::cout << "\n=== STARTING SAVE OF ALL CHUNKS IN MEMORY ===" << std::endl;

  // Get all chunks that have gaussians in memory
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  if (spatial_chunks.size(0) == 0) {
    std::cout << "No spatial chunks to save" << std::endl;
    return;
  }

  // Categorize chunks
  torch::Tensor loaded_chunks_mask =
      torch::isin(spatial_chunks, chunks_loaded_from_disk_);
  torch::Tensor spillover_chunks_mask =
      torch::isin(spatial_chunks, chunks_on_disk_) & (~loaded_chunks_mask);
  torch::Tensor new_chunks_mask =
      (~torch::isin(spatial_chunks, chunks_on_disk_)) & (~loaded_chunks_mask);

  torch::Tensor loaded_chunks = spatial_chunks.index({loaded_chunks_mask});
  torch::Tensor spillover_chunks =
      spatial_chunks.index({spillover_chunks_mask});
  torch::Tensor new_chunks = spatial_chunks.index({new_chunks_mask});

  std::cout << "Loaded chunks: " << loaded_chunks.size(0) << std::endl;
  std::cout << "Spillover chunks: " << spillover_chunks.size(0) << " (skipping)"
            << std::endl;
  std::cout << "New chunks: " << new_chunks.size(0) << std::endl;

  // Save loaded chunks (preserve existing data)
  if (loaded_chunks.size(0) > 0) {
    saveChunks(loaded_chunks);
  }

  // Save new chunks (preserve new data)
  if (new_chunks.size(0) > 0) {
    saveChunks(new_chunks);
  }

  // Don't save spillover chunks (would overwrite better disk data)

  std::cout << "=== SAVE COMPLETE ===" << std::endl;
}

int64_t GaussianModel::countAllGaussians() {
  // Count gaussians in memory
  int64_t in_memory_count = xyz_.size(0);

  // Count only unloaded gaussians on disk
  torch::Tensor unloaded_mask =
      ~torch::isin(chunks_on_disk_, chunks_loaded_from_disk_);
  torch::Tensor unloaded_counts = chunk_gaussian_counts_.index({unloaded_mask});
  int64_t disk_count = torch::sum(unloaded_counts).item<int64_t>();

  return in_memory_count + disk_count;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianModel::filterPointsByChunkDensity(const torch::Tensor& xyz,
                                          const torch::Tensor& colors,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& opacities,
                                          int min_gaussians_per_chunk) {
  if (min_gaussians_per_chunk <= 1) {
    // No filtering needed
    return std::make_tuple(xyz, colors, scales, opacities);
  }

  // Compute chunk IDs for all input points
  torch::Tensor chunk_ids = computeChunkIds(xyz, chunk_size_);

  // Count points per chunk using PyTorch operations
  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(chunk_ids, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  // Create mask for chunks that have enough points
  torch::Tensor valid_chunk_mask = counts >= min_gaussians_per_chunk;

  // Get the chunk IDs that pass the threshold
  torch::Tensor valid_chunk_ids = unique_chunks.index({valid_chunk_mask});

  if (valid_chunk_ids.size(0) == 0) {
    // No chunks have enough points, return empty tensors
    torch::Tensor empty_xyz = torch::empty({0, 3}, xyz.options());
    torch::Tensor empty_colors = torch::empty({0, 3}, colors.options());
    torch::Tensor empty_scales = scales.defined()
                                     ? torch::empty({0, 3}, scales.options())
                                     : torch::Tensor();
    torch::Tensor empty_opacities = torch::empty({0, 1}, opacities.options());
    return std::make_tuple(empty_xyz, empty_colors, empty_scales,
                           empty_opacities);
  }

  // Create mask for points that belong to valid chunks
  torch::Tensor point_valid_mask = torch::isin(chunk_ids, valid_chunk_ids);

  // Filter all tensors using the mask
  torch::Tensor filtered_xyz = xyz.index({point_valid_mask});
  torch::Tensor filtered_colors = colors.index({point_valid_mask});
  torch::Tensor filtered_scales =
      scales.defined() ? scales.index({point_valid_mask}) : torch::Tensor();
  torch::Tensor filtered_opacities = opacities.index({point_valid_mask});

  // Log statistics
  int total_chunks = unique_chunks.size(0);
  int valid_chunks = valid_chunk_ids.size(0);
  int total_points = xyz.size(0);
  int kept_points = filtered_xyz.size(0);

  // std::cout << "[Chunk Density Filter] Chunks: " << valid_chunks << "/"
  //           << total_chunks << " passed (need >= " <<
  //           min_gaussians_per_chunk
  //           << " points)" << std::endl;
  // std::cout << "[Chunk Density Filter] Points: " << kept_points << "/"
  //           << total_points << " kept ("
  //           << (100.0f * kept_points / total_points) << "%)" << std::endl;

  return std::make_tuple(filtered_xyz, filtered_colors, filtered_scales,
                         filtered_opacities);
}

void GaussianModel::handleBatchChunkRedistribution(
    const torch::Tensor& processed_chunk_ids) {
  torch::NoGradGuard no_grad;

  if (processed_chunk_ids.size(0) == 0) {
    return;  // No chunks to process
  }

  std::cout << "[Batch Redistribution] Processing "
            << processed_chunk_ids.size(0) << " chunks" << std::endl;

  // Step 1: Find all gaussians that were in any of the processed chunks
  torch::Tensor processed_chunk_mask =
      torch::zeros_like(gaussian_chunk_ids_, torch::kBool);

  // Create mask for all processed chunks at once
  for (int i = 0; i < processed_chunk_ids.size(0); i++) {
    int64_t chunk_id = processed_chunk_ids[i].item<int64_t>();
    processed_chunk_mask =
        processed_chunk_mask | (gaussian_chunk_ids_ == chunk_id);
  }

  torch::Tensor processed_indices = torch::where(processed_chunk_mask)[0];

  if (processed_indices.size(0) == 0) {
    std::cout << "[Batch Redistribution] No gaussians found in processed chunks"
              << std::endl;
    return;  // No gaussians in any of these chunks
  }

  std::cout << "[Batch Redistribution] Found " << processed_indices.size(0)
            << " gaussians across all processed chunks" << std::endl;

  // Step 2: Recompute actual chunk IDs based on current positions
  torch::Tensor processed_positions = xyz_.index({processed_indices});
  torch::Tensor actual_chunk_ids =
      computeChunkIds(processed_positions, chunk_size_);

  // Step 3: Find gaussians that have moved to different chunks
  torch::Tensor old_chunk_ids = gaussian_chunk_ids_.index({processed_indices});
  torch::Tensor moved_mask = (actual_chunk_ids != old_chunk_ids);

  if (!moved_mask.any().item<bool>()) {
    std::cout << "[Batch Redistribution] No gaussians moved from any "
                 "processed chunks"
              << std::endl;
    return;  // No redistributions needed
  }

  // Get indices of gaussians that moved
  torch::Tensor moved_indices_local = torch::where(moved_mask)[0];
  torch::Tensor moved_indices_global =
      processed_indices.index({moved_indices_local});
  torch::Tensor destination_chunk_ids =
      actual_chunk_ids.index({moved_indices_local});

  std::cout << "[Batch Redistribution] " << moved_indices_local.size(0)
            << " gaussians moved to different chunks" << std::endl;

  // Step 4: Get unique destination chunks that gaussians moved to
  torch::Tensor unique_destinations =
      std::get<0>(torch::_unique2(destination_chunk_ids));

  // Step 5: Pre-load destination chunks to prevent spillover classification
  if (unique_destinations.size(0) > 0) {
    std::cout << "[Batch Redistribution] Pre-loading "
              << unique_destinations.size(0)
              << " destination chunks to prevent spillover" << std::endl;
    loadChunks(unique_destinations);
  }

  // Step 6: CRITICAL - Recompute indices after loadChunks() as eviction may
  // have invalidated them
  torch::Tensor updated_processed_chunk_mask =
      torch::zeros_like(gaussian_chunk_ids_, torch::kBool);

  // Recreate mask for all processed chunks
  for (int i = 0; i < processed_chunk_ids.size(0); i++) {
    int64_t chunk_id = processed_chunk_ids[i].item<int64_t>();
    updated_processed_chunk_mask =
        updated_processed_chunk_mask | (gaussian_chunk_ids_ == chunk_id);
  }

  torch::Tensor updated_processed_indices =
      torch::where(updated_processed_chunk_mask)[0];

  if (updated_processed_indices.size(0) == 0) {
    std::cout << "[Batch Redistribution] WARNING: All gaussians from processed "
                 "chunks "
              << "were evicted during loading!" << std::endl;
    return;
  }

  // Step 7: Recompute which gaussians moved (using updated indices)
  torch::Tensor updated_processed_positions =
      xyz_.index({updated_processed_indices});
  torch::Tensor updated_actual_chunk_ids =
      computeChunkIds(updated_processed_positions, chunk_size_);
  torch::Tensor updated_old_chunk_ids =
      gaussian_chunk_ids_.index({updated_processed_indices});
  torch::Tensor updated_moved_mask =
      (updated_actual_chunk_ids != updated_old_chunk_ids);

  if (!updated_moved_mask.any().item<bool>()) {
    std::cout << "[Batch Redistribution] No gaussians moved after recomputation"
              << std::endl;
  }

  torch::Tensor updated_moved_indices_local =
      torch::where(updated_moved_mask)[0];
  torch::Tensor updated_moved_indices_global =
      updated_processed_indices.index({updated_moved_indices_local});
  torch::Tensor updated_destination_chunk_ids =
      updated_actual_chunk_ids.index({updated_moved_indices_local});

  std::cout << "[Batch Redistribution] After recomputation: "
            << updated_moved_indices_local.size(0)
            << " gaussians still need redistribution" << std::endl;

  // Step 8: Update gaussian_chunk_ids_ for moved gaussians (now safe!)
  gaussian_chunk_ids_.index_put_({updated_moved_indices_global},
                                 updated_destination_chunk_ids);
}

std::string GaussianModel::getChunkFilename(const ChunkCoord& coord) {
  // Using 'p' for positive and 'n' for negative prefixes
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));

  // Add .bin extension for the new binary format
  std::string filename = x_str + "_" + y_str + "_" + z_str + ".bin";

  // Use string concatenation with proper path separator
  return storage_base_path_ + "/" + filename;
}

void GaussianModel::updateChunkIDs() {
  gaussian_chunk_ids_ = computeChunkIds(getXYZ(), chunk_size_);
}