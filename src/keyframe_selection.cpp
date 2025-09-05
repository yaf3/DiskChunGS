#include "include/keyframe_selection.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// Constructor - modified to accept both loss_map and used_times_map
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t queue_size,
                             const std::map<std::size_t, float>* loss_map,
                             std::map<std::size_t, int>* used_times_map)
    : scene_(scene),
      queue_size_(queue_size),
      selection_radius_(100.0f),
      memory_management_radius_(100.0f),
      grid_cell_size_(50.0f),
      rng_(std::random_device{}()),
      loss_map_(loss_map),
      used_times_map_(used_times_map) {
  std::cout << "Created SpatialGridKeyframeQueue with:" << std::endl;
  std::cout << "  Selection radius: " << selection_radius_ << "m" << std::endl;
  std::cout << "  Memory management radius: " << memory_management_radius_
            << "m" << std::endl;
  std::cout << "  Grid cell size: " << grid_cell_size_ << "m" << std::endl;

  // Start the save worker thread
  save_worker_ = std::thread(&KeyframeQueue::saveWorker, this);
}

// Destructor
KeyframeQueue::~KeyframeQueue() {
  // Signal worker to stop
  {
    std::unique_lock<std::mutex> lock(save_mutex_);
    save_worker_stop_ = true;
  }
  save_cv_.notify_all();

  // Wait for worker to finish
  if (save_worker_.joinable()) {
    save_worker_.join();
  }
}

// Convert torch::Tensor to Eigen::Vector3f
Eigen::Vector3f KeyframeQueue::tensorToEigen(
    const torch::Tensor& tensor) const {
  // Ensure tensor is on CPU and contiguous
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();

  // Get pointer to data
  float* data_ptr = cpu_tensor.data_ptr<float>();

  return Eigen::Vector3f(data_ptr[0], data_ptr[1], data_ptr[2]);
}

// Get grid coordinate for a 3D position
ChunkCoord KeyframeQueue::getGridCoord(const Eigen::Vector3f& position) const {
  return getChunkCoord(position, grid_cell_size_);
}

// Get neighboring grid coordinates within a radius
std::vector<ChunkCoord> KeyframeQueue::getNeighborGridCoords(
    const ChunkCoord& center_coord,
    float radius) const {
  std::vector<ChunkCoord> neighbors;

  // Calculate how many grid cells we need to check in each direction
  int cell_radius = static_cast<int>(std::ceil(radius / grid_cell_size_)) + 1;

  // Iterate through all potentially relevant grid cells
  for (int dx = -cell_radius; dx <= cell_radius; ++dx) {
    for (int dy = -cell_radius; dy <= cell_radius; ++dy) {
      for (int dz = -cell_radius; dz <= cell_radius; ++dz) {
        ChunkCoord neighbor_coord{center_coord.x + dx, center_coord.y + dy,
                                  center_coord.z + dz};

        // Calculate the minimum distance from the search center to this grid
        // cell
        Eigen::Vector3f cell_center =
            getChunkCenter(neighbor_coord, grid_cell_size_);
        Eigen::Vector3f search_center =
            getChunkCenter(center_coord, grid_cell_size_);

        // Use the distance to the cell center as an approximation
        // This might include some cells that are slightly outside the radius,
        // but we'll do exact distance checking later
        float cell_distance = (cell_center - search_center).norm();
        float max_cell_distance =
            cell_distance + (grid_cell_size_ * std::sqrt(3.0f) / 2.0f);

        if (max_cell_distance <= radius) {
          neighbors.push_back(neighbor_coord);
        }
      }
    }
  }

  return neighbors;
}

// Get keyframes within radius of a center position
std::vector<std::shared_ptr<GaussianKeyframe>>
KeyframeQueue::getKeyframesInRadius(const Eigen::Vector3f& center_position,
                                    float radius) const {
  std::vector<std::shared_ptr<GaussianKeyframe>> result;

  // Get the grid coordinate for the center position
  ChunkCoord center_coord = getGridCoord(center_position);

  // Get all neighboring grid coordinates
  std::vector<ChunkCoord> neighbor_coords =
      getNeighborGridCoords(center_coord, radius);

  // Collect all keyframes from relevant grid cells
  std::unordered_set<std::size_t> candidate_keyframes;

  for (const auto& coord : neighbor_coords) {
    auto grid_it = spatial_grid_.find(coord);
    if (grid_it != spatial_grid_.end()) {
      for (std::size_t kf_id : grid_it->second) {
        candidate_keyframes.insert(kf_id);
      }
    }
  }

  // Now check exact distances and collect valid keyframes
  for (std::size_t kf_id : candidate_keyframes) {
    auto pos_it = keyframe_positions_.find(kf_id);
    if (pos_it != keyframe_positions_.end()) {
      float distance = (pos_it->second - center_position).norm();

      if (distance <= radius) {
        // Get the actual keyframe object (if it exists in scene)
        auto scene_kf_it = scene_->keyframes().find(kf_id);
        if (scene_kf_it != scene_->keyframes().end()) {
          result.push_back(scene_kf_it->second);
        }
      }
    }
  }

  return result;
}

// Add keyframe to the spatial grid (always keep in grid, regardless of memory
// status)
void KeyframeQueue::addKeyframeToGrid(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  // Get keyframe position
  torch::Tensor center_tensor = keyframe->getCenter();
  Eigen::Vector3f position = tensorToEigen(center_tensor);

  // Get grid coordinate
  ChunkCoord grid_coord = getGridCoord(position);

  // Add to spatial grid (always keep ALL keyframes in the grid)
  spatial_grid_[grid_coord].insert(keyframe->fid_);

  // Store position for quick access (always keep ALL positions)
  keyframe_positions_[keyframe->fid_] = position;

  // Mark as loaded (newly added keyframes are loaded)
  loaded_keyframes_.insert(keyframe->fid_);

  // Update latest keyframe
  latest_keyframe_ = keyframe;
}

// Update keyframe position in grid (if it moved)
void KeyframeQueue::updateKeyframeInGrid(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  // Get new position
  torch::Tensor center_tensor = keyframe->getCenter();
  Eigen::Vector3f new_position = tensorToEigen(center_tensor);

  // Check if position changed significantly
  auto pos_it = keyframe_positions_.find(keyframe->fid_);
  if (pos_it != keyframe_positions_.end()) {
    float distance_moved = (new_position - pos_it->second).norm();

    // Only update if moved more than half a grid cell
    if (distance_moved > grid_cell_size_ * 0.5f) {
      // Remove from old grid cell
      ChunkCoord old_coord = getGridCoord(pos_it->second);
      auto old_grid_it = spatial_grid_.find(old_coord);
      if (old_grid_it != spatial_grid_.end()) {
        old_grid_it->second.erase(keyframe->fid_);
        if (old_grid_it->second.empty()) {
          spatial_grid_.erase(old_grid_it);
        }
      }

      // Add to new grid cell
      ChunkCoord new_coord = getGridCoord(new_position);
      spatial_grid_[new_coord].insert(keyframe->fid_);

      // Update stored position
      keyframe_positions_[keyframe->fid_] = new_position;
    }
  } else {
    // New keyframe, just add it
    addKeyframeToGrid(keyframe);
  }
}

// Check if keyframe is currently loaded in memory
bool KeyframeQueue::isKeyframeLoaded(std::size_t keyframe_id) const {
  return loaded_keyframes_.find(keyframe_id) != loaded_keyframes_.end();
}

// Save keyframe to disk and mark as unloaded
void KeyframeQueue::saveKeyframeToDisk(std::size_t keyframe_id) {
  auto scene_kf_it = scene_->keyframes().find(keyframe_id);
  if (scene_kf_it != scene_->keyframes().end() &&
      scene_kf_it->second->allow_eviction_) {
    scene_kf_it->second->saving_ = true;
    // Queue for async saving
    queueForSaving(scene_kf_it->second);

    // Mark as unloaded
    loaded_keyframes_.erase(keyframe_id);

    std::cout << "Saved keyframe " << keyframe_id
              << " to disk (outside memory radius)" << std::endl;
  }
}

// Load keyframe from disk and mark as loaded
void KeyframeQueue::loadKeyframeFromDisk(std::size_t keyframe_id) {
  auto scene_kf_it = scene_->keyframes().find(keyframe_id);
  if (scene_kf_it != scene_->keyframes().end()) {
    auto keyframe = scene_kf_it->second;

    // Load data if not already loaded
    if (!keyframe->loaded_) {
      keyframe->loadDataFromDisk();
    }

    // Mark as loaded
    loaded_keyframes_.insert(keyframe_id);

    std::cout << "Loaded keyframe " << keyframe_id
              << " from disk (within memory radius)" << std::endl;
  }
}

// Manage keyframe memory based on distance from latest keyframe
void KeyframeQueue::manageKeyframeMemory() {
  if (!latest_keyframe_) return;

  torch::Tensor latest_center = latest_keyframe_->getCenter();
  Eigen::Vector3f latest_position = tensorToEigen(latest_center);

  // Get all keyframes within memory management radius using spatial grid
  std::vector<std::shared_ptr<GaussianKeyframe>> nearby_keyframes =
      getKeyframesInRadius(latest_position, memory_management_radius_);

  // Create a set for fast lookup of keyframes that should be loaded
  std::unordered_set<std::size_t> should_be_loaded;
  for (const auto& kf : nearby_keyframes) {
    should_be_loaded.insert(kf->fid_);
  }

  // Collect keyframes that need to be unloaded
  std::vector<std::size_t> keyframes_to_unload;
  for (std::size_t kf_id : loaded_keyframes_) {
    if (should_be_loaded.find(kf_id) == should_be_loaded.end()) {
      keyframes_to_unload.push_back(kf_id);
    }
  }

  // Unload keyframes that are too far
  for (std::size_t kf_id : keyframes_to_unload) {
    saveKeyframeToDisk(kf_id);  // This already erases from loaded_keyframes_
  }

  // Load keyframes that should be loaded but aren't
  for (std::size_t kf_id : should_be_loaded) {
    if (loaded_keyframes_.find(kf_id) == loaded_keyframes_.end()) {
      loadKeyframeFromDisk(kf_id);
    }
  }
}

void KeyframeQueue::unloadAllKeyframes() {
  for (std::size_t kf_id : loaded_keyframes_) {
    saveKeyframeToDisk(kf_id);  // This already erases from loaded_keyframes_
  }
}

// Helper function to increase keyframe times of use (similar to your original
// code)
void KeyframeQueue::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> keyframe,
    int additional_uses) {
  if (!keyframe) return;
  keyframe->remaining_times_of_use_ += additional_uses;
}

// Main keyframe selection method - MODIFIED to use loss and usage-based
// selection
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  std::lock_guard<std::mutex> lock(grid_mutex_);

  if (!latest_keyframe_) {
    // No keyframes yet, return nullptr
    return nullptr;
  }

  // First, manage memory - save distant keyframes, load nearby ones
  manageKeyframeMemory();

  // Get position of the latest keyframe
  torch::Tensor latest_center = latest_keyframe_->getCenter();
  Eigen::Vector3f latest_position = tensorToEigen(latest_center);

  // Get all keyframes within the selection radius (only loaded ones will be
  // returned)
  std::vector<std::shared_ptr<GaussianKeyframe>> candidates =
      getKeyframesInRadius(latest_position, selection_radius_);

  if (candidates.empty()) {
    // Fallback: use the latest keyframe itself
    candidates.push_back(latest_keyframe_);
  }

  // Apply the loss and usage-based selection logic (similar to your shuffled
  // approach)
  std::shared_ptr<GaussianKeyframe> selected_keyframe = nullptr;

  // First, check if any candidate has remaining times of use > 0
  std::vector<std::shared_ptr<GaussianKeyframe>> available_candidates;
  for (const auto& candidate : candidates) {
    if (candidate->remaining_times_of_use_ > 0) {
      available_candidates.push_back(candidate);
    }
  }

  // If no candidates have remaining uses, increase times of use for all
  // candidates
  if (available_candidates.empty()) {
    for (const auto& candidate : candidates) {
      increaseKeyframeTimesOfUse(candidate, 1);
      if (candidate->remaining_times_of_use_ > 0) {
        available_candidates.push_back(candidate);
      }
    }

    // If we have loss_map and auto_distribute logic should be applied
    if (loss_map_ && !loss_map_->empty()) {
      // Create vector of (keyframe_id, loss) pairs for candidates
      std::vector<std::pair<std::size_t, float>> loss_vec;
      for (const auto& candidate : candidates) {
        auto loss_it = loss_map_->find(candidate->fid_);
        if (loss_it != loss_map_->end()) {
          loss_vec.push_back({candidate->fid_, loss_it->second});
        }
      }

      if (!loss_vec.empty()) {
        // Select top keyframes with highest loss
        int k = std::max(
            1, static_cast<int>(loss_vec.size() / 4));  // Using /4 as default

        // Sort by loss (highest first)
        std::nth_element(loss_vec.begin(), loss_vec.begin() + k, loss_vec.end(),
                         [](const std::pair<std::size_t, float>& a,
                            const std::pair<std::size_t, float>& b) {
                           return a.second > b.second;
                         });

        // Increase times of use for top-k highest loss keyframes
        for (int i = 0; i < k; ++i) {
          auto scene_kf_it = scene_->keyframes().find(loss_vec[i].first);
          if (scene_kf_it != scene_->keyframes().end()) {
            increaseKeyframeTimesOfUse(scene_kf_it->second, 1);
          }
        }

        // Refresh available candidates
        available_candidates.clear();
        for (const auto& candidate : candidates) {
          if (candidate->remaining_times_of_use_ > 0) {
            available_candidates.push_back(candidate);
          }
        }
      }
    }
  }

  // Select from available candidates (those with remaining_times_of_use_ > 0)
  if (!available_candidates.empty()) {
    // For now, randomly select from available candidates
    std::uniform_int_distribution<> distrib(0, available_candidates.size() - 1);
    int random_index = distrib(rng_);
    selected_keyframe = available_candidates[random_index];
  } else {
    // Last resort fallback
    selected_keyframe = candidates[0];
  }

  if (selected_keyframe) {
    // Update usage tracking (increment used times)
    auto used_times_it = used_times_map_->find(selected_keyframe->fid_);
    if (used_times_it == used_times_map_->end()) {
      used_times_map_->emplace(selected_keyframe->fid_, 1);
    } else {
      ++used_times_it->second;
    }

    // Decrease remaining times of use
    --(selected_keyframe->remaining_times_of_use_);
  }

  return selected_keyframe;
}

// Handle new keyframe notification
void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  std::lock_guard<std::mutex> lock(grid_mutex_);

  // Add the new keyframe to our spatial grid (always keep ALL keyframes in the
  // grid)
  addKeyframeToGrid(keyframe);

  // Note: We don't remove any keyframes from the grid anymore.
  // Memory management is handled by manageKeyframeMemory() based on distance,
  // but the spatial grid always contains ALL keyframes for efficient spatial
  // queries.
}

// Print grid statistics for debugging
void KeyframeQueue::printGridStatistics() const {
  std::lock_guard<std::mutex> lock(grid_mutex_);

  std::cout << "=== Keyframe Spatial Grid Statistics ===" << std::endl;
  std::cout << "Total keyframes in grid: " << keyframe_positions_.size()
            << std::endl;
  std::cout << "Keyframes loaded in memory: " << loaded_keyframes_.size()
            << std::endl;
  std::cout << "Total grid cells used: " << spatial_grid_.size() << std::endl;
  std::cout << "Selection radius: " << selection_radius_ << "m" << std::endl;
  std::cout << "Memory management radius: " << memory_management_radius_ << "m"
            << std::endl;
  std::cout << "Grid cell size: " << grid_cell_size_ << "m" << std::endl;

  if (!spatial_grid_.empty()) {
    // Calculate grid occupancy statistics
    size_t total_entries = 0;
    size_t max_entries_per_cell = 0;
    size_t min_entries_per_cell = SIZE_MAX;

    for (const auto& cell : spatial_grid_) {
      size_t entries = cell.second.size();
      total_entries += entries;
      max_entries_per_cell = std::max(max_entries_per_cell, entries);
      min_entries_per_cell = std::min(min_entries_per_cell, entries);
    }

    float avg_entries_per_cell =
        static_cast<float>(total_entries) / spatial_grid_.size();

    std::cout << "Avg keyframes per cell: " << avg_entries_per_cell
              << std::endl;
    std::cout << "Max keyframes per cell: " << max_entries_per_cell
              << std::endl;
    std::cout << "Min keyframes per cell: " << min_entries_per_cell
              << std::endl;
  }

  if (latest_keyframe_) {
    torch::Tensor latest_center = latest_keyframe_->getCenter();
    Eigen::Vector3f latest_pos = tensorToEigen(latest_center);

    auto candidates = getKeyframesInRadius(latest_pos, selection_radius_);
    std::cout << "Keyframes in selection radius: " << candidates.size()
              << std::endl;

    // Count keyframes that should be in memory management radius
    size_t keyframes_in_memory_radius = 0;
    for (const auto& pos_pair : keyframe_positions_) {
      float distance = (pos_pair.second - latest_pos).norm();
      if (distance <= memory_management_radius_) {
        keyframes_in_memory_radius++;
      }
    }
    std::cout << "Keyframes in memory radius: " << keyframes_in_memory_radius
              << std::endl;
  }

  std::cout << "=====================================" << std::endl;
}

// Async saving worker thread
void KeyframeQueue::saveWorker() {
  while (true) {
    std::shared_ptr<GaussianKeyframe> keyframe_to_save;
    {
      std::unique_lock<std::mutex> lock(save_mutex_);
      save_cv_.wait(
          lock, [this] { return !save_queue_.empty() || save_worker_stop_; });

      if (save_worker_stop_ && save_queue_.empty()) {
        break;
      }

      if (!save_queue_.empty()) {
        keyframe_to_save = save_queue_.front();
        save_queue_.pop();
      }
    }

    if (keyframe_to_save) {
      try {
        keyframe_to_save->saveDataToDisk();
      } catch (const std::exception& e) {
        std::cerr << "Error saving keyframe " << keyframe_to_save->fid_ << ": "
                  << e.what() << std::endl;
      }
      keyframe_to_save->saving_ = false;
    }
  }
}

// Queue keyframe for async saving
void KeyframeQueue::queueForSaving(std::shared_ptr<GaussianKeyframe> keyframe) {
  {
    std::unique_lock<std::mutex> lock(save_mutex_);
    save_queue_.push(keyframe);
  }
  save_cv_.notify_one();
}