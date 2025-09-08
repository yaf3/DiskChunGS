#include "include/keyframe_selection.h"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             float chunk_size)
    : scene_(scene),
      chunk_size_(chunk_size),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f),
      level_dist_({0.5, 0.3, 0.2}) {
  chunk_sizes_[0] = 2 * chunk_size_;  // FINE
  chunk_sizes_[1] = 4 * chunk_size_;  // MEDIUM
  chunk_sizes_[2] = 8 * chunk_size_;  // COARSE
}

void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  latest_keyframe_ = keyframe;

  torch::Tensor center_tensor = keyframe->getCenter();
  Eigen::Vector3f position = tensorToEigen(center_tensor);

  for (int level = 0; level < 3; ++level) {
    ChunkCoord chunk_coord = getChunkCoord(position, chunk_sizes_[level]);
    int64_t chunk_id = encodeChunkCoord(chunk_coord);
    chunk_to_keyframes_[level][chunk_id].push_back(keyframe);
  }
}

// Updated keyframe selection using active_frames_gpu_
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (!latest_keyframe_) {
    return nullptr;
  }

  torch::Tensor latest_center = latest_keyframe_->getCenter();
  Eigen::Vector3f latest_position = tensorToEigen(latest_center);

  int level = level_dist_(rng_);
  ChunkCoord chunk_coord = getChunkCoord(latest_position, chunk_sizes_[level]);
  int64_t chunk_id = encodeChunkCoord(chunk_coord);

  auto it = chunk_to_keyframes_[level].find(chunk_id);
  if (it != chunk_to_keyframes_[level].end() && !it->second.empty()) {
    const auto& keyframes_in_chunk = it->second;
    std::uniform_int_distribution<size_t> dist(0,
                                               keyframes_in_chunk.size() - 1);
    size_t random_index = dist(rng_);
    std::shared_ptr<GaussianKeyframe> selected_keyframe =
        keyframes_in_chunk[random_index];

    // Check if already on GPU to avoid redundant transfer
    if (!selected_keyframe->loaded_) {
      selected_keyframe->loadDataFromDisk();
    }

    // More efficient GPU queue management
    // Remove the keyframe if it's already in the queue (avoid duplicates)
    auto queue_it =
        std::find(gpu_queue.begin(), gpu_queue.end(), selected_keyframe);
    if (queue_it != gpu_queue.end()) {
      gpu_queue.erase(queue_it);
    }

    // Add to front (most recently used)
    gpu_queue.push_front(selected_keyframe);

    // Clean up oldest keyframes more efficiently
    while (gpu_queue.size() > max_gpu_keyframes_) {
      std::shared_ptr<GaussianKeyframe> oldest = gpu_queue.back();
      gpu_queue.pop_back();

      // Only transfer to CPU if it's actually loaded and not the selected one
      if (oldest->loaded_ && oldest != selected_keyframe) {
        oldest->saveDataToDisk();
      }
    }

    return selected_keyframe;
  }

  return nullptr;
}

Eigen::Vector3f KeyframeQueue::tensorToEigen(
    const torch::Tensor& tensor) const {
  // Ensure tensor is on CPU and contiguous
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();

  // Get pointer to data
  float* data_ptr = cpu_tensor.data_ptr<float>();

  return Eigen::Vector3f(data_ptr[0], data_ptr[1], data_ptr[2]);
}