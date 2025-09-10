#include "include/keyframe_selection.h"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             float chunk_size,
                             const std::map<std::size_t, float>* loss_map,
                             std::map<std::size_t, int>* used_times_map)
    : scene_(scene),
      chunk_size_(chunk_size),
      loss_map_(loss_map),
      used_times_map_(used_times_map),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f),
      level_dist_({0.5, 0.3, 0.2}) {
  chunk_sizes_[0] = 4 * chunk_size_;   // FINE
  chunk_sizes_[1] = 8 * chunk_size_;   // MEDIUM
  chunk_sizes_[2] = 12 * chunk_size_;  // COARSE
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

  // Select level using the existing distribution
  int level = level_dist_(rng_);
  ChunkCoord chunk_coord = getChunkCoord(latest_position, chunk_sizes_[level]);
  int64_t chunk_id = encodeChunkCoord(chunk_coord);

  auto it = chunk_to_keyframes_[level].find(chunk_id);
  if (it == chunk_to_keyframes_[level].end() || it->second.empty()) {
    return nullptr;
  }

  const auto& candidates = it->second;
  std::shared_ptr<GaussianKeyframe> selected_keyframe = nullptr;

  // Apply loss and usage-based selection logic
  std::vector<std::shared_ptr<GaussianKeyframe>> available_candidates;

  // First, check if any candidate has remaining times of use > 0
  for (const auto& candidate : candidates) {
    if (candidate->remaining_times_of_use_ > 0) {
      available_candidates.push_back(candidate);
    }
  }

  // If no candidates have remaining uses, apply loss-based distribution
  if (available_candidates.empty()) {
    // Basic increase for all candidates
    for (const auto& candidate : candidates) {
      increaseKeyframeTimesOfUse(candidate, 1);
      if (candidate->remaining_times_of_use_ > 0) {
        available_candidates.push_back(candidate);
      }
    }

    // Apply loss-based auto-distribution if loss_map is available
    if (loss_map_ && !loss_map_->empty()) {
      std::vector<std::pair<std::size_t, float>> loss_vec;

      // Collect loss values for candidates in this chunk
      for (const auto& candidate : candidates) {
        auto loss_it = loss_map_->find(candidate->fid_);
        if (loss_it != loss_map_->end()) {
          loss_vec.push_back({candidate->fid_, loss_it->second});
        }
      }

      if (!loss_vec.empty()) {
        // Select top 25% of keyframes with highest loss
        int k = std::max(1, static_cast<int>(loss_vec.size() / 4));

        // Partial sort to get top-k highest loss keyframes
        std::nth_element(loss_vec.begin(), loss_vec.begin() + k, loss_vec.end(),
                         [](const std::pair<std::size_t, float>& a,
                            const std::pair<std::size_t, float>& b) {
                           return a.second > b.second;
                         });
        // std::cout << "Top " << k << " keyframes by loss in chunk " <<
        // chunk_id
        //           << ": ";
        // for (int i = 0; i < k; ++i) {
        //   std::cout << "(KF ID: " << loss_vec[i].first
        //             << ", Loss: " << loss_vec[i].second << ") ";
        // }
        // std::cout << std::endl;

        // Give additional uses to high-loss keyframes
        for (int i = 0; i < k; ++i) {
          auto scene_kf_it = scene_->keyframes().find(loss_vec[i].first);
          if (scene_kf_it != scene_->keyframes().end()) {
            // Find the keyframe in our candidates
            for (const auto& candidate : candidates) {
              if (candidate->fid_ == loss_vec[i].first) {
                increaseKeyframeTimesOfUse(candidate, 1);
                break;
              }
            }
          }
        }

        // Refresh available candidates after loss-based distribution
        available_candidates.clear();
        for (const auto& candidate : candidates) {
          if (candidate->remaining_times_of_use_ > 0) {
            available_candidates.push_back(candidate);
          }
        }
      }
    }
  }

  // Select from available candidates
  if (!available_candidates.empty()) {
    // Weight selection by remaining uses (higher remaining uses = higher
    // probability)
    std::vector<float> weights;
    weights.reserve(available_candidates.size());

    for (const auto& candidate : available_candidates) {
      // Use remaining times as weight, with a minimum weight to ensure variety
      float weight = std::max(
          1.0f, static_cast<float>(candidate->remaining_times_of_use_));
      weights.push_back(weight);
    }

    // Weighted random selection
    std::discrete_distribution<> weighted_dist(weights.begin(), weights.end());
    size_t selected_index = weighted_dist(rng_);
    selected_keyframe = available_candidates[selected_index];
  } else {
    // Last resort: random selection from all candidates
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    size_t random_index = dist(rng_);
    selected_keyframe = candidates[random_index];
  }

  if (selected_keyframe) {
    // Update usage tracking
    if (used_times_map_) {
      auto used_times_it = used_times_map_->find(selected_keyframe->fid_);
      if (used_times_it == used_times_map_->end()) {
        used_times_map_->emplace(selected_keyframe->fid_, 1);
      } else {
        ++used_times_it->second;
      }
    }

    // Decrease remaining times of use
    if (selected_keyframe->remaining_times_of_use_ > 0) {
      --(selected_keyframe->remaining_times_of_use_);
    }

    // Efficient GPU memory management
    if (!selected_keyframe->loaded_) {
      selected_keyframe->loadDataFromDisk();
    }

    // Remove keyframe if already in queue to avoid duplicates
    auto queue_it =
        std::find(gpu_queue.begin(), gpu_queue.end(), selected_keyframe);
    if (queue_it != gpu_queue.end()) {
      gpu_queue.erase(queue_it);
    }

    // Add to front (most recently used)
    gpu_queue.push_front(selected_keyframe);

    // Clean up oldest keyframes
    while (gpu_queue.size() > max_gpu_keyframes_) {
      std::shared_ptr<GaussianKeyframe> oldest = gpu_queue.back();
      gpu_queue.pop_back();

      // Only transfer to CPU if it's loaded and not the selected one
      if (oldest->loaded_ && oldest != selected_keyframe) {
        oldest->saveDataToDisk();
      }
    }
  }

  return selected_keyframe;
}

Eigen::Vector3f KeyframeQueue::tensorToEigen(
    const torch::Tensor& tensor) const {
  // Ensure tensor is on CPU and contiguous
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();

  // Get pointer to data
  float* data_ptr = cpu_tensor.data_ptr<float>();

  return Eigen::Vector3f(data_ptr[0], data_ptr[1], data_ptr[2]);
}

void KeyframeQueue::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> keyframe,
    int additional_uses) {
  if (!keyframe) return;
  keyframe->remaining_times_of_use_ += additional_uses;
}

void KeyframeQueue::updateKeyframeAssociation(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  // Remove old associations first
  for (int level = 0; level < 3; ++level) {
    auto& level_map = chunk_to_keyframes_[level];
    for (auto& chunk_pair : level_map) {
      auto& keyframe_list = chunk_pair.second;
      keyframe_list.erase(
          std::remove(keyframe_list.begin(), keyframe_list.end(), keyframe),
          keyframe_list.end());
    }
  }

  // Add new associations based on current position
  torch::Tensor center_tensor = keyframe->getCenter();
  Eigen::Vector3f position = tensorToEigen(center_tensor);

  for (int level = 0; level < 3; ++level) {
    ChunkCoord chunk_coord = getChunkCoord(position, chunk_sizes_[level]);
    int64_t chunk_id = encodeChunkCoord(chunk_coord);
    chunk_to_keyframes_[level][chunk_id].push_back(keyframe);
  }
}