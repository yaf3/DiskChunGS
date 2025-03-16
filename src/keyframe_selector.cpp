#include "include/keyframe_selector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

#include "include/chunk_manager.h"
#include "include/profiling.h"

// Updated constructor
KeyframeSelector::KeyframeSelector(std::shared_ptr<ChunkManager> chunk_manager)
    : chunk_manager_(chunk_manager) {
  std::cout << "Creating KeyframeSelector" << std::endl;
}

// Select a keyframe using a weighted combination of strategies
std::shared_ptr<GaussianKeyframe> KeyframeSelector::selectKeyframe(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
    const std::map<std::size_t, float>& keyframe_losses,
    const std::map<std::size_t, int>& keyframe_usage_counts,
    int current_iteration) {
  auto timer = ProfilingUtils::Timer("KeyframeSelector::selectKeyframe");

  // Initialize shuffling if needed
  if (!shuffle_initialized_) {
    generateKeyframeShuffles(keyframes);
  }

  // Check if we need to refresh clusters
  auto now = std::chrono::steady_clock::now();
  auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(
                             now - last_selection_time_)
                             .count();

  // Periodically refresh clusters (every 100 frames or 60 seconds)
  if (time_since_last > 60 ||
      (current_iteration % 100 == 0 && current_iteration > 0)) {
    refreshClusters(keyframes, keyframe_losses);
    last_selection_time_ = now;
  }

  // Decrement cluster change countdown
  if (cluster_change_countdown_ > 0) {
    cluster_change_countdown_--;
  }

  // Choose a selection strategy based on the current state
  // Mix strategies for better results
  std::shared_ptr<GaussianKeyframe> selected_keyframe = nullptr;

  // Try to select based on chunk loading efficiency first
  // Prioritize keyframes whose visible chunks are already loaded
  std::vector<std::pair<std::shared_ptr<GaussianKeyframe>, float>> candidates;

  for (const auto& [kfid, keyframe] : keyframes) {
    // Skip keyframes with no remaining uses
    if (keyframe->remaining_times_of_use_ <= 0) {
      continue;
    }

    // Compute a score based on chunk availability and other factors
    float score =
        computeSelectionScore(keyframe, keyframe_losses, keyframe_usage_counts);
    candidates.push_back({keyframe, score});
  }

  // Sort by score (higher is better)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Take top candidate or select with another strategy if no candidates
  if (!candidates.empty()) {
    selected_keyframe = candidates[0].first;
  }

  // Fallback strategies if no good candidates
  if (!selected_keyframe) {
    // Early iterations: prioritize locality to build strong local
    // reconstructions
    if (current_iteration < 1000) {
      selected_keyframe = selectWithSpatialLocality(keyframes);
    }
    // Mid iterations: balance between locality and loss
    else if (current_iteration < 5000) {
      // Alternate between strategies
      if (current_iteration % 3 == 0) {
        selected_keyframe = selectWithHighLoss(keyframes, keyframe_losses);
      } else {
        selected_keyframe = selectWithSpatialLocality(keyframes);
      }
    }
    // Later iterations: prioritize high-loss keyframes and occasionally revisit
    // all frames
    else {
      if (current_iteration % 5 == 0) {
        selected_keyframe = selectWithLRU(keyframes, keyframe_usage_counts);
      } else if (current_iteration % 10 == 0) {
        selected_keyframe = selectWithRandom(keyframes);
      } else {
        selected_keyframe = selectWithHighLoss(keyframes, keyframe_losses);
      }
    }
  }

  // Fallback to random selection if other strategies failed
  if (!selected_keyframe) {
    selected_keyframe = selectWithRandom(keyframes);
  }

  // Update last used keyframe
  if (selected_keyframe) {
    last_keyframe_id_ = selected_keyframe->fid_;
    last_selection_time_ = std::chrono::steady_clock::now();
  }

  return selected_keyframe;
}

// Predict which keyframes will be used next
std::vector<std::shared_ptr<GaussianKeyframe>>
KeyframeSelector::predictUpcomingKeyframes(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
    const std::map<std::size_t, float>& keyframe_losses,
    int count) {
  std::vector<std::shared_ptr<GaussianKeyframe>> upcoming;

  // If we have a selected cluster, predict keyframes from that cluster
  if (!cluster_keyframes_.empty() && current_cluster_ >= 0 &&
      current_cluster_ < static_cast<int>(cluster_keyframes_.size())) {
    const auto& cluster = cluster_keyframes_[current_cluster_];

    // Get keyframes from current cluster
    for (const auto& kfid : cluster) {
      auto it = keyframes.find(kfid);
      if (it != keyframes.end() && it->second->remaining_times_of_use_ > 0) {
        upcoming.push_back(it->second);
        if (upcoming.size() >= static_cast<size_t>(count)) {
          break;
        }
      }
    }
  }

  // If we didn't get enough from the cluster, add some high-loss keyframes
  if (upcoming.size() < static_cast<size_t>(count)) {
    std::vector<std::pair<std::size_t, float>> keyframe_losses_vec;

    for (const auto& [kfid, loss] : keyframe_losses) {
      auto it = keyframes.find(kfid);
      if (it != keyframes.end() && it->second->remaining_times_of_use_ > 0) {
        // Check if already in upcoming
        bool already_included = false;
        for (const auto& kf : upcoming) {
          if (kf->fid_ == kfid) {
            already_included = true;
            break;
          }
        }

        if (!already_included) {
          keyframe_losses_vec.push_back({kfid, loss});
        }
      }
    }

    // Sort by loss (higher first)
    std::sort(keyframe_losses_vec.begin(), keyframe_losses_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Add top loss keyframes
    for (const auto& [kfid, _] : keyframe_losses_vec) {
      upcoming.push_back(keyframes.at(kfid));
      if (upcoming.size() >= static_cast<size_t>(count)) {
        break;
      }
    }
  }

  return upcoming;
}

// Get spatial cluster of a keyframe
int KeyframeSelector::getKeyframeCluster(std::size_t keyframe_id) {
  auto it = keyframe_clusters_.find(keyframe_id);
  if (it != keyframe_clusters_.end()) {
    return it->second;
  }
  return -1;
}

// Refresh clustering of keyframes
void KeyframeSelector::refreshClusters(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
    const std::map<std::size_t, float>& keyframe_losses) {
  clusterKeyframesByPosition(keyframes);

  // Switch to a new cluster or stay in current one
  if (cluster_keyframes_.empty()) {
    current_cluster_ = -1;
    return;
  }

  // If current cluster is valid, decrement countdown, otherwise pick a new one
  if (current_cluster_ >= 0 &&
      current_cluster_ < static_cast<int>(cluster_keyframes_.size()) &&
      cluster_change_countdown_ > 0) {
    return;
  }

  // Pick a new cluster based on:
  // 1. Keyframes remaining in each cluster
  // 2. Loss in each cluster
  std::vector<std::pair<int, float>> cluster_scores;

  for (int i = 0; i < static_cast<int>(cluster_keyframes_.size()); ++i) {
    const auto& cluster = cluster_keyframes_[i];

    int remaining_uses = 0;
    float total_loss = 0.0f;
    int count = 0;

    for (const auto& kfid : cluster) {
      auto it = keyframes.find(kfid);
      if (it != keyframes.end()) {
        remaining_uses += it->second->remaining_times_of_use_;

        // Add loss if available
        auto loss_it = keyframe_losses.find(kfid);
        if (loss_it != keyframe_losses.end()) {
          total_loss += loss_it->second;
          ++count;
        }
      }
    }

    float avg_loss = count > 0 ? total_loss / count : 0.0f;
    float score = remaining_uses * 0.5f + avg_loss * 100.0f;

    cluster_scores.push_back({i, score});
  }

  // Sort by score (higher is better)
  std::sort(cluster_scores.begin(), cluster_scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Pick top cluster
  if (!cluster_scores.empty()) {
    current_cluster_ = cluster_scores[0].first;
    // Set countdown to process this cluster for a while
    // Higher for bigger clusters to ensure we complete them
    cluster_change_countdown_ = std::max(
        20, static_cast<int>(cluster_keyframes_[current_cluster_].size() / 2));
  } else {
    current_cluster_ = -1;
  }
}

// Generate keyframe shuffle for randomization
void KeyframeSelector::generateKeyframeShuffles(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  keyframe_shuffle_.resize(keyframes.size());
  std::iota(keyframe_shuffle_.begin(), keyframe_shuffle_.end(), 0);
  std::shuffle(keyframe_shuffle_.begin(), keyframe_shuffle_.end(), rng_);
  shuffle_idx_ = 0;
  shuffle_initialized_ = true;
}

// Compute score for keyframe selection
float KeyframeSelector::computeSelectionScore(
    std::shared_ptr<GaussianKeyframe> keyframe,
    const std::map<std::size_t, float>& keyframe_losses,
    const std::map<std::size_t, int>& keyframe_usage_counts) {
  if (!keyframe) return 0.0f;

  float score = 0.0f;

  // Factor 1: Loss value (higher loss means we should use it more)
  auto loss_it = keyframe_losses.find(keyframe->fid_);
  if (loss_it != keyframe_losses.end()) {
    score += loss_it->second * 10.0f;  // Weight loss more heavily
  }

  // Factor 2: Remaining uses (more uses = higher priority)
  score += static_cast<float>(keyframe->remaining_times_of_use_) * 0.1f;

  // Factor 3: Locality score (prefer keyframes in current cluster)
  int cluster = getKeyframeCluster(keyframe->fid_);
  if (cluster == current_cluster_) {
    score += 5.0f;  // Strong boost for current cluster
  }

  // Factor 4: Chunk availability (major boost if chunks already loaded)
  // Get visible chunks for this keyframe
  std::array<Eigen::Vector4f, 6> frustum_planes =
      chunk_manager_->computeFrustumPlanes(keyframe);

  // Extract camera position
  Sophus::SE3d camera_pose = keyframe->getPose();
  Sophus::SE3d Twc = camera_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();

  // Get chunks in view frustum
  ChunkCoord camera_chunk = chunk_manager_->getChunkCoord(camera_position);
  int search_radius = 3;  // Smaller radius for quick check
  int chunks_in_frustum = 0;
  int loaded_chunks = 0;

  for (int dx = -search_radius; dx <= search_radius; dx++) {
    for (int dy = -search_radius; dy <= search_radius; dy++) {
      for (int dz = -search_radius; dz <= search_radius; dz++) {
        ChunkCoord check_coord{camera_chunk.x + dx, camera_chunk.y + dy,
                               camera_chunk.z + dz};

        // Is this chunk in frustum?
        if (chunk_manager_->isChunkInFrustum(check_coord, frustum_planes)) {
          chunks_in_frustum++;

          // Is this chunk loaded?
          if (chunk_manager_->getChunkAt(check_coord)) {
            loaded_chunks++;
          }
        }
      }
    }
  }

  // Compute ratio of loaded chunks (0 to 1)
  float chunk_ratio =
      chunks_in_frustum > 0
          ? static_cast<float>(loaded_chunks) / chunks_in_frustum
          : 0.0f;

  // Add major boost for loaded chunks
  score += chunk_ratio * 50.0f;

  // Factor 5: Iteration since last use (LRU factor)
  auto usage_it = keyframe_usage_counts.find(keyframe->fid_);
  if (usage_it == keyframe_usage_counts.end()) {
    // Never used keyframe gets a boost
    score += 2.0f;
  }

  return score;
}

// Cluster keyframes by their spatial position
void KeyframeSelector::clusterKeyframesByPosition(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  // Skip if too few keyframes
  if (keyframes.size() < 5) {
    keyframe_clusters_.clear();
    cluster_keyframes_.clear();
    return;
  }

  // Extract positions
  std::vector<std::pair<std::size_t, Eigen::Vector3f>> keyframe_positions;
  for (const auto& [kfid, keyframe] : keyframes) {
    if (keyframe->remaining_times_of_use_ <= 0) continue;

    Sophus::SE3d Twc = keyframe->getPose().inverse();
    Eigen::Vector3f position = Twc.translation().cast<float>();
    keyframe_positions.push_back({kfid, position});
  }

  // DBSCAN-like clustering
  float epsilon = 10.0f;  // Distance threshold for clustering
  int min_points = 3;     // Minimum points to form a cluster

  std::vector<int> cluster_labels(keyframe_positions.size(),
                                  -1);  // -1 = unprocessed
  int current_label = 0;

  for (size_t i = 0; i < keyframe_positions.size(); ++i) {
    // Skip already processed points
    if (cluster_labels[i] != -1) continue;

    // Find neighbors
    std::vector<size_t> neighbors;
    for (size_t j = 0; j < keyframe_positions.size(); ++j) {
      if (i == j) continue;

      float dist =
          (keyframe_positions[i].second - keyframe_positions[j].second).norm();
      if (dist < epsilon) {
        neighbors.push_back(j);
      }
    }

    // If not enough neighbors, mark as noise (-2)
    if (neighbors.size() < static_cast<size_t>(min_points - 1)) {
      cluster_labels[i] = -2;
      continue;
    }

    // Start a new cluster
    cluster_labels[i] = current_label;

    // Expand cluster
    for (size_t j = 0; j < neighbors.size(); ++j) {
      size_t neighbor_idx = neighbors[j];

      // If unprocessed, add to cluster
      if (cluster_labels[neighbor_idx] == -1) {
        cluster_labels[neighbor_idx] = current_label;

        // Find neighbors of this point
        for (size_t k = 0; k < keyframe_positions.size(); ++k) {
          if (neighbor_idx == k) continue;

          float dist = (keyframe_positions[neighbor_idx].second -
                        keyframe_positions[k].second)
                           .norm();
          if (dist < epsilon && cluster_labels[k] == -1) {
            neighbors.push_back(k);
          }
        }
      }
    }

    current_label++;
  }

  // Organize clusters
  keyframe_clusters_.clear();
  cluster_keyframes_.resize(current_label);

  for (size_t i = 0; i < keyframe_positions.size(); ++i) {
    int label = cluster_labels[i];
    if (label >= 0) {
      keyframe_clusters_[keyframe_positions[i].first] = label;
      cluster_keyframes_[label].push_back(keyframe_positions[i].first);
    } else {
      // Noise points get their own "cluster"
      keyframe_clusters_[keyframe_positions[i].first] = -1;
    }
  }

  // If current cluster no longer exists, reset it
  if (current_cluster_ >= current_label) {
    current_cluster_ = -1;
    cluster_change_countdown_ = 0;
  }
}

// Calculate spatial distance between keyframes
float KeyframeSelector::keyframeDistance(
    std::shared_ptr<GaussianKeyframe> kf1,
    std::shared_ptr<GaussianKeyframe> kf2) {
  if (!kf1 || !kf2) return std::numeric_limits<float>::max();

  Sophus::SE3d Twc1 = kf1->getPose().inverse();
  Sophus::SE3d Twc2 = kf2->getPose().inverse();

  Eigen::Vector3f pos1 = Twc1.translation().cast<float>();
  Eigen::Vector3f pos2 = Twc2.translation().cast<float>();

  // Position distance
  float pos_dist = (pos1 - pos2).norm();

  // Viewing direction similarity
  Eigen::Vector3f view_dir1 =
      -kf1->getPose().rotationMatrix().col(2).cast<float>();
  Eigen::Vector3f view_dir2 =
      -kf2->getPose().rotationMatrix().col(2).cast<float>();

  float view_sim =
      view_dir1.dot(view_dir2);  // 1 = same direction, -1 = opposite

  // Combined distance metric (position and orientation)
  return pos_dist * (2.0f - view_sim);  // Lower means more similar
}

// Check if two keyframes are spatially close
bool KeyframeSelector::isSpatiallyClose(std::shared_ptr<GaussianKeyframe> kf1,
                                        std::shared_ptr<GaussianKeyframe> kf2,
                                        float threshold) {
  return keyframeDistance(kf1, kf2) < threshold;
}

// Strategy: Random selection
std::shared_ptr<GaussianKeyframe> KeyframeSelector::selectWithRandom(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  if (keyframes.empty()) return nullptr;

  // Use the shuffle
  if (!shuffle_initialized_) {
    generateKeyframeShuffles(keyframes);
  }

  std::shared_ptr<GaussianKeyframe> selected = nullptr;

  // Try keyframes in shuffled order
  for (size_t i = 0; i < keyframe_shuffle_.size(); ++i) {
    // Get next shuffled index
    shuffle_idx_ = (shuffle_idx_ + 1) % keyframe_shuffle_.size();

    // Get keyframe at this index
    auto it = keyframes.begin();
    std::advance(it, keyframe_shuffle_[shuffle_idx_] % keyframes.size());

    if (it->second->remaining_times_of_use_ > 0) {
      selected = it->second;
      break;
    }
  }

  // If no unused keyframe found, regenerate shuffle
  if (!selected) {
    generateKeyframeShuffles(keyframes);

    // Try again with fresh shuffle
    for (size_t i = 0; i < keyframe_shuffle_.size(); ++i) {
      shuffle_idx_ = (shuffle_idx_ + 1) % keyframe_shuffle_.size();

      auto it = keyframes.begin();
      std::advance(it, keyframe_shuffle_[shuffle_idx_] % keyframes.size());

      if (it->second->remaining_times_of_use_ > 0) {
        selected = it->second;
        break;
      }
    }
  }

  return selected;
}

// Strategy: Spatial locality (select from current cluster)
std::shared_ptr<GaussianKeyframe> KeyframeSelector::selectWithSpatialLocality(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes) {
  if (keyframes.empty()) return nullptr;

  // If we have clusters and a current cluster
  if (!cluster_keyframes_.empty() && current_cluster_ >= 0 &&
      current_cluster_ < static_cast<int>(cluster_keyframes_.size())) {
    const auto& cluster = cluster_keyframes_[current_cluster_];

    // Find unused keyframes in cluster
    std::vector<std::shared_ptr<GaussianKeyframe>> candidates;

    for (const auto& kfid : cluster) {
      auto it = keyframes.find(kfid);
      if (it != keyframes.end() && it->second->remaining_times_of_use_ > 0) {
        candidates.push_back(it->second);
      }
    }

    // If we have candidates, select one randomly
    if (!candidates.empty()) {
      std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
      return candidates[dist(rng_)];
    }
  }

  // Fallback: find unused keyframe with spatial locality to last used keyframe
  if (last_keyframe_id_ != 0) {
    auto last_it = keyframes.find(last_keyframe_id_);
    if (last_it != keyframes.end()) {
      // Find keyframes within a distance threshold
      std::vector<std::shared_ptr<GaussianKeyframe>> nearby;
      float threshold = 20.0f;  // Distance threshold

      for (const auto& [kfid, keyframe] : keyframes) {
        if (keyframe->remaining_times_of_use_ > 0 &&
            isSpatiallyClose(last_it->second, keyframe, threshold)) {
          nearby.push_back(keyframe);
        }
      }

      // Select one randomly
      if (!nearby.empty()) {
        std::uniform_int_distribution<size_t> dist(0, nearby.size() - 1);
        return nearby[dist(rng_)];
      }
    }
  }

  // Fallback to random
  return selectWithRandom(keyframes);
}

// Strategy: LRU (Least Recently Used)
std::shared_ptr<GaussianKeyframe> KeyframeSelector::selectWithLRU(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
    const std::map<std::size_t, int>& keyframe_usage_counts) {
  if (keyframes.empty()) return nullptr;

  // Get all keyframes with usage count
  std::vector<std::pair<std::shared_ptr<GaussianKeyframe>, int>> candidates;

  for (const auto& [kfid, keyframe] : keyframes) {
    if (keyframe->remaining_times_of_use_ > 0) {
      int usage = 0;
      auto it = keyframe_usage_counts.find(kfid);
      if (it != keyframe_usage_counts.end()) {
        usage = it->second;
      }
      candidates.push_back({keyframe, usage});
    }
  }

  // Sort by usage (lower first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Get least used keyframe
  if (!candidates.empty()) {
    return candidates[0].first;
  }

  return nullptr;
}

// Strategy: High Loss
std::shared_ptr<GaussianKeyframe> KeyframeSelector::selectWithHighLoss(
    const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
    const std::map<std::size_t, float>& keyframe_losses) {
  if (keyframes.empty()) return nullptr;

  // Get all keyframes with loss
  std::vector<std::pair<std::shared_ptr<GaussianKeyframe>, float>> candidates;

  for (const auto& [kfid, loss] : keyframe_losses) {
    auto it = keyframes.find(kfid);
    if (it != keyframes.end() && it->second->remaining_times_of_use_ > 0) {
      candidates.push_back({it->second, loss});
    }
  }

  // Sort by loss (higher first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Get highest loss keyframe
  if (!candidates.empty()) {
    return candidates[0].first;
  }

  return nullptr;
}