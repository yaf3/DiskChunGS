#include "include/keyframe_selection.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>

#include "include/profiling.h"

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t queue_size)
    : scene_(scene),
      queue_size_(queue_size),
      kfid_shuffled_(false),
      kfid_shuffle_idx_(0),
      current_cluster_(0),
      cluster_iterations_(0),
      iterations_per_cluster_(50),  // Default value, can be adjusted
      keyframes_since_last_full_clustering_(0) {
  // Initialize empty
}

void KeyframeQueue::setChunkManager(
    std::shared_ptr<ChunkManager> chunk_manager) {
  chunk_manager_ = chunk_manager;
}

// Original random shuffle method (kept for backward compatibility)
void KeyframeQueue::generateKfidRandomShuffle() {
  if (scene_->keyframes().empty()) return;

  // Create vector of keyframe IDs
  kfid_shuffle_.clear();
  for (const auto& [fid, _] : scene_->keyframes()) {
    kfid_shuffle_.push_back(fid);
  }

  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
  std::cout << "Generated random shuffle of " << kfid_shuffle_.size()
            << " keyframes" << std::endl;
}

// New spatial clustering method
void KeyframeQueue::generateSpatiallyCoherentBatches() {
  auto timer = ProfilingUtils::Timer("generateSpatiallyCoherentBatches");

  if (scene_->keyframes().empty()) return;

  std::cout << "Generating spatially coherent keyframe batches..." << std::endl;

  // Step 1: Extract keyframe positions
  std::vector<std::pair<std::size_t, Eigen::Vector3f>> keyframe_positions;
  keyframe_positions.reserve(scene_->keyframes().size());

  for (const auto& [fid, keyframe] : scene_->keyframes()) {
    if (!keyframe->set_pose_) continue;  // Skip keyframes without valid poses

    // Get world position of camera (inverse of camera-to-world)
    Sophus::SE3d Twc = keyframe->getPose().inverse();
    Eigen::Vector3f position = Twc.translation().cast<float>();
    keyframe_positions.push_back({fid, position});
  }

  if (keyframe_positions.empty()) {
    std::cout
        << "No valid keyframe positions found, falling back to random shuffle"
        << std::endl;
    generateKfidRandomShuffle();
    return;
  }

  // Step 2: Determine appropriate number of clusters based on scene size
  const int min_keyframes_per_cluster =
      15;  // Target minimum keyframes per cluster
  int num_clusters = std::max(1, static_cast<int>(keyframe_positions.size() /
                                                  min_keyframes_per_cluster));
  // num_clusters = std::min(8, num_clusters);  // Cap at 8 clusters

  std::cout << "Creating " << num_clusters << " spatial clusters for "
            << keyframe_positions.size() << " keyframes" << std::endl;

  std::vector<std::vector<std::size_t>> clusters(num_clusters);

  // Initialize cluster centers with furthest point sampling for better
  // distribution
  cluster_centers_.clear();
  cluster_centers_.push_back(
      keyframe_positions[0].second);  // Start with first point

  for (int i = 1; i < num_clusters; i++) {
    // Find furthest point from all existing centers
    float max_dist = -1;
    std::size_t furthest_idx = 0;

    for (std::size_t j = 0; j < keyframe_positions.size(); j++) {
      float min_dist = std::numeric_limits<float>::max();
      for (const auto& center : cluster_centers_) {
        float dist = (keyframe_positions[j].second - center).norm();
        min_dist = std::min(min_dist, dist);
      }

      if (min_dist > max_dist) {
        max_dist = min_dist;
        furthest_idx = j;
      }
    }

    cluster_centers_.push_back(keyframe_positions[furthest_idx].second);
  }

  // Assign keyframes to nearest cluster
  for (const auto& [fid, position] : keyframe_positions) {
    int nearest_cluster = findNearestCluster(position);
    clusters[nearest_cluster].push_back(fid);
  }

  // Step 3: Create queue of clusters, and shuffle keyframes within each cluster
  kfid_shuffle_.clear();
  clusters_.clear();

  std::mt19937 g(std::random_device{}());

  for (auto& cluster : clusters) {
    // Skip empty clusters
    if (cluster.empty()) continue;

    // Shuffle keyframes within the cluster
    std::shuffle(cluster.begin(), cluster.end(), g);

    clusters_.push_back(cluster);
  }

  // Shuffle the order of clusters
  std::shuffle(clusters_.begin(), clusters_.end(), g);

  // If we were already training, try to keep current cluster if it still exists
  int old_current_cluster = current_cluster_;
  current_cluster_ = 0;

  if (kfid_shuffled_ && old_current_cluster < clusters_.size()) {
    // Try to maintain continuity by keeping the same cluster index if possible
    current_cluster_ = old_current_cluster;
  }

  // Start with current cluster
  if (!clusters_.empty()) {
    kfid_shuffle_ = clusters_[current_cluster_];
  }

  kfid_shuffled_ = true;
  kfid_shuffle_idx_ = 0;
  // Don't reset cluster_iterations_ to maintain continuity

  // Reset new keyframe counter
  keyframes_since_last_full_clustering_ = 0;

  // Print cluster information
  std::cout << "Created " << clusters_.size() << " spatial clusters with:";
  for (size_t i = 0; i < clusters_.size(); i++) {
    std::cout << " [" << i << "]:" << clusters_[i].size();
  }
  std::cout << " keyframes" << std::endl;

  // Initialize with current cluster
  prewarmClusterCache();
}

// Pre-warm cache for current cluster
void KeyframeQueue::prewarmClusterCache() {
  if (!chunk_manager_ || clusters_.empty() ||
      current_cluster_ >= clusters_.size()) {
    return;
  }

  std::cout << "Pre-warming cache for cluster " << current_cluster_ << "..."
            << std::endl;

  // Take first few keyframes from current cluster to pre-load their chunks
  const int preload_count =
      std::min(5, static_cast<int>(clusters_[current_cluster_].size()));

  for (int i = 0; i < preload_count; i++) {
    auto kf_id = clusters_[current_cluster_][i];
    auto it = scene_->keyframes().find(kf_id);
    if (it != scene_->keyframes().end()) {
      // Use preloadVisibleChunks with true to use cache
      chunk_manager_->preloadVisibleChunks(it->second, true);
    }
  }
}

// Modified fillQueue to use spatial coherence
void KeyframeQueue::fillQueue() {
  // Check if we should generate clusters
  if (!kfid_shuffled_) {
    generateSpatiallyCoherentBatches();
  }

  // Check if we should move to the next cluster
  if (cluster_iterations_ >= iterations_per_cluster_ && !clusters_.empty()) {
    cluster_iterations_ = 0;
    current_cluster_ = (current_cluster_ + 1) % clusters_.size();

    // Switch to the next cluster
    kfid_shuffle_ = clusters_[current_cluster_];
    kfid_shuffle_idx_ = 0;

    std::cout << "Switching to spatial cluster " << current_cluster_ << " with "
              << kfid_shuffle_.size() << " keyframes" << std::endl;

    // When switching clusters, clear the queue for fresh keyframes
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }

    // Pre-warm cache for new cluster
    prewarmClusterCache();
  }

  // Keep filling until we reach desired size or run out of options
  while (keyframe_queue_.size() < queue_size_ && !kfid_shuffle_.empty()) {
    int start_shuffle_idx = kfid_shuffle_idx_;
    std::shared_ptr<GaussianKeyframe> next_kf = nullptr;

    do {
      // Move to next index, wrapping if needed
      kfid_shuffle_idx_ = (kfid_shuffle_idx_ + 1) % kfid_shuffle_.size();

      // If we've checked all keyframes in this cluster and found none with uses
      // left
      if (kfid_shuffle_idx_ == start_shuffle_idx) {
        // Add 1 time of use to all keyframes in this cluster
        for (auto fid : kfid_shuffle_) {
          auto it = scene_->keyframes().find(fid);
          if (it != scene_->keyframes().end()) {
            it->second->remaining_times_of_use_ += 1;
          }
        }
      }

      // Get keyframe at current shuffle index
      std::size_t kf_id = kfid_shuffle_[kfid_shuffle_idx_];
      auto it = scene_->keyframes().find(kf_id);

      if (it != scene_->keyframes().end()) {
        next_kf = it->second;
      } else {
        next_kf = nullptr;
      }
    } while (next_kf && next_kf->remaining_times_of_use_ <= 0);

    // Add usable keyframe to queue
    if (next_kf && next_kf->remaining_times_of_use_ > 0) {
      keyframe_queue_.push(next_kf);
    } else {
      break;
    }
  }
}

// Helper to find nearest cluster for a keyframe
int KeyframeQueue::findNearestCluster(const Eigen::Vector3f& position) const {
  if (cluster_centers_.empty()) return 0;

  float min_dist = std::numeric_limits<float>::max();
  int nearest_cluster = 0;

  for (int i = 0; i < cluster_centers_.size(); i++) {
    float dist = (position - cluster_centers_[i]).norm();
    if (dist < min_dist) {
      min_dist = dist;
      nearest_cluster = i;
    }
  }

  return nearest_cluster;
}

// Add a single keyframe to existing clusters
void KeyframeQueue::addKeyframeToExistingClusters(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe || !keyframe->set_pose_ || clusters_.empty() ||
      cluster_centers_.empty()) {
    // Can't add to clusters if they don't exist or keyframe has no pose
    return;
  }

  // Get keyframe position
  Sophus::SE3d Twc = keyframe->getPose().inverse();
  Eigen::Vector3f position = Twc.translation().cast<float>();

  // Find nearest cluster
  int nearest_cluster = findNearestCluster(position);

  // Make sure the cluster index is valid
  if (nearest_cluster >= clusters_.size()) {
    nearest_cluster = clusters_.size() - 1;
  }

  // Add keyframe to the nearest cluster
  clusters_[nearest_cluster].push_back(keyframe->fid_);

  // Update the current shuffle if we're in the affected cluster
  if (nearest_cluster == current_cluster_) {
    kfid_shuffle_ = clusters_[current_cluster_];
    // Preserve the current index if possible
    kfid_shuffle_idx_ =
        std::min(kfid_shuffle_idx_, static_cast<int>(kfid_shuffle_.size() - 1));
  }

  std::cout << "Added new keyframe " << keyframe->fid_ << " to spatial cluster "
            << nearest_cluster << " (now has "
            << clusters_[nearest_cluster].size() << " keyframes)" << std::endl;
}

// Notify that a new keyframe was added
void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  if (!kfid_shuffled_ || clusters_.empty() || cluster_centers_.empty()) {
    // No clusters yet, need to do a full clustering
    kfid_shuffled_ = false;
    return;
  }

  // Increment counter for new keyframes
  keyframes_since_last_full_clustering_++;

  // If we've added too many new keyframes, do a full reclustering
  if (keyframes_since_last_full_clustering_ >= RECLUSTER_THRESHOLD) {
    std::cout << "Reached " << keyframes_since_last_full_clustering_
              << " new keyframes, performing full reclustering" << std::endl;

    kfid_shuffled_ = false;
    return;
  }

  // Otherwise, just add to existing clusters
  addKeyframeToExistingClusters(keyframe);
}

// Modified to handle spatial clusters and incremental updates
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (!kfid_shuffled_) {
    generateSpatiallyCoherentBatches();
    // Clear the queue so we incorporate new keyframes immediately
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }
    fillQueue();
  }

  if (keyframe_queue_.empty()) {
    fillQueue();
    if (keyframe_queue_.empty()) return nullptr;
  }

  auto next_kf = keyframe_queue_.front();
  keyframe_queue_.pop();

  // Update usage statistics
  auto viewpoint_fid = next_kf->fid_;
  if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
    kfs_used_times_[viewpoint_fid] = 1;
  else
    ++kfs_used_times_[viewpoint_fid];

  // Decrease remaining times of use
  --(next_kf->remaining_times_of_use_);

  // Increment the cluster iterations counter
  cluster_iterations_++;

  // Refill queue if running low
  if (keyframe_queue_.size() < queue_size_ / 2) {
    fillQueue();
  }

  return next_kf;
}

// Look ahead without modifying queue
std::vector<std::shared_ptr<GaussianKeyframe>>
KeyframeQueue::peekUpcomingKeyframes(size_t count) {
  if (!kfid_shuffled_) {
    generateSpatiallyCoherentBatches();
    // Clear the queue so we incorporate new keyframes immediately
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }
    fillQueue();
  }

  if (keyframe_queue_.empty()) {
    fillQueue();
  }

  std::vector<std::shared_ptr<GaussianKeyframe>> upcoming;
  std::queue<std::shared_ptr<GaussianKeyframe>> temp_queue = keyframe_queue_;
  size_t look_ahead = std::min(count, temp_queue.size());

  for (size_t i = 0; i < look_ahead; i++) {
    upcoming.push_back(temp_queue.front());
    temp_queue.pop();
  }

  return upcoming;
}

// Set the number of iterations per cluster
void KeyframeQueue::setIterationsPerCluster(int iterations) {
  iterations_per_cluster_ = std::max(1, iterations);
  std::cout << "Set iterations per cluster to " << iterations_per_cluster_
            << std::endl;
}

// Get current cluster index
int KeyframeQueue::getCurrentClusterIndex() const { return current_cluster_; }

// Get number of clusters
int KeyframeQueue::getClusterCount() const { return clusters_.size(); }

// Force switch to next cluster
void KeyframeQueue::forceNextCluster() {
  if (clusters_.empty()) return;

  current_cluster_ = (current_cluster_ + 1) % clusters_.size();
  kfid_shuffle_ = clusters_[current_cluster_];
  kfid_shuffle_idx_ = 0;
  cluster_iterations_ = 0;

  // Clear queue to force refill from new cluster
  while (!keyframe_queue_.empty()) {
    keyframe_queue_.pop();
  }

  std::cout << "Forced switch to spatial cluster " << current_cluster_
            << " with " << kfid_shuffle_.size() << " keyframes" << std::endl;

  // Pre-warm cache for the new cluster
  prewarmClusterCache();
}