#include "include/keyframe_selection.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>

#include "include/profiling.h"

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t queue_size,
                             float similarity_threshold,
                             int auto_distribute,
                             const std::map<std::size_t, float>* loss_map)
    : scene_(scene),
      queue_size_(queue_size),
      kfid_shuffled_(false),
      kfid_shuffle_idx_(0),
      current_cluster_(0),
      cluster_iterations_(0),
      iterations_per_cluster_(200),
      keyframes_since_last_full_clustering_(0),
      similarity_threshold_(similarity_threshold),
      auto_distribute_k_factor_(auto_distribute),
      kfs_loss_ptr_(loss_map) {
  // No initialization needed for visibility-based clustering yet
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
  kfid_shuffle_.reserve(scene_->keyframes().size());

  for (const auto& [fid, _] : scene_->keyframes()) {
    kfid_shuffle_.push_back(fid);
  }

  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
  std::cout << "Generated random shuffle of " << kfid_shuffle_.size()
            << " keyframes" << std::endl;
}

// Get or compute visible chunks for a keyframe
std::vector<ChunkCoord> KeyframeQueue::getOrComputeVisibleChunks(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe || !keyframe->set_pose_ || !chunk_manager_) {
    return {};
  }

  // Simply use ChunkManager's frustumCullChunks which has its own cache
  // The true parameter enables the cache
  return chunk_manager_->frustumCullChunks(keyframe, true);
}

// Compute similarity between keyframes based on chunk visibility overlap
float KeyframeQueue::computeChunkOverlapSimilarity(std::size_t kf1_id,
                                                   std::size_t kf2_id) {
  auto it1 = scene_->keyframes().find(kf1_id);
  auto it2 = scene_->keyframes().find(kf2_id);

  if (it1 == scene_->keyframes().end() || it2 == scene_->keyframes().end() ||
      !it1->second || !it2->second) {
    return 0.0f;
  }

  // Get visible chunks for both keyframes
  std::vector<ChunkCoord> chunks1 = getOrComputeVisibleChunks(it1->second);
  std::vector<ChunkCoord> chunks2 = getOrComputeVisibleChunks(it2->second);

  // If either list is empty, there's no overlap
  if (chunks1.empty() || chunks2.empty()) return 0.0f;

  // Count the overlap using a set for efficient lookups
  std::unordered_set<ChunkCoord, ChunkCoordHash> chunks1_set(chunks1.begin(),
                                                             chunks1.end());
  int overlap = 0;

  for (const auto& chunk : chunks2) {
    if (chunks1_set.count(chunk) > 0) {
      overlap++;
    }
  }

  // Jaccard similarity: |A ∩ B| / |A ∪ B|
  int total_unique = chunks1.size() + chunks2.size() - overlap;
  if (total_unique == 0) return 0.0f;  // Avoid division by zero

  return static_cast<float>(overlap) / total_unique;
}

// New method for visibility-based clustering
void KeyframeQueue::generateVisibilityBasedClusters() {
  auto start_time = std::chrono::steady_clock::now();
  auto timer = ProfilingUtils::Timer("generateVisibilityBasedClusters");

  if (scene_->keyframes().empty()) return;

  std::cout << "Generating visibility-based keyframe clusters..." << std::endl;

  // First, ensure we have visibility data for all keyframes
  std::vector<std::size_t> valid_keyframe_ids;
  for (const auto& [kf_id, keyframe] : scene_->keyframes()) {
    if (!keyframe->set_pose_) continue;  // Skip keyframes without valid poses

    // Pre-compute and cache visible chunks
    getOrComputeVisibleChunks(keyframe);
    valid_keyframe_ids.push_back(kf_id);
  }

  if (valid_keyframe_ids.empty()) {
    std::cout << "No valid keyframes found, falling back to random shuffle"
              << std::endl;
    generateKfidRandomShuffle();
    return;
  }

  // For larger datasets, we'll use a more efficient approach:
  // 1. Start with each keyframe in its own cluster
  // 2. Iteratively merge the two most similar clusters

  // Initialize each keyframe as its own cluster
  std::vector<std::vector<std::size_t>> working_clusters;
  for (std::size_t kf_id : valid_keyframe_ids) {
    working_clusters.push_back({kf_id});
  }

  // Merge until best similarity is below threshold
  while (working_clusters.size() > 1) {
    float best_similarity = -1.0f;
    int best_i = -1, best_j = -1;

    // Find the two most similar clusters
    for (size_t i = 0; i < working_clusters.size(); i++) {
      for (size_t j = i + 1; j < working_clusters.size(); j++) {
        // Compute average similarity between all pairs of keyframes in the two
        // clusters
        float total_similarity = 0.0f;
        int pairs_compared = 0;

        // To avoid O(n²) comparisons for large clusters, sample a limited
        // number of pairs
        const int max_pairs_to_check = 9;  // 3x3 pairs
        int pairs_i = std::min(static_cast<int>(working_clusters[i].size()), 3);
        int pairs_j = std::min(static_cast<int>(working_clusters[j].size()), 3);

        for (int ii = 0; ii < pairs_i; ii++) {
          std::size_t kf1_id =
              working_clusters[i][ii * working_clusters[i].size() / pairs_i];
          for (int jj = 0; jj < pairs_j; jj++) {
            std::size_t kf2_id =
                working_clusters[j][jj * working_clusters[j].size() / pairs_j];
            total_similarity += computeChunkOverlapSimilarity(kf1_id, kf2_id);
            pairs_compared++;
          }
        }

        float avg_similarity =
            pairs_compared > 0 ? total_similarity / pairs_compared : 0.0f;

        if (avg_similarity > best_similarity) {
          best_similarity = avg_similarity;
          best_i = i;
          best_j = j;
        }
      }
    }

    // If the best similarity is too low, stop merging
    if (best_similarity < similarity_threshold_ || best_i < 0 || best_j < 0) {
      break;
    }

    // Merge the two most similar clusters
    working_clusters[best_i].insert(working_clusters[best_i].end(),
                                    working_clusters[best_j].begin(),
                                    working_clusters[best_j].end());

    // Remove the now-merged cluster
    working_clusters.erase(working_clusters.begin() + best_j);
  }

  // Shuffle the order of clusters
  std::mt19937 g(std::random_device{}());
  std::shuffle(working_clusters.begin(), working_clusters.end(), g);

  // Save the new clusters
  clusters_ = working_clusters;

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
    // Shuffle the keyframes within the current cluster
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
  }

  kfid_shuffled_ = true;
  kfid_shuffle_idx_ = 0;

  // Reset new keyframe counter
  keyframes_since_last_full_clustering_ = 0;

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // Print cluster information
  std::cout << "Created " << clusters_.size()
            << " visibility-based clusters in " << duration.count()
            << "ms with:";
  for (size_t i = 0; i < clusters_.size(); i++) {
    std::cout << " [" << i << "]:" << clusters_[i].size();
  }
  std::cout << " keyframes" << std::endl;

  // Pre-warm cache for current cluster
  // preloadClusterChunks();

  cullSmallClusters(3);
}

// Helper to pre-warm cache for current cluster
void KeyframeQueue::preloadClusterChunks() {
  if (!chunk_manager_ || clusters_.empty() ||
      current_cluster_ >= clusters_.size()) {
    return;
  }

  std::cout << "Pre-warming cache for cluster " << current_cluster_ << "..."
            << std::endl;

  // Get unique visible chunks for this cluster
  std::unordered_set<ChunkCoord, ChunkCoordHash> cluster_chunks;

  for (std::size_t kf_id : clusters_[current_cluster_]) {
    auto it = scene_->keyframes().find(kf_id);
    if (it != scene_->keyframes().end() && it->second) {
      std::vector<ChunkCoord> kf_chunks = getOrComputeVisibleChunks(it->second);
      for (const auto& chunk : kf_chunks) {
        if (chunk_manager_->chunkExists(chunk)) {
          cluster_chunks.insert(chunk);
        }
      }
    }
  }

  // Preload a limited number of chunks
  int preloaded = 0;

  for (const auto& chunk_coord : cluster_chunks) {
    if (preloaded >= MAX_PRELOAD_CHUNKS) break;

    // Use cache and skip busy chunks
    chunk_manager_->loadChunkAsync(chunk_coord, 3, false, true);
    preloaded++;
  }

  std::cout << "Preloaded " << preloaded << " chunks for cluster "
            << current_cluster_ << std::endl;
}

// Modified fillQueue method to use loss prioritization when all keyframes in a
// cluster are exhausted
void KeyframeQueue::fillQueue() {
  // Check if we should generate clusters
  if (!kfid_shuffled_) {
    generateVisibilityBasedClusters();
    // Clear the queue so we incorporate new keyframes immediately
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }
  }

  // Check if we should move to the next cluster
  if (cluster_iterations_ >= iterations_per_cluster_ && !clusters_.empty()) {
    cluster_iterations_ = 0;
    current_cluster_ = (current_cluster_ + 1) % clusters_.size();

    // Switch to the next cluster
    kfid_shuffle_ = clusters_[current_cluster_];
    kfid_shuffle_idx_ = 0;

    // Shuffle the keyframes within the cluster too
    std::mt19937 g(std::random_device{}());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    // std::cout << "Switching to visibility cluster " << current_cluster_
    //           << " with " << kfid_shuffle_.size() << " keyframes" <<
    //           std::endl;

    // When switching clusters, clear the queue for fresh keyframes
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }

    return;  // Return after switching clusters
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

        // Apply loss-based prioritization here (moved from queue empty check)
        // Only do this when we've exhausted all keyframes in the cluster
        if ((auto_distribute_k_factor_ > 0) && kfs_loss_ptr_ &&
            !kfs_loss_ptr_->empty()) {
          // Create a vector of keyframe ID and loss pairs (only for current
          // cluster)
          std::vector<std::pair<std::size_t, float>> loss_pairs;

          // Only collect losses for keyframes in the current cluster
          for (auto fid : kfid_shuffle_) {
            auto loss_it = kfs_loss_ptr_->find(fid);
            if (loss_it != kfs_loss_ptr_->end()) {
              loss_pairs.push_back(*loss_it);
            }
          }

          if (!loss_pairs.empty()) {
            // Calculate k (number of keyframes to prioritize)
            int k = std::max(1, static_cast<int>(loss_pairs.size() /
                                                 auto_distribute_k_factor_));

            // Use nth_element instead of sort for better efficiency - just like
            // GaussianMapper
            std::nth_element(loss_pairs.begin(), loss_pairs.begin() + k,
                             loss_pairs.end(),
                             [](const std::pair<std::size_t, float>& a,
                                const std::pair<std::size_t, float>& b) {
                               return a.second > b.second;
                             });

            // Give extra usage times to high-loss keyframes
            for (int i = 0; i < k && i < loss_pairs.size(); ++i) {
              auto it = scene_->keyframes().find(loss_pairs[i].first);
              if (it != scene_->keyframes().end() && it->second) {
                // Add additional usage time
                it->second->remaining_times_of_use_ += 1;

                // std::cout << "Added extra usage time to high-loss keyframe "
                //           << loss_pairs[i].first
                //           << " (loss: " << loss_pairs[i].second << ")"
                //           << std::endl;
              }
            }
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

  // Remove the original loss-based prioritization that happened on empty queue
  // This has been moved to the cluster exhaustion point above
}

// Add a single keyframe to existing clusters
void KeyframeQueue::addKeyframeToExistingClusters(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe || !keyframe->set_pose_ || clusters_.empty()) {
    return;
  }

  auto start_time = std::chrono::steady_clock::now();

  // Get visible chunks for this keyframe
  std::vector<ChunkCoord> visible_chunks = getOrComputeVisibleChunks(keyframe);

  // Find the best cluster to add this keyframe to
  float best_similarity = -1.0f;
  int best_cluster = -1;

  for (size_t i = 0; i < clusters_.size(); i++) {
    float total_similarity = 0.0f;
    int comparisons = 0;

    // Sample a few keyframes from the cluster for comparison
    const int max_samples = 5;  // Limit the number of comparisons
    int step = std::max(1, static_cast<int>(clusters_[i].size() / max_samples));

    for (size_t j = 0; j < clusters_[i].size(); j += step) {
      std::size_t kf_id = clusters_[i][j];
      total_similarity += computeChunkOverlapSimilarity(keyframe->fid_, kf_id);
      comparisons++;
    }

    float avg_similarity =
        comparisons > 0 ? total_similarity / comparisons : 0.0f;

    if (avg_similarity > best_similarity) {
      best_similarity = avg_similarity;
      best_cluster = i;
    }
  }

  // Add to the best cluster or create a new one if none is suitable
  if (best_cluster >= 0 && best_similarity > similarity_threshold_) {
    clusters_[best_cluster].push_back(keyframe->fid_);

    // If we're adding to the current cluster, update the shuffle
    if (best_cluster == current_cluster_) {
      kfid_shuffle_ = clusters_[current_cluster_];
      // Preserve the current index if possible
      kfid_shuffle_idx_ = std::min(kfid_shuffle_idx_,
                                   static_cast<int>(kfid_shuffle_.size() - 1));
    }

    // std::cout << "Added new keyframe " << keyframe->fid_
    //           << " to visibility cluster " << best_cluster << " (now has "
    //           << clusters_[best_cluster].size() << " keyframes)" <<
    //           std::endl;
  } else {
    // Create a new cluster for this keyframe
    clusters_.push_back({keyframe->fid_});

    // std::cout << "Created new visibility cluster " << (clusters_.size() - 1)
    //           << " for keyframe " << keyframe->fid_ << std::endl;
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // Print cluster information
  // std::cout << "Adding keyframe to queue took " << duration.count() << "ms"
  //           << std::endl;
}

// Notify that a new keyframe was added
void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;

  if (!kfid_shuffled_ || clusters_.empty()) {
    // No clusters yet, need to do a full clustering
    kfid_shuffled_ = false;
    return;
  }

  // Increment counter for new keyframes
  keyframes_since_last_full_clustering_++;

  // // If we've added too many new keyframes, do a full reclustering
  // if (keyframes_since_last_full_clustering_ >= RECLUSTER_THRESHOLD) {
  //   std::cout << "Reached " << keyframes_since_last_full_clustering_
  //             << " new keyframes, performing full reclustering" << std::endl;

  //   kfid_shuffled_ = false;
  //   return;
  // }

  // Otherwise, just add to existing clusters
  addKeyframeToExistingClusters(keyframe);

  if (keyframes_since_last_full_clustering_ % 15 == 0) {
    cullSmallClusters(3);
  }
}

// Modified to handle visibility-based clusters
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (!kfid_shuffled_) {
    generateVisibilityBasedClusters();
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
  kfs_used_times_[viewpoint_fid]++;

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
    generateVisibilityBasedClusters();
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
  upcoming.reserve(std::min(count, keyframe_queue_.size()));

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

  // Shuffle the keyframes within the current cluster
  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  // Clear queue to force refill from new cluster
  while (!keyframe_queue_.empty()) {
    keyframe_queue_.pop();
  }

  std::cout << "Forced switch to visibility cluster " << current_cluster_
            << " with " << kfid_shuffle_.size() << " keyframes" << std::endl;

  // Pre-warm cache for the new cluster
  // preloadClusterChunks();
}

void KeyframeQueue::cullSmallClusters(int size_threshold) {
  if (clusters_.empty()) return;

  auto start_time = std::chrono::steady_clock::now();
  auto timer = ProfilingUtils::Timer("cullSmallClusters");

  // First, identify small clusters and mark them for culling
  std::vector<bool> cull_marker(clusters_.size(), false);
  std::vector<int> cluster_sizes(clusters_.size());
  int culled_count = 0;

  for (size_t i = 0; i < clusters_.size(); i++) {
    cluster_sizes[i] = clusters_[i].size();
    if (cluster_sizes[i] < size_threshold) {
      cull_marker[i] = true;
      culled_count++;
    }
  }

  if (culled_count == 0) {
    // std::cout << "No small clusters to cull (< " << size_threshold
    //           << " keyframes)" << std::endl;
    return;
  }

  std::cout << "Found " << culled_count << " clusters smaller than "
            << size_threshold << " keyframes to be culled" << std::endl;

  // Now we'll reassign keyframes from small clusters to bigger ones
  std::vector<std::vector<std::size_t>> new_clusters;
  int reassigned_frames = 0;

  // First add all large clusters to the new list
  for (size_t i = 0; i < clusters_.size(); i++) {
    if (!cull_marker[i]) {
      new_clusters.push_back(clusters_[i]);
    }
  }

  // If all clusters are marked for culling, keep at least the largest one
  if (new_clusters.empty()) {
    int largest_idx = 0;
    for (size_t i = 1; i < clusters_.size(); i++) {
      if (cluster_sizes[i] > cluster_sizes[largest_idx]) {
        largest_idx = i;
      }
    }
    cull_marker[largest_idx] = false;
    new_clusters.push_back(clusters_[largest_idx]);
    std::cout << "All clusters were small - keeping the largest one with "
              << cluster_sizes[largest_idx] << " keyframes" << std::endl;
  }

  // For each small cluster, find the best matching large cluster
  for (size_t small_idx = 0; small_idx < clusters_.size(); small_idx++) {
    if (!cull_marker[small_idx]) continue;  // Skip large clusters

    float best_similarity = -1.0f;
    int best_large_idx = 0;

    // Find most similar large cluster
    for (size_t large_idx = 0; large_idx < new_clusters.size(); large_idx++) {
      float total_similarity = 0.0f;
      int comparisons = 0;

      // Compare each keyframe in small cluster with a sample from large cluster
      for (std::size_t small_kf_id : clusters_[small_idx]) {
        // Sample a few frames from the large cluster
        const int max_samples = 5;
        int step = std::max(
            1, static_cast<int>(new_clusters[large_idx].size() / max_samples));

        for (size_t j = 0; j < new_clusters[large_idx].size(); j += step) {
          std::size_t large_kf_id = new_clusters[large_idx][j];
          total_similarity +=
              computeChunkOverlapSimilarity(small_kf_id, large_kf_id);
          comparisons++;
        }
      }

      float avg_similarity =
          comparisons > 0 ? total_similarity / comparisons : 0.0f;

      // Update best match
      if (avg_similarity > best_similarity) {
        best_similarity = avg_similarity;
        best_large_idx = large_idx;
      }
    }

    // Merge small cluster into the best matching large cluster
    std::cout << "Merging cluster " << small_idx << " ("
              << clusters_[small_idx].size() << " keyframes) into cluster "
              << best_large_idx << " with similarity " << best_similarity
              << std::endl;

    new_clusters[best_large_idx].insert(new_clusters[best_large_idx].end(),
                                        clusters_[small_idx].begin(),
                                        clusters_[small_idx].end());

    reassigned_frames += clusters_[small_idx].size();
  }

  // Update the clusters
  clusters_ = new_clusters;

  // If current cluster was culled, reset to the first cluster
  if (current_cluster_ >= clusters_.size()) {
    current_cluster_ = 0;
  }

  // Update the shuffle if needed
  kfid_shuffle_ = clusters_[current_cluster_];
  kfid_shuffle_idx_ =
      std::min(kfid_shuffle_idx_, static_cast<int>(kfid_shuffle_.size() - 1));

  // Shuffle the keyframes within the current cluster
  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  // Reset the queue
  while (!keyframe_queue_.empty()) {
    keyframe_queue_.pop();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "Culled " << culled_count << " small clusters, reassigned "
            << reassigned_frames << " keyframes in " << duration.count()
            << "ms. Now have " << clusters_.size() << " clusters." << std::endl;

  // Pre-warm the cache
  // preloadClusterChunks();
}

void KeyframeQueue::visualizeClusters(const std::string& output_file,
                                      int width,
                                      int height) {
  if (clusters_.empty()) {
    std::cerr << "No clusters to visualize." << std::endl;
    return;
  }

  // Define projection plane (we'll use XZ by default, but you can change this)
  // Options: XY (0,1), XZ (0,2), YZ (1,2)
  int dim1 = 0;  // X
  int dim2 = 2;  // Z

  // Determine bounds of the data for scaling
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();

  // Check keyframe positions
  for (const auto& [fid, keyframe] : scene_->keyframes()) {
    Sophus::SE3d Twc = keyframe->getPose().inverse();
    Eigen::Vector3f position = Twc.translation().cast<float>();
    min_x = std::min(min_x, position(dim1));
    max_x = std::max(max_x, position(dim1));
    min_y = std::min(min_y, position(dim2));
    max_y = std::max(max_y, position(dim2));
  }

  // Add some padding
  float padding = 0.05f;
  float range_x = max_x - min_x;
  float range_y = max_y - min_y;
  min_x -= range_x * padding;
  max_x += range_x * padding;
  min_y -= range_y * padding;
  max_y += range_y * padding;

  // Scale factors to fit within SVG dimensions
  auto scale_x = [&](float x) -> float {
    return width * 0.9f * (x - min_x) / (max_x - min_x) + width * 0.05f;
  };

  auto scale_y = [&](float y) -> float {
    return height * 0.9f * (1.0f - (y - min_y) / (max_y - min_y)) +
           height * 0.05f;
  };

  // Generate random colors for clusters
  std::vector<std::string> colors;
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_int_distribution<int> dist(0, 255);

  for (size_t i = 0; i < clusters_.size(); i++) {
    std::stringstream ss;
    ss << "#";

    // Generate a color that's not too light (for visibility)
    int r = dist(rng) % 200;
    int g = dist(rng) % 200;
    int b = dist(rng) % 200;

    ss << std::hex << std::setfill('0') << std::setw(2) << r;
    ss << std::hex << std::setfill('0') << std::setw(2) << g;
    ss << std::hex << std::setfill('0') << std::setw(2) << b;

    colors.push_back(ss.str());
  }

  // Create SVG file
  std::ofstream svg_file(output_file);
  if (!svg_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_file << std::endl;
    return;
  }

  // Write SVG header
  svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"
           << std::endl;
  svg_file << "<svg width=\"" << width << "\" height=\"" << height
           << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;

  // Add title
  svg_file << "  <title>Cluster Centers Visualization</title>" << std::endl;

  // Add background
  svg_file << "  <rect width=\"100%\" height=\"100%\" fill=\"#f0f0f0\"/>"
           << std::endl;

  // Draw grid lines (optional)
  svg_file << "  <!-- Grid lines -->" << std::endl;
  int grid_steps = 10;
  svg_file << "  <g stroke=\"#cccccc\" stroke-width=\"0.5\">" << std::endl;

  for (int i = 1; i < grid_steps; i++) {
    float pos_x = width * i / static_cast<float>(grid_steps);
    float pos_y = height * i / static_cast<float>(grid_steps);

    // Vertical line
    svg_file << "    <line x1=\"" << pos_x << "\" y1=\"0\" x2=\"" << pos_x
             << "\" y2=\"" << height << "\"/>" << std::endl;

    // Horizontal line
    svg_file << "    <line x1=\"0\" y1=\"" << pos_y << "\" x2=\"" << width
             << "\" y2=\"" << pos_y << "\"/>" << std::endl;
  }
  svg_file << "  </g>" << std::endl;

  // Draw axes labels
  svg_file << "  <!-- Axes labels -->" << std::endl;
  svg_file
      << "  <text x=\"" << width / 2 << "\" y=\"" << height - 10
      << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">"
      << (dim1 == 0 ? "X" : (dim1 == 1 ? "Y" : "Z")) << " Axis</text>"
      << std::endl;
  svg_file
      << "  <text x=\"10\" y=\"" << height / 2
      << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\" "
      << "transform=\"rotate(270 10," << height / 2 << ")\">"
      << (dim2 == 0 ? "X" : (dim2 == 1 ? "Y" : "Z")) << " Axis</text>"
      << std::endl;

  // Create keyframe to cluster CENTER mapping based on nearest distance
  // This is the key fix - we color by actual cluster center, not by cluster
  // index in the shuffled array
  std::unordered_map<std::size_t, size_t> keyframe_to_cluster_center;

  for (size_t cluster_idx = 0; cluster_idx < clusters_.size(); cluster_idx++) {
    for (auto& kf_id : clusters_[cluster_idx]) {
      keyframe_to_cluster_center[kf_id] = cluster_idx;
      std::cout << "Keyframe " << kf_id << " belongs to cluster " << cluster_idx
                << std::endl;
    }
  }

  // Draw keyframes as small dots
  svg_file << "  <!-- Keyframes -->" << std::endl;
  svg_file << "  <g>" << std::endl;

  // Draw each keyframe with its cluster center's color
  for (const auto& [fid, keyframe] : scene_->keyframes()) {
    Sophus::SE3d Twc = keyframe->getPose().inverse();
    Eigen::Vector3f position = Twc.translation().cast<float>();
    float x = scale_x(position(dim1));
    float y = scale_y(position(dim2));

    // Get cluster center index for this keyframe
    std::string color = "#aaaaaa";  // Default gray color
    auto it = keyframe_to_cluster_center.find(fid);
    if (it != keyframe_to_cluster_center.end() && it->second < colors.size()) {
      color = colors[it->second];
    }

    svg_file << "    <circle cx=\"" << x << "\" cy=\"" << y
             << "\" r=\"2\" fill=\"" << color << "\" />" << std::endl;
  }
  svg_file << "  </g>" << std::endl;

  // End SVG
  svg_file << "</svg>" << std::endl;
  svg_file.close();

  std::cout << "Generated cluster visualization at: " << output_file
            << std::endl;
}