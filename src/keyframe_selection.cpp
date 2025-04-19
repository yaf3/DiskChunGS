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
      iterations_per_cluster_(200),
      keyframes_since_last_full_clustering_(0),
      cluster_centers_adapter_(nullptr),
      cluster_kdtree_(nullptr) {
  // KD-tree will be initialized when we have cluster centers
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
  kfid_shuffle_.reserve(
      scene_->keyframes().size());  // Pre-allocate for better performance

  for (const auto& [fid, _] : scene_->keyframes()) {
    kfid_shuffle_.push_back(fid);
  }

  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
  std::cout << "Generated random shuffle of " << kfid_shuffle_.size()
            << " keyframes" << std::endl;
}

// New spatial clustering method with KD-tree optimization
void KeyframeQueue::generateSpatiallyCoherentBatches() {
  auto timer = ProfilingUtils::Timer("generateSpatiallyCoherentBatches");

  if (scene_->keyframes().empty()) return;

  std::cout << "Generating spatially coherent keyframe batches..." << std::endl;

  // Step 1: Extract keyframe positions
  std::vector<std::pair<std::size_t, Eigen::Vector3f>> keyframe_positions;
  keyframe_positions.reserve(scene_->keyframes().size());

  // Clear position cache and rebuild it
  keyframe_positions_cache_.clear();
  keyframe_positions_cache_.reserve(scene_->keyframes().size());

  for (const auto& [fid, keyframe] : scene_->keyframes()) {
    if (!keyframe->set_pose_) continue;  // Skip keyframes without valid poses

    // Get world position of camera (inverse of camera-to-world)
    Sophus::SE3d Twc = keyframe->getPose().inverse();
    Eigen::Vector3f position = Twc.translation().cast<float>();
    keyframe_positions.push_back({fid, position});
    keyframe_positions_cache_[fid] = position;  // Cache the position
  }

  if (keyframe_positions.empty()) {
    std::cout
        << "No valid keyframe positions found, falling back to random shuffle"
        << std::endl;
    generateKfidRandomShuffle();
    return;
  }

  // Step 2: Determine appropriate number of clusters based on scene size
  const int min_keyframes_per_cluster = 15;
  int num_clusters = std::max(1, static_cast<int>(keyframe_positions.size() /
                                                  min_keyframes_per_cluster));

  std::cout << "Creating " << num_clusters << " spatial clusters for "
            << keyframe_positions.size() << " keyframes" << std::endl;

  std::vector<std::vector<std::size_t>> clusters(num_clusters);

  // Initialize cluster centers with furthest point sampling for better
  // distribution
  cluster_centers_.clear();
  cluster_centers_.reserve(
      num_clusters);  // Pre-allocate for better performance
  cluster_centers_.push_back(
      keyframe_positions[0].second);  // Start with first point

  // Furthest point sampling for initial cluster centers
  for (int i = 1; i < num_clusters; i++) {
    float max_dist = -1;
    std::size_t furthest_idx = 0;

    // Process keyframe positions in batches to improve cache efficiency
    for (std::size_t batch_start = 0; batch_start < keyframe_positions.size();
         batch_start += BATCH_SIZE) {
      const std::size_t batch_end =
          std::min(batch_start + BATCH_SIZE, keyframe_positions.size());

      for (std::size_t j = batch_start; j < batch_end; j++) {
        float min_dist = std::numeric_limits<float>::max();

        // Find minimum distance to any existing center
        for (const auto& center : cluster_centers_) {
          float dist =
              (keyframe_positions[j].second - center)
                  .squaredNorm();  // Use squared norm to avoid square root
          min_dist = std::min(min_dist, dist);
        }

        if (min_dist > max_dist) {
          max_dist = min_dist;
          furthest_idx = j;
        }
      }
    }

    cluster_centers_.push_back(keyframe_positions[furthest_idx].second);
  }

  // Initialize nanoflann KD-tree with cluster centers for efficient nearest
  // neighbor search
  cluster_centers_adapter_ =
      std::make_unique<ClusterCentersAdapter>(cluster_centers_);
  cluster_kdtree_ = std::make_unique<ClusterKDTree>(
      3,  // dim
      *cluster_centers_adapter_,
      nanoflann::KDTreeSingleIndexAdaptorParams(10)  // max leaf size
  );
  cluster_kdtree_->buildIndex();

  // Assign keyframes to nearest cluster using KD-tree
  for (const auto& [fid, position] : keyframe_positions) {
    int nearest_cluster = findNearestCluster(position);
    clusters[nearest_cluster].push_back(fid);
  }

  // Step 3: Create queue of clusters, and shuffle keyframes within each cluster
  kfid_shuffle_.clear();
  clusters_.clear();
  clusters_.reserve(num_clusters);

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

    std::cout << "Switching to spatial cluster " << current_cluster_ << " with "
              << kfid_shuffle_.size() << " keyframes" << std::endl;

    // When switching clusters, clear the queue for fresh keyframes
    while (!keyframe_queue_.empty()) {
      keyframe_queue_.pop();
    }

    // Pre-warm cache for new cluster
    prewarmClusterCache();
    return;  // Return after switching clusters and prewarming cache
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

// Helper to find nearest cluster for a keyframe - optimized with nanoflann
// KD-tree
int KeyframeQueue::findNearestCluster(const Eigen::Vector3f& position) const {
  if (cluster_centers_.empty()) return 0;
  if (!cluster_kdtree_) return 0;

  // Use nanoflann KD-tree for nearest neighbor search - O(log n) instead of
  // O(n)
  const float query_point[3] = {position(0), position(1), position(2)};

  // Find nearest neighbor
  size_t index;
  float distance_squared;

  nanoflann::KNNResultSet<float> resultSet(1);
  resultSet.init(&index, &distance_squared);

  // Search for the nearest neighbor
  cluster_kdtree_->findNeighbors(resultSet, query_point,
                                 nanoflann::SearchParameters(10));

  return static_cast<int>(index);
}

// Add a single keyframe to existing clusters - optimized
void KeyframeQueue::addKeyframeToExistingClusters(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe || !keyframe->set_pose_ || clusters_.empty() ||
      cluster_centers_.empty()) {
    return;
  }

  // Get keyframe position
  Sophus::SE3d Twc = keyframe->getPose().inverse();
  Eigen::Vector3f position = Twc.translation().cast<float>();

  // Cache the position
  keyframe_positions_cache_[keyframe->fid_] = position;

  // Find nearest cluster using KD-tree
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
  kfs_used_times_[viewpoint_fid]++;  // Simplified, unordered_map handles
                                     // non-existent keys

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
  upcoming.reserve(std::min(count, keyframe_queue_.size()));  // Pre-allocate

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

void KeyframeQueue::visualizeClusterCenters(const std::string& output_file,
                                            int width,
                                            int height) {
  if (cluster_centers_.empty()) {
    std::cerr << "No cluster centers to visualize." << std::endl;
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

  // Check cluster centers
  for (const auto& center : cluster_centers_) {
    min_x = std::min(min_x, center(dim1));
    max_x = std::max(max_x, center(dim1));
    min_y = std::min(min_y, center(dim2));
    max_y = std::max(max_y, center(dim2));
  }

  // Check keyframe positions
  for (const auto& [_, pos] : keyframe_positions_cache_) {
    min_x = std::min(min_x, pos(dim1));
    max_x = std::max(max_x, pos(dim1));
    min_y = std::min(min_y, pos(dim2));
    max_y = std::max(max_y, pos(dim2));
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

  for (size_t i = 0; i < cluster_centers_.size(); i++) {
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

  // For each keyframe, find the nearest cluster center
  for (const auto& [kf_id, pos] : keyframe_positions_cache_) {
    int nearest_center_idx = findNearestCluster(pos);
    if (nearest_center_idx >= 0 &&
        nearest_center_idx < cluster_centers_.size()) {
      keyframe_to_cluster_center[kf_id] = nearest_center_idx;
    }
  }

  // Draw keyframes as small dots
  svg_file << "  <!-- Keyframes -->" << std::endl;
  svg_file << "  <g>" << std::endl;

  // Draw each keyframe with its cluster center's color
  for (const auto& [kf_id, pos] : keyframe_positions_cache_) {
    float x = scale_x(pos(dim1));
    float y = scale_y(pos(dim2));

    // Get cluster center index for this keyframe
    std::string color = "#aaaaaa";  // Default gray color
    auto it = keyframe_to_cluster_center.find(kf_id);
    if (it != keyframe_to_cluster_center.end() && it->second < colors.size()) {
      color = colors[it->second];
    }

    svg_file << "    <circle cx=\"" << x << "\" cy=\"" << y
             << "\" r=\"2\" fill=\"" << color << "\" />" << std::endl;
  }
  svg_file << "  </g>" << std::endl;

  // Draw cluster centers as larger circles
  svg_file << "  <!-- Cluster Centers -->" << std::endl;
  for (size_t i = 0; i < cluster_centers_.size(); i++) {
    const auto& center = cluster_centers_[i];
    float x = scale_x(center(dim1));
    float y = scale_y(center(dim2));

    // Draw the cluster center
    svg_file << "  <g>" << std::endl;
    svg_file << "    <circle cx=\"" << x << "\" cy=\"" << y
             << "\" r=\"8\" fill=\"" << colors[i]
             << "\" stroke=\"black\" stroke-width=\"1\"/>" << std::endl;

    // Add label
    svg_file
        << "    <text x=\"" << x << "\" y=\"" << y - 10
        << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">"
        << "C" << i << "</text>" << std::endl;
    svg_file << "  </g>" << std::endl;
  }

  // Add legend
  svg_file << "  <!-- Legend -->" << std::endl;
  svg_file << "  <g transform=\"translate(" << (width - 120) << ", 20)\">"
           << std::endl;
  svg_file << "    <rect x=\"0\" y=\"0\" width=\"110\" height=\""
           << (30 + 20 * cluster_centers_.size())
           << "\" fill=\"white\" stroke=\"black\" stroke-width=\"1\"/>"
           << std::endl;
  svg_file << "    <text x=\"5\" y=\"20\" font-family=\"Arial\" "
              "font-size=\"12\" font-weight=\"bold\">Clusters</text>"
           << std::endl;

  for (size_t i = 0; i < cluster_centers_.size(); i++) {
    float y_pos = 40 + i * 20;
    svg_file << "    <circle cx=\"15\" cy=\"" << y_pos - 5
             << "\" r=\"5\" fill=\"" << colors[i] << "\"/>" << std::endl;
    svg_file << "    <text x=\"30\" y=\"" << y_pos
             << "\" font-family=\"Arial\" font-size=\"12\">Cluster " << i
             << " (" << clusters_[i].size() << ")</text>" << std::endl;
  }
  svg_file << "  </g>" << std::endl;

  // End SVG
  svg_file << "</svg>" << std::endl;
  svg_file.close();

  std::cout << "Generated cluster visualization at: " << output_file
            << std::endl;
}