#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations
class ChunkManager;
class GaussianKeyframe;

class KeyframeSelector {
 public:
  KeyframeSelector(std::shared_ptr<ChunkManager> chunk_manager);

  // Select a keyframe using a smart strategy with needed data passed as
  // parameters
  std::shared_ptr<GaussianKeyframe> selectKeyframe(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
      const std::map<std::size_t, float>& keyframe_losses,
      const std::map<std::size_t, int>& keyframe_usage_counts,
      int current_iteration);

  // Predict upcoming keyframes
  std::vector<std::shared_ptr<GaussianKeyframe>> predictUpcomingKeyframes(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
      const std::map<std::size_t, float>& keyframe_losses,
      int count = 5);

  // Get spatial cluster of keyframe
  int getKeyframeCluster(std::size_t keyframe_id);

 private:
  // Now only stores ChunkManager (no GaussianMapper reference)
  std::shared_ptr<ChunkManager> chunk_manager_;

  // Keyframe selection state
  std::vector<std::size_t> keyframe_shuffle_;
  std::size_t shuffle_idx_ = 0;
  bool shuffle_initialized_ = false;

  // Spatial clustering of keyframes
  std::unordered_map<std::size_t, int> keyframe_clusters_;
  std::vector<std::vector<std::size_t>> cluster_keyframes_;
  int current_cluster_ = 0;
  int cluster_change_countdown_ = 0;

  // RNG
  std::mt19937 rng_{std::random_device{}()};

  // Last used keyframe and time
  std::size_t last_keyframe_id_ = 0;
  std::chrono::steady_clock::time_point last_selection_time_ =
      std::chrono::steady_clock::now();

  // Helper methods - updated to accept data as parameters
  float computeSelectionScore(
      std::shared_ptr<GaussianKeyframe> keyframe,
      const std::map<std::size_t, float>& keyframe_losses,
      const std::map<std::size_t, int>& keyframe_usage_counts);

  void clusterKeyframesByPosition(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
          keyframes);

  void refreshClusters(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
      const std::map<std::size_t, float>& keyframe_losses);

  void generateKeyframeShuffles(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
          keyframes);

  bool isSpatiallyClose(std::shared_ptr<GaussianKeyframe> kf1,
                        std::shared_ptr<GaussianKeyframe> kf2,
                        float threshold);

  // Selection strategies - updated to accept data as parameters
  std::shared_ptr<GaussianKeyframe> selectWithRandom(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
          keyframes);

  std::shared_ptr<GaussianKeyframe> selectWithSpatialLocality(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
          keyframes);

  std::shared_ptr<GaussianKeyframe> selectWithLRU(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
      const std::map<std::size_t, int>& keyframe_usage_counts);

  std::shared_ptr<GaussianKeyframe> selectWithHighLoss(
      const std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes,
      const std::map<std::size_t, float>& keyframe_losses);

  // Calculate spatial distance between keyframes
  float keyframeDistance(std::shared_ptr<GaussianKeyframe> kf1,
                         std::shared_ptr<GaussianKeyframe> kf2);
};