#include "include/keyframe_selection.h"

#include <algorithm>
#include <iostream>

// Constructor
KeyframeQueue::KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                             size_t queue_size,
                             float similarity_threshold,
                             int auto_distribute,
                             const std::map<std::size_t, float>* loss_map)
    : scene_(scene),
      queue_size_(queue_size),
      recent_keyframes_count_(queue_size),  // Use queue_size as default for k
      rng_(std::random_device{}()) {
  std::cout << "Created SimpleKeyframeQueue with k=" << recent_keyframes_count_
            << " most recent keyframes" << std::endl;
}

// These methods are just stubs for compatibility with the original interface
void KeyframeQueue::setChunkManager(
    std::shared_ptr<ChunkManager> chunk_manager) {
  chunk_manager_ = chunk_manager;
}

// Get next keyframe (randomly selected from k most recent)
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (keyframe_ids_.empty()) return nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, keyframe_ids_.size() - 1);

  int randomIndex = distrib(gen);
  std::size_t kf_id = keyframe_ids_[randomIndex];

  auto it = scene_->keyframes().find(kf_id);

  return it->second;
}

void KeyframeQueue::notifyNewKeyframeAdded(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  if (!keyframe) return;
  std::unique_lock<std::mutex> lock(mutex_new_keyframe_);

  // Add to the list of keyframe IDs
  if (keyframe_ids_.size() >= queue_size_) {
    // Remove the oldest keyframe ID and erase it from scene
    std::size_t oldest_kf_id = keyframe_ids_.front();
    keyframe_ids_.erase(keyframe_ids_.begin());

    // // Erase the keyframe from the scene entirely, but only if it exists
    auto it = scene_->keyframes().find(oldest_kf_id);
    if (it != scene_->keyframes().end()) {
      it->second->saveDataToDisk();
      // scene_->keyframes().erase(oldest_kf_id);
      // std::cout << "Erased keyframe " << oldest_kf_id
      //           << " from scene (no longer in recent k)" << std::endl;
    }
  }
  keyframe_ids_.push_back(keyframe->fid_);
  // std::cout << "Added keyframe " << keyframe->fid_
  //           << " to queue, queue size now: " << keyframe_ids_.size()
  //           << std::endl;
}