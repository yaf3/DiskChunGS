#include "include/keyframe_selection.h"

void KeyframeQueue::generateKfidRandomShuffle() {
  if (scene_->keyframes().empty()) return;

  std::size_t nkfs = scene_->keyframes().size();
  kfid_shuffle_.resize(nkfs);
  std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
  std::mt19937 g(std::random_device{}());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
}

void KeyframeQueue::fillQueue() {
  // Keep filling until we reach desired size or run out of options

  if (!kfid_shuffled_) generateKfidRandomShuffle();

  while (keyframe_queue_.size() < queue_size_) {
    int start_shuffle_idx = kfid_shuffle_idx_;
    std::shared_ptr<GaussianKeyframe> next_kf = nullptr;

    do {
      // Move to next index, wrapping if needed
      kfid_shuffle_idx_ = (kfid_shuffle_idx_ + 1) % kfid_shuffle_.size();

      // If we've checked all keyframes and found none with uses left
      if (kfid_shuffle_idx_ == start_shuffle_idx) {
        // Add 1 time of use to all keyframes
        for (auto& kfit : scene_->keyframes()) {
          kfit.second->remaining_times_of_use_ += 1;
        }

        // Optional: implement auto_distribute logic here
      }

      // Get keyframe at current shuffle index
      int random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
      auto it = scene_->keyframes().begin();
      std::advance(
          it, std::min<size_t>(random_cam_idx, scene_->keyframes().size() - 1));
      next_kf = it->second;

    } while (next_kf->remaining_times_of_use_ <= 0);

    // Add usable keyframe to queue
    if (next_kf && next_kf->remaining_times_of_use_ > 0) {
      keyframe_queue_.push(next_kf);
    } else {
      break;
    }
  }
}

// Get next keyframe (dequeues it)
std::shared_ptr<GaussianKeyframe> KeyframeQueue::getNextKeyframe() {
  if (!kfid_shuffled_) {
    generateKfidRandomShuffle();
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
    generateKfidRandomShuffle();
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