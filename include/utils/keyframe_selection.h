/**
 * This file is part of DiskChunGS, modified from CaRtGS/Photo-SLAM.
 *
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See the GNU General Public License for more details:
 * <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "scene/gaussian_keyframe.h"
#include "scene/gaussian_scene.h"

class KeyframeQueue {
 public:
  KeyframeQueue(std::shared_ptr<GaussianScene> scene,
                float chunk_size = 20.0f,
                const std::map<std::size_t, float>* loss_map = nullptr,
                std::map<std::size_t, int>* used_times_map = nullptr);

  std::shared_ptr<GaussianKeyframe> getNextKeyframe();

  void updateChunkKeyframeMapping(std::shared_ptr<GaussianKeyframe> keyframe,
                                  bool is_new_keyframe = false);

  int getQueueSize() const;

 private:
  std::shared_ptr<GaussianScene> scene_;
  float chunk_size_;
  float chunk_sizes_[3];  // Different levels: FINE, MEDIUM, COARSE
  int64_t current_active_chunk_id_;

  const std::map<std::size_t, float>* loss_map_;
  std::map<std::size_t, int>* used_times_map_;

  std::shared_ptr<GaussianKeyframe> latest_keyframe_;

  // Random number generation
  std::mt19937 rng_;
  std::uniform_real_distribution<float> uniform_dist_;
  std::discrete_distribution<int> level_dist_;

  std::map<int64_t, std::vector<std::shared_ptr<GaussianKeyframe>>>
      chunk_to_keyframes_[3];

  std::deque<std::shared_ptr<GaussianKeyframe>> gpu_queue;
  size_t max_gpu_keyframes_ = 400;

  Eigen::Vector3f tensorToEigen(const torch::Tensor& tensor) const;

  void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> keyframe,
                                  int additional_uses);
};