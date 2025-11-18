/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

// Frame structure for external pose mode
struct Frame {
  cv::Mat rgb_image;
  cv::Mat depth_image;
  Sophus::SE3f pose;
  double timestamp;

  Frame(const cv::Mat& rgb,
        const cv::Mat& depth,
        const Sophus::SE3f& p,
        double ts);
};

// Leaky frame queue for external pose mode
class LeakyFrameQueue {
 public:
  explicit LeakyFrameQueue(size_t max_size = 20);
  void push(Frame&& frame);
  std::optional<Frame> pop(bool wait = true);
  void stop();
  bool empty() const;
  size_t size() const;

 private:
  std::deque<Frame> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  const size_t max_size_;
  bool stopped_{false};
};
