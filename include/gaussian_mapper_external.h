/**
 * This file is part of DiskChunGS.
 *
 * Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See <http://www.gnu.org/licenses/>.
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
