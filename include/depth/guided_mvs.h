/*
 * Copyright (C) 2025, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * This file is Derivative Works of On-The-Fly-NVS,
 * modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 *
 * For inquiries contact george.drettakis@inria.fr
 */

#pragma once

#include <torch/torch.h>

#include <utility>
#include <vector>

#include "scene/gaussian_keyframe.h"
/**
 * @brief Multi-view stereo depth estimation guided by monocular depth
 *
 * This class implements GPU-accelerated multi-view stereo depth estimation
 * that uses a coarse monocular depth map as guidance for more accurate
 * depth estimation from multiple camera views.
 */
class GuidedMVS {
 private:
  int n_cams;                ///< Number of cameras
  int num_depth_candidates;  ///< Number of depth candidates to test
  float idepth_range;        ///< Inverse depth range for candidate generation

 public:
  /**
   * @brief Construct a new GuidedMVS object
   *
   * @param num_prev_keyframes Number of previous keyframes to use
   * @param num_depth_candidates Number of depth candidates (default: 16)
   */
  GuidedMVS(int num_prev_keyframes,
            int num_depth_candidates = 16,
            float inverse_depth_range = 2e-1f);

  /**
   * @brief Perform guided multi-view stereo depth estimation
   *
   * @param uv UV coordinates tensor [N, 2]
   * @param refKeyframe Reference keyframe containing feature map and intrinsics
   * @param keyframes Vector of neighboring keyframes
   * @return std::pair<torch::Tensor, torch::Tensor> Depth values and validity
   * mask
   */
  std::pair<torch::Tensor, torch::Tensor> operator()(
      const torch::Tensor& uv,
      const std::shared_ptr<GaussianKeyframe> refKeyframe,
      const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes);

  // Getters
  int getNumCams() const { return n_cams; }
  int getNumDepthCandidates() const { return num_depth_candidates; }
  float getIdepthRange() const { return idepth_range; }

  // Setters
  void setIdepthRange(float range) { idepth_range = range; }
};

struct DebugStats {
  int valid_cams_count;
  int best_cam_idx;
  float max_baseline_dist;
  float initial_idepth;
  float final_depth;
  float min_cost;
  float max_cost;
  float cost_ratio;
  bool quadratic_applied;
  float quadratic_variation;
  int best_candidate_idx;
  bool insufficient_baseline;
  bool out_of_bounds;
};

void analyze_debug_stats(const std::vector<DebugStats>& stats,
                         float idepth_range);

void validateFeatureQuality(
    const std::shared_ptr<GaussianKeyframe>& refKeyframe,
    const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes,
    const torch::Tensor& uv);

void saveDepthMapAsPointCloud(const torch::Tensor& depth_map,
                              const torch::Tensor& intrinsics,
                              const torch::Tensor& image,
                              const std::string& filename,
                              float depth_threshold = 0.1f);