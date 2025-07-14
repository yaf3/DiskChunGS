#include "include/guided_mvs.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Forward declaration of CUDA kernel wrapper
extern "C" void launch_uvToDepth(const void* uvs,
                                 const void* refFeatMap,
                                 const void* otherFeatMaps,
                                 const void* Rts,
                                 const void* intrinsics_,
                                 const float* idepthMap,
                                 float* depth,
                                 float* idist,
                                 DebugStats* debug_stats,
                                 float range,
                                 int nPts,
                                 int featMapH,
                                 int featMapW,
                                 int depthMapH,
                                 int depthMapW,
                                 int H,
                                 int W,
                                 int num_depth_candidates);

// Constructor implementation
GuidedMVS::GuidedMVS(int num_prev_keyframes,
                     int num_depth_candidates,
                     float inverse_depth_range)
    : n_cams(num_prev_keyframes),
      num_depth_candidates(num_depth_candidates),
      idepth_range(inverse_depth_range) {}

// Operator implementation
std::pair<torch::Tensor, torch::Tensor> GuidedMVS::operator()(
    const torch::Tensor& uv,
    const std::shared_ptr<GaussianKeyframe> refKeyframe,
    const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes) {
  // Input validation
  if (uv.ndimension() != 2 || uv.size(1) != 2) {
    AT_ERROR("uv must have dimensions (num_points, 2)");
  }

  // Ensure tensors are contiguous and on CUDA
  auto uv_cuda = uv.contiguous().cuda();

  // Get relative poses
  std::vector<torch::Tensor> other2ref_list;
  for (const auto& keyframe : keyframes) {
    auto rel_pose = torch::matmul(keyframe->getRT(),
                                  torch::linalg::inv(refKeyframe->getRT()));
    other2ref_list.push_back(
        rel_pose.slice(0, 0, 3).slice(1, 0, 4));  // [:3, :4]
  }
  auto other2ref = torch::stack(other2ref_list, 0).contiguous().cuda();

  // // Debug pose tensors
  // std::cout << "\n=== C++ POSE DEBUG ===" << std::endl;
  // std::cout << "Number of keyframes: " << keyframes.size() << std::endl;
  // std::cout << "other2ref shape: " << other2ref.sizes() << std::endl;
  // std::cout << "other2ref dtype: " << other2ref.dtype() << std::endl;

  // // Print reference keyframe RT
  // auto ref_rt = refKeyframe->getRT();
  // std::cout << "\nReference keyframe RT shape: " << ref_rt.sizes() <<
  // std::endl; std::cout << "Reference keyframe RT:" << std::endl; std::cout <<
  // ref_rt << std::endl; std::cout << "Reference keyframe ID: " <<
  // refKeyframe->fid_ << std::endl;

  // // Print each relative pose
  // for (size_t i = 0; i < keyframes.size(); ++i) {
  //   auto keyframe_rt = keyframes[i]->getRT();
  //   auto ref_rt_inv = torch::linalg::inv(ref_rt);
  //   auto rel_pose = torch::matmul(keyframe_rt, ref_rt_inv);
  //   auto rel_pose_34 = rel_pose.slice(0, 0, 3).slice(1, 0, 4);

  //   std::cout << "\nKeyframe " << i << " RT:" << std::endl;
  //   std::cout << keyframe_rt << std::endl;
  //   std::cout << "nKeyframe ID: " << keyframes[i]->fid_ << std::endl;
  //   std::cout << "Reference RT inverse:" << std::endl;
  //   std::cout << ref_rt_inv << std::endl;
  //   std::cout << "Relative pose (full 4x4):" << std::endl;
  //   std::cout << rel_pose << std::endl;
  //   std::cout << "Relative pose [:3, :4]:" << std::endl;
  //   std::cout << rel_pose_34 << std::endl;
  //   std::cout << "other2ref[" << i << "]:" << std::endl;
  //   std::cout << other2ref[i] << std::endl;

  //   // Check if they match
  //   auto diff = torch::abs(rel_pose_34 - other2ref[i]);
  //   auto max_diff = torch::max(diff);
  //   std::cout << "Max difference: " << max_diff.item<float>() << std::endl;

  //   if (max_diff.item<float>() < 1e-6) {
  //     std::cout << "✓ Matches other2ref[" << i << "]" << std::endl;
  //   } else {
  //     std::cout << "✗ MISMATCH with other2ref[" << i << "]" << std::endl;
  //   }
  // }

  // std::cout << "=== END C++ POSE DEBUG ===" << std::endl;

  // Get feature maps
  auto refFeatMap = refKeyframe->feature_map_.contiguous().cuda();
  std::vector<torch::Tensor> featMaps_list;
  for (const auto& keyframe : keyframes) {
    featMaps_list.push_back(keyframe->feature_map_.cuda().contiguous());
  }
  auto featMaps = torch::stack(featMaps_list, 0);

  refFeatMap = torch::nn::functional::interpolate(
                   refFeatMap.unsqueeze(0),
                   torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{refKeyframe->image_height_,
                                                  refKeyframe->image_width_})
                       .mode(torch::kBilinear)
                       .align_corners(true))
                   .squeeze(0);

  featMaps = torch::nn::functional::interpolate(
      featMaps, torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{refKeyframe->image_height_,
                                               refKeyframe->image_width_})
                    .mode(torch::kBilinear)
                    .align_corners(true));

  // Get intrinsics
  auto intrinsics = torch::tensor({refKeyframe->intr_[0], refKeyframe->intr_[2],
                                   refKeyframe->intr_[3]})
                        .contiguous()
                        .cuda();

  // Get monocular inverse depth
  auto mono_idepth = refKeyframe->idepth_.contiguous().cuda();

  // std::cout << mono_idepth.sizes() << std::endl;
  // std::cout << mono_idepth.mean().item() << std::endl;
  // std::cout << mono_idepth.min().item() << std::endl;
  // std::cout << mono_idepth.max().item() << std::endl;

  // std::cout << "Mono idepth stats: min="
  //           << torch::min(mono_idepth).item<float>()
  //           << " max=" << torch::max(mono_idepth).item<float>()
  //           << " mean=" << torch::mean(mono_idepth).item<float>() <<
  //           std::endl;

  // Initialize output tensors
  auto depth = -torch::ones(
      {uv_cuda.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto idist = -torch::ones(
      {uv_cuda.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor depth_debug =
      torch::nn::functional::interpolate(
          mono_idepth.reciprocal(),
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{refKeyframe->image_height_,
                                         refKeyframe->image_width_})
              .mode(torch::kBilinear)
              .align_corners(true))
          .squeeze(0)
          .squeeze(0);

  // saveDepthMapAsPointCloud(depth_debug, intrinsics,
  //                          refKeyframe->original_image_, "pcd.ply", 0.0f);

  const int P = uv_cuda.size(0);

  DebugStats* d_debug_stats;
  cudaMalloc(&d_debug_stats, P * sizeof(DebugStats));

  // std::cout << refFeatMap.sizes() << std::endl;
  // std::cout << featMaps.sizes() << std::endl;

  // std::cout << "P: " << P << std::endl;
  // std::cout << "refFeatMap.size(0): " << static_cast<int>(refFeatMap.size(1))
  //           << std::endl;
  // std::cout << "refFeatMap.size(1): " << static_cast<int>(refFeatMap.size(2))
  //           << std::endl;
  // std::cout << "mono_idepth.size(-2): "
  //           << static_cast<int>(mono_idepth.size(-2)) << std::endl;
  // std::cout << "mono_idepth.size(-1): "
  //           << static_cast<int>(mono_idepth.size(-1)) << std::endl;
  // std::cout << "refKeyframe->image_height_: "
  //           << static_cast<int>(refKeyframe->original_image_.size(1))
  //           << std::endl;
  // std::cout << "refKeyframe->image_width_: "
  //           << static_cast<int>(refKeyframe->original_image_.size(2))
  //           << std::endl;
  // std::cout << "num_depth_candidates: " << num_depth_candidates << std::endl;

  // Run feature quality validation
  // validateFeatureQuality(refKeyframe, keyframes, uv_cuda);

  if (P != 0) {
    // Type checks to match Python version and kernel expectations
    // TORCH_CHECK(uv_cuda.scalar_type() == torch::kFloat32,
    //             "uv must be float32 ");
    // TORCH_CHECK(refFeatMap.scalar_type() == torch::kFloat16,
    //             "refFeatMap must be float16");
    // TORCH_CHECK(featMaps.scalar_type() == torch::kFloat16,
    //             "featMaps must be float16");
    // TORCH_CHECK(other2ref.scalar_type() == torch::kFloat32,
    //             "other2ref must be float32");
    // TORCH_CHECK(intrinsics.scalar_type() == torch::kFloat32,
    //             "intrinsics must be float32");
    // TORCH_CHECK(mono_idepth.scalar_type() == torch::kFloat32,
    //             "mono_idepth must be float32");

    // Launch CUDA kernel via wrapper
    launch_uvToDepth(uv_cuda.data_ptr<float>(), refFeatMap.data_ptr<at::Half>(),
                     featMaps.data_ptr<at::Half>(), other2ref.data_ptr<float>(),
                     intrinsics.data_ptr<float>(),
                     mono_idepth.data_ptr<float>(), depth.data_ptr<float>(),
                     idist.data_ptr<float>(), d_debug_stats, idepth_range, P,
                     static_cast<int>(refFeatMap.size(1)),
                     static_cast<int>(refFeatMap.size(2)),
                     static_cast<int>(mono_idepth.size(-2)),
                     static_cast<int>(mono_idepth.size(-1)),
                     static_cast<int>(refKeyframe->original_image_.size(1)),
                     static_cast<int>(refKeyframe->original_image_.size(2)),
                     num_depth_candidates);
  }

  std::vector<DebugStats> h_debug_stats(P);
  if (P > 0) {
    cudaMemcpy(h_debug_stats.data(), d_debug_stats, P * sizeof(DebugStats),
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_debug_stats);

  // analyze_debug_stats(h_debug_stats, idepth_range);

  auto valid_mask = idist >= 0;
  return std::make_pair(depth, valid_mask);
};

void validateFeatureQuality(
    const std::shared_ptr<GaussianKeyframe>& refKeyframe,
    const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes,
    const torch::Tensor& uv) {
  std::cout << "\n========== Feature Quality Validation ==========\n";

  // 1. Check feature map dimensions and statistics
  auto refFeatMap = refKeyframe->feature_map_;
  std::cout << "Reference feature map shape: " << refFeatMap.sizes()
            << std::endl;
  std::cout << "Reference feature map dtype: " << refFeatMap.dtype()
            << std::endl;

  // Convert to float for analysis
  auto refFeatFloat = refFeatMap.to(torch::kFloat32);
  auto refFeatCpu = refFeatFloat.cpu();

  std::cout << "Feature statistics:" << std::endl;
  std::cout << "  - Min: " << refFeatCpu.min().item<float>() << std::endl;
  std::cout << "  - Max: " << refFeatCpu.max().item<float>() << std::endl;
  std::cout << "  - Mean: " << refFeatCpu.mean().item<float>() << std::endl;
  std::cout << "  - Std: " << refFeatCpu.std().item<float>() << std::endl;

  // 2. Check for NaN/Inf values
  auto hasNaN = torch::isnan(refFeatCpu).any().item<bool>();
  auto hasInf = torch::isinf(refFeatCpu).any().item<bool>();
  std::cout << "NaN values: " << (hasNaN ? "YES" : "NO") << std::endl;
  std::cout << "Inf values: " << (hasInf ? "YES" : "NO") << std::endl;

  // 3. Check feature variance across spatial dimensions
  auto spatialVar =
      refFeatCpu.var(std::vector<int64_t>{0, 1}).mean().item<float>();
  std::cout << "Average spatial variance: " << spatialVar << std::endl;

  // 4. Check feature normalization
  auto featNorm = torch::norm(refFeatCpu, 2, 2).mean().item<float>();
  std::cout << "Average L2 norm per pixel: " << featNorm << std::endl;

  // 5. Analyze feature similarity between keyframes
  std::cout << "\n--- Inter-keyframe Feature Analysis ---" << std::endl;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    auto otherFeat = keyframes[i]->feature_map_.to(torch::kFloat32).cpu();

    // Check if features are identical (potential bug)
    bool identical = torch::allclose(refFeatCpu, otherFeat, 1e-6);
    std::cout << "Keyframe " << i
              << " identical to reference: " << (identical ? "YES" : "NO")
              << std::endl;

    // Compute feature similarity at same locations
    auto similarity =
        torch::cosine_similarity(refFeatCpu, otherFeat, 2).mean().item<float>();
    std::cout << "Keyframe " << i
              << " average cosine similarity: " << similarity << std::endl;
  }

  // 6. Sample-based feature validation at UV coordinates
  std::cout << "\n--- UV-based Feature Validation ---" << std::endl;

  if (uv.size(0) > 0) {
    auto uvCpu = uv.cpu();
    int numSamples = std::min(100, static_cast<int>(uv.size(0)));

    std::vector<float> featureVariances;
    std::vector<float> featureNorms;

    for (int i = 0; i < numSamples; ++i) {
      float x = uvCpu[i][0].item<float>();
      float y = uvCpu[i][1].item<float>();

      // Convert to feature map coordinates
      int featH = refFeatCpu.size(0);
      int featW = refFeatCpu.size(1);
      int fx = static_cast<int>(x * featW / refKeyframe->image_width_);
      int fy = static_cast<int>(y * featH / refKeyframe->image_height_);

      // Bounds check
      if (fx >= 0 && fx < featW && fy >= 0 && fy < featH) {
        auto refFeatVec = refFeatCpu[fy][fx];

        // Compute feature statistics
        float featNorm = torch::norm(refFeatVec, 2).item<float>();
        featureNorms.push_back(featNorm);

        // Compare with neighboring keyframes
        std::vector<float> similarities;
        for (size_t j = 0; j < keyframes.size(); ++j) {
          auto otherFeat = keyframes[j]->feature_map_.to(torch::kFloat32).cpu();
          if (fx < otherFeat.size(1) && fy < otherFeat.size(0)) {
            auto otherFeatVec = otherFeat[fy][fx];
            float sim = torch::cosine_similarity(refFeatVec, otherFeatVec, 0)
                            .item<float>();
            similarities.push_back(sim);
          }
        }

        if (!similarities.empty()) {
          float avgSim =
              std::accumulate(similarities.begin(), similarities.end(), 0.0f) /
              similarities.size();
          float varSim = 0.0f;
          for (float sim : similarities) {
            varSim += (sim - avgSim) * (sim - avgSim);
          }
          varSim /= similarities.size();
          featureVariances.push_back(varSim);
        }
      }
    }

    if (!featureNorms.empty()) {
      float avgNorm =
          std::accumulate(featureNorms.begin(), featureNorms.end(), 0.0f) /
          featureNorms.size();
      std::cout << "Average feature norm at UV points: " << avgNorm
                << std::endl;
    }

    if (!featureVariances.empty()) {
      float avgVar = std::accumulate(featureVariances.begin(),
                                     featureVariances.end(), 0.0f) /
                     featureVariances.size();
      std::cout << "Average similarity variance at UV points: " << avgVar
                << std::endl;
    }
  }

  // 7. Feature discrimination test
  std::cout << "\n--- Feature Discrimination Test ---" << std::endl;

  // Sample random pairs of pixels and compute their feature similarity
  int numPairs = 1000;
  std::vector<float> intraFrameSimilarities;
  std::vector<float> interFrameSimilarities;

  for (int i = 0; i < numPairs; ++i) {
    int x1 = rand() % refFeatCpu.size(1);
    int y1 = rand() % refFeatCpu.size(0);
    int x2 = rand() % refFeatCpu.size(1);
    int y2 = rand() % refFeatCpu.size(0);

    auto feat1 = refFeatCpu[y1][x1];
    auto feat2 = refFeatCpu[y2][x2];

    // Intra-frame similarity
    float intraSim = torch::cosine_similarity(feat1, feat2, 0).item<float>();
    intraFrameSimilarities.push_back(intraSim);

    // Inter-frame similarity (with first other keyframe)
    if (!keyframes.empty()) {
      auto otherFeat = keyframes[0]->feature_map_.to(torch::kFloat32).cpu();
      if (x1 < otherFeat.size(1) && y1 < otherFeat.size(0)) {
        auto otherFeatVec = otherFeat[y1][x1];
        float interSim =
            torch::cosine_similarity(feat1, otherFeatVec, 0).item<float>();
        interFrameSimilarities.push_back(interSim);
      }
    }
  }

  if (!intraFrameSimilarities.empty()) {
    float avgIntra = std::accumulate(intraFrameSimilarities.begin(),
                                     intraFrameSimilarities.end(), 0.0f) /
                     intraFrameSimilarities.size();
    std::cout << "Average intra-frame similarity: " << avgIntra << std::endl;
  }

  if (!interFrameSimilarities.empty()) {
    float avgInter = std::accumulate(interFrameSimilarities.begin(),
                                     interFrameSimilarities.end(), 0.0f) /
                     interFrameSimilarities.size();
    std::cout << "Average inter-frame similarity: " << avgInter << std::endl;
  }

  // 8. Feature quality warnings
  std::cout << "\n--- Feature Quality Assessment ---" << std::endl;

  if (hasNaN || hasInf) {
    std::cout << "⚠️  CRITICAL: Features contain NaN/Inf values!" << std::endl;
  }

  if (spatialVar < 0.01f) {
    std::cout << "⚠️  Low spatial variance - features may be too uniform"
              << std::endl;
  }

  if (featNorm < 0.1f) {
    std::cout << "⚠️  Low feature magnitudes - possible normalization issue"
              << std::endl;
  }

  if (!intraFrameSimilarities.empty() && !interFrameSimilarities.empty()) {
    float avgIntra = std::accumulate(intraFrameSimilarities.begin(),
                                     intraFrameSimilarities.end(), 0.0f) /
                     intraFrameSimilarities.size();
    float avgInter = std::accumulate(interFrameSimilarities.begin(),
                                     interFrameSimilarities.end(), 0.0f) /
                     interFrameSimilarities.size();

    if (avgIntra > 0.9f) {
      std::cout << "⚠️  High intra-frame similarity - features may lack "
                   "distinctiveness"
                << std::endl;
    }

    if (avgInter > 0.8f) {
      std::cout << "⚠️  High inter-frame similarity - features may not capture "
                   "motion/parallax"
                << std::endl;
    }

    if (avgIntra - avgInter < 0.1f) {
      std::cout
          << "⚠️  Poor discrimination between same/different frame features"
          << std::endl;
    }
  }

  std::cout << "================================================\n"
            << std::endl;
}

void analyze_debug_stats(const std::vector<DebugStats>& stats,
                         float idepth_range) {
  if (stats.empty()) return;

  std::cout << "\n========== MVS Debug Analysis ==========\n";
  std::cout << "Range: " << idepth_range << ", Points: " << stats.size()
            << std::endl;

  // Basic statistics
  int valid_points = 0;
  int no_valid_cams = 0;
  int insufficient_baseline_count = 0;
  int out_of_bounds_count = 0;
  int successful_estimates = 0;
  int quadratic_applied_count = 0;
  int poor_cost_ratio_count = 0;

  float avg_valid_cams = 0;
  float avg_baseline = 0;
  float avg_cost_ratio = 0;
  float avg_depth_change = 0;

  std::vector<float> depth_changes;
  std::vector<float> cost_ratios;

  for (const auto& stat : stats) {
    // Count points with valid cameras
    if (stat.valid_cams_count > 0) {
      valid_points++;
      avg_valid_cams += stat.valid_cams_count;

      // Only sum baseline distances for valid points
      if (stat.max_baseline_dist > 0) {
        avg_baseline += stat.max_baseline_dist;
      }
    } else {
      no_valid_cams++;

      // Only count failure reasons for points with no valid cameras
      if (stat.insufficient_baseline) insufficient_baseline_count++;
      if (stat.out_of_bounds) out_of_bounds_count++;
    }

    // Count successful depth estimates (not -1)
    if (stat.final_depth > 0) {
      successful_estimates++;

      // Only count quadratic refinement for successful estimates
      if (stat.quadratic_applied) quadratic_applied_count++;

      // Calculate depth change for successful estimates
      if (stat.initial_idepth > 0) {
        float initial_depth = 1.0f / stat.initial_idepth;
        float depth_change_ratio =
            std::abs(stat.final_depth - initial_depth) / initial_depth;
        depth_changes.push_back(depth_change_ratio);
        avg_depth_change += depth_change_ratio;
      }
    }

    // Only analyze costs for points that computed them (not -1)
    if (stat.min_cost > 0) {
      cost_ratios.push_back(stat.cost_ratio);
      avg_cost_ratio += stat.cost_ratio;

      if (stat.cost_ratio < 1.2f) poor_cost_ratio_count++;
    }
  }

  // Calculate averages
  if (valid_points > 0) {
    avg_valid_cams /= valid_points;
    avg_baseline /= valid_points;
  }

  if (!cost_ratios.empty()) {
    avg_cost_ratio /= cost_ratios.size();
  }

  if (!depth_changes.empty()) {
    avg_depth_change /= depth_changes.size();
  }

  // Print summary
  std::cout << "\n--- Camera Validation ---" << std::endl;
  std::cout << "Points with valid cameras: " << valid_points << "/"
            << stats.size() << " (" << std::fixed << std::setprecision(1)
            << (100.0f * valid_points / stats.size()) << "%)" << std::endl;
  std::cout << "No valid cameras: " << no_valid_cams << std::endl;
  std::cout << "  └─ Insufficient baseline: " << insufficient_baseline_count
            << std::endl;
  std::cout << "  └─ Out of bounds: " << out_of_bounds_count << std::endl;

  if (valid_points > 0) {
    std::cout << "Avg valid cameras per point: " << std::fixed
              << std::setprecision(2) << avg_valid_cams << std::endl;
    std::cout << "Avg max baseline distance: " << avg_baseline << std::endl;
  }

  std::cout << "\n--- Depth Estimation Results ---" << std::endl;
  std::cout << "Successful depth estimates: " << successful_estimates << "/"
            << stats.size() << " (" << std::fixed << std::setprecision(1)
            << (100.0f * successful_estimates / stats.size()) << "%)"
            << std::endl;
  std::cout << "Failed estimates (kept as -1): "
            << (stats.size() - successful_estimates) << std::endl;

  if (successful_estimates > 0) {
    std::cout << "Quadratic refinement applied: " << quadratic_applied_count
              << "/" << successful_estimates << " (" << std::fixed
              << std::setprecision(1)
              << (100.0f * quadratic_applied_count / successful_estimates)
              << "%)" << std::endl;
  }

  if (!cost_ratios.empty()) {
    std::cout << "Poor cost discrimination (ratio < 1.2): "
              << poor_cost_ratio_count << "/" << cost_ratios.size() << " ("
              << std::fixed << std::setprecision(1)
              << (100.0f * poor_cost_ratio_count / cost_ratios.size()) << "%)"
              << std::endl;
    std::cout << "Avg cost ratio: " << std::fixed << std::setprecision(2)
              << avg_cost_ratio << std::endl;
  }

  if (!depth_changes.empty()) {
    std::cout << "Avg depth change: " << std::fixed << std::setprecision(1)
              << (avg_depth_change * 100) << "%" << std::endl;

    // Percentile analysis for depth changes
    std::sort(depth_changes.begin(), depth_changes.end());
    std::cout << "Depth change percentiles: ";
    std::cout << "50%=" << std::fixed << std::setprecision(1)
              << (depth_changes[depth_changes.size() / 2] * 100) << "% ";
    std::cout << "90%=" << (depth_changes[depth_changes.size() * 9 / 10] * 100)
              << "% ";
    std::cout << "95%="
              << (depth_changes[depth_changes.size() * 95 / 100] * 100) << "%"
              << std::endl;
  }

  // Flag potential issues
  std::cout << "\n--- Potential Issues ---" << std::endl;
  if (no_valid_cams > stats.size() * 0.3f) {
    std::cout << "⚠️  HIGH: >30% points have no valid cameras" << std::endl;
  }
  if (insufficient_baseline_count > stats.size() * 0.5f) {
    std::cout << "⚠️  Camera baseline too small for many points" << std::endl;
  }
  if (!cost_ratios.empty() &&
      poor_cost_ratio_count > cost_ratios.size() * 0.7f) {
    std::cout << "⚠️  Poor feature discrimination (low cost ratios)"
              << std::endl;
  }
  if (avg_depth_change > 0.5f) {
    std::cout
        << "⚠️  Large depth changes suggest range too wide or poor initial depth"
        << std::endl;
  }
  if (valid_points > 0 &&
      quadratic_applied_count < successful_estimates * 0.3f) {
    std::cout
        << "⚠️  Low quadratic refinement rate suggests poor depth candidates"
        << std::endl;
  }

  std::cout << "============================================\n" << std::endl;
}

void saveDepthMapAsPointCloud(const torch::Tensor& depth_map,
                              const torch::Tensor& intrinsics,
                              const torch::Tensor& image,
                              const std::string& filename,
                              float depth_threshold) {
  // Ensure tensors are on CPU
  auto depth_cpu = depth_map.cpu();
  auto intrinsics_cpu = intrinsics.cpu();
  auto image_cpu = image.cpu();

  std::cout << depth_map.sizes() << std::endl;
  std::cout << image.sizes() << std::endl;

  // Get dimensions
  int H = depth_cpu.size(0);
  int W = depth_cpu.size(1);

  // Extract intrinsics (assuming [fx, cx, cy] format)
  float fx = intrinsics_cpu[0].item<float>();
  float cx = intrinsics_cpu[1].item<float>();
  float cy = intrinsics_cpu[2].item<float>();

  std::vector<std::array<float, 6>> points;  // [x, y, z, r, g, b]

  // Convert depth to point cloud
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float depth = depth_cpu[y][x].item<float>();

      // Skip invalid depths
      if (depth <= depth_threshold || depth > 100.0f) {
        continue;
      }

      // Project to 3D
      float world_x = (x - cx) * depth / fx;
      float world_y = (y - cy) * depth / fx;  // Assuming fy = fx
      float world_z = depth;

      // Get color (assuming RGB image)
      float r = 1.0f, g = 1.0f, b = 1.0f;  // Default white
      if (image_cpu.numel() > 0) {
        if (image_cpu.dim() == 3 && image_cpu.size(0) == 3) {
          r = image_cpu[0][y][x].item<float>();
          g = image_cpu[1][y][x].item<float>();
          b = image_cpu[2][y][x].item<float>();
        } else if (image_cpu.dim() == 2) {
          // Grayscale
          r = g = b = image_cpu[y][x].item<float>();
        }
      }

      // Normalize colors to [0, 1] if needed
      if (r > 1.0f) r /= 255.0f;
      if (g > 1.0f) g /= 255.0f;
      if (b > 1.0f) b /= 255.0f;

      points.push_back({world_x, world_y, world_z, r, g, b});
    }
  }

  // Save as PLY file
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return;
  }

  // Write PLY header
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << points.size() << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property float red\n";
  file << "property float green\n";
  file << "property float blue\n";
  file << "end_header\n";

  // Write points
  for (const auto& point : points) {
    file << point[0] << " " << point[1] << " " << point[2] << " " << point[3]
         << " " << point[4] << " " << point[5] << "\n";
  }

  file.close();
  std::cout << "Saved " << points.size() << " points to " << filename
            << std::endl;
}