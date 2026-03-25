/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023
 * as part of Photo-SLAM, modified by Dapeng Feng in 2024 as part of CaRtGS,
 * and further modified by Casimir Feldmann in 2025 as part of DiskChunGS.
 */

#include "model/triangle_model.h"
#include "rendering/triangle_rasterizer.h"

void TriangleModel::assignOptimizedTensors(
    const std::vector<torch::Tensor>& tensors) {
  triangles_points_ = tensors[0];
  features_dc_ = tensors[1];
  features_rest_ = tensors[2];
  opacity_ = tensors[3];
  sigma_ = tensors[4];
  TRIANGLE_MODEL_TENSORS_TO_VEC
}

void TriangleModel::resetOpacity() {
  torch::Tensor opacities_new = general_utils::inverse_sigmoid(torch::min(
      getOpacityActivation(), torch::ones_like(getOpacityActivation() * 0.01)));
  torch::Tensor optimizable_tensors =
      replaceTensorToOptimizer(opacities_new, 3);  // opacity
  opacity_ = optimizable_tensors;
  Tensor_vec_opacity_ = {opacity_};
}

void TriangleModel::resetOpacityForMask(const torch::Tensor& triangle_mask) {
  torch::NoGradGuard no_grad;

  int num_reset = torch::sum(triangle_mask).item<int>();
  std::cout << "[Opacity Reset] Resetting opacity for " << num_reset
            << " triangles" << std::endl;

  torch::Tensor current_opacity_activated = getOpacityActivation();

  // min(current, 0.05) for masked triangles, then convert back to logit space
  torch::Tensor target_opacity =
      torch::min(current_opacity_activated,
                 torch::ones_like(current_opacity_activated) * 0.05f);
  torch::Tensor new_opacity_values =
      general_utils::inverse_sigmoid(target_opacity);

  opacity_.index_put_({triangle_mask},
                      new_opacity_values.index({triangle_mask}));

  std::cout << "[Opacity Reset] Opacity reset complete - max="
            << torch::sigmoid(opacity_).max().item<float>()
            << ", min=" << torch::sigmoid(opacity_).min().item<float>()
            << std::endl;
}

void TriangleModel::resetPositionLRAndOptimizerState(
    const torch::Tensor& triangle_mask) {
  torch::NoGradGuard no_grad;

  if (!optimizer_) {
    std::cerr << "ERROR: Optimizer is null in "
                 "resetPositionLRAndOptimizerState!"
              << std::endl;
    return;
  }

  int num_reset = torch::sum(triangle_mask).item<int>();
  std::cout << "[Optimizer Reset] Resetting position LR and Adam states for "
            << num_reset << " triangles" << std::endl;

  // Reset position learning rates back to initial value
  position_lrs_.index_put_({triangle_mask}, position_lr_init_);

  // Reset Adam optimizer states for positions (group 0 = xyz)
  auto& param_group = optimizer_->param_groups()[0];
  if (param_group.params().empty()) {
    std::cerr << "ERROR: No parameters in position group!" << std::endl;
    return;
  }

  auto& xyz_param = param_group.params()[0];
  auto& state = optimizer_->state();
  auto key = xyz_param.unsafeGetTensorImpl();

  if (state.find(key) == state.end()) {
    std::cerr << "WARNING: No optimizer state found for positions" << std::endl;
    return;
  }

  auto& param_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
  torch::Tensor exp_avg = param_state.exp_avg();
  torch::Tensor exp_avg_sq = param_state.exp_avg_sq();

  // Expand mask to match triangles_points_ dimensions [N, 3, 3]
  torch::Tensor tri_mask =
      triangle_mask.unsqueeze(1).unsqueeze(2).expand({-1, 3, 3});
  exp_avg.index_put_({tri_mask}, 0.0f);
  exp_avg_sq.index_put_({tri_mask}, 0.0f);

  std::cout << "[Optimizer Reset] Position LRs reset - max="
            << position_lrs_.max().item<float>()
            << ", min=" << position_lrs_.min().item<float>()
            << ", mean=" << position_lrs_.mean().item<float>() << std::endl;
}

torch::Tensor TriangleModel::replaceTensorToOptimizer(torch::Tensor& tensor,
                                                      int tensor_idx) {
  if (!optimizer_) {
    throw std::runtime_error("Null optimizer in replaceTensorToOptimizer");
  }
  if (tensor_idx >= static_cast<int>(optimizer_->param_groups().size())) {
    throw std::runtime_error("Index out of bounds in replaceTensorToOptimizer");
  }

  auto& param_group = optimizer_->param_groups()[tensor_idx];
  if (param_group.params().empty()) {
    throw std::runtime_error("Empty param group in replaceTensorToOptimizer");
  }

  auto& param = param_group.params()[0];
  auto& state = optimizer_->state();
  auto key = param.unsafeGetTensorImpl();

  // Create state if it doesn't exist yet
  if (state.find(key) == state.end()) {
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(0);
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    state[key] = std::move(new_state);
  }

  auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);

  auto new_state = std::make_unique<torch::optim::AdamParamState>();
  new_state->step(stored_state.step());
  new_state->exp_avg(torch::zeros_like(tensor));
  new_state->exp_avg_sq(torch::zeros_like(tensor));

  state.erase(key);
  param = tensor.requires_grad_();
  key = param.unsafeGetTensorImpl();
  state[key] = std::move(new_state);

  return param;
}

void TriangleModel::prunePoints(torch::Tensor& mask) {
  torch::NoGradGuard no_grad;
  auto valid_points_mask = ~mask;
  auto valid_indices = torch::nonzero(valid_points_mask).squeeze(1);

  // Prune optimizer: filter each parameter group to keep only valid points
  std::vector<torch::Tensor> optimizable_tensors(kNumParamGroups);
  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
    auto& param = param_groups[group_idx].params()[0];
    auto key = param.unsafeGetTensorImpl();

    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(stored_state.exp_avg().index_select(0, valid_indices));
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index_select(0, valid_indices));

      state.erase(key);
      param = param.index({valid_points_mask}).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
    } else {
      param = param.index({valid_points_mask}).requires_grad_();
    }
    optimizable_tensors[group_idx] = param;
  }

  assignOptimizedTensors(optimizable_tensors);

  exist_since_iter_ = exist_since_iter_.index({valid_points_mask});
  position_lrs_ = position_lrs_.index({valid_points_mask});
  triangle_chunk_ids_ = triangle_chunk_ids_.index({valid_points_mask});
  triangle_ids_ = triangle_ids_.index({valid_points_mask});
}

void TriangleModel::densificationPostfix(
    torch::Tensor& new_triangles_points,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_sigma,
    torch::Tensor& new_exist_since_iter,
    torch::Tensor& new_chunk_ids,
    torch::Tensor& new_position_lrs,
    torch::Tensor& new_triangle_ids,
    const std::vector<torch::Tensor>& loaded_exp_avg,
    const std::vector<torch::Tensor>& loaded_exp_avg_sq,
    const std::vector<int64_t>& loaded_step_counts) {
  torch::NoGradGuard no_grad;

  std::vector<torch::Tensor> optimizable_tensors(kNumParamGroups);
  std::vector<torch::Tensor> extension_tensors = {
      new_triangles_points, new_features_dc, new_features_rest,
      new_opacities,        new_sigma};

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
    auto& group = param_groups[group_idx];
    assert(group.params().size() == 1);
    auto& extension_tensor = extension_tensors[group_idx];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();

    // Determine extension optimizer state (loaded from disk or zeros)
    bool has_loaded_state =
        group_idx < static_cast<int>(loaded_exp_avg.size()) &&
        loaded_exp_avg[group_idx].defined();
    torch::Tensor ext_exp_avg = has_loaded_state
                                    ? loaded_exp_avg[group_idx]
                                    : torch::zeros_like(extension_tensor);
    torch::Tensor ext_exp_avg_sq = has_loaded_state
                                       ? loaded_exp_avg_sq[group_idx]
                                       : torch::zeros_like(extension_tensor);

    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);

      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(has_loaded_state ? std::max(stored_state.step(),
                                                  loaded_step_counts[group_idx])
                                       : stored_state.step());
      new_state->exp_avg(
          torch::cat({stored_state.exp_avg().clone(), ext_exp_avg}, 0));
      new_state->exp_avg_sq(
          torch::cat({stored_state.exp_avg_sq().clone(), ext_exp_avg_sq}, 0));

      state.erase(key);
      param = torch::cat({param, extension_tensor}, 0).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
    } else {
      int64_t old_size = param.size(0);
      param = torch::cat({param, extension_tensor}, 0).requires_grad_();
      key = param.unsafeGetTensorImpl();

      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      if (has_loaded_state) {
        new_state->step(loaded_step_counts[group_idx]);
        new_state->exp_avg(torch::cat(
            {torch::zeros({old_size}, extension_tensor.options()), ext_exp_avg},
            0));
        new_state->exp_avg_sq(
            torch::cat({torch::zeros({old_size}, extension_tensor.options()),
                        ext_exp_avg_sq},
                       0));
      } else {
        new_state->step(0);
        new_state->exp_avg(torch::zeros_like(param));
        new_state->exp_avg_sq(torch::zeros_like(param));
      }

      state[key] = std::move(new_state);
    }
    optimizable_tensors[group_idx] = param;
  }

  assignOptimizedTensors(optimizable_tensors);

  exist_since_iter_ = torch::cat({exist_since_iter_, new_exist_since_iter}, 0);
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  triangle_chunk_ids_ = torch::cat({triangle_chunk_ids_, new_chunk_ids}, 0);
  triangle_ids_ = torch::cat({triangle_ids_, new_triangle_ids}, 0);
}

void TriangleModel::pruneLowOpacityTriangles(
    std::shared_ptr<TriangleKeyframe> pkf,
    const torch::Tensor& visible_triangle_mask) {
  torch::NoGradGuard no_grad;

  torch::Tensor visible_indices = torch::where(visible_triangle_mask)[0];

  // Keyframe camera parameters
  torch::Tensor keyframe_center = pkf->getCenter();
  float focal_length = pkf->intr_[0];  // fx
  int image_width = pkf->image_width_;

  // Triangle parameters for visible subset
  torch::Tensor positions = getXYZ().index({visible_indices});
  torch::Tensor opacities = getOpacityActivation().index({visible_indices});
  torch::Tensor sigmas = getSigmaActivation().index({visible_indices});  // [V,1]

  int n_triangles = positions.size(0);
  torch::Tensor valid_mask = torch::ones(
      n_triangles,
      torch::TensorOptions().dtype(torch::kBool).device(positions.device()));

  // Remove Triangles with low opacity
  valid_mask &= opacities.squeeze(1) > 0.05f;

  // Remove Triangles that appear too large on screen (use sigma as scale proxy)
  torch::Tensor distances =
      torch::norm(positions - keyframe_center.unsqueeze(0), 2, /*dim=*/1);
  torch::Tensor screen_size =
      focal_length * sigmas.squeeze(1) / distances.clamp_min(1e-6f);
  float max_screen_size = 0.5f * static_cast<float>(image_width);
  valid_mask &= screen_size < max_screen_size;

  // Build full-model prune mask from the visible subset
  torch::Tensor full_model_prune_mask = torch::zeros(
      {triangles_points_.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  full_model_prune_mask.index_put_({visible_indices}, ~valid_mask);

  prunePoints(full_model_prune_mask);
}

void TriangleModel::deleteSparseChunks(int min_triangles_per_chunk) {
  torch::NoGradGuard no_grad;

  if (!is_initialized_ || triangles_points_.size(0) == 0) {
    return;
  }

  // Count triangles per spatial chunk
  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(triangle_chunk_ids_, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  torch::Tensor sparse_mask = counts < min_triangles_per_chunk;
  torch::Tensor sparse_chunk_ids = unique_chunks.index({sparse_mask});

  if (sparse_chunk_ids.size(0) == 0) {
    return;
  }

  // Create removal mask and update tracking state
  torch::Tensor remove_mask =
      torch::isin(triangle_chunk_ids_, sparse_chunk_ids);

  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, sparse_chunk_ids);
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});

  torch::Tensor keep_disk_mask =
      ~torch::isin(chunks_on_disk_, sparse_chunk_ids);
  chunks_on_disk_ = chunks_on_disk_.index({keep_disk_mask});
  chunk_triangle_counts_ = chunk_triangle_counts_.index({keep_disk_mask});

  // Clear access times for deleted chunks
  auto sparse_ids_cpu = sparse_chunk_ids.cpu();
  auto sparse_ids_accessor = sparse_ids_cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < sparse_ids_cpu.size(0); i++) {
    chunk_access_times_.erase(sparse_ids_accessor[i]);
  }

  deleteSparseChunkFiles(sparse_chunk_ids);
  prunePoints(remove_mask);
}

void TriangleModel::deleteSparseChunkFiles(const torch::Tensor& chunk_ids) {
  auto chunks_cpu = chunk_ids.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();

  for (int i = 0; i < chunks_cpu.size(0); ++i) {
    int64_t chunk_id = accessor[i];
    std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

    if (std::filesystem::exists(chunk_filename)) {
      try {
        std::filesystem::remove(chunk_filename);
      } catch (const std::exception& e) {
        std::cerr << "[File Deletion] Failed to delete " << chunk_filename
                  << ": " << e.what() << std::endl;
      }
    }
  }
}