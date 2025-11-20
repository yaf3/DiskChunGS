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

#include "model/gaussian_model.h"
#include "rendering/gaussian_rasterizer.h"

void GaussianModel::resetOpacity() {
  torch::Tensor opacities_new = general_utils::inverse_sigmoid(
      torch::min(this->getOpacityActivation(),
                 torch::ones_like(this->getOpacityActivation() * 0.01)));
  torch::Tensor optimizable_tensors =
      this->replaceTensorToOptimizer(opacities_new, 3);  // "opacity"
  this->opacity_ = optimizable_tensors;
  this->Tensor_vec_opacity_ = {this->opacity_};
}

void GaussianModel::resetOpacityForMask(const torch::Tensor& gaussian_mask) {
  torch::NoGradGuard no_grad;

  int num_reset = torch::sum(gaussian_mask).item<int>();
  std::cout << "[Opacity Reset] Resetting opacity for " << num_reset
            << " gaussians" << std::endl;

  // Get current opacities in sigmoid space
  torch::Tensor current_opacity_activated = this->getOpacityActivation();

  // Create new opacity: min(current, 0.01) for masked gaussians
  torch::Tensor target_opacity =
      torch::min(current_opacity_activated,
                 torch::ones_like(current_opacity_activated) * 0.05f);

  // Apply inverse sigmoid to convert back to optimization space
  torch::Tensor new_opacity_values =
      general_utils::inverse_sigmoid(target_opacity);

  // Selectively update only masked gaussians
  this->opacity_.index_put_({gaussian_mask},
                            new_opacity_values.index({gaussian_mask}));

  std::cout << "[Opacity Reset] Opacity reset complete - max="
            << torch::sigmoid(this->opacity_).max().item<float>()
            << ", min=" << torch::sigmoid(this->opacity_).min().item<float>()
            << std::endl;
}

void GaussianModel::resetPositionLRAndOptimizerState(
    const torch::Tensor& gaussian_mask) {
  torch::NoGradGuard no_grad;

  if (!this->optimizer_) {
    std::cerr << "ERROR: Optimizer is null in "
                 "resetPositionLRAndOptimizerState!"
              << std::endl;
    return;
  }

  // Count how many gaussians we're resetting
  int num_reset = torch::sum(gaussian_mask).item<int>();
  std::cout << "[Optimizer Reset] Resetting position LR and Adam states for "
            << num_reset << " gaussians" << std::endl;

  // 1. Reset position learning rates back to initial value
  position_lrs_.index_put_({gaussian_mask}, position_lr_init_);

  // 2. Reset Adam optimizer states for positions (group 0)
  auto& param_group = this->optimizer_->param_groups()[0];
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

  // Get current exp_avg and exp_avg_sq tensors
  torch::Tensor exp_avg = param_state.exp_avg();
  torch::Tensor exp_avg_sq = param_state.exp_avg_sq();

  // Expand mask to match xyz dimensions [N, 3]
  torch::Tensor xyz_mask = gaussian_mask.unsqueeze(1).expand({-1, 3});

  // Zero out momentum and variance for masked gaussians
  exp_avg.index_put_({xyz_mask}, 0.0f);
  exp_avg_sq.index_put_({xyz_mask}, 0.0f);

  std::cout << "[Optimizer Reset] Position LRs reset - max="
            << position_lrs_.max().item<float>()
            << ", min=" << position_lrs_.min().item<float>()
            << ", mean=" << position_lrs_.mean().item<float>() << std::endl;
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor,
                                                      int tensor_idx) {
  if (!this->optimizer_) {
    std::cerr << "ERROR: Optimizer is null!" << std::endl;
    throw std::runtime_error("Null optimizer in replaceTensorToOptimizer");
  }

  if (tensor_idx >= this->optimizer_->param_groups().size()) {
    std::cerr << "ERROR: tensor_idx " << tensor_idx << " out of bounds!"
              << std::endl;
    throw std::runtime_error("Index out of bounds in replaceTensorToOptimizer");
  }

  auto& param_group = this->optimizer_->param_groups()[tensor_idx];
  if (param_group.params().empty()) {
    std::cerr << "ERROR: No parameters in group " << tensor_idx << std::endl;
    throw std::runtime_error("Empty param group in replaceTensorToOptimizer");
  }

  auto& param = param_group.params()[0];
  auto& state = optimizer_->state();
  auto key = param.unsafeGetTensorImpl();

  if (state.find(key) == state.end()) {
    std::cerr << "WARNING: No optimizer state found for tensor_idx "
              << tensor_idx << std::endl;
    // Create a new state instead of crashing
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(0);
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    state[key] = std::move(new_state);
  }

  try {
    auto& stored_state =
        static_cast<torch::optim::AdamParamState&>(*state[key]);

    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());

    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));

    state.erase(key);

    param = tensor.requires_grad_();
    key = param.unsafeGetTensorImpl();

    state[key] = std::move(new_state);

    return param;
  } catch (const std::exception& e) {
    std::cerr << "ERROR in replaceTensorToOptimizer: " << e.what() << std::endl;
    throw;
  }
}

void GaussianModel::prunePoints(torch::Tensor& mask) {
  torch::NoGradGuard no_grad;
  auto valid_points_mask = ~mask;

  // _prune_optimizer
  std::vector<torch::Tensor> optimizable_tensors(6);
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& param = param_groups[group_idx].params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      auto valid_indices = torch::nonzero(valid_points_mask).squeeze(1);
      new_state->exp_avg(stored_state.exp_avg().index_select(0, valid_indices));
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index_select(0, valid_indices));
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = param.index({valid_points_mask}).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
      optimizable_tensors[group_idx] = param;
    } else {
      param = param.index({valid_points_mask}).requires_grad_();
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ = this->exist_since_iter_.index({valid_points_mask});
  this->position_lrs_ = this->position_lrs_.index({valid_points_mask});
  this->gaussian_chunk_ids_ =
      this->gaussian_chunk_ids_.index({valid_points_mask});
  this->gaussian_ids_ = this->gaussian_ids_.index({valid_points_mask});

  // c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_exist_since_iter,
    torch::Tensor& new_chunk_ids,
    torch::Tensor& new_position_lrs,
    torch::Tensor& new_gaussian_ids,
    const std::vector<torch::Tensor>& loaded_exp_avg,
    const std::vector<torch::Tensor>& loaded_exp_avg_sq,
    const std::vector<int64_t>& loaded_step_counts) {
  torch::NoGradGuard no_grad;
  // cat_tensors_to_optimizer
  std::vector<torch::Tensor> optimizable_tensors(6);
  std::vector<torch::Tensor> tensors_dict = {new_xyz,           new_features_dc,
                                             new_features_rest, new_opacities,
                                             new_scaling,       new_rotation};
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& group = param_groups[group_idx];
    assert(group.params().size() == 1);
    auto& extension_tensor = tensors_dict[group_idx];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();

      // Use loaded optimizer states if provided, otherwise use zeros
      torch::Tensor extension_exp_avg, extension_exp_avg_sq;
      int64_t new_step_count;

      if (group_idx < loaded_exp_avg.size() &&
          loaded_exp_avg[group_idx].defined()) {
        extension_exp_avg = loaded_exp_avg[group_idx];
        extension_exp_avg_sq = loaded_exp_avg_sq[group_idx];
        new_step_count =
            std::max(stored_state.step(), loaded_step_counts[group_idx]);
      } else {
        extension_exp_avg = torch::zeros_like(extension_tensor);
        extension_exp_avg_sq = torch::zeros_like(extension_tensor);
        new_step_count = stored_state.step();
      }

      new_state->step(new_step_count);
      new_state->exp_avg(
          torch::cat({stored_state.exp_avg().clone(), extension_exp_avg},
                     /*dim=*/0));
      new_state->exp_avg_sq(
          torch::cat({stored_state.exp_avg_sq().clone(), extension_exp_avg_sq},
                     /*dim=*/0));
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);

      optimizable_tensors[group_idx] = param;
    } else {
      // No existing state - create initial state for the concatenated param
      int64_t old_size = param.size(0);
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      key = param.unsafeGetTensorImpl();

      // Create optimizer state
      auto new_state = std::make_unique<torch::optim::AdamParamState>();

      if (group_idx < loaded_exp_avg.size() &&
          loaded_exp_avg[group_idx].defined()) {
        // Use loaded states for the extension, zeros for existing
        new_state->step(loaded_step_counts[group_idx]);
        new_state->exp_avg(
            torch::cat({torch::zeros({old_size}, extension_tensor.options()),
                        loaded_exp_avg[group_idx]},
                       0));
        new_state->exp_avg_sq(
            torch::cat({torch::zeros({old_size}, extension_tensor.options()),
                        loaded_exp_avg_sq[group_idx]},
                       0));
      } else {
        // All zeros
        new_state->step(0);
        new_state->exp_avg(torch::zeros_like(param));
        new_state->exp_avg_sq(torch::zeros_like(param));
      }

      state[key] = std::move(new_state);
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ =
      torch::cat({this->exist_since_iter_, new_exist_since_iter}, /*dim=*/0);

  int num_new_primitives = new_xyz.size(0);
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  gaussian_chunk_ids_ =
      torch::cat({gaussian_chunk_ids_, new_chunk_ids}, /*dim=*/0);
  this->gaussian_ids_ =
      torch::cat({this->gaussian_ids_, new_gaussian_ids}, /*dim=*/0);
}

void GaussianModel::pruneLowOpacityGaussians(
    std::shared_ptr<GaussianKeyframe> pkf,
    const torch::Tensor& visible_gaussian_mask) {
  torch::NoGradGuard no_grad;

  torch::Tensor visible_indices = torch::where(visible_gaussian_mask)[0];

  // Get keyframe parameters
  torch::Tensor keyframe_center =
      pkf->getCenter();                // Camera center in world coordinates
  float focal_length = pkf->intr_[0];  // fx
  int image_width = pkf->image_width_;

  // Get Gaussian parameters
  torch::Tensor positions = getXYZ().index({visible_indices});  // [N, 3]
  torch::Tensor opacities = getOpacityActivation().index(
      {visible_indices});  // [N, 1] - already activated
                           // (sigmoid applied)
  torch::Tensor scalings = getScalingActivation().index(
      {visible_indices});  // [N, 3] - already
                           // activated (exp applied)

  int n_gaussians = positions.size(0);

  // Create validity mask (start with all valid)
  torch::Tensor valid_mask = torch::ones(
      n_gaussians,
      torch::TensorOptions().dtype(torch::kBool).device(positions.device()));

  // 1. Remove Gaussians with low opacity (< 0.05)
  torch::Tensor opacity_mask = opacities.squeeze(1) > 0.05f;  // [N]
  valid_mask = valid_mask & opacity_mask;

  // 2. Remove Gaussians that appear too large on screen
  // Compute distance from camera to each Gaussian
  torch::Tensor diff = positions - keyframe_center.unsqueeze(0);  // [N, 3]
  torch::Tensor distances =
      torch::norm(diff, 2, 1);  // [N] - L2 norm along dim 1

  // Compute maximum scaling for each Gaussian
  torch::Tensor max_scaling =
      std::get<0>(torch::max(scalings, /*dim=*/1));  // [N]

  // Compute screen size: focal_length * max_scaling / distance
  torch::Tensor screen_size = focal_length * max_scaling / distances;  // [N]

  // Remove Gaussians that are too large (screen_size >= 0.5 * image_width)
  float max_screen_size = 0.5f * static_cast<float>(image_width);
  torch::Tensor size_mask = screen_size < max_screen_size;  // [N]
  valid_mask = valid_mask & size_mask;

  // 3. Create pruning mask (invert valid_mask since prunePoints expects
  // "points to remove")
  torch::Tensor prune_mask = ~valid_mask;  // [N] - true for points to remove

  // std::cout << "Pruning: " << prune_mask.sum().item<int>() << " points"
  //           << std::endl;

  torch::Tensor full_model_prune_mask = torch::zeros(
      {getXYZ().size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

  full_model_prune_mask.index_put_({visible_indices}, prune_mask);

  prunePoints(full_model_prune_mask);
}

void GaussianModel::deleteSparseChunks(int min_gaussians_per_chunk) {
  torch::NoGradGuard no_grad;

  if (!is_initialized_ || xyz_.size(0) == 0) {
    return;
  }

  // Count gaussians per spatial chunk (regardless of load state)
  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(gaussian_chunk_ids_, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  // Find sparse chunks
  torch::Tensor sparse_mask = counts < min_gaussians_per_chunk;
  torch::Tensor sparse_chunk_ids = unique_chunks.index({sparse_mask});
  torch::Tensor sparse_counts = counts.index({sparse_mask});

  if (sparse_chunk_ids.size(0) == 0) {
    return;
  }

  int total_gaussians_to_delete = sparse_counts.sum().item<int>();
  int num_sparse_chunks = sparse_chunk_ids.size(0);

  // std::cout << "[Sparse Deletion] Deleting " << num_sparse_chunks
  //           << " sparse chunks with " << total_gaussians_to_delete
  //           << " total gaussians (threshold: " << min_gaussians_per_chunk
  //           <<
  //           ")"
  //           << std::endl;

  // Create removal mask for gaussians
  torch::Tensor remove_mask =
      torch::isin(gaussian_chunk_ids_, sparse_chunk_ids);

  // Update all tracking tensors (vectorized)
  // Remove from loaded chunks (if any were loaded)
  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, sparse_chunk_ids);
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});

  // Remove from disk chunks (delete them entirely)
  torch::Tensor keep_disk_mask =
      ~torch::isin(chunks_on_disk_, sparse_chunk_ids);
  chunks_on_disk_ = chunks_on_disk_.index({keep_disk_mask});
  chunk_gaussian_counts_ = chunk_gaussian_counts_.index({keep_disk_mask});

  // Clear access times for deleted chunks
  auto sparse_ids_cpu = sparse_chunk_ids.cpu();
  auto sparse_ids_accessor = sparse_ids_cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < sparse_ids_cpu.size(0); i++) {
    chunk_access_times_.erase(sparse_ids_accessor[i]);
  }

  // Actually delete the disk files (if they exist)
  deleteSparseChunkFiles(sparse_chunk_ids);

  // Remove gaussians from memory
  prunePoints(remove_mask);

  // std::cout << "[Sparse Deletion] Deleted " << total_gaussians_to_delete
  //           << " gaussians from " << num_sparse_chunks
  //           << " sparse chunks. Model now has " << xyz_.size(0) << "
  //           gaussians"
  //           << std::endl;
}

void GaussianModel::deleteSparseChunkFiles(const torch::Tensor& chunk_ids) {
  auto chunks_cpu = chunk_ids.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();

  int files_deleted = 0;

  for (int i = 0; i < chunks_cpu.size(0); ++i) {
    int64_t chunk_id = accessor[i];
    std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

    if (std::filesystem::exists(chunk_filename)) {
      try {
        std::filesystem::remove(chunk_filename);
        files_deleted++;
        // std::cout << "[File Deletion] Deleted " << chunk_filename <<
        // std::endl;
      } catch (const std::exception& e) {
        std::cerr << "[File Deletion] Failed to delete " << chunk_filename
                  << ": " << e.what() << std::endl;
      }
    }
  }

  if (files_deleted > 0) {
    // std::cout << "[File Deletion] Deleted " << files_deleted
    //           << " chunk files from disk" << std::endl;
  }
}

void GaussianModel::prune(float min_opacity,
                          float extent,
                          int max_screen_size) {
  auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
  if (max_screen_size) {
    auto big_points_ws =
        std::get<0>(this->getScalingActivation().max(/*dim=*/1)) >
        0.1f * extent;
    prune_mask = torch::logical_or(prune_mask, big_points_ws);
  }
  this->prunePoints(prune_mask);

  // c10::cuda::CUDACachingAllocator::emptyCache();  // torch.cuda.empty_cache()
}