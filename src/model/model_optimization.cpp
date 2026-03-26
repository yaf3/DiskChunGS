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

void TriangleModel::trainingSetup(
    const TriangleOptimizationParams& training_args) {
  position_lr_init_ = training_args.position_lr_init_ * spatial_lr_scale_;
  position_lr_decay_ = training_args.position_lr_decay_;
  position_lr_min_ = position_lr_init_ * 0.1f;  // position_lr_init_ already includes spatial_lr_scale_

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  optimizer_.reset(
      new SparseTriangleAdam(Tensor_vec_triangles_points_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(0.0f);

  // Per-Triangle position learning rates
  int num_triangles = triangles_points_.size(0);
  position_lrs_ = torch::full(
      {num_triangles}, position_lr_init_,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  // Remaining parameter groups use scalar learning rates
  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_sigma_);
  optimizer_->param_groups()[4].options().set_lr(training_args.sigma_lr_);
}

void TriangleModel::updateLearningRates(const torch::Tensor& visibility) {
  if (visibility.size(0) != position_lrs_.size(0)) {
    throw std::runtime_error(
        "Visibility tensor size doesn't match position_lrs_ size");
  }

  position_lrs_.index_put_(
      {visibility}, position_lrs_.index({visibility}) * position_lr_decay_);
  position_lrs_.clamp_min_(position_lr_min_);
}

void TriangleModel::optimizerStep(torch::Tensor& visibility,
                                   const uint32_t N) {
  torch::NoGradGuard no_grad;

  auto& param_groups = optimizer_->param_groups();

  for (size_t group_idx = 0; group_idx < param_groups.size(); ++group_idx) {
    auto& group = param_groups[group_idx];
    auto& param = group.params()[0];

    if (!param.grad().defined()) continue;

    // Lazily initialize Adam state
    auto& state = optimizer_->state();
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) == state.end()) {
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(0);
      new_state->exp_avg(torch::zeros_like(param));
      new_state->exp_avg_sq(torch::zeros_like(param));
      state[key] = std::move(new_state);
    }

    auto& param_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto options = static_cast<torch::optim::AdamOptions&>(group.options());

    // Group 0 uses per-Triangle position LRs (expanded for [N,3,3]); others scalar
    torch::Tensor lr_tensor;
    if (group_idx == 0) {
      // position_lrs_ is [N]; param is [N,3,3] with M=9 elements per triangle.
      // adamUpdate expects lr_tensor of size N (one LR per triangle row).
      lr_tensor = position_lrs_;
    } else {
      lr_tensor = torch::tensor(
          {static_cast<float>(group.options().get_lr())},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    const uint32_t M = param.numel() / N;
    adamUpdate(param, param.grad(), param_state.exp_avg(),
               param_state.exp_avg_sq(), visibility, lr_tensor,
               std::get<0>(options.betas()), std::get<1>(options.betas()),
               options.eps(), N, M);
  }

  updateLearningRates(visibility);
}
