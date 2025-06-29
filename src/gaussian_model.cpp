/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM, and modified by Dapeng Feng in 2024, as part of
 * CaRtGS.
 */

#include "include/gaussian_model.h"

#include "include/gaussian_rasterizer.h"

GaussianModel::GaussianModel(const int sh_degree)
    : sh_degree_(0),
      position_lr_init_(0.00005),
      position_lr_decay_(0.99998),
      local_iteration_(0) {
  this->sh_degree_ = sh_degree;

  // Device
  if (torch::cuda::is_available())
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

GaussianModel::GaussianModel(const GaussianModelParams& model_params)
    : sh_degree_(0),
      position_lr_init_(0.00005),
      position_lr_decay_(0.99998),
      local_iteration_(0) {
  this->sh_degree_ = model_params.sh_degree_;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

torch::Tensor GaussianModel::getScalingActivation() {
  return torch::exp(this->scaling_);
}

torch::Tensor GaussianModel::getRotationActivation() {
  return torch::nn::functional::normalize(this->rotation_);
}

torch::Tensor GaussianModel::getXYZ() { return this->xyz_; }

torch::Tensor GaussianModel::getFeatures() {
  return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()},
                    /*dim=*/1);
}

torch::Tensor GaussianModel::getOpacityActivation() {
  return torch::sigmoid(this->opacity_);
}

torch::Tensor GaussianModel::getCovarianceActivation(int scaling_modifier) {
  // build_rotation
  auto r = this->rotation_;
  auto R = general_utils::build_rotation(r);

  // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
  auto s = scaling_modifier * this->getScalingActivation();
  auto L = torch::zeros(
      {s.size(0), 3, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
  L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
  L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
  L = R.matmul(L);  // L = R @ L

  // build_covariance_from_scaling_rotation
  auto actual_covariance = L.matmul(L.transpose(1, 2));
  return actual_covariance;
}

void GaussianModel::createFromPcd(const torch::Tensor& fused_point_cloud,
                                  const torch::Tensor& color,
                                  const torch::Tensor& new_scales,
                                  const torch::Tensor& new_opacities) {
  int num_points = static_cast<int>(fused_point_cloud.sizes()[0]);

  torch::Tensor fused_color = sh_utils::RGB2SH(color);
  auto temp = this->sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {fused_color.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      fused_color;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points at initialization : " <<
  // fused_point_cloud.size(0) << std::endl;

  torch::Tensor point_cloud_copy = fused_point_cloud.clone();

  torch::Tensor scales;
  if (new_scales.defined() && new_scales.size(0) > 0) {
    // Use provided scales, convert to log space and repeat for 3 dimensions
    scales = new_scales;
  } else {
    torch::Tensor dist2 =
        torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    scales = torch::log(torch::sqrt(dist2) * 0.1);
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  }
  torch::Tensor rots =
      torch::zeros({fused_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor opacities = new_opacities;

  this->exist_since_iter_ = torch::zeros(
      {fused_point_cloud.size(0)},
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  this->xyz_ = fused_point_cloud.requires_grad_();
  this->features_dc_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->features_rest_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->scaling_ = scales.requires_grad_();
  this->rotation_ = rots.requires_grad_();
  this->opacity_ = opacities.requires_grad_();

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::increasePcd(const torch::Tensor& new_point_cloud,
                                const torch::Tensor& new_colors,
                                const torch::Tensor& new_scales,
                                const torch::Tensor& new_opacities,
                                const int iteration) {
  // auto time1 = std::chrono::steady_clock::now();
  auto num_new_points = new_point_cloud.size(0);
  if (num_new_points == 0) return;

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points increase : "
  //           << num_new_points << std::endl;

  torch::Tensor scales;
  if (new_scales.defined() && new_scales.size(0) > 0) {
    scales = new_scales;
  } else {
    torch::Tensor dist2 =
        torch::clamp_min(distCUDA2(new_point_cloud.clone()), 0.0000001);
    scales = torch::log(torch::sqrt(dist2) * 0.1);
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  }
  torch::Tensor rots =
      torch::zeros({new_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor new_exist_since_iter = torch::full(
      {new_point_cloud.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_point_cloud;
  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities_tensor = new_opacities;
  auto new_scaling = scales;
  auto new_rotation = rots;

  // auto time2 = std::chrono::steady_clock::now();
  // auto time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
  // std::cout << "increasePcd(tensor) preparation time: " << time << " ms"
  // <<std::endl;

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities_tensor, new_scaling, new_rotation,
                       new_exist_since_iter);

  c10::cuda::CUDACachingAllocator::emptyCache();

  // auto time3 = std::chrono::steady_clock::now();
  // time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
  // std::cout << "increasePcd(tensor) postfix time: " << time << " ms"
  // <<std::endl;
}

void GaussianModel::applyScaledTransformation(const float s,
                                              const Sophus::SE3f T) {
  torch::NoGradGuard no_grad;
  // pt <- (s * Ryw * pt + tyw)
  this->xyz_ *= s;
  torch::Tensor T_tensor =
      tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(this->xyz_, T_tensor);

  // torch::Tensor scales;
  // torch::Tensor point_cloud_copy = this->xyz_.clone();
  // torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy),
  // 0.0000001); scales = torch::log(torch::sqrt(dist2)); auto scales_ndimension
  // = scales.ndimension(); scales =
  // scales.unsqueeze(scales_ndimension).repeat({1, 3});
  this->scaling_ *= s;
  scaledTransformationPostfix(this->xyz_, this->scaling_);
}

void GaussianModel::scaledTransformationPostfix(torch::Tensor& new_xyz,
                                                torch::Tensor& new_scaling) {
  // param_groups[0] = xyz_
  torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(new_xyz, 0);
  // param_groups[4] = scaling_
  torch::Tensor optimizable_scaling =
      this->replaceTensorToOptimizer(new_scaling, 4);

  this->xyz_ = optimizable_xyz;
  this->scaling_ = optimizable_scaling;

  this->Tensor_vec_xyz_ = {this->xyz_};
  this->Tensor_vec_scaling_ = {this->scaling_};
}

void GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale) {
  torch::NoGradGuard no_grad;

  // std::cout << "[DEBUG-STPV] Starting with flag tensor size: "
  //           << point_not_transformed_flags.size(0)
  //           << ", xyz size: " << this->xyz_.size(0) << std::endl;

  torch::Tensor points = this->getXYZ();
  torch::Tensor rots = this->getRotationActivation();

  // std::cout << "[DEBUG-STPV] Got points and rotations" << std::endl;
  // torch::Tensor scales = this->scaling_;// * scale;

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  std::cout << "[DEBUG-STPV] Created unstable flags" << std::endl;

  std::cout << "[DEBUG-STPV] Calling transform function" << std::endl;

  scaleAndTransformThenMarkVisiblePoints(
      points, rots, point_not_transformed_flags, point_unstable_flags,
      diff_pose, kf_world_view_transform, kf_full_proj_transform,
      num_transformed, scale);

  // std::cout << "[DEBUG-STPV] Transform complete, transformed "
  //           << num_transformed << " points" << std::endl;

  // torch::Tensor point_cloud_copy = points.clone();
  // torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy),
  // 0.0000001); torch::Tensor scales = torch::log(torch::sqrt(dist2)); auto
  // scales_ndimension = scales.ndimension(); scales =
  // scales.unsqueeze(scales_ndimension).repeat({1, 3});

  // Postfix
  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================

  if (num_transformed > 0) {
    try {
      // std::cout << "[DEBUG-STPV] About to replace xyz tensor" << std::endl;
      torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
      // std::cout << "[DEBUG-STPV] Successfully replaced xyz tensor" <<
      // std::endl;

      // std::cout << "[DEBUG-STPV] About to replace rotation tensor" <<
      // std::endl;
      torch::Tensor optimizable_rots = this->replaceTensorToOptimizer(rots, 5);
      // std::cout << "[DEBUG-STPV] Successfully replaced rotation tensor"
      //           << std::endl;

      this->xyz_ = optimizable_xyz;
      this->rotation_ = optimizable_rots;

      this->Tensor_vec_xyz_ = {this->xyz_};
      this->Tensor_vec_rotation_ = {this->rotation_};

      // std::cout << "[DEBUG-STPV] Updated tensors in-place" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "ERROR during optimizer tensor replacement: " << e.what()
                << std::endl;
      // Recover gracefully instead of crashing
      std::cerr << "Skipping optimizer update for this transformation"
                << std::endl;
    }
  }
}

void GaussianModel::trainingSetup(
    const GaussianOptimizationParams& training_args) {
  setPercentDense(training_args.percent_dense_);
  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));

  position_lr_init_ = training_args.position_lr_init_;
  position_lr_decay_ = training_args.position_lr_decay_;
  position_lr_min_ = position_lr_init_ * 0.1f;

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);  // We'll set individual LRs below
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(
      training_args.position_lr_init_);

  // For per-primitive learning rates, create tensor-based LRs
  int num_gaussians = this->getXYZ().size(0);

  // Position learning rates (per-primitive for positions)
  torch::Tensor position_lrs = torch::full(
      {num_gaussians}, position_lr_init_,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
  this->position_lrs_ = position_lrs;

  // For other parameters, we can still use scalar learning rates
  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);
}

void GaussianModel::updateLearningRates(const torch::Tensor& visibility) {
  // Check if visibility tensor size matches position_lrs_ size
  // This can happen when pruning occurs between radii computation and optimizer
  // step
  if (visibility.size(0) != position_lrs_.size(0)) {
    throw std::runtime_error(
        "[WARNING] Visibility tensor size doesn't match position_lrs_ size");
  }

  // std::cout << "[DEBUG-Optimizer] Pre-update position learning rates: "
  //           << "max =" << position_lrs_.max().item<float>()
  //           << ", min =" << position_lrs_.min().item<float>()
  //           << ", mean =" << position_lrs_.mean().item<float>() << std::endl;

  position_lrs_.index_put_(
      {visibility}, position_lrs_.index({visibility}) * position_lr_decay_);
  position_lrs_.clamp_min_(position_lr_min_);

  // std::cout << "[DEBUG-Optimizer] Updated position learning rates: "
  //           << "max =" << position_lrs_.max().item<float>()
  //           << ", min =" << position_lrs_.min().item<float>()
  //           << ", mean =" << position_lrs_.mean().item<float>() << std::endl;
}

void GaussianModel::optimizerStep(torch::Tensor& visibility, const uint32_t N) {
  torch::NoGradGuard no_grad;

  auto& param_groups = optimizer_->param_groups();

  for (size_t group_idx = 0; group_idx < param_groups.size(); ++group_idx) {
    auto& group = param_groups[group_idx];
    auto& param = group.params()[0];

    if (!param.grad().defined()) continue;

    // Get optimizer state
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

    // Learning rate handling per parameter type
    torch::Tensor lr_tensor;
    if (group_idx == 0) {
      // GROUP 0: Positions - use per-primitive learning rates
      lr_tensor = position_lrs_;
    } else {
      // ALL OTHER GROUPS: Use fixed scalar learning rates
      float scalar_lr = group.options().get_lr();
      lr_tensor = torch::tensor(scalar_lr,
                                torch::TensorOptions().device(param.device()));
    }

    const uint32_t M = param.numel() / N;
    auto options = static_cast<torch::optim::AdamOptions&>(group.options());
    auto exp_avg = param_state.exp_avg();
    auto exp_avg_sq = param_state.exp_avg_sq();
    auto grad = param.grad();
    auto eps = options.eps();

    // Adam update
    adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility, lr_tensor,
               std::get<0>(options.betas()), std::get<1>(options.betas()), eps,
               N, M);
  }

  // Update learning rates AFTER Adam step
  updateLearningRates(visibility);
}

// ==================================
// param_groups[0] = xyz_
// param_groups[1] = feature_dc_
// param_groups[2] = feature_rest_
// param_groups[3] = opacity_
// param_groups[4] = scaling_
// param_groups[5] = rotation_
// ==================================
void GaussianModel::setFeatureLearningRate(float feature_lr) {
  optimizer_->param_groups()[1].options().set_lr(feature_lr);
  optimizer_->param_groups()[2].options().set_lr(feature_lr / 20.0);
}
void GaussianModel::setOpacityLearningRate(float opacity_lr) {
  optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModel::setScalingLearningRate(float scaling_lr) {
  optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModel::setRotationLearningRate(float rot_lr) {
  optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModel::resetOpacity() {
  torch::Tensor opacities_new = general_utils::inverse_sigmoid(
      torch::min(this->getOpacityActivation(),
                 torch::ones_like(this->getOpacityActivation() * 0.01)));
  torch::Tensor optimizable_tensors =
      this->replaceTensorToOptimizer(opacities_new, 3);  // "opacity"
  this->opacity_ = optimizable_tensors;
  this->Tensor_vec_opacity_ = {this->opacity_};
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor,
                                                      int tensor_idx) {
  std::cout
      << "[DEBUG-Optimizer] Starting replaceTensorToOptimizer for tensor_idx: "
      << tensor_idx << std::endl;

  if (!this->optimizer_) {
    std::cerr << "ERROR: Optimizer is null!" << std::endl;
    throw std::runtime_error("Null optimizer in replaceTensorToOptimizer");
  }

  std::cout << "[DEBUG-Optimizer] Param groups size: "
            << this->optimizer_->param_groups().size() << std::endl;

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

  std::cout << "[DEBUG-Optimizer] Checking state for key..." << std::endl;
  if (state.find(key) == state.end()) {
    std::cerr << "WARNING: No optimizer state found for tensor_idx "
              << tensor_idx << std::endl;
    // Create a new state instead of crashing
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(0);
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    state[key] = std::move(new_state);
    std::cout << "[DEBUG-Optimizer] Created new state" << std::endl;
  }

  try {
    auto& stored_state =
        static_cast<torch::optim::AdamParamState&>(*state[key]);
    std::cout << "[DEBUG-Optimizer] Got stored state with step: "
              << stored_state.step() << std::endl;

    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());

    std::cout << "[DEBUG-Optimizer] Creating exp_avg and exp_avg_sq..."
              << std::endl;
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));

    std::cout << "[DEBUG-Optimizer] Erasing old state..." << std::endl;
    state.erase(key);

    std::cout << "[DEBUG-Optimizer] Setting requires_grad..." << std::endl;
    param = tensor.requires_grad_();
    key = param.unsafeGetTensorImpl();

    std::cout << "[DEBUG-Optimizer] Storing new state..." << std::endl;
    state[key] = std::move(new_state);

    std::cout << "[DEBUG-Optimizer] Completed replaceTensorToOptimizer for "
                 "tensor_idx: "
              << tensor_idx << std::endl;
    return param;
  } catch (const std::exception& e) {
    std::cerr << "ERROR in replaceTensorToOptimizer: " << e.what() << std::endl;
    throw;
  }
}

void GaussianModel::prunePoints(torch::Tensor& mask) {
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
      new_state->exp_avg(
          stored_state.exp_avg().index({valid_points_mask}).clone());
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index({valid_points_mask}).clone());
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

  this->xyz_gradient_accum_ =
      this->xyz_gradient_accum_.index({valid_points_mask});

  this->denom_ = this->denom_.index({valid_points_mask});
  this->max_radii2D_ = this->max_radii2D_.index({valid_points_mask});
  this->position_lrs_ = this->position_lrs_.index({valid_points_mask});
}

void GaussianModel::densificationPostfix(torch::Tensor& new_xyz,
                                         torch::Tensor& new_features_dc,
                                         torch::Tensor& new_features_rest,
                                         torch::Tensor& new_opacities,
                                         torch::Tensor& new_scaling,
                                         torch::Tensor& new_rotation,
                                         torch::Tensor& new_exist_since_iter) {
  auto old_max_radii2D = this->max_radii2D_.clone();
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
      new_state->step(stored_state.step());
      new_state->exp_avg(torch::cat(
          {stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)},
          /*dim=*/0));
      new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(),
                                        torch::zeros_like(extension_tensor)},
                                       /*dim=*/0));
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);

      optimizable_tensors[group_idx] = param;
    } else {
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
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

  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));
  auto new_max_radii2D_extension = torch::zeros(
      {new_xyz.size(0)}, torch::TensorOptions().device(device_type_));
  this->max_radii2D_ =
      torch::cat({old_max_radii2D, new_max_radii2D_extension}, /*dim=*/0);

  int num_new_primitives = new_xyz.size(0);
  torch::Tensor new_position_lrs =
      torch::full({num_new_primitives}, position_lr_init_,
                  torch::TensorOptions().device(device_type_));
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
}

void GaussianModel::densifyAndSplit(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent,
                                    int N) {
  int n_init_points = this->getXYZ().size(0);
  // Extract points that satisfy the gradient condition
  auto padded_grad = torch::zeros({n_init_points},
                                  torch::TensorOptions().device(device_type_));
  padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0))
      .copy_(grads.squeeze());
  auto selected_pts_mask =
      torch::where(padded_grad >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) >
          percentDense() * scene_extent);

  auto new_xyz = this->getXYZ().index({selected_pts_mask}).repeat({N, 1});
  auto samples = torch::normal(0.0f, 1.0f, new_xyz.sizes()).to(device_type_);
  samples.index_put_({samples.abs() > 3.0f}, 0.0f);
  auto new_covs = this->getCovarianceActivation()
                      .index({selected_pts_mask})
                      .repeat({N, 1, 1});
  new_xyz += new_covs.matmul(samples.unsqueeze(-1)).squeeze(-1);
  auto new_scaling = torch::log(
      this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1}) /
      (0.8 * N));  // scaling_inverse_activation
  auto new_rotation = this->rotation_.index({selected_pts_mask}).repeat({N, 1});
  auto new_features_dc =
      this->features_dc_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_features_rest =
      this->features_rest_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_opacity = this->opacity_.index({selected_pts_mask}).repeat({N, 1});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask}).repeat({N});

  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacity, new_scaling, new_rotation,
                             new_exist_since_iter);

  auto prune_filter = torch::cat(
      {selected_pts_mask,
       torch::zeros(
           {(N * selected_pts_mask.sum()).item<int>()},
           torch::TensorOptions().device(device_type_).dtype(torch::kBool))});
  this->prunePoints(prune_filter);
}

void GaussianModel::densifyAndClone(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent) {
  // Extract points that satisfy the gradient condition
  auto selected_pts_mask = torch::where(
      torch::frobenius_norm(grads, /*dim=*/-1) >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) <=
          percentDense() * scene_extent);

  auto new_xyz = this->xyz_.index({selected_pts_mask});
  auto new_features_dc = this->features_dc_.index({selected_pts_mask});
  auto new_features_rest = this->features_rest_.index({selected_pts_mask});
  auto new_opacities = this->opacity_.index({selected_pts_mask});
  auto new_scaling = this->scaling_.index({selected_pts_mask});
  auto new_rotation = this->rotation_.index({selected_pts_mask});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask});

  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacities, new_scaling, new_rotation,
                             new_exist_since_iter);
}

void GaussianModel::densifyAndPrune(float max_grad,
                                    float min_opacity,
                                    float extent,
                                    int max_screen_size) {
  auto grads = this->xyz_gradient_accum_ / this->denom_;
  grads.index_put_({grads.isnan()}, 0.0f);

  std::cout << "=== DENSIFY AND PRUNE DEBUG ===" << std::endl;
  std::cout << "max_screen_size: " << max_screen_size << std::endl;
  std::cout << "min_opacity: " << min_opacity << std::endl;

  this->densifyAndClone(grads, max_grad, extent);
  this->densifyAndSplit(grads, max_grad, extent);

  auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
  auto opacity_prune_count = prune_mask.sum().item<int>();
  std::cout << "Points to prune due to low opacity: " << opacity_prune_count
            << std::endl;

  if (max_screen_size) {
    auto big_points_vs = this->max_radii2D_ > max_screen_size;
    auto big_points_count = big_points_vs.sum().item<int>();
    std::cout << "Points with radii > " << max_screen_size << ": "
              << big_points_count << std::endl;

    // Debug: show some max_radii2D_ values
    auto max_radii_stats =
        torch::tensor({this->max_radii2D_.min().item<float>(),
                       this->max_radii2D_.max().item<float>(),
                       this->max_radii2D_.mean().item<float>()});
    std::cout << "max_radii2D_ - min: " << max_radii_stats[0].item<float>()
              << ", max: " << max_radii_stats[1].item<float>()
              << ", mean: " << max_radii_stats[2].item<float>() << std::endl;

    prune_mask = torch::logical_or(prune_mask, big_points_vs);
  }

  auto total_prune_count = prune_mask.sum().item<int>();
  auto total_points = prune_mask.size(0);
  std::cout << "Total points to prune: " << total_prune_count << " out of "
            << total_points << std::endl;

  this->prunePoints(prune_mask);
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::prune(float min_opacity, int max_screen_size) {
  auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
  // auto opacity_prune_count = prune_mask.sum().item<int>();
  // std::cout << "Points to prune due to low opacity: " << opacity_prune_count
  //           << std::endl;

  if (max_screen_size) {
    auto big_points_vs = this->max_radii2D_ > max_screen_size;
    auto big_points_count = big_points_vs.sum().item<int>();
    // std::cout << "Points with radii > " << max_screen_size << ": "
    //           << big_points_count << std::endl;

    // Debug: show some max_radii2D_ values
    // auto max_radii_stats =
    //     torch::tensor({this->max_radii2D_.min().item<float>(),
    //                    this->max_radii2D_.max().item<float>(),
    //                    this->max_radii2D_.mean().item<float>()});
    // std::cout << "max_radii2D_ - min: " << max_radii_stats[0].item<float>()
    //           << ", max: " << max_radii_stats[1].item<float>()
    //           << ", mean: " << max_radii_stats[2].item<float>() << std::endl;

    prune_mask = torch::logical_or(prune_mask, big_points_vs);
  }

  // auto total_prune_count = prune_mask.sum().item<int>();
  // auto total_points = prune_mask.size(0);
  // std::cout << "Total points to prune: " << total_prune_count << " out of "
  //           << total_points << std::endl;

  this->prunePoints(prune_mask);
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::addDensificationStats(
    const torch::Tensor& viewspace_point_tensor,
    const torch::Tensor& update_filter) {
  this->xyz_gradient_accum_.index_put_(
      {update_filter},
      torch::frobenius_norm(viewspace_point_tensor.grad().index(
                                {update_filter, torch::indexing::Slice(0, 2)}),
                            /*dim=*/-1,
                            /*keepdim=*/true),
      /*accumulate=*/true);

  this->denom_.index_put_({update_filter},
                          this->denom_.index({update_filter}) + 1);
}

// void GaussianModel::increasePointsIterationsOfExistence(const int i)
// {
//     this->exist_since_iter_ += i;
// }

void GaussianModel::loadPly(std::filesystem::path ply_path) {
  std::ifstream instream_binary(ply_path, std::ios::binary);
  if (!instream_binary.is_open() || instream_binary.fail())
    throw std::runtime_error("Fail to open ply file at " + ply_path.string());
  instream_binary.seekg(0, std::ios::beg);

  tinyply::PlyFile ply_file;
  ply_file.parse_header(instream_binary);

  std::cout << "\t[ply_header] Type: "
            << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
  for (const auto& c : ply_file.get_comments())
    std::cout << "\t[ply_header] Comment: " << c << std::endl;
  for (const auto& c : ply_file.get_info())
    std::cout << "\t[ply_header] Info: " << c << std::endl;

  for (const auto& e : ply_file.get_elements()) {
    std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
              << std::endl;
    for (const auto& p : e.properties) {
      std::cout << "\t[ply_header] \tproperty: " << p.name
                << " (type=" << tinyply::PropertyTable[p.propertyType].str
                << ")";
      if (p.isList)
        std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                  << ")";
      std::cout << std::endl;
    }
  }

  std::shared_ptr<tinyply::PlyData> xyz, f_dc, f_rest, opacity, scales, rot;

  try {
    xyz = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    f_dc = ply_file.request_properties_from_element(
        "vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  int n_f_rest = ((sh_degree_ + 1) * (sh_degree_ + 1) - 1) * 3;
  if (n_f_rest >= 0) {
    std::vector<std::string> f_rest_element_names(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
      f_rest_element_names[i] = "f_rest_" + std::to_string(i);
    try {
      f_rest = ply_file.request_properties_from_element("vertex",
                                                        f_rest_element_names);
    } catch (const std::exception& e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
  }

  try {
    opacity = ply_file.request_properties_from_element("vertex", {"opacity"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    scales = ply_file.request_properties_from_element(
        "vertex", {"scale_0", "scale_1", "scale_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    rot = ply_file.request_properties_from_element(
        "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  ply_file.read(instream_binary);

  if (xyz) std::cout << "\tRead " << xyz->count << " total xyz " << std::endl;
  if (f_dc)
    std::cout << "\tRead " << f_dc->count << " total f_dc " << std::endl;
  if (f_rest)
    std::cout << "\tRead " << f_rest->count << " total f_rest " << std::endl;
  if (opacity)
    std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
  if (scales)
    std::cout << "\tRead " << scales->count << " total scales " << std::endl;
  if (rot) std::cout << "\tRead " << rot->count << " total rot " << std::endl;

  // Data to std::vector
  const int num_points = xyz->count;

  const std::size_t n_xyz_bytes = xyz->buffer.size_bytes();
  std::vector<float> xyz_vector(xyz->count * 3);
  std::memcpy(xyz_vector.data(), xyz->buffer.get(), n_xyz_bytes);

  const std::size_t n_f_dc_bytes = f_dc->buffer.size_bytes();
  std::vector<float> f_dc_vector(f_dc->count * 3);
  std::memcpy(f_dc_vector.data(), f_dc->buffer.get(), n_f_dc_bytes);

  const std::size_t n_f_rest_bytes = f_rest->buffer.size_bytes();
  std::vector<float> f_rest_vector(f_rest->count * n_f_rest);
  std::memcpy(f_rest_vector.data(), f_rest->buffer.get(), n_f_rest_bytes);

  const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
  std::vector<float> opacity_vector(opacity->count * 1);
  std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

  const std::size_t n_scales_bytes = scales->buffer.size_bytes();
  std::vector<float> scales_vector(scales->count * 3);
  std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

  const std::size_t n_rot_bytes = rot->buffer.size_bytes();
  std::vector<float> rot_vector(rot->count * 4);
  std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

  // std::vector to torch::Tensor
  this->xyz_ = torch::from_blob(xyz_vector.data(), {num_points, 3},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .to(device_type_);

  this->features_dc_ =
      torch::from_blob(f_dc_vector.data(), {num_points, 3, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->features_rest_ =
      torch::from_blob(f_rest_vector.data(), {num_points, 3, n_f_rest / 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->opacity_ =
      torch::from_blob(opacity_vector.data(), {num_points, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->scaling_ =
      torch::from_blob(scales_vector.data(), {num_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->rotation_ =
      torch::from_blob(rot_vector.data(), {num_points, 4},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->sh_degree_ = sh_degree_;
}

void GaussianModel::savePly(std::filesystem::path result_path) {
  // Prepare data to write
  torch::Tensor xyz = this->xyz_.detach().cpu();
  torch::Tensor normals = torch::zeros_like(xyz);
  torch::Tensor f_dc =
      this->features_dc_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
  torch::Tensor f_rest = this->features_rest_.detach()
                             .transpose(1, 2)
                             .flatten(1)
                             .contiguous()
                             .cpu();
  torch::Tensor opacities = this->opacity_.detach().cpu();
  torch::Tensor scale = this->scaling_.detach().cpu();
  torch::Tensor rotation = this->rotation_.detach().cpu();

  std::filebuf fb_binary;
  fb_binary.open(result_path, std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + result_path.string());

  tinyply::PlyFile result_file;

  // xyz
  result_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.size(0),
      reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()), tinyply::Type::INVALID,
      0);

  // normals
  result_file.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normals.size(0),
      reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_dc
  std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
  std::vector<std::string> property_names_f_dc(n_f_dc);
  for (int i = 0; i < n_f_dc; ++i)
    property_names_f_dc[i] = "f_dc_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_dc, tinyply::Type::FLOAT32,
      this->features_dc_.size(0),
      reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_rest
  std::size_t n_f_rest =
      this->features_rest_.size(1) * this->features_rest_.size(2);
  std::vector<std::string> property_names_f_rest(n_f_rest);
  for (int i = 0; i < n_f_rest; ++i)
    property_names_f_rest[i] = "f_rest_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_rest, tinyply::Type::FLOAT32,
      this->features_rest_.size(0),
      reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // opacities
  result_file.add_properties_to_element(
      "vertex", {"opacity"}, tinyply::Type::FLOAT32, opacities.size(0),
      reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // scale
  std::size_t n_scale = scale.size(1);
  std::vector<std::string> property_names_scale(n_scale);
  for (int i = 0; i < n_scale; ++i)
    property_names_scale[i] = "scale_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_scale, tinyply::Type::FLOAT32, scale.size(0),
      reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // rotation
  std::size_t n_rotation = rotation.size(1);
  std::vector<std::string> property_names_rotation(n_rotation);
  for (int i = 0; i < n_rotation; ++i)
    property_names_rotation[i] = "rot_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_rotation, tinyply::Type::FLOAT32,
      rotation.size(0), reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // Write the file
  result_file.write(outstream_binary, true);

  fb_binary.close();
}

float GaussianModel::percentDense() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return percent_dense_;
}

void GaussianModel::setPercentDense(const float percent_dense) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  percent_dense_ = percent_dense;
}

void GaussianModel::save_checkpoint(const std::string& path) {
  // Create directory if it doesn't exist
  std::filesystem::create_directories(std::filesystem::path(path));

  // Save model tensors as before
  torch::serialize::OutputArchive model_archive;
  model_archive.write("xyz_", xyz_);
  model_archive.write("features_dc_", features_dc_);
  model_archive.write("features_rest_", features_rest_);
  model_archive.write("scaling_", scaling_);
  model_archive.write("rotation_", rotation_);
  model_archive.write("opacity_", opacity_);
  model_archive.write("max_radii2D_", max_radii2D_);
  model_archive.write("xyz_gradient_accum_", xyz_gradient_accum_);
  model_archive.write("denom_", denom_);
  model_archive.write("exist_since_iter_", exist_since_iter_);
  model_archive.write("position_lrs_", position_lrs_);
  model_archive.save_to(path + "/model.pt");

  assert(sh_degree_ <= 3);
  assert(position_lr_init_ >= 0);
  assert(position_lr_decay_ > 0);
  assert(position_lr_min_ >= 0);
  assert(local_iteration_ >= 0);

  assert(percent_dense_ > 0 && percent_dense_ < 1);

  // Save configuration as before
  torch::serialize::OutputArchive config_archive;
  config_archive.write("sh_degree_", torch::tensor(sh_degree_));
  config_archive.write("position_lr_init_", torch::tensor(position_lr_init_));
  config_archive.write("position_lr_decay_", torch::tensor(position_lr_decay_));
  config_archive.write("position_lr_min_", torch::tensor(position_lr_min_));
  config_archive.write("local_iteration_", torch::tensor(local_iteration_));
  config_archive.write("percent_dense_", torch::tensor(percent_dense_));

  // Manual learning rate saving
  if (optimizer_) {
    // std::cout << "Saving learning rates:" << std::endl;
    for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
      float lr = optimizer_->param_groups()[i].options().get_lr();

      // Safety check for NaN or inf learning rates
      if (std::isnan(lr) || std::isinf(lr) || lr <= 0.0f || lr > 1.0f) {
        std::cerr << "ERROR: Found invalid learning rate (" << lr
                  << ") for group " << i << std::endl;
        throw std::runtime_error("Invalid LR to save");
      }

      std::string lr_name = "param_group_" + std::to_string(i) + "_lr";
      config_archive.write(lr_name, torch::tensor(static_cast<float>(lr)));
      // std::cout << "  Group " << i << " LR: " << lr << std::endl;
    }
  }
  config_archive.save_to(path + "/config.pt");

  // Manually save optimizer state
  if (optimizer_) {
    torch::serialize::OutputArchive opt_archive;

    // Save number of param groups
    opt_archive.write(
        "num_param_groups",
        torch::tensor(static_cast<int64_t>(optimizer_->param_groups().size())));

    // For each parameter group and parameter, save the state
    auto& state = optimizer_->state();
    int param_idx = 0;

    for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
         ++group_idx) {
      auto& group = optimizer_->param_groups()[group_idx];

      // Save parameter group size
      opt_archive.write(
          "group_" + std::to_string(group_idx) + "_size",
          torch::tensor(static_cast<int64_t>(group.params().size())));

      for (size_t j = 0; j < group.params().size(); ++j) {
        auto& param = group.params()[j];
        auto key = param.unsafeGetTensorImpl();

        // Save shape information
        std::vector<int64_t> sizes_vec;
        for (int d = 0; d < param.dim(); ++d) {
          sizes_vec.push_back(param.size(d));
        }
        opt_archive.write("param_" + std::to_string(param_idx) + "_shape",
                          torch::tensor(sizes_vec));

        if (state.find(key) != state.end()) {
          auto& param_state =
              static_cast<torch::optim::AdamParamState&>(*state[key]);

          // Save state components
          opt_archive.write(
              "param_" + std::to_string(param_idx) + "_step",
              torch::tensor(static_cast<int64_t>(param_state.step())));
          opt_archive.write("param_" + std::to_string(param_idx) + "_exp_avg",
                            param_state.exp_avg());
          opt_archive.write(
              "param_" + std::to_string(param_idx) + "_exp_avg_sq",
              param_state.exp_avg_sq());
        } else {
          // Save placeholder for parameters without state
          opt_archive.write("param_" + std::to_string(param_idx) + "_step",
                            torch::tensor(static_cast<int64_t>(0)));
          opt_archive.write("param_" + std::to_string(param_idx) + "_exp_avg",
                            torch::zeros_like(param));
          opt_archive.write(
              "param_" + std::to_string(param_idx) + "_exp_avg_sq",
              torch::zeros_like(param));
        }

        param_idx++;
      }
    }

    opt_archive.save_to(path + "/optimizer_manual.pt");
    // std::cout << "Optimizer state manually saved" << std::endl;
  }

  // std::cout << "Checkpoint saved to " << path << std::endl;
}

void GaussianModel::load_checkpoint_incremental(
    const std::string& path,
    const GaussianOptimizationParams& training_args,
    bool load_auxiliary_tensors,
    bool load_optimizer_state,
    bool load_existence_info,
    bool normalize_quaternions,
    bool clear_cache_after_load) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Checkpoint directory does not exist: " + path);
  }

  // 1. Load model tensors
  torch::serialize::InputArchive model_archive;
  if (std::filesystem::exists(path + "/model.pt")) {
    try {
      model_archive.load_from(path + "/model.pt");
    } catch (const std::exception& e) {
      throw std::runtime_error("Failed to load model: " +
                               std::string(e.what()));
    }
  } else {
    throw std::runtime_error("No model file found at " + path);
  }

  // Always load core tensors
  model_archive.read("xyz_", xyz_);
  model_archive.read("features_dc_", features_dc_);
  model_archive.read("features_rest_", features_rest_);
  model_archive.read("scaling_", scaling_);
  model_archive.read("rotation_", rotation_);
  model_archive.read("opacity_", opacity_);

  // Load auxiliary tensors
  try {
    model_archive.read("max_radii2D_", max_radii2D_);
    model_archive.read("xyz_gradient_accum_", xyz_gradient_accum_);
    model_archive.read("denom_", denom_);
    model_archive.read("exist_since_iter_", exist_since_iter_);
    model_archive.read("position_lrs_", position_lrs_);
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to load auxiliary tensors info: " << e.what()
              << std::endl;
  }

  // Load model configuration
  torch::serialize::InputArchive config_archive;
  if (std::filesystem::exists(path + "/config.pt")) {
    try {
      config_archive.load_from(path + "/config.pt");
    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to load chunk config: " << e.what()
                << std::endl;
    }
  } else {
    std::cerr << "Warning: Chunk config not found at " << path + "/config.pt"
              << std::endl;
  }

  // Temporary tensors to hold the loaded scalar values
  torch::Tensor sh_degree_tensor, local_iteration_tensor;
  torch::Tensor percent_dense_tensor, position_lr_init_tensor,
      position_lr_decay_tensor, position_lr_min_tensor;

  config_archive.read("sh_degree_", sh_degree_tensor);
  config_archive.read("local_iteration_", local_iteration_tensor);

  // Load float values
  config_archive.read("percent_dense_", percent_dense_tensor);
  config_archive.read("position_lr_init_", position_lr_init_tensor);
  config_archive.read("position_lr_decay_", position_lr_decay_tensor);
  config_archive.read("position_lr_min_", position_lr_min_tensor);

  // Convert tensors back to native types
  sh_degree_ = sh_degree_tensor.item<int>();
  assert(sh_degree_ <= 3);

  local_iteration_ = local_iteration_tensor.item<int>();
  assert(local_iteration_ >= 0);

  percent_dense_ = percent_dense_tensor.item<float>();
  assert(percent_dense_ > 0 && percent_dense_ < 1);
  position_lr_init_ = position_lr_init_tensor.item<float>();
  assert(position_lr_init_ > 0);
  position_lr_decay_ = position_lr_decay_tensor.item<float>();
  assert(position_lr_decay_ > 0);
  position_lr_min_ = position_lr_min_tensor.item<float>();
  assert(position_lr_min_ > 0);

  // std::cout << "lr_init_tensor " << lr_init_tensor << std::endl;
  // std::cout << "lr_init_ " << lr_init_ << std::endl;

  // 3. Prepare tensors for optimizer (with optional normalization)
  // if (normalize_quaternions) {
  //   rotation_ = torch::nn::functional::normalize(
  //       rotation_, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  // }

  // Ensure all tensors require gradients
  xyz_ = xyz_.requires_grad_(true);
  features_dc_ = features_dc_.contiguous().requires_grad_(true);
  features_rest_ = features_rest_.contiguous().requires_grad_(true);
  scaling_ = scaling_.requires_grad_(true);
  rotation_ = rotation_.requires_grad_(true);
  opacity_ = opacity_.requires_grad_(true);

  // Convert tensors to vectors
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Load learning rates with strict validation
  std::vector<float> learning_rates;

  // Define default learning rates
  std::vector<float> default_learning_rates = {
      training_args.position_lr_init_,   training_args.feature_lr_,
      training_args.feature_lr_ / 20.0f, training_args.opacity_lr_,
      training_args.scaling_lr_,         training_args.rotation_lr_};

  if (std::filesystem::exists(path + "/config.pt")) {
    try {
      config_archive.load_from(path + "/config.pt");

      // Try to load individual learning rates with validation
      torch::Tensor lr_tensor;
      // std::cout << "Loading learning rates:" << std::endl;

      for (int i = 0; i < 6; ++i) {  // We know there are 6 param groups
        std::string lr_name = "param_group_" + std::to_string(i) + "_lr";
        float lr = default_learning_rates[i];  // Default value

        try {
          config_archive.read(lr_name, lr_tensor);
          float loaded_lr = lr_tensor.item<float>();

          // Validate the loaded learning rate
          if (!std::isnan(loaded_lr) && !std::isinf(loaded_lr) &&
              loaded_lr > 0.0f && loaded_lr < 1.0f) {
            lr = loaded_lr;
          } else {
            std::cerr << "WARNING: Invalid learning rate (" << loaded_lr
                      << ") for group " << i << ". Using default "
                      << default_learning_rates[i] << " instead." << std::endl;
          }
        } catch (...) {
          std::cout << "  Group " << i << " LR not found, using default: " << lr
                    << std::endl;
        }

        learning_rates.push_back(lr);
        // std::cout << "  Group " << i << " LR: " << lr << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Error loading config: " << e.what() << std::endl;
      std::cerr << "Using default learning rates" << std::endl;
      learning_rates = default_learning_rates;
    }
  } else {
    std::cout << "No config file found, using default learning rates"
              << std::endl;
    learning_rates = default_learning_rates;
  }

  // 4. Create optimizer with properly set learning rates
  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(learning_rates[0]);  // Set initial LR for first group
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(learning_rates[0]);

  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(learning_rates[1]);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(learning_rates[2]);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(learning_rates[3]);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(learning_rates[4]);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(learning_rates[5]);

  // std::cout << "Verifying optimizer learning rates:" << std::endl;
  for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
    float actual_lr = optimizer_->param_groups()[i].options().get_lr();
    // std::cout << "  Group " << i << " actual LR: " << actual_lr << std::endl;

    // Double-check for NaN (this should never happen with our validation)
    if (std::isnan(actual_lr)) {
      std::cerr
          << "CRITICAL: NaN learning rate detected after setting! Fixing..."
          << std::endl;
      optimizer_->param_groups()[i].options().set_lr(default_learning_rates[i]);
    }
  }

  // 5. Manually load optimizer state
  if (load_optimizer_state &&
      std::filesystem::exists(path + "/optimizer_manual.pt")) {
    try {
      torch::serialize::InputArchive opt_archive;
      opt_archive.load_from(path + "/optimizer_manual.pt");

      torch::Tensor num_groups_tensor;
      opt_archive.read("num_param_groups", num_groups_tensor);
      int64_t num_groups = num_groups_tensor.item<int64_t>();

      if (num_groups != optimizer_->param_groups().size()) {
        std::cerr << "Warning: Saved optimizer had " << num_groups
                  << " parameter groups, but current optimizer has "
                  << optimizer_->param_groups().size() << std::endl;
        throw std::runtime_error("Parameter group count mismatch");
      }

      int param_idx = 0;
      auto& state = optimizer_->state();

      for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
           ++group_idx) {
        // Check group size
        torch::Tensor group_size_tensor;
        opt_archive.read("group_" + std::to_string(group_idx) + "_size",
                         group_size_tensor);
        int64_t group_size = group_size_tensor.item<int64_t>();

        auto& group = optimizer_->param_groups()[group_idx];
        if (group_size != group.params().size()) {
          std::cerr << "Warning: Group " << group_idx
                    << " size mismatch: saved=" << group_size
                    << ", current=" << group.params().size() << std::endl;
          throw std::runtime_error("Parameter count mismatch in group");
        }

        for (size_t j = 0; j < group.params().size(); ++j) {
          auto& param = group.params()[j];
          auto key = param.unsafeGetTensorImpl();

          // Check parameter shape
          torch::Tensor shape_tensor;
          opt_archive.read("param_" + std::to_string(param_idx) + "_shape",
                           shape_tensor);
          std::vector<int64_t> shape;
          for (int i = 0; i < shape_tensor.size(0); ++i) {
            shape.push_back(shape_tensor[i].item<int64_t>());
          }

          // Create the parameter state
          torch::Tensor step_tensor, exp_avg, exp_avg_sq;
          opt_archive.read("param_" + std::to_string(param_idx) + "_step",
                           step_tensor);
          opt_archive.read("param_" + std::to_string(param_idx) + "_exp_avg",
                           exp_avg);
          opt_archive.read("param_" + std::to_string(param_idx) + "_exp_avg_sq",
                           exp_avg_sq);

          // Verify tensor size
          bool sizes_match = true;
          if (param.dim() != shape.size()) {
            sizes_match = false;
          } else {
            for (int d = 0; d < param.dim(); ++d) {
              if (param.size(d) != shape[d]) {
                sizes_match = false;
                break;
              }
            }
          }

          if (!sizes_match) {
            // Print the expected shape from the archive
            std::cerr << "Expected shape: [";
            for (size_t d = 0; d < shape.size(); ++d) {
              std::cerr << shape[d];
              if (d < shape.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;

            // Print the actual shape of the parameter
            std::cerr << "Actual shape: [";
            for (int d = 0; d < param.dim(); ++d) {
              std::cerr << param.size(d);
              if (d < param.dim() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;

            std::cerr << "Warning: Parameter shape mismatch for param "
                      << param_idx << std::endl;
            throw std::runtime_error("Parameter shape mismatch");
            // Create zero state instead of throwing
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(0);
            new_state->exp_avg(torch::zeros_like(param));
            new_state->exp_avg_sq(torch::zeros_like(param));
            state[key] = std::move(new_state);
          } else {
            // Create state with loaded values
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(step_tensor.item<int64_t>());
            new_state->exp_avg(exp_avg.to(param.device()));
            new_state->exp_avg_sq(exp_avg_sq.to(param.device()));

            // // Optional: clip extreme values
            // if (new_state->exp_avg().abs().max().item<float>() > 1e3) {
            //   std::cout << "Clipping extreme exp_avg values for param "
            //             << param_idx << std::endl;
            //   new_state->exp_avg().clamp_(-1e3, 1e3);
            // }
            // if (new_state->exp_avg_sq().max().item<float>() > 1e6) {
            //   std::cout << "Clipping extreme exp_avg_sq values for param "
            //             << param_idx << std::endl;
            //   new_state->exp_avg_sq().clamp_(0, 1e6);
            // }

            state[key] = std::move(new_state);
          }

          param_idx++;
        }
      }

      // std::cout << "Optimizer state manually loaded successfully" <<
      // std::endl;

      // Verify loaded state
      bool state_ok = true;
      for (auto& param_group : optimizer_->param_groups()) {
        for (auto& param : param_group.params()) {
          auto key = param.unsafeGetTensorImpl();
          if (state.find(key) == state.end()) {
            state_ok = false;
            throw std::runtime_error(
                "Error: Parameter missing from state after loading");
          }
        }
      }

      // if (state_ok) {
      //   std::cout << "All parameters have state after loading" << std::endl;
      // }

      // Debug print the actual learning rates
      // std::cout << "\n==== ACTUAL LEARNING RATES AFTER LOADING ====\n";
      // for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
      //   std::cout << "Group " << i
      //             << " LR: " <<
      //             optimizer_->param_groups()[i].options().get_lr()
      //             << std::endl;
      // }

      // {
      //   std::cout << "Checking for high momentum values after loading..."
      //             << std::endl;
      //   auto& state = optimizer_->state();
      //   int clipped_count = 0;

      //   for (auto& param_group : optimizer_->param_groups()) {
      //     for (auto& param : param_group.params()) {
      //       auto key = param.unsafeGetTensorImpl();
      //       if (state.find(key) != state.end()) {
      //         auto& param_state =
      //             static_cast<torch::optim::AdamParamState&>(*state[key]);

      //         // Check for empty tensors before calling max()
      //         if (param_state.exp_avg().numel() == 0 ||
      //             param_state.exp_avg_sq().numel() == 0) {
      //           std::cout << "Warning: Empty momentum tensor detected,
      //           skipping"
      //                     << std::endl;
      //           continue;
      //         }

      //         // Check for extreme values (safely)
      //         float max_exp_avg = 0.0f;
      //         float max_exp_avg_sq = 0.0f;

      //         try {
      //           max_exp_avg =
      //           param_state.exp_avg().abs().max().item<float>();
      //         } catch (const std::exception& e) {
      //           std::cout << "Warning: Error getting max of exp_avg: "
      //                     << e.what() << std::endl;
      //         }

      //         try {
      //           max_exp_avg_sq =
      //           param_state.exp_avg_sq().max().item<float>();
      //         } catch (const std::exception& e) {
      //           std::cout << "Warning: Error getting max of exp_avg_sq: "
      //                     << e.what() << std::endl;
      //         }

      //         // More conservative clipping
      //         if (max_exp_avg > 10.0f || std::isnan(max_exp_avg)) {
      //           clipped_count++;
      //           // Scale down rather than hard clipping
      //           param_state.exp_avg().mul_(10.0f / (max_exp_avg + 1e-6f));
      //         }

      //         if (max_exp_avg_sq > 100.0f || std::isnan(max_exp_avg_sq)) {
      //           clipped_count++;
      //           // Scale down squared values
      //           param_state.exp_avg_sq().mul_(100.0f /
      //                                         (max_exp_avg_sq + 1e-6f));
      //         }

      //         // Final check for NaN
      //         if (param_state.exp_avg().isnan().any().item<bool>() ||
      //             param_state.exp_avg_sq().isnan().any().item<bool>()) {
      //           std::cout << "  Resetting NaN values in momentum" <<
      //           std::endl; param_state.exp_avg().nan_to_num_();
      //           param_state.exp_avg_sq().nan_to_num_();
      //         }
      //       }
      //     }
      //   }

      //   std::cout << "  Clipped " << clipped_count << " momentum tensors"
      //             << std::endl;
      // }

    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to load optimizer state: " << e.what()
                << std::endl;
      throw std::runtime_error("Warning: Failed to load optimizer state");
      std::cerr << "Continuing with newly initialized optimizer" << std::endl;

      // Initialize fresh state
      auto& state = optimizer_->state();
      state.clear();
      for (auto& param_group : optimizer_->param_groups()) {
        for (auto& param : param_group.params()) {
          auto key = param.unsafeGetTensorImpl();
          auto new_state = std::make_unique<torch::optim::AdamParamState>();
          new_state->step(0);
          new_state->exp_avg(torch::zeros_like(param));
          new_state->exp_avg_sq(torch::zeros_like(param));
          state[key] = std::move(new_state);
        }
      }
    }
  } else if (load_optimizer_state) {
    std::cout << "No optimizer state found at " << path + "/optimizer_manual.pt"
              << std::endl;
    throw std::runtime_error("Optimizer couldn't be loaded");
    std::cout << "Initializing fresh optimizer state" << std::endl;

    // Initialize fresh state
    auto& state = optimizer_->state();
    for (auto& param_group : optimizer_->param_groups()) {
      for (auto& param : param_group.params()) {
        auto key = param.unsafeGetTensorImpl();
        auto new_state = std::make_unique<torch::optim::AdamParamState>();
        new_state->step(0);
        new_state->exp_avg(torch::zeros_like(param));
        new_state->exp_avg_sq(torch::zeros_like(param));
        state[key] = std::move(new_state);
      }
    }
  }

  // Force recompute covariances
  // {
  //   torch::NoGradGuard no_grad;
  //   getCovarianceActivation();
  // }

  // Optional CUDA cache clear
  if (clear_cache_after_load) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  // std::cout << "Loaded " << xyz_.size(0) << " points from checkpoint"
  //           << std::endl;
  // std::cout << "Checkpoint loaded from " << path << std::endl;
}

void GaussianModel::initializeFromExistingGaussians(
    torch::Tensor& points,
    torch::Tensor& features_dc,
    torch::Tensor& features_rest,
    torch::Tensor& opacities,
    torch::Tensor& scaling,
    torch::Tensor& rotation,
    torch::Tensor& exist_since_iter,
    const GaussianOptimizationParams& training_args) {
  // Initialize tensors with explicit cloning to ensure independent storage
  this->xyz_ = points.detach().clone().requires_grad_();
  this->features_dc_ = features_dc.detach().clone().requires_grad_();
  this->features_rest_ = features_rest.detach().clone().requires_grad_();
  this->opacity_ = opacities.detach().clone().requires_grad_();
  this->scaling_ = scaling.detach().clone().requires_grad_();
  this->rotation_ = rotation.detach().clone().requires_grad_();
  this->exist_since_iter_ = exist_since_iter.detach().clone();

  // Initialize tensor vectors
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Initialize tracking variables
  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));

  // Setup optimizer
  setPercentDense(training_args.percent_dense_);
  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));

  // Initialize optimizer with properly set learning rates
  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  int num_gaussians = this->getXYZ().size(0);

  // Position learning rates (per-primitive for positions)
  torch::Tensor position_lrs = torch::full(
      {num_gaussians}, training_args.position_lr_init_,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(
      training_args.position_lr_init_);

  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);
}