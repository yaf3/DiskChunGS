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
      spatial_lr_scale_(0.0),
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
      spatial_lr_scale_(0.0),
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
                                  const torch::Tensor& new_opacities,
                                  const int iteration,
                                  const float spatial_lr_scale) {
  this->spatial_lr_scale_ = spatial_lr_scale;
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

  this->exist_since_iter_ = torch::full(
      {fused_point_cloud.size(0)}, iteration,
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

  // c10::cuda::CUDACachingAllocator::emptyCache();

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
    torch::Tensor& point_transformed_flags,
    const torch::Tensor& diff_pose,
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

  int n_elements = 3;
  std::cout << "First " << n_elements << " exist_since_iter_ elements: ";
  for (int i = 0; i < std::min(n_elements, (int)exist_since_iter_.size(0));
       i++) {
    std::cout << exist_since_iter_[i].item<float>() << " ";
  }
  std::cout << std::endl;

  std::cout << "Creation iter: " << kf_creation_iter << std::endl;

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  std::cout << "[DEBUG] Points unstable mask true count: "
            << point_unstable_flags.sum().item<int>() << std::endl;

  std::cout << "[DEBUG-STPV] Created unstable flags" << std::endl;

  std::cout << "[DEBUG-STPV] Calling transform function" << std::endl;

  scaleAndTransformThenMarkVisiblePoints(
      points, rots, point_transformed_flags, point_unstable_flags, diff_pose,
      kf_world_view_transform, kf_full_proj_transform, num_transformed, scale);

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

  position_lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
  position_lr_decay_ = training_args.position_lr_decay_;
  position_lr_min_ = position_lr_init_ * 0.1f * this->spatial_lr_scale_;

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);  // We'll set individual LRs below
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  // We don't use the pytorch lr for group 0
  optimizer_->param_groups()[0].options().set_lr(0.0f);

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
    auto options = static_cast<torch::optim::AdamOptions&>(group.options());

    if (group_idx == 0) {
      // GROUP 0: Positions - use sparse optimizer with per-primitive learning
      // rates
      const uint32_t M =
          param.numel() / N;  // Parameters per Gaussian (3 for xyz)

      adamUpdate(param, param.grad(), param_state.exp_avg(),
                 param_state.exp_avg_sq(), visibility, position_lrs_,
                 std::get<0>(options.betas()), std::get<1>(options.betas()),
                 options.eps(), N, M);
    } else {
      // ALL OTHER GROUPS: Use basic optimizer with scalar learning rates
      float scalar_lr = group.options().get_lr();

      adamUpdateBasic(param, param.grad(), param_state.exp_avg(),
                      param_state.exp_avg_sq(), scalar_lr,
                      std::get<0>(options.betas()),
                      std::get<1>(options.betas()), options.eps());
    }
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
  // std::cout
  //     << "[DEBUG-Optimizer] Starting replaceTensorToOptimizer for tensor_idx:
  //     "
  //     << tensor_idx << std::endl;

  if (!this->optimizer_) {
    std::cerr << "ERROR: Optimizer is null!" << std::endl;
    throw std::runtime_error("Null optimizer in replaceTensorToOptimizer");
  }

  // std::cout << "[DEBUG-Optimizer] Param groups size: "
  //           << this->optimizer_->param_groups().size() << std::endl;

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

  // std::cout << "[DEBUG-Optimizer] Checking state for key..." << std::endl;
  if (state.find(key) == state.end()) {
    std::cerr << "WARNING: No optimizer state found for tensor_idx "
              << tensor_idx << std::endl;
    // Create a new state instead of crashing
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(0);
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    state[key] = std::move(new_state);
    // std::cout << "[DEBUG-Optimizer] Created new state" << std::endl;
  }

  try {
    auto& stored_state =
        static_cast<torch::optim::AdamParamState&>(*state[key]);
    // std::cout << "[DEBUG-Optimizer] Got stored state with step: "
    //           << stored_state.step() << std::endl;

    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());

    // std::cout << "[DEBUG-Optimizer] Creating exp_avg and exp_avg_sq..."
    //           << std::endl;
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));

    // std::cout << "[DEBUG-Optimizer] Erasing old state..." << std::endl;
    state.erase(key);

    // std::cout << "[DEBUG-Optimizer] Setting requires_grad..." << std::endl;
    param = tensor.requires_grad_();
    key = param.unsafeGetTensorImpl();

    // std::cout << "[DEBUG-Optimizer] Storing new state..." << std::endl;
    state[key] = std::move(new_state);

    // std::cout << "[DEBUG-Optimizer] Completed replaceTensorToOptimizer for "
    //              "tensor_idx: "
    //           << tensor_idx << std::endl;
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
  // c10::cuda::CUDACachingAllocator::emptyCache();
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
  // c10::cuda::CUDACachingAllocator::emptyCache();
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

GaussianTransferData GaussianModel::extractGaussiansWithStates(
    const torch::Tensor& mask) {
  torch::NoGradGuard no_grad;

  GaussianTransferData transfer_data;

  // Extract basic tensors
  transfer_data.points = this->xyz_.index({mask}).detach().clone();
  transfer_data.features_dc = this->features_dc_.index({mask}).detach().clone();
  transfer_data.features_rest =
      this->features_rest_.index({mask}).detach().clone();
  transfer_data.opacities = this->opacity_.index({mask}).detach().clone();
  transfer_data.scaling = this->scaling_.index({mask}).detach().clone();
  transfer_data.rotation = this->rotation_.index({mask}).detach().clone();
  transfer_data.exist_since =
      this->exist_since_iter_.index({mask}).detach().clone();

  // Extract auxiliary states
  transfer_data.position_lrs =
      this->position_lrs_.index({mask}).detach().clone();
  transfer_data.xyz_gradient_accum =
      this->xyz_gradient_accum_.index({mask}).detach().clone();
  transfer_data.denom = this->denom_.index({mask}).detach().clone();
  transfer_data.max_radii2D = this->max_radii2D_.index({mask}).detach().clone();

  // Extract optimizer states for each parameter group
  transfer_data.exp_avg_states.resize(6);
  transfer_data.exp_avg_sq_states.resize(6);
  transfer_data.step_states.resize(6);

  if (this->optimizer_) {
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      auto& param = param_groups[group_idx].params()[0];
      auto key = param.unsafeGetTensorImpl();

      if (state.find(key) != state.end()) {
        auto& param_state =
            static_cast<torch::optim::AdamParamState&>(*state[key]);

        // Extract states for the masked indices
        transfer_data.exp_avg_states[group_idx] =
            param_state.exp_avg().index({mask}).detach().clone();
        transfer_data.exp_avg_sq_states[group_idx] =
            param_state.exp_avg_sq().index({mask}).detach().clone();

        // Step is scalar per parameter, so we replicate it
        int64_t step_value = param_state.step();
        transfer_data.step_states[group_idx] = torch::full(
            {mask.sum().item<int>()}, step_value,
            torch::TensorOptions().dtype(torch::kInt64).device(mask.device()));
      } else {
        // Create zero states if no state exists
        int num_points = mask.sum().item<int>();
        auto tensor_shape = param.index({mask}).sizes().vec();

        transfer_data.exp_avg_states[group_idx] = torch::zeros(
            tensor_shape, torch::TensorOptions().device(param.device()));
        transfer_data.exp_avg_sq_states[group_idx] = torch::zeros(
            tensor_shape, torch::TensorOptions().device(param.device()));
        transfer_data.step_states[group_idx] = torch::zeros(
            {num_points},
            torch::TensorOptions().dtype(torch::kInt64).device(param.device()));
      }
    }
  }

  return transfer_data;
}

// Enhanced method to add Gaussians with their optimizer states
void GaussianModel::addGaussiansWithStates(
    const GaussianTransferData& transfer_data) {
  // First, add the basic tensors using existing method
  auto old_size = this->getXYZ().size(0);
  auto new_points = transfer_data.points.clone();
  auto new_features_dc = transfer_data.features_dc.clone();
  auto new_features_rest = transfer_data.features_rest.clone();
  auto new_opacities = transfer_data.opacities.clone();
  auto new_scaling = transfer_data.scaling.clone();
  auto new_rotation = transfer_data.rotation.clone();
  auto new_exist_since = transfer_data.exist_since.clone();

  this->densificationPostfix(new_points, new_features_dc, new_features_rest,
                             new_opacities, new_scaling, new_rotation,
                             new_exist_since);

  // Now handle the auxiliary states - replace the automatically created ones
  auto new_size = this->getXYZ().size(0);
  auto num_new_points = new_size - old_size;

  // Replace the newly added auxiliary states with transferred ones
  if (transfer_data.position_lrs.defined()) {
    this->position_lrs_.slice(0, old_size, new_size)
        .copy_(transfer_data.position_lrs);
  }

  if (transfer_data.xyz_gradient_accum.defined()) {
    this->xyz_gradient_accum_.slice(0, old_size, new_size)
        .copy_(transfer_data.xyz_gradient_accum);
  }

  if (transfer_data.denom.defined()) {
    this->denom_.slice(0, old_size, new_size).copy_(transfer_data.denom);
  }

  if (transfer_data.max_radii2D.defined()) {
    this->max_radii2D_.slice(0, old_size, new_size)
        .copy_(transfer_data.max_radii2D);
  }

  // Handle optimizer states
  if (this->optimizer_ && !transfer_data.exp_avg_states.empty()) {
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      auto& param = param_groups[group_idx].params()[0];
      auto key = param.unsafeGetTensorImpl();

      if (state.find(key) != state.end() &&
          transfer_data.exp_avg_states[group_idx].defined()) {
        auto& param_state =
            static_cast<torch::optim::AdamParamState&>(*state[key]);

        // Get the shape for the new portion of the parameter
        auto param_slice = param.slice(0, old_size, new_size);

        // Replace the newly added optimizer states with transferred ones
        param_state.exp_avg()
            .slice(0, old_size, new_size)
            .copy_(transfer_data.exp_avg_states[group_idx]);
        param_state.exp_avg_sq()
            .slice(0, old_size, new_size)
            .copy_(transfer_data.exp_avg_sq_states[group_idx]);

        // For step, we take the maximum of existing and transferred
        // (since step is per-parameter, not per-primitive)
        if (transfer_data.step_states[group_idx].defined() &&
            transfer_data.step_states[group_idx].numel() > 0) {
          int64_t transferred_step =
              transfer_data.step_states[group_idx][0].item<int64_t>();
          int64_t current_step = param_state.step();
          param_state.step(std::max(current_step, transferred_step));
        }
      }
    }
  }
}

// Additional method needed in GaussianModel class
void GaussianModel::initializeFromTransferData(
    const GaussianTransferData& transfer_data,
    const GaussianOptimizationParams& training_args,
    const float spatial_lr_scale) {
  // Initialize tensors from transfer data
  this->xyz_ = transfer_data.points.clone().requires_grad_();
  this->features_dc_ = transfer_data.features_dc.clone().requires_grad_();
  this->features_rest_ = transfer_data.features_rest.clone().requires_grad_();
  this->opacity_ = transfer_data.opacities.clone().requires_grad_();
  this->scaling_ = transfer_data.scaling.clone().requires_grad_();
  this->rotation_ = transfer_data.rotation.clone().requires_grad_();
  this->exist_since_iter_ = transfer_data.exist_since.clone();

  // Initialize auxiliary states from transfer data
  this->position_lrs_ = transfer_data.position_lrs.clone();
  this->xyz_gradient_accum_ = transfer_data.xyz_gradient_accum.clone();
  this->denom_ = transfer_data.denom.clone();
  this->max_radii2D_ = transfer_data.max_radii2D.clone();

  // Initialize tensor vectors
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Setup optimizer with preserved learning rates
  setPercentDense(training_args.percent_dense_);

  position_lr_init_ = training_args.position_lr_init_ * spatial_lr_scale;
  position_lr_decay_ = training_args.position_lr_decay_;
  position_lr_min_ = position_lr_init_ * 0.1f * spatial_lr_scale;

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(0.0f);

  // Add other parameter groups
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

  // Restore optimizer states from transfer data
  if (!transfer_data.exp_avg_states.empty()) {
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      auto& param = param_groups[group_idx].params()[0];
      auto key = param.unsafeGetTensorImpl();

      if (transfer_data.exp_avg_states[group_idx].defined()) {
        auto new_state = std::make_unique<torch::optim::AdamParamState>();

        // Use the step value from transfer data (take max if multiple)
        int64_t step_value = 0;
        if (transfer_data.step_states[group_idx].defined() &&
            transfer_data.step_states[group_idx].numel() > 0) {
          step_value =
              transfer_data.step_states[group_idx].max().item<int64_t>();
        }
        new_state->step(step_value);

        new_state->exp_avg(transfer_data.exp_avg_states[group_idx].clone());
        new_state->exp_avg_sq(
            transfer_data.exp_avg_sq_states[group_idx].clone());

        state[key] = std::move(new_state);
      } else {
        // Create zero state if no transfer data available
        auto new_state = std::make_unique<torch::optim::AdamParamState>();
        new_state->step(0);
        new_state->exp_avg(torch::zeros_like(param));
        new_state->exp_avg_sq(torch::zeros_like(param));
        state[key] = std::move(new_state);
      }
    }
  }
}

void GaussianModel::saveTensorBinary(const torch::Tensor& tensor,
                                     std::ofstream& file) {
  TensorHeader header = {};
  header.dims = tensor.dim();

  for (int i = 0; i < tensor.dim(); ++i) {
    header.sizes[i] = static_cast<uint32_t>(tensor.size(i));
  }
  header.dtype = static_cast<uint32_t>(tensor.scalar_type());
  header.data_size = tensor.nbytes();

  // Write header
  file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Move tensor to CPU if needed and write data
  torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
  file.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr()),
             header.data_size);
}

torch::Tensor GaussianModel::loadTensorBinary(std::ifstream& file) {
  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));

  // Reconstruct tensor sizes
  std::vector<int64_t> sizes(header.dims);
  for (uint32_t i = 0; i < header.dims; ++i) {
    sizes[i] = header.sizes[i];
  }

  // Create tensor with correct type and device
  torch::TensorOptions options =
      torch::TensorOptions()
          .dtype(static_cast<torch::ScalarType>(header.dtype))
          .device(device_type_);

  torch::Tensor tensor = torch::empty(sizes, options.device(torch::kCPU));

  // Read data
  file.read(reinterpret_cast<char*>(tensor.data_ptr()), header.data_size);

  // Move to target device if needed
  return tensor.to(device_type_);
}

void GaussianModel::save_checkpoint_fast(const std::string& path) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + path);
  }

  // Write a simple magic number and version for validation
  uint32_t magic = 0x47415553;  // "GAUS" in hex
  uint32_t version = 2;         // Incremented for optimizer state support
  file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));

  // Validate model parameters before saving
  assert(sh_degree_ <= 3);
  assert(position_lr_init_ >= 0);
  assert(position_lr_decay_ > 0);
  assert(position_lr_min_ >= 0);
  assert(local_iteration_ >= 0);
  assert(percent_dense_ > 0 && percent_dense_ < 1);

  // Write model metadata
  file.write(reinterpret_cast<const char*>(&sh_degree_), sizeof(sh_degree_));
  file.write(reinterpret_cast<const char*>(&local_iteration_),
             sizeof(local_iteration_));
  file.write(reinterpret_cast<const char*>(&percent_dense_),
             sizeof(percent_dense_));
  file.write(reinterpret_cast<const char*>(&spatial_lr_scale_),
             sizeof(spatial_lr_scale_));
  file.write(reinterpret_cast<const char*>(&position_lr_init_),
             sizeof(position_lr_init_));
  file.write(reinterpret_cast<const char*>(&position_lr_decay_),
             sizeof(position_lr_decay_));
  file.write(reinterpret_cast<const char*>(&position_lr_min_),
             sizeof(position_lr_min_));

  // Prepare optimizer header
  OptimizerHeader opt_header = {};
  opt_header.has_optimizer_state = (optimizer_ != nullptr) ? 1 : 0;

  if (optimizer_) {
    opt_header.num_param_groups =
        static_cast<uint32_t>(optimizer_->param_groups().size());

    // Save learning rates with validation
    for (size_t i = 0; i < optimizer_->param_groups().size() && i < 6; ++i) {
      float lr = optimizer_->param_groups()[i].options().get_lr();

      // Safety check for NaN or inf learning rates
      if (std::isnan(lr) || std::isinf(lr) || lr < 0.0f || lr > 1.0f) {
        std::cerr << "ERROR: Found invalid learning rate (" << lr
                  << ") for group " << i << std::endl;
        throw std::runtime_error("Invalid LR to save");
      }

      opt_header.learning_rates[i] = lr;
    }

    // Count parameters per group
    for (size_t i = 0; i < optimizer_->param_groups().size() && i < 6; ++i) {
      opt_header.param_counts[i] =
          static_cast<uint32_t>(optimizer_->param_groups()[i].params().size());
    }

    // Calculate total state data size (will be filled later)
    opt_header.state_data_size = 0;
  }

  // Write optimizer header
  file.write(reinterpret_cast<const char*>(&opt_header), sizeof(opt_header));

  // Save main tensors in order
  saveTensorBinary(xyz_, file);
  saveTensorBinary(features_dc_, file);
  saveTensorBinary(features_rest_, file);
  saveTensorBinary(scaling_, file);
  saveTensorBinary(rotation_, file);
  saveTensorBinary(opacity_, file);
  saveTensorBinary(max_radii2D_, file);
  saveTensorBinary(xyz_gradient_accum_, file);
  saveTensorBinary(denom_, file);
  saveTensorBinary(exist_since_iter_, file);
  saveTensorBinary(position_lrs_, file);

  // Save optimizer state if available
  if (optimizer_) {
    auto& state = optimizer_->state();
    int param_idx = 0;

    for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
         ++group_idx) {
      auto& group = optimizer_->param_groups()[group_idx];

      for (size_t j = 0; j < group.params().size(); ++j) {
        auto& param = group.params()[j];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end()) {
          auto& param_state =
              static_cast<torch::optim::AdamParamState&>(*state[key]);

          // Save step count
          int64_t step = param_state.step();
          file.write(reinterpret_cast<const char*>(&step), sizeof(step));

          // Save momentum tensors
          saveTensorBinary(param_state.exp_avg(), file);
          saveTensorBinary(param_state.exp_avg_sq(), file);
        } else {
          // Save placeholder for parameters without state
          int64_t step = 0;
          file.write(reinterpret_cast<const char*>(&step), sizeof(step));
          saveTensorBinary(torch::zeros_like(param), file);
          saveTensorBinary(torch::zeros_like(param), file);
        }

        param_idx++;
      }
    }
  }

  file.close();
}

void GaussianModel::load_checkpoint_fast(
    const std::string& path,
    const GaussianOptimizationParams& training_args,
    bool load_auxiliary_tensors,
    bool load_optimizer_state,
    bool load_existence_info,
    bool normalize_quaternions) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + path);
  }

  // Validate magic number and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));

  if (magic != 0x47415553) {
    throw std::runtime_error("Invalid checkpoint file format");
  }

  bool has_optimizer_data = (version >= 2);

  // Read model metadata
  file.read(reinterpret_cast<char*>(&sh_degree_), sizeof(sh_degree_));
  file.read(reinterpret_cast<char*>(&local_iteration_),
            sizeof(local_iteration_));
  file.read(reinterpret_cast<char*>(&percent_dense_), sizeof(percent_dense_));
  file.read(reinterpret_cast<char*>(&spatial_lr_scale_),
            sizeof(spatial_lr_scale_));
  file.read(reinterpret_cast<char*>(&position_lr_init_),
            sizeof(position_lr_init_));
  file.read(reinterpret_cast<char*>(&position_lr_decay_),
            sizeof(position_lr_decay_));
  file.read(reinterpret_cast<char*>(&position_lr_min_),
            sizeof(position_lr_min_));

  // Validate loaded parameters
  assert(sh_degree_ <= 3);
  assert(local_iteration_ >= 0);
  assert(percent_dense_ > 0 && percent_dense_ < 1);
  assert(position_lr_init_ > 0);
  assert(position_lr_decay_ > 0);
  assert(position_lr_min_ > 0);

  // Read optimizer header if available
  OptimizerHeader opt_header = {};
  std::vector<float> learning_rates;
  std::vector<float> default_learning_rates = {
      0.0f,
      training_args.feature_lr_,
      training_args.feature_lr_ / 20.0f,
      training_args.opacity_lr_,
      training_args.scaling_lr_,
      training_args.rotation_lr_};

  if (has_optimizer_data) {
    file.read(reinterpret_cast<char*>(&opt_header), sizeof(opt_header));

    // Load learning rates with validation
    for (int i = 0; i < 6; ++i) {
      float lr = default_learning_rates[i];

      if (i < opt_header.num_param_groups) {
        float loaded_lr = opt_header.learning_rates[i];

        // Validate the loaded learning rate
        if (!std::isnan(loaded_lr) && !std::isinf(loaded_lr) &&
            loaded_lr >= 0.0f && loaded_lr < 1.0f) {
          lr = loaded_lr;
        } else {
          std::cerr << "WARNING: Invalid learning rate (" << loaded_lr
                    << ") for group " << i << ". Using default "
                    << default_learning_rates[i] << " instead." << std::endl;
        }
      }

      learning_rates.push_back(lr);
    }
  } else {
    learning_rates = default_learning_rates;
  }

  // Load main tensors in same order as saved
  xyz_ = loadTensorBinary(file);
  features_dc_ = loadTensorBinary(file);
  features_rest_ = loadTensorBinary(file);
  scaling_ = loadTensorBinary(file);
  rotation_ = loadTensorBinary(file);
  opacity_ = loadTensorBinary(file);

  if (load_auxiliary_tensors) {
    max_radii2D_ = loadTensorBinary(file);
    xyz_gradient_accum_ = loadTensorBinary(file);
    denom_ = loadTensorBinary(file);
  } else {
    // Skip these tensors by reading headers and seeking past data
    for (int i = 0; i < 3; ++i) {
      TensorHeader header;
      file.read(reinterpret_cast<char*>(&header), sizeof(header));
      file.seekg(header.data_size, std::ios::cur);
    }
    // Initialize with empty tensors
    max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type_));
    xyz_gradient_accum_ =
        torch::empty(0, torch::TensorOptions().device(device_type_));
    denom_ = torch::empty(0, torch::TensorOptions().device(device_type_));
  }

  if (load_existence_info) {
    exist_since_iter_ = loadTensorBinary(file);
  } else {
    // Skip tensor
    TensorHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.seekg(header.data_size, std::ios::cur);
    exist_since_iter_ =
        torch::empty(0, torch::TensorOptions().device(device_type_));
  }

  // Always load position_lrs for optimizer setup
  position_lrs_ = loadTensorBinary(file);

  // Normalize quaternions if requested
  if (normalize_quaternions) {
    rotation_ = torch::nn::functional::normalize(
        rotation_, torch::nn::functional::NormalizeFuncOptions().dim(1));
  }

  // Ensure all tensors require gradients
  xyz_ = xyz_.requires_grad_(true);
  features_dc_ = features_dc_.contiguous().requires_grad_(true);
  features_rest_ = features_rest_.contiguous().requires_grad_(true);
  scaling_ = scaling_.requires_grad_(true);
  rotation_ = rotation_.requires_grad_(true);
  opacity_ = opacity_.requires_grad_(true);

  // Update tensor vectors
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Setup optimizer with proper learning rates
  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(learning_rates[0]);
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(0.0f);

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

  // Verify optimizer learning rates
  for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
    float actual_lr = optimizer_->param_groups()[i].options().get_lr();

    // Double-check for NaN (this should never happen with our validation)
    if (std::isnan(actual_lr)) {
      std::cerr
          << "CRITICAL: NaN learning rate detected after setting! Fixing..."
          << std::endl;
      optimizer_->param_groups()[i].options().set_lr(default_learning_rates[i]);
    }
  }

  // Load optimizer state if requested and available
  if (load_optimizer_state && has_optimizer_data &&
      opt_header.has_optimizer_state) {
    try {
      auto& state = optimizer_->state();
      int param_idx = 0;

      for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
           ++group_idx) {
        auto& group = optimizer_->param_groups()[group_idx];

        // Verify parameter count matches
        if (group_idx < 6 &&
            group.params().size() != opt_header.param_counts[group_idx]) {
          std::cerr << "Warning: Group " << group_idx
                    << " size mismatch: saved="
                    << opt_header.param_counts[group_idx]
                    << ", current=" << group.params().size() << std::endl;
          throw std::runtime_error("Parameter count mismatch in group");
        }

        for (size_t j = 0; j < group.params().size(); ++j) {
          auto& param = group.params()[j];
          auto key = param.unsafeGetTensorImpl();

          // Load step count
          int64_t step;
          file.read(reinterpret_cast<char*>(&step), sizeof(step));

          // Load momentum tensors
          torch::Tensor exp_avg = loadTensorBinary(file);
          torch::Tensor exp_avg_sq = loadTensorBinary(file);

          // Verify tensor shapes match current parameter
          bool sizes_match = true;
          if (param.dim() != exp_avg.dim() || param.dim() != exp_avg_sq.dim()) {
            sizes_match = false;
          } else {
            for (int d = 0; d < param.dim(); ++d) {
              if (param.size(d) != exp_avg.size(d) ||
                  param.size(d) != exp_avg_sq.size(d)) {
                sizes_match = false;
                break;
              }
            }
          }

          if (!sizes_match) {
            std::cerr << "Warning: Parameter shape mismatch for param "
                      << param_idx << std::endl;
            // Create zero state instead of using mismatched tensors
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(0);
            new_state->exp_avg(torch::zeros_like(param));
            new_state->exp_avg_sq(torch::zeros_like(param));
            state[key] = std::move(new_state);
          } else {
            // Create state with loaded values
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(step);
            new_state->exp_avg(exp_avg.to(param.device()));
            new_state->exp_avg_sq(exp_avg_sq.to(param.device()));

            state[key] = std::move(new_state);
          }

          param_idx++;
        }
      }

      // Verify loaded state
      for (auto& param_group : optimizer_->param_groups()) {
        for (auto& param : param_group.params()) {
          auto key = param.unsafeGetTensorImpl();
          if (state.find(key) == state.end()) {
            throw std::runtime_error(
                "Error: Parameter missing from state after loading");
          }
        }
      }

    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to load optimizer state: " << e.what()
                << std::endl;
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
    // Initialize fresh optimizer state if no saved state available
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

  file.close();

  // Clear CUDA cache if on GPU
  if (device_type_ == torch::kCUDA) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
}

void GaussianModel::save_checkpoint_mmap(const std::string& path) {
  auto start_time = std::chrono::high_resolution_clock::now();
  torch::NoGradGuard no_grad;

  auto num_points = getXYZ().size(0);
  auto n_features_rest = features_rest_.size(1);

  CompleteMMapHeader header = {};
  header.num_points = static_cast<uint32_t>(num_points);
  header.sh_degree = static_cast<uint32_t>(sh_degree_);
  header.spatial_lr_scale = spatial_lr_scale_;
  header.position_lr_init = position_lr_init_;
  header.position_lr_decay = position_lr_decay_;
  header.position_lr_min = position_lr_min_;
  header.percent_dense = percent_dense_;
  header.local_iteration = local_iteration_;

  // Set tensor shapes
  header.xyz_size[0] = num_points;
  header.xyz_size[1] = 3;
  header.features_dc_size[0] = num_points;
  header.features_dc_size[1] = features_dc_.size(1);
  header.features_dc_size[2] = features_dc_.size(2);
  header.features_rest_size[0] = num_points;
  header.features_rest_size[1] = n_features_rest;
  header.features_rest_size[2] = features_rest_.size(2);
  header.scaling_size[0] = num_points;
  header.scaling_size[1] = 3;
  header.rotation_size[0] = num_points;
  header.rotation_size[1] = 4;
  header.opacity_size[0] = num_points;
  header.opacity_size[1] = 1;
  header.max_radii2D_size[0] = num_points;
  header.xyz_gradient_accum_size[0] = num_points;
  header.xyz_gradient_accum_size[1] = 1;
  header.denom_size[0] = num_points;
  header.denom_size[1] = 1;
  header.exist_since_iter_size[0] = num_points;
  header.position_lrs_size[0] = num_points;

  // Calculate offsets for main tensors
  uint64_t offset = sizeof(CompleteMMapHeader);

  header.xyz_offset = offset;
  offset += xyz_.nbytes();

  header.features_dc_offset = offset;
  offset += features_dc_.nbytes();

  header.features_rest_offset = offset;
  offset += features_rest_.nbytes();

  header.scaling_offset = offset;
  offset += scaling_.nbytes();

  header.rotation_offset = offset;
  offset += rotation_.nbytes();

  header.opacity_offset = offset;
  offset += opacity_.nbytes();

  header.max_radii2D_offset = offset;
  offset += max_radii2D_.nbytes();

  header.xyz_gradient_accum_offset = offset;
  offset += xyz_gradient_accum_.nbytes();

  header.denom_offset = offset;
  offset += denom_.nbytes();

  header.exist_since_iter_offset = offset;
  offset += exist_since_iter_.nbytes();

  header.position_lrs_offset = offset;
  offset += position_lrs_.nbytes();

  // Setup optimizer state information
  header.optimizer_state_offset = offset;
  header.has_optimizer_state = (optimizer_ != nullptr) ? 1 : 0;

  OptimizerStateLayout opt_layout;
  uint64_t optimizer_state_size = 0;

  if (optimizer_) {
    header.num_param_groups =
        static_cast<uint32_t>(optimizer_->param_groups().size());

    // Save learning rates
    for (size_t i = 0; i < optimizer_->param_groups().size() && i < 6; ++i) {
      float lr = optimizer_->param_groups()[i].options().get_lr();
      if (std::isnan(lr) || std::isinf(lr) || lr < 0.0f || lr > 1.0f) {
        throw std::runtime_error("Invalid learning rate detected during save");
      }
      header.learning_rates[i] = lr;
      header.param_counts[i] =
          static_cast<uint32_t>(optimizer_->param_groups()[i].params().size());
    }

    // Calculate optimizer state layout
    auto& state = optimizer_->state();

    // Step data offset (one int64_t per parameter group)
    header.step_data_offset = offset;
    offset += header.num_param_groups * sizeof(int64_t);

    // Calculate offsets for each parameter group's optimizer tensors
    for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
         ++group_idx) {
      auto& group = optimizer_->param_groups()[group_idx];

      if (!group.params().empty()) {
        auto& param = group.params()[0];  // Assume one param per group

        // Store parameter shape for validation during loading
        auto param_shape = param.sizes().vec();
        opt_layout.param_shapes.push_back(param_shape);

        // exp_avg offset
        header.exp_avg_offsets[group_idx] = offset;
        offset += param.nbytes();

        // exp_avg_sq offset
        header.exp_avg_sq_offsets[group_idx] = offset;
        offset += param.nbytes();
      }
    }

    optimizer_state_size = offset - header.optimizer_state_offset;
  }

  header.optimizer_state_size = optimizer_state_size;
  header.total_file_size = offset;

  // Create file and memory map
  int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  if (fd == -1) {
    throw std::runtime_error("Failed to create file: " + path);
  }

  if (ftruncate(fd, header.total_file_size) == -1) {
    close(fd);
    throw std::runtime_error("Failed to set file size");
  }

  void* mapped = mmap(nullptr, header.total_file_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to memory map file");
  }

  // Write header
  std::memcpy(mapped, &header, sizeof(header));

  // Helper to write tensor data
  auto write_tensor = [&](const torch::Tensor& tensor, uint64_t offset) {
    auto cpu_tensor =
        tensor.is_cuda() ? tensor.detach().cpu() : tensor.detach();
    std::memcpy(static_cast<char*>(mapped) + offset, cpu_tensor.data_ptr(),
                cpu_tensor.nbytes());
  };

  // Write main tensor data
  write_tensor(xyz_, header.xyz_offset);
  write_tensor(features_dc_, header.features_dc_offset);
  write_tensor(features_rest_, header.features_rest_offset);
  write_tensor(scaling_, header.scaling_offset);
  write_tensor(rotation_, header.rotation_offset);
  write_tensor(opacity_, header.opacity_offset);
  write_tensor(max_radii2D_, header.max_radii2D_offset);
  write_tensor(xyz_gradient_accum_, header.xyz_gradient_accum_offset);
  write_tensor(denom_, header.denom_offset);
  write_tensor(exist_since_iter_, header.exist_since_iter_offset);
  write_tensor(position_lrs_, header.position_lrs_offset);

  // Write optimizer state if available
  if (optimizer_) {
    auto& state = optimizer_->state();

    // Write step counts
    std::vector<int64_t> step_counts(header.num_param_groups, 0);
    for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
         ++group_idx) {
      auto& group = optimizer_->param_groups()[group_idx];
      if (!group.params().empty()) {
        auto& param = group.params()[0];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end()) {
          auto& param_state =
              static_cast<torch::optim::AdamParamState&>(*state[key]);
          step_counts[group_idx] = param_state.step();
        }
      }
    }

    std::memcpy(static_cast<char*>(mapped) + header.step_data_offset,
                step_counts.data(), step_counts.size() * sizeof(int64_t));

    // Write exp_avg and exp_avg_sq tensors for each group
    for (size_t group_idx = 0; group_idx < optimizer_->param_groups().size();
         ++group_idx) {
      auto& group = optimizer_->param_groups()[group_idx];

      if (!group.params().empty()) {
        auto& param = group.params()[0];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end()) {
          auto& param_state =
              static_cast<torch::optim::AdamParamState&>(*state[key]);

          // Write exp_avg
          write_tensor(param_state.exp_avg(),
                       header.exp_avg_offsets[group_idx]);

          // Write exp_avg_sq
          write_tensor(param_state.exp_avg_sq(),
                       header.exp_avg_sq_offsets[group_idx]);
        } else {
          // Write zero tensors for missing state
          auto zero_tensor = torch::zeros_like(param);
          write_tensor(zero_tensor, header.exp_avg_offsets[group_idx]);
          write_tensor(zero_tensor, header.exp_avg_sq_offsets[group_idx]);
        }
      }
    }
  }

  // Force write to disk
  msync(mapped, header.total_file_size, MS_SYNC);

  // Cleanup
  munmap(mapped, header.total_file_size);
  close(fd);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Complete memory-mapped save with optimizer state completed in "
            << duration.count() << "ms" << std::endl;
}

void GaussianModel::load_checkpoint_mmap(
    const std::string& path,
    const GaussianOptimizationParams& training_args,
    bool load_auxiliary_tensors,
    bool load_optimizer_state,
    bool load_existence_info,
    bool normalize_quaternions) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Open and map file
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error("Failed to get file size");
  }

  void* mapped = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to memory map file");
  }

  // Read and validate header
  const CompleteMMapHeader* header =
      static_cast<const CompleteMMapHeader*>(mapped);
  if (header->magic != 0x474D4150 || header->version != 3) {
    munmap(mapped, sb.st_size);
    close(fd);
    throw std::runtime_error("Invalid file format or version");
  }

  // Validate file size
  if (sb.st_size < header->total_file_size) {
    munmap(mapped, sb.st_size);
    close(fd);
    throw std::runtime_error("File size mismatch - file may be corrupted");
  }

  // Load model parameters
  sh_degree_ = header->sh_degree;
  spatial_lr_scale_ = header->spatial_lr_scale;
  position_lr_init_ = header->position_lr_init;
  position_lr_decay_ = header->position_lr_decay;
  position_lr_min_ = header->position_lr_min;
  percent_dense_ = header->percent_dense;
  local_iteration_ = header->local_iteration;

  // Validate loaded parameters
  if (sh_degree_ > 3 || position_lr_init_ <= 0 || position_lr_decay_ <= 0 ||
      position_lr_min_ <= 0 || percent_dense_ <= 0 || percent_dense_ >= 1) {
    munmap(mapped, sb.st_size);
    close(fd);
    throw std::runtime_error("Invalid model parameters in checkpoint");
  }

  // Helper to create tensor from memory-mapped data
  auto create_tensor = [&](uint64_t offset, const uint64_t* shape, int dims,
                           torch::ScalarType dtype) -> torch::Tensor {
    // Validate offset
    if (offset >= sb.st_size) {
      throw std::runtime_error("Invalid tensor offset in checkpoint");
    }

    std::vector<int64_t> sizes(dims);
    size_t total_elements = 1;
    for (int i = 0; i < dims; ++i) {
      sizes[i] = static_cast<int64_t>(shape[i]);
      total_elements *= sizes[i];
    }

    // Validate tensor size doesn't exceed file bounds
    size_t tensor_bytes = total_elements * torch::elementSize(dtype);
    if (offset + tensor_bytes > sb.st_size) {
      throw std::runtime_error("Tensor data exceeds file bounds");
    }

    const void* data_ptr = static_cast<const char*>(mapped) + offset;

    if (device_type_ == torch::kCUDA) {
      torch::Tensor cpu_tensor = torch::from_blob(
          const_cast<void*>(data_ptr), sizes,
          torch::TensorOptions().dtype(dtype).device(torch::kCPU));
      return cpu_tensor.to(device_type_, /*non_blocking=*/false).clone();
    } else {
      torch::Tensor result = torch::from_blob(
          const_cast<void*>(data_ptr), sizes,
          torch::TensorOptions().dtype(dtype).device(torch::kCPU));
      return result.clone();
    }
  };

  // Load main tensors
  xyz_ =
      create_tensor(header->xyz_offset, header->xyz_size, 2, torch::kFloat32);
  features_dc_ = create_tensor(header->features_dc_offset,
                               header->features_dc_size, 3, torch::kFloat32);
  features_rest_ =
      create_tensor(header->features_rest_offset, header->features_rest_size, 3,
                    torch::kFloat32);
  scaling_ = create_tensor(header->scaling_offset, header->scaling_size, 2,
                           torch::kFloat32);
  rotation_ = create_tensor(header->rotation_offset, header->rotation_size, 2,
                            torch::kFloat32);
  opacity_ = create_tensor(header->opacity_offset, header->opacity_size, 2,
                           torch::kFloat32);

  // Load auxiliary tensors conditionally
  if (load_auxiliary_tensors) {
    max_radii2D_ = create_tensor(header->max_radii2D_offset,
                                 header->max_radii2D_size, 1, torch::kFloat32);
    xyz_gradient_accum_ =
        create_tensor(header->xyz_gradient_accum_offset,
                      header->xyz_gradient_accum_size, 2, torch::kFloat32);
    denom_ = create_tensor(header->denom_offset, header->denom_size, 2,
                           torch::kFloat32);
  } else {
    max_radii2D_ = torch::zeros({header->num_points},
                                torch::TensorOptions().device(device_type_));
    xyz_gradient_accum_ = torch::zeros(
        {header->num_points, 1}, torch::TensorOptions().device(device_type_));
    denom_ = torch::zeros({header->num_points, 1},
                          torch::TensorOptions().device(device_type_));
  }

  if (load_existence_info) {
    exist_since_iter_ =
        create_tensor(header->exist_since_iter_offset,
                      header->exist_since_iter_size, 1, torch::kInt32);
  } else {
    exist_since_iter_ = torch::zeros(
        {header->num_points},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
  }

  position_lrs_ = create_tensor(header->position_lrs_offset,
                                header->position_lrs_size, 1, torch::kFloat32);

  // Normalize quaternions if requested
  if (normalize_quaternions) {
    rotation_ = torch::nn::functional::normalize(
        rotation_, torch::nn::functional::NormalizeFuncOptions().dim(1));
  }

  // Set requires_grad
  xyz_ = xyz_.requires_grad_(true);
  features_dc_ = features_dc_.contiguous().requires_grad_(true);
  features_rest_ = features_rest_.contiguous().requires_grad_(true);
  scaling_ = scaling_.requires_grad_(true);
  rotation_ = rotation_.requires_grad_(true);
  opacity_ = opacity_.requires_grad_(true);

  // Update tensor vectors
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Setup optimizer with proper learning rates
  std::vector<float> learning_rates;
  std::vector<float> default_learning_rates = {
      0.0f,
      training_args.feature_lr_,
      training_args.feature_lr_ / 20.0f,
      training_args.opacity_lr_,
      training_args.scaling_lr_,
      training_args.rotation_lr_};

  // Use saved learning rates if available, otherwise use defaults
  bool has_valid_optimizer_data =
      (header->has_optimizer_state && header->num_param_groups <= 6);

  for (int i = 0; i < 6; ++i) {
    float lr = default_learning_rates[i];

    if (has_valid_optimizer_data && i < header->num_param_groups) {
      float loaded_lr = header->learning_rates[i];
      if (!std::isnan(loaded_lr) && !std::isinf(loaded_lr) &&
          loaded_lr >= 0.0f && loaded_lr < 1.0f) {
        lr = loaded_lr;
      } else {
        std::cerr << "WARNING: Invalid learning rate (" << loaded_lr
                  << ") for group " << i << ". Using default." << std::endl;
      }
    }
    learning_rates.push_back(lr);
  }

  // Initialize optimizer
  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0f);
  adam_options.eps() = 1e-15;

  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(0.0f);

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

  // Load optimizer state if requested and available
  if (load_optimizer_state && has_valid_optimizer_data) {
    try {
      auto& state = optimizer_->state();

      // Load step counts
      const int64_t* step_data = reinterpret_cast<const int64_t*>(
          static_cast<const char*>(mapped) + header->step_data_offset);

      // Load optimizer state for each parameter group
      for (size_t group_idx = 0;
           group_idx < optimizer_->param_groups().size() &&
           group_idx < header->num_param_groups;
           ++group_idx) {
        auto& group = optimizer_->param_groups()[group_idx];

        if (!group.params().empty()) {
          auto& param = group.params()[0];
          auto key = param.unsafeGetTensorImpl();

          // Validate that optimizer state data exists for this group
          if (header->exp_avg_offsets[group_idx] >= sb.st_size ||
              header->exp_avg_sq_offsets[group_idx] >= sb.st_size) {
            std::cerr << "WARNING: Invalid optimizer state offset for group "
                      << group_idx << ". Skipping." << std::endl;
            continue;
          }

          // Create tensors for optimizer state
          auto param_shape = param.sizes().vec();

          torch::Tensor exp_avg = create_tensor(
              header->exp_avg_offsets[group_idx],
              reinterpret_cast<const uint64_t*>(param_shape.data()),
              param_shape.size(), torch::kFloat32);

          torch::Tensor exp_avg_sq = create_tensor(
              header->exp_avg_sq_offsets[group_idx],
              reinterpret_cast<const uint64_t*>(param_shape.data()),
              param_shape.size(), torch::kFloat32);

          // Validate tensor shapes match
          bool shapes_match =
              (param.dim() == exp_avg.dim() && param.dim() == exp_avg_sq.dim());
          if (shapes_match) {
            for (int d = 0; d < param.dim(); ++d) {
              if (param.size(d) != exp_avg.size(d) ||
                  param.size(d) != exp_avg_sq.size(d)) {
                shapes_match = false;
                break;
              }
            }
          }

          if (shapes_match) {
            // Create and store optimizer state
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(step_data[group_idx]);
            new_state->exp_avg(exp_avg.to(param.device()));
            new_state->exp_avg_sq(exp_avg_sq.to(param.device()));
            state[key] = std::move(new_state);
          } else {
            std::cerr << "WARNING: Optimizer state shape mismatch for group "
                      << group_idx << ". Creating zero state." << std::endl;

            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(0);
            new_state->exp_avg(torch::zeros_like(param));
            new_state->exp_avg_sq(torch::zeros_like(param));
            state[key] = std::move(new_state);
          }
        }
      }

      std::cout << "Successfully loaded optimizer state for "
                << std::min(optimizer_->param_groups().size(),
                            (size_t)header->num_param_groups)
                << " parameter groups." << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "WARNING: Failed to load optimizer state: " << e.what()
                << std::endl;
      std::cerr << "Initializing fresh optimizer state." << std::endl;

      // Fall back to fresh state
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
  } else {
    // Initialize fresh optimizer state
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

  // Cleanup
  munmap(mapped, sb.st_size);
  close(fd);

  // Clear CUDA cache
  if (device_type_ == torch::kCUDA) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Complete memory-mapped load with optimizer state completed in "
            << duration.count() << "ms" << std::endl;
}