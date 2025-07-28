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

GaussianModel::GaussianModel(const GaussianModelParams& model_params,
                             std::string storage_base_path)
    : storage_base_path_(storage_base_path),
      sh_degree_(0),
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

  // int n_elements = 3;
  // std::cout << "First " << n_elements << " exist_since_iter_ elements: ";
  // for (int i = 0; i < std::min(n_elements, (int)exist_since_iter_.size(0));
  //      i++) {
  //   std::cout << exist_since_iter_[i].item<float>() << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "Creation iter: " << kf_creation_iter << std::endl;

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  // std::cout << "[DEBUG] Points unstable mask true count: "
  //           << point_unstable_flags.sum().item<int>() << std::endl;

  // std::cout << "[DEBUG-STPV] Created unstable flags" << std::endl;

  // std::cout << "[DEBUG-STPV] Calling transform function" << std::endl;

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
  std::cout << "Calling trainingSetup" << std::endl;
  std::cout << "XYZ tensor sizes: " << getXYZ().sizes() << std::endl;
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
  this->gaussian_chunk_ids_ =
      this->gaussian_chunk_ids_.index({valid_points_mask});

  updateChunksInMemory();
}

void GaussianModel::densificationPostfix(torch::Tensor& new_xyz,
                                         torch::Tensor& new_features_dc,
                                         torch::Tensor& new_features_rest,
                                         torch::Tensor& new_opacities,
                                         torch::Tensor& new_scaling,
                                         torch::Tensor& new_rotation,
                                         torch::Tensor& new_exist_since_iter,
                                         torch::Tensor& new_position_lrs) {
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
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  torch::Tensor new_chunk_ids = computeChunkIds(new_xyz);
  gaussian_chunk_ids_ =
      torch::cat({gaussian_chunk_ids_, new_chunk_ids}, /*dim=*/0);

  updateChunksInMemory();
}

std::vector<ChunkCoord> GaussianModel::frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  if (!keyframe) {
    return {};
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  // Check cache (keep original cache logic)
  if (use_cache) {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto cache_it = visibility_cache_.find(keyframe_id);
    if (cache_it != visibility_cache_.end()) {
      auto& entry = cache_it->second;
      if ((now - entry.timestamp) < cache_expiry_time_ &&
          pose_nearly_equal(current_pose, entry.pose)) {
        entry.timestamp = now;
        return entry.visible_chunks;
      }
    }
  }

  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);
  torch::Tensor tensor_matrix = keyframe->projection_matrix_;

  // Ensure tensor is on CPU and contiguous
  tensor_matrix = tensor_matrix.cpu().contiguous();

  // Get data pointer and create Eigen matrix
  float* data_ptr = tensor_matrix.data_ptr<float>();
  Eigen::Matrix4f proj_matrix = Eigen::Map<Eigen::Matrix4f>(data_ptr);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Get camera position for chunk search
  Sophus::SE3d Twc = current_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position, chunk_size_);

  // Calculate parameters
  int search_radius =
      std::ceil(keyframe->zfar_ / chunk_size_ * std::sqrt(3.0f)) + 2;
  float max_distance = keyframe->zfar_ + chunk_size_ * 1.732f;

  // Call hierarchical culling
  std::vector<ChunkCoord> visible_chunks =
      cullChunksHierarchical(vp_matrix, camera_position, camera_chunk,
                             search_radius, chunk_size_, max_distance);

  // Update cache (keep original cache update logic)
  if (use_cache) {
    std::lock_guard<std::mutex> lock(visibility_cache_mutex_);
    VisibilityCacheEntry entry;
    entry.pose = current_pose;
    entry.visible_chunks = visible_chunks;
    entry.timestamp = std::chrono::steady_clock::now();
    visibility_cache_[keyframe_id] = entry;
  }

  return visible_chunks;
}

torch::Tensor GaussianModel::cullVisibleGaussians(
    std::shared_ptr<GaussianKeyframe> keyframe) {
  std::vector<ChunkCoord> visible_chunks =
      frustumCullChunks(keyframe, /*use_cache=*/true);

  if (visible_chunks.empty()) {
    // Return all-false mask
    return torch::zeros(
        {xyz_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  // STAGE 2: Convert chunk visibility to gaussian visibility
  torch::Tensor gaussian_visibility_mask =
      createGaussianMaskFromChunks(visible_chunks);

  return gaussian_visibility_mask;
}

torch::Tensor GaussianModel::createGaussianMaskFromChunks(
    const std::vector<ChunkCoord>& visible_chunks) {
  // std::cout << "[DEBUG] Creating gaussian mask from " <<
  // visible_chunks.size()
  //           << " visible chunks" << std::endl;

  // Convert to tensor
  std::vector<int64_t> visible_chunk_ids;
  for (const auto& coord : visible_chunks) {
    int64_t encoded = encodeChunkCoord(coord);
    visible_chunk_ids.push_back(encoded);
    // std::cout << "[DEBUG] Chunk (" << coord.x << "," << coord.y << ","
    //           << coord.z << ") -> encoded: " << encoded << std::endl;
  }

  if (visible_chunk_ids.empty()) {
    std::cout << "[DEBUG] No visible chunks, returning all-false mask"
              << std::endl;
    return torch::zeros(
        {gaussian_chunk_ids_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  torch::Tensor visible_chunks_tensor = torch::tensor(
      visible_chunk_ids,
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  // std::cout << "[DEBUG] Total gaussians: " << gaussian_chunk_ids_.size(0)
  //           << std::endl;
  // std::cout << "[DEBUG] Unique chunk IDs in model: "
  //           << std::get<0>(torch::_unique2(gaussian_chunk_ids_)).size(0)
  //           << std::endl;
  // std::cout << "[DEBUG] Min chunk ID in model: "
  //           << gaussian_chunk_ids_.min().item<int64_t>() << std::endl;
  // std::cout << "[DEBUG] Max chunk ID in model: "
  //           << gaussian_chunk_ids_.max().item<int64_t>() << std::endl;

  // VECTORIZED: Use PyTorch's optimized isin function
  torch::Tensor visibility_mask =
      torch::isin(gaussian_chunk_ids_, visible_chunks_tensor);

  // int visible_count = visibility_mask.sum().item<int>();
  // float visible_percentage =
  //     (float)visible_count / gaussian_chunk_ids_.size(0) * 100.0f;

  // std::cout << "[DEBUG] Visible gaussians: " << visible_count << " / "
  //           << gaussian_chunk_ids_.size(0) << " (" << visible_percentage <<
  //           "%)"
  //           << std::endl;

  return visibility_mask;
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

  torch::Tensor full_model_prune_mask = torch::zeros(
      {getXYZ().size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

  full_model_prune_mask.index_put_({visible_indices}, prune_mask);

  prunePoints(full_model_prune_mask);
}

torch::Tensor GaussianModel::computeChunkIds(const torch::Tensor& positions) {
  // Same logic as your existing getChunkCoord, but vectorized
  float half_chunk = chunk_size_ * 0.5f;

  // Vectorized coordinate computation
  torch::Tensor shifted_positions = positions + half_chunk;
  torch::Tensor chunk_coords = torch::floor(shifted_positions / chunk_size_);

  // Convert to int64 for encoding
  chunk_coords = chunk_coords.to(torch::kInt64);

  // Vectorized encoding: convert (x,y,z) coordinates to single integers
  auto x = chunk_coords.index({torch::indexing::Slice(), 0});
  auto y = chunk_coords.index({torch::indexing::Slice(), 1});
  auto z = chunk_coords.index({torch::indexing::Slice(), 2});

  // Encode to single integer per chunk
  const int64_t OFFSET = 10000;
  const int64_t STRIDE = 20000;

  torch::Tensor chunk_ids =
      (x + OFFSET) * STRIDE * STRIDE + (y + OFFSET) * STRIDE + (z + OFFSET);

  return chunk_ids;
}

void GaussianModel::addPoints(const torch::Tensor& new_xyz,
                              const torch::Tensor& new_colors,
                              const torch::Tensor& new_scales,
                              const torch::Tensor& new_opacities,
                              int iteration,
                              float spatial_lr_scale) {
  if (!is_initialized_) {
    // First call - initialize the model
    initializeFromPoints(new_xyz, new_colors, new_scales, new_opacities,
                         iteration, spatial_lr_scale);
  } else {
    // Subsequent calls - append to existing model
    appendPoints(new_xyz, new_colors, new_scales, new_opacities, iteration);
  }
}

void GaussianModel::initializeFromPoints(const torch::Tensor& initial_xyz,
                                         const torch::Tensor& initial_colors,
                                         const torch::Tensor& initial_scales,
                                         const torch::Tensor& initial_opacities,
                                         int iteration,
                                         float spatial_lr_scale) {
  std::cout << "[Gaussian Model] Initializing from points: "
            << initial_xyz.sizes() << std::endl;
  this->spatial_lr_scale_ = spatial_lr_scale;
  int num_points = static_cast<int>(initial_xyz.sizes()[0]);
  torch::Tensor initial_chunk_ids = computeChunkIds(initial_xyz);

  torch::Tensor fused_color = sh_utils::RGB2SH(initial_colors);
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

  torch::Tensor point_cloud_copy = initial_xyz.clone();

  torch::Tensor scales;
  if (initial_scales.defined() && initial_scales.size(0) > 0) {
    // Use provided scales, convert to log space and repeat for 3 dimensions
    scales = initial_scales;
  } else {
    torch::Tensor dist2 =
        torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    scales = torch::log(torch::sqrt(dist2) * 0.1);
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  }
  torch::Tensor rots = torch::zeros(
      {initial_xyz.size(0), 4}, torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor opacities = initial_opacities;

  this->exist_since_iter_ = torch::full(
      {initial_xyz.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  this->xyz_ = initial_xyz.requires_grad_();
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

  gaussian_chunk_ids_ = initial_chunk_ids;

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));

  updateChunksInMemory();

  is_initialized_ = true;
}

void GaussianModel::appendPoints(const torch::Tensor& new_xyzs,
                                 const torch::Tensor& new_colors,
                                 const torch::Tensor& new_scales,
                                 const torch::Tensor& new_opacities,
                                 int iteration) {
  // std::cout << "[Gaussian Model] Appending points: " << new_xyzs.sizes()
  //           << std::endl;
  auto num_new_points = new_xyzs.size(0);
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
        torch::clamp_min(distCUDA2(new_xyzs.clone()), 0.0000001);
    scales = torch::log(torch::sqrt(dist2) * 0.1);
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  }
  torch::Tensor rots = torch::zeros(
      {new_xyzs.size(0), 4}, torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor new_exist_since_iter = torch::full(
      {new_xyzs.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_xyzs;
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

  torch::Tensor new_position_lrs =
      torch::full({num_new_points}, position_lr_init_,
                  torch::TensorOptions().device(device_type_));

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities_tensor, new_scaling, new_rotation,
                       new_exist_since_iter, new_position_lrs);
}

std::string GaussianModel::getChunkFilename(const ChunkCoord& coord) {
  // Using 'p' for positive and 'n' for negative prefixes
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));

  // Add .bin extension for the new binary format
  std::string filename = x_str + "_" + y_str + "_" + z_str + ".bin";

  // Use string concatenation with proper path separator
  return storage_base_path_ + "/" + filename;
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

void GaussianModel::updateChunksInMemory() {
  // Get unique chunk IDs first (GPU operation)
  auto unique_result = torch::_unique2(gaussian_chunk_ids_,
                                       /*sorted=*/false,
                                       /*return_inverse=*/false,
                                       /*return_counts=*/false);
  torch::Tensor unique_chunk_ids = std::get<0>(unique_result);
  auto unique_cpu = unique_chunk_ids.cpu();
  auto accessor = unique_cpu.accessor<int64_t, 1>();

  chunks_in_memory_.clear();  // Fresh start
  for (int i = 0; i < unique_cpu.size(0); ++i) {
    chunks_in_memory_.insert(accessor[i]);
  }
}

void GaussianModel::loadChunks(const std::vector<int64_t>& chunk_ids_to_load) {
  if (chunk_ids_to_load.empty()) return;

  std::vector<GaussianModel::ChunkData> chunks_to_append;
  std::vector<int64_t> loaded_chunk_ids;

  // Load each chunk from disk (can be parallelized)
  for (int64_t chunk_id : chunk_ids_to_load) {
    if (chunks_in_memory_.count(chunk_id)) {
      continue;  // Already in memory
    }

    auto chunk_data = loadSingleChunkFromDisk(chunk_id);
    if (chunk_data.has_value()) {
      chunks_to_append.push_back(chunk_data.value());
      loaded_chunk_ids.push_back(chunk_id);
    }
  }

  if (!chunks_to_append.empty()) {
    // Batch append all loaded chunks at once
    appendLoadedChunks(chunks_to_append, loaded_chunk_ids);

    // Update tracking
    for (int64_t chunk_id : loaded_chunk_ids) {
      chunks_in_memory_.insert(chunk_id);
    }
  }
}

void GaussianModel::saveSingleChunkToDisk(int64_t chunk_id,
                                          const ChunkData& chunk_data) {
  std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

  // Create directory if it doesn't exist
  std::filesystem::path chunk_path(chunk_filename);
  std::filesystem::create_directories(chunk_path.parent_path());

  std::ofstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + chunk_filename);
  }

  try {
    // Write magic number and version for validation
    uint32_t magic = 0x43484E4B;  // "CHNK" in hex
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write chunk metadata
    file.write(reinterpret_cast<const char*>(&chunk_id), sizeof(chunk_id));
    uint32_t num_points = static_cast<uint32_t>(chunk_data.num_points);
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));

    // Save tensors in same order as loading
    saveTensorBinary(chunk_data.xyz, file);
    saveTensorBinary(chunk_data.features_dc, file);
    saveTensorBinary(chunk_data.features_rest, file);
    saveTensorBinary(chunk_data.scaling, file);
    saveTensorBinary(chunk_data.rotation, file);
    saveTensorBinary(chunk_data.opacity, file);
    saveTensorBinary(chunk_data.exist_since, file);
    saveTensorBinary(chunk_data.position_lrs, file);

    file.close();
    std::cout << "Saved chunk " << chunk_id << " with " << num_points
              << " points to " << chunk_filename << std::endl;

  } catch (const std::exception& e) {
    file.close();
    std::filesystem::remove(chunk_filename);  // Clean up partial file
    throw std::runtime_error("Failed to save chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

std::optional<GaussianModel::ChunkData> GaussianModel::loadSingleChunkFromDisk(
    int64_t chunk_id) {
  std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

  if (!std::filesystem::exists(chunk_filename)) {
    return std::nullopt;
  }

  // Use your existing memory-mapped loading
  std::ifstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    return std::nullopt;
  }

  // Validate magic number and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));

  if (magic != 0x43484E4B) {
    throw std::runtime_error("Invalid checkpoint file format");
  }

  int64_t stored_chunk_id;
  uint32_t stored_num_points;

  file.read(reinterpret_cast<char*>(&stored_chunk_id), sizeof(stored_chunk_id));
  file.read(reinterpret_cast<char*>(&stored_num_points),
            sizeof(stored_num_points));

  if (stored_chunk_id != chunk_id) {
    throw std::runtime_error("Saved chunk ID does not match requested ID");
  }

  ChunkData data;
  try {
    // Load tensors in the same order as saved
    data.xyz = loadTensorBinary(file);
    data.features_dc = loadTensorBinary(file);
    data.features_rest = loadTensorBinary(file);
    data.scaling = loadTensorBinary(file);
    data.rotation = loadTensorBinary(file);
    data.opacity = loadTensorBinary(file);
    data.exist_since = loadTensorBinary(file);
    data.position_lrs = loadTensorBinary(file);

    file.close();

    data.num_points = data.xyz.size(0);

    // Validate loaded data
    if (data.num_points != static_cast<int>(stored_num_points)) {
      std::cerr << "Point count mismatch in chunk file: " << chunk_filename
                << std::endl;
      return std::nullopt;
    }

    // Create chunk IDs tensor (all points belong to this chunk)
    data.chunk_ids = torch::full(
        {data.num_points}, chunk_id,
        torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

    std::cout << "Loaded chunk " << chunk_id << " with " << data.num_points
              << " points from " << chunk_filename << std::endl;

    return data;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load chunk " << chunk_id << ": " << e.what()
              << std::endl;
    return std::nullopt;
  }
}

void GaussianModel::appendLoadedChunks(
    const std::vector<ChunkData>& chunks_data,
    const std::vector<int64_t>& chunk_ids) {
  if (chunks_data.empty()) return;

  // Concatenate all chunk data
  std::vector<torch::Tensor> all_xyz, all_features_dc, all_features_rest;
  std::vector<torch::Tensor> all_scaling, all_rotation, all_opacity;
  std::vector<torch::Tensor> all_exist_since, all_position_lrs, all_chunk_ids;

  for (const auto& chunk : chunks_data) {
    all_xyz.push_back(chunk.xyz);
    all_features_dc.push_back(chunk.features_dc);
    all_features_rest.push_back(chunk.features_rest);
    all_scaling.push_back(chunk.scaling);
    all_rotation.push_back(chunk.rotation);
    all_opacity.push_back(chunk.opacity);
    all_exist_since.push_back(chunk.exist_since);
    all_position_lrs.push_back(chunk.position_lrs);
    all_chunk_ids.push_back(chunk.chunk_ids);
  }

  // Single concatenation operations
  torch::Tensor batch_xyz = torch::cat(all_xyz, 0);
  torch::Tensor batch_features_dc = torch::cat(all_features_dc, 0);
  torch::Tensor batch_features_rest = torch::cat(all_features_rest, 0);
  torch::Tensor batch_scaling = torch::cat(all_scaling, 0);
  torch::Tensor batch_rotation = torch::cat(all_rotation, 0);
  torch::Tensor batch_opacity = torch::cat(all_opacity, 0);
  torch::Tensor batch_exist_since = torch::cat(all_exist_since, 0);
  torch::Tensor batch_position_lrs = torch::cat(all_position_lrs, 0);
  torch::Tensor batch_chunk_ids = torch::cat(all_chunk_ids, 0);

  // Use existing densificationPostfix to append everything at once
  densificationPostfix(batch_xyz, batch_features_dc, batch_features_rest,
                       batch_opacity, batch_scaling, batch_rotation,
                       batch_exist_since, batch_position_lrs);

  std::cout << "Loaded " << batch_xyz.size(0) << " gaussians from "
            << chunks_data.size() << " chunks" << std::endl;
}

void GaussianModel::saveChunks(const std::vector<int64_t>& chunk_ids_to_save) {
  if (chunk_ids_to_save.empty()) return;

  for (int64_t chunk_id : chunk_ids_to_save) {
    if (!chunks_in_memory_.count(chunk_id)) {
      continue;  // Not in memory, skip
    }

    // Extract gaussians belonging to this chunk
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    int num_chunk_gaussians = chunk_mask.sum().item<int>();

    if (num_chunk_gaussians == 0) {
      continue;  // No gaussians in this chunk
    }

    ChunkData chunk_data = extractChunkData(chunk_mask);
    saveSingleChunkToDisk(chunk_id, chunk_data);

    // Update tracking
    chunks_on_disk_.insert(chunk_id);
  }
}

GaussianModel::ChunkData GaussianModel::extractChunkData(
    const torch::Tensor& chunk_mask) {
  ChunkData data;

  data.xyz = xyz_.index({chunk_mask}).detach().clone();
  data.features_dc = features_dc_.index({chunk_mask}).detach().clone();
  data.features_rest = features_rest_.index({chunk_mask}).detach().clone();
  data.scaling = scaling_.index({chunk_mask}).detach().clone();
  data.rotation = rotation_.index({chunk_mask}).detach().clone();
  data.opacity = opacity_.index({chunk_mask}).detach().clone();
  data.exist_since = exist_since_iter_.index({chunk_mask}).detach().clone();
  data.position_lrs = position_lrs_.index({chunk_mask}).detach().clone();
  data.num_points = data.xyz.size(0);

  return data;
}

void GaussianModel::saveAndEvictChunks(const std::vector<int64_t>& chunk_ids) {
  if (chunk_ids.empty()) return;

  std::cout << "Saving and evicting " << chunk_ids.size() << " chunks..."
            << std::endl;

  // Step 1: Save chunks to disk
  saveChunks(chunk_ids);

  // Step 2: Create mask of gaussians to REMOVE (opposite of keep)
  torch::Tensor remove_mask = torch::zeros(
      {xyz_.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(device_type_));

  int total_gaussians_to_remove = 0;
  for (int64_t chunk_id : chunk_ids) {
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    remove_mask = remove_mask | chunk_mask;  // OR operation to combine masks
    total_gaussians_to_remove += chunk_mask.sum().item<int>();

    // Update tracking
    chunks_in_memory_.erase(chunk_id);
  }

  std::cout << "Removing " << total_gaussians_to_remove
            << " gaussians from unified model" << std::endl;

  // Step 3: Actually remove the gaussians (this shrinks all tensors)
  if (total_gaussians_to_remove > 0) {
    prunePoints(
        remove_mask);  // This updates ALL tensors including gaussian_chunk_ids_

    std::cout << "Unified model now has " << xyz_.size(0) << " gaussians"
              << std::endl;
  }
}

size_t GaussianModel::getCurrentGPUMemoryUsage() const {
  if (torch::cuda::is_available()) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    // Get current allocated bytes (this is what we want to track for chunks)
    c10Alloc::Stat alloc_bytes =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    return alloc_bytes.current;
  }
  return 0;
}

// void GaussianModel::checkMemoryPressure() {
//   auto now = std::chrono::steady_clock::now();
//   if (now - last_memory_check_ < std::chrono::seconds(5)) {
//     return;  // Check every 5 seconds
//   }
//   last_memory_check_ = now;

//   size_t current_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);
//   size_t target_memory_mb = static_cast<size_t>(max_memory_gb_ * 1024);

//   if (current_memory_mb > target_memory_mb) {
//     // Find least recently used chunks to save
//     std::vector<int64_t> chunks_to_save =
//         findLRUChunks(current_memory_mb - target_memory_mb);

//     // Save and remove from memory
//     saveAndEvictChunks(chunks_to_save);
//   }
// }

void GaussianModel::testSaveLoadEvictCycle() {
  std::cout << "\n=== STARTING SAVE/LOAD/EVICT CYCLE TEST ===" << std::endl;

  if (!is_initialized_) {
    std::cout << "ERROR: Model not initialized, cannot run test" << std::endl;
    return;
  }

  auto test_start = std::chrono::steady_clock::now();

  // Step 1: Record initial state
  int initial_gaussians = xyz_.size(0);
  size_t initial_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

  std::cout << "INITIAL STATE:" << std::endl;
  std::cout << "  Gaussians: " << initial_gaussians << std::endl;
  std::cout << "  Memory: " << initial_memory_mb << "MB" << std::endl;
  std::cout << "  Chunks in memory: " << chunks_in_memory_.size() << std::endl;

  // Step 2: Get all unique chunks currently in memory
  std::unordered_set<int64_t> all_chunk_ids_set;
  {
    auto chunk_ids_cpu = gaussian_chunk_ids_.cpu();
    auto accessor = chunk_ids_cpu.accessor<int64_t, 1>();
    for (int i = 0; i < chunk_ids_cpu.size(0); ++i) {
      all_chunk_ids_set.insert(accessor[i]);
    }
  }

  std::vector<int64_t> all_chunk_ids(all_chunk_ids_set.begin(),
                                     all_chunk_ids_set.end());
  std::cout << "  Unique chunks: " << all_chunk_ids.size() << std::endl;

  if (all_chunk_ids.empty()) {
    std::cout << "ERROR: No chunks found in model" << std::endl;
    return;
  }

  // Step 5: Evict all chunks from memory
  std::cout << "\nEVICTING ALL CHUNKS..." << std::endl;
  auto evict_start = std::chrono::steady_clock::now();

  try {
    // Force evict all chunks (bypass LRU logic)
    saveAndEvictChunks(all_chunk_ids);

    auto evict_end = std::chrono::steady_clock::now();
    auto evict_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(evict_end -
                                                              evict_start)
            .count();

    int remaining_gaussians = xyz_.size(0);
    size_t remaining_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Eviction completed in " << evict_duration_ms << "ms"
              << std::endl;
    std::cout << "  Remaining gaussians: " << remaining_gaussians << std::endl;
    std::cout << "  Remaining memory: " << remaining_memory_mb << "MB"
              << std::endl;
    std::cout << "  Chunks in memory: " << chunks_in_memory_.size()
              << std::endl;
    std::cout << "  Memory freed: " << (initial_memory_mb - remaining_memory_mb)
              << "MB" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during eviction: " << e.what() << std::endl;
    return;
  }

  // Step 6: Reload all chunks from disk
  std::cout << "\nRELOADING ALL CHUNKS..." << std::endl;
  auto load_start = std::chrono::steady_clock::now();

  try {
    loadChunks(all_chunk_ids);

    auto load_end = std::chrono::steady_clock::now();
    auto load_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                              load_start)
            .count();

    int final_gaussians = xyz_.size(0);
    size_t final_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Load completed in " << load_duration_ms << "ms"
              << std::endl;
    std::cout << "  Final gaussians: " << final_gaussians << std::endl;
    std::cout << "  Final memory: " << final_memory_mb << "MB" << std::endl;
    std::cout << "  Chunks in memory: " << chunks_in_memory_.size()
              << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during load: " << e.what() << std::endl;
    return;
  }

  // Step 8: Final summary
  auto test_end = std::chrono::steady_clock::now();
  auto total_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(test_end -
                                                            test_start)
          .count();

  std::cout << "\n=== TEST SUMMARY ===" << std::endl;
  std::cout << "  Total time: " << total_duration_ms << "ms" << std::endl;
  std::cout << "  Gaussians: " << initial_gaussians << " -> " << xyz_.size(0)
            << " ("
            << (xyz_.size(0) == initial_gaussians ? "MATCH" : "MISMATCH") << ")"
            << std::endl;
  std::cout << "  Memory: " << initial_memory_mb << "MB -> "
            << (getCurrentGPUMemoryUsage() / (1024 * 1024)) << "MB"
            << std::endl;

  std::cout << "=== SAVE/LOAD/EVICT CYCLE TEST COMPLETE ===\n" << std::endl;
}