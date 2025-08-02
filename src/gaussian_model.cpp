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
      local_iteration_(0),
      base_scale_threshold_(6.0f),
      detail_scale_threshold_(3.0f) {
  this->sh_degree_ = sh_degree;

  // Device
  if (torch::cuda::is_available())
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

  chunks_on_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunks_loaded_from_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunk_gaussian_counts_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
}

GaussianModel::GaussianModel(const GaussianModelParams& model_params,
                             std::string storage_base_path,
                             float chunk_size)
    : storage_base_path_(storage_base_path),
      chunk_size_(chunk_size),
      sh_degree_(0),
      spatial_lr_scale_(0.0),
      position_lr_init_(0.00005),
      position_lr_decay_(0.99998),
      local_iteration_(0),
      base_scale_threshold_(6.0f),
      detail_scale_threshold_(3.0f) {
  this->sh_degree_ = model_params.sh_degree_;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

  chunks_on_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunks_loaded_from_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunk_gaussian_counts_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  std::cout << "[GaussianModel] Initialized with storage path: "
            << storage_base_path_ << " and chunk size: " << chunk_size_
            << std::endl;
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
  this->position_lrs_ = this->position_lrs_.index({valid_points_mask});
  this->gaussian_chunk_ids_ =
      this->gaussian_chunk_ids_.index({valid_points_mask});
  this->gaussian_lod_levels_ =
      this->gaussian_lod_levels_.index({valid_points_mask});

  c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::densificationPostfix(torch::Tensor& new_xyz,
                                         torch::Tensor& new_features_dc,
                                         torch::Tensor& new_features_rest,
                                         torch::Tensor& new_opacities,
                                         torch::Tensor& new_scaling,
                                         torch::Tensor& new_rotation,
                                         torch::Tensor& new_exist_since_iter,
                                         torch::Tensor& new_chunk_ids,
                                         torch::Tensor& new_position_lrs,
                                         torch::Tensor& new_lod_levels) {
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

  int num_new_primitives = new_xyz.size(0);
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  gaussian_chunk_ids_ =
      torch::cat({gaussian_chunk_ids_, new_chunk_ids}, /*dim=*/0);
  // Append LoD levels
  gaussian_lod_levels_ = torch::cat({gaussian_lod_levels_, new_lod_levels}, 0);
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
  torch::NoGradGuard no_grad;

  // Frustum cull chunks
  std::vector<ChunkCoord> visible_chunks =
      frustumCullChunks(keyframe, /*use_cache=*/true);

  if (visible_chunks.empty()) {
    // Return all-false mask
    return torch::zeros(
        {xyz_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  torch::Tensor visible_chunk_coords = chunkCoordVectorToTensor(visible_chunks);
  torch::Tensor visible_chunk_ids =
      encodeChunkCoordsTensor(visible_chunk_coords);

  if (visible_chunk_ids.size(0) > 0) {
    loadChunks(visible_chunk_ids);
  }

  // Convert chunk visibility to gaussian visibility
  torch::Tensor chunk_visibility_mask =
      createGaussianMaskFromChunks(visible_chunk_ids);

  // return chunk_visibility_mask;

  // Apply LoD filtering based on distance
  torch::Tensor camera_position = keyframe->getCenter().squeeze();
  torch::Tensor lod_filtered_mask =
      selectCumulativeLoD(chunk_visibility_mask, camera_position);

  // Debug: Print culling statistics
  // int chunk_visible = chunk_visibility_mask.sum().item<int>();
  // int lod_visible = lod_filtered_mask.sum().item<int>();
  // int total_gaussians = xyz_.size(0);
  // float reduction =
  //     (chunk_visible > 0)
  //         ? (1.0f - float(lod_visible) / float(chunk_visible)) * 100.0f
  //         : 0.0f;

  // std::cout << "[LoD Debug] Culling stats - Total: " << total_gaussians
  //           << ", Chunk visible: " << chunk_visible
  //           << ", LoD visible: " << lod_visible << " (reduction: " <<
  //           reduction
  //           << "%)" << std::endl;

  //  Update access times for all visible chunks
  updateChunkAccess(visible_chunk_ids);

  return lod_filtered_mask;
}

torch::Tensor GaussianModel::createGaussianMaskFromChunks(
    const torch::Tensor& visible_chunk_ids) {
  if (visible_chunk_ids.size(0) == 0) {
    return torch::zeros(
        {gaussian_chunk_ids_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  return torch::isin(gaussian_chunk_ids_, visible_chunk_ids);
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
  torch::NoGradGuard no_grad;
  float half_chunk = chunk_size_ * 0.5f;

  torch::Tensor shifted_positions = positions + half_chunk;
  torch::Tensor chunk_coords = torch::floor(shifted_positions / chunk_size_);
  chunk_coords = chunk_coords.to(torch::kInt64);

  // Use the SAME encoding as encodeChunkCoordsTensor
  return encodeChunkCoordsTensor(chunk_coords);
}

void GaussianModel::addPoints(const torch::Tensor& new_xyz,
                              const torch::Tensor& new_colors,
                              const torch::Tensor& new_scales,
                              const torch::Tensor& new_opacities,
                              int iteration,
                              float spatial_lr_scale) {
  torch::NoGradGuard no_grad;

  auto start_time = std::chrono::steady_clock::now();

  // Apply chunk density filtering

  auto filter_start_time = std::chrono::steady_clock::now();
  auto [filtered_xyz, filtered_colors, filtered_scales, filtered_opacities] =
      filterPointsByChunkDensity(new_xyz, new_colors, new_scales, new_opacities,
                                 new_gaussian_chunk_density_);
  auto filter_end_time = std::chrono::steady_clock::now();
  auto filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      filter_end_time - filter_start_time);
  // std::cout << "filterPointsByChunkDensity completed in "
  //           << filter_duration.count() << "ms" << std::endl;

  if (filtered_xyz.size(0) == 0) {
    std::cout
        << "[Gaussian Model] Warning: No points passed chunk density filter"
        << std::endl;
    return;
  }

  // std::cout << "[Gaussian Model] Filtered points: " << filtered_xyz.size(0)
  //           << " -> " << filtered_xyz.size(0) << " (kept "
  //           << (100.0f * filtered_xyz.size(0) / new_xyz.size(0)) << "%)"
  //           << std::endl;

  // Only care about existing disk chunks that need loading
  torch::Tensor affected_chunks =
      std::get<0>(torch::_unique2(computeChunkIds(filtered_xyz)));
  torch::Tensor existing_disk_chunks =
      torch::isin(affected_chunks, chunks_on_disk_);
  torch::Tensor unloaded_disk_chunks = affected_chunks.index(
      {existing_disk_chunks &
       ~torch::isin(affected_chunks, chunks_loaded_from_disk_)});

  if (unloaded_disk_chunks.size(0) > 0) {
    // Load existing disk chunks that we're about to add points to
    loadChunks(unloaded_disk_chunks);
  }

  if (!is_initialized_) {
    // First call - initialize the model
    initializeFromPoints(filtered_xyz, filtered_colors, filtered_scales,
                         filtered_opacities, iteration, spatial_lr_scale);
  } else {
    // Subsequent calls - append to existing model
    appendPoints(filtered_xyz, filtered_colors, filtered_scales,
                 filtered_opacities, iteration);
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // std::cout << "addPoints completed in " << duration.count() << "ms"
  //           << std::endl;
}

void GaussianModel::initializeFromPoints(const torch::Tensor& initial_xyz,
                                         const torch::Tensor& initial_colors,
                                         const torch::Tensor& initial_scales,
                                         const torch::Tensor& initial_opacities,
                                         int iteration,
                                         float spatial_lr_scale) {
  torch::NoGradGuard no_grad;
  std::cout << "[Gaussian Model] Initializing from points: "
            << initial_xyz.sizes() << std::endl;

  this->spatial_lr_scale_ = spatial_lr_scale;
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

  torch::Tensor scales;
  if (initial_scales.defined() && initial_scales.size(0) > 0) {
    scales = initial_scales;
  } else {
    torch::Tensor point_cloud_copy = initial_xyz.clone();
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

  gaussian_chunk_ids_ = computeChunkIds(initial_xyz);

  // Assign LoD levels based on density (distance to nearest neighbors)
  torch::Tensor point_cloud_copy = initial_xyz.clone();
  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
  torch::Tensor nearest_distances = torch::sqrt(dist2);
  gaussian_lod_levels_ = assignLoDByDensity(nearest_distances);

  // Debug: Print initial LoD assignment statistics
  auto lod_counts = torch::bincount(gaussian_lod_levels_, torch::Tensor(), 3);
  // std::cout << "[LoD Debug] Initialized " << gaussian_lod_levels_.size(0)
  //           << " gaussians - "
  //           << "LoD0: " << lod_counts[0].item<int>() << ", "
  //           << "LoD1: " << lod_counts[1].item<int>() << ", "
  //           << "LoD2: " << lod_counts[2].item<int>()
  //           << " (density range: " << nearest_distances.min().item<float>()
  //           << " - " << nearest_distances.max().item<float>() << ")"
  //           << std::endl;

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  c10::cuda::CUDACachingAllocator::emptyCache();

  is_initialized_ = true;
}

void GaussianModel::appendPoints(const torch::Tensor& new_xyzs,
                                 const torch::Tensor& new_colors,
                                 const torch::Tensor& new_scales,
                                 const torch::Tensor& new_opacities,
                                 int iteration) {
  torch::NoGradGuard no_grad;
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
  auto new_xyz_tensor = new_xyzs;
  auto new_scaling = scales;
  auto new_rotation = rots;

  torch::Tensor new_position_lrs =
      torch::full({new_xyzs.size(0)}, position_lr_init_,
                  torch::TensorOptions().device(device_type_));

  // Assign LoD levels based on density (distance to nearest neighbors)
  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(new_xyzs.clone()), 0.0000001);
  torch::Tensor nearest_distances = torch::sqrt(dist2);
  torch::Tensor new_lod_levels = assignLoDByDensity(nearest_distances);

  // Debug: Print LoD assignment statistics
  auto lod_counts = torch::bincount(new_lod_levels, torch::Tensor(), 3);
  // std::cout << "[LoD Debug] Added " << new_lod_levels.size(0) << " gaussians
  // - "
  //           << "LoD0: " << lod_counts[0].item<int>() << ", "
  //           << "LoD1: " << lod_counts[1].item<int>() << ", "
  //           << "LoD2: " << lod_counts[2].item<int>()
  //           << " (density range: " << nearest_distances.min().item<float>()
  //           << " - " << nearest_distances.max().item<float>() << ")"
  //           << std::endl;

  torch::Tensor new_chunk_ids = computeChunkIds(new_xyzs);

  densificationPostfix(new_xyz_tensor, new_features_dc, new_features_rest,
                       new_opacities_tensor, new_scaling, new_rotation,
                       new_exist_since_iter, new_chunk_ids, new_position_lrs,
                       new_lod_levels);

  c10::cuda::CUDACachingAllocator::emptyCache();
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

void GaussianModel::loadChunks(const torch::Tensor& chunk_id_requests) {
  torch::NoGradGuard no_grad;
  if (chunk_id_requests.size(0) == 0) return;

  // Requested that are on disk
  torch::Tensor on_disk_mask = torch::isin(chunk_id_requests, chunks_on_disk_);

  // Requested that aren't loaded from disk
  torch::Tensor not_loaded_mask =
      ~torch::isin(chunk_id_requests, chunks_loaded_from_disk_);

  // Requested that are on disk and not loaded from disk
  torch::Tensor on_disk_not_loaded_mask = on_disk_mask & not_loaded_mask;

  // IDs of requested that are on disk but not loaded from disk
  torch::Tensor chunks_ids_needing_load =
      chunk_id_requests.index({on_disk_not_loaded_mask});

  // Filter out chunks that are already loaded
  if (chunks_ids_needing_load.size(0) == 0) {
    // std::cout << "[Load] All requested chunks already loaded" << std::endl;
    return;
  }

  // STEP 0: Pre-emptive eviction using exact gaussian counts
  if (chunks_ids_needing_load.any().item<bool>()) {
    // Find indices of loadable chunks in chunks_on_disk_
    int64_t exact_gaussians_to_load = 0;
    auto to_load_cpu = chunks_ids_needing_load.cpu();
    auto chunks_on_disk_cpu = chunks_on_disk_.cpu();
    auto counts_cpu = chunk_gaussian_counts_.cpu();

    auto to_load_accessor = to_load_cpu.accessor<int64_t, 1>();
    auto chunks_on_disk_accessor = chunks_on_disk_cpu.accessor<int64_t, 1>();
    auto counts_accessor = counts_cpu.accessor<int64_t, 1>();

    // Iterate through chunks needing load
    for (int64_t i = 0; i < to_load_accessor.size(0); i++) {
      int64_t to_load_chunk_id = to_load_accessor[i];

      // Find this chunk in chunks_on_disk_
      for (int64_t j = 0; j < chunks_on_disk_cpu.size(0); j++) {
        if (chunks_on_disk_accessor[j] == to_load_chunk_id) {
          exact_gaussians_to_load += counts_accessor[j];
          break;  // Found it, move to next requested chunk
        }
      }
    }

    int64_t current_gaussians = xyz_.size(0);
    int64_t target_gaussians = 3000000;
    int64_t projected_total = current_gaussians + exact_gaussians_to_load;

    std::cout << "[Load] Planning to load " << chunks_ids_needing_load.size(0)
              << " chunks (exactly " << exact_gaussians_to_load << " gaussians)"
              << std::endl;

    if (projected_total > target_gaussians) {
      int64_t excess = projected_total - target_gaussians;
      std::cout << "[Load] Pre-emptive eviction needed: current="
                << current_gaussians << ", incoming=" << exact_gaussians_to_load
                << ", projected=" << projected_total << ", excess=" << excess
                << std::endl;

      // Get evictable chunks and find LRU ones to free up 'excess' gaussians
      torch::Tensor spatial_chunks =
          std::get<0>(torch::_unique2(gaussian_chunk_ids_));
      if (spatial_chunks.size(0) > 0) {
        torch::Tensor lru_chunks = findLRUChunks(spatial_chunks, excess);
        if (lru_chunks.size(0) > 0) {
          saveAndEvictChunks(lru_chunks);
        } else {
          throw std::runtime_error("No evictable chunks found");
        }
      } else {
        throw std::runtime_error(
            "Literally no chunks exist but we need to evict?");
      }
    }
  }

  // Parallel load from disk
  if (chunks_ids_needing_load.size(0) > 0) {
    // Remove spillover first
    // torch::Tensor spillover_mask =
    //     torch::isin(gaussian_chunk_ids_, loadable_chunks);
    // int spillover_count = spillover_mask.sum().item<int>();

    // if (spillover_count > 0) {
    //   std::cout << "[Load] Removing " << spillover_count
    //             << " spillover gaussians before loading "
    //             << loadable_chunks.size(0) << " chunks from disk" <<
    //             std::endl;
    //   prunePoints(spillover_mask);
    // }

    // PARALLEL LOADING
    auto chunks_ids_needing_load_cpu = chunks_ids_needing_load.cpu();
    auto accessor = chunks_ids_needing_load_cpu.accessor<int64_t, 1>();
    int num_chunks = chunks_ids_needing_load_cpu.size(0);

    std::vector<std::future<std::pair<int64_t, std::optional<ChunkData>>>>
        futures;

    // Launch parallel load tasks
    for (int i = 0; i < num_chunks; ++i) {
      int64_t chunk_id = accessor[i];

      auto future = std::async(std::launch::async, [this, chunk_id]() {
        return std::make_pair(chunk_id, loadSingleChunkFromDisk(chunk_id));
      });

      futures.push_back(std::move(future));
    }

    // Collect results
    std::vector<GaussianModel::ChunkData> chunks_to_append;
    std::vector<int64_t> loaded_chunk_ids;

    for (auto& future : futures) {
      auto [chunk_id, chunk_data] = future.get();
      if (chunk_data.has_value()) {
        chunks_to_append.push_back(chunk_data.value());
        loaded_chunk_ids.push_back(chunk_id);
      }
    }

    // Append results
    if (!chunks_to_append.empty()) {
      appendLoadedChunks(chunks_to_append, loaded_chunk_ids);
    }

    // Mark as loaded
    chunks_loaded_from_disk_ =
        torch::cat({chunks_loaded_from_disk_, chunks_ids_needing_load}, 0);

    // Clean up duplicates
    chunks_loaded_from_disk_ =
        std::get<0>(torch::_unique2(chunks_loaded_from_disk_));
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
    saveTensorBinary(chunk_data.lod_levels, file);

    // Save optimizer states
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      // Write step count
      file.write(
          reinterpret_cast<const char*>(&chunk_data.step_counts[group_idx]),
          sizeof(int64_t));

      // Save momentum tensors
      if (chunk_data.exp_avg_states[group_idx].defined()) {
        saveTensorBinary(chunk_data.exp_avg_states[group_idx], file);
        saveTensorBinary(chunk_data.exp_avg_sq_states[group_idx], file);
      } else {
        // Save empty tensors as placeholders
        torch::Tensor empty = torch::empty({0});
        saveTensorBinary(empty, file);
        saveTensorBinary(empty, file);
      }
    }

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

// Add this to your loadSingleChunkFromDisk function to isolate bottlenecks:

std::optional<GaussianModel::ChunkData> GaussianModel::loadSingleChunkFromDisk(
    int64_t chunk_id) {
  ChunkCoord chunk_coord = decodeChunkCoord(chunk_id);
  std::string chunk_filename = getChunkFilename(chunk_coord);

  if (!std::filesystem::exists(chunk_filename)) {
    throw std::runtime_error("Chunk file does not exist: " + chunk_filename);
    return std::nullopt;
  }

  // Use your existing memory-mapped loading
  std::ifstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + chunk_filename);
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
    data.lod_levels = loadTensorBinary(file);

    // Load optimizer states
    data.exp_avg_states.resize(6);
    data.exp_avg_sq_states.resize(6);
    data.step_counts.resize(6);

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      // Load step count
      file.read(reinterpret_cast<char*>(&data.step_counts[group_idx]),
                sizeof(int64_t));

      // Load momentum tensors
      data.exp_avg_states[group_idx] = loadTensorBinary(file);
      data.exp_avg_sq_states[group_idx] = loadTensorBinary(file);
    }

    data.num_points = data.xyz.size(0);

    file.close();

    // Validate loaded data
    if (data.num_points != static_cast<int>(stored_num_points)) {
      std::cerr << "Point count mismatch in chunk file: " << chunk_filename
                << std::endl;
      return std::nullopt;
    }

    // Create chunk IDs tensor (all points belong to this chunk)
    data.chunk_id = chunk_id;

    std::cout << "Loaded chunk " << chunk_id << " with " << data.num_points
              << " points from " << chunk_filename << std::endl;

    return data;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load chunk " << chunk_id << ": " << e.what()
              << std::endl;
    throw std::runtime_error("Failed to load chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
    return std::nullopt;
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

void GaussianModel::appendLoadedChunks(
    const std::vector<ChunkData>& chunks_data,
    const std::vector<int64_t>& chunk_ids) {
  torch::NoGradGuard no_grad;
  if (chunks_data.empty()) return;

  // Concatenate all chunk data
  std::vector<torch::Tensor> all_xyz, all_features_dc, all_features_rest;
  std::vector<torch::Tensor> all_scaling, all_rotation, all_opacity;
  std::vector<torch::Tensor> all_exist_since, all_chunk_ids, all_position_lrs,
      all_lod_levels;

  // NEW: Concatenate optimizer states
  std::vector<std::vector<torch::Tensor>> all_exp_avg(6), all_exp_avg_sq(6);
  std::vector<int64_t> max_step_counts(6, 0);

  for (const auto& chunk : chunks_data) {
    all_xyz.push_back(chunk.xyz);
    all_features_dc.push_back(chunk.features_dc);
    all_features_rest.push_back(chunk.features_rest);
    all_scaling.push_back(chunk.scaling);
    all_rotation.push_back(chunk.rotation);
    all_opacity.push_back(chunk.opacity);
    all_exist_since.push_back(chunk.exist_since);
    all_position_lrs.push_back(chunk.position_lrs);
    all_lod_levels.push_back(chunk.lod_levels);

    torch::Tensor chunk_ids = torch::full(
        {chunk.num_points}, chunk.chunk_id,
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));
    all_chunk_ids.push_back(chunk_ids);

    // NEW: Collect optimizer states
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
      if (chunk.exp_avg_states[group_idx].defined()) {
        all_exp_avg[group_idx].push_back(chunk.exp_avg_states[group_idx]);
        all_exp_avg_sq[group_idx].push_back(chunk.exp_avg_sq_states[group_idx]);
      }
      max_step_counts[group_idx] =
          std::max(max_step_counts[group_idx], chunk.step_counts[group_idx]);
    }
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
  torch::Tensor batch_lod_levels = torch::cat(all_lod_levels, 0);
  torch::Tensor batch_chunk_ids = torch::cat(all_chunk_ids, 0);

  // Get starting index for new gaussians
  int old_size = xyz_.size(0);

  // Use existing densificationPostfix to append everything at once
  densificationPostfix(batch_xyz, batch_features_dc, batch_features_rest,
                       batch_opacity, batch_scaling, batch_rotation,
                       batch_exist_since, batch_chunk_ids, batch_position_lrs,
                       batch_lod_levels);

  // NEW: Restore optimizer states for the loaded range
  int new_size = xyz_.size(0);
  restoreOptimizerStatesForRange(all_exp_avg, all_exp_avg_sq, max_step_counts,
                                 old_size, new_size);

  std::cout << "Loaded " << batch_xyz.size(0) << " gaussians from "
            << chunks_data.size() << " chunks with full optimizer states"
            << std::endl;
}

void GaussianModel::restoreOptimizerStatesForRange(
    const std::vector<std::vector<torch::Tensor>>& all_exp_avg,
    const std::vector<std::vector<torch::Tensor>>& all_exp_avg_sq,
    const std::vector<int64_t>& max_step_counts,
    int start_idx,
    int end_idx) {
  if (!optimizer_) return;

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    if (group_idx >= param_groups.size()) continue;
    if (all_exp_avg[group_idx].empty()) continue;

    auto& param_group = param_groups[group_idx];
    if (param_group.params().empty()) continue;

    auto& param = param_group.params()[0];
    auto key = param.unsafeGetTensorImpl();

    // Ensure optimizer state exists
    if (state.find(key) == state.end()) {
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(0);
      new_state->exp_avg(torch::zeros_like(param));
      new_state->exp_avg_sq(torch::zeros_like(param));
      state[key] = std::move(new_state);
    }

    auto& param_state = static_cast<torch::optim::AdamParamState&>(*state[key]);

    // Concatenate states from all chunks
    torch::Tensor concat_exp_avg = torch::cat(all_exp_avg[group_idx], 0);
    torch::Tensor concat_exp_avg_sq = torch::cat(all_exp_avg_sq[group_idx], 0);

    // Update the loaded range in optimizer state
    param_state.exp_avg().slice(0, start_idx, end_idx).copy_(concat_exp_avg);
    param_state.exp_avg_sq()
        .slice(0, start_idx, end_idx)
        .copy_(concat_exp_avg_sq);

    // Update step count to maximum from loaded chunks
    if (max_step_counts[group_idx] > param_state.step()) {
      param_state.step(max_step_counts[group_idx]);
    }
  }
}

void GaussianModel::saveChunks(const torch::Tensor& chunk_ids_to_save) {
  torch::NoGradGuard no_grad;
  if (chunk_ids_to_save.size(0) == 0) return;

  auto start_time = std::chrono::steady_clock::now();

  std::cout << "[Chunk Save] Saving " << chunk_ids_to_save.size(0)
            << " chunks to disk" << std::endl;

  auto chunks_cpu = chunk_ids_to_save.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num_chunks = chunks_cpu.size(0);

  std::vector<std::pair<int64_t, ChunkData>> prepared_chunks;
  std::unordered_map<int64_t, int64_t> chunk_id_to_count;
  for (int i = 0; i < num_chunks; ++i) {
    int64_t chunk_id = accessor[i];
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    ChunkData chunk_data = extractChunkData(chunk_mask, chunk_id);
    chunk_id_to_count[chunk_id] = chunk_data.num_points;
    prepared_chunks.emplace_back(chunk_id, std::move(chunk_data));
  }

  // PARALLEL SAVING - NEW CODE
  std::vector<std::future<std::pair<int64_t, bool>>> futures;
  for (auto& [chunk_id, chunk_data] : prepared_chunks) {
    auto future = std::async(
        std::launch::async,
        [this](int64_t id, ChunkData data) {
          try {
            saveSingleChunkToDisk(id, data);  // Pure CPU/I/O work
            return std::make_pair(id, true);
          } catch (const std::exception& e) {
            std::cerr << "Failed to save chunk " << id << ": " << e.what()
                      << std::endl;
            return std::make_pair(id, false);
          }
        },
        chunk_id, std::move(chunk_data));

    futures.push_back(std::move(future));
  }

  // Collect results
  std::vector<int64_t> successfully_saved;
  for (auto& future : futures) {
    auto [chunk_id, success] = future.get();
    if (success) {
      successfully_saved.push_back(chunk_id);
    }
  }

  // Check if chunk_gaussian_counts_ and chunks_on_disk_ have gone out of sync
  if (chunk_gaussian_counts_.size(0) != chunks_on_disk_.size(0)) {
    throw std::runtime_error(
        "chunk_gaussian_counts_ doesn't match chunks_on_disk_ size!");
  }

  // Now add new chunk entries to chunks_on_disk_ and update
  // chunk_gaussian_counts_ for existing and new
  std::vector<int64_t> saved_counts;
  for (int64_t chunk_id : successfully_saved) {
    torch::Tensor chunk_id_tensor = torch::tensor(
        {chunk_id},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));
    torch::Tensor gaussian_count_tensor = torch::tensor(
        {chunk_id_to_count[chunk_id]},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));

    auto mask = torch::eq(chunks_on_disk_, chunk_id_tensor);
    bool found = torch::any(mask).item<bool>();

    // If we find existing entry, update the gaussian count
    if (found) {
      auto indices = torch::where(mask)[0];

      // If there are multiple entries with this chunk id, we messed up
      // somewhere
      if (indices.size(0) > 1) {
        throw std::runtime_error("chunks_on_disk_ has duplicates");
      }
      int64_t first_index = indices[0].item<int64_t>();
      chunk_gaussian_counts_[first_index] = chunk_id_to_count[chunk_id];

      // If first time observing this chunk id add it and the count
    } else {
      chunks_on_disk_ = torch::cat({chunks_on_disk_, chunk_id_tensor}, 0);
      chunk_gaussian_counts_ =
          torch::cat({chunk_gaussian_counts_, gaussian_count_tensor}, 0);
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "saveChunks completed in " << duration.count() << "ms"
            << std::endl;
}

GaussianModel::ChunkData GaussianModel::extractChunkData(
    const torch::Tensor& chunk_mask,
    int64_t chunk_id) {
  ChunkData data;

  // Extract main tensors
  data.xyz = xyz_.index({chunk_mask}).detach().clone();
  data.features_dc = features_dc_.index({chunk_mask}).detach().clone();
  data.features_rest = features_rest_.index({chunk_mask}).detach().clone();
  data.scaling = scaling_.index({chunk_mask}).detach().clone();
  data.rotation = rotation_.index({chunk_mask}).detach().clone();
  data.opacity = opacity_.index({chunk_mask}).detach().clone();
  data.exist_since = exist_since_iter_.index({chunk_mask}).detach().clone();
  data.position_lrs = position_lrs_.index({chunk_mask}).detach().clone();
  data.lod_levels = gaussian_lod_levels_.index({chunk_mask}).detach().clone();
  data.num_points = data.xyz.size(0);
  data.chunk_id = chunk_id;

  // NEW: Extract optimizer states
  data.exp_avg_states.resize(6);
  data.exp_avg_sq_states.resize(6);
  data.step_counts.resize(6);

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& param_group = param_groups[group_idx];
    auto& param = param_group.params()[0];
    auto key = param.unsafeGetTensorImpl();

    if (state.find(key) != state.end()) {
      // Extract existing optimizer state
      auto& param_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);

      data.exp_avg_states[group_idx] =
          param_state.exp_avg().index({chunk_mask}).detach().clone();
      data.exp_avg_sq_states[group_idx] =
          param_state.exp_avg_sq().index({chunk_mask}).detach().clone();
      data.step_counts[group_idx] = param_state.step();
    } else {
      throw std::runtime_error("No param state found");
    }
  }

  return data;
}

void GaussianModel::saveAndEvictChunks(const torch::Tensor& chunk_ids) {
  if (chunk_ids.size(0) == 0) return;

  // Step 1: Save loaded chunks (same as before)
  torch::Tensor loaded_mask = torch::isin(chunk_ids, chunks_loaded_from_disk_);
  torch::Tensor chunks_to_save = chunk_ids.index({loaded_mask});

  if (chunks_to_save.size(0) > 0) {
    saveChunks(chunks_to_save);
  }

  // Step 2: Categorize non-loaded chunks
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));
  torch::Tensor has_gaussians_mask = torch::isin(chunk_ids, spatial_chunks);
  torch::Tensor non_loaded_with_gaussians =
      chunk_ids.index({has_gaussians_mask & (~loaded_mask)});

  if (non_loaded_with_gaussians.size(0) > 0) {
    // Distinguish spillover vs new chunks
    torch::Tensor is_spillover_mask =
        torch::isin(non_loaded_with_gaussians, chunks_on_disk_);
    torch::Tensor spillover_chunks =
        non_loaded_with_gaussians.index({is_spillover_mask});
    torch::Tensor new_chunks =
        non_loaded_with_gaussians.index({~is_spillover_mask});

    // Handle spillover chunks - discard without saving
    if (spillover_chunks.size(0) > 0) {
      std::cout << "[Eviction] Discarding " << spillover_chunks.size(0)
                << " spillover-only chunks" << std::endl;
    }

    // Handle new chunks - save them!
    if (new_chunks.size(0) > 0) {
      std::cout << "[Eviction] Saving " << new_chunks.size(0) << " new chunks"
                << std::endl;
      saveChunks(new_chunks);  // Save the new chunks to disk
    }
  }

  // Step 3: Remove all gaussians from evicted chunks
  torch::Tensor remove_mask = torch::isin(gaussian_chunk_ids_, chunk_ids);
  if (remove_mask.sum().item<int>() > 0) {
    prunePoints(remove_mask);
  }

  // Step 4: Update tracking
  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, chunk_ids);
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});
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

void GaussianModel::checkMemoryPressure() {
  auto now = std::chrono::steady_clock::now();
  // if (now - last_memory_check_ < std::chrono::milliseconds(100)) {
  //   return;
  // }
  // last_memory_check_ = now;

  int current_gaussians = getXYZ().size(0);
  int threshold_gaussians = 3000000;

  if (current_gaussians <= threshold_gaussians) {
    return;  // No pressure, exit early
  }

  // Keep evicting until we reach our memory goal or run out of chunks
  while (true) {
    current_gaussians = getXYZ().size(0);
    if (current_gaussians <= threshold_gaussians) {
      // Goal reached, exit the loop
      break;
    }

    // Get chunks that can be evicted (any chunk with gaussians in memory)
    torch::Tensor evictable_chunks =
        std::get<0>(torch::_unique2(gaussian_chunk_ids_));

    if (evictable_chunks.size(0) == 0) {
      // No chunks found to evict, break to avoid infinite loop
      std::cout << "[Memory] Warning: No evictable chunks found" << std::endl;
      break;
    }

    // Calculate how many gaussians to evict this iteration
    int64_t excess_gaussians = current_gaussians - threshold_gaussians;
    int64_t gaussians_to_evict =
        std::max(excess_gaussians,
                 static_cast<int64_t>(100000));  // Minimum 100k per iteration

    // Get LRU chunks that total at least gaussians_to_evict
    torch::Tensor lru_chunks =
        findLRUChunks(evictable_chunks, gaussians_to_evict);

    if (lru_chunks.size(0) == 0) {
      std::cout << "[Memory] Warning: No LRU chunks found to evict"
                << std::endl;
      break;
    }

    std::cout << "[Memory] Evicting " << lru_chunks.size(0)
              << " LRU chunks (current Gaussians: " << current_gaussians
              << " target: " << threshold_gaussians << std::endl;

    // Use updated saveAndEvictChunks
    saveAndEvictChunks(lru_chunks);
  }
}

torch::Tensor GaussianModel::findLRUChunks(
    const torch::Tensor& candidate_chunks,
    int64_t target_gaussian_count) {
  if (candidate_chunks.size(0) == 0 || target_gaussian_count <= 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  }

  auto chunks_cpu = candidate_chunks.cpu();
  auto chunks_accessor = chunks_cpu.accessor<int64_t, 1>();

  // Create vector of (chunk_id, access_time, gaussian_count) tuples
  std::vector<std::tuple<int64_t, float, int64_t>> chunk_data;
  for (int64_t i = 0; i < chunks_cpu.size(0); i++) {
    int64_t chunk_id = chunks_accessor[i];
    float access_time = chunk_access_times_.count(chunk_id)
                            ? chunk_access_times_[chunk_id]
                            : 0.0f;

    // Get gaussian count for this chunk
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    int64_t gaussian_count = chunk_mask.sum().item<int64_t>();

    chunk_data.emplace_back(chunk_id, access_time, gaussian_count);
  }

  // Sort by access time (oldest first)
  std::sort(chunk_data.begin(), chunk_data.end(),
            [](const auto& a, const auto& b) {
              return std::get<1>(a) < std::get<1>(b);
            });

  // Debug output
  if (!chunk_data.empty()) {
    float oldest_time = std::get<1>(chunk_data.front());
    float newest_time = std::get<1>(chunk_data.back());
    float time_delta = newest_time - oldest_time;
    std::cout << "[LRU DEBUG] After sorting (oldest first):" << std::endl;
    std::cout << "[LRU DEBUG] Time range: oldest=" << oldest_time
              << ", newest=" << newest_time << ", delta=" << time_delta << "ms"
              << std::endl;
  }

  // Accumulate chunks until we reach target gaussian count
  std::vector<int64_t> selected_chunks;
  int64_t accumulated_gaussians = 0;

  for (const auto& [chunk_id, access_time, gaussian_count] : chunk_data) {
    selected_chunks.push_back(chunk_id);
    accumulated_gaussians += gaussian_count;

    if (accumulated_gaussians >= target_gaussian_count) {
      break;
    }
  }

  std::cout << "[LRU DEBUG] Selected " << selected_chunks.size() << " chunks ("
            << accumulated_gaussians << " gaussians) to reach target eviction"
            << target_gaussian_count << std::endl;

  // Convert to tensor
  torch::Tensor result = torch::empty(
      {static_cast<int64_t>(selected_chunks.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto result_accessor = result.accessor<int64_t, 1>();

  for (size_t i = 0; i < selected_chunks.size(); i++) {
    result_accessor[i] = selected_chunks[i];
  }

  return result.to(device_type_);
}

void GaussianModel::testSaveLoadEvictCycle() {
  std::cout << "\n=== STARTING VECTORIZED SAVE/LOAD/EVICT CYCLE TEST ==="
            << std::endl;

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
  std::cout << "  Loaded chunks: " << chunks_loaded_from_disk_.size(0)
            << std::endl;
  std::cout << "  Chunks on disk: " << chunks_on_disk_.size(0) << std::endl;

  // Step 2: Get all unique chunks currently in model (spatial chunks)
  torch::Tensor all_spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  std::cout << "  Spatial chunks: " << all_spatial_chunks.size(0) << std::endl;
  std::cout << "  Loaded chunks: " << chunks_loaded_from_disk_.size(0)
            << std::endl;

  if (all_spatial_chunks.size(0) == 0) {
    std::cout << "ERROR: No chunks found in model" << std::endl;
    return;
  }

  // Step 3: Evict all chunks from memory (vectorized)
  std::cout << "\nEVICTING ALL CHUNKS (VECTORIZED)..." << std::endl;
  auto evict_start = std::chrono::steady_clock::now();

  // Store the chunks we're testing so we only reload exactly these
  torch::Tensor chunks_to_test = all_spatial_chunks.clone();

  try {
    // Force evict all spatial chunks using vectorized operations
    saveAndEvictChunks(all_spatial_chunks);

    auto evict_end = std::chrono::steady_clock::now();
    auto evict_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(evict_end -
                                                              evict_start)
            .count();

    int remaining_gaussians = xyz_.size(0);
    size_t remaining_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Vectorized eviction completed in " << evict_duration_ms
              << "ms" << std::endl;
    std::cout << "  Remaining gaussians: " << remaining_gaussians << std::endl;
    std::cout << "  Remaining memory: " << remaining_memory_mb << "MB"
              << std::endl;
    std::cout << "  Loaded chunks after eviction: "
              << chunks_loaded_from_disk_.size(0) << std::endl;
    std::cout << "  Chunks on disk after eviction: " << chunks_on_disk_.size(0)
              << std::endl;
    std::cout << "  Memory freed: " << (initial_memory_mb - remaining_memory_mb)
              << "MB" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during vectorized eviction: " << e.what() << std::endl;
    return;
  }

  // Step 4: Reload all chunks from disk (vectorized)
  std::cout << "\nRELOADING ALL CHUNKS (VECTORIZED)..." << std::endl;
  auto load_start = std::chrono::steady_clock::now();

  try {
    // Load only the chunks we originally evicted, not all chunks on disk
    // (chunks_on_disk_ may contain additional chunks saved during memory
    // pressure)
    if (chunks_to_test.size(0) > 0) {
      loadChunks(chunks_to_test);
    } else {
      std::cout << "  WARNING: No chunks to reload!" << std::endl;
    }

    auto load_end = std::chrono::steady_clock::now();
    auto load_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                              load_start)
            .count();

    int final_gaussians = xyz_.size(0);
    size_t final_memory_mb = getCurrentGPUMemoryUsage() / (1024 * 1024);

    std::cout << "  Vectorized load completed in " << load_duration_ms << "ms"
              << std::endl;
    std::cout << "  Final gaussians: " << final_gaussians << std::endl;
    std::cout << "  Final memory: " << final_memory_mb << "MB" << std::endl;
    std::cout << "  Loaded chunks after reload: "
              << chunks_loaded_from_disk_.size(0) << std::endl;
    std::cout << "  Spatial chunks after reload: "
              << std::get<0>(torch::_unique2(gaussian_chunk_ids_)).size(0)
              << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR during vectorized load: " << e.what() << std::endl;
    return;
  }

  // Step 5: Validation and summary
  auto test_end = std::chrono::steady_clock::now();
  auto total_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(test_end -
                                                            test_start)
          .count();

  // Check if we recovered all gaussians
  bool gaussians_match = (xyz_.size(0) == initial_gaussians);
  bool chunks_properly_loaded = (chunks_loaded_from_disk_.size(0) > 0);

  std::cout << "\n=== VECTORIZED TEST SUMMARY ===" << std::endl;
  std::cout << "  Total time: " << total_duration_ms << "ms" << std::endl;
  std::cout << "  Gaussians: " << initial_gaussians << " -> " << xyz_.size(0)
            << " (" << (gaussians_match ? "MATCH" : "MISMATCH") << ")"
            << std::endl;
  std::cout << "  Memory: " << initial_memory_mb << "MB -> "
            << (getCurrentGPUMemoryUsage() / (1024 * 1024)) << "MB"
            << std::endl;
  std::cout << "  Chunks properly loaded: "
            << (chunks_properly_loaded ? "YES" : "NO") << std::endl;

  // Additional validation
  if (!gaussians_match) {
    std::cout << "  WARNING: Gaussian count mismatch - possible data loss!"
              << std::endl;
  }

  if (!chunks_properly_loaded) {
    std::cout << "  WARNING: No chunks marked as loaded - state tracking issue!"
              << std::endl;
  }

  if (gaussians_match && chunks_properly_loaded) {
    std::cout << "  ✅ TEST PASSED: Save/Load/Evict cycle successful"
              << std::endl;
  } else {
    std::cout << "  ❌ TEST FAILED: Issues detected in save/load cycle"
              << std::endl;
  }

  std::cout << "=== VECTORIZED SAVE/LOAD/EVICT CYCLE TEST COMPLETE ===\n"
            << std::endl;
}

void GaussianModel::updateChunkAccess(const torch::Tensor& accessed_chunk_ids) {
  if (accessed_chunk_ids.size(0) == 0) return;

  float current_time = std::chrono::duration<float>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();

  auto chunk_ids_cpu = accessed_chunk_ids.cpu();
  auto chunk_ids_accessor = chunk_ids_cpu.accessor<int64_t, 1>();

  for (int64_t i = 0; i < chunk_ids_cpu.size(0); i++) {
    int64_t chunk_id = chunk_ids_accessor[i];
    chunk_access_times_[chunk_id] = current_time;
  }
}

void GaussianModel::saveAllChunks() {
  std::cout << "\n=== STARTING SAVE OF ALL CHUNKS IN MEMORY ===" << std::endl;

  // Get all chunks that have gaussians in memory
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  if (spatial_chunks.size(0) == 0) {
    std::cout << "No spatial chunks to save" << std::endl;
    return;
  }

  // Categorize chunks
  torch::Tensor loaded_chunks_mask =
      torch::isin(spatial_chunks, chunks_loaded_from_disk_);
  torch::Tensor spillover_chunks_mask =
      torch::isin(spatial_chunks, chunks_on_disk_) & (~loaded_chunks_mask);
  torch::Tensor new_chunks_mask =
      (~torch::isin(spatial_chunks, chunks_on_disk_)) & (~loaded_chunks_mask);

  torch::Tensor loaded_chunks = spatial_chunks.index({loaded_chunks_mask});
  torch::Tensor spillover_chunks =
      spatial_chunks.index({spillover_chunks_mask});
  torch::Tensor new_chunks = spatial_chunks.index({new_chunks_mask});

  std::cout << "Loaded chunks: " << loaded_chunks.size(0) << std::endl;
  std::cout << "Spillover chunks: " << spillover_chunks.size(0) << " (skipping)"
            << std::endl;
  std::cout << "New chunks: " << new_chunks.size(0) << std::endl;

  // Save loaded chunks (preserve existing data)
  if (loaded_chunks.size(0) > 0) {
    saveChunks(loaded_chunks);
  }

  // Save new chunks (preserve new data)
  if (new_chunks.size(0) > 0) {
    saveChunks(new_chunks);
  }

  // Don't save spillover chunks (would overwrite better disk data)

  std::cout << "=== SAVE COMPLETE ===" << std::endl;
}

int64_t GaussianModel::countAllGaussians() {
  std::cout << "[Gaussian Count] Starting lightweight count..." << std::endl;

  int64_t total_count = 0;

  // Count gaussians in memory
  int64_t in_memory_count = xyz_.size(0);
  total_count += in_memory_count;

  std::cout << "[Gaussian Count] Gaussians in memory: " << in_memory_count
            << std::endl;

  // Count gaussians on disk by reading chunk headers only
  int64_t disk_count = 0;
  int disk_chunks_read = 0;

  // Filter chunks_on_disk_ to exclude those already loaded in memory
  // We need to avoid double-counting loaded chunks
  torch::Tensor not_loaded_mask =
      ~torch::isin(chunks_on_disk_, chunks_loaded_from_disk_);
  torch::Tensor disk_only_chunks = chunks_on_disk_.index({not_loaded_mask});

  std::cout << "[Gaussian Count] Checking " << disk_only_chunks.size(0)
            << " disk-only chunks (excluding "
            << chunks_loaded_from_disk_.size(0) << " loaded chunks)"
            << std::endl;

  if (disk_only_chunks.size(0) == 0) {
    std::cout << "[Gaussian Count] No disk-only chunks to check" << std::endl;
    std::cout << "[Gaussian Count] Total gaussians: " << total_count
              << std::endl;
    return total_count;
  }

  // Read headers for disk-only chunks (file operations still need to be
  // sequential)
  auto chunks_cpu = disk_only_chunks.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();

  for (int i = 0; i < chunks_cpu.size(0); ++i) {
    int64_t chunk_id = accessor[i];

    // Read just the header to get point count
    std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

    if (!std::filesystem::exists(chunk_filename)) {
      std::cerr << "[Gaussian Count] Warning: Chunk " << chunk_id
                << " tracked as on-disk but file doesn't exist: "
                << chunk_filename << std::endl;
      continue;
    }

    try {
      std::ifstream file(chunk_filename, std::ios::binary);
      if (!file.is_open()) {
        continue;
      }

      // Read magic and version
      uint32_t magic, version;
      file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      file.read(reinterpret_cast<char*>(&version), sizeof(version));

      if (magic != 0x43484E4B) {  // "CHNK"
        std::cerr << "[Gaussian Count] Invalid file format for chunk "
                  << chunk_id << std::endl;
        file.close();
        continue;
      }

      // Read chunk ID and point count from header
      int64_t stored_chunk_id;
      uint32_t num_points;
      file.read(reinterpret_cast<char*>(&stored_chunk_id),
                sizeof(stored_chunk_id));
      file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

      file.close();

      if (stored_chunk_id == chunk_id) {
        disk_count += num_points;
        disk_chunks_read++;
      } else {
        std::cerr << "[Gaussian Count] Chunk ID mismatch in file "
                  << chunk_filename << ": expected " << chunk_id << ", got "
                  << stored_chunk_id << std::endl;
      }

    } catch (const std::exception& e) {
      std::cerr << "[Gaussian Count] Error reading chunk header for chunk "
                << chunk_id << ": " << e.what() << std::endl;
      continue;
    }
  }

  total_count += disk_count;

  std::cout << "[Gaussian Count] Gaussians on disk: " << disk_count << " (from "
            << disk_chunks_read << " chunks)" << std::endl;
  std::cout << "[Gaussian Count] Total gaussians: " << total_count << std::endl;

  // Optional: Show breakdown for clarity
  std::cout << "[Gaussian Count] Breakdown:" << std::endl;
  std::cout << "  - In memory: " << in_memory_count << std::endl;
  std::cout << "  - Loaded chunks: " << chunks_loaded_from_disk_.size(0)
            << " (already counted in memory)" << std::endl;
  std::cout << "  - Disk-only chunks: " << disk_chunks_read << " with "
            << disk_count << " gaussians" << std::endl;

  return total_count;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianModel::filterPointsByChunkDensity(const torch::Tensor& xyz,
                                          const torch::Tensor& colors,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& opacities,
                                          int min_gaussians_per_chunk) {
  if (min_gaussians_per_chunk <= 1) {
    // No filtering needed
    return std::make_tuple(xyz, colors, scales, opacities);
  }

  // Compute chunk IDs for all input points
  torch::Tensor chunk_ids = computeChunkIds(xyz);

  // Count points per chunk using PyTorch operations
  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(chunk_ids, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  // Create mask for chunks that have enough points
  torch::Tensor valid_chunk_mask = counts >= min_gaussians_per_chunk;

  // Get the chunk IDs that pass the threshold
  torch::Tensor valid_chunk_ids = unique_chunks.index({valid_chunk_mask});

  if (valid_chunk_ids.size(0) == 0) {
    // No chunks have enough points, return empty tensors
    torch::Tensor empty_xyz = torch::empty({0, 3}, xyz.options());
    torch::Tensor empty_colors = torch::empty({0, 3}, colors.options());
    torch::Tensor empty_scales = scales.defined()
                                     ? torch::empty({0, 3}, scales.options())
                                     : torch::Tensor();
    torch::Tensor empty_opacities = torch::empty({0, 1}, opacities.options());
    return std::make_tuple(empty_xyz, empty_colors, empty_scales,
                           empty_opacities);
  }

  // Create mask for points that belong to valid chunks
  torch::Tensor point_valid_mask = torch::isin(chunk_ids, valid_chunk_ids);

  // Filter all tensors using the mask
  torch::Tensor filtered_xyz = xyz.index({point_valid_mask});
  torch::Tensor filtered_colors = colors.index({point_valid_mask});
  torch::Tensor filtered_scales =
      scales.defined() ? scales.index({point_valid_mask}) : torch::Tensor();
  torch::Tensor filtered_opacities = opacities.index({point_valid_mask});

  // Log statistics
  int total_chunks = unique_chunks.size(0);
  int valid_chunks = valid_chunk_ids.size(0);
  int total_points = xyz.size(0);
  int kept_points = filtered_xyz.size(0);

  // std::cout << "[Chunk Density Filter] Chunks: " << valid_chunks << "/"
  //           << total_chunks << " passed (need >= " <<
  //           min_gaussians_per_chunk
  //           << " points)" << std::endl;
  // std::cout << "[Chunk Density Filter] Points: " << kept_points << "/"
  //           << total_points << " kept ("
  //           << (100.0f * kept_points / total_points) << "%)" << std::endl;

  return std::make_tuple(filtered_xyz, filtered_colors, filtered_scales,
                         filtered_opacities);
}

void GaussianModel::initializeEmpty(float spatial_lr_scale) {
  std::cout << "[Gaussian Model] Initializing empty model for loading"
            << std::endl;

  this->spatial_lr_scale_ = spatial_lr_scale;

  // Initialize with empty tensors but correct shapes
  this->xyz_ =
      torch::empty(
          {0, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->features_dc_ =
      torch::empty(
          {0, 1, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->features_rest_ =
      torch::empty(
          {0, (sh_degree_ + 1) * (sh_degree_ + 1) - 1, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->scaling_ =
      torch::empty(
          {0, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->rotation_ =
      torch::empty(
          {0, 4},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->opacity_ =
      torch::empty(
          {0, 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();

  // Initialize auxiliary tensors
  this->exist_since_iter_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
  this->gaussian_chunk_ids_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  this->gaussian_lod_levels_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  // Initialize tensor vectors for optimizer
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Mark as initialized
  is_initialized_ = true;

  std::cout << "[Gaussian Model] Empty model initialized" << std::endl;
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

// Vectorized encoding - much faster than loop
torch::Tensor GaussianModel::encodeChunkCoordsTensor(
    const torch::Tensor& chunk_coords) {
  // chunk_coords: [N, 3] with coordinates roughly in range [-500, 500]

  // For 10km range with 20m chunks: 10000m ÷ 20m = 500 chunks per axis
  // Use 12 bits per coordinate = 4096 range = [-2048, +2047] chunks
  // That's 40km+ range per axis - plenty of headroom
  const int32_t OFFSET = 2048;  // Supports [-2048, +2047] range

  auto x = chunk_coords.index({torch::indexing::Slice(), 0}) + OFFSET;
  auto y = chunk_coords.index({torch::indexing::Slice(), 1}) + OFFSET;
  auto z = chunk_coords.index({torch::indexing::Slice(), 2}) + OFFSET;

  // 12 bits per coordinate = 36 total bits
  torch::Tensor encoded = x * (1 << 24) + y * (1 << 12) + z;

  return encoded;  // Max value: ~8 billion instead of 16 trillion
}

// Vectorized decoding
torch::Tensor GaussianModel::decodeChunkCoordsTensor(
    const torch::Tensor& encoded_ids) {
  const int32_t OFFSET = 2048;

  torch::Tensor z = (encoded_ids % (1 << 12)) - OFFSET;
  torch::Tensor y = ((encoded_ids / (1 << 12)) % (1 << 12)) - OFFSET;
  torch::Tensor x = (encoded_ids / (1 << 24)) - OFFSET;

  return torch::stack({x, y, z}, /*dim=*/1);
}

torch::Tensor GaussianModel::chunkCoordVectorToTensor(
    const std::vector<ChunkCoord>& coords) {
  if (coords.empty()) {
    return torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  }

  // Use from_blob for zero-copy conversion (ChunkCoord is POD with int64_t
  // x,y,z)
  torch::Tensor coord_tensor =
      torch::from_blob(const_cast<ChunkCoord*>(coords.data()),
                       {static_cast<int64_t>(coords.size()), 3},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();  // Clone to own the memory

  return coord_tensor.to(device_type_);
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
        std::cout << "[File Deletion] Deleted " << chunk_filename << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "[File Deletion] Failed to delete " << chunk_filename
                  << ": " << e.what() << std::endl;
      }
    }
  }

  if (files_deleted > 0) {
    std::cout << "[File Deletion] Deleted " << files_deleted
              << " chunk files from disk" << std::endl;
  }
}

torch::Tensor GaussianModel::assignLoDByScale(
    const torch::Tensor& scale_magnitudes) {
  int n_gaussians = scale_magnitudes.size(0);
  torch::Tensor lod_levels = torch::zeros(
      {n_gaussians},
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  // Debug: Print thresholds (only once per session)
  static bool printed_thresholds = false;
  if (!printed_thresholds) {
    // std::cout << "[LoD Debug] Scale thresholds - LoD0: >="
    //           << base_scale_threshold_ << ", LoD1: [" <<
    //           detail_scale_threshold_
    //           << " - " << base_scale_threshold_ << "), LoD2: <"
    //           << detail_scale_threshold_ << std::endl;
    printed_thresholds = true;
  }

  // LoD 0: Large gaussians (base structure)
  torch::Tensor large_mask = scale_magnitudes >= base_scale_threshold_;
  lod_levels.masked_fill_(large_mask, 0);

  // LoD 1: Medium gaussians (details)
  torch::Tensor medium_mask = (scale_magnitudes >= detail_scale_threshold_) &
                              (scale_magnitudes < base_scale_threshold_);
  lod_levels.masked_fill_(medium_mask, 1);

  // LoD 2: Small gaussians (fine details) - default assignment (already 0)
  torch::Tensor small_mask = scale_magnitudes < detail_scale_threshold_;
  lod_levels.masked_fill_(small_mask, 2);

  return lod_levels;
}

torch::Tensor GaussianModel::assignLoDByDensity(
    const torch::Tensor& nearest_distances) {
  int n_gaussians = nearest_distances.size(0);
  torch::Tensor lod_levels = torch::zeros(
      {n_gaussians},
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  // Calculate density thresholds based on distance distribution
  // Larger distances = sparser areas = should be LoD 0/1 (visible from far)
  // Smaller distances = denser areas = can be LoD 2 (only visible close up)

  float mean_dist = nearest_distances.mean().item<float>();
  float dense_threshold = mean_dist * 0.5f;   // Bottom 25% (very dense areas)
  float sparse_threshold = mean_dist * 1.5f;  // Top 25% (sparse areas)

  // Debug: Print density thresholds (only once per session)
  static bool printed_density_thresholds = false;
  if (!printed_density_thresholds) {
    // std::cout << "[LoD Debug] Density thresholds - LoD0: >=" <<
    // sparse_threshold
    //           << ", LoD1: [" << dense_threshold << " - " <<
    //           sparse_threshold
    //           << "), LoD2: <" << dense_threshold << " (mean dist: " <<
    //           mean_dist
    //           << ")" << std::endl;
    printed_density_thresholds = true;
  }

  // LoD 0: Sparse areas (large distances between points) - visible from far
  torch::Tensor sparse_mask = nearest_distances >= sparse_threshold;
  lod_levels.masked_fill_(sparse_mask, 0);

  // LoD 1: Medium density areas
  torch::Tensor medium_mask = (nearest_distances >= dense_threshold) &
                              (nearest_distances < sparse_threshold);
  lod_levels.masked_fill_(medium_mask, 1);

  // LoD 2: Dense areas (small distances between points) - only visible close
  // up
  torch::Tensor dense_mask = nearest_distances < dense_threshold;
  lod_levels.masked_fill_(dense_mask, 2);

  return lod_levels;
}

torch::Tensor GaussianModel::selectCumulativeLoD(
    const torch::Tensor& visible_gaussian_mask,
    const torch::Tensor& camera_position) {
  // Get visible indices and their positions
  torch::Tensor visible_indices = torch::where(visible_gaussian_mask)[0];
  if (visible_indices.size(0) == 0) {
    return visible_gaussian_mask;
  }

  torch::Tensor visible_positions = xyz_.index({visible_indices});

  // Compute distances for visible gaussians
  torch::Tensor distances = torch::norm(
      visible_positions - camera_position.unsqueeze(0), /*p=*/2, /*dim=*/1);

  // Logarithmic LoD level calculation
  float d_max = 8.0f * chunk_size_;
  torch::Tensor required_lod = torch::clamp(
      torch::log2(d_max / torch::clamp_min(distances, 0.1f)), 0.0f, 2.0f);

  // Get pre-assigned LoD levels for visible gaussians
  torch::Tensor visible_lod_levels =
      gaussian_lod_levels_.index({visible_indices});

  // Debug: Print distance and LoD statistics
  float min_dist = distances.min().item<float>();
  float max_dist = distances.max().item<float>();
  float mean_required_lod = required_lod.mean().item<float>();

  auto visible_lod_counts =
      torch::bincount(visible_lod_levels, torch::Tensor(), 3);
  // std::cout << "[LoD Debug] Distance range: " << min_dist << " - " <<
  // max_dist
  //           << ", Mean required LoD: " << mean_required_lod
  //           << ", Available LoDs - L0: " <<
  //           visible_lod_counts[0].item<int>()
  //           << ", L1: " << visible_lod_counts[1].item<int>()
  //           << ", L2: " << visible_lod_counts[2].item<int>() << std::endl;

  // Cumulative selection: include gaussian if its assigned LoD >= required
  // LoD Note: We invert the logic since LoD 0 = large (base), LoD 2 = small
  // (fine)
  torch::Tensor lod_mask = visible_lod_levels.to(torch::kFloat) <= required_lod;

  // Create final mask
  torch::Tensor final_mask = torch::zeros_like(visible_gaussian_mask);
  final_mask.index_put_({visible_indices}, lod_mask);

  return final_mask;
}