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
  vertices_ = tensors[0];
  features_dc_ = tensors[1];
  features_rest_ = tensors[2];
  vertex_weight_ = tensors[3];
  TRIANGLE_MODEL_TENSORS_TO_VEC
}

void TriangleModel::resetVertexWeight() {
  torch::Tensor target = torch::min(
      getVertexWeightActivation(),
      torch::ones_like(getVertexWeightActivation()) * 0.01);
  torch::Tensor weights_new = inverseVertexWeightActivation(target);
  torch::Tensor optimizable_tensors =
      replaceTensorToOptimizer(weights_new, 3);  // vertex_weight
  vertex_weight_ = optimizable_tensors;
  Tensor_vec_vertex_weight_ = {vertex_weight_};
}

void TriangleModel::resetVertexWeightForMask(const torch::Tensor& triangle_mask) {
  torch::NoGradGuard no_grad;

  // Map triangle mask to vertex mask
  const int64_t V = vertices_.size(0);
  torch::Tensor vertex_mask = torch::zeros(
      {V}, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  vertex_mask.index_put_(
      {triangle_indices_.index({triangle_mask}).flatten().to(torch::kLong)},
      true);

  int num_reset = torch::sum(vertex_mask).item<int>();
  std::cout << "[Vertex Weight Reset] Resetting vertex weight for " << num_reset
            << " vertices" << std::endl;

  torch::Tensor current_weight_activated = getVertexWeightActivation();

  torch::Tensor target_weight =
      torch::min(current_weight_activated,
                 torch::ones_like(current_weight_activated) * 0.05f);
  torch::Tensor new_weight_values =
      inverseVertexWeightActivation(target_weight);

  vertex_weight_.index_put_({vertex_mask},
                      new_weight_values.index({vertex_mask}));

  std::cout << "[Vertex Weight Reset] Reset complete - max="
            << torch::sigmoid(vertex_weight_).max().item<float>()
            << ", min=" << torch::sigmoid(vertex_weight_).min().item<float>()
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

  // Map triangle mask to vertex mask
  const int64_t V = vertices_.size(0);
  torch::Tensor vertex_mask = torch::zeros(
      {V}, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  vertex_mask.index_put_(
      {triangle_indices_.index({triangle_mask}).flatten().to(torch::kLong)},
      true);

  int num_reset = torch::sum(vertex_mask).item<int>();
  std::cout << "[Optimizer Reset] Resetting position LR and Adam states for "
            << num_reset << " vertices" << std::endl;

  position_lrs_.index_put_({vertex_mask}, position_lr_init_);

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

  // Expand vertex mask to match [V, 3]
  torch::Tensor vert_mask_3d = vertex_mask.unsqueeze(1).expand({-1, 3});
  exp_avg.index_put_({vert_mask_3d}, 0.0f);
  exp_avg_sq.index_put_({vert_mask_3d}, 0.0f);

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
  auto valid_triangles_mask = ~mask;

  triangle_indices_ = triangle_indices_.index({valid_triangles_mask});
  exist_since_iter_ = exist_since_iter_.index({valid_triangles_mask});
  triangle_chunk_ids_ = triangle_chunk_ids_.index({valid_triangles_mask});
  triangle_ids_ = triangle_ids_.index({valid_triangles_mask});

  gcUnreferencedVertices();
}

void TriangleModel::gcUnreferencedVertices() {
  const int64_t V = vertices_.size(0);
  if (V == 0 || triangle_indices_.size(0) == 0) return;

  torch::Tensor referenced = torch::zeros(
      {V}, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  referenced.index_put_(
      {triangle_indices_.flatten().to(torch::kLong)}, true);

  if (referenced.all().item<bool>()) return;

  // Build old→new vertex ID mapping
  torch::Tensor new_ids = torch::full(
      {V}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_type_));
  torch::Tensor kept = torch::nonzero(referenced).squeeze(1);
  new_ids.index_put_({kept}, torch::arange(kept.size(0),
      torch::TensorOptions().dtype(torch::kLong).device(device_type_)));

  triangle_indices_ =
      new_ids.index({triangle_indices_.to(torch::kLong)}).to(torch::kInt32);

  chunk_delaunay_.clear();
  chunk_delaunay_vert_map_.clear();

  pruneVertexData(referenced);
}

void TriangleModel::pruneVertexData(const torch::Tensor& vertex_keep_mask) {
  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();
  std::vector<torch::Tensor> optimized_tensors(kNumParamGroups);

  for (int i = 0; i < kNumParamGroups; ++i) {
    auto& group = param_groups[i];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();

    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(stored_state.exp_avg().index({vertex_keep_mask}));
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index({vertex_keep_mask}));

      state.erase(key);
      param = param.index({vertex_keep_mask}).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
    } else {
      param = param.index({vertex_keep_mask}).requires_grad_();
    }
    optimized_tensors[i] = param;
  }

  assignOptimizedTensors(optimized_tensors);
  position_lrs_ = position_lrs_.index({vertex_keep_mask});
}

void TriangleModel::densificationPostfix(
    torch::Tensor& new_vertices,
    torch::Tensor& new_triangle_indices,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_vertex_weight,
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
      new_vertices, new_features_dc, new_features_rest, new_vertex_weight};

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
    auto& group = param_groups[group_idx];
    assert(group.params().size() == 1);
    auto& extension_tensor = extension_tensors[group_idx];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();

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

  triangle_indices_ = torch::cat({triangle_indices_, new_triangle_indices}, 0);
  exist_since_iter_ = torch::cat({exist_since_iter_, new_exist_since_iter}, 0);
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  triangle_chunk_ids_ = torch::cat({triangle_chunk_ids_, new_chunk_ids}, 0);
  triangle_ids_ = torch::cat({triangle_ids_, new_triangle_ids}, 0);
  refreshActiveChunkIds();
}

void TriangleModel::pruneLowWeightTriangles(
    std::shared_ptr<TriangleKeyframe> pkf,
    const torch::Tensor& visible_triangle_mask,
    const torch::Tensor& full_model_scaling) {
  torch::NoGradGuard no_grad;

  torch::Tensor visible_indices = torch::where(visible_triangle_mask)[0];

  // Per-vertex weights [V, 1] → gather per-triangle min weight
  torch::Tensor vert_weights = getVertexWeightActivation();  // [V, 1]
  torch::Tensor vis_tri_idx =
      triangle_indices_.index({visible_indices}).to(torch::kLong);  // [T_vis, 3]
  // [T_vis, 3] — weight of each corner vertex
  torch::Tensor tri_vert_weights =
      vert_weights.squeeze(1).index({vis_tri_idx});
  // Min weight across the 3 vertices → [T_vis]
  torch::Tensor min_weights = std::get<0>(tri_vert_weights.min(/*dim=*/1));

  torch::Tensor screen_size = full_model_scaling.index({visible_indices});

  const float kMaxScreenSize = 0.5f * static_cast<float>(pkf->image_width_);

  torch::Tensor valid_mask = (min_weights > 0.05f) &
                             (screen_size < kMaxScreenSize);

  torch::Tensor full_model_prune_mask = torch::zeros(
      {triangle_indices_.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  full_model_prune_mask.index_put_({visible_indices}, ~valid_mask);

  prunePoints(full_model_prune_mask);
}

void TriangleModel::pruneDepthInconsistent(
    std::shared_ptr<TriangleKeyframe> pkf,
    const torch::Tensor& visible_triangle_mask,
    const torch::Tensor& gt_inv_depth,
    const torch::Tensor& view_matrix,
    float threshold,
    const torch::Tensor& depth_confidence,
    const torch::Tensor& rendered_inv_depth) {
  torch::NoGradGuard no_grad;

  torch::Tensor visible_indices = torch::where(visible_triangle_mask)[0];
  auto vis_tri_idx = triangle_indices_.index({visible_indices}).to(torch::kLong);
  auto vis_vert_ids = std::get<0>(torch::_unique(vis_tri_idx.flatten()));

  torch::Tensor verts = vertices_.index({vis_vert_ids});
  float fx = pkf->intr_[0], fy = pkf->intr_[1];
  float cx = pkf->intr_[2], cy = pkf->intr_[3];
  int H = pkf->image_height_, W = pkf->image_width_;

  torch::Tensor ones = torch::ones({verts.size(0), 1}, verts.options());
  torch::Tensor verts_h = torch::cat({verts, ones}, 1);
  torch::Tensor verts_view = verts_h.mm(view_matrix);
  torch::Tensor vz = verts_view.select(1, 2);

  torch::Tensor px = (verts_view.select(1, 0) / vz) * fx + cx;
  torch::Tensor py = (verts_view.select(1, 1) / vz) * fy + cy;

  torch::Tensor inside = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (vz > 0);
  if (!inside.any().item<bool>()) return;

  torch::Tensor xi = torch::clamp(torch::round(px.index({inside})).to(torch::kLong), 0, W - 1);
  torch::Tensor yi = torch::clamp(torch::round(py.index({inside})).to(torch::kLong), 0, H - 1);
  torch::Tensor vz_k = vz.index({inside});

  // Skip occluded vertices: if vertex is behind the rendered surface, another
  // triangle covers it and the GT depth comparison would be against the wrong surface.
  if (rendered_inv_depth.defined()) {
    torch::Tensor rend_inv = rendered_inv_depth.squeeze(0).index({yi, xi});
    torch::Tensor rend_depth = 1.0f / rend_inv.clamp_min(1e-8f);
    torch::Tensor not_occluded = vz_k < rend_depth * 1.05f;
    xi = xi.index({not_occluded});
    yi = yi.index({not_occluded});
    vz_k = vz_k.index({not_occluded});
    // Update inside to reflect the non-occluded subset
    torch::Tensor inside_idx = torch::where(inside)[0];
    inside.fill_(false);
    inside.index_put_({inside_idx.index({not_occluded})}, true);
    if (xi.size(0) == 0) return;
  }

  torch::Tensor sampled_inv = gt_inv_depth.squeeze(0).index({yi, xi});
  torch::Tensor valid = (sampled_inv > 0) & torch::isfinite(sampled_inv);
  if (!valid.any().item<bool>()) return;

  torch::Tensor ref_depth = 1.0f / sampled_inv.index({valid});
  torch::Tensor v_depth = vz_k.index({valid});
  torch::Tensor depth_diff = ref_depth - v_depth;

  torch::Tensor per_vertex_threshold;
  if (depth_confidence.defined()) {
    torch::Tensor conf_map = depth_confidence.squeeze(0).squeeze(0);
    torch::Tensor xi_valid = xi.index({valid});
    torch::Tensor yi_valid = yi.index({valid});
    torch::Tensor sampled_conf = conf_map.index({yi_valid, xi_valid}).clamp(0.05f, 1.0f);
    per_vertex_threshold = threshold / sampled_conf;
  } else {
    per_vertex_threshold = torch::full_like(depth_diff, threshold);
  }
  torch::Tensor too_far = torch::abs(depth_diff) > per_vertex_threshold;

  int n_too_far = too_far.sum().item<int>();
  int n_valid = valid.sum().item<int>();
  std::cout << "[DepthPrune] verts checked=" << n_valid
            << " bad=" << n_too_far
            << " max_in_front=" << depth_diff.max().item<float>()
            << " median=" << depth_diff.median().item<float>()
            << " threshold=" << threshold
            << std::endl;

  if (!too_far.any().item<bool>()) return;

  // Map bad vertices back to full vertex ids
  torch::Tensor inside_indices = torch::where(inside)[0];
  torch::Tensor valid_indices = torch::where(valid)[0];
  torch::Tensor bad_in_inside = inside_indices.index({valid_indices.index({torch::where(too_far)[0]})});
  torch::Tensor bad_vert_ids = vis_vert_ids.index({bad_in_inside});

  torch::Tensor bad_verts = torch::zeros(
      {vertices_.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  bad_verts.index_put_({bad_vert_ids}, true);

  torch::Tensor tri_v0 = bad_verts.index({triangle_indices_.select(1, 0).to(torch::kLong)});
  torch::Tensor tri_v1 = bad_verts.index({triangle_indices_.select(1, 1).to(torch::kLong)});
  torch::Tensor tri_v2 = bad_verts.index({triangle_indices_.select(1, 2).to(torch::kLong)});
  torch::Tensor prune_mask = tri_v0 | tri_v1 | tri_v2;

  int n_pruned = prune_mask.sum().item<int>();
  if (n_pruned > 0) {
    std::cout << "[DepthPrune] Removing " << n_pruned << " triangles" << std::endl;
    prunePoints(prune_mask);
  }
}

void TriangleModel::subdivideMidpoint(const torch::Tensor& triangle_mask,
                                      int current_iter) {
  torch::NoGradGuard no_grad;

  int64_t S = triangle_mask.sum().item<int64_t>();
  if (S == 0) return;

  torch::Tensor selected_indices = torch::where(triangle_mask)[0];
  torch::Tensor selected_tri =
      triangle_indices_.index({selected_indices}).to(torch::kLong);

  // Collect edges [0,1], [0,2], [1,2] and sort each pair
  torch::Tensor col0 = selected_tri.select(1, 0);
  torch::Tensor col1 = selected_tri.select(1, 1);
  torch::Tensor col2 = selected_tri.select(1, 2);
  torch::Tensor edges = torch::cat({
      torch::stack({col0, col1}, 1),
      torch::stack({col0, col2}, 1),
      torch::stack({col1, col2}, 1)
  }, 0);
  edges = std::get<0>(torch::sort(edges, 1));

  // Pack sorted edges into single int64 keys for deduplication
  constexpr int64_t kEdgeKeyMul = 10000000LL;
  torch::Tensor edge_keys =
      edges.select(1, 0) * kEdgeKeyMul + edges.select(1, 1);
  auto [unique_keys, inverse_indices, counts] =
      torch::_unique2(edge_keys, /*sorted=*/true, /*return_inverse=*/true,
                       /*return_counts=*/true);
  int64_t M = unique_keys.size(0);
  int64_t new_vertex_base = vertices_.size(0);

  // Unpack unique edge keys back to vertex pairs
  torch::Tensor u = torch::floor_divide(unique_keys, kEdgeKeyMul);
  torch::Tensor v = unique_keys - u * kEdgeKeyMul;

  // Build edge_to_midpoint map on CPU
  auto uk_cpu = unique_keys.cpu();
  auto uk_acc = uk_cpu.accessor<int64_t, 1>();
  std::unordered_map<int64_t, int64_t> edge_to_midpoint;
  for (int64_t i = 0; i < M; i++) {
    edge_to_midpoint[uk_acc[i]] = new_vertex_base + i;
  }
  torch::Tensor new_vertices = (vertices_.index({u}) + vertices_.index({v})) / 2.0f;
  torch::Tensor new_features_dc =
      (features_dc_.index({u}) + features_dc_.index({v})) / 2.0f;
  torch::Tensor new_features_rest =
      (features_rest_.index({u}) + features_rest_.index({v})) / 2.0f;

  // Average weight in activated space, then invert
  torch::Tensor wu = getVertexWeightActivation().index({u});
  torch::Tensor wv = getVertexWeightActivation().index({v});
  torch::Tensor avg_weight = ((wu + wv) / 2.0f).clamp(1e-6f, 1.0f - 1e-6f);
  torch::Tensor new_vertex_weight = inverseVertexWeightActivation(avg_weight);

  // Build new triangles on CPU
  auto sel_cpu = selected_tri.cpu();
  auto sel_acc = sel_cpu.accessor<int64_t, 2>();
  std::vector<int32_t> new_tri_data;
  new_tri_data.reserve(S * 4 * 3);

  auto lookup = [&](int64_t a, int64_t b) -> int64_t {
    if (a > b) std::swap(a, b);
    return edge_to_midpoint[a * kEdgeKeyMul + b];
  };

  for (int64_t i = 0; i < S; i++) {
    int64_t a = sel_acc[i][0], b = sel_acc[i][1], c = sel_acc[i][2];
    int64_t m_ab = lookup(a, b);
    int64_t m_ac = lookup(a, c);
    int64_t m_bc = lookup(b, c);

    new_tri_data.insert(new_tri_data.end(),
        {(int32_t)a, (int32_t)m_ab, (int32_t)m_ac});
    new_tri_data.insert(new_tri_data.end(),
        {(int32_t)b, (int32_t)m_ab, (int32_t)m_bc});
    new_tri_data.insert(new_tri_data.end(),
        {(int32_t)c, (int32_t)m_ac, (int32_t)m_bc});
    new_tri_data.insert(new_tri_data.end(),
        {(int32_t)m_ab, (int32_t)m_bc, (int32_t)m_ac});
  }

  int64_t num_new_tri = S * 4;
  torch::Tensor new_triangle_indices =
      torch::from_blob(new_tri_data.data(), {num_new_tri, 3},
                        torch::kInt32).to(device_type_).clone();

  // Inherit chunk_ids from parent triangles (each parent → 4 children)
  torch::Tensor parent_chunk_ids =
      triangle_chunk_ids_.index({selected_indices});
  torch::Tensor new_chunk_ids =
      parent_chunk_ids.unsqueeze(1).expand({S, 4}).reshape({num_new_tri});

  torch::Tensor new_exist_since =
      torch::full({num_new_tri}, current_iter,
                  torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  torch::Tensor new_position_lrs =
      torch::full({M}, position_lr_init_,
                  torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  torch::Tensor new_triangle_ids =
      torch::arange(next_triangle_id_, next_triangle_id_ + num_new_tri,
                    torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  next_triangle_id_ += num_new_tri;

  // Empty vectors → densificationPostfix will use zeros for optimizer state
  std::vector<torch::Tensor> empty_exp_avg, empty_exp_avg_sq;
  std::vector<int64_t> empty_steps;

  densificationPostfix(new_vertices, new_triangle_indices, new_features_dc,
                       new_features_rest, new_vertex_weight, new_exist_since,
                       new_chunk_ids, new_position_lrs, new_triangle_ids,
                       empty_exp_avg, empty_exp_avg_sq, empty_steps);

  // Remove original triangles (extend mask for newly appended triangles)
  torch::Tensor prune_mask = torch::zeros(
      {triangle_indices_.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  prune_mask.slice(0, 0, triangle_mask.size(0)).copy_(triangle_mask);
  prunePoints(prune_mask);
}

void TriangleModel::deleteSparseChunks(int min_triangles_per_chunk) {
  torch::NoGradGuard no_grad;

  if (!is_initialized_ || triangle_indices_.size(0) == 0) {
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
  chunk_vertex_counts_ = chunk_vertex_counts_.index({keep_disk_mask});

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