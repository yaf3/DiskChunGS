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

TriangleModel::TriangleModel(const TriangleModelParams& model_params,
                             std::string storage_base_path,
                             float chunk_size)
    : storage_base_path_(storage_base_path),
      chunk_size_(chunk_size),
      max_triangles_in_memory_(model_params.max_triangles_in_memory_),
      sh_degree_(0),
      spatial_lr_scale_(1.0),
      position_lr_init_(0.00005),
      position_lr_decay_(0.99998),
      local_iteration_(0),
      triangle_visibility_cache_(chunk_size) {
  this->sh_degree_ = model_params.sh_degree_;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  TRIANGLE_MODEL_INIT_TENSORS(this->device_type_)

  chunks_on_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunks_loaded_from_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunk_triangle_counts_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  triangle_ids_ = torch::empty(
      0, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  std::cout << "[TriangleModel] Initialized with storage path: "
            << storage_base_path_ << " and chunk size: " << chunk_size_
            << std::endl;
}

torch::Tensor TriangleModel::getTrianglesPoints() {
  return this->triangles_points_;
}

torch::Tensor TriangleModel::getSigmaActivation() {
  return 0.01f + torch::exp(this->sigma_);
}

// Returns centroid of 3 vertices: triangles_points_ [N,3,3] → [N,3]
torch::Tensor TriangleModel::getXYZ() {
  return this->triangles_points_.mean(/*dim=*/1);
}

torch::Tensor TriangleModel::getFeatures() {
  return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()},
                    /*dim=*/1);
}

torch::Tensor TriangleModel::getOpacityActivation() {
  return torch::sigmoid(this->opacity_);
}


void TriangleModel::applyScaledTransformation(const float s,
                                              const Sophus::SE3f T) {
  torch::NoGradGuard no_grad;

  torch::Tensor T_tensor =
      tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_)
          .transpose(0, 1);

  // Transform all 3 vertices: triangles_points_ [N,3,3]
  // Process each vertex column: reshape to [N*3, 3], transform, reshape back
  auto N = this->triangles_points_.size(0);
  auto pts_flat = this->triangles_points_.reshape({N * 3, 3});
  pts_flat *= s;
  transformPoints(pts_flat, T_tensor);
  this->triangles_points_ = pts_flat.reshape({N, 3, 3});

  // sigma in log-space: sigma_activated = 0.01 + exp(sigma_)
  // Scaling by s: new_activated = s * old_activated ≈ exp(log(s) + sigma_)
  // So sigma_ += log(s) (approximate, ignoring the 0.01 offset)
  this->sigma_ = this->sigma_ + std::log(s);

  scaledTransformationPostfix(this->triangles_points_, this->sigma_);
}

void TriangleModel::scaledTransformationPostfix(
    torch::Tensor& new_triangles_points,
    torch::Tensor& new_sigma) {
  // param_groups[0] = triangles_points_
  torch::Tensor optimizable_tri_pts =
      this->replaceTensorToOptimizer(new_triangles_points, 0);
  // param_groups[4] = sigma_
  torch::Tensor optimizable_sigma =
      this->replaceTensorToOptimizer(new_sigma, 4);

  this->triangles_points_ = optimizable_tri_pts;
  this->sigma_ = optimizable_sigma;

  this->Tensor_vec_triangles_points_ = {this->triangles_points_};
  this->Tensor_vec_sigma_ = {this->sigma_};
}

void TriangleModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_transformed_flags,
    const torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale) {
  torch::NoGradGuard no_grad;

  // Use centroids for visibility detection; dummy rots (not used anymore)
  torch::Tensor centroids = this->getXYZ();  // [N, 3]
  torch::Tensor dummy_rots = torch::zeros(
      {centroids.size(0), 4},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  dummy_rots.index({torch::indexing::Slice(), 0}) = 1.0f;  // identity quaternion

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  // scaleAndTransformThenMarkVisiblePoints transforms centroids in-place
  // and marks which points are visible+unstable.
  scaleAndTransformThenMarkVisiblePoints(
      centroids, dummy_rots, point_transformed_flags, point_unstable_flags,
      diff_pose, kf_world_view_transform, kf_full_proj_transform,
      num_transformed, scale);

  if (num_transformed > 0) {
    // Apply the same diff_pose transform to all 3 vertices of each triangle.
    // The transform applied by scaleAndTransformThenMarkVisiblePoints is:
    //   new_pt = scale * diff_pose * pt
    // We replicate this for all vertices of transformed triangles.
    torch::Tensor transformed_mask = point_transformed_flags.to(torch::kBool);
    auto N_total = this->triangles_points_.size(0);
    auto pts_flat = this->triangles_points_.reshape({N_total * 3, 3});

    // Build index: for each transformed triangle, transform all 3 vertices
    torch::Tensor tri_indices = torch::where(transformed_mask)[0];  // [K]
    torch::Tensor vert_indices = (tri_indices.unsqueeze(1) * 3 +
        torch::arange(3, tri_indices.options()).unsqueeze(0))
        .reshape({-1});  // [K*3]

    auto selected = pts_flat.index({vert_indices});  // [K*3, 3]
    selected *= scale;
    // diff_pose is a [4,4] world-view transform; apply it as a linear transform
    auto pts_h = torch::cat({selected,
        torch::ones({selected.size(0), 1}, selected.options())}, /*dim=*/1);
    auto transformed = pts_h.matmul(diff_pose);
    pts_flat.index_put_({vert_indices}, transformed.slice(1, 0, 3));

    this->triangles_points_ = pts_flat.reshape({N_total, 3, 3});

    // Postfix: replace in optimizer
    // param_groups[0] = triangles_points_
    torch::Tensor optimizable_tri_pts =
        this->replaceTensorToOptimizer(this->triangles_points_, 0);
    this->triangles_points_ = optimizable_tri_pts;
    this->Tensor_vec_triangles_points_ = {this->triangles_points_};
  }
}

void TriangleModel::addPoints(const torch::Tensor& new_xyz,
                              const torch::Tensor& new_colors,
                              const torch::Tensor& new_scales,
                              const torch::Tensor& new_opacities,
                              int iteration,
                              float spatial_lr_scale) {
  torch::NoGradGuard no_grad;

  auto [filtered_xyz, filtered_colors, filtered_scales, filtered_opacities] =
      filterPointsByChunkDensity(new_xyz, new_colors, new_scales, new_opacities,
                                 new_triangle_chunk_density_);

  if (filtered_xyz.size(0) == 0) {
    std::cout
        << "[Triangle Model] Warning: No points passed chunk density filter"
        << std::endl;
    return;
  }

  // Only care about existing disk chunks that need loading
  torch::Tensor affected_chunks =
      std::get<0>(torch::_unique2(computeChunkIds(filtered_xyz, chunk_size_)));
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
                         filtered_opacities, iteration);
  } else {
    // Subsequent calls - append to existing model
    appendPoints(filtered_xyz, filtered_colors, filtered_scales,
                 filtered_opacities, iteration);
  }
}

// Generate 3 triangle vertices per input point using Fibonacci sphere directions
// and per-point random rotations.
// xyz [N,3], radii [N] → triangles_points [N,3,3]
static torch::Tensor generateTriangleVertices(const torch::Tensor& xyz,
                                              const torch::Tensor& radii) {
  const int64_t N = xyz.size(0);
  auto opts = xyz.options().device(torch::kCPU).dtype(torch::kFloat32);

  auto xyz_cpu = xyz.cpu().to(torch::kFloat32);
  auto radii_cpu = radii.cpu().to(torch::kFloat32);

  // 3 Fibonacci sphere base directions
  // i={0,1,2}: z=1-2*i/2={1,0,-1}, theta=PI*(3-sqrt(5))*i
  const float golden_angle = static_cast<float>(M_PI) * (3.0f - std::sqrt(5.0f));
  float base_dirs[3][3];
  for (int i = 0; i < 3; ++i) {
    float z = 1.0f - 2.0f * i / 2.0f;
    float theta = golden_angle * i;
    float r_xy = std::sqrt(std::max(0.0f, 1.0f - z * z));
    base_dirs[i][0] = r_xy * std::cos(theta);
    base_dirs[i][1] = r_xy * std::sin(theta);
    base_dirs[i][2] = z;
  }

  auto tri_pts = torch::zeros({N, 3, 3}, opts);
  auto xyz_acc = xyz_cpu.accessor<float, 2>();
  auto radii_acc = radii_cpu.accessor<float, 1>();
  auto tri_acc = tri_pts.accessor<float, 3>();

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int64_t i = 0; i < N; ++i) {
    float radius = std::max(1e-4f, radii_acc[i]);  // ensure positive

    // Random rotation matrix via Rodrigues formula
    float ax = dist(rng) * 2.0f - 1.0f;
    float ay = dist(rng) * 2.0f - 1.0f;
    float az = dist(rng) * 2.0f - 1.0f;
    float len = std::sqrt(ax*ax + ay*ay + az*az);
    if (len < 1e-6f) { ax = 1.0f; ay = 0.0f; az = 0.0f; len = 1.0f; }
    ax /= len; ay /= len; az /= len;
    float angle = dist(rng) * static_cast<float>(2.0 * M_PI);
    float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    // R = [[t*ax*ax+c, t*ax*ay-s*az, t*ax*az+s*ay],
    //      [t*ax*ay+s*az, t*ay*ay+c, t*ay*az-s*ax],
    //      [t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c]]
    float R[3][3] = {
      {t*ax*ax+c,    t*ax*ay-s*az, t*ax*az+s*ay},
      {t*ax*ay+s*az, t*ay*ay+c,    t*ay*az-s*ax},
      {t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c   }
    };

    for (int k = 0; k < 3; ++k) {
      // rotated_dir = R @ base_dirs[k]
      float dx = R[0][0]*base_dirs[k][0] + R[0][1]*base_dirs[k][1] + R[0][2]*base_dirs[k][2];
      float dy = R[1][0]*base_dirs[k][0] + R[1][1]*base_dirs[k][1] + R[1][2]*base_dirs[k][2];
      float dz = R[2][0]*base_dirs[k][0] + R[2][1]*base_dirs[k][1] + R[2][2]*base_dirs[k][2];
      tri_acc[i][k][0] = xyz_acc[i][0] + radius * dx;
      tri_acc[i][k][1] = xyz_acc[i][1] + radius * dy;
      tri_acc[i][k][2] = xyz_acc[i][2] + radius * dz;
    }
  }

  return tri_pts;
}

void TriangleModel::initializeFromPoints(const torch::Tensor& initial_xyz,
                                         const torch::Tensor& initial_colors,
                                         const torch::Tensor& initial_scales,
                                         const torch::Tensor& initial_opacities,
                                         int iteration) {
  torch::NoGradGuard no_grad;
  std::cout << "[Triangle Model] Initializing from points: "
            << initial_xyz.sizes() << std::endl;

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

  torch::Tensor opacities = initial_opacities;

  this->exist_since_iter_ = torch::full(
      {initial_xyz.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  // Compute per-point radius from initial scales (mean across dims)
  // initial_scales is in log-space; exp() gives actual metric scale
  torch::Tensor radii = torch::exp(initial_scales).mean(1).clamp_min(1e-4f);  // [N]

  // Generate 3 triangle vertices per input point
  torch::Tensor tri_pts = generateTriangleVertices(initial_xyz, radii);

  // Sigma init (inverse of 0.01 + exp activation): log(radius - 0.01)
  torch::Tensor sigma_init =
      torch::log((radii - 0.01f).clamp_min(1e-5f)).unsqueeze(1);  // [N, 1]

  this->triangles_points_ =
      tri_pts.to(device_type_).contiguous().requires_grad_();
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
  this->sigma_ = sigma_init.to(device_type_).requires_grad_();
  this->opacity_ = opacities.requires_grad_();

  triangle_chunk_ids_ = computeChunkIds(initial_xyz, chunk_size_);

  triangle_ids_ = torch::arange(
      next_triangle_id_, next_triangle_id_ + initial_xyz.size(0),
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  next_triangle_id_ += initial_xyz.size(0);

  TRIANGLE_MODEL_TENSORS_TO_VEC

  is_initialized_ = true;
}

void TriangleModel::appendPoints(const torch::Tensor& new_xyzs,
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

  // Compute per-point radius and generate triangle vertices
  // new_scales is in log-space; exp() gives actual metric scale
  torch::Tensor radii = torch::exp(new_scales).mean(1).clamp_min(1e-4f);  // [M]
  torch::Tensor new_tri_pts =
      generateTriangleVertices(new_xyzs, radii).to(device_type_).contiguous();
  torch::Tensor new_sigma =
      torch::log((radii - 0.01f).clamp_min(1e-5f)).unsqueeze(1).to(device_type_);

  torch::Tensor new_position_lrs =
      torch::full({new_xyzs.size(0)}, position_lr_init_,
                  torch::TensorOptions().device(device_type_));

  torch::Tensor new_triangle_ids = torch::arange(
      next_triangle_id_, next_triangle_id_ + new_xyzs.size(0),
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  next_triangle_id_ += new_xyzs.size(0);

  torch::Tensor new_chunk_ids = computeChunkIds(new_xyzs, chunk_size_);

  densificationPostfix(new_tri_pts, new_features_dc, new_features_rest,
                       new_opacities_tensor, new_sigma, new_exist_since_iter,
                       new_chunk_ids, new_position_lrs, new_triangle_ids);
}

std::vector<ChunkCoord> TriangleModel::frustumCullChunks(
    std::shared_ptr<TriangleKeyframe> keyframe,
    bool use_cache) {
  // Delegate to standalone function with our cache
  FrustumCullingCache* cache_ptr =
      use_cache ? &triangle_visibility_cache_ : nullptr;
  return ::frustumCullChunks(keyframe, chunk_size_, cache_ptr);
}

torch::Tensor TriangleModel::cullVisibleTriangles(
    std::shared_ptr<TriangleKeyframe> keyframe,
    bool manage_memory) {
  torch::NoGradGuard no_grad;

  // Frustum cull chunks
  std::vector<ChunkCoord> visible_chunks =
      frustumCullChunks(keyframe, /*use_cache=*/true);

  if (visible_chunks.empty()) {
    // Return all-false mask
    return torch::zeros(
        {triangles_points_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  torch::Tensor visible_chunk_coords =
      chunkCoordVectorToTensor(visible_chunks, device_type_);
  torch::Tensor visible_chunk_ids =
      encodeChunkCoordsTensor(visible_chunk_coords);

  if (manage_memory && visible_chunk_ids.size(0) > 0) {
    loadChunks(visible_chunk_ids);
  }

  // Convert chunk visibility to triangle visibility
  torch::Tensor chunk_visibility_mask =
      createTriangleMaskFromChunks(visible_chunk_ids);

  updateChunkAccess(visible_chunk_ids);

  return chunk_visibility_mask;
}

torch::Tensor TriangleModel::createTriangleMaskFromChunks(
    const torch::Tensor& visible_chunk_ids) {
  if (visible_chunk_ids.size(0) == 0) {
    return torch::zeros(
        {triangle_chunk_ids_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  return torch::isin(triangle_chunk_ids_, visible_chunk_ids);
}

void TriangleModel::initializeEmpty(float spatial_lr_scale) {
  std::cout << "[Triangle Model] Initializing empty model for loading"
            << std::endl;

  this->spatial_lr_scale_ = spatial_lr_scale;

  // Initialize with empty tensors but correct shapes
  this->triangles_points_ =
      torch::empty(
          {0, 3, 3},
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
  this->sigma_ =
      torch::empty(
          {0, 1},
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
  this->triangle_chunk_ids_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  // Initialize tensor vectors for optimizer
  TRIANGLE_MODEL_TENSORS_TO_VEC

  // Mark as initialized
  is_initialized_ = true;

  std::cout << "[Triangle Model] Empty model initialized" << std::endl;
}