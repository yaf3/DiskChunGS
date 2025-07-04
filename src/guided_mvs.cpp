#include "include/guided_mvs.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
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
GuidedMVS::GuidedMVS(int num_prev_keyframes, int num_depth_candidates)
    : n_cams(num_prev_keyframes),
      num_depth_candidates(num_depth_candidates),
      idepth_range(2e-1f) {}

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

  // Get feature maps
  auto refFeatMap = refKeyframe->feature_map_.contiguous().cuda();
  std::vector<torch::Tensor> featMaps_list;
  for (const auto& keyframe : keyframes) {
    featMaps_list.push_back(keyframe->feature_map_.cuda().contiguous());
  }
  auto featMaps = torch::stack(featMaps_list, 0).contiguous();

  // Get intrinsics
  auto intrinsics = torch::tensor({refKeyframe->intr_[0], refKeyframe->intr_[2],
                                   refKeyframe->intr_[3]})
                        .contiguous()
                        .cuda();

  // Get monocular inverse depth
  auto mono_idepth = refKeyframe->depth_image_.contiguous().cuda();

  // Initialize output tensors
  auto depth = -torch::ones(
      {uv_cuda.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto idist = -torch::ones(
      {uv_cuda.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  const int P = uv_cuda.size(0);

  // std::cout << "P: " << P << std::endl;
  // std::cout << "refFeatMap.size(0): " << static_cast<int>(refFeatMap.size(0))
  //           << std::endl;
  // std::cout << "refFeatMap.size(1): " << static_cast<int>(refFeatMap.size(1))
  //           << std::endl;
  // std::cout << "mono_idepth.size(-2): "
  //           << static_cast<int>(mono_idepth.size(-2)) << std::endl;
  // std::cout << "mono_idepth.size(-1): "
  //           << static_cast<int>(mono_idepth.size(-1)) << std::endl;
  // std::cout << "refKeyframe->image_height_: "
  //           << static_cast<int>(refKeyframe->image_height_) << std::endl;
  // std::cout << "refKeyframe->image_width_: "
  //           << static_cast<int>(refKeyframe->image_width_) << std::endl;
  // std::cout << "num_depth_candidates: " << num_depth_candidates << std::endl;

  if (P != 0) {
    // Type checks to match Python version and kernel expectations
    // TORCH_CHECK(uv_cuda.scalar_type() == torch::kFloat32, "uv must be
    // float32"); TORCH_CHECK(refFeatMap.scalar_type() == torch::kFloat16,
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
    launch_uvToDepth(
        uv_cuda.data_ptr<float>(), refFeatMap.data_ptr<at::Half>(),
        featMaps.data_ptr<at::Half>(), other2ref.data_ptr<float>(),
        intrinsics.data_ptr<float>(), mono_idepth.data_ptr<float>(),
        depth.data_ptr<float>(), idist.data_ptr<float>(), idepth_range, P,
        static_cast<int>(refFeatMap.size(0)),
        static_cast<int>(refFeatMap.size(1)),
        static_cast<int>(mono_idepth.size(-2)),
        static_cast<int>(mono_idepth.size(-1)),
        static_cast<int>(refKeyframe->image_height_),
        static_cast<int>(refKeyframe->image_width_), num_depth_candidates);
  }

  auto valid_mask = idist >= 0;
  return std::make_pair(depth, valid_mask);
};