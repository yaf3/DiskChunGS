#include "utils.h"

namespace gs {
namespace utils {

Sophus::SE3f tensor_to_pose(torch::Tensor pose_tensor) {
  // Make sure pose_tensor is a 4x4 matrix
  if (pose_tensor.sizes().size() != 2 || pose_tensor.size(0) != 4 ||
      pose_tensor.size(1) != 4) {
    throw std::runtime_error("Pose tensor must be a 4x4 matrix");
  }

  // Convert to CPU and float32 if needed
  pose_tensor = pose_tensor.to(torch::kCPU, torch::kFloat32).contiguous();

  // Extract rotation and translation separately to construct SE3 properly
  // Extract 3x3 rotation matrix
  torch::Tensor R_tensor = pose_tensor.slice(0, 0, 3).slice(1, 0, 3);
  Eigen::Matrix3f R = Eigen::Map<Eigen::Matrix3f>(R_tensor.data_ptr<float>());

  // Extract translation vector
  torch::Tensor t_tensor =
      pose_tensor.slice(0, 0, 3).slice(1, 3, 4).reshape({3});
  Eigen::Vector3f t = Eigen::Map<Eigen::Vector3f>(t_tensor.data_ptr<float>());

  // Ensure the rotation matrix is valid
  Eigen::Quaternionf quat(R);

  // Create Sophus::SE3f from rotation and translation
  return Sophus::SE3f(quat, t);
}

torch::Tensor cv_mat_to_tensor(const cv::Mat& mat) {
  torch::Tensor output;

  if (mat.channels() == 3 && mat.type() == CV_32FC3) {
    // Create tensor with the right shape [H, W, C]
    output = torch::zeros({mat.rows, mat.cols, 3}, torch::kFloat32);

    // Copy data from cv::Mat to tensor
    std::memcpy(output.data_ptr(), mat.data,
                sizeof(float) * mat.rows * mat.cols * 3);

    // Convert to [C, H, W] format (PyTorch standard)
    output = output.permute({2, 0, 1}).contiguous();
  } else {
    throw std::runtime_error(
        "Unsupported cv::Mat format. Expected 3-channel float32 image "
        "(CV_32FC3)");
  }

  return output;
}

}  // namespace utils
}  // namespace gs