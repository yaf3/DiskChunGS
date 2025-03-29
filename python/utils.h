#pragma once

#include <torch/extension.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace gs {
namespace utils {

// Convert PyTorch tensor (4x4 matrix) to Sophus::SE3f pose
Sophus::SE3f tensor_to_pose(torch::Tensor pose_tensor);

// Convert cv::Mat to torch::Tensor
torch::Tensor cv_mat_to_tensor(const cv::Mat& mat);

}  // namespace utils
}  // namespace gs