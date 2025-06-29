#pragma once

#include <torch/extension.h>

#include <string>
#include <vector>

namespace gs {

// Initialize the GaussianMapper with config and model path
bool initialize(const std::string& gaussian_cfg_path,
                const std::string& result_path);

// Render from a given pose (4x4 transformation matrix)
torch::Tensor renderFromPose(torch::Tensor pose_tensor, int width, int height);

}  // namespace gs