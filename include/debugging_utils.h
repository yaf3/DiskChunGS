#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

void visualizePointCloud(const torch::Tensor &points3D,
                         const torch::Tensor &colors,
                         const std::string &output_path = "point_cloud.ply");

void saveColorizedDepthMap(const torch::Tensor &depth,
                           int height,
                           int width,
                           const std::string &filepath,
                           float min_depth = 0.0f,
                           float max_depth = 20.0f);