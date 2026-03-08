/*
 * From fused-ssim (https://github.com/rahul-goel/fused-ssim)
 * Copyright (C) 2024 Rahul Goel et al.
 * Licensed under the MIT License.
 * See: https://github.com/rahul-goel/fused-ssim/blob/main/LICENSE
 */

#pragma once
#include <torch/all.h>

#include <cstdio>
#include <string>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(float C1,
          float C2,
          torch::Tensor &img1,
          torch::Tensor &img2,
          bool train);

torch::Tensor fusedssim_backward(float C1,
                                 float C2,
                                 torch::Tensor &img1,
                                 torch::Tensor &img2,
                                 torch::Tensor &dL_dmap,
                                 torch::Tensor &dm_dmu1,
                                 torch::Tensor &dm_dsigma1_sq,
                                 torch::Tensor &dm_dsigma12);