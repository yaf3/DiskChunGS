/**
 * This file is part of DiskChunGS, modified from CaRtGS/Photo-SLAM.
 *
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Casimir Feldmann (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See the GNU General Public License for more details:
 * <http://www.gnu.org/licenses/>.
 */

#include "triangle_mapper.h"

int TriangleMapper::getIteration() {
  std::unique_lock<std::mutex> lock(mutex_status_);
  return iteration_;
}

void TriangleMapper::increaseIteration(const int inc) {
  std::unique_lock<std::mutex> lock(mutex_status_);
  iteration_ += inc;
}

float TriangleMapper::positionLearningRateInit() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.position_lr_init_;
}

float TriangleMapper::featureLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.feature_lr_;
}

float TriangleMapper::opacityLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_lr_;
}

float TriangleMapper::sigmaLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.sigma_lr_;
}

float TriangleMapper::lambdaDssim() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_dssim_;
}

float TriangleMapper::lambdaDepth() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_depth_;
}

float TriangleMapper::lambdaNormal() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_normal_;
}

int TriangleMapper::normalLossMode() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.normal_loss_mode_;
}

float TriangleMapper::lambdaEquilateral() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_equilateral_;
}

int TriangleMapper::newKeyframeTimesOfUse() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return new_keyframe_times_of_use_;
}

int TriangleMapper::stableNumIterExistence() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return stable_num_iter_existence_;
}

bool TriangleMapper::isKeepingTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return keep_training_;
}

void TriangleMapper::setLambdaDssim(const float lambda_dssim) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.lambda_dssim_ = lambda_dssim;
}

void TriangleMapper::setNewKeyframeTimesOfUse(const int times) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  new_keyframe_times_of_use_ = times;
}

void TriangleMapper::setStableNumIterExistence(const int niter) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  stable_num_iter_existence_ = niter;
}

void TriangleMapper::setKeepTraining(const bool keep) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  keep_training_ = keep;
}
