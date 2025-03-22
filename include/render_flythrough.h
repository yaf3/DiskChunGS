#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "include/gaussian_keyframe.h"

void smoothPath(std::vector<Eigen::Vector3d> &path, float smoothness_factor);
std::vector<Eigen::Vector3d> createSmoothPath(
    const std::vector<Eigen::Vector3d> &keypoints,
    int points_per_segment,
    float tension = 0.5f);

Eigen::Vector3d catmullRomInterpolate(const Eigen::Vector3d &p0,
                                      const Eigen::Vector3d &p1,
                                      const Eigen::Vector3d &p2,
                                      const Eigen::Vector3d &p3,
                                      double t,
                                      float tension = 0.5f);

std::vector<double> computeArcLengths(const std::vector<Eigen::Vector3d> &path);

void samplePathConstantSpeed(
    const std::vector<Eigen::Vector3d> &path,
    const std::vector<Eigen::Vector3d> &keyframe_positions,
    const std::vector<Eigen::Quaterniond> &keyframe_orientations,
    int num_samples,
    std::vector<Eigen::Vector3d> &sampled_positions,
    std::vector<Eigen::Quaterniond> &sampled_orientations);

void mapKeyframesToPath(const std::vector<Eigen::Vector3d> &keyframe_positions,
                        const std::vector<Eigen::Vector3d> &path,
                        const std::vector<double> &arc_lengths,
                        std::vector<double> &keyframe_parameters);

Eigen::Vector3d samplePositionAtArcLength(
    const std::vector<Eigen::Vector3d> &path,
    const std::vector<double> &arc_lengths,
    double target_length);

Eigen::Quaterniond interpolateOrientation(
    double param,
    const std::vector<double> &keyframe_parameters,
    const std::vector<Eigen::Quaterniond> &keyframe_orientations);

std::vector<Eigen::Matrix3d> computePathFrames(
    const std::vector<Eigen::Vector3d> &path);

double calculateSceneScale(const std::vector<Eigen::Vector3d> &positions);