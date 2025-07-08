#pragma once

#include <torch/torch.h>

#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "slam_deps/depth-anything-tensorrt/depth_anything.h"  // Include the TensorRT DepthAnything
#include "tensor_utils.h"

/**
 * @brief MonoDepth estimation class using TensorRT DepthAnything
 *
 * This class provides monocular depth estimation using the DepthAnything model
 * with TensorRT for fast inference.
 */
class MonoDepth {
 public:
  /**
   * @brief Constructor
   * @param model_path Path to the TensorRT engine file or ONNX model file
   */
  MonoDepth(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~MonoDepth() = default;

  /**
   * @brief Estimate depth from monocular image
   * @param image Input monocular image
   * @param focal_length Camera focal length (optional, for scale)
   * @return Tuple of depth tensor and confidence tensor
   */
  std::tuple<torch::Tensor, torch::Tensor> estimate_depth(
      const cv::Mat& image,
      float focal_length = 0.0f);

  /**
   * @brief Estimate relative depth from monocular image
   * @param image Input monocular image
   * @return Depth map as OpenCV Mat
   */
  cv::Mat estimate_relative_depth(const cv::Mat& image);

  /**
   * @brief Align depth using sparse keypoints
   * @param mono_depth_map Normalized depth from model
   * @param keypoint_pixels Pixel coordinates of keypoints
   * @param keypoint_depths Metric depths of keypoints in meters
   * @param width Image width
   * @param height Image height
   * @return Aligned depth tensor
   */
  torch::Tensor align_depth_equivalent(
      const torch::Tensor& mono_depth_map,
      const std::vector<float>& keypoint_pixels,
      const std::vector<float>& keypoint_depths,
      int width,
      int height) const;

 private:
  // TensorRT DepthAnything instance
  std::unique_ptr<DepthAnything> depth_anything_;

  // Model information
  int input_height_;
  int input_width_;
  int img_height_;
  int img_width_;

  // Sobel kernels for gradient computation
  torch::Tensor sobel_x_;
  torch::Tensor sobel_y_;

  /**
   * @brief Initialize the TensorRT model
   * @param model_path Path to model file
   */
  void initialize_model(const std::string& model_path);

  /**
   * @brief Get median and median absolute deviation for depth normalization
   * @param depth Input depth tensor
   * @return Tuple of median (t) and MAD (s)
   */
  std::tuple<torch::Tensor, torch::Tensor> get_t_s(
      const torch::Tensor& depth) const;

  /**
   * @brief Align samples by finding scale and offset
   * @param tri_idepth Target inverse depths
   * @param mono_idepth Source inverse depths
   * @return Tuple of aligned depths, scale, and offset
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> align_samples(
      const torch::Tensor& tri_idepth,
      const torch::Tensor& mono_idepth) const;

  /**
   * @brief Sample depth values at given pixel coordinates
   * @param depth_map Input depth map
   * @param pixel_coords Pixel coordinates
   * @param width Image width
   * @param height Image height
   * @return Sampled depth values
   */
  torch::Tensor sample_depth_at_pixels(const torch::Tensor& depth_map,
                                       const torch::Tensor& pixel_coords,
                                       int width,
                                       int height) const;
};