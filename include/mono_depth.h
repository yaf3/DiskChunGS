#pragma once

#include <onnxruntime_cxx_api.h>
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

#include "tensor_utils.h"

/**
 * @brief Fast ACVNet depth estimation class using ONNX Runtime
 *
 * This class provides stereo depth estimation using the Fast ACVNet model
 * with ONNX Runtime for inference and OpenCV for image processing.
 */
class MonoDepth {
 public:
  /**
   * @brief Constructor
   * @param model_path Path to the ONNX model file
   * @param max_dist Maximum distance for depth visualization
   */
  MonoDepth(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~MonoDepth() = default;

  /**
   * @brief Estimate depth from stereo images
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @return Disparity map
   */
  std::tuple<torch::Tensor, torch::Tensor> estimate_depth(const cv::Mat& image,
                                                          float focal_length);

  /**
   * @brief Estimate depth from stereo images and convert to metric depth
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @param focal_length Camera focal length in pixels (for original image
   * resolution)
   * @param baseline Stereo baseline distance in meters
   * @return Depth map in meters
   */
  cv::Mat estimate_relative_depth(const cv::Mat& image);

  torch::Tensor align_depth_to_metric(const torch::Tensor& relative_depth_map,
                                      const std::vector<float>& keypoint_pixels,
                                      const std::vector<float>& keypoint_depths,
                                      int width,
                                      int height) const;

  torch::Tensor align_depth_to_metric_direct(
      const torch::Tensor& relative_depth_map,
      const std::vector<float>& keypoint_pixels,
      const std::vector<float>& keypoint_depths,
      int width,
      int height) const;

  torch::Tensor align_depth_least_squares(
      const torch::Tensor& relative_depth_map,
      const std::vector<float>& keypoint_pixels,
      const std::vector<float>& keypoint_depths,
      int width,
      int height) const;

  /**
   * @brief Estimate depth using both RGB image and reference depth, with depth
   * taking precedence
   * @param rgb_image Input RGB image for monocular depth estimation
   * @param depth_image Reference depth image (can be sparse or incomplete)
   * @param focal_length Camera focal length for metric conversion
   * @param depth_threshold Minimum valid depth value (depths below this are
   * considered invalid)
   * @param blend_sigma Standard deviation for Gaussian blending around depth
   * boundaries
   * @return Merged depth map with confidence
   */
  std::tuple<torch::Tensor, torch::Tensor> estimate_depth_with_reference(
      const cv::Mat& rgb_image,
      const cv::Mat& depth_image,
      float focal_length,
      float depth_threshold = 0.1f,
      float blend_sigma = 5.0f);

 private:
  // ONNX Runtime components
  std::unique_ptr<Ort::Session> session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;

  // Model information
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_char_;
  std::vector<const char*> output_names_char_;

  std::vector<int64_t> input_shape_;
  int input_height_;
  int input_width_;
  int input_size_;  // Pre-calculated input size
  int img_height_;
  int img_width_;

  // Results
  cv::Mat disparity_map_;
  cv::Mat depth_map_;

  // Configuration
  float max_dist_;

  torch::Tensor sobel_x_;
  torch::Tensor sobel_y_;

  float resize_scale_;

  /**
   * @brief Initialize ONNX model
   * @param model_path Path to ONNX model file
   */
  void initialize_model(const std::string& model_path);

  /**
   * @brief Get input layer details from model
   */
  void get_input_details();

  /**
   * @brief Get output layer details from model
   */
  void get_output_details();

  /**
   * @brief Prepare input image for inference (Optimized version)
   * @param img Input image
   * @return Preprocessed data as vector
   */
  std::vector<float> prepare_input_metric3d(const cv::Mat& img,
                                            cv::Size& original_size,
                                            std::vector<int>& pad_info);

  /**
   * @brief Run inference on input data (Optimized version)
   * @param left_input Left image data
   * @param right_input Right image data
   * @return Disparity map
   */
  cv::Mat inference_optimized(const std::vector<float>& input);

  void debug_preprocessing(const std::vector<float>& input);

  std::tuple<torch::Tensor, torch::Tensor> get_t_s(
      const torch::Tensor& depth) const;

  std::tuple<torch::Tensor, float, float> align_samples(
      const torch::Tensor& tri_idepth,
      const torch::Tensor& mono_idepth) const;

  torch::Tensor sample_depth_at_pixels(const torch::Tensor& depth_map,
                                       const torch::Tensor& pixel_coords,
                                       int width,
                                       int height) const;

  /**
   * @brief Create a distance field from valid depth pixels for smooth blending
   * @param depth_mask Binary mask of valid depth pixels
   * @param max_distance Maximum distance to compute
   * @return Distance field as torch tensor
   */
  torch::Tensor compute_distance_field(const torch::Tensor& depth_mask,
                                       float max_distance = 20.0f);

  /**
   * @brief Align monocular depth to reference depth using valid pixels
   * @param mono_depth Monocular depth prediction
   * @param ref_depth Reference depth image
   * @param valid_mask Mask of valid reference depth pixels
   * @return Aligned monocular depth
   */
  torch::Tensor align_mono_to_reference_depth(const torch::Tensor& mono_depth,
                                              const torch::Tensor& ref_depth,
                                              const torch::Tensor& valid_mask);
  ;
};
