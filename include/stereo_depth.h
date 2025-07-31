#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

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
#include <fstream>

/**
 * @brief Fast ACVNet depth estimation class using TensorRT
 *
 * This class provides stereo depth estimation using the Fast ACVNet model
 * with TensorRT for inference and OpenCV for image processing.
 */
class StereoDepth {
 public:
  /**
   * @brief Constructor
   * @param model_path Path to the ONNX model file
   * @param max_dist Maximum distance for depth visualization
   */
  StereoDepth(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~StereoDepth();

  /**
   * @brief Estimate depth from stereo images
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @return Disparity map
   */
  cv::Mat estimate_depth(const cv::Mat& left_img, const cv::Mat& right_img);

  /**
   * @brief Estimate depth from stereo images and convert to metric depth
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @param focal_length Camera focal length in pixels (for original image
   * resolution)
   * @param baseline Stereo baseline distance in meters
   * @return Depth map in meters
   */
  cv::Mat estimate_metric_depth(const cv::Mat& left_img,
                                const cv::Mat& right_img,
                                const float focal_length,
                                const float baseline);

 private:
  /**
   * @brief TensorRT Logger class
   */
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
      }
    }
  };

  // TensorRT components
  Logger logger_;
  std::unique_ptr<nvinfer1::IBuilder> builder_;
  std::unique_ptr<nvinfer1::INetworkDefinition> network_;
  std::unique_ptr<nvinfer1::IBuilderConfig> config_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Model information
  int input_height_;
  int input_width_;
  int input_size_;  // Pre-calculated input size
  int img_height_;
  int img_width_;

  // GPU memory buffers
  void* buffers_[3];  // left_input, right_input, output
  cudaStream_t stream_;
  float* output_data_;

  // Results
  cv::Mat disparity_map_;
  cv::Mat depth_map_;

  // Configuration
  float max_dist_;
  std::string engine_cache_path_;

  /**
   * @brief Initialize TensorRT model
   * @param model_path Path to ONNX model file
   */
  void initialize_model(const std::string& model_path);

  /**
   * @brief Build TensorRT engine from ONNX model
   * @param onnx_path Path to ONNX model file
   */
  void build_engine(const std::string& onnx_path);

  /**
   * @brief Load TensorRT engine from cache
   * @param engine_path Path to engine cache file
   */
  bool load_engine(const std::string& engine_path);

  /**
   * @brief Save TensorRT engine to cache
   * @param engine_path Path to save engine cache
   */
  bool save_engine(const std::string& engine_path);

  /**
   * @brief Allocate GPU memory buffers
   */
  void allocate_buffers();

  /**
   * @brief Free GPU memory buffers
   */
  void free_buffers();

  /**
   * @brief Get model input/output dimensions
   */
  void get_model_info();

  /**
   * @brief Prepare input image for inference
   * @param img Input image
   * @return Preprocessed data as vector
   */
  std::vector<float> prepare_input_optimized(const cv::Mat& img);

  /**
   * @brief Run TensorRT inference
   * @param left_input Left image data
   * @param right_input Right image data
   * @return Disparity map
   */
  cv::Mat inference_tensorrt(const std::vector<float>& left_input,
                             const std::vector<float>& right_input);
};