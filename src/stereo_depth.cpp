#include <include/stereo_depth.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

// StereoDepth implementation
StereoDepth::StereoDepth(const std::string& model_path)
    : output_data_(nullptr),
      pinned_left_input_(nullptr),
      pinned_right_input_(nullptr) {
  // Initialize CUDA stream
  cudaStreamCreate(&stream_);

  // Initialize buffers to nullptr
  for (int i = 0; i < 3; ++i) {
    buffers_[i] = nullptr;
  }

  initialize_model(model_path);
}

StereoDepth::~StereoDepth() { free_buffers(); }

void StereoDepth::initialize_model(const std::string& model_path) {
  // Check if model file exists
  std::ifstream file(model_path);
  if (!file.good()) {
    throw std::runtime_error("Model file not found: " + model_path);
  }
  file.close();

  std::cout << "Loading ONNX model for TensorRT: " << model_path << std::endl;

  // Generate engine cache path
  engine_cache_path_ = model_path + ".trt";

  try {
    // Try to load cached engine first
    if (load_engine(engine_cache_path_)) {
      std::cout << "Loaded cached TensorRT engine" << std::endl;
    } else {
      std::cout << "Building TensorRT engine from ONNX..." << std::endl;
      build_engine(model_path);
      save_engine(engine_cache_path_);
      std::cout << "TensorRT engine saved to cache" << std::endl;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Failed to create execution context");
    }

    // Get model dimensions
    get_model_info();

    // Allocate GPU memory buffers
    allocate_buffers();

    std::cout << "TensorRT model initialization completed" << std::endl;

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize TensorRT model: " +
                             std::string(e.what()));
  }
}

void StereoDepth::build_engine(const std::string& onnx_path) {
  // Create builder
  builder_.reset(nvinfer1::createInferBuilder(logger_));
  if (!builder_) {
    throw std::runtime_error("Failed to create TensorRT builder");
  }

  // Create network definition
  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network_.reset(builder_->createNetworkV2(explicit_batch));
  if (!network_) {
    throw std::runtime_error("Failed to create network definition");
  }

  // Create ONNX parser
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, logger_));
  if (!parser) {
    throw std::runtime_error("Failed to create ONNX parser");
  }

  // Parse ONNX model
  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::string error_msg = "Failed to parse ONNX file: " + onnx_path + "\n";
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      error_msg += parser->getError(i)->desc();
      error_msg += "\n";
    }
    throw std::runtime_error(error_msg);
  }

  // Create builder configuration
  config_.reset(builder_->createBuilderConfig());
  if (!config_) {
    throw std::runtime_error("Failed to create builder config");
  }

  // Set memory pool limits
  config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                              1U << 30);  // 1GB

  // Enable FP16 precision if available
  if (builder_->platformHasFastFp16()) {
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "FP16 precision enabled" << std::endl;
  }

  // Build serialized network
  auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
      builder_->buildSerializedNetwork(*network_, *config_));
  if (!serialized_engine) {
    throw std::runtime_error("Failed to build TensorRT engine");
  }

  // Create runtime and deserialize engine
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  engine_.reset(runtime_->deserializeCudaEngine(serialized_engine->data(),
                                                serialized_engine->size()));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize TensorRT engine");
  }

  std::cout << "TensorRT engine built successfully" << std::endl;
}

bool StereoDepth::load_engine(const std::string& engine_path) {
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (!engine_file.good()) {
    return false;
  }

  engine_file.seekg(0, std::ios::end);
  const size_t model_size = engine_file.tellg();
  engine_file.seekg(0, std::ios::beg);

  std::unique_ptr<char[]> engine_data(new char[model_size]);
  engine_file.read(engine_data.get(), model_size);
  engine_file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.get(), model_size));
  return engine_ != nullptr;
}

bool StereoDepth::save_engine(const std::string& engine_path) {
  if (!engine_) {
    return false;
  }

  auto serialized_engine =
      std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
  if (!serialized_engine) {
    return false;
  }

  std::ofstream engine_file(engine_path, std::ios::binary);
  if (!engine_file.good()) {
    return false;
  }

  engine_file.write(static_cast<const char*>(serialized_engine->data()),
                    serialized_engine->size());
  engine_file.close();
  return true;
}

void StereoDepth::get_model_info() {
  // Get input dimensions from first input tensor
  auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
  input_size_ = 3 * input_height_ * input_width_;

  std::cout << "Model input size: " << input_width_ << "x" << input_height_
            << std::endl;
}

void StereoDepth::allocate_buffers() {
  const size_t input_size = input_size_ * sizeof(float);
  const size_t output_size = input_height_ * input_width_ * sizeof(float);

  // Allocate GPU memory for inputs and output
  cudaMalloc(&buffers_[0], input_size);   // left input
  cudaMalloc(&buffers_[1], input_size);   // right input
  cudaMalloc(&buffers_[2], output_size);  // output

  // Allocate pinned host memory for inputs (faster CPU->GPU transfers)
  cudaMallocHost(&pinned_left_input_, input_size_ * sizeof(float));
  cudaMallocHost(&pinned_right_input_, input_size_ * sizeof(float));

  // Allocate host memory for output
  output_data_ = new float[input_height_ * input_width_];

  // Pre-allocate GPU preprocessing buffers
  gpu_channels_.resize(3);

  std::cout << "GPU buffers allocated (with pinned host memory)" << std::endl;
}

void StereoDepth::free_buffers() {
  for (int i = 0; i < 3; ++i) {
    if (buffers_[i]) {
      cudaFree(buffers_[i]);
      buffers_[i] = nullptr;
    }
  }

  if (pinned_left_input_) {
    cudaFreeHost(pinned_left_input_);
    pinned_left_input_ = nullptr;
  }

  if (pinned_right_input_) {
    cudaFreeHost(pinned_right_input_);
    pinned_right_input_ = nullptr;
  }

  if (output_data_) {
    delete[] output_data_;
    output_data_ = nullptr;
  }

  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

// GPU-accelerated preprocessing - all operations on GPU
void StereoDepth::prepare_input_optimized(const cv::Mat& img,
                                          float* output_buffer) {
  // Upload to GPU
  cv::cuda::GpuMat gpu_img;
  gpu_img.upload(img);

  // Resize on GPU
  cv::cuda::resize(gpu_img, gpu_resized_, cv::Size(input_width_, input_height_),
                   0, 0, cv::INTER_LINEAR);

  // Convert to float and scale [0,255] → [0,1] on GPU
  if (gpu_resized_.type() != CV_32FC3) {
    gpu_resized_.convertTo(gpu_float_, CV_32FC3, 1.0 / 255.0);
  } else {
    gpu_float_ = gpu_resized_;
  }

  // ImageNet normalization on GPU - RGB order
  cv::Scalar mean(0.485, 0.456, 0.406);
  cv::Scalar std(0.229, 0.224, 0.225);

  cv::cuda::subtract(gpu_float_, mean, gpu_normalized_);
  cv::cuda::divide(gpu_normalized_, std, gpu_normalized_);

  // Split channels on GPU
  cv::cuda::split(gpu_normalized_, gpu_channels_);

  // Download CHW data directly to pinned memory
  const int channel_size = input_width_ * input_height_;
  for (int c = 0; c < 3; c++) {
    gpu_channels_[c].download(cv::Mat(input_height_, input_width_, CV_32F,
                                      output_buffer + c * channel_size));
  }
}

cv::Mat StereoDepth::inference_tensorrt() {
  try {
    const size_t input_size_bytes = input_size_ * sizeof(float);

    // Copy input data from pinned memory to GPU (faster transfer)
    cudaMemcpyAsync(buffers_[0], pinned_left_input_, input_size_bytes,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(buffers_[1], pinned_right_input_, input_size_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // Set tensor addresses for the execution context
    context_->setTensorAddress(engine_->getIOTensorName(0),
                               buffers_[0]);  // left input
    context_->setTensorAddress(engine_->getIOTensorName(1),
                               buffers_[1]);  // right input
    context_->setTensorAddress(engine_->getIOTensorName(2),
                               buffers_[2]);  // output

    // Run inference
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT inference execution failed");
    }

    // Copy output data back to host
    const size_t output_size_bytes =
        input_height_ * input_width_ * sizeof(float);
    cudaMemcpyAsync(output_data_, buffers_[2], output_size_bytes,
                    cudaMemcpyDeviceToHost, stream_);

    // Synchronize stream to ensure completion
    cudaStreamSynchronize(stream_);

    // Create result matrix - no clone needed, output_data_ is persistent
    return cv::Mat(input_height_, input_width_, CV_32F, output_data_).clone();

  } catch (const std::exception& e) {
    throw std::runtime_error("TensorRT inference error: " +
                             std::string(e.what()));
  }
}

cv::Mat StereoDepth::estimate_depth(const cv::Mat& left_img,
                                    const cv::Mat& right_img) {
  img_height_ = left_img.rows;
  img_width_ = left_img.cols;

  // Use optimized preprocessing with pinned memory and TensorRT inference
  auto start_prep = std::chrono::high_resolution_clock::now();
  prepare_input_optimized(left_img, pinned_left_input_);
  prepare_input_optimized(right_img, pinned_right_input_);
  auto end_prep = std::chrono::high_resolution_clock::now();

  auto start_inf = std::chrono::high_resolution_clock::now();
  cv::Mat raw_disparity = inference_tensorrt();
  auto end_inf = std::chrono::high_resolution_clock::now();

  // Optional: Print timing breakdown
  auto prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_prep - start_prep);
  auto inf_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_inf - start_inf);
  // std::cout << "Preprocessing: " << prep_time.count()
  //           << "ms, Inference: " << inf_time.count() << "ms" << std::endl;

  return raw_disparity;
}

/**
 * @brief Estimate depth from stereo images and convert to metric depth
 * @param left_img Left stereo image
 * @param right_img Right stereo image
 * @param focal_length Camera focal length in pixels (for original image
 * resolution)
 * @param baseline Stereo baseline distance in meters
 * @return Depth map in meters
 */
cv::Mat StereoDepth::estimate_metric_depth(const cv::Mat& left_img,
                                           const cv::Mat& right_img,
                                           const float focal_length,
                                           const float baseline) {
  // First get disparity
  cv::Mat disparity = estimate_depth(left_img, right_img);

  // Scale disparity back to original image resolution
  cv::Mat disparity_map;
  cv::resize(disparity, disparity_map, cv::Size(img_width_, img_height_), 0, 0,
             cv::INTER_LINEAR);

  // Scale disparity values to account for resolution change
  float scale_factor =
      static_cast<float>(img_width_) / static_cast<float>(input_width_);
  disparity_map *= scale_factor;

  // Convert to depth
  cv::Mat depth_map;
  cv::divide(focal_length * baseline, disparity_map, depth_map);
  return depth_map;
}