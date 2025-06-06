#include <include/DepthAnything.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

// DepthAnything implementation
DepthAnything::DepthAnything(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthAnything") {
  initialize_model(model_path);
}

void DepthAnything::initialize_model(const std::string& model_path) {
  // Check if model file exists
  std::ifstream file(model_path);
  if (!file.good()) {
    throw std::runtime_error("Model file not found: " + model_path);
  }
  file.close();

  std::cout << "Loading ONNX model: " << model_path << std::endl;

  // Configure session options for optimal performance
  session_options_.SetIntraOpNumThreads(
      1);  // Single thread often performs better for GPU inference
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Enable memory pattern optimization
  session_options_.EnableMemPattern();
  session_options_.EnableCpuMemArena();

  // Try CUDA provider with optimized settings
  try {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 1;  // Extend by doubling
    cuda_options.gpu_mem_limit = SIZE_MAX;   // No memory limit
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;

    session_options_.AppendExecutionProvider_CUDA(cuda_options);
    std::cout << "CUDA provider enabled with optimizations" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "CUDA provider not available, using CPU: " << e.what()
              << std::endl;
  }

  try {
    // Create session
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                              session_options_);
    std::cout << "ONNX session created successfully" << std::endl;

    get_input_details();
    get_output_details();

    // Note: memory_info_ will be created as needed in inference methods

    std::cout << "Model initialization completed" << std::endl;

  } catch (const Ort::Exception& e) {
    throw std::runtime_error("ONNX Runtime error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize model: " +
                             std::string(e.what()));
  }
}

void DepthAnything::get_input_details() {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session_->GetInputCount();

  std::cout << "=== Model Input Analysis ===" << std::endl;
  std::cout << "Number of input nodes: " << num_input_nodes << std::endl;

  input_names_.clear();
  input_names_char_.clear();

  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name_ptr = session_->GetInputNameAllocated(i, allocator);
    std::string input_name(input_name_ptr.get());
    input_names_.push_back(input_name);
    std::cout << "Input " << i << " name: '" << input_name << "'" << std::endl;

    // Get input type and shape info
    auto input_type_info = session_->GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();

    std::cout << "Input " << i << " shape: [";
    for (size_t j = 0; j < input_shape.size(); ++j) {
      std::cout << input_shape[j];
      if (j < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    if (i == 0) {  // Use first input as primary
      input_shape_ = input_shape;
      if (input_shape.size() >= 4) {
        input_height_ = input_shape[2];
        input_width_ = input_shape[3];
        std::cout << "Model expects input: " << input_width_ << "x"
                  << input_height_ << std::endl;
      }
    }
  }

  // Convert to char pointers
  for (const auto& name : input_names_) {
    input_names_char_.push_back(name.c_str());
  }
}

void DepthAnything::get_output_details() {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_output_nodes = session_->GetOutputCount();

  std::cout << "=== Model Output Analysis ===" << std::endl;
  std::cout << "Number of output nodes: " << num_output_nodes << std::endl;

  output_names_.clear();
  output_names_char_.clear();

  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
    std::string output_name(output_name_ptr.get());
    output_names_.push_back(output_name);
    std::cout << "Output " << i << " name: '" << output_name << "'"
              << std::endl;

    // Get output type and shape info
    auto output_type_info = session_->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_shape = output_tensor_info.GetShape();

    std::cout << "Output " << i << " shape: [";
    for (size_t j = 0; j < output_shape.size(); ++j) {
      std::cout << output_shape[j];
      if (j < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  // Convert to char pointers
  for (const auto& name : output_names_) {
    output_names_char_.push_back(name.c_str());
  }
}

// Optimized preprocessing without PyTorch
std::vector<float> DepthAnything::prepare_input_optimized(const cv::Mat& img) {
  // Resize image to target dimensions
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(input_width_, input_height_), 0, 0,
             cv::INTER_LINEAR);

  cv::Mat float_img = resized_img;  // Already float [0,1]

  // ImageNet normalization
  cv::Scalar mean(0.485, 0.456, 0.406);  // RGB order: R, G, B
  cv::Scalar std(0.229, 0.224, 0.225);   // RGB order: R, G, B

  cv::Mat normalized_img;
  cv::subtract(float_img, mean, normalized_img);
  cv::divide(normalized_img, std, normalized_img);

  // Convert from HWC to CHW format and flatten
  std::vector<cv::Mat> channels(3);
  cv::split(normalized_img, channels);

  std::vector<float> input_data;
  input_data.reserve(input_width_ * input_height_ * 3);

  // Pack channels in CHW order (C, H, W)
  for (int c = 0; c < 3; c++) {
    cv::Mat flat_channel = channels[c].reshape(1, 1);  // Flatten to 1D
    std::vector<float> channel_data;
    flat_channel.copyTo(channel_data);
    input_data.insert(input_data.end(), channel_data.begin(),
                      channel_data.end());
  }

  return input_data;
}

cv::Mat DepthAnything::inference_optimized(const std::vector<float>& input) {
  // Create input shape
  std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};

  // Create memory info for this inference
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Create input tensors directly from vectors
  std::vector<Ort::Value> input_tensors;

  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(input.data()), input.size(),
      input_shape.data(), input_shape.size()));

  try {
    // Run inference
    auto output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, input_names_char_.data(),
                      input_tensors.data(), input_tensors.size(),
                      output_names_char_.data(), output_names_char_.size());

    // Get output tensor info
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // Debug the actual output shape
    std::cout << "ONNX output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
      std::cout << output_shape[i];
      if (i < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Handle different possible output shapes
    cv::Mat result;
    if (output_shape.size() == 3) {
      // Shape is [1, H, W] - skip batch dimension
      int height = static_cast<int>(output_shape[1]);
      int width = static_cast<int>(output_shape[2]);
      result = cv::Mat(height, width, CV_32F, output_data).clone();
    } else if (output_shape.size() == 4) {
      // Shape is [1, 1, H, W] - skip batch and channel dimensions
      int height = static_cast<int>(output_shape[2]);
      int width = static_cast<int>(output_shape[3]);
      result = cv::Mat(height, width, CV_32F, output_data).clone();
    } else {
      throw std::runtime_error("Unexpected output tensor dimensions: " +
                               std::to_string(output_shape.size()));
    }

    std::cout << "Created cv::Mat with size: " << result.rows << "x"
              << result.cols << std::endl;

    return result;

  } catch (const Ort::Exception& e) {
    throw std::runtime_error("ONNX Runtime inference error: " +
                             std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Inference error: " + std::string(e.what()));
  }
}

// Add this to your estimate_depth function
cv::Mat DepthAnything::estimate_depth(const cv::Mat& image) {
  img_height_ = image.rows;
  img_width_ = image.cols;

  std::cout << "Input image type: " << image.type() << std::endl;

  auto start_prep = std::chrono::high_resolution_clock::now();
  std::vector<float> input = prepare_input_optimized(image);
  auto end_prep = std::chrono::high_resolution_clock::now();

  // Add debug validation
  debug_preprocessing(input);

  auto start_inf = std::chrono::high_resolution_clock::now();
  cv::Mat raw_depth = inference_optimized(input);
  auto end_inf = std::chrono::high_resolution_clock::now();

  std::cout << "Raw depth shape: " << raw_depth.rows << "x" << raw_depth.cols
            << std::endl;

  // Check raw depth statistics
  double raw_min, raw_max;
  cv::minMaxLoc(raw_depth, &raw_min, &raw_max);
  cv::Scalar raw_mean = cv::mean(raw_depth);
  std::cout << "Raw depth range: " << raw_min << " - " << raw_max << std::endl;
  std::cout << "Raw depth mean: " << raw_mean[0] << std::endl;

  auto prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_prep - start_prep);
  auto inf_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_inf - start_inf);
  std::cout << "Preprocessing: " << prep_time.count()
            << "ms, Inference: " << inf_time.count() << "ms" << std::endl;

  return raw_depth;
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
cv::Mat DepthAnything::estimate_metric_depth(const cv::Mat& image) {
  cv::Mat depth_output = estimate_depth(image);
  cv::Mat depth_map;
  cv::resize(depth_output, depth_map, cv::Size(img_width_, img_height_), 0, 0,
             cv::INTER_LINEAR);
  return depth_map;
}

void DepthAnything::debug_preprocessing(const std::vector<float>& input) {
  std::cout << "=== Debug Preprocessing ===" << std::endl;
  std::cout << "Input tensor size: " << input.size() << std::endl;
  std::cout << "Expected size: " << (3 * input_height_ * input_width_)
            << std::endl;

  // Check value ranges for each channel
  const int channel_size = input_height_ * input_width_;

  for (int c = 0; c < 3; c++) {
    const float* channel_data = input.data() + c * channel_size;
    float min_val =
        *std::min_element(channel_data, channel_data + channel_size);
    float max_val =
        *std::max_element(channel_data, channel_data + channel_size);
    float mean_val =
        std::accumulate(channel_data, channel_data + channel_size, 0.0f) /
        channel_size;

    std::cout << "Channel " << c << " - Min: " << min_val
              << ", Max: " << max_val << ", Mean: " << mean_val << std::endl;
  }

  // Expected ranges after ImageNet normalization should be roughly:
  // Channel 0 (R): [-2.1, 2.6]
  // Channel 1 (G): [-2.0, 2.8]
  // Channel 2 (B): [-1.8, 2.3]
  std::cout << "=== End Debug ===" << std::endl;
}