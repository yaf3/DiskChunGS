#include <include/DepthAnything.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

// DepthAnything implementation - Modified for Metric3D
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

    std::cout << "Model initialization completed" << std::endl;

  } catch (const Ort::Exception& e) {
    throw std::runtime_error("ONNX Runtime error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize model: " +
                             std::string(e.what()));
  }

  // Initialize Sobel kernels (following Python implementation exactly)
  sobel_x_ = torch::tensor(
      {{{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  sobel_y_ = torch::tensor(
      {{{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
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
  }

  input_height_ = 616;
  input_width_ = 1064;  // Hardcoded for Metric3D model

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

// Modified preprocessing for Metric3D model
std::vector<float> DepthAnything::prepare_input_metric3d(
    const cv::Mat& img,
    cv::Size& original_size,
    std::vector<int>& pad_info) {
  // Store original size
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

// Modified estimate_depth function for Metric3D
std::tuple<torch::Tensor, torch::Tensor> DepthAnything::estimate_depth(
    const cv::Mat& image,
    float focal_length) {
  img_height_ = image.rows;
  img_width_ = image.cols;

  std::cout << "Input image type: " << image.type() << std::endl;
  std::cout << "Input image size: " << img_width_ << "x" << img_height_
            << std::endl;

  // Prepare input with Metric3D preprocessing
  cv::Size original_size;
  std::vector<int> pad_info;

  auto start_prep = std::chrono::high_resolution_clock::now();
  std::vector<float> input =
      prepare_input_metric3d(image, original_size, pad_info);
  auto end_prep = std::chrono::high_resolution_clock::now();

  // Run inference
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

  // Convert to torch tensor for unpadding
  cv::Mat resized_depth;
  cv::resize(raw_depth, resized_depth, cv::Size(img_width_, img_height_), 0, 0,
             cv::INTER_LINEAR);

  // Apply de-canonical transform for metric depth
  // Hardcoded intrinsics as requested - you should replace with actual values
  float canonical_to_real_scale = focal_length / 1000.0;
  cv::Mat metric_depth = resized_depth * canonical_to_real_scale;

  torch::Tensor depth =
      tensor_utils::cvMat2TorchTensor_Float32(metric_depth, torch::kCUDA);

  // auto [t, s] = get_t_s(depth);
  // depth = (depth - t) / s;

  if (depth.dim() == 2) {
    depth = depth.unsqueeze(0).unsqueeze(0);
  } else if (depth.dim() == 3) {
    depth = depth.unsqueeze(0);
  }

  // Compute gradients using Sobel filters
  torch::Tensor grad_x = torch::nn::functional::conv2d(
      depth, sobel_x_, torch::nn::functional::Conv2dFuncOptions().padding(1));

  torch::Tensor grad_y = torch::nn::functional::conv2d(
      depth, sobel_y_, torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute edge magnitude and confidence
  torch::Tensor edges = torch::cat({grad_x, grad_y}, 0);
  torch::Tensor edges_sq_norm = (edges.pow(2)).sum(0, true);

  float var = 0.2f;
  torch::Tensor confidence = torch::exp(-edges_sq_norm / var);

  return std::make_tuple(depth, confidence);
}

/**
 * Get median and median absolute deviation for depth normalization
 * Following the Python implementation: get_t_s(d)
 */
std::tuple<torch::Tensor, torch::Tensor> DepthAnything::get_t_s(
    const torch::Tensor& depth) const {
  torch::Tensor t = depth.median();
  torch::Tensor s = (depth - t).abs().median();
  return std::make_tuple(t, s);
}

/**
 * Align samples by finding scale and offset
 * Following the Python implementation: align_samples(tri_idepth, mono_idepth)
 */
std::tuple<torch::Tensor, float, float> DepthAnything::align_samples(
    const torch::Tensor& tri_idepth,
    const torch::Tensor& mono_idepth) const {
  auto [t_tri_tensor, s_tri_tensor] = get_t_s(tri_idepth);
  auto [t_mono_tensor, s_mono_tensor] = get_t_s(mono_idepth);

  float t_tri = t_tri_tensor.item<float>();
  float s_tri = s_tri_tensor.item<float>();
  float t_mono = t_mono_tensor.item<float>();
  float s_mono = s_mono_tensor.item<float>();

  // Add debug prints
  std::cout << "DEBUG align_samples:" << std::endl;
  std::cout << "  tri_idepth range: " << tri_idepth.min().item<float>() << " - "
            << tri_idepth.max().item<float>() << std::endl;
  std::cout << "  mono_idepth range: " << mono_idepth.min().item<float>()
            << " - " << mono_idepth.max().item<float>() << std::endl;
  std::cout << "  t_tri=" << t_tri << ", s_tri=" << s_tri << std::endl;
  std::cout << "  t_mono=" << t_mono << ", s_mono=" << s_mono << std::endl;

  float scale = s_tri / s_mono;
  float offset = t_tri - t_mono * scale;

  torch::Tensor aligned = mono_idepth * scale + offset;

  return std::make_tuple(aligned, scale, offset);
}

/**
 * Sample depth values at given pixel coordinates
 */
torch::Tensor DepthAnything::sample_depth_at_pixels(
    const torch::Tensor& depth_map,
    const torch::Tensor& pixel_coords,
    int width,
    int height) const {
  // Convert pixel coordinates to normalized coordinates [-1, 1]
  torch::Tensor normalized_coords = pixel_coords.clone().to(torch::kFloat32);
  normalized_coords.select(1, 0) =
      (normalized_coords.select(1, 0) / (width - 1)) * 2.0 - 1.0;
  normalized_coords.select(1, 1) =
      (normalized_coords.select(1, 1) / (height - 1)) * 2.0 - 1.0;

  // Reshape for grid_sample: [1, 1, N, 2]
  torch::Tensor grid = normalized_coords.view({1, 1, -1, 2});

  std::cout << "Grid shape: " << grid.sizes() << std::endl;
  std::cout << "Depth shape: " << depth_map.sizes() << std::endl;

  // Sample using bilinear interpolation
  torch::Tensor sampled = torch::nn::functional::grid_sample(
      depth_map, grid,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .align_corners(true));

  return sampled.view({-1});
}

/**
 * Align mono depth map with triangulated depth from keypoints
 * Following the Python implementation exactly
 * Takes your existing depth output and makes it metric
 */
torch::Tensor DepthAnything::align_depth_to_metric(
    const torch::Tensor& relative_depth_map,
    const std::vector<float>& keypoint_pixels,
    const std::vector<float>& keypoint_depths,
    int width,
    int height) const {
  auto start_align = std::chrono::high_resolution_clock::now();
  if (keypoint_pixels.empty() || keypoint_depths.empty() ||
      keypoint_pixels.size() != keypoint_depths.size() * 2) {
    std::cerr << "Warning: No valid keypoints for depth alignment" << std::endl;
    return relative_depth_map;
  }

  int num_keypoints = keypoint_depths.size();

  // Convert keypoint data to tensors
  torch::Tensor pixel_coords =
      torch::from_blob(const_cast<float*>(keypoint_pixels.data()),
                       {num_keypoints, 2},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(relative_depth_map.device());

  torch::Tensor metric_depths =
      torch::from_blob(const_cast<float*>(keypoint_depths.data()),
                       {num_keypoints},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(relative_depth_map.device());

  // Sample relative depths at keypoint locations
  torch::Tensor relative_depths_sampled =
      sample_depth_at_pixels(relative_depth_map, pixel_coords, width, height);

  std::cout << "Sampled relative depths range: "
            << relative_depths_sampled.min().item<float>() << " - "
            << relative_depths_sampled.max().item<float>() << std::endl;
  std::cout << "Metric depths range: " << metric_depths.min().item<float>()
            << " - " << metric_depths.max().item<float>() << std::endl;

  // Simple least squares: solve for [scale, offset] in metric = scale *
  // relative + offset Set up system: [relative_depths, ones] * [scale; offset]
  // = metric_depths
  torch::Tensor A = torch::stack(
      {relative_depths_sampled, torch::ones_like(relative_depths_sampled)}, 1);
  torch::Tensor b = metric_depths;

  // Solve using pseudo-inverse: x = (A^T A)^{-1} A^T b
  torch::Tensor AtA = torch::matmul(A.t(), A);
  torch::Tensor Atb = torch::matmul(A.t(), b);
  torch::Tensor solution = torch::linalg_solve(AtA, Atb);

  float scale = solution[0].item<float>();
  float offset = solution[1].item<float>();

  auto end_align = std::chrono::high_resolution_clock::now();

  auto align_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_align - start_align);
  std::cout << "Alignment: " << align_time.count() << "ms" << std::endl;

  std::cout << "Simple alignment: scale=" << scale << ", offset=" << offset
            << " (using " << num_keypoints << " keypoints)" << std::endl;

  // Apply transformation to entire depth map
  torch::Tensor metric_depth_map = relative_depth_map * scale + offset;

  std::cout << "Final metric depth range: "
            << metric_depth_map.min().item<float>() << " - "
            << metric_depth_map.max().item<float>() << std::endl;

  return metric_depth_map;
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