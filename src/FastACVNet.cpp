#include <include/FastACVNet.h>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

// FastACVNet implementation
FastACVNet::FastACVNet(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "FastACVNet"){
    
    initialize_model(model_path);
}

void FastACVNet::initialize_model(const std::string& model_path) {
    // Check if model file exists
    std::ifstream file(model_path);
    if (!file.good()) {
        throw std::runtime_error("Model file not found: " + model_path);
    }
    file.close();
    
    std::cout << "Loading ONNX model: " << model_path << std::endl;
    
    // Configure session options for optimal performance
    session_options_.SetIntraOpNumThreads(1);  // Single thread often performs better for GPU inference
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
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
        std::cout << "CUDA provider not available, using CPU: " << e.what() << std::endl;
    }
    
    try {
        // Create session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        std::cout << "ONNX session created successfully" << std::endl;
        
        get_input_details();
        get_output_details();
        
        // Note: memory_info_ will be created as needed in inference methods
        
        std::cout << "Model initialization completed" << std::endl;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize model: " + std::string(e.what()));
    }
}

void FastACVNet::get_input_details() {
    // Get input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    
    std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
    
    input_names_.clear();
    input_names_char_.clear();
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        try {
            auto input_name_ptr = session_->GetInputNameAllocated(i, allocator);
            std::string input_name(input_name_ptr.get());
            
            std::cout << "Input " << i << " name: '" << input_name << "'" << std::endl;
            
            if (input_name.empty()) {
                throw std::runtime_error("Empty input name at index " + std::to_string(i));
            }
            
            input_names_.push_back(input_name);
            
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get input name for index " + std::to_string(i) + ": " + e.what());
        }
    }
    
    // Convert to char pointers AFTER all strings are stored
    for (const auto& name : input_names_) {
        input_names_char_.push_back(name.c_str());
    }
    
    // Get input shape from the first input
    if (num_input_nodes > 0) {
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (input_shape_.size() >= 4) {
            input_height_ = input_shape_[2];
            input_width_ = input_shape_[3];
            input_size_ = 3 * input_height_ * input_width_;  // Pre-calculate size
            std::cout << "Model input size: " << input_width_ << "x" << input_height_ << std::endl;
        } else {
            throw std::runtime_error("Unexpected input shape size: " + std::to_string(input_shape_.size()));
        }
    } else {
        throw std::runtime_error("No input nodes found in the model");
    }
}

void FastACVNet::get_output_details() {
    // Get output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_output_nodes = session_->GetOutputCount();
    
    std::cout << "Number of output nodes: " << num_output_nodes << std::endl;
    
    output_names_.clear();
    output_names_char_.clear();
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        try {
            auto output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
            std::string output_name(output_name_ptr.get());
            
            std::cout << "Output " << i << " name: '" << output_name << "'" << std::endl;
            
            if (output_name.empty()) {
                throw std::runtime_error("Empty output name at index " + std::to_string(i));
            }
            
            output_names_.push_back(output_name);
            
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get output name for index " + std::to_string(i) + ": " + e.what());
        }
    }
    
    // Convert to char pointers AFTER all strings are stored
    for (const auto& name : output_names_) {
        output_names_char_.push_back(name.c_str());
    }
}

// Optimized preprocessing without PyTorch
std::vector<float> FastACVNet::prepare_input_optimized(const cv::Mat& img) {
    // Pre-allocate output vector
    std::vector<float> input_data;
    input_data.resize(input_size_);
    
    // ImageNet normalization constants
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_inv[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    const float scale = 1.0f / 255.0f;
    
    // Resize image directly to target size
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);
    
    // Process all channels in parallel and normalize in single pass
    // Since input is already RGB, no color conversion needed
    const int channel_size = input_height_ * input_width_;
    float* data_ptr = input_data.data();
    
    for (int h = 0; h < input_height_; h++) {
        const uint8_t* row_ptr = resized_img.ptr<uint8_t>(h);
        
        for (int w = 0; w < input_width_; w++) {
            const int pixel_idx = h * input_width_ + w;
            const int rgb_idx = w * 3;
            
            // Direct RGB processing (no conversion needed)
            // Channel 0 (R) = RGB[0]
            data_ptr[pixel_idx] = ((float)row_ptr[rgb_idx] * scale - mean[0]) * std_inv[0];
            // Channel 1 (G) = RGB[1]  
            data_ptr[channel_size + pixel_idx] = ((float)row_ptr[rgb_idx + 1] * scale - mean[1]) * std_inv[1];
            // Channel 2 (B) = RGB[2]
            data_ptr[2 * channel_size + pixel_idx] = ((float)row_ptr[rgb_idx + 2] * scale - mean[2]) * std_inv[2];
        }
    }
    
    return input_data;
}

cv::Mat FastACVNet::inference_optimized(const std::vector<float>& left_input, 
                                       const std::vector<float>& right_input) {
    // Create input shape
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    
    // Create memory info for this inference
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    
    // Create input tensors directly from vectors
    std::vector<Ort::Value> input_tensors;
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(left_input.data()), 
        left_input.size(),
        input_shape.data(), 
        input_shape.size()
    ));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(right_input.data()), 
        right_input.size(),
        input_shape.data(), 
        input_shape.size()
    ));
    
    try {
        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names_char_.data(), 
                                          input_tensors.data(), 
                                          input_tensors.size(),
                                          output_names_char_.data(), 
                                          output_names_char_.size());
        
        // Convert output back to cv::Mat
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Create result matrix: output should be [1, 1, H, W] -> we want [H, W]
        cv::Mat result(output_shape[2], output_shape[3], CV_32F, output_data);
        
        return result.clone(); // Make a copy since the original data will be destroyed
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime inference error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference error: " + std::string(e.what()));
    }
}

cv::Mat FastACVNet::estimate_depth(const cv::Mat& left_img, const cv::Mat& right_img) {
  img_height_ = left_img.rows;
  img_width_ = left_img.cols;
  
  // Use optimized preprocessing and inference
  auto start_prep = std::chrono::high_resolution_clock::now();
  std::vector<float> left_input = prepare_input_optimized(left_img);
  std::vector<float> right_input = prepare_input_optimized(right_img);
  auto end_prep = std::chrono::high_resolution_clock::now();
  
  auto start_inf = std::chrono::high_resolution_clock::now();
  cv::Mat raw_disparity = inference_optimized(left_input, right_input);
  auto end_inf = std::chrono::high_resolution_clock::now();
  
  // Optional: Print timing breakdown
  auto prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_prep - start_prep);
  auto inf_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_inf - start_inf);
  std::cout << "Preprocessing: " << prep_time.count() << "ms, Inference: " << inf_time.count() << "ms" << std::endl;
  
  return raw_disparity;
}

// Keep the old PyTorch-based method for backward compatibility
torch::Tensor FastACVNet::prepare_input(const cv::Mat& img) {
    // No BGR to RGB conversion needed since input is already RGB
    
    // Resize
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_AREA);
    
    // Convert to float and normalize
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    
    // ImageNet normalization
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    
    std::vector<cv::Mat> channels;
    cv::split(resized_img, channels);
    
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    
    cv::merge(channels, resized_img);
    
    // Convert to tensor and add batch dimension
    torch::Tensor tensor = torch::from_blob(resized_img.data, {1, input_height_, input_width_, 3}, torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}); // NHWC to NCHW
    
    return tensor.contiguous();
}

cv::Mat FastACVNet::inference(const torch::Tensor& left_input, const torch::Tensor& right_input) {
    // Convert tensors to ONNX Runtime format
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    
    // Create memory info for this inference
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    
    std::vector<Ort::Value> input_tensors;
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(left_input.data_ptr<float>()), 
        left_input.numel(),
        input_shape.data(), 
        input_shape.size()
    ));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(right_input.data_ptr<float>()), 
        right_input.numel(),
        input_shape.data(), 
        input_shape.size()
    ));
    
    try {
        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names_char_.data(), 
                                          input_tensors.data(), 
                                          input_tensors.size(),
                                          output_names_char_.data(), 
                                          output_names_char_.size());
        
        // Convert output back to cv::Mat
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Create result matrix: output should be [1, 1, 480, 640] -> we want [480, 640]
        cv::Mat result(output_shape[2], output_shape[3], CV_32F, output_data);
        
        return result.clone(); // Make a copy since the original data will be destroyed
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime inference error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference error: " + std::string(e.what()));
    }
}

/**
 * @brief Estimate depth from stereo images and convert to metric depth
 * @param left_img Left stereo image
 * @param right_img Right stereo image  
 * @param focal_length Camera focal length in pixels (for original image resolution)
 * @param baseline Stereo baseline distance in meters
 * @return Depth map in meters
 */
cv::Mat FastACVNet::estimate_metric_depth(const cv::Mat& left_img, 
  const cv::Mat& right_img,
  const float focal_length,
  const float baseline) {
  // First get disparity
  cv::Mat disparity = estimate_depth(left_img, right_img);

  // Scale disparity back to original image resolution
  cv::Mat disparity_map;
  cv::resize(disparity, disparity_map, cv::Size(img_width_, img_height_), 0, 0, cv::INTER_LINEAR);

  // Scale disparity values to account for resolution change
  float scale_factor = static_cast<float>(img_width_) / static_cast<float>(input_width_);
  disparity_map *= scale_factor;

  // Convert to depth
  cv::Mat depth_map;
  cv::divide(focal_length * baseline, disparity_map, depth_map);
  return depth_map;
}