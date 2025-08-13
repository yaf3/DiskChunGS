#include "gs_core.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "include/gaussian_mapper.h"
#include "include/types.h"
#include "utils.h"

namespace gs {

// Global GaussianMapper instance
std::shared_ptr<GaussianMapper> g_pGausMapper = nullptr;

bool initialize(const std::string& gaussian_cfg_path,
                const std::string& result_path) {
  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Using GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "CUDA not available. Using CPU." << std::endl;
    device_type = torch::kCPU;
  }

  try {
    // Create GaussianMapper
    g_pGausMapper = std::make_shared<GaussianMapper>(
        std::filesystem::path(gaussian_cfg_path),
        std::filesystem::path(result_path), 0, device_type);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing GaussianMapper: " << e.what() << std::endl;
    return false;
  }
}

torch::Tensor renderFromPose(torch::Tensor pose_tensor, int width, int height) {
  if (!g_pGausMapper) {
    throw std::runtime_error(
        "GaussianMapper not initialized. Call initialize() first.");
  }

  try {
    // Convert pose tensor to Sophus::SE3f
    // std::cout << "Converting pose tensor to Sophus::SE3f..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    Sophus::SE3f Tcw = utils::tensor_to_pose(pose_tensor);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // std::cout << "Pose conversion took" << duration.count() << "ms"
    //           << std::endl;

    // Print the Sophus SE3 pose for debugging
    // std::cout << "Converted SE3 pose:" << std::endl;
    // std::cout << "Rotation matrix:\n" << Tcw.rotationMatrix() << std::endl;
    // std::cout << "Translation vector: " << Tcw.translation().transpose()
    //           << std::endl;

    // Render the image
    // std::cout << "Rendering image from pose with dimensions: " << width <<
    // "x"
    //           << height << std::endl;
    start_time = std::chrono::steady_clock::now();
    auto render_result =
        g_pGausMapper->renderFromPose(Tcw, width, height, true);
    cv::Mat rendered_image =
        std::get<0>(render_result);  // Get RGB image, ignore depth
    end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // std::cout << "Rendering took " << duration.count() << "ms" << std::endl;
    // Check if the rendered image is valid
    if (rendered_image.empty()) {
      throw std::runtime_error(
          "Rendered image is empty. This may indicate an issue with the "
          "rendering process.");
    }

    // std::cout << "Image rendered successfully with size: "
    //           << rendered_image.cols << "x" << rendered_image.rows
    //           << " and type: " << rendered_image.type() << std::endl;

    // Convert the OpenCV Mat to a PyTorch tensor
    // The rendered image is CV_32FC3 (float, 3 channels)
    start_time = std::chrono::steady_clock::now();
    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor tensor = torch::zeros({height, width, 3}, options);

    // Copy data from OpenCV Mat to PyTorch tensor
    std::memcpy(tensor.data_ptr(), rendered_image.data,
                sizeof(float) * height * width * 3);
    end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // std::cout << "Conversion to tensor took " << duration.count() << "ms"
    //           << std::endl;

    // Return tensor in standard PyTorch format (C, H, W)
    return tensor.permute({2, 0, 1}).contiguous();

  } catch (const std::exception& e) {
    std::cerr << "Error rendering image: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "Unknown error occurred during rendering" << std::endl;
    throw std::runtime_error("Unknown error occurred during rendering");
  }
}

}  // namespace gs