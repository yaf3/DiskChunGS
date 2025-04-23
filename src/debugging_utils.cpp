#include "include/debugging_utils.h"

void visualizePointCloud(const torch::Tensor& points3D,
                         const torch::Tensor& colors,
                         const std::string& output_path) {
  // Convert tensors to CPU if they're on GPU
  auto points_cpu = points3D.cpu();
  auto colors_cpu = colors.cpu();

  // Open file for writing
  std::ofstream ply_file(output_path);
  if (!ply_file.is_open()) {
    throw std::runtime_error("Failed to open file for writing point cloud"); 
  }

  // Write PLY header
  int num_points = points_cpu.size(0);
  ply_file << "ply\n";
  ply_file << "format ascii 1.0\n";
  ply_file << "element vertex " << num_points << "\n";
  ply_file << "property float x\n";
  ply_file << "property float y\n";
  ply_file << "property float z\n";
  ply_file << "property uchar red\n";
  ply_file << "property uchar green\n";
  ply_file << "property uchar blue\n";
  ply_file << "end_header\n";

  // Write points and colors
  auto points_accessor = points_cpu.accessor<float, 2>();
  auto colors_accessor = colors_cpu.accessor<float, 2>();

  for (int i = 0; i < num_points; i++) {
    // Write XYZ coordinates
    ply_file << points_accessor[i][0] << " " << points_accessor[i][1] << " "
             << points_accessor[i][2] << " ";

    // Convert float RGB values [0,1] to uchar [0,255] and write
    int r = static_cast<int>(colors_accessor[i][0] * 255);
    int g = static_cast<int>(colors_accessor[i][1] * 255);
    int b = static_cast<int>(colors_accessor[i][2] * 255);
    ply_file << r << " " << g << " " << b << "\n";
  }

  ply_file.close();
  std::cout << "Point cloud saved to " << output_path << std::endl;
  std::cout << "Number of points saved: " << num_points << std::endl;
}

void saveColorizedDepthMap(const torch::Tensor& depth,
                           int height,
                           int width,
                           const std::string& filepath,
                           float min_depth,
                           float max_depth) {
  try {
    torch::NoGradGuard no_grad;

    // Print tensor info for debugging
    std::cout << "Depth tensor shape: " << depth.sizes() << std::endl;
    std::cout << "Min value: " << torch::min(depth).item<float>() << std::endl;
    std::cout << "Max value: " << torch::max(depth).item<float>() << std::endl;

    // Reshape flattened depth tensor to 2D using provided dimensions
    auto depth_2d = depth.reshape({height, width});

    // Clamp values to valid range
    depth_2d = torch::clamp(depth_2d, min_depth, max_depth);

    // Normalize to 0-1 range
    depth_2d = (depth_2d - min_depth) / (max_depth - min_depth);

    // Convert to CPU and uint8
    auto depth_cpu = (depth_2d * 255).to(torch::kCPU).to(torch::kUInt8);

    // Convert to OpenCV Mat
    cv::Mat depth_mat(height, width, CV_8UC1);
    std::memcpy(depth_mat.data, depth_cpu.data_ptr(),
                depth_cpu.numel() * sizeof(uint8_t));

    // Apply colormap
    cv::Mat colored_depth;
    cv::applyColorMap(depth_mat, colored_depth, cv::COLORMAP_TURBO);

    // Save the colorized depth map using the provided filepath
    bool success = cv::imwrite(filepath, colored_depth);
    if (!success) {
      std::cerr << "Failed to save depth map to: " << filepath << std::endl;
    } else {
      std::cout << "Successfully saved depth map to: " << filepath << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error in saveColorizedDepthMap: " << e.what() << std::endl;
  }
}