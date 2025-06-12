#include <include/stereo_depth.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  try {
    // Configuration - you can easily change resolution for speed/quality
    // tradeoff
    cv::Size model_resolution(
        1280, 384);  // Options: (192,144), (320,240), (480,360), (640,480),
                     // (832,624), (1024,768)

    // KITTI camera configuration with focal length scaled for model resolution
    float baseline = 0.53716f;  // KITTI baseline in meters
    float kitti_focal_length =
        721.5377f;  // KITTI focal length for 1242x375 images
    // float scaled_focal_length = kitti_focal_length * model_resolution.width /
    // 1242.0f;  // Scale for model resolution

    float max_distance = 80.0f;  // KITTI scenes can be quite far

    // Initialize model
    // std::string model_path =
    // "./models/fast_acvnet_plus_onnx_no_gridsample/fast_acvnet_plus_kitti_2015_opset11_"
    // +
    //                         std::to_string(model_resolution.height) + "x" +
    //                         std::to_string(model_resolution.width) + ".onnx";
    std::string model_path =
        "./models/fast_acvnet_plus_onnx_gridsample/"
        "fast_acvnet_plus_kitti_2015_opset16_" +
        std::to_string(model_resolution.height) + "x" +
        std::to_string(model_resolution.width) + ".onnx";

    std::cout << "Loading StereoDepth model from: " << model_path << std::endl;
    std::cout << "Model resolution: " << model_resolution.width << "x"
              << model_resolution.height << std::endl;
    // std::cout << "Scaled focal length: " << scaled_focal_length << " pixels"
    // << std::endl;

    StereoDepth depth_estimator(model_path);
    std::cout << "Model loaded successfully!" << std::endl;

    // Load images
    std::string left_path = "./models/kitti_left.png";
    std::string right_path = "./models/kitti_right.png";

    cv::Mat left_img = cv::imread(left_path);
    cv::Mat right_img = cv::imread(right_path);

    if (!left_img.empty() && !right_img.empty()) {
      std::cout << "Processing KITTI stereo pair..." << std::endl;
      std::cout << "Original image size: " << left_img.cols << "x"
                << left_img.rows << std::endl;

      // Time the inference
      auto start = std::chrono::high_resolution_clock::now();

      // Estimate depth
      cv::Mat depth_map = depth_estimator.estimate_metric_depth(
          left_img, right_img, kitti_focal_length, baseline);

      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

      // Get results in different formats
      // cv::Mat model_disparity = depth_estimator.get_disparity_map(false); //
      // Model resolution cv::Mat original_disparity =
      // depth_estimator.get_disparity_map(true);   // Original image resolution
      // cv::Mat original_depth = depth_estimator.get_depth_map(true); //
      // Original image resolution

      // std::cout << "Model resolution disparity: " << model_disparity.size()
      // << std::endl; std::cout << "Original resolution disparity: " <<
      // original_disparity.size() << std::endl;

      // Create visualizations (these are automatically at original resolution)
      // cv::Mat color_disparity = depth_estimator.draw_disparity();
      // cv::Mat color_depth = depth_estimator.draw_depth();

      // Get some depth statistics
      if (!depth_map.empty()) {
        double min_depth, max_depth;
        cv::minMaxLoc(depth_map, &min_depth, &max_depth);
        cv::Scalar mean_depth = cv::mean(depth_map);
        std::cout << "Depth range: " << min_depth << " - " << max_depth
                  << " meters" << std::endl;
        std::cout << "Mean depth: " << mean_depth[0] << " meters" << std::endl;
      }

      float max_dist = 80;
      cv::Mat norm_depth_map = 255.0 * (1.0 - depth_map / max_dist);

      // Clamp values
      cv::threshold(norm_depth_map, norm_depth_map, 0, 0, cv::THRESH_TOZERO);
      cv::threshold(norm_depth_map, norm_depth_map, 255, 0,
                    cv::THRESH_TOZERO_INV);

      cv::Mat depth_8u;
      norm_depth_map.convertTo(depth_8u, CV_8U);

      cv::Mat colored_depth;
      cv::applyColorMap(depth_8u, colored_depth, cv::COLORMAP_JET);

      cv::imwrite("kitti_depth_map.png", colored_depth);

      // // Create different visualization options
      // cv::Mat combined_overlay, side_by_side, depth_only;

      // // Option 1: Overlay depth on original image
      // if (!color_depth.empty()) {
      //     cv::addWeighted(left_img, 0.6, color_depth, 0.4, 0,
      //     combined_overlay);
      // } else {
      //     combined_overlay = left_img.clone();
      // }

      // // Option 2: Side by side comparison
      // cv::hconcat(left_img, color_disparity, side_by_side);

      // // Option 3: Just the depth map
      // depth_only = color_depth.clone();

      // Display options
      // cv::namedWindow("1. Original + Depth Overlay", cv::WINDOW_NORMAL);
      // cv::namedWindow("2. Original + Disparity", cv::WINDOW_NORMAL);
      // cv::namedWindow("3. Depth Map Only", cv::WINDOW_NORMAL);

      // cv::imshow("1. Original + Depth Overlay", combined_overlay);
      // cv::imshow("2. Original + Disparity", side_by_side);
      // cv::imshow("3. Depth Map Only", depth_only);

      // // Save results
      // cv::imwrite("kitti_depth_overlay.png", combined_overlay);
      // cv::imwrite("kitti_disparity_comparison.png", side_by_side);
      // cv::imwrite("kitti_depth_map.png", depth_only);
      // cv::imwrite("kitti_raw_disparity.exr", original_disparity);  // Save
      // raw disparity as EXR cv::imwrite("kitti_raw_depth.exr",
      // original_depth);          // Save raw depth as EXR

      // std::cout << "\nResults saved:" << std::endl;
      // std::cout << "- kitti_depth_overlay.png (visualization)" << std::endl;
      // std::cout << "- kitti_disparity_comparison.png (comparison)" <<
      // std::endl; std::cout << "- kitti_depth_map.png (depth only)" <<
      // std::endl; std::cout << "- kitti_raw_disparity.exr (raw disparity
      // data)" << std::endl; std::cout << "- kitti_raw_depth.exr (raw depth
      // data)" << std::endl;

      // std::cout << "\nPress any key to continue..." << std::endl;
      // cv::waitKey(0);
      // cv::destroyAllWindows();

    } else {
      std::cerr << "Could not load images. Please check paths:" << std::endl;
      std::cerr << "Left: " << left_path << std::endl;
      std::cerr << "Right: " << right_path << std::endl;
    }

    std::cout << "Demo completed successfully!" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}