/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <thread>

#include "ORB-SLAM3/include/System.h"
#include "gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void LoadImages(const string &strSequence,
                vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
void saveTrackingTime(std::vector<float> &vTimesTrack,
                      const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

void saveSlowdownFactor(float slowdown_factor,
                        const std::filesystem::path &output_dir);

// New function to calculate slowdown factor based on the sequence
float calculateSlowdownFactor(
    const std::string &sequencePath,
    float targetSpeedKmh = 0.0) {  // Changed default to 0.0
  // If targetSpeedKmh is 0 or negative, return 1.0 (no slowdown)
  if (targetSpeedKmh <= 0.0) {
    std::cout << "No target speed specified, using default slowdown factor: "
                 "1.0x (no slowdown)"
              << std::endl;
    return 1.0f;
  }

  // Extract sequence number from path
  std::string seqNum = "";
  std::size_t found = sequencePath.find_last_of("/\\");
  if (found != std::string::npos) {
    std::string dirName = sequencePath.substr(found + 1);
    // Try to extract sequence number (last digits in the path)
    for (auto it = dirName.rbegin(); it != dirName.rend(); ++it) {
      if (std::isdigit(*it)) {
        seqNum = *it + seqNum;
      } else if (!seqNum.empty()) {
        break;
      }
    }
  }

  // Predefined speeds for known KITTI sequences (km/h)
  std::map<std::string, float> sequenceSpeeds = {
      {"00", 29.52},
      {"01", 80.21},
      {"02", 39.14},
      {"03", 25.21},
      {"04", 52.29},
      {"05", 28.76},
      {"06", 40.31},
      {"07", 21.24},
      {"08", 28.50},
      {"09", 38.58},
      {"10", 27.56}
  };

  // Default slowdown factor if sequence not found
  float defaultFactor = 1.0f;  // Changed from 6.87 to 1.0

  // Calculate slowdown factor based on original speed
  if (!seqNum.empty() && sequenceSpeeds.find(seqNum) != sequenceSpeeds.end()) {
    float factor = sequenceSpeeds[seqNum] / targetSpeedKmh;
    std::cout << "Sequence " << seqNum
              << " detected: Original speed = " << sequenceSpeeds[seqNum]
              << " km/h, Slowdown factor = " << factor << "x to achieve "
              << targetSpeedKmh << " km/h" << std::endl;
    return factor;
  }

  std::cout << "Sequence not recognized, using default slowdown factor: "
            << defaultFactor << "x" << std::endl;
  return defaultFactor;
}

// Function to calculate speed based on frames, length and framerate
float calculateOriginalSpeed(int numFrames,
                             float lengthInMeters,
                             float frameRate = 10.0) {
  // Time taken in seconds
  float timeInSeconds = numFrames / frameRate;

  // Distance in kilometers
  float distanceInKm = lengthInMeters / 1000.0f;

  // Speed in km/h
  return (distanceInKm / timeInSeconds) * 3600.0f;
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7 && argc != 8 && argc != 9) {
    std::cerr << std::endl
              << "Usage: " << argv[0] << " path_to_vocabulary" /*1*/
              << " path_to_ORB_SLAM3_settings"                 /*2*/
              << " path_to_gaussian_mapping_settings"          /*3*/
              << " path_to_sequence"                           /*4*/
              << " path_to_trajectory_output_directory/"       /*5*/
              << " (optional)no_viewer"                        /*6*/
              << " (optional)target_speed_kmh"                 /*7*/
              << " (optional)adjust_fps_in_config"             /*8*/
              << std::endl;
    return 1;
  }

  bool use_viewer = true;
  if (argc >= 7)
    use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

  float target_speed_kmh = 0.0;  // Default target speed (walking pace)
  if (argc >= 8) {
    try {
      target_speed_kmh = std::stof(argv[7]);
      if (target_speed_kmh <= 0) {
        std::cerr << "Target speed must be positive!" << std::endl;
      }
    } catch (std::exception &e) {
      std::cerr << "Invalid target speed. Using default of 5.0 km/h."
                << std::endl;
    }
  }

  // Whether to adjust FPS in the ORB-SLAM3 config file
  bool adjust_fps_in_config = false;
  if (argc == 9) {
    adjust_fps_in_config = (std::string(argv[8]) == "true" ? true : false);
  }

  std::string sequence_path = std::string(argv[4]);
  // Calculate the appropriate slowdown factor based on the sequence
  float slowdown_factor =
      calculateSlowdownFactor(sequence_path, target_speed_kmh);

  std::string output_directory = std::string(argv[5]);
  if (output_directory.back() != '/') output_directory += "/";
  std::filesystem::path output_dir(output_directory);

  // Retrieve paths to images
  vector<string> vstrImageFilenamesRGB;
  vector<double> vTimestamps;
  LoadImages(string(argv[4]), vstrImageFilenamesRGB, vTimestamps);

  // Check consistency in the number of images
  int nImages = vstrImageFilenamesRGB.size();
  if (vstrImageFilenamesRGB.empty()) {
    std::cerr << std::endl << "No images found in provided path." << std::endl;
    return 1;
  }

  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }

  // Optionally adjust the FPS in the config file to match the slowdown
  std::string orbslam_settings_path = std::string(argv[2]);
  if (adjust_fps_in_config) {
    std::string adjusted_settings_path = orbslam_settings_path + ".adjusted";
    float adjusted_fps =
        std::max(1, static_cast<int>(std::round(10.0f / slowdown_factor)));

    std::cout << "Adjusting Camera.fps from 10.0 to " << adjusted_fps
              << " to match slowdown factor of " << slowdown_factor << "x"
              << std::endl;

    // Read the original config file
    std::ifstream original_config(orbslam_settings_path);
    std::ofstream adjusted_config(adjusted_settings_path);

    if (original_config.is_open() && adjusted_config.is_open()) {
      std::string line;
      while (std::getline(original_config, line)) {
        if (line.find("Camera.fps:") != std::string::npos) {
          // Replace the fps line
          adjusted_config << "Camera.fps: " << adjusted_fps << std::endl;
        } else {
          // Keep original line
          adjusted_config << line << std::endl;
        }
      }
      original_config.close();
      adjusted_config.close();

      // Use the adjusted config file instead
      orbslam_settings_path = adjusted_settings_path;
    } else {
      std::cerr << "Warning: Could not adjust fps in config file. Using "
                   "original settings."
                << std::endl;
    }
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  std::shared_ptr<ORB_SLAM3::System> pSLAM =
      std::make_shared<ORB_SLAM3::System>(
          argv[1], orbslam_settings_path.c_str(), ORB_SLAM3::System::MONOCULAR);
  float imageScale = pSLAM->GetImageScale();

  // Create GaussianMapper
  std::filesystem::path gaussian_cfg_path(argv[3]);
  std::shared_ptr<GaussianMapper> pGausMapper =
      std::make_shared<GaussianMapper>(pSLAM, gaussian_cfg_path, output_dir, 0,
                                       device_type);
  std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

  // Create Gaussian Viewer
  std::thread viewer_thd;
  std::shared_ptr<ImGuiViewer> pViewer;
  if (use_viewer) {
    pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
    viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
  }

  // Vector for tracking time statistics
  std::vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  std::cout.precision(17);

  std::cout << std::endl << "-------" << std::endl;
  std::cout << "Start processing sequence ..." << std::endl;
  std::cout << "Images in the sequence: " << nImages << std::endl;
  std::cout << "Target speed: " << target_speed_kmh << " km/h" << std::endl;
  std::cout << "Slowdown factor: " << slowdown_factor << "x" << std::endl
            << std::endl;

  // Main loop
  cv::Mat im;
  double start_timestamp =
      vTimestamps[0];  // Store the first timestamp as reference
  for (int ni = 0; ni < nImages; ni++) {
    if (pSLAM->isShutDown()) break;

    // Read image from file
    im = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
    cv::cvtColor(im, im, CV_BGR2RGB);

    // Get original timestamp
    double original_tframe = vTimestamps[ni];

    // Scale timestamp to match slowdown (relative to start time)
    // This ensures timestamps grow at the slowed down rate
    double scaled_tframe =
        start_timestamp + (original_tframe - start_timestamp) * slowdown_factor;

    if (im.empty()) {
      std::cerr << std::endl
                << "Failed to load image at: " << vstrImageFilenamesRGB[ni]
                << std::endl;
      return 1;
    }

    if (imageScale != 1.f) {
      int width = im.cols * imageScale;
      int height = im.rows * imageScale;
      cv::resize(im, im, cv::Size(width, height));
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Pass the image to the SLAM system with scaled timestamp
    pSLAM->TrackMonocular(im, scaled_tframe,
                          std::vector<ORB_SLAM3::IMU::Point>(),
                          vstrImageFilenamesRGB[ni]);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimestamps[ni + 1] -
          vTimestamps[ni];  // Use original timestamps for interval calculation
    else if (ni > 0)
      T = vTimestamps[ni] - vTimestamps[ni - 1];

    T *= slowdown_factor;  // Apply the calculated slowdown factor

    if (ttrack < T) usleep((T - ttrack) * 1e6);
  }

  // Stop all threads
  pSLAM->Shutdown();
  training_thd.join();
  if (use_viewer) viewer_thd.join();

  // GPU peak usage
  saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

  // Tracking time statistics
  saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

  saveSlowdownFactor(slowdown_factor, output_dir);

  // Save camera trajectory
  pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
  pSLAM->SaveKeyFrameTrajectoryTUM(
      (output_dir / "KeyFrameTrajectory_TUM.txt").string());
  pSLAM->SaveTrajectoryEuRoC(
      (output_dir / "CameraTrajectory_EuRoC.txt").string());
  pSLAM->SaveKeyFrameTrajectoryEuRoC(
      (output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
  pSLAM->SaveTrajectoryKITTI(
      (output_dir / "CameraTrajectory_KITTI.txt").string());

  return 0;
}

void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_2/";

  const int nTimes = vTimestamps.size();
  vstrImageFilenames.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
  }
}

void saveTrackingTime(std::vector<float> &vTimesTrack,
                      const std::string &strSavePath) {
  std::ofstream out;
  out.open(strSavePath.c_str());
  std::size_t nImages = vTimesTrack.size();
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    out << std::fixed << std::setprecision(4) << vTimesTrack[ni] << std::endl;
    totaltime += vTimesTrack[ni];
  }

  // std::sort(vTimesTrack.begin(), vTimesTrack.end());
  // out << "-------" << std::endl;
  // out << std::fixed << std::setprecision(4)
  //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
  // out << std::fixed << std::setprecision(4)
  //     << "mean tracking time: " << totaltime / nImages << std::endl;

  out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave) {
  namespace c10Alloc = c10::cuda::CUDACachingAllocator;
  c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

  c10Alloc::Stat reserved_bytes =
      mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
  float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

  c10Alloc::Stat alloc_bytes =
      mem_stats
          .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
  float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

  std::ofstream out(pathSave);
  out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
  out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
  out.close();
}

void saveSlowdownFactor(float slowdown_factor,
                        const std::filesystem::path &output_dir) {
  std::ofstream out((output_dir / "slowdown_factor.txt").string());
  if (out.is_open()) {
    out << std::fixed << std::setprecision(6) << slowdown_factor << std::endl;
    std::cout << "Saved slowdown factor " << slowdown_factor << "x to "
              << (output_dir / "slowdown_factor.txt").string() << std::endl;
    out.close();
  } else {
    std::cerr << "Warning: Could not save slowdown factor to "
              << (output_dir / "slowdown_factor.txt").string() << std::endl;
  }
}