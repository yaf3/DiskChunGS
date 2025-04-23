#pragma once

#include <cv_bridge/cv_bridge.h>
#include <dlfcn.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <string>
#include <thread>
#include <vector>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

class GaussianSLAMWrapper {
 public:
  GaussianSLAMWrapper(ros::NodeHandle &nh, ros::NodeHandle &pnh);
  ~GaussianSLAMWrapper();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Topic subscribers
  message_filters::Subscriber<sensor_msgs::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::Image> right_sub_;
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  ros::Subscriber mono_sub_;
  ros::Subscriber imu_sub_;  // Added IMU subscriber

  // Synchronization
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      sync_pol;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> rgbd_sync_;

  // TF listener for external poses
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  bool first_frame = true;
  Sophus::SE3f T_init;

  // SLAM system
  std::shared_ptr<ORB_SLAM3::System> slam_system_;
  std::shared_ptr<GaussianMapper> gaussian_mapper_;
  std::shared_ptr<ImGuiViewer> viewer_;

  // Threads
  std::thread mapper_thread_;
  std::thread viewer_thread_;

  // Parameters
  std::string vocabulary_path_;
  std::string orb_settings_path_;
  std::string gaussian_settings_path_;
  std::string output_directory_;
  bool use_viewer_;
  std::string mode_;
  std::string slam_mode_;
  std::string left_topic_;
  std::string right_topic_;
  std::string mono_topic_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string imu_topic_;  // Added IMU topic
  std::string target_frame_;
  std::string source_frame_;

  // IMU-related members
  std::vector<ORB_SLAM3::IMU::Point> imu_buffer_;  // IMU measurements buffer
  std::mutex imu_mutex_;  // Mutex for thread-safe access to IMU buffer
  double last_processed_image_ts_ = 0;  // Timestamp of the last processed image

  // Callback methods
  void monoCallback(const sensor_msgs::ImageConstPtr &msg);
  void stereoCallback(const sensor_msgs::ImageConstPtr &left,
                      const sensor_msgs::ImageConstPtr &right);
  void rgbdCallback(const sensor_msgs::ImageConstPtr &rgb,
                    const sensor_msgs::ImageConstPtr &depth);
  void imuCallback(const sensor_msgs::ImuConstPtr &msg);  // Added IMU callback

  // Helper methods
  void initializeSLAMSystem();
  void initializeGaussianMapper();
  bool getExternalPose(Sophus::SE3f &pose, double timestamp);
};