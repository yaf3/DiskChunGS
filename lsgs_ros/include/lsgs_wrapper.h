// gaussian_slam_wrapper.h
#pragma once

#include <cv_bridge/cv_bridge.h>
#include <dlfcn.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <thread>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

class GaussianSLAMWrapper {
 public:
  GaussianSLAMWrapper(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~GaussianSLAMWrapper();

 private:
  void stereoCallback(const sensor_msgs::ImageConstPtr& msg_left,
                      const sensor_msgs::ImageConstPtr& msg_right);
  void monoCallback(const sensor_msgs::ImageConstPtr& msg);
  void rgbdCallback(const sensor_msgs::ImageConstPtr& msg_rgb,
                    const sensor_msgs::ImageConstPtr& msg_depth);
  void initializeSLAMSystem();
  void initializeGaussianMapper();

  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Subscribers for stereo
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      sync_pol;
  message_filters::Subscriber<sensor_msgs::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::Image> right_sub_;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_;

  // Subscribers for RGB-D
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> rgbd_sync_;

  // Subscriber for mono
  ros::Subscriber mono_sub_;

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
  std::string mode_;  // "mono", "stereo", or "rgbd"

  std::string left_topic_;
  std::string right_topic_;

  std::string mono_topic_;

  std::string rgb_topic_;
  std::string depth_topic_;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  std::string slam_mode_;
  std::string target_frame_;
  std::string source_frame_;

  bool first_frame = true;
  Sophus::SE3f T_init;

  // Add helper function
  bool getExternalPose(Sophus::SE3f& Twc);
};