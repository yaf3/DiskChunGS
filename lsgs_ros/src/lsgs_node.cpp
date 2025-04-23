// gaussian_slam_node.cpp
#include <ros/ros.h>

#include "lsgs_wrapper.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "gaussian_slam_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try {
    GaussianSLAMWrapper gaussian_slam(nh, pnh);
    ros::spin();
    ROS_INFO("Gaussian SLAM node shutting down");
  } catch (const std::exception& e) {
    ROS_ERROR("Exception in Gaussian SLAM node: %s", e.what());
    return 1;
  }

  return 0;
}