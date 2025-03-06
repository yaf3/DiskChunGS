// gaussian_slam_wrapper.cpp
#include "lsgs_wrapper.h"

GaussianSLAMWrapper::GaussianSLAMWrapper(ros::NodeHandle &nh,
                                         ros::NodeHandle &pnh)
    : nh_(nh),
      pnh_(pnh),
      tfListener(tfBuffer) {  // Remove subscriber initialization from here
  ROS_INFO("GaussianSLAMWrapper constructor starting...");

  // Dl_info dl_info;
  // dladdr((void*)cv::getBuildInformation, &dl_info);
  // std::cout << "Loading OpenCV from: " << dl_info.dli_fname << std::endl;
  // std::cout << "OpenCV Build Info:\n" << cv::getBuildInformation() <<
  // std::endl;

  // Load parameters from parameter server
  if (!pnh_.getParam("vocabulary_path", vocabulary_path_)) {
    throw std::runtime_error("Failed to load vocabulary_path parameter");
  }
  if (!pnh_.getParam("orb_settings_path", orb_settings_path_)) {
    throw std::runtime_error("Failed to load orb_settings_path parameter");
  }
  if (!pnh_.getParam("gaussian_settings_path", gaussian_settings_path_)) {
    throw std::runtime_error("Failed to load gaussian_settings_path parameter");
  }
  if (!pnh_.getParam("output_directory", output_directory_)) {
    throw std::runtime_error("Failed to load output_directory parameter");
  }
  pnh_.param<bool>("use_viewer", use_viewer_, false);
  pnh_.param<std::string>("mode", mode_, "stereo");
  pnh_.param<std::string>("left_topic", left_topic_, "/camera/rgb/image_raw");
  pnh_.param<std::string>("right_topic", right_topic_, "/camera/rgb/image_raw");
  pnh_.param<std::string>("mono_topic", mono_topic_, "/camera/image_raw");
  pnh_.param<std::string>("rgb_topic", rgb_topic_, "/camera/rgb/image_raw");
  pnh_.param<std::string>("depth_topic", depth_topic_,
                          "/camera/depth/image_raw");
  pnh_.param<std::string>("slam_mode", slam_mode_, "orbslam");
  pnh_.param<std::string>("target_frame", target_frame_, "map");
  pnh_.param<std::string>("source_frame", source_frame_,
                          "zed2i_left_camera_frame");

  ROS_INFO("Parameters loaded:");
  ROS_INFO("  mode: %s", mode_.c_str());
  ROS_INFO("  vocabulary_path: %s", vocabulary_path_.c_str());
  ROS_INFO("  orb_settings_path: %s", orb_settings_path_.c_str());
  ROS_INFO("  gaussian_settings_path: %s", gaussian_settings_path_.c_str());
  ROS_INFO("  output_directory: %s", output_directory_.c_str());
  ROS_INFO("  use_viewer: %d", use_viewer_);
  ROS_INFO("  slam_mode: %s", slam_mode_.c_str());
  if (slam_mode_ == "external" || slam_mode_ == "hybrid") {
    ROS_INFO("  target_frame: %s", target_frame_.c_str());
    ROS_INFO("  source_frame: %s", source_frame_.c_str());
  }

  // Verify files exist
  if (!std::filesystem::exists(vocabulary_path_)) {
    throw std::runtime_error("Vocabulary file not found: " + vocabulary_path_);
  }
  if (!std::filesystem::exists(orb_settings_path_)) {
    throw std::runtime_error("ORB settings file not found: " +
                             orb_settings_path_);
  }
  if (!std::filesystem::exists(gaussian_settings_path_)) {
    throw std::runtime_error("Gaussian settings file not found: " +
                             gaussian_settings_path_);
  }

  // Initialize subscribers based on mode
  if (mode_ == "stereo") {
    left_sub_.subscribe(nh_, left_topic_, 1);
    right_sub_.subscribe(nh_, right_topic_, 1);
    sync_.reset(new message_filters::Synchronizer<sync_pol>(
        sync_pol(30), left_sub_, right_sub_));
  } else if (mode_ == "rgbd") {
    rgb_sub_.subscribe(nh_, rgb_topic_, 1);
    depth_sub_.subscribe(nh_, depth_topic_, 1);
    rgbd_sync_.reset(new message_filters::Synchronizer<sync_pol>(
        sync_pol(30), rgb_sub_, depth_sub_));
  }

  if (slam_mode_ == "orbslam" || slam_mode_ == "hybrid") {
    initializeSLAMSystem();
  }

  initializeGaussianMapper();

  // Register appropriate callback based on mode
  if (mode_ == "stereo") {
    ROS_INFO("Registering stereo callback...");
    ROS_INFO("Left topic: %s", left_sub_.getTopic().c_str());
    ROS_INFO("Right topic: %s", right_sub_.getTopic().c_str());
    sync_->registerCallback(
        boost::bind(&GaussianSLAMWrapper::stereoCallback, this, _1, _2));
  } else if (mode_ == "rgbd") {
    ROS_INFO("Registering RGB-D callback...");
    ROS_INFO("RGB topic: %s", rgb_sub_.getTopic().c_str());
    ROS_INFO("Depth topic: %s", depth_sub_.getTopic().c_str());
    rgbd_sync_->registerCallback(
        boost::bind(&GaussianSLAMWrapper::rgbdCallback, this, _1, _2));
  } else if (mode_ == "mono") {
    ROS_INFO("Registering mono callback...");
    mono_sub_ =
        nh_.subscribe(mono_topic_, 1, &GaussianSLAMWrapper::monoCallback, this);
    ROS_INFO("Mono topic: %s", mono_topic_.c_str());
  } else {
    throw std::runtime_error("Invalid mode: " + mode_);
  }

  ROS_INFO("GaussianSLAMWrapper initialization complete!");
}

bool GaussianSLAMWrapper::getExternalPose(Sophus::SE3f &pose) {
  try {
    geometry_msgs::TransformStamped transformStamped =
        tfBuffer.lookupTransform(target_frame_, source_frame_, ros::Time(0));

    // Get the rotation quaternion and translation
    Eigen::Quaternionf quat(transformStamped.transform.rotation.w,
                            transformStamped.transform.rotation.x,
                            transformStamped.transform.rotation.y,
                            transformStamped.transform.rotation.z);

    Eigen::Vector3f trans(transformStamped.transform.translation.x,
                          transformStamped.transform.translation.y,
                          transformStamped.transform.translation.z);

    // This is Twc (world to camera) from ROS
    Sophus::SE3f Twc_ros(quat, trans);

    if (first_frame) {
      T_init = Twc_ros;
      first_frame = false;
      pose = Sophus::SE3f();  // Identity for first frame
      return true;
    }

    // Sophus::SE3f T_init;

    // Get relative transform from first frame
    Sophus::SE3f Twc_relative = T_init.inverse() * Twc_ros;

    // ORBSLAM expects Tcw (camera to world), so invert
    pose = Twc_relative.inverse();

    return true;
  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s", ex.what());
    return false;
  }
}

void GaussianSLAMWrapper::initializeSLAMSystem() {
  ROS_INFO("Initializing SLAM system...");
  try {
    ORB_SLAM3::System::eSensor system_mode;
    if (mode_ == "stereo") {
      system_mode = ORB_SLAM3::System::STEREO;
    } else if (mode_ == "rgbd") {
      system_mode = ORB_SLAM3::System::RGBD;
    } else {
      system_mode = ORB_SLAM3::System::MONOCULAR;
    }

    slam_system_ = std::make_shared<ORB_SLAM3::System>(
        vocabulary_path_, orb_settings_path_, system_mode);
    ROS_INFO("SLAM system object created successfully");
  } catch (const std::exception &e) {
    ROS_ERROR("Exception during SLAM system initialization: %s", e.what());
    throw;
  }
}

void GaussianSLAMWrapper::initializeGaussianMapper() {
  ROS_INFO("Creating Gaussian Mapper...");

  // // Add multiple version checks
  // std::cout << "OpenCV version from macro: " << CV_VERSION << std::endl;
  // std::cout << "OpenCV version from string: " << cv::getVersionString()
  //           << std::endl;

  // Dl_info dl_info;
  // dladdr((void*)cv::getBuildInformation, &dl_info);
  // std::cout << "OpenCV library location: " << dl_info.dli_fname << std::endl;

  // // Print build information before creating mapper
  // std::cout << "OpenCV Build Information:\n"
  //           << cv::getBuildInformation() << std::endl;
  if (slam_mode_ == "external") {
    SystemSensorType sensor_type;

    if (mode_ == "mono") {
      sensor_type = MONOCULAR;
    } else if (mode_ == "stereo") {
      sensor_type = STEREO;
    } else if (mode_ == "rgbd") {
      sensor_type = RGBD;
    } else {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }

    gaussian_mapper_ = std::make_shared<GaussianMapper>(
        sensor_type, orb_settings_path_, gaussian_settings_path_,
        output_directory_,
        0,            // stream id
        torch::kCUDA  // assuming CUDA is available
    );

    mapper_thread_ = std::thread(&GaussianMapper::run_external_poses,
                                 gaussian_mapper_.get());

  } else if (slam_mode_ == "orbslam" || slam_mode_ == "hybrid") {
    gaussian_mapper_ = std::make_shared<GaussianMapper>(
        slam_system_, gaussian_settings_path_, output_directory_,
        0,            // stream id
        torch::kCUDA  // assuming CUDA is available
    );

    mapper_thread_ = std::thread(&GaussianMapper::run, gaussian_mapper_.get());
  }

  // // Add version check after mapper creation
  // std::cout << "OpenCV version after mapper creation: " << CV_VERSION
  //           << std::endl;

  if (use_viewer_) {
    ROS_INFO("Initializing viewer...");
    viewer_ = std::make_shared<ImGuiViewer>(slam_system_, gaussian_mapper_,
                                            true, (slam_mode_ == "external"));
    viewer_thread_ = std::thread(&ImGuiViewer::run, viewer_.get());
    ROS_INFO("Viewer initialized and started.");
  }
}

void GaussianSLAMWrapper::monoCallback(const sensor_msgs::ImageConstPtr &msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    if (msg->encoding == "bayer_rggb8") {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    }

    // ROS_INFO("Image size: %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);

    double timestamp = msg->header.stamp.toSec();
    // ROS_INFO("Processing frame with timestamp: %.6f", timestamp);

    if (slam_mode_ == "external") {
      ROS_ERROR("Mono doesn't support external mode right now!");
      return;
    }

    slam_system_->TrackMonocular(cv_ptr->image, timestamp,
                                 std::vector<ORB_SLAM3::IMU::Point>(),
                                 std::to_string(msg->header.seq));
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  } catch (const std::exception &e) {
    ROS_ERROR("Exception in TrackMonocular: %s", e.what());
  }
}

void GaussianSLAMWrapper::stereoCallback(
    const sensor_msgs::ImageConstPtr &msg_left,
    const sensor_msgs::ImageConstPtr &msg_right) {
  // ROS_INFO("Received stereo images. Left encoding: %s, Right encoding: %s",
  //          msg_left->encoding.c_str(), msg_right->encoding.c_str());

  cv_bridge::CvImageConstPtr cv_left, cv_right;
  try {
    // If images are Bayer RGGB8, convert to RGB8
    if (msg_left->encoding == "bayer_rggb8") {
      cv_left =
          cv_bridge::toCvCopy(msg_left, sensor_msgs::image_encodings::RGB8);
      cv_right =
          cv_bridge::toCvCopy(msg_right, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_left =
          cv_bridge::toCvShare(msg_left, sensor_msgs::image_encodings::RGB8);
      cv_right =
          cv_bridge::toCvShare(msg_right, sensor_msgs::image_encodings::RGB8);
    }

    // ROS_INFO("Image sizes - Left: %dx%d, Right: %dx%d", cv_left->image.cols,
    //          cv_left->image.rows, cv_right->image.cols,
    //          cv_right->image.rows);

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Get timestamp from message
  double timestamp = msg_left->header.stamp.toSec();
  // ROS_INFO("Processing frame with timestamp: %.6f", timestamp);

  try {
    if (slam_mode_ == "external") {
      Sophus::SE3f Twc;
      if (getExternalPose(Twc)) {
        // Process frame with GT pose
        gaussian_mapper_->handleNewFrameExternal(
            cv_left->image, cv_right->image, Twc, timestamp);
      }
    } else if (slam_mode_ == "hybrid") {
      ROS_ERROR("Hyrbid not yet implemented for stereo");
    } else if (slam_mode_ == "orbslam") {
      if (!slam_system_) {
        ROS_ERROR("SLAM system pointer is null!");
        return;
      }

      try {
        slam_system_->TrackStereo(cv_left->image, cv_right->image, timestamp,
                                  std::vector<ORB_SLAM3::IMU::Point>(),
                                  std::to_string(msg_left->header.seq));
      } catch (const std::exception &e) {
        ROS_ERROR("Exception in TrackStereo: %s", e.what());
      }
    }
  } catch (const std::exception &e) {
    ROS_ERROR("Exception in TrackStereo: %s", e.what());
  }
}

void GaussianSLAMWrapper::rgbdCallback(
    const sensor_msgs::ImageConstPtr &msg_rgb,
    const sensor_msgs::ImageConstPtr &msg_depth) {
  cv_bridge::CvImageConstPtr cv_rgb, cv_depth;
  try {
    // Convert RGB image
    if (msg_rgb->encoding == "bayer_rggb8") {
      cv_rgb = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_rgb =
          cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::RGB8);
    }

    // Handle depth image based on its encoding
    cv::Mat depth_converted;
    if (msg_depth->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      cv_depth = cv_bridge::toCvShare(msg_depth,
                                      sensor_msgs::image_encodings::TYPE_32FC1);
      // Convert from meters to millimeters
      cv_depth->image.convertTo(depth_converted, CV_16UC1, 1000);
      // Replace inf/nan with 0
      depth_converted.setTo(0, cv_depth->image <= 0);
    } else {
      // Assume 16UC1 and use as is
      cv_depth = cv_bridge::toCvShare(msg_depth,
                                      sensor_msgs::image_encodings::TYPE_16UC1);
      depth_converted = cv_depth->image;
    }

    // ROS_INFO("Image sizes - RGB: %dx%d, Depth: %dx%d", cv_rgb->image.cols,
    //          cv_rgb->image.rows, cv_depth->image.cols, cv_depth->image.rows);

    // Get timestamp from message
    double timestamp = msg_rgb->header.stamp.toSec();

    try {
      if (slam_mode_ == "external") {
        Sophus::SE3f Twc;
        if (getExternalPose(Twc)) {
          gaussian_mapper_->handleNewFrameExternal(
              cv_rgb->image, cv_depth->image, Twc, timestamp);
        }

      } else if (slam_mode_ == "hybrid") {
        Sophus::SE3f Twc;
        if (getExternalPose(Twc)) {
          try {
            slam_system_->TrackRGBDWithPose(
                cv_rgb->image.clone(), depth_converted.clone(), Twc, timestamp,
                std::to_string(msg_rgb->header.seq));
          } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
          } catch (...) {
            std::cerr << "Unknown exception!" << std::endl;
          }
        }

      } else if (slam_mode_ == "orbslam") {
        if (!slam_system_) {
          ROS_ERROR("SLAM system pointer is null!");
          return;
        }

        try {
          slam_system_->TrackRGBD(cv_rgb->image, depth_converted, timestamp,
                                  std::vector<ORB_SLAM3::IMU::Point>(),
                                  std::to_string(msg_rgb->header.seq));
        } catch (const std::exception &e) {
          ROS_ERROR("Exception in TrackRGBD: %s", e.what());
        }
      }
    } catch (const std::exception &e) {
      ROS_ERROR("Exception in TrackRGBD: %s", e.what());
    }

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

GaussianSLAMWrapper::~GaussianSLAMWrapper() {
  if (slam_system_) {
    slam_system_->Shutdown();
  }

  if (mapper_thread_.joinable()) {
    mapper_thread_.join();
  }

  if (viewer_thread_.joinable()) {
    viewer_thread_.join();
  }
}