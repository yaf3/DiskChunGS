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
  pnh_.param<std::string>("imu_topic", imu_topic_, "/boxi/zed2i/imu/data");
  pnh_.param<std::string>("slam_mode", slam_mode_, "orbslam");
  pnh_.param<std::string>("target_frame", target_frame_, "map");
  pnh_.param<std::string>("source_frame", source_frame_,
                          "zed2i_left_camera_frame");
  pnh_.param<double>("timeout_duration", timeout_duration_, 20.0);

  ROS_INFO("Parameters loaded:");
  ROS_INFO("  mode: %s", mode_.c_str());
  ROS_INFO("  vocabulary_path: %s", vocabulary_path_.c_str());
  ROS_INFO("  orb_settings_path: %s", orb_settings_path_.c_str());
  ROS_INFO("  gaussian_settings_path: %s", gaussian_settings_path_.c_str());
  ROS_INFO("  output_directory: %s", output_directory_.c_str());
  ROS_INFO("  timeout_duration: %.1f seconds", timeout_duration_);
  ROS_INFO("  use_viewer: %d", use_viewer_);
  ROS_INFO("  slam_mode: %s", slam_mode_.c_str());
  if (slam_mode_ == "external" || slam_mode_ == "hybrid") {
    ROS_INFO("  target_frame: %s", target_frame_.c_str());
    ROS_INFO("  source_frame: %s", source_frame_.c_str());
  }
  if (mode_ == "rgbd-imu" || mode_ == "stereo-imu") {
    ROS_INFO("  imu_topic: %s", imu_topic_.c_str());
  }

  timeout_timer_ = nh_.createTimer(ros::Duration(1.0),
                                   &GaussianSLAMWrapper::timeoutCallback, this);
  status_check_timer_ = nh_.createTimer(
      ros::Duration(1.0), &GaussianSLAMWrapper::checkMappingStatus, this);

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
  if (mode_ == "stereo" || mode_ == "stereo-imu") {
    left_sub_.subscribe(nh_, left_topic_, 1);
    right_sub_.subscribe(nh_, right_topic_, 1);
    sync_.reset(new message_filters::Synchronizer<sync_pol>(
        sync_pol(30), left_sub_, right_sub_));
  } else if (mode_ == "rgbd" || mode_ == "rgbd-imu") {
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
    // Subscribe to IMU data if using an IMU mode
  } else if (mode_ == "rgbd-imu") {
    ROS_INFO("Registering RGB-D callback...");
    ROS_INFO("RGB topic: %s", rgb_sub_.getTopic().c_str());
    ROS_INFO("Depth topic: %s", depth_sub_.getTopic().c_str());
    rgbd_sync_->registerCallback(
        boost::bind(&GaussianSLAMWrapper::rgbdCallback, this, _1, _2));
    imu_sub_ = nh_.subscribe(imu_topic_, 1000,
                             &GaussianSLAMWrapper::imuCallback, this);
    ROS_INFO("Subscribed to IMU topic: %s", imu_topic_.c_str());

  } else {
    throw std::runtime_error("Invalid mode: " + mode_);
  }

  ROS_INFO("GaussianSLAMWrapper initialization complete!");
}

bool GaussianSLAMWrapper::getExternalPose(Sophus::SE3f &pose,
                                          double timestamp) {
  // First attempt with waitForTransform to block until the transform is
  // available
  if (!tfBuffer.canTransform(target_frame_, source_frame_, ros::Time(timestamp),
                             ros::Duration(0.5))) {
    ROS_WARN("Transform from %s to %s not available yet, waiting...",
             source_frame_.c_str(), target_frame_.c_str());
    return false;
  }
  try {
    // Now try to lookup the transform
    geometry_msgs::TransformStamped transformStamped = tfBuffer.lookupTransform(
        target_frame_, source_frame_, ros::Time(timestamp));

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

    // if (first_frame) {
    //   T_init = Twc_ros;
    //   first_frame = false;
    //   pose = Sophus::SE3f();  // Identity for first frame
    //   return true;
    // }

    // Sophus::SE3f T_init;

    // Get relative transform from first frame
    // Sophus::SE3f Twc_relative = T_init.inverse() * Twc_ros;

    // ORBSLAM expects Tcw (camera to world), so invert
    pose = Twc_ros.inverse();

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
    // Select the appropriate sensor mode
    if (mode_ == "stereo") {
      system_mode = ORB_SLAM3::System::STEREO;
    } else if (mode_ == "stereo-imu") {
      system_mode = ORB_SLAM3::System::IMU_STEREO;
    } else if (mode_ == "rgbd") {
      system_mode = ORB_SLAM3::System::RGBD;
    } else if (mode_ == "rgbd-imu") {
      system_mode =
          ORB_SLAM3::System::IMU_RGBD;  // Using the built-in IMU_RGBD mode
    } else {
      system_mode = ORB_SLAM3::System::MONOCULAR;
    }

    slam_system_ = std::make_shared<ORB_SLAM3::System>(
        vocabulary_path_, orb_settings_path_, system_mode);
    ROS_INFO("SLAM system object created successfully with mode: %d",
             static_cast<int>(system_mode));
    ROS_INFO("SLAM system object created successfully");
  } catch (const std::exception &e) {
    ROS_ERROR("Exception during SLAM system initialization: %s", e.what());
    throw;
  }
}

void GaussianSLAMWrapper::imuCallback(const sensor_msgs::ImuConstPtr &msg) {
  std::lock_guard<std::mutex> lock(imu_mutex_);

  // Extract IMU measurements
  const double ax = msg->linear_acceleration.x;
  const double ay = msg->linear_acceleration.y;
  const double az = msg->linear_acceleration.z;
  const double gx = msg->angular_velocity.x;
  const double gy = msg->angular_velocity.y;
  const double gz = msg->angular_velocity.z;
  const double timestamp = msg->header.stamp.toSec();

  // Create IMU measurement point and add to buffer
  ORB_SLAM3::IMU::Point imu_point(ax, ay, az, gx, gy, gz, timestamp);
  imu_buffer_.push_back(imu_point);

  // Optionally, limit buffer size to prevent unbounded growth
  const size_t MAX_IMU_BUFFER_SIZE = 1000;
  if (imu_buffer_.size() > MAX_IMU_BUFFER_SIZE) {
    imu_buffer_.erase(imu_buffer_.begin());
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

    gaussian_mapper_->setCompletionCallback(
        [this]() { this->mapping_completed_.store(true); });

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
  updateCallbackTime();
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
  updateCallbackTime();
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
      if (getExternalPose(Twc, timestamp)) {
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
  updateCallbackTime();
  cv_bridge::CvImageConstPtr cv_rgb, cv_depth;
  try {
    // Convert RGB image (existing code)
    if (msg_rgb->encoding == "bayer_rggb8") {
      cv_rgb = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_rgb =
          cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::RGB8);
    }

    // Handle depth image (existing code)
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

    // Get timestamp from message
    double timestamp = msg_rgb->header.stamp.toSec();

    // CRITICAL: Process IMU data for the frame
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);

      // Find relevant IMU measurements for this frame
      if (mode_ == "rgbd-imu") {
        // Only use IMU measurements between the last frame and this one
        double min_time = last_processed_image_ts_;
        std::cout << "min_time: " << min_time << std::endl;
        if (min_time == 0) {
          // For the first frame, use a window before the current timestamp
          min_time = timestamp - 0.1;  // 100ms window before first frame
        }

        // Collect all IMU measurements in the window
        for (const auto &imu_point : imu_buffer_) {
          if (imu_point.t >= min_time && imu_point.t <= timestamp) {
            vImuMeas.push_back(imu_point);
          }
        }

        // Log IMU integration status
        if (vImuMeas.empty()) {
          ROS_WARN(
              "No IMU measurements for frame at time %.3f (last frame: %.3f)",
              timestamp, last_processed_image_ts_);
          ROS_WARN("Buffer has %zu IMU measurements", imu_buffer_.size());
          if (!imu_buffer_.empty()) {
            ROS_WARN("IMU buffer time range: %.3f to %.3f",
                     imu_buffer_.front().t, imu_buffer_.back().t);
          }
        } else {
          ROS_INFO("Using %zu IMU measurements for frame at time %.3f",
                   vImuMeas.size(), timestamp);
        }

        // Update last processed timestamp
        last_processed_image_ts_ = timestamp;
      }
    }

    // Tracking logic
    try {
      if (slam_mode_ == "external") {
        Sophus::SE3f Twc;
        if (getExternalPose(Twc, timestamp)) {
          // std::cout << "handleNewFrameExternal called" << std::endl;
          gaussian_mapper_->handleNewFrameExternal(
              cv_rgb->image, cv_depth->image, Twc, timestamp);
          // std::cout << "handleNewFrameExternal finished" << std::endl;
        }

      } else if (slam_mode_ == "hybrid") {
        Sophus::SE3f Twc;
        if (getExternalPose(Twc, timestamp)) {
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

        // Track with or without IMU data
        slam_system_->TrackRGBD(cv_rgb->image, depth_converted, timestamp,
                                vImuMeas, std::to_string(msg_rgb->header.seq));
      }
      // Other modes (external, hybrid) remain unchanged

    } catch (const std::exception &e) {
      ROS_ERROR("Exception in TrackRGBD: %s", e.what());
    }

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}
void GaussianSLAMWrapper::timeoutCallback(const ros::TimerEvent &event) {
  std::lock_guard<std::mutex> lock(timeout_mutex_);

  // Only check for timeout if we've started receiving data and haven't already
  // stopped
  if (data_started_ && !stopped_) {
    ros::Duration elapsed = ros::Time::now() - last_callback_time_;

    if (elapsed.toSec() > timeout_duration_) {
      ROS_INFO(
          "No callbacks received for %.1f seconds, signaling data stream "
          "stopped",
          elapsed.toSec());

      // Signal to the gaussian mapper that the external data has stopped
      if (gaussian_mapper_) {
        ROS_INFO("Calling signalExternalDataStopped on gaussian mapper");
        gaussian_mapper_->signalExternalDataStopped();
        stopped_ = true;
        ROS_INFO("Called signalExternalDataStopped on gaussian mapper");
      }
    }
  }
}

// Add a helper method to update the last callback time
void GaussianSLAMWrapper::updateCallbackTime() {
  std::lock_guard<std::mutex> lock(timeout_mutex_);
  last_callback_time_ = ros::Time::now();
  if (!data_started_) {
    data_started_ = true;
    ROS_INFO("Data stream has started, timeout monitoring active");
  }
}

void GaussianSLAMWrapper::checkMappingStatus(const ros::TimerEvent &event) {
  // Check if mapper has signaled completion
  if (mapping_completed_) {
    ROS_INFO("Mapping process complete, shutting down node.");
    ros::shutdown();
  }
}

GaussianSLAMWrapper::~GaussianSLAMWrapper() {
  // Stop the timeout timer
  timeout_timer_.stop();

  std::cout << "Shutting down GaussianSLAMWrapper..." << std::endl;

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