cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(lsgs_ros LANGUAGES CXX CUDA)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_DEPENDS_IN_PROJECT_ONLY ON)

# Set OpenCV_DIR to the installation directory
set(OpenCV_DIR "/workspaces/large_scale_gaussian_slam/third_party/install/opencv/lib/cmake/opencv4")
if(NOT EXISTS "${OpenCV_DIR}")
    message(FATAL_ERROR "Custom OpenCV installation not found at ${OpenCV_DIR}")
endif()

# Find OpenCV first
find_package(OpenCV 4.10 REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

# Set RPATH settings
list(APPEND CMAKE_INSTALL_RPATH 
    "${LSGS_MAIN_DIR}/third_party/install/opencv/lib"
    "${LSGS_MAIN_DIR}/lib"
    "${ORB_SLAM3_SOURCE_DIR}/lib"
    "${LSGS_MAIN_DIR}/third_party/libtorch/lib"
)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Ensure no system directories are added to RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)

# Add debug symbols
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add debug flags for both C++ and CUDA
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# Find catkin and required ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
  tf2
  tf2_ros
  tf2_eigen
)

# Same configurations as your original project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CAFFE2_USE_CUDNN on)

set(LSGS_MAIN_DIR "/workspaces/large_scale_gaussian_slam")

# Point directly to the Torch config file
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${LSGS_MAIN_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(jsoncpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

# Set the path to your project root and ORB_SLAM3
set(GAUSSIAN_SLAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ORB_SLAM3_SOURCE_DIR "${GAUSSIAN_SLAM_DIR}/third_party/ORB-SLAM3")

# Specify catkin package information
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport message_filters
  DEPENDS EIGEN3 OpenCV
)

# Include directories - note the BEFORE keyword to prioritize custom OpenCV
include_directories(BEFORE
    ${OpenCV_INCLUDE_DIRS}
    include
    ${catkin_INCLUDE_DIRS}
    ${LSGS_MAIN_DIR}
    ${LSGS_MAIN_DIR}/third_party
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Add the ROS wrapper library
add_library(${PROJECT_NAME}
  src/lsgs_wrapper.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${LSGS_MAIN_DIR}/lib/libgaussian_mapper.so
  ${LSGS_MAIN_DIR}/lib/libgaussian_viewer.so
  ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TORCH_LIBRARIES}
  Eigen3::Eigen
)

# Add the ROS node executable
add_executable(${PROJECT_NAME}_node src/lsgs_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/pose_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)