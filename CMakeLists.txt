cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(lsgs LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY JOB_POOLS link_pool=2)
set(CMAKE_JOB_POOL_LINK link_pool)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Check for AVX support
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX)
# check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)

# if(COMPILER_SUPPORTS_AVX AND COMPILER_SUPPORTS_AVX2)
#     message(STATUS "Compiler supports AVX and AVX2 - enabling optimizations")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2")
#     add_definitions(-DUSE_AVX)
# else()
#     message(STATUS "Compiler does not support AVX - using fallback implementation")
# endif()

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

# Updated path to libtorch in the base Docker image
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/workspace/third_party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# Updated path to OpenCV - using the pre-built version in the base image
set(OpenCV_DIR "/workspace/third_party/install/opencv/lib/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

find_package(jsoncpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(realsense2)
find_package(OpenMP REQUIRED)

# Find ONNX Runtime
find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS /usr/local/lib /usr/lib
    REQUIRED
)

find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include /usr/include
    REQUIRED
)

message(STATUS "Found ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")


# # Find TensorRT
# find_path(TENSORRT_INCLUDE_DIR 
#     NAMES NvInfer.h
#     PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
#     REQUIRED
# )

# find_library(TENSORRT_LIB 
#     NAMES nvinfer
#     PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
#     REQUIRED
# )

# # Check for TensorRT provider library (modern approach)
# find_library(ONNXRUNTIME_TENSORRT_PROVIDER_LIB 
#     NAMES onnxruntime_providers_tensorrt
#     PATHS /usr/local/lib /usr/lib
# )

# if(ONNXRUNTIME_TENSORRT_PROVIDER_LIB)
#     message(STATUS "ONNX Runtime TensorRT provider found: ${ONNXRUNTIME_TENSORRT_PROVIDER_LIB}")
#     set(HAVE_TENSORRT_PROVIDER TRUE)
# else()
#     message(WARNING "ONNX Runtime TensorRT provider not found - TensorRT support disabled")
#     set(HAVE_TENSORRT_PROVIDER FALSE)
# endif()

# # Add TensorRT and ONNX Runtime include directories
# include_directories(
#     ${TENSORRT_INCLUDE_DIR}
#     ${ONNXRUNTIME_INCLUDE_DIR}
# )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread ${OpenMP_CXX_FLAGS}")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER)
    set(TENSORRT_FOUND TRUE)
    set(TENSORRT_DIR ${TENSORRT_INCLUDE_DIR}/..)
endif()

if(NOT TENSORRT_FOUND)
    message(WARNING "TensorRT not found. TensorRT depth estimation will be disabled.")
else()
    # Read TensorRT version
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version REGEX "#define NV_TENSORRT_MAJOR +[0-9]+")
    string(REGEX MATCH "[0-9]+" tensorrt_version_major ${tensorrt_version})
    
    # Check TensorRT version and set library accordingly
    if (WIN32)
        if (tensorrt_version_major EQUAL 10)
            set(TENSORRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
        else()
            set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
        endif()
    else()
        if (tensorrt_version_major EQUAL 10)
            set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
        else()
            set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
        endif()
    endif()
    
    message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT version: ${tensorrt_version_major}")
endif()


set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/slam_deps/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/slam_deps
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    simple_knn
    SHARED
    slam_deps/simple-knn/simple_knn.cu
    slam_deps/simple-knn/simple_knn.h
    slam_deps/simple-knn/spatial.cu
    slam_deps/simple-knn/spatial.h
)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn ${TORCH_LIBRARIES})

add_library(
    xfeat
    SHARED
    slam_deps/xfeat_cpp/src/XFModel.cc
    slam_deps/xfeat_cpp/src/XFeat.cc
    slam_deps/xfeat_cpp/src/InterpolateSparse2d.cc
)
target_include_directories(xfeat PRIVATE ${PROJECT_SOURCE_DIR}/slam_deps/xfeat_cpp/include)
target_compile_options(xfeat PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(xfeat ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})

add_library(
    depth_anything_tensorrt
    SHARED
    slam_deps/depth-anything-tensorrt/depth_anything.cpp
    slam_deps/depth-anything-tensorrt/utils.cpp
)
target_include_directories(depth_anything_tensorrt PRIVATE 
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
    depth_anything_tensorrt
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

add_library(
    cuda_rasterizer
    SHARED
    include/operate_points.h
    src/operate_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    src/guided_mvs.cu
    include/cuda_utils.h
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterize_points.h
    cuda_rasterizer/rasterize_points.cu
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h
    cuda_rasterizer/adam.cu
    cuda_rasterizer/adam.h
    cuda_rasterizer/ssim.cu
    cuda_rasterizer/ssim.h
)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(
    cuda_rasterizer
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(cuda_rasterizer glm::glm ${TORCH_LIBRARIES} Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    imgui
    SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h
)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    gaussian_mapper
    SHARED
    slam_deps/tinyply/tinyply.h
    slam_deps/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    include/profiling.h
    include/chunk_types.h
    include/render_flythrough.h
    include/keyframe_selection.h
    include/nanoflann.hpp
    include/debugging_utils.h
    include/stereo_depth.h
    include/mono_depth.h
    include/guided_mvs.h
    include/frustum_culler.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_mapper.cpp
    src/render_flythrough.cpp
    src/keyframe_selection.cpp
    src/debugging_utils.cpp
    src/stereo_depth.cpp
    src/mono_depth.cpp
    src/guided_mvs.cpp
    src/frustum_culler.cpp
)
target_link_libraries(
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    xfeat
    OpenMP::OpenMP_CXX
    ${ONNXRUNTIME_LIB}
    depth_anything_tensorrt  # <-- ADD THIS LINE
    ${TENSORRT_LIBS}         # <-- ADD THIS LINE
)

# if(HAVE_TENSORRT_PROVIDER)
#     target_compile_definitions(gaussian_mapper PRIVATE -DHAVE_TENSORRT_PROVIDER)
#     target_link_libraries(gaussian_mapper ${TENSORRT_LIB} ${ONNXRUNTIME_TENSORRT_PROVIDER_LIB})
# endif()

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    gaussian_viewer
    SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h
)
target_link_libraries(
    gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    ${TORCH_LIBRARIES}
    glm::glm
    glfw
    OpenGL::GL
)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result gaussian_viewer gaussian_mapper)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Replica Monocular
add_executable(replica_mono examples/replica_mono.cpp)
target_link_libraries(
    replica_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# Replica Monocular
add_executable(replica_rgbd examples/replica_rgbd.cpp)
target_link_libraries(
    replica_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# TUM Monocular
add_executable(tum_mono examples/tum_mono.cpp)
target_link_libraries(
    tum_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# TUM RGBD
add_executable(tum_rgbd examples/tum_rgbd.cpp)
target_link_libraries(
    tum_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# KITTI Monocular
add_executable(kitti_mono examples/kitti_mono.cpp)
target_link_libraries(kitti_mono
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# KITTI Stereo
add_executable(kitti_stereo examples/kitti_stereo.cpp)
target_link_libraries(kitti_stereo
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# KITTI RGBD
add_executable(kitti_rgbd examples/kitti_rgbd.cpp)
target_link_libraries(kitti_rgbd
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# VKITTI RGBD
add_executable(vkitti_rgbd examples/vkitti_rgbd.cpp)
target_link_libraries(vkitti_rgbd
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# ACVNET Stereo Test
add_executable(test_acvnet examples/test_acvnet.cpp)
target_link_libraries(test_acvnet
    ${ONNXRUNTIME_LIB}
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# ACVNET Stereo Test
# add_executable(inspect_onnx examples/inspect_onnx.cpp)
# target_link_libraries(inspect_onnx
#     ${ONNXRUNTIME_LIB}
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# Camera Mono
add_executable(camera_mono examples/camera_mono.cpp)
target_link_libraries(camera_mono
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)


# Xfeat matching
add_executable(match examples/match.cpp)
target_link_libraries(match
    gaussian_mapper)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Intel Realsense
if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC ${realsense_INCLUDE_DIR})
    target_link_libraries(
        realsense_rgbd
        gaussian_viewer
        gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${realsense2_LIBRARY}
    )
endif()