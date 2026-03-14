cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(diskchungs LANGUAGES CXX CUDA)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Building for ARM / Jetson (aarch64)")
    set(IS_ARM TRUE)
else()
    message(STATUS "Building for x86/x86_64")
    set(IS_ARM FALSE)
endif()

set_property(GLOBAL PROPERTY JOB_POOLS link_pool=2)
set(CMAKE_JOB_POOL_LINK link_pool)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

if(IS_ARM)
    # Updated path to libtorch in the jetson Docker image
    execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_CMAKE_PREFIX_PATH})
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    # Updated path to libtorch in the base Docker image
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/workspace/third_party/libtorch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
endif()

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# Updated path to OpenCV - using the pre-built version in the base image
set(OpenCV_DIR "/workspace/third_party/install/opencv/lib/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

find_package(jsoncpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)


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
    ${PROJECT_SOURCE_DIR}/include
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
    include/geometry/operate_points.h
    src/geometry/operate_points.cu
    include/depth/stereo_vision.h
    src/depth/stereo_vision.cu
    src/depth/guided_mvs.cu
    include/utils/cuda_utils.h
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

if(IS_ARM)
    set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "87")
else()
    set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "86")
endif()

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
    # Core headers
    include/gaussian_mapper.h
    include/camera.h
    include/types.h
    include/chunk_types.h
    # Scene
    include/scene/gaussian_keyframe.h
    include/scene/gaussian_parameters.h
    include/scene/gaussian_scene.h
    include/scene/keyframe_selection.h
    # Model
    include/model/gaussian_model.h
    # Rendering
    include/rendering/gaussian_rasterizer.h
    include/rendering/gaussian_renderer.h
    include/rendering/frustum_culler.h
    # Depth
    include/depth/stereo_depth.h
    include/depth/mono_depth.h
    include/depth/guided_mvs.h
    # Geometry
    include/geometry/point_cloud.h
    include/geometry/point2d.h
    include/geometry/point3d.h
    # Utils
    include/utils/general_utils.h
    include/utils/graphics_utils.h
    include/utils/loss_utils.h
    include/utils/sh_utils.h
    include/utils/tensor_utils.h
    include/utils/profiling.h
    include/utils/debugging_utils.h
    include/utils/trajectory_viewer.h
    # Gaussian mapper implementation
    src/mapper/mapper_config.cpp
    src/mapper/mapper_initialization.cpp
    src/mapper/mapper_core.cpp
    src/mapper/mapper_operations.cpp
    src/mapper/mapper_io.cpp
    src/mapper/mapper_parameters.cpp
    src/mapper/mapper_external.cpp
    src/mapper/mapper_depth.cpp
    # Model implementation
    src/model/model_core.cpp
    src/model/model_memory.cpp
    src/model/model_optimization.cpp
    src/model/model_pruning.cpp
    src/model/model_storage.cpp
    # Scene implementation
    src/scene/gaussian_keyframe.cpp
    src/scene/gaussian_parameters.cpp
    src/scene/gaussian_scene.cpp
    src/scene/keyframe_selection.cpp
    # Rendering implementation
    src/rendering/gaussian_rasterizer.cpp
    src/rendering/gaussian_renderer.cpp
    src/rendering/frustum_culler.cpp
    # Depth implementation
    src/depth/stereo_depth.cpp
    src/depth/mono_depth.cpp
    src/depth/guided_mvs.cpp
    # Utils implementation
    src/utils/debugging_utils.cpp
    src/utils/trajectory_viewer.cpp
)
target_link_libraries(
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    cuda_rasterizer
    xfeat
    OpenMP::OpenMP_CXX
    depth_anything_tensorrt 
    ${TENSORRT_LIBS} 
)

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

# Replica RGBD
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

# KITTI Stereo
add_executable(kitti_stereo examples/kitti_stereo.cpp)
target_link_libraries(kitti_stereo
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################