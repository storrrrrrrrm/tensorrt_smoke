cmake_minimum_required(VERSION 2.8.0)
project(smoke_trt)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER NAMES nvinfer)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_library(NVONNXPARSER nvonnxparser)


include_directories(
  ${CUDA_INCLUDE_DIRS}
  # ~/TensorRT-8.0.3.4/include
  plugins/modulated_deform_conv
)

cuda_add_library(trt_modulated_deform_conv_plugin SHARED
  plugins/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu
  plugins/modulated_deform_conv/trt_modulated_deform_conv.cpp
)

add_executable(${PROJECT_NAME}
  smoke.cpp
  main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${NVINFER}
  ${NVONNXPARSER}
  ${NVINFER_PLUGIN}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  trt_modulated_deform_conv_plugin
)
