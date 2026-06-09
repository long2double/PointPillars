# 设置CMake的最小版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(trt_infer LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "120")

# 查找CUDA库
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("/usr/local/TensorRT-10.16.1.11/include")
link_directories("/usr/local/TensorRT-10.16.1.11/lib")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE infer_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
# 指定生成目标
add_executable(trt_infer ${infer_src} trt_infer.cpp)

# 链接库
target_link_libraries(trt_infer cudart nvinfer nvinfer_plugin ${CUDA_LIBRARIES})