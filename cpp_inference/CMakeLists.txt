# CMakeLists.txt for C++ Inference Core
# 混合架构的C++推理模块构建系统

cmake_minimum_required(VERSION 3.16)

project(BambooInferenceCore
    VERSION 1.0.0
    DESCRIPTION "Bamboo Detection System C++ Inference Core with Python Bindings"
    LANGUAGES CXX CUDA
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# CUDA编译选项
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖包
find_package(PkgConfig REQUIRED)

# 查找CUDA
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA not found! Please install CUDA Toolkit.")
endif()

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_TARGET_DIR}")

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Please install OpenCV.")
endif()

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")

# 查找TensorRT
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT installation root")

# 尝试多个可能的TensorRT路径
set(TENSORRT_POSSIBLE_PATHS
    "/usr/local/TensorRT"
    "/opt/TensorRT"
    "/usr/local/tensorrt"
    "/opt/tensorrt"
    "$ENV{TENSORRT_ROOT}"
)

foreach(path ${TENSORRT_POSSIBLE_PATHS})
    if(EXISTS "${path}/include/NvInfer.h")
        set(TENSORRT_ROOT ${path})
        break()
    endif()
endforeach()

if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT not found! Please set TENSORRT_ROOT or install TensorRT.")
endif()

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

# 检查TensorRT库文件
set(TENSORRT_LIBS
    nvinfer
    nvinfer_plugin  
    nvonnxparser
    nvparsers
)

foreach(lib ${TENSORRT_LIBS})
    find_library(${lib}_LIBRARY
        NAMES ${lib}
        PATHS ${TENSORRT_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    if(NOT ${lib}_LIBRARY)
        message(FATAL_ERROR "TensorRT library ${lib} not found in ${TENSORRT_LIB_DIR}")
    endif()
    
    message(STATUS "Found TensorRT ${lib}: ${${lib}_LIBRARY}")
endforeach()

message(STATUS "TensorRT Root: ${TENSORRT_ROOT}")
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIB_DIR}")

# 查找pybind11
find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found! Please install pybind11.")
endif()

message(STATUS "pybind11 Version: ${pybind11_VERSION}")

# 查找Python
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS "Python Version: ${Python_VERSION}")
message(STATUS "Python Include: ${Python_INCLUDE_DIRS}")
message(STATUS "NumPy Include: ${Python_NumPy_INCLUDE_DIRS}")

# 包含目录
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)

# 工具函数：添加实现文件
write_to_file(
<path>cpp_inference/utils.cpp</path>
<content>/**
 * 工具函数实现
 */

#include "inference_core.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

namespace BambooUtils {

CudaDeviceInfo getCudaDeviceInfo() {
    CudaDeviceInfo info;
    
    // 获取CUDA设备数量
    cudaGetDeviceCount(&info.device_count);
    
    if (info.device_count > 0) {
        // 获取当前设备
        cudaGetDevice(&info.current_device);
        
        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, info.current_device);
        
        info.device_name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        
        // 获取可用内存
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;
    }
    
    return info;
}

std::string getTensorRTVersion() {
    std::stringstream ss;
    ss << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH;
    return ss.str();
}

std::string getOpenCVVersion() {
    return CV_VERSION;
}

bool setupLogging(const std::string& log_level) {
    // 简单的日志设置实现
    if (log_level == "DEBUG" || log_level == "INFO" || log_level == "WARNING" || log_level == "ERROR") {
        std::cout << "[Logger] 设置日志级别为: " << log_level << std::endl;
        return true;
    }
    return false;
}

} // namespace BambooUtils
</content>
<line_count>51</line_count>
)

# 添加缺失的性能统计方法
file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/inference_core.cpp "

// 更新性能统计
void BambooDetector::updatePerformanceStats(float processing_time_ms) {
    perf_stats_.total_detections++;
    perf_stats_.total_time_ms += processing_time_ms;
    perf_stats_.avg_time_ms = perf_stats_.total_time_ms / perf_stats_.total_detections;
    
    if (perf_stats_.total_detections == 1) {
        perf_stats_.min_time_ms = processing_time_ms;
        perf_stats_.max_time_ms = processing_time_ms;
    } else {
        if (processing_time_ms < perf_stats_.min_time_ms) {
            perf_stats_.min_time_ms = processing_time_ms;
        }
        if (processing_time_ms > perf_stats_.max_time_ms) {
            perf_stats_.max_time_ms = processing_time_ms;
        }
    }
    
    perf_stats_.fps = perf_stats_.avg_time_ms > 0.0f ? 1000.0f / perf_stats_.avg_time_ms : 0.0f;
}

// 重置性能统计
void BambooDetector::resetPerformanceStats() {
    perf_stats_ = PerformanceStats();
}

// 批量检测
std::vector<DetectionResult> BambooDetector::detectBatch(const std::vector<cv::Mat>& images) {
    std::vector<DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(detect(image));
    }
    
    return results;
}
")

# 源文件列表
set(INFERENCE_CORE_SOURCES
    inference_core.cpp
    utils.cpp
)

set(PYTHON_BINDING_SOURCES
    python_bindings.cpp
)

# 创建C++推理核心静态库
add_library(bamboo_inference_core STATIC ${INFERENCE_CORE_SOURCES})

# 设置编译属性
target_include_directories(bamboo_inference_core PUBLIC ${INCLUDE_DIRS})

# 链接库
target_link_libraries(bamboo_inference_core
    ${OpenCV_LIBS}
    ${nvinfer_LIBRARY}
    ${nvinfer_plugin_LIBRARY}
    ${nvonnxparser_LIBRARY}
    ${nvparsers_LIBRARY}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cudnn
)

# 设置CUDA属性
set_property(TARGET bamboo_inference_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 创建Python绑定模块
pybind11_add_module(cpp_inference_core ${PYTHON_BINDING_SOURCES})

# 链接推理核心库
target_link_libraries(cpp_inference_core PRIVATE bamboo_inference_core)

# 设置编译属性
target_include_directories(cpp_inference_core PRIVATE ${INCLUDE_DIRS})

# Python模块属性
target_compile_definitions(cpp_inference_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# 设置输出名称
set_target_properties(cpp_inference_core PROPERTIES
    OUTPUT_NAME cpp_inference_core
    PREFIX ""
    SUFFIX ".so"
)

# 安装规则
install(TARGETS bamboo_inference_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS cpp_inference_core
    LIBRARY DESTINATION python_core
)

# 复制头文件
install(FILES inference_core.h
    DESTINATION include
)

# 创建pkg-config文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bamboo-inference-core.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/bamboo-inference-core.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bamboo-inference-core.pc
    DESTINATION lib/pkgconfig
)

# 测试目标（可选）
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 示例程序（可选）
option(BUILD_EXAMPLES "Build examples" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 打印构建信息
message(STATUS "")
message(STATUS "=== Bamboo Inference Core Build Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT: ${TENSORRT_ROOT}")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  Python: ${Python_VERSION}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  bamboo_inference_core (static library)")
message(STATUS "  cpp_inference_core (Python module)")
message(STATUS "")