# C++推理核心 CMake配置
# 版本: 4.0.0 (Python LVGL Architecture)
# C++推理后端 + Python LVGL前端的混合架构

cmake_minimum_required(VERSION 3.16)
project(BambooInferenceCore VERSION 4.0.0 LANGUAGES C CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# AI优化特性选项
option(ENABLE_AI_OPTIMIZATION "Enable AI optimizations" ON)
option(ENABLE_NAM_ATTENTION "Enable NAM Attention" ON)
option(ENABLE_GHOST_CONV "Enable GhostConv" ON)
option(ENABLE_VOV_GSCSP "Enable VoV-GSCSP" ON)
option(ENABLE_WISE_IOU "Enable Wise-IoU" ON)
option(ENABLE_SAHI_SLICING "Enable SAHI Slicing" ON)
option(ENABLE_HARDWARE_ACCELERATION "Enable Hardware Acceleration" ON)
option(ENABLE_TENSORRT "Enable TensorRT" ON)
option(ENABLE_CUDA "Enable CUDA" ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")

# 检测目标平台
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(TARGET_ARCH "aarch64")
    message(STATUS "Target architecture: ARM64/AArch64 (Jetson)")
    add_compile_definitions(JETSON_PLATFORM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a78")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a78")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(TARGET_ARCH "x86_64")
    message(STATUS "Target architecture: x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 查找依赖库
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    set(ENABLE_OPENCV ON)
else()
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENCV QUIET opencv4)
        if(NOT OPENCV_FOUND)
            pkg_check_modules(OPENCV QUIET opencv)
        endif()
        
        if(OPENCV_FOUND)
            message(STATUS "Found OpenCV via pkg-config: ${OPENCV_VERSION}")
            set(ENABLE_OPENCV ON)
            set(OpenCV_FOUND TRUE)
            set(OpenCV_LIBS ${OPENCV_LIBRARIES})
            set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
            include_directories(${OPENCV_INCLUDE_DIRS})
            link_directories(${OPENCV_LIBRARY_DIRS})
        endif()
    endif()
endif()

if(NOT ENABLE_OPENCV)
    message(FATAL_ERROR "OpenCV not found! Please install OpenCV")
endif()

# GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")

# CUDA配置（如果启用）
if(ENABLE_CUDA)
    if(TARGET_ARCH STREQUAL "aarch64")
        message(STATUS "Configuring CUDA for Jetson...")
        
        # Jetson CUDA路径
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
        
        find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
            PATHS
                "/usr/local/cuda/include"
                "/usr/local/cuda-12.6/include"
                "/usr/local/cuda-12/include"
            NO_DEFAULT_PATH)
        
        find_library(CUDA_RUNTIME_LIBRARY cudart
            PATHS
                "/usr/local/cuda/lib64"
                "/usr/local/cuda-12.6/targets/aarch64-linux/lib"
                "/usr/local/cuda/targets/aarch64-linux/lib"
            NO_DEFAULT_PATH)
        
        if(CUDA_RUNTIME_INCLUDE AND CUDA_RUNTIME_LIBRARY)
            message(STATUS "Found CUDA for Jetson: ${CUDA_RUNTIME_LIBRARY}")
            set(CUDA_FOUND TRUE)
            enable_language(CUDA)
            include_directories(${CUDA_RUNTIME_INCLUDE})
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")
            set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})
        else()
            message(WARNING "CUDA not found - GPU功能将被禁用")
            set(ENABLE_CUDA OFF)
        endif()
    else()
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA: ${CUDA_VERSION}")
            enable_language(CUDA)
            include_directories(${CUDA_INCLUDE_DIRS})
        else()
            message(WARNING "CUDA not found - GPU功能将被禁用")
            set(ENABLE_CUDA OFF)
        endif()
    endif()
endif()

# TensorRT配置（如果启用）
if(ENABLE_TENSORRT AND ENABLE_CUDA)
    if(TARGET_ARCH STREQUAL "aarch64")
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS "/usr/include/aarch64-linux-gnu" "/usr/include"
            NO_DEFAULT_PATH)
        
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            PATHS "/usr/lib/aarch64-linux-gnu"
            NO_DEFAULT_PATH)
        
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY)
            set(TENSORRT_FOUND ON)
            message(STATUS "Found TensorRT for Jetson: ${TENSORRT_INFER_LIBRARY}")
            include_directories(${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_INFER_LIBRARY})
        else()
            message(WARNING "TensorRT not found - 使用OpenCV DNN后备")
            set(ENABLE_TENSORRT OFF)
        endif()
    else()
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS /usr/include /usr/local/include)
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            PATHS /usr/lib /usr/local/lib)
        
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY)
            set(TENSORRT_FOUND ON)
            message(STATUS "Found TensorRT: ${TENSORRT_INFER_LIBRARY}")
            include_directories(${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_INFER_LIBRARY})
        else()
            message(WARNING "TensorRT not found - 使用OpenCV DNN后备")
            set(ENABLE_TENSORRT OFF)
        endif()
    endif()
endif()

# libmodbus
find_library(MODBUS_LIBRARY modbus
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
find_path(MODBUS_INCLUDE_DIR modbus/modbus.h
    PATHS /usr/include /usr/local/include)

if(MODBUS_LIBRARY AND MODBUS_INCLUDE_DIR)
    message(STATUS "Found libmodbus: ${MODBUS_LIBRARY}")
    set(ENABLE_MODBUS ON)
    include_directories(${MODBUS_INCLUDE_DIR})
else()
    message(WARNING "libmodbus not found - PLC通信功能将被禁用")
    set(ENABLE_MODBUS OFF)
endif()

# pybind11 (for Python bindings)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "Found pybind11: ${pybind11_VERSION}")
    set(ENABLE_PYBIND11 ON)
else()
    message(WARNING "pybind11 not found - Python绑定将被禁用")
    set(ENABLE_PYBIND11 OFF)
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_backend/include
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

# 源文件收集 - 只包含推理相关的源文件
set(INFERENCE_SOURCES
    ../cpp_backend/src/core/logger.cpp
    ../cpp_backend/src/core/system_utils.cpp
    ../cpp_backend/src/vision/detector.cpp
    ../cpp_backend/src/vision/optimized_detector.cpp
    ../cpp_backend/src/vision/camera_manager.cpp
    ../cpp_backend/src/vision/shared_memory_manager.cpp
    ../cpp_backend/src/communication/modbus_server.cpp
    ../cpp_backend/src/communication/tcp_socket_server.cpp
)

# AI优化源文件
if(ENABLE_AI_OPTIMIZATION)
    list(APPEND INFERENCE_SOURCES
        ../cpp_backend/src/vision/nam_attention.cpp
        ../cpp_backend/src/vision/ghost_conv.cpp
        ../cpp_backend/src/vision/vov_gscsp.cpp
        ../cpp_backend/src/vision/sahi_slicing.cpp
        ../cpp_backend/src/vision/wise_iou.cpp
    )
endif()

# CUDA相关源文件
if(ENABLE_CUDA)
    list(APPEND INFERENCE_SOURCES
        ../cpp_backend/src/vision/hardware_accelerated_camera.cpp
    )
endif()

# TensorRT相关源文件
if(ENABLE_TENSORRT)
    list(APPEND INFERENCE_SOURCES
        ../cpp_backend/src/vision/tensorrt_engine.cpp
    )
endif()

# 深度学习相关源文件
list(APPEND INFERENCE_SOURCES
    ../cpp_backend/src/vision/stereo_vision.cpp
    ../cpp_backend/src/vision/deepstream_pipeline.cpp
)

# 过滤存在的源文件
set(VALID_SOURCES)
foreach(src ${INFERENCE_SOURCES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        list(APPEND VALID_SOURCES ${src})
    else()
        message(WARNING "Source file not found: ${src}")
    endif()
endforeach()

# 创建推理库
if(VALID_SOURCES)
    add_library(bamboo_inference SHARED ${VALID_SOURCES})
    
    # 编译定义
    target_compile_definitions(bamboo_inference PRIVATE
        $<$<BOOL:${ENABLE_CUDA}>:ENABLE_CUDA>
        $<$<BOOL:${ENABLE_TENSORRT}>:ENABLE_TENSORRT>
        $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
        $<$<BOOL:${ENABLE_OPENCV}>:ENABLE_OPENCV>
        $<$<BOOL:${ENABLE_AI_OPTIMIZATION}>:ENABLE_AI_OPTIMIZATION>
        $<$<BOOL:${ENABLE_NAM_ATTENTION}>:ENABLE_NAM_ATTENTION>
        $<$<BOOL:${ENABLE_GHOST_CONV}>:ENABLE_GHOST_CONV>
        $<$<BOOL:${ENABLE_VOV_GSCSP}>:ENABLE_VOV_GSCSP>
        $<$<BOOL:${ENABLE_WISE_IOU}>:ENABLE_WISE_IOU>
        $<$<BOOL:${ENABLE_SAHI_SLICING}>:ENABLE_SAHI_SLICING>
        $<$<BOOL:${ENABLE_HARDWARE_ACCELERATION}>:ENABLE_HARDWARE_ACCELERATION>
        $<$<STREQUAL:${TARGET_ARCH},aarch64>:JETSON_PLATFORM>
    )
    
    # 链接库
    target_link_libraries(bamboo_inference
        ${OpenCV_LIBS}
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${GSTREAMER_VIDEO_LIBRARIES}
        Threads::Threads
        pthread
        dl
        m
    )
    
    # 条件链接库
    if(ENABLE_CUDA)
        target_link_libraries(bamboo_inference ${CUDA_LIBRARIES})
    endif()
    
    if(ENABLE_TENSORRT)
        target_link_libraries(bamboo_inference ${TENSORRT_LIBRARIES})
    endif()
    
    if(ENABLE_MODBUS)
        target_link_libraries(bamboo_inference ${MODBUS_LIBRARY})
    endif()
    
    message(STATUS "Bamboo inference library configured with ${list(LENGTH VALID_SOURCES)} source files")
else()
    message(WARNING "No valid source files found, creating minimal library")
    # 创建一个最小的库避免构建失败
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/minimal.cpp 
        "// Minimal library implementation\nextern \"C\" { int bamboo_inference_version() { return 1; } }")
    add_library(bamboo_inference SHARED ${CMAKE_CURRENT_BINARY_DIR}/minimal.cpp)
endif()

# Python绑定（如果启用pybind11）
if(ENABLE_PYBIND11 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_inference/python_bindings.cpp)
    pybind11_add_module(bamboo_py_inference ../cpp_inference/python_bindings.cpp)
    target_link_libraries(bamboo_py_inference PRIVATE bamboo_inference)
    target_compile_definitions(bamboo_py_inference PRIVATE VERSION_INFO=${PROJECT_VERSION})
    message(STATUS "Python bindings enabled")
endif()

# 测试程序（可选）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_backend/src/main.cpp)
    add_executable(bamboo_inference_test ../cpp_backend/src/main.cpp)
    target_link_libraries(bamboo_inference_test bamboo_inference)
    message(STATUS "Test executable configured")
endif()

# 安装配置
install(TARGETS bamboo_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT Runtime
)

if(TARGET bamboo_inference_test)
    install(TARGETS bamboo_inference_test
        RUNTIME DESTINATION bin
        COMPONENT Runtime
    )
endif()

if(TARGET bamboo_py_inference)
    install(TARGETS bamboo_py_inference
        LIBRARY DESTINATION lib
        COMPONENT Runtime
    )
endif()

# 构建信息显示
message(STATUS "")
message(STATUS "=== C++推理核心构建配置 ===")
message(STATUS "项目名称: ${PROJECT_NAME}")
message(STATUS "版本: ${PROJECT_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "目标架构: ${TARGET_ARCH}")
message(STATUS "")
message(STATUS "AI优化技术:")
message(STATUS "  NAM注意力: ${ENABLE_NAM_ATTENTION}")
message(STATUS "  GhostConv: ${ENABLE_GHOST_CONV}")
message(STATUS "  VoV-GSCSP: ${ENABLE_VOV_GSCSP}")
message(STATUS "  Wise-IoU: ${ENABLE_WISE_IOU}")
message(STATUS "  SAHI切片: ${ENABLE_SAHI_SLICING}")
message(STATUS "  硬件加速: ${ENABLE_HARDWARE_ACCELERATION}")
message(STATUS "")
message(STATUS "依赖库状态:")
message(STATUS "  OpenCV: ${ENABLE_OPENCV}")
message(STATUS "  GStreamer: ${GSTREAMER_FOUND}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  TensorRT: ${ENABLE_TENSORRT}")
message(STATUS "  libmodbus: ${ENABLE_MODBUS}")
message(STATUS "  pybind11: ${ENABLE_PYBIND11}")
message(STATUS "===============================")