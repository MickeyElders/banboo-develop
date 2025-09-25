# C++ LVGL一体化竹子识别系统 CMake配置
# 版本: 5.0.0 (C++ LVGL Integrated Architecture)
# C++推理后端 + LVGL界面 + Modbus通信的完整一体化系统

cmake_minimum_required(VERSION 3.16)
project(BambooRecognitionSystem VERSION 5.0.0 LANGUAGES C CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 特性选项
option(ENABLE_AI_OPTIMIZATION "Enable AI optimizations" ON)
option(ENABLE_NAM_ATTENTION "Enable NAM Attention" ON)
option(ENABLE_GHOST_CONV "Enable GhostConv" ON)
option(ENABLE_VOV_GSCSP "Enable VoV-GSCSP" ON)
option(ENABLE_WISE_IOU "Enable Wise-IoU" ON)
option(ENABLE_SAHI_SLICING "Enable SAHI Slicing" ON)
option(ENABLE_HARDWARE_ACCELERATION "Enable Hardware Acceleration" ON)
option(ENABLE_TENSORRT "Enable TensorRT" ON)
option(ENABLE_CUDA "Enable CUDA" ON)
option(ENABLE_MODBUS "Enable Modbus Communication" ON)
option(ENABLE_LVGL "Enable LVGL UI" ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -march=native")

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
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")

# LVGL
if(ENABLE_LVGL)
    find_path(LVGL_INCLUDE_DIR lvgl/lvgl.h
        PATHS
            /usr/include
            /usr/local/include
        PATH_SUFFIXES lvgl)
        
    find_library(LVGL_LIBRARY NAMES lvgl liblvgl
        PATHS
            /usr/lib
            /usr/local/lib
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib/aarch64-linux-gnu
            /usr/lib64
            /usr/local/lib64)
    
    # 尝试查找pkg-config的LVGL
    pkg_check_modules(LVGL_PKG QUIET lvgl)
    if(LVGL_PKG_FOUND)
        set(LVGL_INCLUDE_DIR ${LVGL_PKG_INCLUDE_DIRS})
        set(LVGL_LIBRARY ${LVGL_PKG_LIBRARIES})
        set(LVGL_FOUND TRUE)
        message(STATUS "Found LVGL via pkg-config: ${LVGL_PKG_VERSION}")
    elseif(LVGL_INCLUDE_DIR AND LVGL_LIBRARY)
        set(LVGL_FOUND TRUE)
        message(STATUS "Found LVGL: ${LVGL_LIBRARY}")
        message(STATUS "LVGL include dir: ${LVGL_INCLUDE_DIR}")
    else()
        # 检查是否LVGL头文件在标准位置但没有单独的包含目录
        find_path(LVGL_INCLUDE_DIR_ALT lvgl.h
            PATHS
                /usr/include/lvgl
                /usr/local/include/lvgl)
        if(LVGL_INCLUDE_DIR_ALT AND LVGL_LIBRARY)
            set(LVGL_INCLUDE_DIR ${LVGL_INCLUDE_DIR_ALT}/..)
            set(LVGL_FOUND TRUE)
            message(STATUS "Found LVGL (alternative path): ${LVGL_LIBRARY}")
        else()
            message(WARNING "LVGL not found! LVGL UI功能将被禁用")
            set(ENABLE_LVGL OFF)
            set(LVGL_FOUND FALSE)
        endif()
    endif()
    
    if(LVGL_FOUND)
        include_directories(${LVGL_INCLUDE_DIR})
    endif()
endif()

# CUDA配置
if(ENABLE_CUDA)
    if(TARGET_ARCH STREQUAL "aarch64")
        find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
            PATHS "/usr/local/cuda/include" "/usr/local/cuda-12.6/include"
            NO_DEFAULT_PATH)
        find_library(CUDA_RUNTIME_LIBRARY cudart
            PATHS "/usr/local/cuda/lib64" "/usr/local/cuda/targets/aarch64-linux/lib"
            NO_DEFAULT_PATH)
            
        if(CUDA_RUNTIME_INCLUDE AND CUDA_RUNTIME_LIBRARY)
            set(CUDA_FOUND TRUE)
            message(STATUS "Found CUDA for Jetson: ${CUDA_RUNTIME_LIBRARY}")
            include_directories(${CUDA_RUNTIME_INCLUDE})
            set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})
        else()
            message(WARNING "CUDA not found - GPU功能将被禁用")
            set(ENABLE_CUDA OFF)
        endif()
    else()
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA: ${CUDA_VERSION}")
            include_directories(${CUDA_INCLUDE_DIRS})
        else()
            message(WARNING "CUDA not found - GPU功能将被禁用")
            set(ENABLE_CUDA OFF)
        endif()
    endif()
endif()

# TensorRT配置
if(ENABLE_TENSORRT AND ENABLE_CUDA)
    if(TARGET_ARCH STREQUAL "aarch64")
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS "/usr/include/aarch64-linux-gnu" "/usr/include")
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            PATHS "/usr/lib/aarch64-linux-gnu")
    else()
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS /usr/include /usr/local/include)
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            PATHS /usr/lib /usr/local/lib)
    endif()
    
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

# libmodbus
if(ENABLE_MODBUS)
    find_library(MODBUS_LIBRARY modbus
        PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
    find_path(MODBUS_INCLUDE_DIR modbus/modbus.h
        PATHS /usr/include /usr/local/include)
    
    if(MODBUS_LIBRARY AND MODBUS_INCLUDE_DIR)
        message(STATUS "Found libmodbus: ${MODBUS_LIBRARY}")
        include_directories(${MODBUS_INCLUDE_DIR})
    else()
        message(WARNING "libmodbus not found - PLC通信功能将被禁用")
        set(ENABLE_MODBUS OFF)
    endif()
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

# 核心源文件
set(CORE_SOURCES
    src/core/data_bridge.cpp
    src/core/bamboo_system.cpp
)

# AI推理源文件
set(INFERENCE_SOURCES
    src/inference/bamboo_detector.cpp
)

# LVGL界面源文件
if(ENABLE_LVGL AND LVGL_FOUND)
    set(UI_SOURCES
        src/ui/lvgl_interface.cpp
        src/ui/lvgl_display_driver.cpp
        src/ui/lvgl_input_driver.cpp
    )
else()
    set(UI_SOURCES)
endif()

# 通信源文件
if(ENABLE_MODBUS)
    set(COMMUNICATION_SOURCES
        src/communication/modbus_interface.cpp
    )
endif()

# 工具源文件
set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/config_loader.cpp
    src/utils/system_monitor.cpp
)

# 合并所有源文件
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${INFERENCE_SOURCES}
    ${UI_SOURCES}
    ${COMMUNICATION_SOURCES}
    ${UTILS_SOURCES}
)

# 过滤存在的源文件
set(VALID_SOURCES)
foreach(src ${ALL_SOURCES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        list(APPEND VALID_SOURCES ${src})
    else()
        message(WARNING "Source file not found: ${src}")
    endif()
endforeach()

# 创建主库
add_library(bamboo_system SHARED ${VALID_SOURCES})

# 编译定义
target_compile_definitions(bamboo_system PRIVATE
    $<$<BOOL:${ENABLE_CUDA}>:ENABLE_CUDA>
    $<$<BOOL:${ENABLE_TENSORRT}>:ENABLE_TENSORRT>
    $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
    $<$<BOOL:${ENABLE_LVGL}>:ENABLE_LVGL>
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
target_link_libraries(bamboo_system
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
    target_link_libraries(bamboo_system ${CUDA_LIBRARIES})
endif()

if(ENABLE_TENSORRT)
    target_link_libraries(bamboo_system ${TENSORRT_LIBRARIES})
endif()

if(ENABLE_MODBUS)
    target_link_libraries(bamboo_system ${MODBUS_LIBRARY})
endif()

if(ENABLE_LVGL AND LVGL_FOUND)
    target_link_libraries(bamboo_system ${LVGL_LIBRARY})
endif()

# 主程序可执行文件
add_executable(bamboo_recognition src/main.cpp)
target_link_libraries(bamboo_recognition bamboo_system)

# 设置可执行文件输出名称
set_target_properties(bamboo_recognition PROPERTIES OUTPUT_NAME "bamboo-recognition")

# 安装配置
install(TARGETS bamboo_system
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT Runtime
)

install(TARGETS bamboo_recognition
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    COMPONENT Development
)

# 安装配置文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../config/
    DESTINATION etc/bamboo-recognition
    FILES_MATCHING PATTERN "*.yaml" PATTERN "*.json"
    COMPONENT Runtime
)

# 创建systemd服务文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../deploy/systemd/bamboo-cpp-lvgl.service.in
    ${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service
    DESTINATION lib/systemd/system
    COMPONENT Runtime
)

# CPack配置
set(CPACK_PACKAGE_NAME "bamboo-recognition-system")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AI Bamboo Recognition System - C++ LVGL Edition")
set(CPACK_PACKAGE_VENDOR "Bamboo Industries")
set(CPACK_PACKAGE_CONTACT "admin@bamboo-industries.com")

if(TARGET_ARCH STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
endif()

set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev, libgstreamer1.0-dev, libmodbus-dev, liblvgl-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "opencv-devel, gstreamer1-devel, libmodbus-devel, lvgl-devel")

include(CPack)

# 构建信息显示
message(STATUS "")
message(STATUS "=== C++ LVGL一体化竹子识别系统构建配置 ===")
message(STATUS "项目名称: ${PROJECT_NAME}")
message(STATUS "版本: ${PROJECT_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "目标架构: ${TARGET_ARCH}")
message(STATUS "")
message(STATUS "功能模块:")
message(STATUS "  AI推理: ${ENABLE_AI_OPTIMIZATION}")
message(STATUS "  LVGL界面: ${ENABLE_LVGL}")
message(STATUS "  Modbus通信: ${ENABLE_MODBUS}")
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
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  GStreamer: ${GSTREAMER_VERSION}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  TensorRT: ${ENABLE_TENSORRT}")
message(STATUS "  libmodbus: ${ENABLE_MODBUS}")
message(STATUS "  LVGL: ${ENABLE_LVGL}")
message(STATUS "===============================================")