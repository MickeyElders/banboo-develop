cmake_minimum_required(VERSION 3.16)
project(BambooCutBackend VERSION 2.0.0 LANGUAGES CXX C)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 启用 FetchContent 用于自动下载依赖
include(FetchContent)

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 检测目标平台
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(TARGET_ARCH "aarch64")
    message(STATUS "Target architecture: ARM64/AArch64 (Jetson)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(TARGET_ARCH "x86_64")
    message(STATUS "Target architecture: x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 包含自定义的CMake模块
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ================================
# 查找依赖库
# ================================

# OpenCV - 多种方式查找以提高兼容性
set(ENABLE_OPENCV OFF)
set(OpenCV_FOUND FALSE)

# 方法1: 尝试使用find_package查找OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV via find_package: ${OpenCV_VERSION}")
    set(ENABLE_OPENCV ON)
    # OpenCV变量已经由find_package设置
    set(OpenCV_LIBS ${OpenCV_LIBS})
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
else()
    # 方法2: 尝试使用pkg-config查找OpenCV
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENCV QUIET opencv4)
        if(NOT OPENCV_FOUND)
            pkg_check_modules(OPENCV QUIET opencv)
        endif()
        
        if(OPENCV_FOUND)
            message(STATUS "Found OpenCV via pkg-config: ${OPENCV_VERSION}")
            set(ENABLE_OPENCV ON)
            set(OpenCV_FOUND TRUE)
            # 设置变量以兼容原代码
            set(OpenCV_LIBS ${OPENCV_LIBRARIES})
            set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
            # 添加包含目录和链接库
            include_directories(${OPENCV_INCLUDE_DIRS})
            link_directories(${OPENCV_LIBRARY_DIRS})
            add_definitions(${OPENCV_CFLAGS_OTHER})
        endif()
    endif()
endif()

# 方法3: 手动查找OpenCV (Jetson特定路径)
if(NOT ENABLE_OPENCV)
    # Jetson常用的OpenCV安装路径
    set(OPENCV_SEARCH_PATHS
        /usr/include/opencv4
        /usr/local/include/opencv4
        /opt/opencv/include/opencv4
        /usr/include/opencv2
        /usr/local/include/opencv2
        /opt/opencv/include/opencv2
        /usr/include
        /usr/local/include
    )
    
    find_path(OpenCV_INCLUDE_DIR opencv2/opencv.hpp
        PATHS ${OPENCV_SEARCH_PATHS}
        NO_DEFAULT_PATH)
    
    if(OpenCV_INCLUDE_DIR)
        # 查找OpenCV库
        find_library(OpenCV_CORE_LIB opencv_core
            PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
            NO_DEFAULT_PATH)
        
        if(OpenCV_CORE_LIB)
            message(STATUS "Found OpenCV manually: ${OpenCV_INCLUDE_DIR}")
            set(ENABLE_OPENCV ON)
            set(OpenCV_FOUND TRUE)
            set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
            
            # 查找常用的OpenCV库
            set(OpenCV_LIBS "")
            foreach(lib IN ITEMS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_video opencv_calib3d opencv_features2d opencv_objdetect opencv_dnn opencv_highgui)
                find_library(${lib}_LIB ${lib}
                    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
                    NO_DEFAULT_PATH)
                if(${lib}_LIB)
                    list(APPEND OpenCV_LIBS ${${lib}_LIB})
                endif()
            endforeach()
            
            include_directories(${OpenCV_INCLUDE_DIRS})
            message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
        endif()
    endif()
endif()

if(NOT ENABLE_OPENCV)
    message(FATAL_ERROR "OpenCV not found! Please install OpenCV:
    Ubuntu/Debian: sudo apt-get install libopencv-dev
    Or build from source: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html")
endif()

# 检查OpenCV扩展模块 (contrib)
set(ENABLE_OPENCV_CONTRIB OFF)
if(ENABLE_OPENCV)
    # 尝试查找OpenCV扩展模块
    find_path(OPENCV_CONTRIB_INCLUDE opencv2/ximgproc.hpp
        PATHS ${OpenCV_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS}
        PATH_SUFFIXES opencv4/opencv2 opencv2
        NO_DEFAULT_PATH)
    
    if(OPENCV_CONTRIB_INCLUDE)
        message(STATUS "Found OpenCV contrib modules")
        set(ENABLE_OPENCV_CONTRIB ON)
    else()
        message(WARNING "OpenCV contrib modules not found - some advanced features will be disabled")
        message(WARNING "To enable full functionality, install: sudo apt-get install libopencv-contrib-dev")
    endif()
endif()

# PkgConfig for GStreamer
find_package(PkgConfig REQUIRED)
# GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")
set(ENABLE_GSTREAMER ON)

# CUDA
find_package(CUDA REQUIRED)
# 额外检查CUDA头文件是否真的存在
find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include
    REQUIRED)

message(STATUS "Found CUDA: ${CUDA_VERSION}")
set(ENABLE_CUDA ON)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT (增强配置，支持 FP16 和完整功能)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

# 查找完整的 TensorRT 库
find_library(TENSORRT_INFER_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY)
    set(TENSORRT_FOUND ON)
    message(STATUS "Found TensorRT: ${TENSORRT_INFER_LIBRARY}")
    set(ENABLE_TENSORRT ON)
    set(ENABLE_FP16 ON)  # 启用 FP16 半精度
    message(STATUS "FP16 precision enabled")
else()
    message(WARNING "TensorRT not found, using fallback inference")
    set(ENABLE_TENSORRT OFF)
    set(ENABLE_FP16 OFF)
endif()

# DeepStream SDK (用于优化的 GStreamer 流水线)
find_path(DEEPSTREAM_INCLUDE_DIR nvds_meta.h
    HINTS ${DEEPSTREAM_ROOT} /opt/nvidia/deepstream/deepstream-6.4
    PATH_SUFFIXES include)

find_library(DEEPSTREAM_LIBRARY nvdsgst_meta
    HINTS ${DEEPSTREAM_ROOT} /opt/nvidia/deepstream/deepstream-6.4
    PATH_SUFFIXES lib lib64)

if(DEEPSTREAM_INCLUDE_DIR AND DEEPSTREAM_LIBRARY)
    set(DEEPSTREAM_FOUND ON)
    message(STATUS "Found DeepStream: ${DEEPSTREAM_LIBRARY}")
    set(ENABLE_DEEPSTREAM ON)
else()
    message(WARNING "DeepStream not found, using standard GStreamer")
    set(ENABLE_DEEPSTREAM OFF)
endif()

# Threads
find_package(Threads REQUIRED)

# JSON库 (nlohmann/json) - 优先使用系统安装版本
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found system nlohmann_json: ${nlohmann_json_VERSION}")
    set(ENABLE_JSON ON)
else()
    message(WARNING "System nlohmann_json not found, trying FetchContent")
    # 回退到FetchContent下载
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3  # 使用稳定版本
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
    
    if(nlohmann_json_POPULATED)
        message(STATUS "nlohmann_json fetched and built successfully")
        set(ENABLE_JSON ON)
    else()
        message(ERROR "Failed to find or fetch nlohmann_json")
        set(ENABLE_JSON OFF)
    endif()
endif()

# ModbusTCP库 - 使用 FetchContent 自动下载和构建
FetchContent_Declare(
    libmodbus
    GIT_REPOSITORY https://github.com/stephane/libmodbus.git
    GIT_TAG v3.1.10  # 使用稳定版本
    GIT_SHALLOW TRUE
)

# 配置 libmodbus 构建选项
set(libmodbus_BUILD_TESTS OFF CACHE BOOL "Build libmodbus tests" FORCE)
set(libmodbus_BUILD_EXAMPLES OFF CACHE BOOL "Build libmodbus examples" FORCE)

# 下载并构建 libmodbus
FetchContent_MakeAvailable(libmodbus)

if(libmodbus_POPULATED)
    message(STATUS "libmodbus fetched and built successfully")
    set(ENABLE_MODBUS ON)
    set(MODBUS_LIBRARY modbus)
    # 设置 libmodbus 的 include 路径
    set(MODBUS_INCLUDE_DIRS ${libmodbus_SOURCE_DIR}/src)
    include_directories(${MODBUS_INCLUDE_DIRS})
else()
    message(WARNING "Failed to fetch libmodbus, trying system installation")
    # 回退到系统安装的版本
    find_library(MODBUS_LIBRARY modbus PATHS /usr/lib /usr/local/lib)
    if(MODBUS_LIBRARY)
        message(STATUS "Found system libmodbus: ${MODBUS_LIBRARY}")
        set(ENABLE_MODBUS ON)
        # 系统安装的 libmodbus include 路径
        include_directories(/usr/include /usr/local/include)
    else()
        message(WARNING "libmodbus not found, will use custom implementation")
        set(ENABLE_MODBUS OFF)
    endif()
endif()

# ================================
# 配置头文件
# ================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bamboo_cut/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/bamboo_cut/config.h
)

# ================================
# 包含目录
# ================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${MODBUS_INCLUDE_DIRS}
)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(GSTREAMER_FOUND)
    include_directories(
        ${GSTREAMER_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    )
endif()

# ================================
# 源文件
# ================================
# 只包含实际存在的源文件
set(SOURCES
    src/core/logger.cpp
    src/vision/stereo_vision.cpp
    src/vision/detector.cpp
    src/vision/camera_manager.cpp
    src/vision/optimized_detector.cpp
    src/communication/modbus_server.cpp
)

# ================================
# 库目标 (供测试使用)
# ================================
add_library(bamboo_cut_lib STATIC
    ${SOURCES}
)

# 设置include路径
target_include_directories(bamboo_cut_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenCV_FOUND)
    target_include_directories(bamboo_cut_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

if(GSTREAMER_FOUND)
    target_include_directories(bamboo_cut_lib PUBLIC
        ${GSTREAMER_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    )
endif()

# 为库目标设置编译定义
target_compile_definitions(bamboo_cut_lib PUBLIC
    $<$<BOOL:${ENABLE_CUDA}>:ENABLE_CUDA>
    $<$<BOOL:${ENABLE_TENSORRT}>:ENABLE_TENSORRT>
    $<$<BOOL:${ENABLE_FP16}>:ENABLE_FP16>
    $<$<BOOL:${ENABLE_DEEPSTREAM}>:ENABLE_DEEPSTREAM>
    $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
    $<$<BOOL:${ENABLE_JSON}>:ENABLE_JSON>
    $<$<BOOL:${ENABLE_OPENCV}>:ENABLE_OPENCV>
    $<$<BOOL:${ENABLE_GSTREAMER}>:ENABLE_GSTREAMER>
    $<$<BOOL:${ENABLE_OPENCV_CONTRIB}>:ENABLE_OPENCV_CONTRIB>
)

# 为库目标链接依赖
target_link_libraries(bamboo_cut_lib
    Threads::Threads
)

if(OpenCV_FOUND)
    target_link_libraries(bamboo_cut_lib ${OpenCV_LIBS})
endif()

if(GSTREAMER_FOUND)
    target_link_libraries(bamboo_cut_lib 
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${GSTREAMER_VIDEO_LIBRARIES}
    )
endif()

if(TENSORRT_FOUND)
    target_link_libraries(bamboo_cut_lib 
        ${TENSORRT_INFER_LIBRARY}
        ${TENSORRT_ONNX_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
    )
endif()

if(DEEPSTREAM_FOUND)
    target_link_libraries(bamboo_cut_lib ${DEEPSTREAM_LIBRARY})
endif()

if(CUDA_FOUND)
    target_link_libraries(bamboo_cut_lib ${CUDA_LIBRARIES})
endif()

# 链接 FetchContent 下载的库
if(libmodbus_POPULATED)
    target_link_libraries(bamboo_cut_lib modbus)
elseif(MODBUS_LIBRARY)
    target_link_libraries(bamboo_cut_lib ${MODBUS_LIBRARY})
endif()

if(nlohmann_json_POPULATED)
    target_link_libraries(bamboo_cut_lib nlohmann_json::nlohmann_json)
elseif(nlohmann_json_FOUND)
    target_link_libraries(bamboo_cut_lib nlohmann_json::nlohmann_json)
endif()

# ================================
# 可执行文件
# ================================
add_executable(bamboo_cut_backend
    src/main.cpp
)

# 链接库到可执行文件 (编译定义会从库自动继承，因为库使用了PUBLIC)
target_link_libraries(bamboo_cut_backend bamboo_cut_lib)

# ================================
# 安装规则
# ================================
install(TARGETS bamboo_cut_backend bamboo_cut_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ================================
# 打包信息
# ================================
set(CPACK_PACKAGE_NAME "bamboo-cut-backend")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION "智能切竹机后端服务")
set(CPACK_PACKAGE_CONTACT "developer@bamboo-cut.com")

if(TARGET_ARCH STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
endif()

include(CPack) 