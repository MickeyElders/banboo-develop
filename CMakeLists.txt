# 一体化竹子识别系统 CMake配置
# 版本: 3.0.0 (Integrated Architecture)
# 整合cpp_backend和lvgl_frontend为单一高性能进程

cmake_minimum_required(VERSION 3.16)
project(BambooIntegratedSystem VERSION 3.0.0 LANGUAGES C CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 启用 FetchContent 用于自动下载依赖
include(FetchContent)

# 编译选项 - 整合两个项目的优化选项
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

# ================================
# 查找依赖库
# ================================
option(ENABLE_WAYLAND "Enable Wayland client/compositor support" OFF)

# 查找PkgConfig
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenCV - 多种方式查找以提高兼容性
set(ENABLE_OPENCV OFF)
set(OpenCV_FOUND FALSE)

# 优先使用find_package查找OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV via find_package: ${OpenCV_VERSION}")
    set(ENABLE_OPENCV ON)
else()
    # 回退到pkg-config查找
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
    # 手动查找OpenCV (Jetson特定路径)
    set(OPENCV_SEARCH_PATHS
        /usr/include/opencv4
        /usr/local/include/opencv4
        /opt/opencv/include/opencv4
        /usr/include/opencv2
        /usr/local/include/opencv2
        /usr/include
        /usr/local/include
    )
    
    find_path(OpenCV_INCLUDE_DIR opencv2/opencv.hpp PATHS ${OPENCV_SEARCH_PATHS})
    if(OpenCV_INCLUDE_DIR)
        find_library(OpenCV_CORE_LIB opencv_core
            PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
        
        if(OpenCV_CORE_LIB)
            message(STATUS "Found OpenCV manually: ${OpenCV_INCLUDE_DIR}")
            set(ENABLE_OPENCV ON)
            set(OpenCV_FOUND TRUE)
            set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
            
            # 查找常用的OpenCV库
            set(OpenCV_LIBS "")
            foreach(lib IN ITEMS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_video opencv_calib3d opencv_features2d opencv_objdetect opencv_dnn opencv_highgui)
                find_library(${lib}_LIB ${lib}
                    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
                if(${lib}_LIB)
                    list(APPEND OpenCV_LIBS ${${lib}_LIB})
                endif()
            endforeach()
            
            include_directories(${OpenCV_INCLUDE_DIRS})
        endif()
    endif()
endif()

if(NOT ENABLE_OPENCV)
    message(FATAL_ERROR "OpenCV not found! Please install OpenCV")
endif()

# GStreamer - 整合两个项目的GStreamer配置
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")
set(ENABLE_GSTREAMER ON)

# CUDA - Jetson Orin NX特殊配置
if(TARGET_ARCH STREQUAL "aarch64")
    message(STATUS "Configuring CUDA for Jetson Orin NX...")
    
    # Jetson Orin NX的CUDA路径配置
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    
    # 查找CUDA头文件 - 用户环境有cuda, cuda-12, cuda-12.6三个版本
    find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
        PATHS
            "/usr/local/cuda/include"
            "/usr/local/cuda-12.6/include"
            "/usr/local/cuda-12/include"
        NO_DEFAULT_PATH)
    
    # 查找CUDA库
    find_library(CUDA_RUNTIME_LIBRARY cudart
        PATHS
            "/usr/local/cuda/lib64"
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib"
            "/usr/local/cuda/targets/aarch64-linux/lib"
            "/usr/lib/aarch64-linux-gnu"
        NO_DEFAULT_PATH)
    
    if(CUDA_RUNTIME_INCLUDE AND CUDA_RUNTIME_LIBRARY)
        message(STATUS "Found CUDA for Jetson: ${CUDA_RUNTIME_LIBRARY}")
        set(ENABLE_CUDA ON)
        set(CUDA_FOUND TRUE)
        
        # 启用CUDA语言支持
        enable_language(CUDA)
        
        # 设置CUDA包含目录
        include_directories(${CUDA_RUNTIME_INCLUDE})
        
        # 添加CUDA库目录到链接器搜索路径
        link_directories(${CUDA_LIB_PATHS})
        
        # 设置CUDA编译器标志
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")  # Orin NX GPU架构
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        
        # 查找其他CUDA库 - 适配用户的多CUDA版本环境
        set(CUDA_LIB_PATHS
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib"
            "/usr/local/cuda-12/targets/aarch64-linux/lib"
            "/usr/local/cuda/targets/aarch64-linux/lib"
            "/usr/local/cuda-12.6/lib64"
            "/usr/local/cuda-12/lib64"
            "/usr/local/cuda/lib64"
            "/usr/lib/aarch64-linux-gnu"
        )
        
        find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
        find_library(CURAND_LIBRARY curand PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
        find_library(CUFFT_LIBRARY cufft PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
        
        set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})
        if(CUBLAS_LIBRARY)
            list(APPEND CUDA_LIBRARIES ${CUBLAS_LIBRARY})
        endif()
        if(CURAND_LIBRARY)
            list(APPEND CUDA_LIBRARIES ${CURAND_LIBRARY})
        endif()
        if(CUFFT_LIBRARY)
            list(APPEND CUDA_LIBRARIES ${CUFFT_LIBRARY})
        endif()
        
        message(STATUS "Jetson CUDA libraries: ${CUDA_LIBRARIES}")
        
    else()
        message(FATAL_ERROR "CUDA not found on Jetson Orin NX. Please ensure CUDA is properly installed.")
    endif()
    
else()
    # x86_64平台的标准CUDA查找
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include /usr/local/cuda/include)
        
        if(CUDA_RUNTIME_INCLUDE)
            message(STATUS "Found CUDA: ${CUDA_VERSION}")
            set(ENABLE_CUDA ON)
            enable_language(CUDA)
            include_directories(${CUDA_INCLUDE_DIRS})
        else()
            message(WARNING "CUDA headers not found - GPU功能将被禁用")
            set(ENABLE_CUDA OFF)
        endif()
    else()
        message(WARNING "CUDA not found - GPU功能将被禁用")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# TensorRT - Jetson Orin NX特殊配置
if(ENABLE_CUDA)
    if(TARGET_ARCH STREQUAL "aarch64")
        message(STATUS "Configuring TensorRT for Jetson Orin NX...")
        
        # Jetson Orin NX的TensorRT路径
        set(TENSORRT_SEARCH_PATHS
            "/usr/include/aarch64-linux-gnu"
            "/usr/include"
            "/usr/local/include"
            "/opt/nvidia/vpi2/include"
        )
        
        set(TENSORRT_LIB_PATHS
            "/usr/lib/aarch64-linux-gnu"
            "/usr/local/lib"
            "/opt/nvidia/vpi2/lib64"
        )
        
        # 查找TensorRT头文件
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS ${TENSORRT_SEARCH_PATHS}
            NO_DEFAULT_PATH)
        
        # 查找TensorRT库
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            PATHS ${TENSORRT_LIB_PATHS}
            NO_DEFAULT_PATH)
        
        find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
            PATHS ${TENSORRT_LIB_PATHS}
            NO_DEFAULT_PATH)
        
        find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
            PATHS ${TENSORRT_LIB_PATHS}
            NO_DEFAULT_PATH)
        
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY)
            set(TENSORRT_FOUND ON)
            set(ENABLE_TENSORRT ON)
            set(ENABLE_FP16 ON)  # Jetson支持FP16
            
            message(STATUS "Found TensorRT for Jetson:")
            message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
            message(STATUS "  nvinfer: ${TENSORRT_INFER_LIBRARY}")
            
            set(TENSORRT_LIBRARIES ${TENSORRT_INFER_LIBRARY})
            if(TENSORRT_ONNX_LIBRARY)
                list(APPEND TENSORRT_LIBRARIES ${TENSORRT_ONNX_LIBRARY})
                message(STATUS "  nvonnxparser: ${TENSORRT_ONNX_LIBRARY}")
            endif()
            if(TENSORRT_PLUGIN_LIBRARY)
                list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PLUGIN_LIBRARY})
                message(STATUS "  nvinfer_plugin: ${TENSORRT_PLUGIN_LIBRARY}")
            endif()
            
            include_directories(${TENSORRT_INCLUDE_DIR})
            
        else()
            message(FATAL_ERROR "TensorRT not found on Jetson Orin NX. Please ensure TensorRT is properly installed.")
        endif()
        
    else()
        # x86_64平台的标准TensorRT查找
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include
            PATHS /usr/include/x86_64-linux-gnu /usr/include)
        
        find_library(TENSORRT_INFER_LIBRARY nvinfer
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64
            PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
        
        find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64
            PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
        
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY)
            set(TENSORRT_FOUND ON)
            set(ENABLE_TENSORRT ON)
            message(STATUS "Found TensorRT: ${TENSORRT_INFER_LIBRARY}")
            set(TENSORRT_LIBRARIES ${TENSORRT_INFER_LIBRARY})
            if(TENSORRT_ONNX_LIBRARY)
                list(APPEND TENSORRT_LIBRARIES ${TENSORRT_ONNX_LIBRARY})
            endif()
            include_directories(${TENSORRT_INCLUDE_DIR})
        else()
            message(WARNING "TensorRT not found - 使用OpenCV DNN后备")
            set(ENABLE_TENSORRT OFF)
        endif()
    endif()
else()
    set(ENABLE_TENSORRT OFF)
endif()

# libmodbus - 可选依赖
find_library(MODBUS_LIBRARY modbus
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
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

# nlohmann/json - 可选依赖
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
    set(ENABLE_JSON ON)
else()
    message(WARNING "nlohmann_json not found - JSON功能将被禁用")
    set(ENABLE_JSON OFF)
endif()

# Wayland - 标准Wayland Compositor架构 (可选，默认关闭以支持无合成器)
if(ENABLE_WAYLAND)
    pkg_check_modules(WAYLAND_CLIENT QUIET wayland-client)
    pkg_check_modules(WAYLAND_EGL QUIET wayland-egl)
    pkg_check_modules(WAYLAND_PROTOCOLS QUIET wayland-protocols)

    if(WAYLAND_CLIENT_FOUND AND WAYLAND_EGL_FOUND)
        message(STATUS "Found Wayland client: ${WAYLAND_CLIENT_VERSION}")
        message(STATUS "Found Wayland EGL: ${WAYLAND_EGL_VERSION}")
        include_directories(${WAYLAND_CLIENT_INCLUDE_DIRS} ${WAYLAND_EGL_INCLUDE_DIRS})
        link_directories(${WAYLAND_CLIENT_LIBRARY_DIRS} ${WAYLAND_EGL_LIBRARY_DIRS})
        
        if(WAYLAND_PROTOCOLS_FOUND)
            message(STATUS "Found Wayland protocols: ${WAYLAND_PROTOCOLS_VERSION}")
        endif()
    else()
        message(WARNING "Wayland client libraries not found - Wayland support disabled")
        set(ENABLE_WAYLAND OFF)
    endif()
endif()

# 查找EGL库 (Wayland需要EGL支持)
find_library(EGL_LIBRARY EGL
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_path(EGL_INCLUDE_DIR EGL/egl.h
    PATHS /usr/include /usr/local/include)

if(EGL_LIBRARY AND EGL_INCLUDE_DIR)
    message(STATUS "Found EGL: ${EGL_LIBRARY}")
    set(ENABLE_EGL ON)
    set(EGL_LIBRARIES ${EGL_LIBRARY})
    include_directories(${EGL_INCLUDE_DIR})
else()
    message(FATAL_ERROR "EGL not found - Wayland需要EGL支持")
endif()

# 查找GBM库 (DRM EGL后端需要)
find_library(GBM_LIBRARY gbm
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_path(GBM_INCLUDE_DIR gbm.h
    PATHS /usr/include /usr/local/include)

if(GBM_LIBRARY AND GBM_INCLUDE_DIR)
    message(STATUS "Found GBM: ${GBM_LIBRARY}")
    set(ENABLE_GBM ON)
    set(GBM_LIBRARIES ${GBM_LIBRARY})
    include_directories(${GBM_INCLUDE_DIR})
    list(APPEND EGL_LIBRARIES ${GBM_LIBRARY})
else()
    message(FATAL_ERROR "GBM not found - DRM EGL后端需要GBM支持")
endif()

# 查找OpenGL ES库 (DRM EGL渲染需要)
find_library(GLES2_LIBRARY GLESv2
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_path(GLES2_INCLUDE_DIR GLES2/gl2.h
    PATHS /usr/include /usr/local/include)

if(GLES2_LIBRARY AND GLES2_INCLUDE_DIR)
    message(STATUS "Found OpenGL ES: ${GLES2_LIBRARY}")
    set(ENABLE_GLES2 ON)
    set(GLES2_LIBRARIES ${GLES2_LIBRARY})
    include_directories(${GLES2_INCLUDE_DIR})
    list(APPEND EGL_LIBRARIES ${GLES2_LIBRARY})
else()
    message(FATAL_ERROR "OpenGL ES not found - DRM EGL渲染需要OpenGL ES支持")
endif()

# 查找DRM库 (DRM直接访问需要)
find_library(DRM_LIBRARY drm
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_path(DRM_INCLUDE_DIR libdrm/drm.h
    PATHS /usr/include /usr/local/include)

if(DRM_LIBRARY AND DRM_INCLUDE_DIR)
    message(STATUS "Found DRM: ${DRM_LIBRARY}")
    message(STATUS "DRM include dir: ${DRM_INCLUDE_DIR}")
    set(ENABLE_DRM_LIB ON)
    set(DRM_LIBRARIES ${DRM_LIBRARY})
    include_directories(${DRM_INCLUDE_DIR})
    include_directories(${DRM_INCLUDE_DIR}/libdrm)
    list(APPEND EGL_LIBRARIES ${DRM_LIBRARY})
else()
    message(FATAL_ERROR "DRM library not found - DRM EGL后端需要libdrm支持")
endif()

# xkbcommon - Wayland键盘输入支持
pkg_check_modules(XKBCOMMON QUIET xkbcommon)
if(XKBCOMMON_FOUND)
    message(STATUS "Found xkbcommon: ${XKBCOMMON_VERSION}")
    set(ENABLE_XKBCOMMON ON)
    include_directories(${XKBCOMMON_INCLUDE_DIRS})
    link_directories(${XKBCOMMON_LIBRARY_DIRS})
else()
    message(WARNING "xkbcommon not found - 键盘输入可能受限")
    set(ENABLE_XKBCOMMON OFF)
endif()

# 启用DRM EGL共享架构（替代DRM直接访问）
set(ENABLE_DRM OFF)  # 禁用旧的DRM直接访问
set(ENABLE_NVIDIA_DRM OFF)
set(ENABLE_TEGRA_DRM OFF)
# ENABLE_GBM, ENABLE_GLES2, ENABLE_DRM_LIB 在上面已设置
message(STATUS "DRM直接访问已禁用 - 使用DRM EGL共享架构")

# 移除Python绑定相关配置 - 项目现在是纯C++架构
set(ENABLE_PYBIND11 OFF)
set(ENABLE_PYTHON OFF)
message(STATUS "Python绑定已禁用 - 使用纯C++一体化架构")

# ================================
# LVGL配置 - 使用系统安装的LVGL
# ================================

# 查找系统安装的LVGL - 优先使用pkg-config
pkg_check_modules(LVGL QUIET lvgl)
if(LVGL_FOUND)
    message(STATUS "Found LVGL via pkg-config: ${LVGL_VERSION}")
    set(ENABLE_LVGL ON)
    include_directories(${LVGL_INCLUDE_DIRS})
    link_directories(${LVGL_LIBRARY_DIRS})
    
    # 创建导入的目标 - 修复路径验证
    if(EXISTS "${LVGL_INCLUDE_DIRS}")
        add_library(lvgl SHARED IMPORTED)
        set_target_properties(lvgl PROPERTIES
            IMPORTED_LOCATION "${LVGL_LIBRARY_DIRS}/liblvgl.so"
            INTERFACE_INCLUDE_DIRECTORIES "${LVGL_INCLUDE_DIRS}"
        )
    else()
        message(WARNING "LVGL include directory does not exist: ${LVGL_INCLUDE_DIRS}")
        # 尝试简单的包含目录设置
        include_directories(${LVGL_INCLUDE_DIRS})
    endif()
else()
    # 回退到find_package
    find_package(lvgl QUIET)
    if(lvgl_FOUND)
        message(STATUS "Found system LVGL package: ${lvgl_VERSION}")
        set(ENABLE_LVGL ON)
    else()
        # 手动查找系统安装的LVGL
        find_path(LVGL_INCLUDE_DIR lvgl/lvgl.h
            PATHS
                /usr/local/include
                /usr/include
                /opt/lvgl/include
            NO_DEFAULT_PATH)
        
        find_library(LVGL_LIBRARY lvgl
            PATHS
                /usr/local/lib
                /usr/lib
                /usr/lib/aarch64-linux-gnu
                /usr/lib/x86_64-linux-gnu
                /opt/lvgl/lib
            NO_DEFAULT_PATH)
        
        if(LVGL_INCLUDE_DIR AND LVGL_LIBRARY)
            message(STATUS "Found system LVGL manually:")
            message(STATUS "  Include: ${LVGL_INCLUDE_DIR}")
            message(STATUS "  Library: ${LVGL_LIBRARY}")
            set(ENABLE_LVGL ON)
            set(LVGL_FOUND TRUE)
            
            # 创建导入的目标
            add_library(lvgl SHARED IMPORTED)
            set_target_properties(lvgl PROPERTIES
                IMPORTED_LOCATION ${LVGL_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${LVGL_INCLUDE_DIR}
            )
            
            include_directories(${LVGL_INCLUDE_DIR})
        else()
            message(WARNING "System LVGL not found - LVGL功能将被禁用")
            set(ENABLE_LVGL OFF)
            set(LVGL_FOUND FALSE)
        endif()
    endif()
endif()

# 设置LVGL配置文件路径
if(ENABLE_LVGL)
    set(LV_CONF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lv_conf.h CACHE STRING "Path to lv_conf.h")
    message(STATUS "Using lv_conf.h from: ${LV_CONF_PATH}")
endif()

# ================================
# 包含目录
# ================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_backend/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lvgl_frontend/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_inference
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/lvgl
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

if(ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()


# ================================
# 源文件收集
# ================================

# C++后端源文件 - Wayland迁移后的精确配置
# 移除所有DRM相关源文件，添加Wayland接口
set(CPP_BACKEND_SOURCES
    cpp_backend/src/utils/logger.cpp
    cpp_backend/src/core/system_utils.cpp
    cpp_backend/src/core/bamboo_system.cpp
    cpp_backend/src/core/data_bridge.cpp
    cpp_backend/src/vision/stereo_vision.cpp
    cpp_backend/src/vision/detector.cpp
    cpp_backend/src/vision/camera_manager.cpp
    cpp_backend/src/vision/shared_memory_manager.cpp
    cpp_backend/src/vision/optimized_detector.cpp
    cpp_backend/src/vision/nam_attention.cpp
    cpp_backend/src/vision/ghost_conv.cpp
    cpp_backend/src/vision/vov_gscsp.cpp
    cpp_backend/src/vision/sahi_slicing.cpp
    cpp_backend/src/vision/wise_iou.cpp
    cpp_backend/src/vision/tensorrt_engine.cpp
    cpp_backend/src/vision/deepstream_pipeline.cpp
    cpp_backend/src/communication/modbus_server.cpp
    cpp_backend/src/communication/tcp_socket_server.cpp
    cpp_backend/src/communication/modbus_interface.cpp
    cpp_backend/src/inference/bamboo_detector.cpp
    cpp_backend/src/inference/ghost_conv.cpp
    cpp_backend/src/deepstream/deepstream_manager.cpp
    cpp_backend/src/ui/lvgl_ui_utils.cpp
    cpp_backend/src/ui/ui_components.cpp
    cpp_backend/src/ui/ui_updaters.cpp
    # 🔧 注意：以下 panel 文件仅用于 DRM 版本（LVGLInterface）
    # Wayland 版本（LVGLWaylandInterface）使用内联面板创建
    # cpp_backend/src/ui/lvgl_panels/header_panel.cpp
    # cpp_backend/src/ui/lvgl_panels/camera_panel.cpp
    # cpp_backend/src/ui/lvgl_panels/control_panel.cpp
    # cpp_backend/src/ui/lvgl_panels/status_panel.cpp
    # cpp_backend/src/ui/lvgl_panels/footer_panel.cpp
    cpp_backend/src/utils/config_loader.cpp
    cpp_backend/src/utils/system_monitor.cpp
    cpp_backend/src/utils/jetson_monitor.cpp
    cpp_backend/src/ui/egl_context_manager.cpp
)

# 选择接口实现：Wayland 或 DRM
if(ENABLE_WAYLAND)
    list(APPEND CPP_BACKEND_SOURCES cpp_backend/src/ui/lvgl_wayland_interface.cpp)
else()
    list(APPEND CPP_BACKEND_SOURCES cpp_backend/src/ui/lvgl_drm_interface.cpp)
endif()


# 🔧 修复：在构建时从系统 wayland-protocols 生成 xdg-shell 协议文件
if(ENABLE_WAYLAND)
    # 查找 wayland-scanner
    find_program(WAYLAND_SCANNER wayland-scanner REQUIRED)

    # 查找系统的 xdg-shell.xml
    execute_process(
        COMMAND pkg-config --variable=pkgdatadir wayland-protocols
        OUTPUT_VARIABLE WAYLAND_PROTOCOLS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(XDG_SHELL_XML "${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml")

    if(NOT EXISTS "${XDG_SHELL_XML}")
        message(FATAL_ERROR "找不到 xdg-shell.xml: ${XDG_SHELL_XML}")
    endif()

    message(STATUS "Found xdg-shell.xml: ${XDG_SHELL_XML}")

    # 生成协议文件的输出目录
    set(WAYLAND_PROTOCOLS_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/wayland-protocols")
    file(MAKE_DIRECTORY ${WAYLAND_PROTOCOLS_GEN_DIR})

    # 生成 xdg-shell 客户端头文件
    set(XDG_SHELL_CLIENT_HEADER "${WAYLAND_PROTOCOLS_GEN_DIR}/xdg-shell-client-protocol.h")
    add_custom_command(
        OUTPUT ${XDG_SHELL_CLIENT_HEADER}
        COMMAND ${WAYLAND_SCANNER} client-header ${XDG_SHELL_XML} ${XDG_SHELL_CLIENT_HEADER}
        DEPENDS ${XDG_SHELL_XML}
        COMMENT "Generating xdg-shell client header"
        VERBATIM
    )

    # 生成 xdg-shell 协议实现代码
    set(XDG_SHELL_PROTOCOL_CODE "${WAYLAND_PROTOCOLS_GEN_DIR}/xdg-shell-protocol.c")
    add_custom_command(
        OUTPUT ${XDG_SHELL_PROTOCOL_CODE}
        COMMAND ${WAYLAND_SCANNER} private-code ${XDG_SHELL_XML} ${XDG_SHELL_PROTOCOL_CODE}
        DEPENDS ${XDG_SHELL_XML}
        COMMENT "Generating xdg-shell protocol code"
        VERBATIM
    )

    # 创建自定义目标确保协议文件在编译前生成
    add_custom_target(generate_wayland_protocols
        DEPENDS ${XDG_SHELL_CLIENT_HEADER} ${XDG_SHELL_PROTOCOL_CODE}
    )

    # 将生成的协议实现添加到源文件列表
    list(APPEND CPP_BACKEND_SOURCES ${XDG_SHELL_PROTOCOL_CODE})

    # 添加生成文件的包含目录
    include_directories(${WAYLAND_PROTOCOLS_GEN_DIR})

    message(STATUS "Wayland protocols will be generated at build time")

    # 添加lv_drivers Wayland源文件 (如果存在)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lv_drivers")
        set(LV_DRIVERS_WAYLAND_SOURCES
            third_party/lv_drivers/wayland/wayland.c
            third_party/lv_drivers/indev/wayland_pointer.c
            third_party/lv_drivers/indev/wayland_pointeraxis.c
            third_party/lv_drivers/indev/wayland_keyboard.c
            third_party/lv_drivers/indev/wayland_touch.c
        )
        
        # 过滤存在的lv_drivers源文件
        set(VALID_LV_DRIVERS_SOURCES)
        foreach(src ${LV_DRIVERS_WAYLAND_SOURCES})
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
                list(APPEND VALID_LV_DRIVERS_SOURCES ${src})
                message(STATUS "Found lv_drivers source: ${src}")
            endif()
        endforeach()
        
        if(VALID_LV_DRIVERS_SOURCES)
            list(APPEND CPP_BACKEND_SOURCES ${VALID_LV_DRIVERS_SOURCES})
            include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/lv_drivers)
            message(STATUS "Added lv_drivers Wayland sources: ${VALID_LV_DRIVERS_SOURCES}")
        endif()
    endif()
endif()

# 条件包含CUDA相关文件
if(ENABLE_CUDA)
    list(APPEND CPP_BACKEND_SOURCES
        cpp_backend/src/vision/hardware_accelerated_camera.cpp
    )
endif()

# 排除不需要的源文件 - 避免编译错误
list(REMOVE_ITEM CPP_BACKEND_SOURCES
    "cpp_backend/src/ffi/flutter_interface.cpp"  # 排除Flutter接口
)

# LVGL前端源文件 - 基于原有CMakeLists.txt的精确配置
set(LVGL_FRONTEND_SOURCES
    # 应用层
    lvgl_frontend/src/app/main_app.cpp
    lvgl_frontend/src/app/event_manager.cpp
    lvgl_frontend/src/app/config_manager.cpp
    
    # 系统层
    lvgl_frontend/src/system/lv_port_tick.c
    
    # 显示系统
    lvgl_frontend/src/display/framebuffer_driver.cpp
    lvgl_frontend/src/display/lvgl_display.cpp
    lvgl_frontend/src/display/gpu_accelerated.cpp
    
    # 输入系统
    lvgl_frontend/src/input/touch_driver.cpp
    lvgl_frontend/src/input/input_calibration.cpp
    
    # 摄像头系统
    lvgl_frontend/src/camera/v4l2_camera.cpp
    lvgl_frontend/src/camera/camera_manager.cpp
    lvgl_frontend/src/camera/shared_memory_reader.cpp
    lvgl_frontend/src/camera/gstreamer_receiver.cpp
    
    # 后端通信系统
    lvgl_frontend/src/backend/backend_client.cpp
    lvgl_frontend/src/backend/tcp_socket_client.cpp
    
    # AI推理系统
    lvgl_frontend/src/ai/tensorrt_engine.cpp
    lvgl_frontend/src/ai/yolo_detector.cpp
    lvgl_frontend/src/ai/detection_processor.cpp
    
    # GUI组件
    lvgl_frontend/src/gui/video_view.cpp
    lvgl_frontend/src/gui/control_panel.cpp
    lvgl_frontend/src/gui/status_bar.cpp
    lvgl_frontend/src/gui/settings_page.cpp
)

# 条件包含CUDA相关文件
if(ENABLE_CUDA)
    list(APPEND LVGL_FRONTEND_SOURCES
        lvgl_frontend/src/camera/cuda_processor.cpp
    )
endif()

# C++推理核心源文件 - 来自cpp_inference项目
set(CPP_INFERENCE_SOURCES)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp_inference/inference_core.cpp)
    set(CPP_INFERENCE_SOURCES
        cpp_inference/inference_core.cpp
        cpp_inference/utils.cpp
    )
    message(STATUS "Found cpp_inference components:")
    message(STATUS "  inference_core.cpp: ${CMAKE_CURRENT_SOURCE_DIR}/cpp_inference/inference_core.cpp")
    message(STATUS "  utils.cpp: ${CMAKE_CURRENT_SOURCE_DIR}/cpp_inference/utils.cpp")
    set(ENABLE_CPP_INFERENCE ON)
else()
    message(WARNING "cpp_inference components not found at: ${CMAKE_CURRENT_SOURCE_DIR}/cpp_inference/inference_core.cpp")
    set(ENABLE_CPP_INFERENCE OFF)
endif()

# 过滤不存在的源文件
set(ALL_SOURCES ${CPP_BACKEND_SOURCES} ${LVGL_FRONTEND_SOURCES} ${CPP_INFERENCE_SOURCES})
set(VALID_SOURCES)
foreach(src ${ALL_SOURCES})
    # 如果是绝对路径（如生成的协议文件），直接添加
    if(IS_ABSOLUTE ${src})
        list(APPEND VALID_SOURCES ${src})
        message(STATUS "Adding generated file: ${src}")
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        list(APPEND VALID_SOURCES ${src})
    else()
        message(WARNING "Source file not found: ${src}")
    endif()
endforeach()

# ================================
# 主程序 - 一体化可执行文件
# ================================
add_executable(bamboo_integrated
    integrated_main.cpp
    ${VALID_SOURCES}
)

if(ENABLE_WAYLAND)
    # 确保协议文件在编译前生成
    add_dependencies(bamboo_integrated generate_wayland_protocols)
endif()

# 确保只有我们的主程序是默认构建目标
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT bamboo_integrated)

# 编译定义 - Wayland架构
target_compile_definitions(bamboo_integrated PRIVATE
    INTEGRATED_BUILD=1
    LV_CONF_INCLUDE_SIMPLE=1
    $<$<BOOL:${ENABLE_WAYLAND}>:USE_WAYLAND=1>
    $<$<BOOL:${ENABLE_WAYLAND}>:LV_USE_WAYLAND=1>
    $<$<BOOL:${ENABLE_WAYLAND}>:ENABLE_WAYLAND=1>
    HAS_DRM_EGL_BACKEND=1
    $<$<BOOL:${ENABLE_CUDA}>:ENABLE_CUDA>
    $<$<BOOL:${ENABLE_TENSORRT}>:ENABLE_TENSORRT>
    $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
    $<$<BOOL:${ENABLE_JSON}>:ENABLE_JSON>
    $<$<BOOL:${ENABLE_OPENCV}>:ENABLE_OPENCV>
    $<$<BOOL:${ENABLE_GSTREAMER}>:ENABLE_GSTREAMER>
    $<$<BOOL:${ENABLE_LVGL}>:ENABLE_LVGL>
    $<$<BOOL:${ENABLE_CPP_INFERENCE}>:ENABLE_CPP_INFERENCE>
    $<$<BOOL:${ENABLE_WAYLAND}>:ENABLE_WAYLAND>
    $<$<BOOL:${ENABLE_EGL}>:ENABLE_EGL>
    $<$<BOOL:${ENABLE_GBM}>:ENABLE_GBM>
    $<$<BOOL:${ENABLE_GLES2}>:ENABLE_GLES2>
    $<$<BOOL:${ENABLE_DRM_LIB}>:ENABLE_DRM_LIB>
    $<$<BOOL:${ENABLE_XKBCOMMON}>:ENABLE_XKBCOMMON>
    $<$<STREQUAL:${TARGET_ARCH},aarch64>:JETSON_ORIN_NX>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(bamboo_integrated PRIVATE DEBUG=1)
endif()

# 链接库
target_link_libraries(bamboo_integrated 
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
    target_link_libraries(bamboo_integrated ${CUDA_LIBRARIES})
endif()

if(ENABLE_TENSORRT)
    target_link_libraries(bamboo_integrated ${TENSORRT_LIBRARIES})
endif()

if(ENABLE_MODBUS)
    target_link_libraries(bamboo_integrated ${MODBUS_LIBRARY})
endif()

if(ENABLE_JSON)
    target_link_libraries(bamboo_integrated nlohmann_json::nlohmann_json)
endif()

# Wayland库 - 标准Wayland客户端架构
if(ENABLE_WAYLAND)
    target_link_libraries(bamboo_integrated
        ${WAYLAND_CLIENT_LIBRARIES}
        ${WAYLAND_EGL_LIBRARIES}
    )
endif()

# EGL库 - Wayland需要EGL支持
if(ENABLE_EGL)
    target_link_libraries(bamboo_integrated ${EGL_LIBRARIES})
endif()

# xkbcommon库 - Wayland键盘输入支持
if(ENABLE_XKBCOMMON)
    target_link_libraries(bamboo_integrated ${XKBCOMMON_LIBRARIES})
endif()

# LVGL库 - 确保正确链接
if(ENABLE_LVGL)
    if(TARGET lvgl)
        target_link_libraries(bamboo_integrated lvgl)
        message(STATUS "Linking LVGL library to bamboo_integrated")
    elseif(TARGET lvgl::lvgl)
        target_link_libraries(bamboo_integrated lvgl::lvgl)
        message(STATUS "Linking LVGL namespace library to bamboo_integrated")
    elseif(LVGL_FOUND AND LVGL_LIBRARIES)
        # 使用pkg-config找到的LVGL库
        target_link_libraries(bamboo_integrated ${LVGL_LIBRARIES})
        message(STATUS "Linking pkg-config LVGL libraries: ${LVGL_LIBRARIES}")
    else()
        # 查找已安装的LVGL
        find_package(lvgl QUIET)
        if(lvgl_FOUND)
            target_link_libraries(bamboo_integrated lvgl::lvgl)
            message(STATUS "Linking system LVGL library to bamboo_integrated")
        else()
            message(WARNING "LVGL library not found for linking")
            set(ENABLE_LVGL OFF)
        endif()
    endif()
else()
    message(WARNING "LVGL not available for linking")
endif()


# ================================
# 服务文件配置和安装
# ================================

# 配置systemd服务文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deploy/systemd/bamboo-cpp-lvgl.service.in")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/deploy/systemd/bamboo-cpp-lvgl.service.in"
        "${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service"
        @ONLY
    )
    message(STATUS "Service file configured: ${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service")
else()
    message(WARNING "Service template file not found: deploy/systemd/bamboo-cpp-lvgl.service.in")
endif()

# ================================
# 安装配置
# ================================
install(TARGETS bamboo_integrated
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

# 安装配置文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
    install(DIRECTORY config/
        DESTINATION share/bamboo/config
        COMPONENT Runtime
    )
endif()

# 安装模型文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
    install(DIRECTORY models/
        DESTINATION share/bamboo/models
        COMPONENT Runtime
    )
endif()

# 安装systemd服务文件
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/bamboo-cpp-lvgl.service"
        DESTINATION /etc/systemd/system
        COMPONENT Runtime
    )
endif()

# ================================
# 构建信息显示
# ================================
message(STATUS "")
message(STATUS "=== 一体化竹子识别系统构建配置 ===")
message(STATUS "项目名称: ${PROJECT_NAME}")
message(STATUS "版本: ${PROJECT_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "目标架构: ${TARGET_ARCH}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "安装前缀: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "依赖库状态:")
message(STATUS "  OpenCV: ${ENABLE_OPENCV} (${OpenCV_VERSION})")
message(STATUS "  GStreamer: ${ENABLE_GSTREAMER} (${GSTREAMER_VERSION})")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  TensorRT: ${ENABLE_TENSORRT}")
message(STATUS "  libmodbus: ${ENABLE_MODBUS}")
message(STATUS "  nlohmann_json: ${ENABLE_JSON}")
message(STATUS "  Wayland: ${ENABLE_WAYLAND}")
message(STATUS "    - wayland-client: ${WAYLAND_CLIENT_VERSION}")
message(STATUS "    - wayland-egl: ${WAYLAND_EGL_VERSION}")
if(WAYLAND_PROTOCOLS_FOUND)
message(STATUS "    - wayland-protocols: ${WAYLAND_PROTOCOLS_VERSION}")
endif()
message(STATUS "  EGL: ${ENABLE_EGL}")
message(STATUS "  GBM: ${ENABLE_GBM}")
message(STATUS "  OpenGL ES: ${ENABLE_GLES2}")
message(STATUS "  DRM库: ${ENABLE_DRM_LIB}")
message(STATUS "  xkbcommon: ${ENABLE_XKBCOMMON}")
message(STATUS "  LVGL: ${ENABLE_LVGL}")
message(STATUS "  cpp_inference: ${ENABLE_CPP_INFERENCE}")
message(STATUS "")
message(STATUS "架构迁移:")
message(STATUS "  DRM直接访问: 已替换为DRM EGL共享架构")
message(STATUS "  GBM显示后端: 已启用 (DRM EGL)")
message(STATUS "  OpenGL ES渲染: 已启用")
message(STATUS "  Wayland合成器: 已启用")
message(STATUS "  lv_drivers/wayland: 已集成")
message(STATUS "")
message(STATUS "组件模块:")
message(STATUS "  cpp_backend: 已集成")
message(STATUS "  lvgl_frontend: 已集成")
if(ENABLE_CPP_INFERENCE)
    message(STATUS "  cpp_inference: 已集成")
else()
    message(STATUS "  cpp_inference: 未找到")
endif()
message(STATUS "")
message(STATUS "有效源文件数量: ${VALID_SOURCES}")
message(STATUS "编译器: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "========================================")
