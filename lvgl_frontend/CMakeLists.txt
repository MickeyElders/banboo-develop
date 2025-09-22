cmake_minimum_required(VERSION 3.16)
project(BambooControllerLVGL VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

# Jetson平台特定优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_definitions(JETSON_PLATFORM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a78")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a78")
endif()

# 查找依赖包
find_package(PkgConfig REQUIRED)

# OpenCV - 使用Jetson优化版本
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4.8 REQUIRED)

# CUDA和TensorRT
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# TensorRT路径
set(TENSORRT_ROOT /usr/include/x86_64-linux-gnu)
if(EXISTS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_ROOT /usr/include/aarch64-linux-gnu)
endif()

# LVGL配置 - 手动集成以避免examples/demos冲突
set(LVGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lvgl)

# 收集LVGL源文件
file(GLOB_RECURSE LVGL_SOURCES
    ${LVGL_DIR}/src/*.c
    ${LVGL_DIR}/src/*.h
)

# 排除examples和demos目录
list(FILTER LVGL_SOURCES EXCLUDE REGEX ".*/examples/.*")
list(FILTER LVGL_SOURCES EXCLUDE REGEX ".*/demos/.*")

# 创建LVGL库
add_library(lvgl STATIC ${LVGL_SOURCES})

# LVGL头文件目录
target_include_directories(lvgl PUBLIC
    ${LVGL_DIR}
    ${LVGL_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}  # 包含lv_conf.h
)

# 编译定义
target_compile_definitions(lvgl PUBLIC
    LV_CONF_INCLUDE_SIMPLE=1
)

# 头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lvgl
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_ROOT}
    /usr/local/cuda/include
)

# 源文件
set(SOURCES
    # 主程序
    src/main.cpp
    
    # 应用层
    src/app/main_app.cpp
    src/app/event_manager.cpp
    src/app/config_manager.cpp
    
    # 系统层 - LVGL移植
    src/system/lv_port_tick.c
    
    # 显示系统
    src/display/framebuffer_driver.cpp
    src/display/lvgl_display.cpp
    src/display/gpu_accelerated.cpp
    
    # 输入系统
    src/input/touch_driver.cpp
    src/input/input_calibration.cpp
    
    # 摄像头系统
    src/camera/v4l2_camera.cpp
    src/camera/cuda_processor.cpp
    src/camera/camera_manager.cpp
    
    # 后端通信系统
    src/backend/backend_client.cpp
    
    # AI推理系统
    src/ai/tensorrt_engine.cpp
    src/ai/yolo_detector.cpp
    src/ai/detection_processor.cpp
    
    # GUI组件
    src/gui/video_view.cpp
    src/gui/control_panel.cpp
    src/gui/status_bar.cpp
    src/gui/settings_page.cpp
    
    # Font Awesome图标字体
    resources/fonts/lv_font_awesome_16.c
)

# 头文件
set(HEADERS
    include/app/main_app.h
    include/app/event_manager.h
    include/app/config_manager.h
    include/system/lv_port_tick.h
    include/display/framebuffer_driver.h
    include/display/lvgl_display.h
    include/display/gpu_accelerated.h
    include/input/touch_driver.h
    include/input/input_calibration.h
    include/camera/v4l2_camera.h
    include/camera/cuda_processor.h
    include/camera/camera_manager.h
    include/ai/tensorrt_engine.h
    include/ai/yolo_detector.h
    include/ai/detection_processor.h
    include/gui/video_view.h
    include/gui/control_panel.h
    include/gui/status_bar.h
    include/gui/settings_page.h
    include/common/types.h
    include/common/utils.h
)

# 可执行文件
add_executable(bamboo_controller_lvgl ${SOURCES} ${HEADERS})

# Jetson设备特定的CUDA库路径配置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 根据实际检查结果配置CUDA 12.6库路径
    set(CUDA_LIB_PATHS
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib"
        "/usr/local/cuda/targets/aarch64-linux/lib"
        "/usr/local/cuda/lib64"
        "/usr/lib/aarch64-linux-gnu"
    )
    
    # TensorRT库路径
    set(TENSORRT_LIB_PATHS
        "/usr/lib/aarch64-linux-gnu"
        "/usr/local/lib"
        "/opt/nvidia/vpi2/lib64"
    )
    
    # 使用find_library确保库文件存在并获取完整路径
    find_library(CUDART_LIB cudart PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(CUBLAS_LIB cublas PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(CURAND_LIB curand PATHS ${CUDA_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_PATHS} NO_DEFAULT_PATH)
    
    # 检查库是否找到
    message(STATUS "Jetson Orin NX detected - configuring CUDA 12.6 libraries")
    message(STATUS "CUDA library paths: ${CUDA_LIB_PATHS}")
    message(STATUS "Found CUDART: ${CUDART_LIB}")
    message(STATUS "Found CUBLAS: ${CUBLAS_LIB}")
    message(STATUS "Found CURAND: ${CURAND_LIB}")
    message(STATUS "Found NVINFER: ${NVINFER_LIB}")
    message(STATUS "Found NVONNXPARSER: ${NVONNXPARSER_LIB}")
    
    # 如果找不到库，显示警告并使用备用方案
    if(NOT CUDART_LIB)
        message(WARNING "CUDART library not found, trying fallback search")
        find_library(CUDART_LIB cudart)
    endif()
    
    if(NOT CUBLAS_LIB)
        message(WARNING "CUBLAS library not found, trying fallback search")
        find_library(CUBLAS_LIB cublas)
    endif()
    
    if(NOT CURAND_LIB)
        message(WARNING "CURAND library not found, trying fallback search")
        find_library(CURAND_LIB curand)
    endif()
endif()

# 链接库
target_link_libraries(bamboo_controller_lvgl
    PRIVATE
    lvgl
    ${OpenCV_LIBS}
    pthread
    dl
    m
)

# Jetson设备的CUDA和TensorRT库
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 只链接找到的库
    set(JETSON_LIBS)
    
    if(CUDART_LIB)
        list(APPEND JETSON_LIBS ${CUDART_LIB})
    endif()
    
    if(CUBLAS_LIB)
        list(APPEND JETSON_LIBS ${CUBLAS_LIB})
    endif()
    
    if(CURAND_LIB)
        list(APPEND JETSON_LIBS ${CURAND_LIB})
    endif()
    
    if(NVINFER_LIB)
        list(APPEND JETSON_LIBS ${NVINFER_LIB})
    endif()
    
    if(NVONNXPARSER_LIB)
        list(APPEND JETSON_LIBS ${NVONNXPARSER_LIB})
    endif()
    
    if(JETSON_LIBS)
        target_link_libraries(bamboo_controller_lvgl PRIVATE ${JETSON_LIBS})
        message(STATUS "CUDA and TensorRT libraries linked for Jetson: ${JETSON_LIBS}")
    else()
        message(WARNING "No CUDA/TensorRT libraries found - AI功能将被禁用")
        # 暂时禁用CUDA和TensorRT相关的编译定义
        target_compile_definitions(bamboo_controller_lvgl PRIVATE DISABLE_CUDA DISABLE_TENSORRT)
    endif()
endif()

# 编译定义
target_compile_definitions(bamboo_controller_lvgl
    PRIVATE
    LV_CONF_INCLUDE_SIMPLE
    USE_FRAMEBUFFER
    ENABLE_TENSORRT
    ENABLE_CUDA
    JETSON_ORIN_NX
)

# 安装规则
install(TARGETS bamboo_controller_lvgl
    RUNTIME DESTINATION /opt/bamboo/bin
)

# 配置文件安装
install(DIRECTORY resources/config/
    DESTINATION /opt/bamboo/config
)

# 资源安装
install(DIRECTORY resources/images/
    DESTINATION /opt/bamboo/images
    OPTIONAL
)

# 创建systemd服务文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bamboo-controller.service.in
    ${CMAKE_CURRENT_BINARY_DIR}/bamboo-controller.service
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bamboo-controller.service
    DESTINATION /etc/systemd/system
)