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

# LVGL配置
set(LVGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lvgl)
add_subdirectory(${LVGL_DIR})

# 头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lvgl
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
    
    # AI推理系统
    src/ai/tensorrt_engine.cpp
    src/ai/yolo_detector.cpp
    src/ai/detection_processor.cpp
    
    # GUI组件
    src/gui/video_view.cpp
    src/gui/control_panel.cpp
    src/gui/status_bar.cpp
    src/gui/settings_page.cpp
)

# 头文件
set(HEADERS
    include/app/main_app.h
    include/app/event_manager.h
    include/app/config_manager.h
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

# 链接库
target_link_libraries(bamboo_controller_lvgl
    PRIVATE
    lvgl
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    cudart
    cublas
    curand
    pthread
    dl
    m
)

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

# 字体和资源安装  
install(DIRECTORY resources/fonts/
    DESTINATION /opt/bamboo/fonts
)

install(DIRECTORY resources/images/
    DESTINATION /opt/bamboo/images
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