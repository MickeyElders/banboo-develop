# 一体化竹子识别系统 CMake配置
# 版本: 3.0.0 (Integrated Architecture)
cmake_minimum_required(VERSION 3.16)

project(BambooIntegratedSystem VERSION 3.0.0 LANGUAGES C CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 编译选项
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")

# 查找依赖包
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# libmodbus (可选)
pkg_check_modules(LIBMODBUS libmodbus)
if(LIBMODBUS_FOUND)
    message(STATUS "Found libmodbus: ${LIBMODBUS_VERSION}")
    add_definitions(-DENABLE_MODBUS)
else()
    message(WARNING "libmodbus not found - PLC通信功能将被禁用")
endif()

# GStreamer (可选) 
pkg_check_modules(GSTREAMER gstreamer-1.0)
if(GSTREAMER_FOUND)
    message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")
    add_definitions(-DENABLE_GSTREAMER)
else()
    message(WARNING "GStreamer not found - 高级视频功能将被禁用")
endif()

# CUDA (可选)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
else()
    message(WARNING "CUDA not found - GPU加速将被禁用")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_backend/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lvgl_frontend/include
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/lvgl
    ${OpenCV_INCLUDE_DIRS}
)

if(LIBMODBUS_FOUND)
    include_directories(${LIBMODBUS_INCLUDE_DIRS})
endif()

if(GSTREAMER_FOUND)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
endif()

# LVGL源文件
file(GLOB_RECURSE LVGL_SOURCES 
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/lvgl/src/*.c"
)

# C++后端源文件
file(GLOB_RECURSE CPP_BACKEND_SOURCES 
    "cpp_backend/src/*.cpp"
    "cpp_backend/src/*.c"
)

# LVGL前端源文件
file(GLOB_RECURSE LVGL_FRONTEND_SOURCES 
    "lvgl_frontend/src/*.cpp"
    "lvgl_frontend/src/*.c"
)

# 排除原有的main.cpp文件
list(FILTER CPP_BACKEND_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
list(FILTER LVGL_FRONTEND_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

# 主程序 - 一体化可执行文件
add_executable(bamboo_integrated 
    integrated_main.cpp
    ${CPP_BACKEND_SOURCES}
    ${LVGL_FRONTEND_SOURCES}
    ${LVGL_SOURCES}
)

# 链接库
target_link_libraries(bamboo_integrated 
    ${OpenCV_LIBS}
    Threads::Threads
    m
    dl
)

if(LIBMODBUS_FOUND)
    target_link_libraries(bamboo_integrated ${LIBMODBUS_LIBRARIES})
    target_link_directories(bamboo_integrated PRIVATE ${LIBMODBUS_LIBRARY_DIRS})
endif()

if(GSTREAMER_FOUND)
    target_link_libraries(bamboo_integrated ${GSTREAMER_LIBRARIES})
    target_link_directories(bamboo_integrated PRIVATE ${GSTREAMER_LIBRARY_DIRS})
endif()

if(CUDA_FOUND)
    target_link_libraries(bamboo_integrated ${CUDA_LIBRARIES})
endif()

# 编译定义
target_compile_definitions(bamboo_integrated PRIVATE
    INTEGRATED_BUILD=1
    LV_CONF_INCLUDE_SIMPLE
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(bamboo_integrated PRIVATE DEBUG=1)
endif()

# 安装配置
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

# 显示构建信息
message(STATUS "=== 一体化竹子识别系统构建配置 ===")
message(STATUS "项目名称: ${PROJECT_NAME}")
message(STATUS "版本: ${PROJECT_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "安装前缀: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "编译器: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

if(LIBMODBUS_FOUND)
    message(STATUS "libmodbus: 已启用")
else()
    message(STATUS "libmodbus: 已禁用")
endif()

if(GSTREAMER_FOUND)
    message(STATUS "GStreamer: 已启用")
else()
    message(STATUS "GStreamer: 已禁用")
endif()

if(CUDA_FOUND)
    message(STATUS "CUDA: 已启用")
else()
    message(STATUS "CUDA: 已禁用")
endif()

message(STATUS "=========================================")