cmake_minimum_required(VERSION 3.16)
project(BambooControllerQt VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt配置
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Quick QuickWidgets OpenGL OpenGLWidgets Qml QuickControls2 Multimedia MultimediaWidgets SerialPort Network)

# OpenGL ES配置（适配Jetson设备）
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLES2 REQUIRED glesv2)
pkg_check_modules(EGL REQUIRED egl)

# OpenCV配置 - 适配Jetson OpenCV 4.8.0
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4.8 REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    # 备用方案：直接设置路径
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    set(OpenCV_LIBS "opencv_core;opencv_imgproc;opencv_highgui;opencv_imgcodecs")
    message(STATUS "Using fallback OpenCV paths")
endif()

# 源文件
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/videorenderer.cpp
    src/cameramanager.cpp
    src/bamboodetector.cpp
    src/configmanager.cpp
    src/systemcontroller.cpp
    src/touchcontroller.cpp
)

set(HEADERS
    include/mainwindow.h
    include/videorenderer.h
    include/cameramanager.h
    include/bamboodetector.h
    include/configmanager.h
    include/systemcontroller.h
    include/touchcontroller.h
)

set(QML_FILES
    qml/main.qml
    qml/components/VideoDisplay.qml
    qml/components/ControlPanel.qml
    qml/components/StatusBar.qml
    qml/components/SettingsDialog.qml
)

# 资源文件
qt6_add_resources(RESOURCES resources/resources.qrc)

# 设置Qt自动处理MOC、UIC、RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 可执行文件
qt6_add_executable(bamboo_controller_qt ${SOURCES} ${HEADERS} ${RESOURCES})

# Qt模块链接
target_link_libraries(bamboo_controller_qt PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Quick
    Qt6::QuickWidgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    Qt6::Qml
    Qt6::QuickControls2
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    Qt6::SerialPort
    Qt6::Network
    ${OpenCV_LIBS}
    ${GLES2_LIBRARIES}
    ${EGL_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

# 头文件目录
target_include_directories(bamboo_controller_qt PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4
    ${GLES2_INCLUDE_DIRS}
    ${EGL_INCLUDE_DIRS}
)

# 编译选项
target_compile_options(bamboo_controller_qt PRIVATE
    ${OpenCV_CFLAGS_OTHER}
)

# 编译定义
target_compile_definitions(bamboo_controller_qt PRIVATE
    USE_OPENGL_ES
    JETSON_PLATFORM
)

# 安装配置
install(TARGETS bamboo_controller_qt
    RUNTIME DESTINATION bin
)

# QML文件安装
install(DIRECTORY qml/
    DESTINATION share/bamboo_controller_qt/qml
)