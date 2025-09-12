cmake_minimum_required(VERSION 3.16)
project(BambooControllerQt VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt配置
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Quick QuickWidgets OpenGL Multimedia MultimediaWidgets)

# OpenGL ES配置（适配Jetson设备）
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLES2 REQUIRED glesv2)
pkg_check_modules(EGL REQUIRED egl)

# OpenCV配置
pkg_check_modules(OpenCV REQUIRED opencv4)

# 源文件
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/videorenderer.cpp
    src/touchcontroller.cpp
    src/cameramanager.cpp
    src/bamboodetector.cpp
    src/systemcontroller.cpp
    src/configmanager.cpp
)

set(HEADERS
    include/mainwindow.h
    include/videorenderer.h
    include/touchcontroller.h
    include/cameramanager.h
    include/bamboodetector.h
    include/systemcontroller.h
    include/configmanager.h
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

# 可执行文件
qt6_add_executable(bamboo_controller_qt ${SOURCES} ${HEADERS} ${RESOURCES})

# Qt模块链接
target_link_libraries(bamboo_controller_qt
    Qt6::Core
    Qt6::Widgets
    Qt6::Quick
    Qt6::QuickWidgets
    Qt6::OpenGL
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    ${OpenCV_LIBRARIES}
    ${GLES2_LIBRARIES}
    ${EGL_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

# 头文件目录
target_include_directories(bamboo_controller_qt PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${GLES2_INCLUDE_DIRS}
    ${EGL_INCLUDE_DIRS}
)

# 编译定义
target_compile_definitions(bamboo_controller_qt PRIVATE
    ${OpenCV_CFLAGS_OTHER}
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