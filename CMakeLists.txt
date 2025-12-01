cmake_minimum_required(VERSION 3.16)
project(BambooQt VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_GSTREAMER "Build with GStreamer/DeepStream pipeline" ON)
option(ENABLE_MODBUS "Build with libmodbus Modbus TCP client" ON)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Multimedia)

# Modbus (optional)
set(MODBUS_FOUND FALSE)
if(ENABLE_MODBUS)
    pkg_check_modules(MODBUS QUIET modbus)
    if(NOT MODBUS_FOUND)
        pkg_check_modules(MODBUS QUIET libmodbus)
    endif()
    if(MODBUS_FOUND)
        message(STATUS "Found Modbus: ${MODBUS_VERSION}")
    else()
        message(WARNING "libmodbus not found; Modbus client will be disabled. Install libmodbus-dev to enable.")
    endif()
endif()

if(ENABLE_GSTREAMER)
    pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
    pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
    pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
    pkg_check_modules(GSTREAMER_RTSP QUIET gstrtspserver-1.0)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(QT_UI_SOURCES
    qt_ui/main.cpp
    qt_ui/StateModels.cpp
    qt_ui/ModbusClient.cpp
    qt_ui/DeepStreamRunner.cpp
)

set(QML_RESOURCES qt_ui/qml/qml.qrc)

add_executable(bamboo_qt_ui
    ${QT_UI_SOURCES}
    ${QML_RESOURCES}
)

target_include_directories(bamboo_qt_ui PRIVATE qt_ui)
if(MODBUS_FOUND)
    target_include_directories(bamboo_qt_ui PRIVATE ${MODBUS_INCLUDE_DIRS})
endif()

target_compile_definitions(bamboo_qt_ui PRIVATE
    MODEL_PATH_DEFAULT="${CMAKE_SOURCE_DIR}/models/best.onnx"
)

target_link_libraries(bamboo_qt_ui PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Qml
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Multimedia
    Threads::Threads
)

if(MODBUS_FOUND)
    target_link_libraries(bamboo_qt_ui PRIVATE ${MODBUS_LIBRARIES})
    target_compile_definitions(bamboo_qt_ui PRIVATE ENABLE_MODBUS=1)
endif()

if(ENABLE_GSTREAMER)
    target_include_directories(bamboo_qt_ui PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS}
        ${GSTREAMER_RTSP_INCLUDE_DIRS}
    )
    target_link_libraries(bamboo_qt_ui PRIVATE
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${GSTREAMER_VIDEO_LIBRARIES}
        ${GSTREAMER_RTSP_LIBRARIES}
    )
    target_compile_definitions(bamboo_qt_ui PRIVATE ENABLE_GSTREAMER=1)
endif()

install(TARGETS bamboo_qt_ui RUNTIME DESTINATION bin)
