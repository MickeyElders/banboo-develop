cmake_minimum_required(VERSION 3.24)
project(BambooQt VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_GSTREAMER "Build with GStreamer/DeepStream pipeline" ON)
option(ENABLE_MODBUS "Build with libmodbus Modbus TCP client" ON)

# Toolchain roots (Jetson: Qt6 custom prefix, DeepStream root)
set(QT6_ROOT "/opt/qt6" CACHE PATH "Qt6 install prefix (Qt6 >= 6.6 recommended)")
list(APPEND CMAKE_PREFIX_PATH "${QT6_ROOT}")
set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream root path")

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

set(QT_MIN_VERSION 6.6)
find_package(Qt6 ${QT_MIN_VERSION} REQUIRED COMPONENTS
    Core Gui Qml Quick QuickControls2 Multimedia)
qt_standard_project_setup()

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
    pkg_check_modules(GSTREAMER_RTSP QUIET gstreamer-rtsp-server-1.0)
    if(GSTREAMER_RTSP_FOUND)
        set(ENABLE_RTSP TRUE)
        message(STATUS "Found gstreamer-rtsp-server: ${GSTREAMER_RTSP_VERSION}")
    else()
        set(ENABLE_RTSP FALSE)
        message(WARNING "gstreamer-rtsp-server not found; RTSP server fallback disabled")
    endif()
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
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Multimedia
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
        ${DEEPSTREAM_ROOT}/sources/includes
    )
    target_link_directories(bamboo_qt_ui PRIVATE ${DEEPSTREAM_ROOT}/lib)
    target_link_libraries(bamboo_qt_ui PRIVATE
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${GSTREAMER_VIDEO_LIBRARIES}
    )
    find_library(NVBUFSURFACE_LIB nvbufsurface PATHS ${DEEPSTREAM_ROOT}/lib)
    find_library(NVDGST_HELPER_LIB nvdsgst_helper PATHS ${DEEPSTREAM_ROOT}/lib)
    find_library(NVDS_META_LIB nvds_meta PATHS ${DEEPSTREAM_ROOT}/lib)
    if(NVBUFSURFACE_LIB AND NVDGST_HELPER_LIB AND NVDS_META_LIB)
        target_link_libraries(bamboo_qt_ui PRIVATE
            ${NVBUFSURFACE_LIB}
            ${NVDGST_HELPER_LIB}
            ${NVDS_META_LIB}
        )
    else()
        message(WARNING "DeepStream libs not found under ${DEEPSTREAM_ROOT}; zero-copy path will be disabled until installed.")
    endif()
    if(ENABLE_RTSP)
        target_include_directories(bamboo_qt_ui PRIVATE ${GSTREAMER_RTSP_INCLUDE_DIRS})
        target_link_libraries(bamboo_qt_ui PRIVATE ${GSTREAMER_RTSP_LIBRARIES})
        target_compile_definitions(bamboo_qt_ui PRIVATE ENABLE_RTSP=1)
    endif()
    target_compile_definitions(bamboo_qt_ui PRIVATE ENABLE_GSTREAMER=1)
endif()

install(TARGETS bamboo_qt_ui RUNTIME DESTINATION bin)
