#!/bin/bash

# =====================================================================
# 智能切竹机项目统一启动脚本
# 支持编译和运行整个项目（C++后端 + Qt前端）
# =====================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 项目信息
PROJECT_NAME="智能切竹机控制系统"
VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_header() {
    clear
    echo "========================================="
    echo "🎋 $PROJECT_NAME"
    echo "📋 版本: $VERSION"
    echo "📁 路径: $SCRIPT_DIR"
    echo "========================================="
    echo ""
}

show_menu() {
    echo "请选择操作："
    echo ""
    echo "1. 编译整个项目"
    echo "2. 编译C++后端"
    echo "3. 编译Qt前端"
    echo "4. 运行C++后端"
    echo "5. 运行Qt前端"
    echo "6. 运行完整系统（后端+前端）"
    echo "7. 清理构建文件"
    echo "8. 查看系统信息"
    echo "9. 调试Qt前端（详细日志）"
    echo "0. 退出"
    echo ""
    echo -n "请输入选择 [0-9]: "
}

build_cpp_backend() {
    log_info "编译C++后端..."
    cd "$SCRIPT_DIR/cpp_backend"
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    log_success "C++后端编译完成"
}

build_qt_frontend() {
    log_info "编译Qt前端..."
    cd "$SCRIPT_DIR/qt_frontend"
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    log_success "Qt前端编译完成"
}

build_all() {
    log_info "开始编译整个项目..."
    
    # 使用系统Makefile
    cd "$SCRIPT_DIR"
    if [ -f "Makefile" ]; then
        make all
    else
        build_cpp_backend
        build_qt_frontend
    fi
    
    log_success "整个项目编译完成！"
}

# 设置设备权限
setup_device_permissions() {
    log_info "配置设备权限..."
    
    # 摄像头权限
    for device in /dev/video*; do
        if [ -e "$device" ]; then
            sudo chmod 666 "$device" 2>/dev/null || true
            log_info "设置摄像头权限: $device"
        fi
    done
    
    # 串口权限
    for device in /dev/ttyUSB* /dev/ttyACM* /dev/ttyS*; do
        if [ -e "$device" ]; then
            sudo chmod 666 "$device" 2>/dev/null || true
        fi
    done
    
    # 帧缓冲权限
    for device in /dev/fb*; do
        if [ -e "$device" ]; then
            sudo chmod 666 "$device" 2>/dev/null || true
        fi
    done
}

# 智能平台插件检测和设置
detect_and_set_qt_platform() {
    log_info "检测可用的Qt平台插件..."
    
    # 优先尝试EGLFS（适合Jetson等嵌入式设备）
    if [ -c "/dev/fb0" ] || [ -d "/sys/class/drm" ]; then
        # 设置EGLFS环境
        export QT_QPA_PLATFORM=eglfs
        export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
        export QT_OPENGL=es2
        export QT_QPA_FONTDIR=/usr/share/fonts
        
        log_info "使用EGLFS平台 (推荐用于Jetson设备)"
        return 0
    fi
    
    # 尝试Wayland
    if [ -n "$WAYLAND_DISPLAY" ]; then
        export QT_QPA_PLATFORM=wayland
        log_info "使用Wayland平台"
        return 0
    fi
    
    # 尝试XCB (X11)
    if [ -n "$DISPLAY" ]; then
        export QT_QPA_PLATFORM=xcb
        log_info "使用XCB平台"
        return 0
    fi
    
    # 最后尝试LinuxFB
    if [ -c "/dev/fb0" ]; then
        export QT_QPA_PLATFORM=linuxfb
        export QT_QPA_FB_DISABLE_INPUT=1
        log_info "使用LinuxFB平台"
        return 0
    fi
    
    # 如果都不可用，使用minimal平台
    export QT_QPA_PLATFORM=minimal
    log_warning "使用Minimal平台（无显示输出）"
    return 0
}

# 设置Qt环境变量
setup_qt_environment() {
    # Qt模块路径
    export QT_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/qt6/plugins:/usr/lib/qt6/plugins"
    export QML_IMPORT_PATH="/usr/lib/aarch64-linux-gnu/qt6/qml:/usr/lib/qt6/qml"
    
    # OpenGL设置
    export QT_OPENGL=es2
    export QT_QUICK_BACKEND=software
    
    # 禁用Qt日志过滤
    export QT_LOGGING_RULES="*=true"
    
    # 设置字体路径
    export QT_QPA_FONTDIR="/usr/share/fonts"
    
    log_info "Qt环境变量已配置"
}

<<<<<<< HEAD
# 运行C++后端
run_cpp_backend() {
    log_info "运行C++后端..."
    cd "$SCRIPT_DIR/cpp_backend"
    
    if [ ! -f "build/bamboo_cut_system" ]; then
        log_error "C++后端未编译，请先编译"
        return 1
    fi
    
    cd build
    log_info "启动智能切竹机后端系统..."
    ./bamboo_cut_system
}

# 尝试不同平台启动
try_different_platforms() {
    local app_path="$1"
    local debug="$2"
=======
# 尝试不同平台启动
try_different_platforms() {
    local app_path="$1"
>>>>>>> 708576450b153b7e54001f5d377eaf976605ee7c
    local platforms=("eglfs" "wayland" "xcb" "linuxfb" "minimal")
    
    for platform in "${platforms[@]}"; do
        log_info "尝试 $platform 平台..."
<<<<<<< HEAD
        if [ "$debug" == "debug" ]; then
            # 调试模式，显示详细信息
            if QT_QPA_PLATFORM=$platform "$app_path"; then
                log_success "成功使用 $platform 平台启动"
                return 0
            fi
        else
            # 正常模式，隐藏错误输出
            if QT_QPA_PLATFORM=$platform "$app_path" 2>/dev/null; then
                log_success "成功使用 $platform 平台启动"
                return 0
            fi
=======
        if QT_QPA_PLATFORM=$platform "$app_path" 2>/dev/null; then
            log_success "成功使用 $platform 平台启动"
            return 0
>>>>>>> 708576450b153b7e54001f5d377eaf976605ee7c
        fi
    done
    
    log_error "所有平台都启动失败"
    return 1
}

<<<<<<< HEAD
# 调试Qt前端
debug_qt_frontend() {
    log_info "调试模式运行Qt前端..."
    cd "$SCRIPT_DIR/qt_frontend"
    
    if [ ! -f "build/bamboo_controller_qt" ]; then
        log_error "Qt前端未编译，请先编译"
        return 1
    fi
    
    log_info "配置Qt环境和平台插件（调试模式）..."
    
    # 执行配置（不设置设备权限）
    setup_qt_environment
    detect_and_set_qt_platform
    
    # 显示当前配置
    log_info "当前Qt配置："
    echo "  平台插件: ${QT_QPA_PLATFORM:-未设置}"
    echo "  OpenGL: ${QT_OPENGL:-未设置}"
    echo "  插件路径: ${QT_PLUGIN_PATH:-未设置}"
    
    # 显示更多调试信息
    log_info "调试信息："
    echo "  Qt版本: $(qmake -v 2>/dev/null | grep Qt || echo '未找到')"
    echo "  可执行文件: $(file build/bamboo_controller_qt)"
    echo "  依赖库检查:"
    ldd build/bamboo_controller_qt | head -10
    
    # 运行应用程序（显示详细错误）
    cd build
    log_info "启动智能切竹机控制程序（调试模式）..."
    
    # 直接运行，显示所有错误信息
    if ! ./bamboo_controller_qt; then
        log_warning "默认平台启动失败，尝试其他平台（调试模式）..."
        try_different_platforms "./bamboo_controller_qt" "debug"
    else
        log_success "Qt前端启动成功"
    fi
}

# 运行完整系统
run_full_system() {
    log_info "运行完整系统（后端+前端）..."
    
    # 检查编译状态
    if [ ! -f "$SCRIPT_DIR/cpp_backend/build/bamboo_cut_system" ]; then
        log_error "C++后端未编译，请先编译"
        return 1
    fi
    
    if [ ! -f "$SCRIPT_DIR/qt_frontend/build/bamboo_controller_qt" ]; then
        log_error "Qt前端未编译，请先编译"
        return 1
    fi
    
    log_info "启动后端系统..."
    cd "$SCRIPT_DIR/cpp_backend/build"
    ./bamboo_cut_system &
    BACKEND_PID=$!
    
    sleep 2  # 等待后端启动
    
    log_info "启动前端界面..."
    cd "$SCRIPT_DIR"
    run_qt_frontend
    
    # 清理后端进程
    if kill -0 $BACKEND_PID 2>/dev/null; then
        log_info "停止后端系统..."
        kill $BACKEND_PID
    fi
}

=======
>>>>>>> 708576450b153b7e54001f5d377eaf976605ee7c
run_qt_frontend() {
    log_info "运行Qt前端..."
    cd "$SCRIPT_DIR/qt_frontend"
    
    if [ ! -f "build/bamboo_controller_qt" ]; then
        log_error "Qt前端未编译，请先编译"
        return 1
    fi
    
    log_info "配置Qt环境和平台插件..."
    
    # 执行配置
    setup_device_permissions
    setup_qt_environment
    detect_and_set_qt_platform
    
    # 显示当前配置
    log_info "当前Qt配置："
    echo "  平台插件: ${QT_QPA_PLATFORM:-未设置}"
    echo "  OpenGL: ${QT_OPENGL:-未设置}"
    echo "  插件路径: ${QT_PLUGIN_PATH:-未设置}"
    
    # 运行应用程序
    cd build
    log_info "启动智能切竹机控制程序..."
    
    # 尝试运行，如果失败则尝试其他平台
    if ! ./bamboo_controller_qt 2>/dev/null; then
        log_warning "默认平台启动失败，尝试其他平台..."
        try_different_platforms "./bamboo_controller_qt"
    else
        log_success "Qt前端启动成功"
    fi
}

clean_builds() {
    log_info "清理构建文件..."
    
    # 清理C++后端
    if [ -d "$SCRIPT_DIR/cpp_backend/build" ]; then
        rm -rf "$SCRIPT_DIR/cpp_backend/build"
        log_success "已清理C++后端构建文件"
    fi
    
    # 清理Qt前端
    if [ -d "$SCRIPT_DIR/qt_frontend/build" ]; then
        rm -rf "$SCRIPT_DIR/qt_frontend/build"
        log_success "已清理Qt前端构建文件"
    fi
    
    log_success "构建文件清理完成"
}

show_system_info() {
    log_info "系统信息："
    echo ""
    echo "🖥️  操作系统: $(lsb_release -d 2>/dev/null | cut -f2 || uname -s)"
    echo "🏗️  架构: $(uname -m)"
    echo "🧠 CPU核心: $(nproc)"
    echo "💾 内存: $(free -h | grep Mem | awk '{print $2}')"
    echo ""
    
    # Qt信息
    if command -v qt6-config >/dev/null 2>&1; then
        echo "🎨 Qt版本: $(qt6-config --version)"
    elif command -v qmake >/dev/null 2>&1; then
        echo "🎨 Qt版本: $(qmake -v | grep Qt)"
    else
        echo "⚠️  Qt未安装或未在PATH中"
    fi
    
    # OpenCV信息
    if pkg-config --exists opencv4; then
        echo "👁️  OpenCV版本: $(pkg-config --modversion opencv4)"
    elif pkg-config --exists opencv; then
        echo "👁️  OpenCV版本: $(pkg-config --modversion opencv)"
    else
        echo "⚠️  OpenCV未安装"
    fi
    
    # GCC信息
    if command -v gcc >/dev/null 2>&1; then
        echo "🔨 GCC版本: $(gcc --version | head -n1)"
    fi
    
    # 显示可用的Qt平台插件
    echo ""
    echo "🎯 Qt平台插件信息："
    if [ -n "$DISPLAY" ]; then
        echo "  DISPLAY: $DISPLAY"
    fi
    if [ -n "$WAYLAND_DISPLAY" ]; then
        echo "  WAYLAND_DISPLAY: $WAYLAND_DISPLAY"
    fi
    if [ -c "/dev/fb0" ]; then
        echo "  帧缓冲: /dev/fb0 可用"
    fi
    
    echo ""
}

main() {
    while true; do
        show_header
        show_menu
        
        read -r choice
        echo ""
        
        case $choice in
            1)
                build_all
                ;;
            2)
                build_cpp_backend
                ;;
            3)
                build_qt_frontend
                ;;
            4)
                run_qt_frontend
                ;;
            5)
                clean_builds
                ;;
            6)
                show_system_info
                ;;
            0)
                log_info "退出程序"
                exit 0
                ;;
            *)
                log_error "无效选择，请输入0-6之间的数字"
                ;;
        esac
        
        echo ""
        echo -n "按回车继续..."
        read -r
    done
}

# 参数处理
case "${1:-}" in
    "build")
        build_all
        ;;
    "clean")
        clean_builds
        ;;
    "run")
        run_qt_frontend
        ;;
    "info")
        show_system_info
        ;;
    *)
        main
        ;;
esac