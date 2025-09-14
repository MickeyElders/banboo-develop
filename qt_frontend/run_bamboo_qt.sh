#!/bin/bash

# =====================================================================
# 智能切竹机Qt前端启动脚本
# 适用于Jetson设备和其他Linux嵌入式系统
# =====================================================================

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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
PROJECT_NAME="智能切竹机Qt前端"
VERSION="2.0.0"
BUILD_DIR="build"
EXECUTABLE="bamboo_controller_qt"

log_info "🚀 启动 $PROJECT_NAME v$VERSION..."

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "工作目录: $SCRIPT_DIR"

# 检查可执行文件是否存在
if [ ! -f "$BUILD_DIR/$EXECUTABLE" ]; then
    log_error "找不到可执行文件: $BUILD_DIR/$EXECUTABLE"
    log_info "请先编译项目："
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make -j4"
    exit 1
fi

log_success "找到可执行文件: $BUILD_DIR/$EXECUTABLE"

# 设备权限设置
setup_permissions() {
    log_info "设置设备权限..."
    
    # 摄像头权限
    if ls /dev/video* >/dev/null 2>&1; then
        sudo chmod 666 /dev/video* 2>/dev/null || log_warning "无法设置摄像头权限，可能需要手动设置"
        log_success "摄像头设备权限已设置"
    else
        log_warning "未找到摄像头设备"
    fi
    
    # 帧缓冲设备权限
    if ls /dev/fb* >/dev/null 2>&1; then
        sudo chmod 666 /dev/fb* 2>/dev/null || log_warning "无法设置帧缓冲权限"
        log_success "帧缓冲设备权限已设置"
    fi
    
    # 串口权限
    if ls /dev/ttyUSB* >/dev/null 2>&1 || ls /dev/ttyACM* >/dev/null 2>&1; then
        sudo chmod 666 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || log_warning "无法设置串口权限"
        log_success "串口设备权限已设置"
    fi
}

# 检测可用的Qt平台插件
detect_qt_platform() {
    log_info "检测可用的Qt平台插件..."
    
    # 尝试获取可用平台列表
    AVAILABLE_PLATFORMS=$(./$BUILD_DIR/$EXECUTABLE -platform help 2>&1 | grep -oE "(eglfs|wayland|xcb|linuxfb|minimal)" | sort -u)
    
    if [ -n "$AVAILABLE_PLATFORMS" ]; then
        log_success "可用平台: $(echo $AVAILABLE_PLATFORMS | tr '\n' ' ')"
    else
        log_warning "无法检测Qt平台插件"
    fi
    
    echo "$AVAILABLE_PLATFORMS"
}

# 尝试不同的Qt平台
try_qt_platforms() {
    local platforms=("$@")
    
    for platform in "${platforms[@]}"; do
        log_info "尝试使用 $platform 平台..."
        
        # 设置平台特定的环境变量
        case $platform in
            "eglfs")
                export QT_QPA_PLATFORM=eglfs
                export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
                export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
                export QT_OPENGL=es2
                ;;
            "wayland-egl"|"wayland")
                export QT_QPA_PLATFORM=wayland-egl
                ;;
            "xcb")
                export QT_QPA_PLATFORM=xcb
                export DISPLAY=${DISPLAY:-:0}
                ;;
            "linuxfb")
                export QT_QPA_PLATFORM=linuxfb
                export QT_QPA_FB_DEVICE=/dev/fb0
                ;;
            "minimal")
                export QT_QPA_PLATFORM=minimal
                ;;
        esac
        
        log_info "使用平台: $QT_QPA_PLATFORM"
        
        # 尝试运行应用程序
        if timeout 10s ./$BUILD_DIR/$EXECUTABLE --version >/dev/null 2>&1; then
            log_success "平台 $platform 可用！"
            return 0
        else
            log_warning "平台 $platform 不可用"
        fi
    done
    
    return 1
}

# 设置完整的运行环境
setup_environment() {
    log_info "设置运行环境..."
    
    # 基本Qt环境
    export QT_LOGGING_RULES="*.debug=false"  # 减少日志输出
    export QML2_IMPORT_PATH="/usr/lib/aarch64-linux-gnu/qt6/qml:/usr/lib/qt6/qml"
    
    # OpenGL设置
    export LIBGL_ALWAYS_SOFTWARE=0
    export MESA_GL_VERSION_OVERRIDE=3.3
    
    # 字体设置
    export QT_QPA_FONTDIR="/usr/share/fonts"
    
    log_success "环境变量已设置"
}

# 主函数
main() {
    log_info "========================================="
    log_info "$PROJECT_NAME 启动器"
    log_info "版本: $VERSION"
    log_info "========================================="
    
    # 设置权限
    setup_permissions
    
    # 设置环境
    setup_environment
    
    # 检测平台
    AVAILABLE_PLATFORMS=$(detect_qt_platform)
    
    # 定义尝试顺序（适合Jetson设备）
    PLATFORM_ORDER=("eglfs" "wayland-egl" "wayland" "linuxfb" "xcb" "minimal")
    
    # 过滤可用平台
    FILTERED_PLATFORMS=()
    for platform in "${PLATFORM_ORDER[@]}"; do
        if echo "$AVAILABLE_PLATFORMS" | grep -q "$platform"; then
            FILTERED_PLATFORMS+=("$platform")
        fi
    done
    
    if [ ${#FILTERED_PLATFORMS[@]} -eq 0 ]; then
        FILTERED_PLATFORMS=("${PLATFORM_ORDER[@]}")  # 回退到所有平台
    fi
    
    log_info "尝试平台顺序: ${FILTERED_PLATFORMS[*]}"
    
    # 尝试运行
    if try_qt_platforms "${FILTERED_PLATFORMS[@]}"; then
        log_success "找到合适的平台，正在启动应用程序..."
        log_info "========================================="
        log_info "🎯 $PROJECT_NAME 正在运行..."
        log_info "使用平台: $QT_QPA_PLATFORM"
        log_info "按 Ctrl+C 退出"
        log_info "========================================="
        
        # 运行应用程序
        cd "$BUILD_DIR"
        exec ./$EXECUTABLE
    else
        log_error "无法找到合适的Qt平台插件"
        log_info "请尝试以下解决方案："
        echo "  1. 安装X11服务器: sudo apt install xorg lightdm"
        echo "  2. 启动显示服务: sudo systemctl start lightdm"
        echo "  3. 设置显示变量: export DISPLAY=:0"
        echo "  4. 手动指定平台: export QT_QPA_PLATFORM=eglfs"
        exit 1
    fi
}

# 信号处理
trap 'log_info "正在退出..."; exit 0' INT TERM

# 参数处理
case "${1:-}" in
    "-h"|"--help"|"help")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  -h, --help    显示此帮助信息"
        echo "  --debug       启用调试模式"
        echo "  --platform=X  强制使用指定平台 (eglfs|wayland|xcb|linuxfb)"
        echo ""
        echo "环境变量:"
        echo "  QT_QPA_PLATFORM    指定Qt平台插件"
        echo "  DISPLAY            X11显示变量"
        echo ""
        exit 0
        ;;
    "--debug")
        export QT_LOGGING_RULES="*.debug=true"
        log_info "调试模式已启用"
        ;;
    --platform=*)
        FORCE_PLATFORM="${1#--platform=}"
        export QT_QPA_PLATFORM="$FORCE_PLATFORM"
        log_info "强制使用平台: $FORCE_PLATFORM"
        cd "$BUILD_DIR"
        exec ./$EXECUTABLE
        ;;
esac

# 运行主函数
main "$@"