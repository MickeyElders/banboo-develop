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
    echo "4. 运行Qt前端"
    echo "5. 清理构建文件"
    echo "6. 查看系统信息"
    echo "0. 退出"
    echo ""
    echo -n "请输入选择 [0-6]: "
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

run_qt_frontend() {
    log_info "运行Qt前端..."
    cd "$SCRIPT_DIR/qt_frontend"
    
    if [ ! -f "build/bamboo_controller_qt" ]; then
        log_error "Qt前端未编译，请先编译"
        return 1
    fi
    
    # 使用启动脚本
    if [ -f "run_bamboo_qt.sh" ]; then
        chmod +x run_bamboo_qt.sh
        ./run_bamboo_qt.sh
    else
        # 直接运行
        cd build
        ./bamboo_controller_qt
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