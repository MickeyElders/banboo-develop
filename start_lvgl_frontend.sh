#!/bin/bash

# 智能切竹机控制系统 - LVGL前端一键启动脚本
# 版本: 2.0.0
# 作者: BambooTech开发团队

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_NAME="智能切竹机控制系统 - LVGL版本"
FRONTEND_DIR="lvgl_frontend"
BUILD_DIR="$FRONTEND_DIR/build"
BINARY_NAME="bamboo_controller_lvgl"
INSTALL_PREFIX="/opt/bamboo"

# 打印函数
print_banner() {
    echo -e "${CYAN}"
    echo "=================================================="
    echo "     $PROJECT_NAME"
    echo "=================================================="
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[步骤]${NC} $1"
}

# 检查系统环境
check_environment() {
    print_step "检查系统环境..."
    
    # 检查是否在Linux环境下
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "此脚本需要在Linux环境下运行（推荐Jetson设备）"
        exit 1
    fi
    
    # 检查是否为aarch64架构（Jetson设备）
    if [[ $(uname -m) == "aarch64" ]]; then
        print_success "检测到Jetson设备 (aarch64架构)"
        export JETSON_DEVICE=true
    else
        print_warn "当前不是Jetson设备，部分功能可能受限"
        export JETSON_DEVICE=false
    fi
    
    # 检查基础工具
    local tools=("cmake" "make" "g++" "git" "pkg-config")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "缺少必要工具: $tool"
            print_info "请运行: sudo apt install -y build-essential cmake git pkg-config"
            exit 1
        fi
    done
    
    print_success "系统环境检查通过"
}

# 检查依赖
check_dependencies() {
    print_step "检查项目依赖..."
    
    # 检查OpenCV
    if ! pkg-config --exists opencv4; then
        print_error "未找到OpenCV 4.x"
        print_info "请安装OpenCV: sudo apt install -y libopencv-dev"
        exit 1
    fi
    
    # 检查CUDA（仅在Jetson设备上）
    if [[ "$JETSON_DEVICE" == "true" ]]; then
        if ! command -v nvcc &> /dev/null; then
            print_warn "未找到CUDA编译器，GPU加速功能可能不可用"
        else
            print_success "检测到CUDA支持"
        fi
    fi
    
    # 检查设备文件
    local devices=("/dev/fb0" "/dev/input/event0" "/dev/video0")
    for device in "${devices[@]}"; do
        if [[ -e "$device" ]]; then
            print_info "检测到设备: $device"
        else
            print_warn "未检测到设备: $device"
        fi
    done
    
    print_success "依赖检查完成"
}

# 初始化LVGL前端
initialize_frontend() {
    print_step "初始化LVGL前端..."
    
    # 检查前端目录是否存在
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        print_error "LVGL前端目录不存在: $FRONTEND_DIR"
        print_info "请确保已正确设置项目结构"
        exit 1
    fi
    
    cd "$FRONTEND_DIR"
    
    # 下载LVGL依赖
    if [[ ! -d "third_party/lvgl" ]]; then
        print_info "下载LVGL库..."
        mkdir -p third_party
        cd third_party
        git clone --depth 1 --branch release/v8.3 https://github.com/lvgl/lvgl.git
        cd ..
        
        # 复制配置文件
        if [[ -f "lv_conf.h" ]]; then
            cp lv_conf.h third_party/lvgl/
            print_info "LVGL配置文件已复制"
        fi
    else
        print_info "LVGL库已存在"
    fi
    
    print_success "前端初始化完成"
}

# 构建项目
build_project() {
    print_step "构建LVGL前端项目..."
    
    # 清理旧的构建
    if [[ -d "build" ]]; then
        print_info "清理旧的构建目录..."
        rm -rf build
    fi
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # CMake配置
    print_info "配置CMake..."
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_CXX_STANDARD=17"
        "-DCMAKE_C_STANDARD=11"
    )
    
    # Jetson优化
    if [[ "$JETSON_DEVICE" == "true" ]]; then
        cmake_args+=(
            "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
            "-DCMAKE_C_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math"
            "-DCMAKE_CXX_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math"
        )
        
        # 查找Jetson OpenCV
        local opencv_dir="/usr/lib/aarch64-linux-gnu/cmake/opencv4"
        if [[ -d "$opencv_dir" ]]; then
            cmake_args+=("-DOpenCV_DIR=$opencv_dir")
        fi
    fi
    
    # 运行CMake
    cmake "${cmake_args[@]}" ..
    
    # 编译
    print_info "编译项目..."
    local num_cores=$(nproc)
    print_info "使用 $num_cores 个CPU核心进行编译"
    make -j$num_cores
    
    cd ..
    print_success "项目构建完成"
}

# 设置权限
setup_permissions() {
    print_step "设置设备权限..."
    
    # 检查是否为root用户
    if [[ $EUID -eq 0 ]]; then
        print_info "以root用户运行，跳过权限设置"
        return
    fi
    
    # 设置设备权限
    local devices=("/dev/fb0" "/dev/input/event0" "/dev/video0" "/dev/nvidia0")
    for device in "${devices[@]}"; do
        if [[ -e "$device" ]]; then
            sudo chmod 666 "$device" 2>/dev/null || true
            print_info "设置设备权限: $device"
        fi
    done
    
    print_success "权限设置完成"
}

# 启动应用程序
start_application() {
    print_step "启动LVGL前端应用..."
    
    # 检查可执行文件
    local binary_path="build/$BINARY_NAME"
    if [[ ! -f "$binary_path" ]]; then
        print_error "找不到可执行文件: $binary_path"
        exit 1
    fi
    
    # 设置环境变量
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export DISPLAY=${DISPLAY:-:0}
    
    # 配置文件路径
    local config_file="resources/config/default_config.json"
    if [[ ! -f "$config_file" ]]; then
        print_warn "配置文件不存在: $config_file"
        config_file=""
    fi
    
    print_success "正在启动应用程序..."
    print_info "可执行文件: $binary_path"
    print_info "配置文件: ${config_file:-"使用默认配置"}"
    print_info "按 Ctrl+C 退出应用程序"
    echo
    
    # 启动应用程序
    if [[ -n "$config_file" ]]; then
        exec "./$binary_path" -c "$config_file" -f
    else
        exec "./$binary_path" -f
    fi
}

# 清理函数
cleanup() {
    print_info "正在清理..."
    # 恢复原始目录
    cd "$SCRIPT_DIR"
}

# 显示帮助信息
show_help() {
    echo "智能切竹机控制系统 - LVGL前端一键启动脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -b, --build-only    仅构建，不启动"
    echo "  -s, --skip-build    跳过构建，直接启动"
    echo "  -c, --clean         清理构建后重新构建"
    echo "  -i, --install       构建并安装到系统"
    echo "  --debug             调试模式构建"
    echo
    echo "示例:"
    echo "  $0                  # 完整的构建和启动流程"
    echo "  $0 -b               # 仅构建，不启动"
    echo "  $0 -s               # 跳过构建，直接启动"
    echo "  $0 -c               # 清理后重新构建和启动"
}

# 主函数
main() {
    local build_only=false
    local skip_build=false
    local clean_build=false
    local install_system=false
    local debug_mode=false
    
    # 保存脚本目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build-only)
                build_only=true
                shift
                ;;
            -s|--skip-build)
                skip_build=true
                shift
                ;;
            -c|--clean)
                clean_build=true
                shift
                ;;
            -i|--install)
                install_system=true
                shift
                ;;
            --debug)
                debug_mode=true
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置清理陷阱
    trap cleanup EXIT
    
    # 显示欢迎横幅
    print_banner
    
    # 执行主要流程
    check_environment
    check_dependencies
    initialize_frontend
    
    # 根据参数决定是否构建
    if [[ "$skip_build" != "true" ]]; then
        if [[ "$clean_build" == "true" ]]; then
            print_info "清理模式：将重新构建项目"
        fi
        build_project
        
        if [[ "$install_system" == "true" ]]; then
            print_step "安装到系统..."
            cd build
            sudo make install
            cd ..
            print_success "系统安装完成"
        fi
    fi
    
    # 根据参数决定是否启动
    if [[ "$build_only" != "true" ]]; then
        setup_permissions
        start_application
    else
        print_success "构建完成！"
        print_info "可执行文件位置: $FRONTEND_DIR/build/$BINARY_NAME"
    fi
}

# 运行主函数
main "$@"