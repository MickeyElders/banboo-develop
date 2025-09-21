#!/bin/bash

# 智能切竹机控制系统 - 完整系统启动脚本
# 版本: 2.0.0
# 作者: BambooTech开发团队
# 包含C++后端和LVGL前端

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
PROJECT_NAME="智能切竹机控制系统 - C++后端+LVGL前端"
FRONTEND_DIR="lvgl_frontend"
BACKEND_DIR="cpp_backend"
BUILD_DIR="$FRONTEND_DIR/build"
BACKEND_BUILD_DIR="$BACKEND_DIR/build"
BINARY_NAME="bamboo_controller_lvgl"
BACKEND_BINARY="bamboo_cut_backend"
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
    
    # 检查项目目录结构
    if [[ ! -d "$BACKEND_DIR" ]] || [[ ! -d "$FRONTEND_DIR" ]]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
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

# 清理老版本和缓存
cleanup_old_versions() {
    print_step "清理老版本进程和配置..."
    
    # 停止可能运行的老版本进程
    local process_names=("bamboo_controller_lvgl" "bamboo_controller" "qt_frontend" "bamboo_cut_backend")
    for process in "${process_names[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            print_warn "发现运行中的进程: $process"
            print_info "正在终止进程..."
            pkill -f "$process" || true
            sleep 2
            
            # 强制终止如果还在运行
            if pgrep -f "$process" > /dev/null; then
                print_warn "强制终止进程: $process"
                pkill -9 -f "$process" || true
                sleep 1
            fi
            print_success "进程 $process 已终止"
        fi
    done
    
    # 杀死占用显示器的关键进程
    print_info "检查并终止占用显示器的进程..."
    local display_processes=(
        "yegra"
        "camrtc"
        "capture-control"
        "capture"
        "vi-output"
        "nvidiaPvaAllowd"
        "gnome-shell"
        "unity"
        "xorg"
        "wayland"
        "gdm"
        "lightdm"
        "sddm"
    )
    
    # 第一轮：尝试优雅终止
    for process in "${display_processes[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            print_warn "发现占用显示器的进程: $process"
            print_info "尝试优雅终止进程: $process"
            pkill -TERM -f "$process" 2>/dev/null || true
        fi
    done
    
    # 等待进程退出
    sleep 3
    
    # 第二轮：强制终止仍在运行的进程
    for process in "${display_processes[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            print_warn "强制终止顽固进程: $process"
            pkill -9 -f "$process" 2>/dev/null || true
            sleep 1
            
            # 再次检查并强制终止
            if pgrep -f "$process" > /dev/null; then
                print_warn "进程 $process 仍在运行，使用killall强制终止"
                killall -9 "$process" 2>/dev/null || true
            fi
        fi
    done
    
    # 最终检查
    sleep 2
    local remaining_processes=()
    for process in "${display_processes[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            remaining_processes+=("$process")
        fi
    done
    
    if [ ${#remaining_processes[@]} -gt 0 ]; then
        print_warn "以下进程仍在运行: ${remaining_processes[*]}"
        print_info "尝试通过PID直接终止..."
        
        # 通过PID直接终止
        for process in "${remaining_processes[@]}"; do
            local pids=$(pgrep -f "$process" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                for pid in $pids; do
                    print_info "强制终止PID: $pid ($process)"
                    kill -9 "$pid" 2>/dev/null || true
                done
            fi
        done
    else
        print_success "所有显示器占用进程已成功终止"
    fi
    
    # 清理构建缓存和临时文件
    print_info "清理构建缓存..."
    local cleanup_dirs=(
        "$FRONTEND_DIR/build"
        "$FRONTEND_DIR/.cmake"
        "$FRONTEND_DIR/CMakeCache.txt"
        "$FRONTEND_DIR/third_party/lvgl/build"
        "/tmp/bamboo_*"
        "/var/tmp/bamboo_*"
    )
    
    for cleanup_path in "${cleanup_dirs[@]}"; do
        if [[ -e "$cleanup_path" ]]; then
            print_info "删除: $cleanup_path"
            rm -rf "$cleanup_path" 2>/dev/null || true
        fi
    done
    
    # 清理可能存在的配置冲突文件
    print_info "清理配置冲突文件..."
    local config_files=(
        "$FRONTEND_DIR/third_party/lvgl/lv_conf_internal.h.bak"
        "$FRONTEND_DIR/third_party/lvgl/examples"
        "$FRONTEND_DIR/third_party/lvgl/demos"
    )
    
    for config_file in "${config_files[@]}"; do
        if [[ -e "$config_file" ]]; then
            print_info "删除配置文件: $config_file"
            rm -rf "$config_file" 2>/dev/null || true
        fi
    done
    
    # 清理systemd服务（如果存在）
    if systemctl is-enabled bamboo-controller.service >/dev/null 2>&1; then
        print_info "停止并禁用旧版本systemd服务..."
        sudo systemctl stop bamboo-controller.service || true
        sudo systemctl disable bamboo-controller.service || true
    fi
    
    # 清理共享内存和IPC资源
    print_info "清理共享资源..."
    ipcs -m | grep $(whoami) | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true
    ipcs -s | grep $(whoami) | awk '{print $2}' | xargs -r ipcrm -s 2>/dev/null || true
    
    print_success "老版本清理完成"
}

# 初始化LVGL前端
initialize_frontend() {
    print_step "初始化LVGL前端..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
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

# 构建C++后端
build_backend() {
    print_step "构建C++后端项目..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    # 检查后端目录是否存在
    if [[ ! -d "$BACKEND_DIR" ]]; then
        print_error "后端目录不存在: $BACKEND_DIR"
        print_info "请确保已正确设置项目结构"
        exit 1
    fi
    
    cd "$BACKEND_DIR"
    
    # 创建构建目录
    if [[ ! -d "build" ]]; then
        mkdir build
    fi
    
    cd build
    
    # CMake配置
    print_info "配置后端CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    # 编译
    print_info "编译后端项目..."
    local num_cores=$(nproc)
    make -j$num_cores
    
    cd ../..
    print_success "C++后端构建完成"
}

# 构建LVGL前端
build_frontend() {
    print_step "构建LVGL前端项目 (Grid/Flex布局版本)..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    # 检查前端目录是否存在
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        print_error "前端目录不存在: $FRONTEND_DIR"
        print_info "请确保已正确设置项目结构"
        exit 1
    fi
    
    cd "$FRONTEND_DIR"
    
    # 验证新布局相关的源文件是否存在
    print_info "验证Grid/Flex布局源文件..."
    local required_files=(
        "src/gui/status_bar.cpp"
        "src/gui/video_view.cpp"
        "src/gui/control_panel.cpp"
        "src/gui/settings_page.cpp"
        "include/gui/status_bar.h"
        "include/gui/video_view.h"
        "include/gui/control_panel.h"
        "include/gui/settings_page.h"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "缺少必需的源文件: $file"
            exit 1
        fi
    done
    print_success "所有Grid/Flex布局源文件验证通过"
    
    # 清理旧的构建
    if [[ -d "build" ]]; then
        print_info "清理旧的构建目录..."
        rm -rf build
    fi
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # CMake配置
    print_info "配置前端CMake (Grid/Flex布局支持)..."
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_CXX_STANDARD=17"
        "-DCMAKE_C_STANDARD=11"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )
    
    # Jetson优化
    if [[ "$JETSON_DEVICE" == "true" ]]; then
        cmake_args+=(
            "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
            "-DCMAKE_C_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math -DLVGL_GRID_LAYOUT=1 -DLVGL_FLEX_LAYOUT=1"
            "-DCMAKE_CXX_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math -DLVGL_GRID_LAYOUT=1 -DLVGL_FLEX_LAYOUT=1"
        )
        
        # 查找Jetson OpenCV
        local opencv_dir="/usr/lib/aarch64-linux-gnu/cmake/opencv4"
        if [[ -d "$opencv_dir" ]]; then
            cmake_args+=("-DOpenCV_DIR=$opencv_dir")
        fi
    else
        # 非Jetson设备也启用Grid/Flex布局
        cmake_args+=(
            "-DCMAKE_C_FLAGS=-O3 -ffast-math -DLVGL_GRID_LAYOUT=1 -DLVGL_FLEX_LAYOUT=1"
            "-DCMAKE_CXX_FLAGS=-O3 -ffast-math -DLVGL_GRID_LAYOUT=1 -DLVGL_FLEX_LAYOUT=1"
        )
    fi
    
    # 运行CMake
    print_info "CMake配置参数: ${cmake_args[*]}"
    cmake "${cmake_args[@]}" ..
    
    # 检查CMake是否成功
    if [[ $? -ne 0 ]]; then
        print_error "CMake配置失败"
        exit 1
    fi
    
    # 编译前显示编译信息
    print_info "开始编译Grid/Flex布局LVGL前端..."
    local num_cores=$(nproc)
    print_info "使用 $num_cores 个CPU核心进行编译"
    print_info "启用LVGL Grid和Flex布局支持"
    
    # 编译
    make -j$num_cores VERBOSE=1
    
    # 检查编译是否成功
    if [[ $? -ne 0 ]]; then
        print_error "前端编译失败"
        print_info "请检查编译错误信息"
        exit 1
    fi
    
    # 验证可执行文件是否生成
    if [[ ! -f "$BINARY_NAME" ]]; then
        print_error "可执行文件未生成: $BINARY_NAME"
        exit 1
    fi
    
    # 显示可执行文件信息
    print_info "可执行文件大小: $(du -h $BINARY_NAME | cut -f1)"
    print_info "可执行文件路径: $(pwd)/$BINARY_NAME"
    
    cd ../..
    print_success "LVGL前端构建完成 (Grid/Flex布局)"
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

# 配置framebuffer和显示器
configure_framebuffer() {
    print_step "配置framebuffer和显示器..."
    
    # 检查framebuffer设备
    if [[ ! -e "/dev/fb0" ]]; then
        print_error "找不到framebuffer设备 /dev/fb0"
        exit 1
    fi
    
    # 显示framebuffer信息
    print_info "Framebuffer设备信息:"
    if command -v fbset >/dev/null 2>&1; then
        fbset -fb /dev/fb0 | head -10
    else
        ls -la /dev/fb* 2>/dev/null || true
        cat /sys/class/graphics/fb0/virtual_size 2>/dev/null || echo "无法获取framebuffer尺寸"
    fi
    
    # 设置framebuffer环境变量
    export FRAMEBUFFER=/dev/fb0
    export DISPLAY=""
    export WAYLAND_DISPLAY=""
    export XDG_SESSION_TYPE=""
    export QT_QPA_PLATFORM=""
    
    # 配置framebuffer权限
    print_info "配置framebuffer权限..."
    chmod 666 /dev/fb* 2>/dev/null || true
    chmod 666 /dev/input/event* 2>/dev/null || true
    chmod 666 /dev/tty* 2>/dev/null || true
    
    # 停止所有可能干扰的显示服务
    print_info "停止显示服务..."
    systemctl stop gdm3 2>/dev/null || true
    systemctl stop lightdm 2>/dev/null || true
    systemctl stop sddm 2>/dev/null || true
    systemctl stop xdm 2>/dev/null || true
    
    # 禁用console光标和文本显示
    print_info "配置console显示..."
    echo 0 > /sys/class/graphics/fbcon/cursor_blink 2>/dev/null || true
    
    # 解绑console from framebuffer
    echo 0 > /sys/class/vtconsole/vtcon0/bind 2>/dev/null || true
    echo 0 > /sys/class/vtconsole/vtcon1/bind 2>/dev/null || true
    
    # 清空framebuffer
    print_info "清空framebuffer..."
    if command -v dd >/dev/null 2>&1; then
        dd if=/dev/zero of=/dev/fb0 bs=1024 count=8192 2>/dev/null || true
    fi
    
    # 检查并释放其他可能占用framebuffer的进程
    print_info "检查framebuffer占用情况..."
    lsof /dev/fb0 2>/dev/null || true
    
    print_success "Framebuffer配置完成"
}

# 启动C++后端
start_backend() {
    print_step "启动C++后端..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    # 检查后端可执行文件
    local backend_binary_path="$BACKEND_DIR/build/$BACKEND_BINARY"
    if [[ ! -f "$backend_binary_path" ]]; then
        print_error "找不到后端可执行文件: $backend_binary_path"
        exit 1
    fi
    
    # 启动后端进程
    cd "$BACKEND_DIR/build"
    print_info "启动后端进程..."
    nohup "./$BACKEND_BINARY" > ../../backend.log 2>&1 &
    local backend_pid=$!
    cd ../..
    
    # 等待后端启动
    sleep 3
    
    if kill -0 $backend_pid 2>/dev/null; then
        print_success "C++后端启动成功 (PID: $backend_pid)"
        echo $backend_pid > backend.pid
    else
        print_error "C++后端启动失败，检查backend.log"
        exit 1
    fi
}

# 启动LVGL前端
start_frontend() {
    print_step "启动LVGL前端应用..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    cd "$FRONTEND_DIR"
    
    # 检查可执行文件
    local binary_path="build/$BINARY_NAME"
    if [[ ! -f "$binary_path" ]]; then
        print_error "找不到前端可执行文件: $binary_path"
        exit 1
    fi
    
    # 配置framebuffer
    configure_framebuffer
    
    # 设置环境变量
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    
    # 配置文件路径
    local config_file="resources/config/default_config.json"
    if [[ ! -f "$config_file" ]]; then
        print_warn "配置文件不存在: $config_file"
        config_file=""
    fi
    
    print_success "正在启动前端应用程序..."
    print_info "前端可执行文件: $binary_path"
    print_info "后端日志: backend.log"
    print_info "配置文件: ${config_file:-"使用默认配置"}"
    print_info "按 Ctrl+C 退出应用程序"
    echo
    
    # 启动前端应用程序
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
    
    # 清理老版本（在初始化和构建之前）
    cleanup_old_versions
    
    initialize_frontend
    
    # 根据参数决定是否构建
    if [[ "$skip_build" != "true" ]]; then
        if [[ "$clean_build" == "true" ]]; then
            print_info "清理模式：将重新构建项目"
        fi
        build_backend
        build_frontend
        
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
        start_backend
        start_frontend
    else
        print_success "构建完成！"
        print_info "可执行文件位置: $FRONTEND_DIR/build/$BINARY_NAME"
    fi
}

# 运行主函数
main "$@"