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
PROJECT_NAME="智能切竹机控制系统 - C++后端(libmodbus+UNIX Socket)+LVGL前端"
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
    
    # 检查libmodbus
    if ! pkg-config --exists libmodbus; then
        print_error "未找到libmodbus库"
        print_info "请安装libmodbus: sudo apt install -y libmodbus-dev"
        exit 1
    else
        print_success "检测到libmodbus库"
    fi
    
    # 检查CUDA（仅在Jetson设备上）
    if [[ "$JETSON_DEVICE" == "true" ]]; then
        if ! command -v nvcc &> /dev/null; then
            print_warn "未找到CUDA编译器，GPU加速功能可能不可用"
        else
            print_success "检测到CUDA支持"
        fi
    fi
    
    # 检查设备文件 - 优先检查 event2
    print_info "检查触摸设备..."
    local touch_devices=("/dev/input/event2" "/dev/input/event1" "/dev/input/event0")
    local touch_found=false
    for device in "${touch_devices[@]}"; do
        if [[ -e "$device" ]]; then
            print_success "检测到触摸设备: $device"
            touch_found=true
            break
        fi
    done
    
    if [[ "$touch_found" != "true" ]]; then
        print_warn "未检测到优先触摸设备"
    fi
    
    # 检查其他设备文件
    local other_devices=("/dev/fb0" "/dev/video0")
    for device in "${other_devices[@]}"; do
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
    
    # 清理GStreamer流相关资源
    print_info "清理GStreamer流资源..."
    
    # 停止可能运行的GStreamer进程
    pkill -f "gst-launch" 2>/dev/null || true
    pkill -f "gstreamer" 2>/dev/null || true
    
    # 清理可能残留的流临时文件
    rm -f /tmp/bamboo_stream_* 2>/dev/null || true
    
    # 清理UNIX Domain Socket文件
    print_info "清理UNIX Domain Socket文件..."
    rm -f /tmp/bamboo_socket 2>/dev/null || true
    
    # 释放可能占用的网络端口
    print_info "检查并释放流端口..."
    local stream_ports=(5000 5001 5002 5003)
    for port in "${stream_ports[@]}"; do
        local pid=$(lsof -ti:$port 2>/dev/null || echo "")
        if [[ -n "$pid" ]]; then
            print_info "释放端口 $port (PID: $pid)"
            kill -9 $pid 2>/dev/null || true
        fi
    done
    
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
    
    # 设置设备权限 - 包含所有触摸设备
    local devices=("/dev/fb0" "/dev/video0" "/dev/nvidia0")
    # 优先设置触摸设备权限
    local touch_devices=("/dev/input/event2" "/dev/input/event1" "/dev/input/event0" "/dev/input/event3" "/dev/input/event4")
    
    print_info "设置触摸设备权限..."
    for device in "${touch_devices[@]}"; do
        if [[ -e "$device" ]]; then
            sudo chmod 666 "$device" 2>/dev/null || true
            print_info "设置触摸设备权限: $device"
        fi
    done
    
    # 设置其他设备权限
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
        print_warn "找不到framebuffer设备 /dev/fb0，跳过framebuffer配置"
        return
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
    print_step "启动C++后端（Modbus+UNIX Domain Socket模式）..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    # 检查后端可执行文件
    local backend_binary_path="$BACKEND_DIR/build/$BACKEND_BINARY"
    if [[ ! -f "$backend_binary_path" ]]; then
        print_error "找不到后端可执行文件: $backend_binary_path"
        exit 1
    fi
    
    # 创建共享内存key文件（如果不存在）
    local shared_memory_key="/tmp/bamboo_camera_shm"
    if [[ ! -f "$shared_memory_key" ]]; then
        print_info "创建共享内存key文件: $shared_memory_key"
        touch "$shared_memory_key"
        chmod 666 "$shared_memory_key"
    fi
    
    # 清理UNIX Domain Socket文件（如果存在）
    local unix_socket_path="/tmp/bamboo_socket"
    if [[ -e "$unix_socket_path" ]]; then
        print_info "清理旧的UNIX Domain Socket: $unix_socket_path"
        rm -f "$unix_socket_path"
    fi
    
    # 设置环境变量
    export BAMBOO_SHARED_MEMORY_KEY="$shared_memory_key"
    export BAMBOO_ENABLE_SHARED_MEMORY="true"
    export BAMBOO_UNIX_SOCKET_PATH="$unix_socket_path"
    export BAMBOO_PLC_IP="${BAMBOO_PLC_IP:-192.168.1.100}"
    export BAMBOO_PLC_PORT="${BAMBOO_PLC_PORT:-502}"
    
    # 启动后端进程
    cd "$BACKEND_DIR/build"
    print_info "启动后端进程（Modbus+UNIX Socket模式）..."
    print_info "共享内存key: $shared_memory_key"
    print_info "UNIX Socket路径: $unix_socket_path"
    print_info "PLC地址: ${BAMBOO_PLC_IP}:${BAMBOO_PLC_PORT}"
    print_info "支持无PLC连接启动（健壮性模式）"
    nohup "./$BACKEND_BINARY" > ../../backend.log 2>&1 &
    local backend_pid=$!
    cd ../..
    
    # 等待后端启动并检查状态
    print_info "检查后端进程启动状态..."
    sleep 3  # 给后端3秒时间启动
    
    # 检查进程是否还在运行
    if ! kill -0 $backend_pid 2>/dev/null; then
        print_warn "后端进程可能启动失败，检查日志"
        if [[ -f "backend.log" ]]; then
            cat backend.log | tail -10
        fi
        print_info "前端将继续启动，可能显示离线状态"
    else
        print_success "后端进程启动成功"
    fi
    
    # 检查UNIX Domain Socket是否创建
    local socket_check_count=0
    while [[ $socket_check_count -lt 5 ]] && [[ ! -e "$unix_socket_path" ]]; do
        sleep 1
        ((socket_check_count++))
    done
    
    if [[ -e "$unix_socket_path" ]]; then
        print_success "UNIX Domain Socket已创建: $unix_socket_path"
    else
        print_warn "UNIX Domain Socket尚未创建，前端将重试连接"
    fi
    
    # 检查是否有共享内存创建（视频流可选）
    if command -v ftok >/dev/null 2>&1; then
        local shm_key=$(ftok "$shared_memory_key" 66 2>/dev/null || echo "")
        if [[ -n "$shm_key" ]] && ipcs -m | grep -q "$shm_key" 2>/dev/null; then
            print_success "共享内存已可用 (key: $shm_key)"
        else
            print_info "共享内存尚未创建，视频流将异步连接"
        fi
    else
        print_info "ftok命令不可用，跳过共享内存检查"
    fi
    
    if kill -0 $backend_pid 2>/dev/null; then
        print_success "C++后端启动成功 (PID: $backend_pid)"
        echo $backend_pid > backend.pid
    else
        print_warn "C++后端可能启动失败，但前端将继续启动"
        if [[ -f "backend.log" ]]; then
            print_info "后端日志信息："
            cat backend.log | tail -10
        fi
    fi
}

# 启动LVGL前端
start_frontend() {
    local debug_mode_param="$1"
    print_step "启动LVGL前端应用（Modbus+UNIX Socket+GStreamer流模式）..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    cd "$FRONTEND_DIR"
    
    # 检查可执行文件
    local binary_path="build/$BINARY_NAME"
    if [[ ! -f "$binary_path" ]]; then
        print_error "找不到前端可执行文件: $binary_path"
        exit 1
    fi
    
    # 设置GStreamer流环境变量
    local stream_host="${BAMBOO_STREAM_HOST:-127.0.0.1}"
    local stream_port="${BAMBOO_STREAM_PORT:-5000}"
    local stream_format="${BAMBOO_STREAM_FORMAT:-H264}"
    
    print_info "设置GStreamer流接收模式..."
    print_info "流地址: udp://$stream_host:$stream_port"
    print_info "流格式: $stream_format"
    
    # 配置framebuffer
    configure_framebuffer
    
    # 设置环境变量
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export BAMBOO_STREAM_URL="udp://$stream_host:$stream_port"
    export BAMBOO_STREAM_FORMAT="$stream_format"
    export BAMBOO_FRONTEND_MODE="gstreamer_stream"
    
    # 设置UNIX Domain Socket通信
    export BAMBOO_UNIX_SOCKET_PATH="/tmp/bamboo_socket"
    export BAMBOO_ENABLE_UNIX_SOCKET="true"
    
    # 检查GStreamer插件可用性
    print_info "检查GStreamer插件..."
    if command -v gst-inspect-1.0 >/dev/null 2>&1; then
        # 检查必要的GStreamer插件
        local required_plugins=("udpsrc" "rtph264depay" "avdec_h264" "videoconvert" "appsink")
        local missing_plugins=()
        
        for plugin in "${required_plugins[@]}"; do
            if ! gst-inspect-1.0 "$plugin" >/dev/null 2>&1; then
                missing_plugins+=("$plugin")
            fi
        done
        
        if [[ ${#missing_plugins[@]} -eq 0 ]]; then
            print_success "所有必要的GStreamer插件已安装"
        else
            print_warn "缺少GStreamer插件: ${missing_plugins[*]}"
            print_info "请安装: sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav"
        fi
    else
        print_warn "未找到gst-inspect-1.0工具，无法验证插件"
    fi
    
    # 配置文件路径
    local config_file="resources/config/default_config.json"
    if [[ ! -f "$config_file" ]]; then
        print_warn "配置文件不存在: $config_file"
        config_file=""
    fi
    
    # 检查触摸配置文件
    local touch_config_file="resources/config/touch_config.json"
    if [[ -f "$touch_config_file" ]]; then
        print_info "找到触摸配置文件: $touch_config_file"
    fi
    
    print_success "正在启动前端应用程序（Modbus+UNIX Socket+GStreamer流模式）..."
    print_info "前端可执行文件: $binary_path"
    print_info "后端日志: backend.log"
    print_info "流URL: $BAMBOO_STREAM_URL"
    print_info "流格式: $BAMBOO_STREAM_FORMAT"
    print_info "UNIX Socket: $BAMBOO_UNIX_SOCKET_PATH"
    print_info "配置文件: ${config_file:-"使用默认配置"}"
    if [[ "$debug_mode_param" == "true" ]]; then
        print_info "调试模式: 已启用 (详细触摸信息将显示)"
    fi
    print_info "架构模式: 后端libmodbus+UNIX Socket + GStreamer视频流 + 前端实时通信"
    print_info "通信特性: PLC实时数据 + 按键控制 + 状态同步 + 健壮性连接"
    print_info "UI渲染: 立即开始，支持离线模式和实时数据切换"
    print_info "按 Ctrl+C 退出应用程序"
    echo
    
    # 启动前端应用程序（非阻塞模式，支持错误处理）
    if [[ -n "$config_file" ]]; then
        if [[ "$debug_mode_param" == "true" ]]; then
            "./$binary_path" -c "$config_file" -f --debug
        else
            "./$binary_path" -c "$config_file" -f
        fi
    else
        if [[ "$debug_mode_param" == "true" ]]; then
            "./$binary_path" -f --debug
        else
            "./$binary_path" -f
        fi
    fi
    
    # 检查前端启动结果
    local frontend_exit_code=$?
    if [[ $frontend_exit_code -eq 0 ]]; then
        print_success "前端应用程序正常退出"
    else
        print_error "前端应用程序异常退出，退出码: $frontend_exit_code"
        print_info "请检查错误信息，可能的原因："
        print_info "1. 显示设备(/dev/fb0)访问权限"
        print_info "2. 触摸设备(/dev/input/event*)权限"
        print_info "3. 后端通信连接问题"
        print_info "4. LVGL依赖库问题"
    fi
}

# 测试GStreamer流架构
test_gstreamer_architecture() {
    print_step "测试GStreamer流架构..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    print_info "验证GStreamer流架构："
    print_info "1. 检查配置文件流设置"
    print_info "2. 验证GStreamer接收器实现"
    print_info "3. 测试前端异步流初始化"
    print_info "4. 检查后端流输出配置"
    echo
    
    # 检查前端配置文件
    local config_file="$FRONTEND_DIR/resources/config/default_config.json"
    if [[ -f "$config_file" ]]; then
        print_info "检查前端配置文件..."
        if grep -q '"stream_url"' "$config_file"; then
            print_success "✓ 流URL配置已添加"
        else
            print_warn "流URL配置可能缺失"
        fi
        
        if grep -q '"stream_format"' "$config_file"; then
            print_success "✓ 流格式配置已添加"
        else
            print_warn "流格式配置可能缺失"
        fi
        
        if grep -q '"display_width"' "$config_file" && grep -q '"display_height"' "$config_file"; then
            print_success "✓ 显示分辨率配置已保留"
        else
            print_warn "显示分辨率配置可能缺失"
        fi
    else
        print_error "找不到前端配置文件: $config_file"
    fi
    
    # 检查GStreamer接收器
    local receiver_header="$FRONTEND_DIR/include/camera/gstreamer_receiver.h"
    local receiver_impl="$FRONTEND_DIR/src/camera/gstreamer_receiver.cpp"
    
    if [[ -f "$receiver_header" ]]; then
        print_info "检查GStreamer接收器头文件..."
        if grep -q "gstreamer_receiver_t" "$receiver_header"; then
            print_success "✓ GStreamer接收器结构体已定义"
        else
            print_warn "GStreamer接收器结构体可能缺失"
        fi
        
        if grep -q "gstreamer_receiver_create" "$receiver_header"; then
            print_success "✓ GStreamer接收器API已定义"
        else
            print_warn "GStreamer接收器API可能缺失"
        fi
    else
        print_error "找不到GStreamer接收器头文件: $receiver_header"
    fi
    
    if [[ -f "$receiver_impl" ]]; then
        print_info "检查GStreamer接收器实现..."
        if grep -q "udpsrc" "$receiver_impl" && grep -q "rtph264depay" "$receiver_impl"; then
            print_success "✓ H.264 UDP流解码实现已添加"
        else
            print_warn "H.264 UDP流解码可能未实现"
        fi
        
        if grep -q "souphttpsrc" "$receiver_impl" && grep -q "jpegdec" "$receiver_impl"; then
            print_success "✓ MJPEG HTTP流解码实现已添加"
        else
            print_warn "MJPEG HTTP流解码可能未实现"
        fi
    else
        print_error "找不到GStreamer接收器实现: $receiver_impl"
    fi
    
    # 检查摄像头管理器更新
    local camera_mgr_file="$FRONTEND_DIR/src/camera/camera_manager.cpp"
    if [[ -f "$camera_mgr_file" ]]; then
        print_info "检查摄像头管理器更新..."
        if grep -q "gstreamer_receiver" "$camera_mgr_file"; then
            print_success "✓ 摄像头管理器已集成GStreamer接收器"
        else
            print_warn "摄像头管理器可能未集成GStreamer接收器"
        fi
        
        if grep -q "gstreamer_read_thread" "$camera_mgr_file"; then
            print_success "✓ GStreamer流读取线程已实现"
        else
            print_warn "GStreamer流读取线程可能未实现"
        fi
    else
        print_error "找不到摄像头管理器文件: $camera_mgr_file"
    fi
    
    # 检查后端流输出
    local backend_camera_mgr="$BACKEND_DIR/src/vision/camera_manager.cpp"
    if [[ -f "$backend_camera_mgr" ]]; then
        print_info "检查后端流输出..."
        if grep -q "initializeVideoStream" "$backend_camera_mgr"; then
            print_success "✓ 后端视频流初始化已实现"
        else
            print_warn "后端视频流初始化可能未实现"
        fi
        
        if grep -q "pushFrameToStream" "$backend_camera_mgr"; then
            print_success "✓ 后端帧推送到流已实现"
        else
            print_warn "后端帧推送到流可能未实现"
        fi
        
        if grep -q "buildStreamPipeline" "$backend_camera_mgr"; then
            print_success "✓ 后端流管道构建已实现"
        else
            print_warn "后端流管道构建可能未实现"
        fi
    else
        print_error "找不到后端摄像头管理器文件: $backend_camera_mgr"
    fi
    
    print_success "GStreamer流架构验证完成"
}

# 测试GStreamer流架构
test_gstreamer_stream_architecture() {
    print_step "测试GStreamer流架构..."
    
    # 确保在项目根目录
    cd "$SCRIPT_DIR"
    
    # 先运行架构验证
    test_gstreamer_architecture
    
    # 检查GStreamer环境
    print_info "检查GStreamer运行环境..."
    if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
        print_error "未找到gst-launch-1.0命令"
        print_info "请安装: sudo apt install gstreamer1.0-tools"
        exit 1
    fi
    
    # 测试基本的GStreamer流管道
    print_info "测试GStreamer流管道..."
    local stream_port=5000
    
    # 检查端口可用性
    if lsof -i:$stream_port >/dev/null 2>&1; then
        print_warn "测试端口 $stream_port 被占用，跳过流测试"
        return
    fi
    
    # 测试UDP接收器
    print_info "测试UDP接收器管道..."
    local test_pipeline="videotestsrc num-buffers=10 ! video/x-raw,width=640,height=480,framerate=10/1 ! videoconvert ! x264enc tune=zerolatency ! video/x-h264,stream-format=byte-stream ! rtph264pay ! udpsink host=127.0.0.1 port=$stream_port"
    
    timeout 5s gst-launch-1.0 $test_pipeline >/dev/null 2>&1 &
    local sender_pid=$!
    
    sleep 1
    
    # 测试接收
    local receive_pipeline="udpsrc port=$stream_port ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! fakesink"
    
    timeout 3s gst-launch-1.0 $receive_pipeline >/dev/null 2>&1
    local receiver_result=$?
    
    # 清理测试进程
    kill $sender_pid 2>/dev/null || true
    wait $sender_pid 2>/dev/null || true
    
    if [[ $receiver_result -eq 0 ]]; then
        print_success "GStreamer流管道测试通过"
    else
        print_warn "GStreamer流管道测试失败，但系统可能仍然工作"
        print_info "这可能是由于测试环境限制造成的"
    fi
    
    print_success "GStreamer流架构测试完成"
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
    echo "  -t, --test          测试GStreamer流架构"
    echo "  --debug             调试模式构建"
    echo
    echo "示例:"
    echo "  $0                  # 完整的构建和启动流程"
    echo "  $0 -b               # 仅构建，不启动"
    echo "  $0 -s               # 跳过构建，直接启动"
    echo "  $0 -c               # 清理后重新构建和启动"
    echo "  $0 -t               # 测试GStreamer流架构"
}

# 主函数
main() {
    local build_only=false
    local skip_build=false
    local clean_build=false
    local install_system=false
    local debug_mode=false
    local test_mode=false
    
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
            -t|--test)
                test_mode=true
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
    
    # 根据参数决定是否测试或启动
    if [[ "$test_mode" == "true" ]]; then
        test_gstreamer_stream_architecture
        print_success "测试完成！"
        exit 0
    elif [[ "$build_only" != "true" ]]; then
        print_info "开始启动应用程序..."
        setup_permissions
        start_backend
        print_info "后端启动完成，准备启动前端..."
        start_frontend "$debug_mode"
    else
        print_success "构建完成！"
        print_info "可执行文件位置: $FRONTEND_DIR/build/$BINARY_NAME"
        if [[ "$debug_mode" == "true" ]]; then
            print_info "调试模式已启用，运行时使用 --debug 参数"
        fi
    fi
}

# 运行主函数
main "$@"