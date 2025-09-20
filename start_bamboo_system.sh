#!/bin/bash

# 智能切竹机系统统一启动脚本
# 启动C++后端和LVGL前端

echo "=================================="
echo "智能切竹机系统 v2.0 启动脚本"
echo "C++后端 + LVGL前端"
echo "=================================="

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

# 检查运行环境
check_environment() {
    log_info "检查运行环境..."
    
    # 检查是否在项目根目录
    if [ ! -d "cpp_backend" ] || [ ! -d "lvgl_frontend" ]; then
        log_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 检查framebuffer设备
    if [ ! -e "/dev/fb0" ]; then
        log_error "找不到framebuffer设备 /dev/fb0"
        exit 1
    fi
    
    # 检查摄像头设备
    if [ ! -e "/dev/video0" ]; then
        log_warning "找不到摄像头设备 /dev/video0"
    fi
    
    # 检查CUDA设备（可选）
    if [ ! -e "/dev/nvidia0" ]; then
        log_warning "未检测到NVIDIA GPU设备"
    fi
    
    log_success "环境检查完成"
}

# 编译C++后端
build_backend() {
    log_info "编译C++后端..."
    
    cd cpp_backend
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    if [ $? -ne 0 ]; then
        log_error "CMake配置失败"
        exit 1
    fi
    
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        log_error "后端编译失败"
        exit 1
    fi
    
    cd ../..
    log_success "C++后端编译完成"
}

# 编译LVGL前端
build_frontend() {
    log_info "编译LVGL前端..."
    
    cd lvgl_frontend
    make clean
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        log_error "前端编译失败"
        exit 1
    fi
    
    cd ..
    log_success "LVGL前端编译完成"
}

# 停止现有进程
stop_existing_processes() {
    log_info "停止现有进程..."
    
    # 停止前端进程
    pkill -f "bamboo_controller_lvgl" > /dev/null 2>&1
    pkill -f "lvgl_frontend" > /dev/null 2>&1
    
    # 停止后端进程
    pkill -f "bamboo_cut_backend" > /dev/null 2>&1
    
    # 停止framebuffer占用进程
    pkill -f "weston" > /dev/null 2>&1
    pkill -f "X" > /dev/null 2>&1
    
    sleep 2
    log_success "现有进程已停止"
}

# 启动C++后端
start_backend() {
    log_info "启动C++后端..."
    
    if [ ! -f "cpp_backend/build/bamboo_cut_backend" ]; then
        log_error "后端可执行文件不存在，请先编译"
        exit 1
    fi
    
    cd cpp_backend/build
    nohup ./bamboo_cut_backend > ../../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ../..
    
    # 等待后端启动
    sleep 3
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        log_success "C++后端启动成功 (PID: $BACKEND_PID)"
        echo $BACKEND_PID > backend.pid
    else
        log_error "C++后端启动失败"
        exit 1
    fi
}

# 启动LVGL前端
start_frontend() {
    log_info "启动LVGL前端..."
    
    if [ ! -f "lvgl_frontend/build/bamboo_controller_lvgl" ]; then
        log_error "前端可执行文件不存在，请先编译"
        exit 1
    fi
    
    # 设置framebuffer权限
    sudo chmod 666 /dev/fb0 2>/dev/null || log_warning "无法设置framebuffer权限"
    
    cd lvgl_frontend
    export DISPLAY=""  # 禁用X11
    ./build/bamboo_controller_lvgl
    cd ..
}

# 清理函数
cleanup() {
    log_info "正在清理进程..."
    
    # 停止后端进程
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            log_success "后端进程已停止"
        fi
        rm -f backend.pid
    fi
    
    # 停止其他相关进程
    pkill -f "bamboo_cut_backend" > /dev/null 2>&1
    pkill -f "bamboo_controller_lvgl" > /dev/null 2>&1
    
    log_success "清理完成"
    exit 0
}

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -b, --build-only    仅编译，不运行"
    echo "  -s, --stop         停止所有进程"
    echo "  -h, --help         显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                 完整启动系统"
    echo "  $0 --build-only    仅编译项目"
    echo "  $0 --stop          停止系统"
}

# 主函数
main() {
    # 解析命令行参数
    case "$1" in
        -b|--build-only)
            check_environment
            build_backend
            build_frontend
            log_success "编译完成，使用 '$0' 启动系统"
            exit 0
            ;;
        -s|--stop)
            stop_existing_processes
            cleanup
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        "")
            # 默认：完整启动
            ;;
        *)
            log_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
    
    # 设置信号处理
    trap cleanup SIGINT SIGTERM
    
    # 完整启动流程
    check_environment
    stop_existing_processes
    build_backend
    build_frontend
    start_backend
    
    log_success "系统启动完成!"
    log_info "后端日志: backend.log"
    log_info "按 Ctrl+C 停止系统"
    
    # 启动前端（阻塞运行）
    start_frontend
}

# 检查root权限（部分操作需要）
if [ "$EUID" -eq 0 ]; then
    log_warning "检测到root权限，建议使用普通用户运行"
fi

# 运行主函数
main "$@"