#!/bin/bash

# 智能切竹机控制系统部署脚本
# 版本: 2.0.0
# 支持: libmodbus + UNIX Domain Socket + systemd

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检查root权限
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        exit 1
    fi
}

# 检查系统依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    local deps=("cmake" "gcc" "g++" "make" "pkg-config")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    # 检查libmodbus
    if ! pkg-config --exists libmodbus; then
        missing_deps+=("libmodbus-dev")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少以下依赖: ${missing_deps[*]}"
        log_info "请运行以下命令安装依赖:"
        echo "sudo apt update"
        echo "sudo apt install -y cmake gcc g++ make pkg-config libmodbus-dev"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 构建项目
build_project() {
    log_info "开始构建项目..."
    
    if ! make clean; then
        log_error "清理失败"
        exit 1
    fi
    
    if ! make all; then
        log_error "构建失败"
        exit 1
    fi
    
    log_success "项目构建完成"
}

# 安装项目
install_project() {
    log_info "安装项目到系统..."
    
    if ! make install; then
        log_error "安装失败"
        exit 1
    fi
    
    log_success "项目安装完成"
}

# 安装systemd服务
install_services() {
    log_info "安装systemd服务..."
    
    # 复制服务文件
    cp deploy/systemd/bamboo-backend.service /etc/systemd/system/
    cp deploy/systemd/bamboo-frontend.service /etc/systemd/system/
    
    # 重新加载systemd
    systemctl daemon-reload
    
    # 启用服务
    systemctl enable bamboo-backend.service
    systemctl enable bamboo-frontend.service
    
    log_success "systemd服务安装完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."
    
    # 启动后端服务
    if systemctl start bamboo-backend.service; then
        log_success "后端服务启动成功"
    else
        log_error "后端服务启动失败"
        exit 1
    fi
    
    # 等待2秒确保后端启动完成
    sleep 2
    
    # 启动前端服务
    if systemctl start bamboo-frontend.service; then
        log_success "前端服务启动成功"
    else
        log_warning "前端服务启动失败，请检查图形环境"
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    echo "=========================="
    echo "后端服务状态:"
    systemctl status bamboo-backend.service --no-pager || true
    echo ""
    echo "前端服务状态:"
    systemctl status bamboo-frontend.service --no-pager || true
    echo "=========================="
}

# 主函数
main() {
    log_info "智能切竹机控制系统部署开始..."
    
    check_root
    check_dependencies
    build_project
    install_project
    install_services
    start_services
    show_status
    
    log_success "部署完成！"
    echo ""
    log_info "服务管理命令:"
    echo "  启动: make start"
    echo "  停止: make stop"
    echo "  重启: make restart"
    echo "  状态: make status"
    echo "  日志: make logs"
    echo ""
    log_info "或使用systemctl命令:"
    echo "  systemctl status bamboo-backend"
    echo "  systemctl status bamboo-frontend"
    echo "  journalctl -u bamboo-backend -f"
}

# 运行主函数
main "$@"