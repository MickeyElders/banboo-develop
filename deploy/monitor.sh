#!/bin/bash

# 智能切竹机系统监控脚本
# 监控服务状态、资源使用情况和通信状态

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 服务名称
BACKEND_SERVICE="bamboo-backend.service"
FRONTEND_SERVICE="bamboo-frontend.service"
SOCKET_PATH="/tmp/bamboo_socket"

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

# 检查服务状态
check_service_status() {
    local service=$1
    local status
    
    if systemctl is-active --quiet "$service"; then
        status="${GREEN}运行中${NC}"
    else
        status="${RED}已停止${NC}"
    fi
    
    echo -e "  $service: $status"
}

# 检查进程资源使用
check_resource_usage() {
    local service=$1
    local pid
    
    pid=$(systemctl show --property MainPID --value "$service" 2>/dev/null)
    
    if [ "$pid" != "0" ] && [ -n "$pid" ]; then
        local cpu=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null || echo "N/A")
        local mem=$(ps -p "$pid" -o %mem --no-headers 2>/dev/null || echo "N/A")
        local rss=$(ps -p "$pid" -o rss --no-headers 2>/dev/null || echo "N/A")
        
        echo -e "    PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | RSS: ${rss}KB"
    else
        echo -e "    ${RED}进程未运行${NC}"
    fi
}

# 检查Socket连接
check_socket_connection() {
    if [ -S "$SOCKET_PATH" ]; then
        log_success "UNIX Domain Socket存在: $SOCKET_PATH"
        
        # 检查Socket权限
        local perms=$(ls -l "$SOCKET_PATH" | awk '{print $1}')
        echo "  权限: $perms"
        
        # 尝试测试连接
        if timeout 1 nc -U "$SOCKET_PATH" </dev/null 2>/dev/null; then
            log_success "Socket连接测试成功"
        else
            log_warning "Socket连接测试失败"
        fi
    else
        log_error "UNIX Domain Socket不存在: $SOCKET_PATH"
    fi
}

# 检查PLC连接
check_plc_connection() {
    local plc_ip="192.168.1.100"
    local plc_port="502"
    
    log_info "测试PLC连接 ($plc_ip:$plc_port)..."
    
    if timeout 3 nc -z "$plc_ip" "$plc_port" 2>/dev/null; then
        log_success "PLC连接正常"
    else
        log_warning "PLC连接失败或超时"
    fi
}

# 检查系统负载
check_system_load() {
    local load=$(uptime | awk -F'load average:' '{print $2}')
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    echo "系统负载: $load"
    echo "内存使用: ${mem_usage}%"
    echo "磁盘使用: ${disk_usage}%"
    
    # 警告阈值
    if (( $(echo "$mem_usage > 80" | bc -l) )); then
        log_warning "内存使用率过高: ${mem_usage}%"
    fi
    
    if [ "$disk_usage" -gt 90 ]; then
        log_warning "磁盘使用率过高: ${disk_usage}%"
    fi
}

# 显示最近日志
show_recent_logs() {
    local service=$1
    local lines=${2:-5}
    
    echo "=== $service 最近日志 ==="
    journalctl -u "$service" --no-pager -n "$lines" 2>/dev/null || echo "无法获取日志"
    echo ""
}

# 主监控函数
main_monitor() {
    clear
    echo "========================================"
    echo "智能切竹机系统监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    log_info "服务状态检查:"
    check_service_status "$BACKEND_SERVICE"
    check_resource_usage "$BACKEND_SERVICE"
    echo ""
    check_service_status "$FRONTEND_SERVICE"
    check_resource_usage "$FRONTEND_SERVICE"
    echo ""
    
    log_info "通信状态检查:"
    check_socket_connection
    echo ""
    check_plc_connection
    echo ""
    
    log_info "系统资源:"
    check_system_load
    echo ""
    
    log_info "最近日志:"
    show_recent_logs "$BACKEND_SERVICE" 3
    show_recent_logs "$FRONTEND_SERVICE" 3
}

# 实时监控模式
real_time_monitor() {
    while true; do
        main_monitor
        echo "按Ctrl+C退出实时监控..."
        sleep 5
    done
}

# 脚本参数处理
case "${1:-}" in
    "realtime"|"rt")
        log_info "启动实时监控模式..."
        trap 'echo ""; log_info "退出监控"; exit 0' INT
        real_time_monitor
        ;;
    "once"|"")
        main_monitor
        ;;
    "logs")
        log_info "显示服务日志:"
        show_recent_logs "$BACKEND_SERVICE" 10
        show_recent_logs "$FRONTEND_SERVICE" 10
        ;;
    "help"|"-h")
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  once     - 执行一次监控检查 (默认)"
        echo "  realtime - 实时监控模式"
        echo "  logs     - 显示最近日志"
        echo "  help     - 显示此帮助"
        ;;
    *)
        log_error "未知选项: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac