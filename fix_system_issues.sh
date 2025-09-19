#!/bin/bash

# 智能切竹机系统问题修复脚本
# 修复systemd权限和Tegra摄像头问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 修复systemd服务文件权限
fix_systemd_permissions() {
    log_info "🔧 修复systemd服务文件权限问题..."
    
    # 需要修复权限的服务文件
    PROBLEMATIC_SERVICES=(
        "/etc/systemd/system/nv-l4t-usb-device-mode.service"
        "/etc/systemd/system/nv-l4t-bootloader-config.service"  
        "/etc/systemd/system/l4t-rootfs-validation-config.service"
    )
    
    for service_file in "${PROBLEMATIC_SERVICES[@]}"; do
        if [ -f "$service_file" ]; then
            log_info "修复权限: $service_file"
            sudo chmod 644 "$service_file"
            log_success "✅ 已修复权限: $service_file"
        else
            log_warning "⚠️ 服务文件不存在: $service_file"
        fi
    done
    
    # 修复/run/systemd/system/权限
    if [ -f "/run/systemd/system/netplan-ovs-cleanup.service" ]; then
        log_info "修复netplan-ovs-cleanup.service权限"
        sudo chmod 644 "/run/systemd/system/netplan-ovs-cleanup.service"
        log_success "✅ 已修复netplan-ovs-cleanup.service权限"
    fi
    
    # 重新加载systemd配置
    log_info "重新加载systemd配置..."
    sudo systemctl daemon-reload
    log_success "✅ systemd配置已重新加载"
}

# 修复Tegra摄像头问题
fix_tegra_camera_issues() {
    log_info "📹 修复Tegra摄像头问题..."
    
    # 检查当前的摄像头模块状态
    log_info "检查当前摄像头模块状态..."
    lsmod | grep -E "(tegra|imx219|ov5693)" || log_warning "未找到加载的摄像头模块"
    
    # 尝试重新加载tegra-capture-vi模块
    log_info "尝试重新配置tegra-capture-vi模块..."
    
    # 停止可能使用摄像头的进程
    sudo pkill -f "tegra" 2>/dev/null || true
    sudo pkill -f "nvargus" 2>/dev/null || true
    
    # 等待进程完全停止
    sleep 2
    
    # 重新启动tegra相关服务
    if systemctl is-active --quiet nvargus-daemon 2>/dev/null; then
        log_info "重启nvargus-daemon服务..."
        sudo systemctl restart nvargus-daemon
        log_success "✅ nvargus-daemon服务已重启"
    else
        log_info "启动nvargus-daemon服务..."
        sudo systemctl start nvargus-daemon || log_warning "⚠️ 无法启动nvargus-daemon"
    fi
    
    # 检查摄像头设备
    log_info "检查摄像头设备状态..."
    
    # 列出所有video设备
    if ls /dev/video* >/dev/null 2>&1; then
        log_info "找到的video设备:"
        ls -la /dev/video* | while read line; do
            log_info "  $line"
        done
    else
        log_warning "⚠️ 未找到video设备"
    fi
    
    # 列出所有media设备  
    if ls /dev/media* >/dev/null 2>&1; then
        log_info "找到的media设备:"
        ls -la /dev/media* | while read line; do
            log_info "  $line"
        done
    else
        log_warning "⚠️ 未找到media设备"
    fi
    
    # 检查内核模块
    log_info "检查tegra相关内核模块..."
    if lsmod | grep -q "tegra_vi"; then
        log_success "✅ tegra_vi模块已加载"
    else
        log_warning "⚠️ tegra_vi模块未加载"
    fi
    
    if lsmod | grep -q "tegra_isp"; then
        log_success "✅ tegra_isp模块已加载"  
    else
        log_warning "⚠️ tegra_isp模块未加载"
    fi
}

# 优化系统配置减少日志噪音
optimize_system_logging() {
    log_info "🔇 优化系统日志配置..."
    
    # 创建或更新rsyslog配置来过滤tegra相关的重复错误
    RSYSLOG_FILTER="/etc/rsyslog.d/99-tegra-filter.conf"
    
    cat > /tmp/tegra-filter.conf << 'EOF'
# 过滤Tegra摄像头的重复错误消息
:msg, contains, "tegra-camrtc-capture-vi" ~
:msg, contains, "corr_err: discarding frame" ~
:msg, contains, "err_data 131072" ~

# 过滤systemd的配置警告
:msg, contains, "Unknown key name 'RestartMode'" ~
:msg, contains, "is marked world-writable" ~
:msg, contains, "is marked world-inaccessible" ~
EOF

    sudo mv /tmp/tegra-filter.conf "$RSYSLOG_FILTER"
    sudo chown root:root "$RSYSLOG_FILTER"
    sudo chmod 644 "$RSYSLOG_FILTER"
    
    # 重启rsyslog服务
    sudo systemctl restart rsyslog
    log_success "✅ 系统日志过滤器已配置"
}

# 创建摄像头健康检查脚本
create_camera_health_check() {
    log_info "📊 创建摄像头健康检查脚本..."
    
    HEALTH_CHECK_SCRIPT="/opt/bamboo-cut/camera_health_check.sh"
    
    cat > /tmp/camera_health_check.sh << 'EOF'
#!/bin/bash

# 摄像头健康检查脚本

echo "🔍 摄像头系统健康检查"
echo "======================="

# 检查Tegra服务状态
echo "📋 Tegra服务状态:"
if systemctl is-active --quiet nvargus-daemon; then
    echo "  ✅ nvargus-daemon: 运行中"
else
    echo "  ❌ nvargus-daemon: 未运行"
fi

# 检查内核模块
echo "📋 内核模块状态:"
if lsmod | grep -q tegra_vi; then
    echo "  ✅ tegra_vi: 已加载"
else
    echo "  ❌ tegra_vi: 未加载"
fi

if lsmod | grep -q tegra_isp; then
    echo "  ✅ tegra_isp: 已加载"
else
    echo "  ❌ tegra_isp: 未加载"
fi

# 检查设备文件
echo "📋 设备文件:"
video_count=$(ls /dev/video* 2>/dev/null | wc -l)
media_count=$(ls /dev/media* 2>/dev/null | wc -l)

echo "  📹 Video设备数量: $video_count"
echo "  📺 Media设备数量: $media_count"

if [ $video_count -gt 0 ]; then
    echo "  📹 Video设备列表:"
    ls -la /dev/video* 2>/dev/null | sed 's/^/    /'
fi

if [ $media_count -gt 0 ]; then
    echo "  📺 Media设备列表:"
    ls -la /dev/media* 2>/dev/null | sed 's/^/    /'
fi

# 检查最近的错误
echo "📋 最近的摄像头错误 (最后10条):"
journalctl --since "10 minutes ago" | grep -E "(tegra|camera|camrtc)" | tail -10 | sed 's/^/  /' || echo "  ✅ 无最近错误"

echo ""
echo "🔧 建议操作:"
if [ $video_count -eq 0 ]; then
    echo "  - 检查摄像头硬件连接"
    echo "  - 重启nvargus-daemon服务: sudo systemctl restart nvargus-daemon"
    echo "  - 检查内核模块: sudo modprobe tegra_vi"
fi

echo "  - 查看实时日志: sudo journalctl -f | grep tegra"
echo "  - 重新部署系统: sudo ./jetpack_deploy.sh"
EOF

    sudo mkdir -p "$(dirname "$HEALTH_CHECK_SCRIPT")"
    sudo mv /tmp/camera_health_check.sh "$HEALTH_CHECK_SCRIPT"
    sudo chown root:root "$HEALTH_CHECK_SCRIPT"
    sudo chmod +x "$HEALTH_CHECK_SCRIPT"
    
    log_success "✅ 摄像头健康检查脚本已创建: $HEALTH_CHECK_SCRIPT"
}

# 主函数
main() {
    log_info "🚀 开始系统问题修复..."
    
    # 检查是否以root权限运行
    if [ "$EUID" -ne 0 ]; then
        log_error "请以root权限运行此脚本: sudo $0"
        exit 1
    fi
    
    fix_systemd_permissions
    fix_tegra_camera_issues
    optimize_system_logging
    create_camera_health_check
    
    log_success "🎉 系统问题修复完成!"
    echo ""
    echo "📋 修复摘要:"
    echo "  ✅ 修复了systemd服务文件权限问题"
    echo "  ✅ 重新配置了Tegra摄像头服务"
    echo "  ✅ 优化了系统日志配置"
    echo "  ✅ 创建了摄像头健康检查工具"
    echo ""
    echo "🔧 建议操作:"
    echo "  1. 重启系统以确保所有更改生效: sudo reboot"
    echo "  2. 重启后运行健康检查: sudo /opt/bamboo-cut/camera_health_check.sh"
    echo "  3. 重新部署智能切竹机: sudo ./jetpack_deploy.sh"
    echo ""
    echo "💡 如果问题仍然存在，可能需要:"
    echo "  - 检查摄像头硬件连接"
    echo "  - 更新JetPack SDK驱动程序"
    echo "  - 联系硬件供应商支持"
}

# 运行主函数
main "$@"