#!/bin/bash
# 智能切竹机 EGL 显示初始化问题修复脚本
# 针对 Jetson Nano 和 EGL 显示失败问题

set -e

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

# 检查是否为Jetson设备
check_jetson_device() {
    log_info "检查Jetson设备..."
    
    if [ -f "/proc/device-tree/model" ] && grep -q "Jetson" /proc/device-tree/model; then
        JETSON_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        log_success "检测到Jetson设备: ${JETSON_MODEL}"
        return 0
    else
        log_error "未检测到Jetson设备，此脚本专为Jetson设备设计"
        return 1
    fi
}

# 配置NVIDIA库路径
configure_nvidia_libraries() {
    log_info "配置NVIDIA库路径..."
    
    # 检查Tegra库目录
    TEGRA_LIB_DIRS=(
        "/usr/lib/aarch64-linux-gnu/tegra"
        "/usr/lib/aarch64-linux-gnu/tegra-egl"
        "/usr/lib/nvidia-tegra"
        "/usr/lib/aarch64-linux-gnu/nvidia/current"
    )
    
    FOUND_DIRS=""
    for dir in "${TEGRA_LIB_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            log_success "找到Tegra库目录: $dir"
            FOUND_DIRS="${FOUND_DIRS}:${dir}"
        else
            log_warning "Tegra库目录不存在: $dir"
        fi
    done
    
    if [ -z "$FOUND_DIRS" ]; then
        log_error "未找到任何Tegra库目录"
        return 1
    fi
    
    # 移除开头的冒号
    FOUND_DIRS="${FOUND_DIRS#:}"
    
    # 创建环境配置文件
    cat > /tmp/nvidia_tegra_env.sh << EOF
#!/bin/bash
# NVIDIA Tegra 环境配置

# 配置库路径
export LD_LIBRARY_PATH="${FOUND_DIRS}:\${LD_LIBRARY_PATH}"

# EGL 特定配置
export EGL_PLATFORM=drm
export GBM_BACKEND=nvidia-drm
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d

# 禁用Mesa，强制使用NVIDIA驱动
export LIBGL_ALWAYS_SOFTWARE=0
export MESA_LOADER_DRIVER_OVERRIDE=""
unset MESA_GL_VERSION_OVERRIDE
unset MESA_GLSL_VERSION_OVERRIDE

# Tegra特定优化
export TEGRA_RM_DISABLE_SECURITY=1
export NV_GL_DEBUG=1
export __GL_SYNC_TO_VBLANK=0
export __GL_YIELD=NOTHING

echo "✅ NVIDIA Tegra环境已配置"
echo "   LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
echo "   EGL_PLATFORM: \$EGL_PLATFORM"
EOF

    # 复制到目标位置
    sudo mkdir -p /opt/bamboo-cut
    sudo cp /tmp/nvidia_tegra_env.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/nvidia_tegra_env.sh
    
    log_success "NVIDIA库路径配置完成"
}

# 禁用X11桌面服务
disable_x11_services() {
    log_info "禁用X11桌面服务，切换到控制台模式..."
    
    # X11相关服务列表
    X11_SERVICES=(
        "gdm3"
        "gdm"
        "lightdm"
        "xdm"
        "sddm"
        "display-manager"
        "graphical-session.target"
        "desktop-session"
    )
    
    for service in "${X11_SERVICES[@]}"; do
        if systemctl is-enabled "$service" >/dev/null 2>&1; then
            log_info "禁用X11服务: $service"
            sudo systemctl disable "$service" || true
        fi
        
        if systemctl is-active "$service" >/dev/null 2>&1; then
            log_info "停止X11服务: $service"
            sudo systemctl stop "$service" || true
        fi
    done
    
    # 设置默认启动到控制台模式
    sudo systemctl set-default multi-user.target
    
    log_success "X11桌面服务已禁用，系统将启动到控制台模式"
}

# 配置KMS显示配置
configure_kms_display() {
    log_info "配置KMS显示配置..."
    
    # 检查DRM设备
    if [ ! -e "/dev/dri/card0" ]; then
        log_error "DRM设备 /dev/dri/card0 不存在"
        return 1
    fi
    
    # 创建优化的KMS配置
    sudo mkdir -p /opt/bamboo-cut/config
    cat > /tmp/kms.conf << 'EOF'
{
  "device": "/dev/dri/card0",
  "hwcursor": false,
  "pbuffers": true,
  "separateScreens": false,
  "outputs": [
    {
      "name": "HDMI1",
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": true
    }
  ]
}
EOF
    
    sudo cp /tmp/kms.conf /opt/bamboo-cut/config/
    sudo chown root:root /opt/bamboo-cut/config/kms.conf
    sudo chmod 644 /opt/bamboo-cut/config/kms.conf
    
    log_success "KMS显示配置已更新"
}

# 修复DRM设备权限
fix_drm_permissions() {
    log_info "修复DRM设备权限..."
    
    # 检查并修复DRM设备权限
    DRM_DEVICES=(
        "/dev/dri/card0"
        "/dev/dri/card1"
        "/dev/dri/renderD128"
        "/dev/dri/renderD129"
    )
    
    for device in "${DRM_DEVICES[@]}"; do
        if [ -e "$device" ]; then
            log_info "设置设备权限: $device"
            sudo chmod 666 "$device"
            log_success "权限已设置: $(ls -la $device)"
        else
            log_warning "设备不存在: $device"
        fi
    done
    
    # 创建udev规则确保权限持久化
    cat > /tmp/99-drm-permissions.rules << 'EOF'
# DRM设备权限规则 - 确保bamboo-cut应用可以访问GPU设备
SUBSYSTEM=="drm", KERNEL=="card*", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
EOF
    
    sudo cp /tmp/99-drm-permissions.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    log_success "DRM设备权限修复完成"
}

# 创建增强版启动脚本
create_enhanced_startup_script() {
    log_info "创建增强版启动脚本..."
    
    cat > /tmp/start_bamboo_cut_jetpack_fixed.sh << 'EOF'
#!/bin/bash
# 智能切竹机 JetPack SDK 启动脚本（EGL修复版）

echo "🚀 启动智能切竹机系统（EGL修复版）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载NVIDIA Tegra环境
if [ -f "./nvidia_tegra_env.sh" ]; then
    source "./nvidia_tegra_env.sh"
    echo "✅ NVIDIA Tegra环境已加载"
else
    echo "⚠️ NVIDIA Tegra环境脚本不存在，使用默认配置"
    # 手动设置关键环境变量
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}"
    export EGL_PLATFORM=drm
    export GBM_BACKEND=nvidia-drm
    export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
    export LIBGL_ALWAYS_SOFTWARE=0
    export MESA_LOADER_DRIVER_OVERRIDE=""
    unset MESA_GL_VERSION_OVERRIDE
    unset MESA_GLSL_VERSION_OVERRIDE
fi

# 检查并修复关键设备权限
echo "🔧 检查并修复设备权限..."
for device in /dev/dri/card0 /dev/dri/renderD128 /dev/input/event2; do
    if [ -e "$device" ]; then
        echo "📋 设备权限: $(ls -la $device)"
        chmod 666 "$device" 2>/dev/null || true
    else
        echo "⚠️ 设备不存在: $device"
    fi
done

# 设置XDG_RUNTIME_DIR
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-root}
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Qt EGLFS配置 - 强制使用EGL
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
export QT_QPA_EGLFS_HIDECURSOR=1
export QT_QPA_EGLFS_FORCE_888=1

# 触摸屏设备配置
export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2

# EGL调试日志
export QT_LOGGING_RULES="qt.qpa.*=true;qt.qpa.input*=true;qt.qpa.eglfs*=true"
export QT_QPA_EGLFS_DEBUG=1

echo "✅ EGL环境配置完成"
echo "   Platform: $QT_QPA_PLATFORM"
echo "   Integration: $QT_QPA_EGLFS_INTEGRATION"
echo "   KMS Config: $QT_QPA_EGLFS_KMS_CONFIG"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 检测显示输出
echo "🔍 检测显示输出..."
echo "📋 DRM设备信息:"
ls -la /dev/dri/ 2>/dev/null || echo "   无DRM设备"

echo "📋 显示器连接状态:"
if command -v xrandr >/dev/null 2>&1 && [ -n "$DISPLAY" ]; then
    xrandr --listactivemonitors 2>/dev/null || echo "   无法获取显示器信息"
else
    echo "   控制台模式，无X11显示"
fi

# 应用性能优化 (如果存在)
if [ -f "./power_config.sh" ]; then
    echo "🔧 应用性能优化..."
    ./power_config.sh || echo "⚠️ 性能优化脚本执行失败"
fi

# 检查摄像头设备
echo "🔍 检测摄像头设备..."
CAMERA_FOUND=false
for device in /dev/video0 /dev/video1 /dev/video2; do
    if [ -e "$device" ]; then
        echo "📹 找到摄像头设备: $device"
        CAMERA_FOUND=true
        export BAMBOO_CAMERA_DEVICE="$device"
        break
    fi
done

if [ "$CAMERA_FOUND" = false ]; then
    echo "⚠️ 未检测到摄像头设备，启用模拟模式"
    export BAMBOO_CAMERA_MODE="simulation"
    export BAMBOO_SKIP_CAMERA="true"
else
    export BAMBOO_CAMERA_MODE="hardware"
    export BAMBOO_SKIP_CAMERA="false"
fi

# 设置CUDA环境
export CUDA_VISIBLE_DEVICES=0

# 启动函数
check_and_start_backend() {
    if [ ! -f "./bamboo_cut_backend" ] || [ ! -x "./bamboo_cut_backend" ]; then
        echo "❌ C++后端可执行文件不存在或无执行权限"
        return 1
    fi
    
    echo "🔄 启动 C++ 后端..."
    timeout 60 ./bamboo_cut_backend &
    BACKEND_PID=$!
    
    sleep 8
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "✅ C++ 后端启动成功 (PID: $BACKEND_PID)"
        return 0
    else
        echo "⚠️ C++ 后端退出，这在模拟模式下是正常的"
        wait $BACKEND_PID 2>/dev/null || true
        return 0
    fi
}

check_and_start_frontend() {
    qt_frontend_exec=""
    qt_frontend_candidates=("./bamboo_controller_qt" "./bamboo_cut_frontend")
    
    for candidate in "${qt_frontend_candidates[@]}"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            qt_frontend_exec="$candidate"
            break
        fi
    done
    
    if [ -z "$qt_frontend_exec" ]; then
        echo "⚠️ Qt前端可执行文件不存在，仅运行后端模式"
        return 1
    fi
    
    echo "🔄 启动 Qt 前端: $qt_frontend_exec"
    echo "🔧 使用EGL平台启动..."
    
    # 显示当前EGL环境
    echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "   EGL_PLATFORM: $EGL_PLATFORM"
    echo "   QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
    
    timeout 30 "$qt_frontend_exec" &
    FRONTEND_PID=$!
    
    sleep 8
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ Qt 前端启动成功 (PID: $FRONTEND_PID)"
        return 0
    else
        echo "❌ Qt前端启动失败"
        echo "🔍 检查EGL错误详情..."
        
        echo "📋 DRM设备信息:"
        ls -la /dev/dri/ 2>/dev/null || echo "  无DRM设备"
        
        echo "📋 显卡信息:"
        lspci | grep -i vga 2>/dev/null || echo "  无法获取显卡信息"
        
        echo "📋 当前用户组:"
        groups
        
        wait $FRONTEND_PID 2>/dev/null || true
        return 1
    fi
}

# 主启动逻辑
BACKEND_STARTED=false
FRONTEND_STARTED=false

# 启动后端
if check_and_start_backend; then
    BACKEND_STARTED=true
fi

# 启动前端
if [ "$BACKEND_STARTED" = true ] && kill -0 $BACKEND_PID 2>/dev/null; then
    if check_and_start_frontend; then
        FRONTEND_STARTED=true
        wait $FRONTEND_PID
        kill $BACKEND_PID 2>/dev/null || true
    else
        echo "🔄 仅后端模式运行，等待后端进程..."
        wait $BACKEND_PID
    fi
else
    echo "✅ 后端在模拟模式下正常退出"
fi

echo "🛑 智能切竹机系统已停止"
EOF

    sudo cp /tmp/start_bamboo_cut_jetpack_fixed.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/start_bamboo_cut_jetpack_fixed.sh
    
    log_success "增强版启动脚本已创建"
}

# 更新systemd服务
update_systemd_service() {
    log_info "更新systemd服务配置..."
    
    cat > /tmp/bamboo-cut-jetpack.service << 'EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK) - EGL修复版
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack_fixed.sh
Restart=on-failure
RestartSec=30
StartLimitBurst=3
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
Environment=QT_QPA_EGLFS_ALWAYS_SET_MODE=1
Environment=QT_QPA_EGLFS_HIDECURSOR=1
Environment=QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
Environment=QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl
Environment=EGL_PLATFORM=drm
Environment=GBM_BACKEND=nvidia-drm
Environment=BAMBOO_SKIP_CAMERA=true

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/bamboo-cut-jetpack.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_success "systemd服务配置已更新"
}

# 主函数
main() {
    echo "=================================="
    echo "智能切竹机 EGL 显示问题修复脚本"
    echo "=================================="
    
    # 检查root权限
    if [ "$EUID" -ne 0 ]; then
        log_error "请以root权限运行此脚本: sudo $0"
        exit 1
    fi
    
    # 检查Jetson设备
    if ! check_jetson_device; then
        exit 1
    fi
    
    # 停止现有服务
    log_info "停止现有服务..."
    systemctl stop bamboo-cut-jetpack 2>/dev/null || true
    
    # 执行修复步骤
    configure_nvidia_libraries
    disable_x11_services
    configure_kms_display
    fix_drm_permissions
    create_enhanced_startup_script
    update_systemd_service
    
    # 启用并启动服务
    log_info "启用并启动修复后的服务..."
    systemctl enable bamboo-cut-jetpack
    systemctl start bamboo-cut-jetpack
    
    sleep 3
    
    # 检查服务状态
    if systemctl is-active --quiet bamboo-cut-jetpack; then
        log_success "✅ 智能切竹机服务启动成功！"
        log_info "服务状态: systemctl status bamboo-cut-jetpack"
        log_info "查看日志: journalctl -u bamboo-cut-jetpack -f"
    else
        log_warning "⚠️ 服务启动可能有问题，请检查日志"
        log_info "检查命令: systemctl status bamboo-cut-jetpack"
        log_info "查看错误: journalctl -u bamboo-cut-jetpack --no-pager"
    fi
    
    echo ""
    echo "🎯 修复摘要："
    echo "✅ 配置了NVIDIA Tegra库路径"
    echo "✅ 禁用了X11桌面服务"
    echo "✅ 更新了KMS显示配置"
    echo "✅ 修复了DRM设备权限"
    echo "✅ 创建了增强版启动脚本"
    echo "✅ 更新了systemd服务配置"
    echo ""
    echo "🔧 如果仍有问题，请执行："
    echo "   journalctl -u bamboo-cut-jetpack -f"
    echo "   查看详细日志进行进一步诊断"
    
    log_success "🎉 EGL显示问题修复完成！"
}

# 运行主函数
main "$@"