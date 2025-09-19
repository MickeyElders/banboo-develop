#!/bin/bash
# Jetson Nano Tegra SoC 专用 EGL 修复脚本
# 针对 NVIDIA Tegra GPU 的 EGLDevice/EGLStream 配置

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

# 检查Jetson Nano设备
check_jetson_nano() {
    log_info "检查Jetson Nano设备..."
    
    if [ -f "/proc/device-tree/model" ]; then
        DEVICE_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        echo "设备型号: $DEVICE_MODEL"
        
        if [[ "$DEVICE_MODEL" == *"Jetson Nano"* ]]; then
            log_success "确认为Jetson Nano设备"
            return 0
        fi
    fi
    
    # 检查Tegra芯片
    if [ -f "/proc/device-tree/compatible" ]; then
        COMPATIBLE=$(cat /proc/device-tree/compatible | tr -d '\0')
        if [[ "$COMPATIBLE" == *"tegra210"* ]]; then
            log_success "检测到Tegra210 SoC (Jetson Nano)"
            return 0
        fi
    fi
    
    log_error "未检测到Jetson Nano设备"
    return 1
}

# 配置Jetson Nano专用的NVIDIA库环境
configure_jetson_libraries() {
    log_info "配置Jetson Nano专用NVIDIA库环境..."
    
    # Jetson Nano库路径
    JETSON_LIB_PATHS=(
        "/usr/lib/aarch64-linux-gnu/tegra"
        "/usr/lib/aarch64-linux-gnu/tegra-egl"
        "/usr/lib/nvidia-tegra"
        "/usr/lib/aarch64-linux-gnu/nvidia/current"
    )
    
    VALID_PATHS=""
    for path in "${JETSON_LIB_PATHS[@]}"; do
        if [ -d "$path" ]; then
            log_success "找到库路径: $path"
            VALID_PATHS="${VALID_PATHS}:${path}"
            
            # 列出关键EGL库
            echo "  EGL库文件:"
            ls -la "$path"/libEGL* 2>/dev/null || echo "    无EGL库文件"
            ls -la "$path"/libGL* 2>/dev/null || echo "    无GL库文件"
        else
            log_warning "库路径不存在: $path"
        fi
    done
    
    if [ -z "$VALID_PATHS" ]; then
        log_error "未找到任何NVIDIA Tegra库路径"
        return 1
    fi
    
    # 移除开头的冒号
    VALID_PATHS="${VALID_PATHS#:}"
    
    # 创建Jetson专用环境配置
    cat > /tmp/jetson_nano_env.sh << EOF
#!/bin/bash
# Jetson Nano Tegra SoC 专用环境配置

echo "🔧 配置Jetson Nano Tegra环境..."

# 设置库路径
export LD_LIBRARY_PATH="${VALID_PATHS}:\${LD_LIBRARY_PATH}"

# Jetson Nano 专用 EGL 配置
export EGL_PLATFORM=device
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS=/etc/egl/egl_external_platform.d

# 禁用GBM，使用EGLDevice
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
export QT_QPA_EGLFS_KMS_ATOMIC=1

# NVIDIA Tegra 专用设置
export CUDA_VISIBLE_DEVICES=0
export TEGRA_RM_DISABLE_SECURITY=1

# OpenGL设置
export __GL_SYNC_TO_VBLANK=0
export __GL_YIELD=NOTHING
export LIBGL_ALWAYS_SOFTWARE=0

# 禁用Mesa干扰
unset MESA_LOADER_DRIVER_OVERRIDE
unset MESA_GL_VERSION_OVERRIDE
unset MESA_GLSL_VERSION_OVERRIDE

echo "✅ Jetson Nano环境配置完成"
echo "   LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
echo "   EGL_PLATFORM: \$EGL_PLATFORM"
echo "   QT_EGLFS_INTEGRATION: \$QT_QPA_EGLFS_INTEGRATION"
EOF

    sudo mkdir -p /opt/bamboo-cut
    sudo cp /tmp/jetson_nano_env.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/jetson_nano_env.sh
    
    log_success "Jetson Nano环境配置完成"
}

# 创建Jetson Nano专用KMS配置
create_jetson_kms_config() {
    log_info "创建Jetson Nano专用KMS配置..."
    
    sudo mkdir -p /opt/bamboo-cut/config
    
    # Jetson Nano专用KMS配置
    cat > /tmp/jetson_kms.conf << 'EOF'
{
  "device": "/dev/dri/card0",
  "hwcursor": false,
  "pbuffers": true,
  "separateScreens": false,
  "format": "argb8888",
  "outputs": [
    {
      "name": "HDMI-A-1",
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": true,
      "format": "xrgb8888"
    }
  ]
}
EOF
    
    sudo cp /tmp/jetson_kms.conf /opt/bamboo-cut/config/kms.conf
    sudo chown root:root /opt/bamboo-cut/config/kms.conf
    sudo chmod 644 /opt/bamboo-cut/config/kms.conf
    
    log_success "Jetson Nano KMS配置已创建"
}

# 检查Tegra GPU状态（替代nvidia-smi）
check_tegra_gpu() {
    log_info "检查Tegra GPU状态..."
    
    echo "📋 Tegra GPU信息："
    
    # 检查Tegra设备树信息
    if [ -f "/proc/device-tree/gpu@57000000/compatible" ]; then
        GPU_COMPATIBLE=$(cat /proc/device-tree/gpu@57000000/compatible 2>/dev/null | tr -d '\0')
        echo "  GPU兼容性: $GPU_COMPATIBLE"
    fi
    
    # 检查GPU频率设置
    if [ -f "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/available_frequencies" ]; then
        GPU_FREQS=$(cat /sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/available_frequencies 2>/dev/null)
        echo "  可用频率: $GPU_FREQS"
    fi
    
    if [ -f "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq" ]; then
        GPU_FREQ=$(cat /sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq 2>/dev/null)
        echo "  当前频率: $GPU_FREQ Hz"
    fi
    
    # 检查3D控制器（替代显卡信息）
    echo "📋 3D控制器信息："
    lspci | grep -i "3d\|vga\|display" || echo "  未找到PCIe显示设备（正常，Tegra为集成GPU）"
    
    # 检查OpenGL信息
    echo "📋 OpenGL渲染器信息："
    if command -v glxinfo >/dev/null 2>&1; then
        glxinfo | grep -i "renderer\|vendor\|version" 2>/dev/null || echo "  无法获取OpenGL信息"
    else
        echo "  glxinfo未安装"
    fi
    
    log_success "Tegra GPU状态检查完成"
}

# 创建Jetson Nano专用启动脚本
create_jetson_startup_script() {
    log_info "创建Jetson Nano专用启动脚本..."
    
    cat > /tmp/start_bamboo_jetson_nano.sh << 'EOF'
#!/bin/bash
# Jetson Nano 专用智能切竹机启动脚本

echo "🚀 启动智能切竹机系统（Jetson Nano专用版）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载Jetson Nano环境
if [ -f "./jetson_nano_env.sh" ]; then
    source "./jetson_nano_env.sh"
    echo "✅ Jetson Nano环境已加载"
else
    echo "⚠️ Jetson Nano环境脚本不存在，使用内置配置"
    
    # 内置Jetson Nano配置
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}"
    export EGL_PLATFORM=device
    export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
    export __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS=/etc/egl/egl_external_platform.d
    export QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
    export QT_QPA_EGLFS_KMS_ATOMIC=1
    export CUDA_VISIBLE_DEVICES=0
    export TEGRA_RM_DISABLE_SECURITY=1
    export __GL_SYNC_TO_VBLANK=0
    export __GL_YIELD=NOTHING
    export LIBGL_ALWAYS_SOFTWARE=0
    unset MESA_LOADER_DRIVER_OVERRIDE
    unset MESA_GL_VERSION_OVERRIDE
    unset MESA_GLSL_VERSION_OVERRIDE
fi

# 设置XDG运行时目录
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-root}
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Qt EGLFS配置 - Jetson Nano专用
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
export QT_QPA_EGLFS_HIDECURSOR=1
export QT_QPA_EGLFS_KMS_ATOMIC=1

# 触摸屏配置
export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2

# EGL调试日志
export QT_LOGGING_RULES="qt.qpa.*=true;qt.qpa.eglfs*=true"
export QT_QPA_EGLFS_DEBUG=1

echo "✅ Jetson Nano EGL环境配置完成"
echo "   Platform: $QT_QPA_PLATFORM"
echo "   Integration: $QT_QPA_EGLFS_INTEGRATION"
echo "   EGL Platform: $EGL_PLATFORM"
echo "   KMS Config: $QT_QPA_EGLFS_KMS_CONFIG"

# 检查关键设备权限
echo "🔧 检查设备权限..."
for device in /dev/dri/card0 /dev/dri/renderD128 /dev/input/event2; do
    if [ -e "$device" ]; then
        echo "📋 设备: $(ls -la $device)"
        chmod 666 "$device" 2>/dev/null || true
    else
        echo "⚠️ 设备不存在: $device"
    fi
done

# 检查Tegra GPU状态（非nvidia-smi）
echo "🔍 检查Tegra GPU状态..."
if [ -f "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq" ]; then
    GPU_FREQ=$(cat /sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq 2>/dev/null)
    echo "📋 当前GPU频率: $GPU_FREQ Hz"
else
    echo "📋 Tegra GPU: 集成在SoC中（无需nvidia-smi）"
fi

# 检查DRM设备
echo "📋 DRM设备信息:"
ls -la /dev/dri/ 2>/dev/null || echo "   无DRM设备"

# 检查EGL库
echo "📋 EGL库检查:"
if [ -f "/usr/lib/aarch64-linux-gnu/tegra/libEGL.so" ]; then
    echo "   ✅ Tegra EGL库存在"
    ldd /usr/lib/aarch64-linux-gnu/tegra/libEGL.so | head -3
else
    echo "   ⚠️ Tegra EGL库不存在"
fi

# 应用性能优化
if [ -f "./power_config.sh" ]; then
    echo "🔧 应用Jetson性能优化..."
    ./power_config.sh || echo "⚠️ 性能优化失败"
fi

# 摄像头检测
echo "🔍 检测摄像头设备..."
CAMERA_FOUND=false
for device in /dev/video0 /dev/video1 /dev/video2; do
    if [ -e "$device" ]; then
        echo "📹 摄像头设备: $device"
        CAMERA_FOUND=true
        export BAMBOO_CAMERA_DEVICE="$device"
        break
    fi
done

if [ "$CAMERA_FOUND" = false ]; then
    echo "⚠️ 未检测到摄像头，启用模拟模式"
    export BAMBOO_CAMERA_MODE="simulation"
    export BAMBOO_SKIP_CAMERA="true"
else
    export BAMBOO_CAMERA_MODE="hardware"
    export BAMBOO_SKIP_CAMERA="false"
fi

# 启动后端
start_backend() {
    if [ ! -f "./bamboo_cut_backend" ] || [ ! -x "./bamboo_cut_backend" ]; then
        echo "❌ C++后端可执行文件不存在"
        return 1
    fi
    
    echo "🔄 启动C++后端..."
    timeout 60 ./bamboo_cut_backend &
    BACKEND_PID=$!
    
    sleep 8
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "✅ C++后端启动成功 (PID: $BACKEND_PID)"
        return 0
    else
        echo "⚠️ C++后端退出（模拟模式下正常）"
        wait $BACKEND_PID 2>/dev/null || true
        return 0
    fi
}

# 启动前端
start_frontend() {
    qt_frontend_exec=""
    for candidate in "./bamboo_controller_qt" "./bamboo_cut_frontend"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            qt_frontend_exec="$candidate"
            break
        fi
    done
    
    if [ -z "$qt_frontend_exec" ]; then
        echo "⚠️ Qt前端不存在，仅后端模式"
        return 1
    fi
    
    echo "🔄 启动Qt前端: $qt_frontend_exec"
    echo "🔧 使用Jetson Nano EGLDevice模式..."
    
    # 显示当前配置
    echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo "   EGL_PLATFORM: ${EGL_PLATFORM}"
    echo "   QT_QPA_PLATFORM: ${QT_QPA_PLATFORM}"
    echo "   QT_QPA_EGLFS_INTEGRATION: ${QT_QPA_EGLFS_INTEGRATION}"
    
    # 启动Qt前端
    timeout 30 "$qt_frontend_exec" &
    FRONTEND_PID=$!
    
    sleep 8
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ Qt前端启动成功 (PID: $FRONTEND_PID)"
        return 0
    else
        echo "❌ Qt前端启动失败"
        echo "🔍 EGL错误诊断:"
        
        # Jetson专用诊断
        echo "📋 Tegra驱动状态:"
        find /sys -name "*tegra*" -type d 2>/dev/null | head -5 || echo "   无Tegra驱动信息"
        
        echo "📋 EGL设备:"
        find /dev -name "nvidia*" 2>/dev/null || echo "   无NVIDIA设备节点"
        
        wait $FRONTEND_PID 2>/dev/null || true
        return 1
    fi
}

# 主启动逻辑
echo "🚀 开始启动应用..."

# 启动后端
BACKEND_STARTED=false
if start_backend; then
    BACKEND_STARTED=true
fi

# 启动前端
if [ "$BACKEND_STARTED" = true ] && kill -0 $BACKEND_PID 2>/dev/null; then
    if start_frontend; then
        wait $FRONTEND_PID
        kill $BACKEND_PID 2>/dev/null || true
    else
        echo "🔄 仅后端模式运行"
        wait $BACKEND_PID
    fi
else
    echo "✅ 后端在模拟模式下完成"
fi

echo "🛑 Jetson Nano智能切竹机系统已停止"
EOF

    sudo cp /tmp/start_bamboo_jetson_nano.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/start_bamboo_jetson_nano.sh
    
    log_success "Jetson Nano专用启动脚本已创建"
}

# 更新systemd服务为Jetson Nano专用
update_jetson_systemd_service() {
    log_info "更新systemd服务为Jetson Nano专用..."
    
    cat > /tmp/bamboo-cut-jetpack.service << 'EOF'
[Unit]
Description=智能切竹机系统 (Jetson Nano专用)
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_jetson_nano.sh
Restart=on-failure
RestartSec=30
StartLimitBurst=3
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
Environment=QT_QPA_EGLFS_KMS_ATOMIC=1
Environment=EGL_PLATFORM=device
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TEGRA_RM_DISABLE_SECURITY=1

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/bamboo-cut-jetpack.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_success "Jetson Nano专用systemd服务已更新"
}

# 主函数
main() {
    echo "========================================"
    echo "Jetson Nano Tegra SoC EGL 专用修复脚本"
    echo "========================================"
    
    # 检查root权限
    if [ "$EUID" -ne 0 ]; then
        log_error "请以root权限运行: sudo $0"
        exit 1
    fi
    
    # 检查Jetson Nano
    if ! check_jetson_nano; then
        exit 1
    fi
    
    # 停止现有服务
    log_info "停止现有服务..."
    systemctl stop bamboo-cut-jetpack 2>/dev/null || true
    
    # 执行Jetson Nano专用修复
    configure_jetson_libraries
    create_jetson_kms_config
    check_tegra_gpu
    create_jetson_startup_script
    update_jetson_systemd_service
    
    # 启动服务
    log_info "启动Jetson Nano专用服务..."
    systemctl enable bamboo-cut-jetpack
    systemctl start bamboo-cut-jetpack
    
    sleep 3
    
    # 检查结果
    if systemctl is-active --quiet bamboo-cut-jetpack; then
        log_success "✅ Jetson Nano智能切竹机服务启动成功！"
        log_info "查看状态: systemctl status bamboo-cut-jetpack"
        log_info "查看日志: journalctl -u bamboo-cut-jetpack -f"
    else
        log_warning "⚠️ 服务启动可能有问题"
        log_info "检查详情: journalctl -u bamboo-cut-jetpack --no-pager"
    fi
    
    echo ""
    echo "🎯 Jetson Nano专用修复摘要："
    echo "✅ 配置了Tegra专用库路径和EGL环境"
    echo "✅ 使用EGLDevice而非GBM模式"
    echo "✅ 创建了Jetson Nano专用KMS配置"
    echo "✅ 添加了Tegra GPU状态检查（非nvidia-smi）"
    echo "✅ 更新了启动脚本和systemd服务"
    echo ""
    echo "🔧 注意：Jetson Nano使用集成Tegra GPU，无需nvidia-smi"
    
    log_success "🎉 Jetson Nano EGL专用修复完成！"
}

# 运行主函数
main "$@"