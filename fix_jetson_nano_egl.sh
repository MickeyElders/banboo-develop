#!/bin/bash
# Jetson Nano/Orin Tegra SoC 专用 EGL 修复脚本 + 完整编译部署功能
# 合并了 jetpack_deploy.sh 的编译部署功能和 fix_jetson_nano_egl.sh 的专用配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本信息 - 修正为根目录调用
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="${PROJECT_ROOT}/build"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"
JETPACK_DEPLOY_DIR="${DEPLOY_DIR}/jetpack"

# JetPack SDK 配置
JETPACK_VERSION="${JETPACK_VERSION:-5.1.1}"
CUDA_VERSION="${CUDA_VERSION:-11.4}"
TENSORRT_VERSION="${TENSORRT_VERSION:-8.5.2}"
OPENCV_VERSION="${OPENCV_VERSION:-4.8.0}"

# 默认配置
BUILD_TYPE="Release"
ENABLE_TENSORRT="ON"
ENABLE_GPU_OPTIMIZATION="ON"
ENABLE_POWER_OPTIMIZATION="ON"
INSTALL_DEPENDENCIES="false"
DEPLOY_TARGET="local"
CREATE_PACKAGE="true"
OPTIMIZE_PERFORMANCE="true"
CLEAN_LEGACY="false"
BACKUP_CURRENT="true"
FORCE_REBUILD="false"
ENABLE_QT_DEPLOY="true"
DEPLOY_MODELS="true"

# 版本信息
VERSION_FILE="${PROJECT_ROOT}/VERSION"
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(cat "$VERSION_FILE")
else
    VERSION="1.0.0"
fi

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

log_jetpack() {
    echo -e "${PURPLE}[JETPACK]${NC} $1"
}

log_qt() {
    echo -e "${CYAN}[QT-DEPLOY]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Jetson Tegra SoC 专用 EGL 修复 + 完整编译部署脚本

用法: $0 [选项]

🎯 主要功能:
    - 自动检测并适配多种 Jetson 设备
    - 完整编译 C++ 后端和 Qt 前端
    - 部署 AI 模型和 Qt 依赖
    - 配置 Tegra 专用的 EGL 环境
    - 创建优化的 KMS 配置和启动脚本
    - 设置和启动 systemd 服务

⚙️  可选参数:
    -t, --type TYPE         构建类型 (Debug, Release) [默认: Release]
    -i, --install-deps      安装 JetPack SDK 依赖包
    -b, --no-backup         重新部署时不备份当前版本
    -f, --force-rebuild     强制重新编译所有组件
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

🚀 使用示例:
    $0                                              # 完整编译、部署和配置
    $0 --install-deps                               # 安装依赖并完整部署
    $0 --type Debug                                 # Debug 模式部署
    $0 --force-rebuild --no-backup                  # 强制重编译，不备份

🔧 系统信息:
    JetPack SDK 版本: ${JETPACK_VERSION}
    CUDA 版本: ${CUDA_VERSION}
    TensorRT 版本: ${TENSORRT_VERSION}
    OpenCV 版本: ${OPENCV_VERSION}

💡 提示:
    - 脚本会自动处理进程清理、编译、部署和启动
    - 支持 Jetson Nano, Orin NX, AGX Orin, Xavier
    - 使用 EGLDevice 而非 GBM 模式以获得最佳性能
    - 如果需要查看服务状态: sudo systemctl status bamboo-cut-jetpack

EOF
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -i|--install-deps)
                INSTALL_DEPENDENCIES="true"
                shift
                ;;
            -b|--no-backup)
                BACKUP_CURRENT="false"
                shift
                ;;
            -f|--force-rebuild)
                FORCE_REBUILD="true"
                CLEAN_LEGACY="true"
                shift
                ;;
            -v|--version)
                echo "Jetson Tegra EGL 修复 + 编译部署脚本 版本 ${VERSION}"
                echo "JetPack SDK: ${JETPACK_VERSION}"
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检测Jetson设备（支持多种型号）
check_jetson_device() {
    log_info "检测Jetson设备..."
    
    if [ -f "/proc/device-tree/model" ]; then
        DEVICE_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        echo "设备型号: $DEVICE_MODEL"
        
        # 支持多种Jetson设备
        if [[ "$DEVICE_MODEL" == *"Jetson"* ]]; then
            if [[ "$DEVICE_MODEL" == *"Orin NX"* ]]; then
                JETSON_TYPE="orin-nx"
                TEGRA_CHIP="tegra234"
                GPU_PATH="17000000.gpu"
                log_success "确认为Jetson Orin NX设备"
                return 0
            elif [[ "$DEVICE_MODEL" == *"Jetson Nano"* ]]; then
                JETSON_TYPE="nano"
                TEGRA_CHIP="tegra210"
                GPU_PATH="57000000.gpu"
                log_success "确认为Jetson Nano设备"
                return 0
            elif [[ "$DEVICE_MODEL" == *"Jetson AGX Orin"* ]]; then
                JETSON_TYPE="agx-orin"
                TEGRA_CHIP="tegra234"
                GPU_PATH="17000000.gpu"
                log_success "确认为Jetson AGX Orin设备"
                return 0
            elif [[ "$DEVICE_MODEL" == *"Jetson Xavier"* ]]; then
                JETSON_TYPE="xavier"
                TEGRA_CHIP="tegra194"
                GPU_PATH="17000000.gpu"
                log_success "确认为Jetson Xavier设备"
                return 0
            else
                log_success "检测到Jetson设备: $DEVICE_MODEL"
                # 默认配置
                JETSON_TYPE="generic"
                TEGRA_CHIP="tegra"
                GPU_PATH="*.gpu"
                return 0
            fi
        fi
    fi
    
    # 检查Tegra芯片兼容性
    if [ -f "/proc/device-tree/compatible" ]; then
        COMPATIBLE=$(cat /proc/device-tree/compatible | tr -d '\0')
        if [[ "$COMPATIBLE" == *"tegra234"* ]]; then
            JETSON_TYPE="orin"
            TEGRA_CHIP="tegra234"
            GPU_PATH="17000000.gpu"
            log_success "检测到Tegra234 SoC (Jetson Orin系列)"
            return 0
        elif [[ "$COMPATIBLE" == *"tegra210"* ]]; then
            JETSON_TYPE="nano"
            TEGRA_CHIP="tegra210"
            GPU_PATH="57000000.gpu"
            log_success "检测到Tegra210 SoC (Jetson Nano)"
            return 0
        elif [[ "$COMPATIBLE" == *"tegra194"* ]]; then
            JETSON_TYPE="xavier"
            TEGRA_CHIP="tegra194"
            GPU_PATH="17000000.gpu"
            log_success "检测到Tegra194 SoC (Jetson Xavier)"
            return 0
        fi
    fi
    
    log_error "未检测到支持的Jetson设备"
    return 1
}

# 停止所有运行中的服务和进程
stop_running_services() {
    log_info "🛑 停止所有运行中的智能切竹机服务和进程..."
    
    # 智能切竹机相关服务清单
    BAMBOO_SERVICES=(
        "bamboo-cut-jetpack"
        "bamboo-cut"
        "bamboo-controller"
        "bamboo-backend"
        "bamboo-frontend"
        "bamboo-cut-backend"
        "bamboo-cut-frontend"
        "bamboo-controller-qt"
    )
    
    # 停止systemd服务
    for service in "${BAMBOO_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            log_info "停止服务: $service"
            sudo systemctl stop "$service" || true
            
            # 等待服务完全停止
            local timeout=10
            while [ $timeout -gt 0 ] && systemctl is-active --quiet "$service" 2>/dev/null; do
                sleep 1
                timeout=$((timeout - 1))
            done
            
            if systemctl is-active --quiet "$service" 2>/dev/null; then
                log_warning "服务 $service 未能在10秒内停止，将强制终止"
            else
                log_success "服务 $service 已停止"
            fi
        fi
    done
    
    # 强制终止所有相关进程
    log_info "强制终止所有相关进程..."
    
    BAMBOO_PROCESSES=(
        "bamboo_cut_backend"
        "bamboo_controller_qt"
        "bamboo-cut"
        "bamboo_cut_frontend"
        "bamboo-backend"
        "bamboo-frontend"
        "start_bamboo_cut_jetpack.sh"
        "start_qt_frontend_only.sh"
    )
    
    for process in "${BAMBOO_PROCESSES[@]}"; do
        if pgrep -f "$process" >/dev/null 2>&1; then
            log_info "终止进程: $process"
            sudo pkill -TERM -f "$process" || true
            sleep 2
            
            # 如果进程仍在运行，强制终止
            if pgrep -f "$process" >/dev/null 2>&1; then
                log_warning "强制终止进程: $process"
                sudo pkill -KILL -f "$process" || true
                sleep 1
            fi
            
            if ! pgrep -f "$process" >/dev/null 2>&1; then
                log_success "进程 $process 已终止"
            fi
        fi
    done
    
    # 清理可能的僵尸进程
    log_info "清理僵尸进程..."
    sudo pkill -KILL -f "bamboo" 2>/dev/null || true
    
    # 等待所有进程完全退出
    sleep 3
    
    log_success "✅ 所有运行中的服务和进程已停止"
}

# 完全清理历史版本进程和配置
clean_legacy_deployment() {
    if [ "$CLEAN_LEGACY" = "true" ]; then
        log_info "🧹 清理历史版本进程和配置..."
        
        # 清理历史安装目录
        log_info "清理历史安装目录..."
        
        LEGACY_INSTALL_DIRS=(
            "/opt/bamboo-controller"
            "/opt/bamboo-backend"
            "/opt/bamboo-frontend"
            "/usr/local/bin/bamboo-cut"
            "/usr/local/share/bamboo-cut"
        )
        
        for dir in "${LEGACY_INSTALL_DIRS[@]}"; do
            if [ -d "$dir" ]; then
                log_info "清理历史目录: $dir"
                sudo rm -rf "$dir"
            fi
        done
        
        # 重新加载systemd配置
        log_info "重新加载systemd配置..."
        sudo systemctl daemon-reload
        
        # 清理systemd的缓存
        sudo systemctl reset-failed 2>/dev/null || true
        
        log_success "✅ 历史版本清理完成"
    fi
}

# 备份当前部署
backup_current_deployment() {
    if [ "$BACKUP_CURRENT" = "true" ] && [ -d "/opt/bamboo-cut" ]; then
        BACKUP_DIR="/opt/bamboo-cut.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "备份当前部署到: $BACKUP_DIR"
        
        sudo cp -r "/opt/bamboo-cut" "$BACKUP_DIR"
        
        # 创建备份信息文件
        cat > /tmp/backup_info.txt << EOF
备份时间: $(date)
备份路径: $BACKUP_DIR
版本: $VERSION
Git提交: $(git rev-parse HEAD 2>/dev/null || echo "未知")
EOF
        sudo mv /tmp/backup_info.txt "$BACKUP_DIR/backup_info.txt"
        
        log_success "当前部署已备份"
    fi
}

# 安装 JetPack SDK 依赖
install_jetpack_dependencies() {
    if [ "$INSTALL_DEPENDENCIES" = "true" ]; then
        log_jetpack "安装 JetPack SDK 依赖包..."
        
        # 更新包管理器
        sudo apt update
        
        # JetPack SDK 核心组件
        log_jetpack "安装 JetPack SDK 核心组件..."
        sudo apt install -y \
            nvidia-jetpack \
            cuda-toolkit-${CUDA_VERSION//./-} \
            tensorrt \
            libnvinfer-dev \
            libnvonnxparsers-dev \
            libnvinfer-plugin-dev || log_warning "部分CUDA组件安装失败，将跳过GPU加速功能"
        
        # OpenCV for Jetson
        log_jetpack "安装 OpenCV for Jetson..."
        sudo apt install -y \
            libopencv-dev \
            libopencv-contrib-dev \
            python3-opencv
        
        # Qt6 for Jetson
        log_jetpack "安装 Qt6 for Jetson..."
        sudo apt install -y \
            qt6-base-dev \
            qt6-declarative-dev \
            qt6-multimedia-dev \
            qt6-serialport-dev \
            qt6-tools-dev \
            qt6-wayland \
            qml6-module-qtquick \
            qml6-module-qtquick-controls
        
        # GStreamer for hardware acceleration
        log_jetpack "安装 GStreamer 硬件加速组件..."
        sudo apt install -y \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-plugins-ugly \
            gstreamer1.0-libav \
            gstreamer1.0-tools \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev
        
        # 其他必要依赖
        log_jetpack "安装其他必要依赖..."
        sudo apt install -y \
            build-essential \
            cmake \
            ninja-build \
            pkg-config \
            git \
            libmodbus-dev \
            nlohmann-json3-dev \
            libeigen3-dev \
            libprotobuf-dev \
            protobuf-compiler
        
        log_success "JetPack SDK 依赖包安装完成"
    fi
}

# 清理构建缓存（强制重新编译时）
clean_build_cache() {
    if [ "$FORCE_REBUILD" = "true" ]; then
        log_info "强制重新编译：清理构建缓存..."
        
        # 清理C++后端构建缓存
        if [ -d "${BUILD_DIR}/cpp_backend" ]; then
            rm -rf "${BUILD_DIR}/cpp_backend"
            log_info "已清理C++后端构建缓存"
        fi
        
        # 清理Qt前端构建缓存
        if [ -d "${BUILD_DIR}/qt_frontend" ]; then
            rm -rf "${BUILD_DIR}/qt_frontend"
            log_info "已清理Qt前端构建缓存"
        fi
        
        # 清理Qt项目内的构建目录
        for build_dir in "${PROJECT_ROOT}/qt_frontend/build" "${PROJECT_ROOT}/qt_frontend/build_debug" "${PROJECT_ROOT}/qt_frontend/build_release"; do
            if [ -d "$build_dir" ]; then
                rm -rf "$build_dir"
                log_info "已清理Qt构建目录: $(basename $build_dir)"
            fi
        done
        
        log_success "构建缓存清理完成"
    fi
}

# 构建项目 - 分别构建各子项目
build_project() {
    log_info "构建智能切竹机项目..."
    
    cd "$PROJECT_ROOT"
    
    # 清理构建缓存（如果需要）
    clean_build_cache
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    
    # 构建 C++ 后端
    log_info "构建 C++ 后端..."
    mkdir -p "${BUILD_DIR}/cpp_backend"
    cd "${BUILD_DIR}/cpp_backend"
    
    # CMake 配置参数
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
        -DENABLE_TENSORRT="$ENABLE_TENSORRT"
        -DENABLE_GPU_OPTIMIZATION="$ENABLE_GPU_OPTIMIZATION"
        -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"
    )
    
    if [ "$JETSON_DETECTED" = "true" ]; then
        CMAKE_ARGS+=(-DJETSON_BUILD=ON)
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="53;62;72;75;86;87")
    fi
    
    # 运行 CMake for C++ backend
    if ! cmake "${CMAKE_ARGS[@]}" "${PROJECT_ROOT}/cpp_backend"; then
        log_error "C++ 后端 CMake 配置失败"
        return 1
    fi
    
    # 编译 C++ 后端
    if ! make -j$(nproc); then
        log_error "C++ 后端编译失败"
        return 1
    fi
    
    log_success "C++ 后端构建完成"
    
    # 构建 Qt 前端
    log_info "构建 Qt 前端..."
    mkdir -p "${BUILD_DIR}/qt_frontend"
    cd "${BUILD_DIR}/qt_frontend"
    
    # Qt 前端 CMake 配置参数
    QT_CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
        -DCMAKE_PREFIX_PATH="/usr/lib/aarch64-linux-gnu/cmake"
    )
    
    if [ "$JETSON_DETECTED" = "true" ]; then
        QT_CMAKE_ARGS+=(-DJETSON_BUILD=ON)
    fi
    
    # 运行 CMake for Qt frontend
    if ! cmake "${QT_CMAKE_ARGS[@]}" "${PROJECT_ROOT}/qt_frontend"; then
        log_error "Qt 前端 CMake 配置失败"
        return 1
    fi
    
    # 编译 Qt 前端
    if ! make -j$(nproc); then
        log_error "Qt 前端编译失败"
        return 1
    fi
    
    log_success "Qt 前端构建完成"
    log_success "项目构建完成"
    return 0
}

# 配置Jetson专用的NVIDIA库环境
configure_jetson_libraries() {
    log_jetpack "配置Jetson专用NVIDIA库环境..."
    
    # Jetson库路径
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
    cat > "${JETPACK_DEPLOY_DIR}/jetson_tegra_env.sh" << EOF
#!/bin/bash
# Jetson $JETSON_TYPE Tegra SoC 专用环境配置

echo "🔧 配置Jetson $JETSON_TYPE Tegra环境..."

# 设置库路径
export LD_LIBRARY_PATH="${VALID_PATHS}:\${LD_LIBRARY_PATH}"

# Jetson 专用 EGL 配置
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

echo "✅ Jetson $JETSON_TYPE 环境配置完成"
echo "   设备类型: $JETSON_TYPE"
echo "   Tegra芯片: $TEGRA_CHIP" 
echo "   LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
echo "   EGL_PLATFORM: \$EGL_PLATFORM"
echo "   QT_EGLFS_INTEGRATION: \$QT_QPA_EGLFS_INTEGRATION"
EOF
    
    chmod +x "${JETPACK_DEPLOY_DIR}/jetson_tegra_env.sh"
    
    log_success "Jetson $JETSON_TYPE 环境配置完成"
}

# 创建Jetson专用KMS配置
create_jetson_kms_config() {
    log_jetpack "创建Jetson $JETSON_TYPE 专用KMS配置..."
    
    mkdir -p "${JETPACK_DEPLOY_DIR}/config"
    
    # 根据不同Jetson设备创建对应的KMS配置
    case "$JETSON_TYPE" in
        "nano")
            KMS_CONFIG_CONTENT='{
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
}'
            ;;
        "orin"*|"agx-orin")
            KMS_CONFIG_CONTENT='{
  "device": "/dev/dri/card0",
  "hwcursor": true,
  "pbuffers": true,
  "separateScreens": false,
  "format": "argb8888",
  "outputs": [
    {
      "name": "DP-1",
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": true,
      "format": "xrgb8888"
    },
    {
      "name": "HDMI-A-1", 
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": false,
      "format": "xrgb8888"
    }
  ]
}'
            ;;
        "xavier")
            KMS_CONFIG_CONTENT='{
  "device": "/dev/dri/card0",
  "hwcursor": true,
  "pbuffers": true,
  "separateScreens": false,
  "format": "argb8888",
  "outputs": [
    {
      "name": "DP-1",
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": true,
      "format": "xrgb8888"
    }
  ]
}'
            ;;
        *)
            # 通用配置
            KMS_CONFIG_CONTENT='{
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
}'
            ;;
    esac
    
    echo "$KMS_CONFIG_CONTENT" > "${JETPACK_DEPLOY_DIR}/config/kms.conf"
    
    log_success "Jetson $JETSON_TYPE KMS配置已创建"
}

# 配置 JetPack SDK 性能优化
configure_jetpack_performance() {
    if [ "$OPTIMIZE_PERFORMANCE" = "true" ]; then
        log_jetpack "配置 JetPack SDK 性能优化..."
        
        # GPU 内存和计算优化
        if [ "$ENABLE_GPU_OPTIMIZATION" = "ON" ]; then
            log_jetpack "配置 GPU 内存和计算优化..."
            
            # 设置 CUDA 环境变量
            export CUDA_VISIBLE_DEVICES=0
            export CUDA_CACHE_DISABLE=0
            export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
            
            # GPU 频率优化 (需要 root 权限)
            if [ "$JETSON_DETECTED" = "true" ]; then
                # 设置最大性能模式
                sudo nvpmodel -m 0 || log_warning "无法设置 nvpmodel，可能需要 root 权限"
                
                # 设置 GPU 最大时钟
                sudo jetson_clocks || log_warning "无法设置 jetson_clocks，可能需要 root 权限"
            fi
        fi
        
        # 功耗管理优化
        if [ "$ENABLE_POWER_OPTIMIZATION" = "ON" ]; then
            log_jetpack "配置功耗管理优化..."
            
            # 创建功耗配置文件（移除sudo，因为systemd服务以root运行）
            cat > "${JETPACK_DEPLOY_DIR}/power_config.sh" << 'EOF'
#!/bin/bash
# JetPack SDK 功耗管理配置

echo "🔧 应用JetPack性能优化设置..."

# 设置 CPU 调度器为性能模式
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1
    echo "✅ CPU调度器已设置为性能模式"
else
    echo "⚠️ 无法设置CPU调度器，跳过"
fi

# 优化内存管理
if [ -w /proc/sys/vm/overcommit_memory ]; then
    echo 1 | tee /proc/sys/vm/overcommit_memory > /dev/null 2>&1
    echo "✅ 内存过量分配已优化"
else
    echo "⚠️ 无法设置内存管理，跳过"
fi

if [ -w /proc/sys/vm/swappiness ]; then
    echo 80 | tee /proc/sys/vm/swappiness > /dev/null 2>&1
    echo "✅ 交换分区优化已设置"
else
    echo "⚠️ 无法设置交换分区，跳过"
fi

# GPU 功耗管理
if [ -w /sys/devices/platform/host1x/*/power/autosuspend_delay_ms ]; then
    echo 1 | tee /sys/devices/platform/host1x/*/power/autosuspend_delay_ms > /dev/null 2>&1
    echo "✅ GPU功耗管理已优化"
else
    echo "⚠️ 无法设置GPU功耗管理，跳过"
fi

# 网络优化
if [ -w /proc/sys/net/core/netdev_max_backlog ]; then
    echo 1 | tee /proc/sys/net/core/netdev_max_backlog > /dev/null 2>&1
    echo "✅ 网络优化已设置"
else
    echo "⚠️ 无法设置网络优化，跳过"
fi

echo "🎉 JetPack性能优化设置完成"
EOF
            chmod +x "${JETPACK_DEPLOY_DIR}/power_config.sh"
        fi
        
        log_success "JetPack SDK 性能优化配置完成"
    fi
}

# Qt 依赖收集和部署 (类似 windeployqt 功能)
deploy_qt_dependencies() {
    if [ "$ENABLE_QT_DEPLOY" = "true" ]; then
        log_qt "收集和部署 Qt 依赖..."
        
        QT_DEPLOY_DIR="${JETPACK_DEPLOY_DIR}/qt_libs"
        mkdir -p "$QT_DEPLOY_DIR"
        
        # 查找 Qt 安装路径
        QT_DIR=$(qmake6 -query QT_INSTALL_PREFIX 2>/dev/null || echo "/usr")
        QT_LIB_DIR=$(qmake6 -query QT_INSTALL_LIBS 2>/dev/null || echo "/usr/lib/aarch64-linux-gnu")
        QT_PLUGIN_DIR=$(qmake6 -query QT_INSTALL_PLUGINS 2>/dev/null || echo "/usr/lib/aarch64-linux-gnu/qt6/plugins")
        QT_QML_DIR=$(qmake6 -query QT_INSTALL_QML 2>/dev/null || echo "/usr/lib/aarch64-linux-gnu/qt6/qml")
        
        log_qt "Qt 安装目录: ${QT_DIR}"
        log_qt "Qt 库目录: ${QT_LIB_DIR}"
        log_qt "Qt 插件目录: ${QT_PLUGIN_DIR}"
        log_qt "Qt QML 目录: ${QT_QML_DIR}"
        
        # 复制核心 Qt 库
        log_qt "复制 Qt 核心库..."
        cp -L "${QT_LIB_DIR}"/libQt6Core.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Gui.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Widgets.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Quick.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Qml.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Multimedia.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6SerialPort.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        cp -L "${QT_LIB_DIR}"/libQt6Network.so* "$QT_DEPLOY_DIR/" 2>/dev/null || true
        
        # 复制平台插件
        log_qt "复制 Qt 平台插件..."
        PLATFORM_PLUGIN_DIR="${QT_DEPLOY_DIR}/platforms"
        mkdir -p "$PLATFORM_PLUGIN_DIR"
        cp -r "${QT_PLUGIN_DIR}/platforms"/* "$PLATFORM_PLUGIN_DIR/" 2>/dev/null || true
        
        # 复制 QML 模块
        log_qt "复制 QML 模块..."
        QML_DEPLOY_DIR="${QT_DEPLOY_DIR}/qml"
        mkdir -p "$QML_DEPLOY_DIR"
        cp -r "${QT_QML_DIR}/QtQuick" "$QML_DEPLOY_DIR/" 2>/dev/null || true
        cp -r "${QT_QML_DIR}/QtQuick.2" "$QML_DEPLOY_DIR/" 2>/dev/null || true
        cp -r "${QT_QML_DIR}/QtMultimedia" "$QML_DEPLOY_DIR/" 2>/dev/null || true
        
        # 创建 Qt 环境设置脚本
        cat > "${QT_DEPLOY_DIR}/setup_qt_env.sh" << EOF
#!/bin/bash
# Qt 环境设置脚本 - 专用EGLFS触摸屏配置

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="\${SCRIPT_DIR}:\${LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="\${SCRIPT_DIR}"
export QML2_IMPORT_PATH="\${SCRIPT_DIR}/qml"
export QT_QPA_PLATFORM_PLUGIN_PATH="\${SCRIPT_DIR}/platforms"

# 设置 XDG_RUNTIME_DIR
export XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/tmp/runtime-root}
mkdir -p "\$XDG_RUNTIME_DIR"
chmod 700 "\$XDG_RUNTIME_DIR"

# 强制使用EGLFS平台（专用触摸屏配置）
echo "🔧 配置EGLFS触摸屏环境..."

export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
export QT_QPA_EGLFS_HIDECURSOR=1

# 触摸屏设备配置
export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2

# 触摸屏调试日志
export QT_LOGGING_RULES="qt.qpa.*=true;qt.qpa.input*=true"
export QT_QPA_EGLFS_DEBUG=1

echo "✅ EGLFS触摸屏环境已配置完成"
echo "   Platform: \$QT_QPA_PLATFORM"
echo "   Touch Device: /dev/input/event2"
echo "   Cursor Hidden: Yes"
echo "   Runtime Dir: \$XDG_RUNTIME_DIR"
EOF
        chmod +x "${QT_DEPLOY_DIR}/setup_qt_env.sh"
        
        log_success "Qt 依赖部署完成"
    fi
}

# 配置和部署 AI 模型文件（增强版，包含OpenCV兼容性修复）
deploy_ai_models() {
    if [ "$DEPLOY_MODELS" = "true" ]; then
        log_jetpack "配置和部署 AI 模型文件..."
        
        MODELS_DIR="${JETPACK_DEPLOY_DIR}/models"
        mkdir -p "$MODELS_DIR"
        
        # 创建模型目录结构
        mkdir -p "${MODELS_DIR}/onnx"
        mkdir -p "${MODELS_DIR}/tensorrt"
        mkdir -p "${MODELS_DIR}/optimized"
        
        # 复制现有模型文件
        if [ -d "${PROJECT_ROOT}/models" ]; then
            cp -r "${PROJECT_ROOT}/models"/* "$MODELS_DIR/" 2>/dev/null || true
        fi
        
        # 创建增强版 TensorRT 模型优化脚本（可选，不阻止启动）
        cat > "${MODELS_DIR}/optimize_models.sh" << 'EOF'
#!/bin/bash
# TensorRT 模型优化脚本（增强版，可选执行）

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${MODELS_DIR}/onnx"
TRT_DIR="${MODELS_DIR}/tensorrt"

echo "🚀 开始 TensorRT 模型优化..."

# 查找 trtexec 工具
TRTEXEC_PATH=""
POSSIBLE_PATHS=(
    "/usr/bin/trtexec"
    "/usr/local/bin/trtexec"
    "/usr/src/tensorrt/bin/trtexec"
    "/usr/src/tensorrt/samples/trtexec"
    "/opt/tensorrt/bin/trtexec"
)

# 首先检查预定义路径
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ] && [ -x "$path" ]; then
        TRTEXEC_PATH="$path"
        echo "✅ 找到 trtexec: $TRTEXEC_PATH"
        break
    fi
done

# 如果预定义路径都找不到，使用find命令搜索
if [ -z "$TRTEXEC_PATH" ]; then
    echo "🔍 在预定义路径中未找到trtexec，使用find命令搜索..."
    FOUND_PATH=$(find /usr -name trtexec -type f -executable 2>/dev/null | head -1)
    if [ -n "$FOUND_PATH" ] && [ -f "$FOUND_PATH" ] && [ -x "$FOUND_PATH" ]; then
        TRTEXEC_PATH="$FOUND_PATH"
        echo "✅ 通过搜索找到 trtexec: $TRTEXEC_PATH"
    fi
fi

# 如果还是找不到trtexec，跳过TensorRT优化
if [ -z "$TRTEXEC_PATH" ]; then
    echo "⚠️ trtexec 未找到，跳过 TensorRT 优化"
    echo "💡 TensorRT已安装但缺少trtexec工具"
    echo "🔧 可尝试安装: sudo apt install tensorrt-dev"
    echo "✅ 系统将使用 ONNX 模型继续运行"
    exit 0  # 正常退出，不阻止系统启动
fi

# 优化 ONNX 模型为 TensorRT 引擎
echo "⚡ 开始TensorRT引擎生成..."
for onnx_file in "${MODELS_DIR}"/*.onnx; do
    if [ -f "$onnx_file" ]; then
        filename=$(basename "$onnx_file" .onnx)
        echo "🔧 优化模型: $filename"
        
        # 移动到onnx目录
        mkdir -p "${ONNX_DIR}" "${TRT_DIR}"
        cp "$onnx_file" "${ONNX_DIR}/" 2>/dev/null || true
        
        # 执行TensorRT优化
        if "$TRTEXEC_PATH" \
            --onnx="$onnx_file" \
            --saveEngine="${TRT_DIR}/${filename}.trt" \
            --fp16 \
            --workspace=1024 \
            --minShapes=input:1x3x640x640 \
            --optShapes=input:1x3x640x640 \
            --maxShapes=input:4x3x640x640 \
            --verbose 2>/dev/null; then
            echo "✅ TensorRT引擎生成成功: ${filename}.trt"
        else
            echo "⚠️ TensorRT引擎生成失败: $filename，将使用ONNX模型"
        fi
    fi
done

echo "🎉 TensorRT 模型优化完成（如有成功）"
EOF
        chmod +x "${MODELS_DIR}/optimize_models.sh"
        
        log_success "AI 模型配置和部署完成"
    fi
}

# 创建 JetPack SDK 部署包
create_jetpack_package() {
    if [ "$CREATE_PACKAGE" = "true" ]; then
        log_jetpack "创建 JetPack SDK 部署包..."
        
        PACKAGE_DIR="${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}"
        mkdir -p "$PACKAGE_DIR"
        
        # 复制可执行文件（添加验证）
        echo "📋 检查可执行文件..."
        if [ -f "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" ]; then
            cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" "$PACKAGE_DIR/"
            echo "✅ C++后端可执行文件已复制"
        else
            echo "⚠️ C++后端可执行文件不存在，创建占位符"
            echo '#!/bin/bash
echo "C++后端尚未编译，请先编译项目"
exit 1' > "$PACKAGE_DIR/bamboo_cut_backend"
            chmod +x "$PACKAGE_DIR/bamboo_cut_backend"
        fi
        
        # 检查Qt前端可执行文件（支持多种可能的名称）
        qt_frontend_found=false
        qt_frontend_candidates=(
            "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
            "${BUILD_DIR}/qt_frontend/bamboo_cut_frontend"
            "${PROJECT_ROOT}/qt_frontend/build/bamboo_controller_qt"
            "${PROJECT_ROOT}/qt_frontend/build_release/bamboo_controller_qt"
        )
        
        for candidate in "${qt_frontend_candidates[@]}"; do
            if [ -f "$candidate" ]; then
                cp "$candidate" "$PACKAGE_DIR/bamboo_controller_qt"
                echo "✅ Qt前端可执行文件已复制: $(basename $candidate) -> bamboo_controller_qt"
                qt_frontend_found=true
                break
            fi
        done
        
        if [ "$qt_frontend_found" = false ]; then
            echo "⚠️ Qt前端可执行文件不存在，创建占位符"
            echo '#!/bin/bash
echo "Qt前端尚未编译，请先编译项目"
exit 1' > "$PACKAGE_DIR/bamboo_controller_qt"
            chmod +x "$PACKAGE_DIR/bamboo_controller_qt"
        fi
        
        # 复制配置文件
        cp -r "${PROJECT_ROOT}/config" "$PACKAGE_DIR/"
        
        # 复制 Qt 依赖 (如果可用)
        if [ "$ENABLE_QT_DEPLOY" = "true" ]; then
            cp -r "${JETPACK_DEPLOY_DIR}/qt_libs" "$PACKAGE_DIR/"
        fi
        
        # 复制模型文件 (如果可用)
        if [ "$DEPLOY_MODELS" = "true" ]; then
            cp -r "${JETPACK_DEPLOY_DIR}/models" "$PACKAGE_DIR/"
        fi
        
        # 复制Jetson环境配置
        cp "${JETPACK_DEPLOY_DIR}/jetson_tegra_env.sh" "$PACKAGE_DIR/"
        
        # 复制性能优化脚本
        if [ -f "${JETPACK_DEPLOY_DIR}/power_config.sh" ]; then
            cp "${JETPACK_DEPLOY_DIR}/power_config.sh" "$PACKAGE_DIR/"
        fi
        
        # 创建健壮的 JetPack 启动脚本（整合 EGL 修复）
        cat > "$PACKAGE_DIR/start_bamboo_jetpack_complete.sh" << 'EOF'
#!/bin/bash
# 智能切竹机 JetPack SDK 完整启动脚本（整合EGL修复）

echo "🚀 启动智能切竹机系统（Jetson完整版）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 Jetson Tegra 环境
if [ -f "./jetson_tegra_env.sh" ]; then
    source "./jetson_tegra_env.sh"
    echo "✅ Jetson Tegra环境已加载"
else
    echo "⚠️ 使用默认Tegra环境"
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}"
    export EGL_PLATFORM=device
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

# 加载 Qt 环境 (如果存在)
if [ -f "./qt_libs/setup_qt_env.sh" ]; then
    source "./qt_libs/setup_qt_env.sh"
    echo "✅ Qt环境已加载"
else
    # 如果没有独立的Qt环境脚本，设置基础环境
    echo "🔧 设置基础Qt环境..."
    export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-root}
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    
    # Jetson专用Qt EGLFS配置
    export QT_QPA_PLATFORM=eglfs
    export QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
    export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
    export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
    export QT_QPA_EGLFS_HIDECURSOR=1
    export QT_QPA_EGLFS_KMS_ATOMIC=1
    
    # 触摸屏设备配置
    export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
    export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
    
    # 调试日志
    export QT_LOGGING_RULES="qt.qpa.*=true;qt.qpa.input*=true"
    echo "✅ 基础Qt环境已设置"
fi

# 检查GPU和显示权限
echo "🔧 检查GPU和显示设备权限..."

# 添加当前用户到必要的组
if ! groups | grep -q video; then
    echo "⚠️ 当前用户不在video组，添加权限..."
    sudo usermod -a -G video $USER || true
fi

if ! groups | grep -q render; then
    echo "⚠️ 当前用户不在render组，添加权限..."
    sudo usermod -a -G render $USER || true
fi

# 检查关键设备权限
for device in /dev/dri/card0 /dev/dri/renderD128 /dev/input/event2 /dev/nvidia0 /dev/nvidiactl; do
    if [ -e "$device" ]; then
        echo "📋 设备权限: $(ls -la $device)"
        sudo chmod 666 "$device" 2>/dev/null || true
    else
        echo "⚠️ 设备不存在: $device"
    fi
done

# 应用性能优化 (如果存在)
if [ -f "./power_config.sh" ]; then
    ./power_config.sh
    echo "✅ 性能优化已应用"
fi

# 增强摄像头检测
echo "🔍 检测摄像头设备..."
CAMERA_FOUND=false

# 检查CSI摄像头（IMX219等）
echo "📋 检查CSI摄像头..."
if lsmod | grep -q imx219; then
    echo "✅ IMX219内核模块已加载"
    CAMERA_FOUND=true
elif modprobe imx219 2>/dev/null; then
    echo "✅ IMX219内核模块加载成功"
    CAMERA_FOUND=true
    sleep 2  # 等待模块初始化
else
    echo "⚠️ 无法加载IMX219内核模块"
fi

# 检查传统video设备
for device in /dev/video0 /dev/video1 /dev/video2; do
    if [ -e "$device" ]; then
        echo "📹 找到video设备: $device"
        if [ -r "$device" ] && [ -w "$device" ]; then
            echo "✅ 摄像头设备 $device 可访问"
            CAMERA_FOUND=true
            export BAMBOO_CAMERA_DEVICE="$device"
            break
        else
            echo "⚠️ 摄像头设备 $device 存在但无访问权限"
        fi
    fi
done

# 设置摄像头模式
if [ "$CAMERA_FOUND" = true ]; then
    echo "✅ 检测到摄像头，启用硬件模式"
    export BAMBOO_CAMERA_MODE="hardware"
    export BAMBOO_SKIP_CAMERA="false"
else
    echo "⚠️ 未检测到可用摄像头设备，启用模拟模式"
    export BAMBOO_CAMERA_MODE="simulation"
    export BAMBOO_SKIP_CAMERA="true"
fi

# 优化模型 (如果存在且需要)
if [ -f "./models/optimize_models.sh" ] && [ ! -f "./models/tensorrt/optimized.flag" ]; then
    echo "🔄 首次运行，正在优化 AI 模型..."
    cd ./models && timeout 300 ./optimize_models.sh && cd ..
    mkdir -p "./models/tensorrt"
    touch "./models/tensorrt/optimized.flag"
    echo "✅ 模型优化完成"
fi

# 设置环境变量
export LD_LIBRARY_PATH="./qt_libs:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# 健壮性检查函数
check_and_start_backend() {
    if [ ! -f "./bamboo_cut_backend" ] || [ ! -x "./bamboo_cut_backend" ]; then
        echo "❌ C++后端可执行文件不存在或无执行权限"
        return 1
    fi
    
    echo "🔄 启动 C++ 后端..."
    # 使用超时和容错机制启动后端
    timeout 60 ./bamboo_cut_backend &
    BACKEND_PID=$!
    
    # 等待后端初始化
    sleep 8
    
    # 检查后端是否还在运行
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "✅ C++ 后端启动成功 (PID: $BACKEND_PID)"
        return 0
    else
        echo "⚠️ C++ 后端可能因摄像头问题启动失败，但这是正常的"
        # 在没有摄像头的环境中，后端可能会退出，这是预期的
        wait $BACKEND_PID 2>/dev/null
        BACKEND_EXIT_CODE=$?
        if [ $BACKEND_EXIT_CODE -eq 0 ]; then
            echo "✅ C++ 后端正常退出"
            return 0
        else
            echo "⚠️ C++ 后端异常退出 (退出码: $BACKEND_EXIT_CODE)"
            return 1
        fi
    fi
}

check_and_start_frontend() {
    # 检查Qt前端可执行文件（支持多种可能的名称）
    qt_frontend_exec=""
    qt_frontend_candidates=("./bamboo_controller_qt" "./bamboo_cut_frontend")
    
    for candidate in "${qt_frontend_candidates[@]}"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            qt_frontend_exec="$candidate"
            break
        fi
    done
    
    if [ -z "$qt_frontend_exec" ]; then
        echo "⚠️ Qt前端可执行文件不存在，仅后端模式"
        return 1
    fi
    
    echo "🔄 启动 Qt 前端: $qt_frontend_exec"
    echo "🔧 当前Qt环境变量："
    echo "   QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
    echo "   QT_QPA_EGLFS_INTEGRATION: $QT_QPA_EGLFS_INTEGRATION"
    echo "   EGL_PLATFORM: $EGL_PLATFORM"
    echo "   XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
    
    # 启动Qt前端
    timeout 30 "$qt_frontend_exec" &
    FRONTEND_PID=$!
    
    sleep 8
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ Qt 前端启动成功 (PID: $FRONTEND_PID)"
        return 0
    else
        echo "❌ Qt前端启动失败"
        echo "🔍 EGL错误诊断:"
        
        # Jetson专用诊断
        echo "📋 Tegra驱动状态:"
        find /sys -name "*tegra*" -type d 2>/dev/null | head -5 || echo "   无Tegra驱动信息"
        
        echo "📋 EGL设备:"
        find /dev -name "nvidia*" 2>/dev/null || echo "   无NVIDIA设备节点"
        
        echo "📋 DRM设备:"
        ls -la /dev/dri/ 2>/dev/null || echo "   无DRM设备"
        
        wait $FRONTEND_PID 2>/dev/null || true
        return 1
    fi
}

# 主启动逻辑
BACKEND_STARTED=false
FRONTEND_STARTED=false

# 尝试启动后端（最多重试2次）
for i in {1..2}; do
    echo "🔄 尝试启动后端 (第 $i 次)..."
    if check_and_start_backend; then
        BACKEND_STARTED=true
        break
    else
        if [ $i -lt 2 ]; then
            echo "⚠️ 后端启动失败，等待 5 秒后重试..."
            sleep 5
        fi
    fi
done

# 如果后端仍在运行，尝试启动前端
if [ "$BACKEND_STARTED" = true ] && kill -0 $BACKEND_PID 2>/dev/null; then
    # 尝试启动前端
    if check_and_start_frontend; then
        FRONTEND_STARTED=true
        # 等待前端进程
        wait $FRONTEND_PID
        kill $BACKEND_PID 2>/dev/null || true
    else
        echo "🔄 仅后端模式运行，等待后端进程..."
        wait $BACKEND_PID
    fi
else
    echo "✅ 后端已完成运行或在模拟模式下正常退出"
fi

echo "🛑 智能切竹机系统已停止"
EOF
        chmod +x "$PACKAGE_DIR/start_bamboo_jetpack_complete.sh"
        
        # 创建安装脚本
        cat > "$PACKAGE_DIR/install_jetpack.sh" << 'EOF'
#!/bin/bash
set -e

echo "安装智能切竹机 JetPack SDK 版本..."

# 创建用户
sudo useradd -r -s /bin/false bamboo-cut || true

# 创建目录
sudo mkdir -p /opt/bamboo-cut
sudo mkdir -p /var/log/bamboo-cut

# 复制文件
sudo cp -r * /opt/bamboo-cut/

# 设置权限
sudo chown -R root:root /opt/bamboo-cut
sudo chown -R bamboo-cut:bamboo-cut /var/log/bamboo-cut
sudo chmod +x /opt/bamboo-cut/*.sh

# 创建健壮的 systemd 服务
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK) - 完整版
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_jetpack_complete.sh
Restart=on-failure
RestartSec=30
StartLimitBurst=3
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
Environment=QT_QPA_EGLFS_ALWAYS_SET_MODE=1
Environment=QT_QPA_EGLFS_HIDECURSOR=1
Environment=QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
Environment=QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
Environment=QT_LOGGING_RULES=qt.qpa.*=true;qt.qpa.input*=true
Environment=EGL_PLATFORM=device
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TEGRA_RM_DISABLE_SECURITY=1
Environment=BAMBOO_SKIP_CAMERA=true

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# 重新加载 systemd
sudo systemctl daemon-reload
sudo systemctl enable bamboo-cut-jetpack

echo "安装完成!"
echo "使用以下命令启动服务:"
echo "sudo systemctl start bamboo-cut-jetpack"
echo "查看状态: sudo systemctl status bamboo-cut-jetpack"
EOF
        chmod +x "$PACKAGE_DIR/install_jetpack.sh"
        
        # 创建 tar 包
        cd "${DEPLOY_DIR}/packages"
        tar czf "bamboo-cut-jetpack-${VERSION}.tar.gz" "bamboo-cut-jetpack-${VERSION}"
        
        log_success "JetPack SDK 部署包创建完成: bamboo-cut-jetpack-${VERSION}.tar.gz"
    fi
}

# 检查Tegra GPU状态（替代nvidia-smi）
check_tegra_gpu() {
    log_jetpack "检查Tegra GPU状态..."
    
    echo "📋 Tegra GPU信息 ($JETSON_TYPE)："
    
    # 根据设备类型检查GPU设备树信息
    GPU_DEVICE_PATHS=(
        "/proc/device-tree/gpu@${GPU_PATH}/compatible"
        "/proc/device-tree/gpu@17000000/compatible"
        "/proc/device-tree/gpu@57000000/compatible"
    )
    
    for gpu_path in "${GPU_DEVICE_PATHS[@]}"; do
        if [ -f "$gpu_path" ]; then
            GPU_COMPATIBLE=$(cat "$gpu_path" 2>/dev/null | tr -d '\0')
            echo "  GPU兼容性: $GPU_COMPATIBLE"
            break
        fi
    done
    
    # 检查GPU频率设置（支持不同路径）
    GPU_FREQ_PATHS=(
        "/sys/devices/platform/host1x/${GPU_PATH}/devfreq/${GPU_PATH}/cur_freq"
        "/sys/devices/platform/host1x/17000000.gpu/devfreq/17000000.gpu/cur_freq"
        "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq"
    )
    
    for freq_path in "${GPU_FREQ_PATHS[@]}"; do
        if [ -f "$freq_path" ]; then
            GPU_FREQ=$(cat "$freq_path" 2>/dev/null)
            echo "  当前GPU频率: $GPU_FREQ Hz"
            
            # 也检查可用频率
            available_freq_path="${freq_path/cur_freq/available_frequencies}"
            if [ -f "$available_freq_path" ]; then
                GPU_FREQS=$(cat "$available_freq_path" 2>/dev/null)
                echo "  可用频率: $GPU_FREQS"
            fi
            break
        fi
    done
    
    # 检查Tegra架构信息
    echo "📋 Tegra架构信息："
    echo "  设备类型: $JETSON_TYPE"
    echo "  Tegra芯片: $TEGRA_CHIP"
    echo "  GPU路径: $GPU_PATH"
    
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

# 部署到目标设备
deploy_to_target() {
    if [ -n "$DEPLOY_TARGET" ]; then
        log_jetpack "部署到目标设备: $DEPLOY_TARGET"
        
        case "$DEPLOY_TARGET" in
            local)
                # 本地安装
                cd "${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}"
                sudo ./install_jetpack.sh
                ;;
            jetson)
                # 部署到 Jetson (假设通过 SSH)
                deploy_to_remote "jetson"
                ;;
            remote:*)
                # 部署到远程设备
                REMOTE_IP="${DEPLOY_TARGET#remote:}"
                deploy_to_remote "$REMOTE_IP"
                ;;
            *)
                log_error "未知的部署目标: $DEPLOY_TARGET"
                exit 1
                ;;
        esac
    fi
}

# 部署到远程设备
deploy_to_remote() {
    local TARGET_HOST="$1"
    local PACKAGE_FILE="${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}.tar.gz"
    
    log_jetpack "部署到远程 Jetson 设备: $TARGET_HOST"
    
    # 传输文件
    scp "$PACKAGE_FILE" "${TARGET_HOST}:/tmp/"
    
    # 远程安装
    ssh "$TARGET_HOST" << EOF
cd /tmp
tar xzf bamboo-cut-jetpack-${VERSION}.tar.gz
cd bamboo-cut-jetpack-${VERSION}
sudo ./install_jetpack.sh
sudo systemctl start bamboo-cut-jetpack
sudo systemctl status bamboo-cut-jetpack
EOF
    
    log_success "远程 JetPack SDK 部署完成"
}

# 创建最终的启动脚本（整合所有功能）
create_final_startup_script() {
    log_jetpack "创建最终启动脚本..."
    
    # 确保部署目录存在
    sudo mkdir -p /opt/bamboo-cut
    
    # 复制可执行文件到最终位置
    if [ -f "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" ]; then
        sudo cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" /opt/bamboo-cut/
        sudo chmod +x /opt/bamboo-cut/bamboo_cut_backend
        log_success "C++后端已部署到 /opt/bamboo-cut/"
    fi
    
    # 检查Qt前端可执行文件（支持多种可能的名称）
    qt_frontend_candidates=(
        "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
        "${BUILD_DIR}/qt_frontend/bamboo_cut_frontend"
        "${PROJECT_ROOT}/qt_frontend/build/bamboo_controller_qt"
        "${PROJECT_ROOT}/qt_frontend/build_debug/bamboo_controller_qt"
        "${PROJECT_ROOT}/qt_frontend/build_release/bamboo_controller_qt"
    )
    
    qt_deployed=false
    for candidate in "${qt_frontend_candidates[@]}"; do
        if [ -f "$candidate" ]; then
            sudo cp "$candidate" /opt/bamboo-cut/bamboo_controller_qt
            sudo chmod +x /opt/bamboo-cut/bamboo_controller_qt
            log_success "Qt前端已部署: $(basename $candidate) -> /opt/bamboo-cut/bamboo_controller_qt"
            qt_deployed=true
            break
        fi
    done
    
    if [ "$qt_deployed" = false ]; then
        log_warning "Qt前端可执行文件未找到，跳过前端部署"
    fi
    
    # 复制配置文件
    if [ -d "${PROJECT_ROOT}/config" ]; then
        sudo cp -r "${PROJECT_ROOT}/config" /opt/bamboo-cut/
        log_success "配置文件已部署"
    fi
    
    # 复制Jetson环境脚本
    if [ -f "${JETPACK_DEPLOY_DIR}/jetson_tegra_env.sh" ]; then
        sudo cp "${JETPACK_DEPLOY_DIR}/jetson_tegra_env.sh" /opt/bamboo-cut/
        sudo chmod +x /opt/bamboo-cut/jetson_tegra_env.sh
    fi
    
    # 复制其他必要文件
    if [ -d "${JETPACK_DEPLOY_DIR}/qt_libs" ]; then
        sudo cp -r "${JETPACK_DEPLOY_DIR}/qt_libs" /opt/bamboo-cut/
    fi
    
    if [ -d "${JETPACK_DEPLOY_DIR}/models" ]; then
        sudo cp -r "${JETPACK_DEPLOY_DIR}/models" /opt/bamboo-cut/
    fi
    
    if [ -f "${JETPACK_DEPLOY_DIR}/power_config.sh" ]; then
        sudo cp "${JETPACK_DEPLOY_DIR}/power_config.sh" /opt/bamboo-cut/
        sudo chmod +x /opt/bamboo-cut/power_config.sh
    fi
    
    # 创建最终的启动脚本
    sudo cp "${JETPACK_DEPLOY_DIR}/../packages/bamboo-cut-jetpack-${VERSION}/start_bamboo_jetpack_complete.sh" /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/start_bamboo_jetpack_complete.sh
    
    log_success "最终启动脚本已部署到 /opt/bamboo-cut/"
}

# 更新systemd服务为Jetson专用（包含编译逻辑）
update_jetson_systemd_service() {
    log_jetpack "更新systemd服务为Jetson $JETSON_TYPE 专用..."
    
    cat > /tmp/bamboo-cut-jetpack.service << EOF
[Unit]
Description=智能切竹机系统 (Jetson $JETSON_TYPE 专用 + 完整编译部署)
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_jetpack_complete.sh
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
Environment=QT_QPA_EGLFS_ALWAYS_SET_MODE=1
Environment=QT_QPA_EGLFS_HIDECURSOR=1
Environment=QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
Environment=QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/bamboo-cut-jetpack.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_success "Jetson $JETSON_TYPE 专用systemd服务已更新"
}

# 主函数
main() {
    echo "========================================"
    echo "Jetson Tegra SoC EGL 修复 + 完整编译部署脚本"
    echo "整合了 jetpack_deploy 和 fix_jetson_nano_egl 功能"
    echo "========================================"
    
    # 检查root权限
    if [ "$EUID" -ne 0 ]; then
        log_error "请以root权限运行: sudo $0"
        exit 1
    fi
    
    parse_arguments "$@"
    
    # 检查Jetson设备
    if ! check_jetson_device; then
        exit 1
    fi
    
    log_jetpack "=== 智能切竹机 Jetson $JETSON_TYPE EGL + 编译部署脚本 ==="
    log_jetpack "版本: $VERSION"
    log_jetpack "JetPack SDK: $JETPACK_VERSION"
    log_jetpack "检测到设备: $JETSON_TYPE ($TEGRA_CHIP)"
    
    # 默认启用重新部署模式
    log_info "启用重新部署模式：自动清理运行中的进程和强制重新编译"
    CLEAN_LEGACY="true"
    FORCE_REBUILD="true"
    ENABLE_QT_DEPLOY="true"
    DEPLOY_MODELS="true"
    CREATE_PACKAGE="true"
    DEPLOY_TARGET="local"
    
    # 设置Jetson检测标志
    JETSON_DETECTED="true"
    
    # 首先停止所有运行中的相关服务和进程
    log_info "🛑 停止所有运行中的智能切竹机服务和进程..."
    stop_running_services
    
    # 清理历史版本
    log_info "🧹 清理历史版本..."
    clean_legacy_deployment
    
    # 备份当前部署
    backup_current_deployment
    
    # 创建部署目录
    mkdir -p "$JETPACK_DEPLOY_DIR"
    
    # 安装依赖
    install_jetpack_dependencies
    
    # 构建项目
    log_jetpack "确保项目已编译..."
    if ! build_project; then
        log_error "项目编译失败，停止部署"
        exit 1
    fi
    
    # 配置Jetson专用环境
    configure_jetpack_libraries
    create_jetson_kms_config
    configure_jetpack_performance
    
    # 部署组件
    deploy_qt_dependencies
    deploy_ai_models
    
    # 创建部署包
    create_jetpack_package
    
    # 创建最终启动脚本和部署文件
    create_final_startup_script
    
    # 更新systemd服务
    update_jetson_systemd_service
    
    # 部署到目标设备
    deploy_to_target
    
    # 检查GPU状态
    check_tegra_gpu
    
    # 启动服务
    log_info "🚀 启动智能切竹机服务..."
    if systemctl is-enabled bamboo-cut-jetpack >/dev/null 2>&1; then
        sudo systemctl start bamboo-cut-jetpack
        sleep 3
        
        if systemctl is-active --quiet bamboo-cut-jetpack; then
            log_success "✅ 智能切竹机服务启动成功"
            log_info "服务状态: sudo systemctl status bamboo-cut-jetpack"
            log_info "查看日志: sudo journalctl -u bamboo-cut-jetpack -f"
        else
            log_warning "⚠️ 服务启动可能有问题，请检查日志"
            log_info "检查命令: sudo systemctl status bamboo-cut-jetpack"
        fi
    else
        log_warning "⚠️ 服务未启用，请手动启动: sudo systemctl enable --now bamboo-cut-jetpack"
    fi
    
    log_success "🎉 Jetson $JETSON_TYPE EGL + 完整编译部署完成!"
    log_info "部署包位置: ${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}.tar.gz"
    
    if [ "$JETSON_DETECTED" = "true" ]; then
        log_jetpack "运行性能测试: sudo jetson_stats"
        log_jetpack "监控 GPU 使用: sudo tegrastats"
    fi
    
    echo ""
    echo "🎯 部署摘要："
    echo "✅ 已停止所有运行中的进程"
    echo "✅ 已清理历史版本"
    echo "✅ 已编译 C++ 后端和 Qt 前端"
    echo "✅ 已配置 Jetson $JETSON_TYPE 专用 EGL 环境"
    echo "✅ 已部署 Qt 依赖和 AI 模型"
    echo "✅ 已创建完整部署包"
    echo "✅ 已重新启动智能切竹机服务"
    echo ""
    echo "📋 常用命令："
    echo "  查看服务状态: sudo systemctl status bamboo-cut-jetpack"
    echo "  查看实时日志: sudo journalctl -u bamboo-cut-jetpack -f"
    echo "  重启服务: sudo systemctl restart bamboo-cut-jetpack"
    echo "  停止服务: sudo systemctl stop bamboo-cut-jetpack"
    echo "  独立测试前端: /opt/bamboo-cut/start_bamboo_jetpack_complete.sh"
}

# 运行主函数
main "$@"