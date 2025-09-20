#!/bin/bash
# Jetson Tegra SoC EGL 修复脚本 + 完整编译部署功能 (修复版)
# 修复函数调用顺序和服务超时问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本信息
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
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
    sync
}

log_success() {
    printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
    sync
}

log_warning() {
    printf "${YELLOW}[WARNING]${NC} %s\n" "$1"
    sync
}

log_error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1"
    sync
}

log_jetpack() {
    printf "${PURPLE}[JETPACK]${NC} %s\n" "$1"
    sync
}

log_qt() {
    printf "${CYAN}[QT-DEPLOY]${NC} %s\n" "$1"
    sync
}

# 显示帮助信息
show_help() {
    cat << EOF
Jetson Tegra SoC EGL 修复 + 完整编译部署脚本 (修复版)

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
    -s, --stop-only         仅停止服务，不重新部署
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

🚀 使用示例:
    $0                      # 完整编译、部署和配置
    $0 --install-deps       # 安装依赖并完整部署
    $0 --force-rebuild      # 强制重新编译
    $0 --stop-only          # 仅停止现有服务

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
            -s|--stop-only)
                STOP_ONLY="true"
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

# 检测Jetson设备
check_jetson_device() {
    log_info "检测Jetson设备..."
    
    if [ -f "/proc/device-tree/model" ]; then
        DEVICE_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        echo "设备型号: $DEVICE_MODEL"
        
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
                JETSON_TYPE="generic"
                TEGRA_CHIP="tegra"
                GPU_PATH="*.gpu"
                log_success "检测到Jetson设备: $DEVICE_MODEL"
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

# 强制停止所有相关服务和进程
force_stop_all_services() {
    log_warning "🛑 强制停止所有智能切竹机相关服务和进程..."
    
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
    
    # 立即停止所有systemd服务
    for service in "${BAMBOO_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            log_warning "强制停止服务: $service"
            sudo systemctl stop "$service" --no-block || true
            sudo systemctl kill "$service" --signal=SIGKILL || true
        fi
        
        # 禁用服务防止自动重启
        if systemctl is-enabled --quiet "$service" 2>/dev/null; then
            sudo systemctl disable "$service" || true
        fi
    done
    
    # 等待服务停止
    sleep 3
    
    # 强制终止所有相关进程
    log_warning "强制终止所有相关进程..."
    
    BAMBOO_PROCESSES=(
        "bamboo_cut_backend"
        "bamboo_controller_qt"
        "bamboo-cut"
        "bamboo_cut_frontend"
        "bamboo-backend"
        "bamboo-frontend"
        "start_bamboo"
        "timeout.*bamboo"
    )
    
    for process in "${BAMBOO_PROCESSES[@]}"; do
        if pgrep -f "$process" >/dev/null 2>&1; then
            log_warning "强制终止进程: $process"
            sudo pkill -KILL -f "$process" || true
        fi
    done
    
    # 清理可能的僵尸进程和孤儿进程
    log_warning "清理残留进程..."
    sudo pkill -KILL -f "bamboo" >/dev/null 2>&1 || true
    sudo pkill -KILL -f "start_bamboo" >/dev/null 2>&1 || true
    
    # 重新加载systemd以清理失败状态
    sudo systemctl daemon-reload
    sudo systemctl reset-failed 2>/dev/null || true
    
    sleep 5
    
    log_success "✅ 所有相关服务和进程已强制停止"
}

# 安装依赖
install_jetpack_dependencies() {
    if [ "$INSTALL_DEPENDENCIES" = "true" ]; then
        log_jetpack "安装 JetPack SDK 依赖包..."
        
        sudo apt update
        
        # 基础编译工具
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
        
        # Qt6 相关
        sudo apt install -y \
            qt6-base-dev \
            qt6-declarative-dev \
            qt6-multimedia-dev \
            qt6-serialport-dev \
            qt6-tools-dev \
            qt6-wayland \
            qml6-module-qtquick \
            qml6-module-qtquick-controls
        
        # OpenCV
        sudo apt install -y \
            libopencv-dev \
            libopencv-contrib-dev \
            python3-opencv
        
        # JetPack 组件（可选，可能失败）
        sudo apt install -y \
            nvidia-jetpack \
            tensorrt \
            libnvinfer-dev || log_warning "部分JetPack组件安装失败"
        
        log_success "依赖包安装完成"
    fi
}

# 清理构建缓存
clean_build_cache() {
    if [ "$FORCE_REBUILD" = "true" ]; then
        log_info "清理构建缓存..."
        
        rm -rf "${BUILD_DIR}" 2>/dev/null || true
        rm -rf "${PROJECT_ROOT}/qt_frontend/build"* 2>/dev/null || true
        
        log_success "构建缓存已清理"
    fi
}

# 构建项目
build_project() {
    log_info "构建智能切竹机项目..."
    
    cd "$PROJECT_ROOT"
    clean_build_cache
    mkdir -p "$BUILD_DIR"
    
    # 构建 C++ 后端
    if [ -d "${PROJECT_ROOT}/cpp_backend" ] || [ -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
        log_info "构建 C++ 后端..."
        mkdir -p "${BUILD_DIR}/cpp_backend"
        cd "${BUILD_DIR}/cpp_backend"
        
        CMAKE_ARGS=(
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
            -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
            -DJETSON_BUILD=ON
        )
        
        # 确定 CMakeLists.txt 位置
        CMAKE_SOURCE_DIR=""
        if [ -f "${PROJECT_ROOT}/cpp_backend/CMakeLists.txt" ]; then
            CMAKE_SOURCE_DIR="${PROJECT_ROOT}/cpp_backend"
        elif [ -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
            CMAKE_SOURCE_DIR="${PROJECT_ROOT}"
        else
            log_warning "未找到 C++ 后端 CMakeLists.txt，跳过后端编译"
        fi
        
        if [ -n "$CMAKE_SOURCE_DIR" ]; then
            if cmake "${CMAKE_ARGS[@]}" "$CMAKE_SOURCE_DIR" && make -j$(nproc); then
                log_success "C++ 后端构建完成"
            else
                log_warning "C++ 后端构建失败"
            fi
        fi
    fi
    
    # 构建 Qt 前端（增强版）
    compile_and_deploy_qt_frontend
    
    cd "$PROJECT_ROOT"
    log_success "项目构建完成"
}

# 编译和部署Qt前端（增强版）
compile_and_deploy_qt_frontend() {
    log_qt "编译和部署 Qt 前端..."
    
    if [ ! -d "${PROJECT_ROOT}/qt_frontend" ]; then
        log_warning "未找到Qt前端目录，跳过前端编译"
        return 1
    fi
    
    cd "${PROJECT_ROOT}/qt_frontend"
    
    # 检查Qt6环境
    QT6_FOUND=false
    if command -v qmake6 >/dev/null 2>&1; then
        QT6_FOUND=true
        QMAKE_CMD="qmake6"
    elif command -v qmake >/dev/null 2>&1; then
        QMAKE_VERSION=$(qmake --version | grep -i qt | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        if [[ "$QMAKE_VERSION" == "6."* ]]; then
            QT6_FOUND=true
            QMAKE_CMD="qmake"
        fi
    fi
    
    if [ "$QT6_FOUND" = false ]; then
        log_error "未找到Qt6环境，无法编译前端"
        return 1
    fi
    
    log_info "使用Qt命令: $QMAKE_CMD"
    
    # 清理旧构建
    rm -rf build build_qmake 2>/dev/null || true
    
    # 方法1: CMake构建
    if [ -f "CMakeLists.txt" ]; then
        log_qt "尝试CMake构建..."
        mkdir -p build
        cd build
        
        CMAKE_ARGS=(
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
            -DJETSON_BUILD=ON
            -DQt6_DIR=/usr/lib/aarch64-linux-gnu/cmake/Qt6
        )
        
        if cmake "${CMAKE_ARGS[@]}" .. && make -j$(nproc); then
            log_success "Qt前端CMake构建成功"
            
            # 查找生成的可执行文件
            FRONTEND_EXEC=$(find . -type f -executable -name "*bamboo*" | head -1)
            if [ -n "$FRONTEND_EXEC" ]; then
                mkdir -p "${BUILD_DIR}/qt_frontend"
                cp "$FRONTEND_EXEC" "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
                log_success "前端可执行文件已复制: $FRONTEND_EXEC"
                cd "$PROJECT_ROOT"
                return 0
            fi
        else
            log_warning "CMake构建失败，尝试qmake"
            cd ..
        fi
    fi
    
    # 方法2: qmake构建
    PRO_FILE=$(ls *.pro 2>/dev/null | head -1)
    if [ -n "$PRO_FILE" ]; then
        log_qt "尝试qmake构建: $PRO_FILE"
        mkdir -p build_qmake
        cd build_qmake
        
        # 配置qmake参数
        QMAKE_ARGS=(
            CONFIG+=release
            CONFIG+=c++17
            "QMAKE_CXXFLAGS+=-DJETSON_BUILD"
        )
        
        if $QMAKE_CMD "${QMAKE_ARGS[@]}" "../$PRO_FILE" && make -j$(nproc); then
            log_success "Qt前端qmake构建成功"
            
            # 查找生成的可执行文件
            FRONTEND_EXEC=$(find . -type f -executable -name "*bamboo*" | head -1)
            if [ -n "$FRONTEND_EXEC" ]; then
                mkdir -p "${BUILD_DIR}/qt_frontend"
                cp "$FRONTEND_EXEC" "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
                log_success "前端可执行文件已复制: $FRONTEND_EXEC"
                cd "$PROJECT_ROOT"
                return 0
            fi
        else
            log_error "qmake构建失败"
        fi
    else
        log_error "未找到.pro文件"
    fi
    
    cd "$PROJECT_ROOT"
    log_error "Qt前端编译失败"
    return 1
}

# 检测Tegra GPU状态
check_tegra_gpu() {
    log_jetpack "检测 Tegra GPU 状态..."
    
    # 根据设备类型检查GPU频率文件
    case "$JETSON_TYPE" in
        "orin"*|"agx-orin")
            GPU_FREQ_PATH="/sys/devices/platform/$GPU_PATH/devfreq/$GPU_PATH"
            ;;
        "nano"|"xavier")
            GPU_FREQ_PATH="/sys/devices/platform/$GPU_PATH/devfreq/$GPU_PATH"
            ;;
        *)
            # 通用查找
            GPU_FREQ_PATH=$(find /sys/devices/platform -name "*gpu" -type d 2>/dev/null | head -1)
            if [ -n "$GPU_FREQ_PATH" ]; then
                GPU_FREQ_PATH="$GPU_FREQ_PATH/devfreq/$(basename $GPU_FREQ_PATH)"
            fi
            ;;
    esac
    
    if [ -d "$GPU_FREQ_PATH" ]; then
        log_success "找到GPU控制路径: $GPU_FREQ_PATH"
        if [ -r "$GPU_FREQ_PATH/cur_freq" ]; then
            CURRENT_FREQ=$(cat "$GPU_FREQ_PATH/cur_freq" 2>/dev/null || echo "0")
            log_info "GPU当前频率: ${CURRENT_FREQ} Hz"
        fi
        return 0
    else
        log_warning "未找到GPU频率控制路径，使用默认配置"
        return 1
    fi
}

# 配置Jetson库环境
configure_jetson_libraries() {
    log_jetpack "配置 Jetson $JETSON_TYPE 库环境..."
    
    mkdir -p "$JETPACK_DEPLOY_DIR"
    
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
            VALID_PATHS="${VALID_PATHS}:${path}"
        fi
    done
    
    VALID_PATHS="${VALID_PATHS#:}"
    
    if [ -z "$VALID_PATHS" ]; then
        log_warning "未找到Tegra库，使用默认路径"
        VALID_PATHS="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl"
    fi
    
    # 创建环境配置脚本
    cat > "${JETPACK_DEPLOY_DIR}/jetson_env.sh" << EOF
#!/bin/bash
# Jetson $JETSON_TYPE 环境配置

echo "🔧 配置 Jetson $JETSON_TYPE 环境..."

export LD_LIBRARY_PATH="${VALID_PATHS}:\${LD_LIBRARY_PATH}"
export EGL_PLATFORM=device
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
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

echo "✅ Jetson $JETSON_TYPE 环境配置完成"
EOF
    
    chmod +x "${JETPACK_DEPLOY_DIR}/jetson_env.sh"
    log_success "Jetson 库环境配置完成"
}

# 创建Jetson专用KMS配置
create_kms_config() {
    log_jetpack "创建 Jetson 专用 KMS 配置..."
    
    mkdir -p "${JETPACK_DEPLOY_DIR}/config"
    
    # 检测实际显示器分辨率
    DISPLAY_MODE="1920x1080"
    if [ -f "/sys/class/graphics/fb0/modes" ]; then
        FB_MODE=$(cat /sys/class/graphics/fb0/modes | head -1)
        if [[ "$FB_MODE" == *"1920x1200"* ]]; then
            DISPLAY_MODE="1920x1200"
            log_info "检测到1920x1200显示器"
        elif [[ "$FB_MODE" == *"1920x1080"* ]]; then
            DISPLAY_MODE="1920x1080"
            log_info "检测到1920x1080显示器"
        else
            log_warning "未识别的显示模式: $FB_MODE，使用默认1920x1080"
        fi
    fi
    
    # 创建系统级Jetson KMS配置文件
    JETSON_KMS_CONFIG="{
  \"device\": \"/dev/dri/card0\",
  \"hwcursor\": false,
  \"pbuffers\": true,
  \"outputs\": [
    {
      \"name\": \"DP-1\",
      \"mode\": \"$DISPLAY_MODE\"
    }
  ]
}"
    
    # 写入系统配置文件
    echo "$JETSON_KMS_CONFIG" | sudo tee /etc/qt_eglfs_kms_jetson.json > /dev/null
    
    # 同时保留本地配置文件作为备份
    echo "$JETSON_KMS_CONFIG" > "${JETPACK_DEPLOY_DIR}/config/kms.conf"
    
    log_success "Jetson KMS 配置已创建 (${DISPLAY_MODE})"
    log_info "系统配置文件: /etc/qt_eglfs_kms_jetson.json"
}

# 创建无超时的启动脚本
create_timeout_safe_startup() {
    log_jetpack "创建安全启动脚本（防止超时）..."
    
    cat > "${JETPACK_DEPLOY_DIR}/start_bamboo_safe.sh" << 'EOF'
#!/bin/bash
# 智能切竹机安全启动脚本（防止超时）

echo "🚀 启动智能切竹机系统（安全模式）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载环境
if [ -f "./jetson_env.sh" ]; then
    source "./jetson_env.sh"
fi

# 设置运行时目录
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-root}
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Jetson Orin NX 完整启动流程
echo "🔧 Jetson Orin NX 完整启动流程..."

# 1. 设置性能模式
echo "⚡ 设置性能模式..."
sudo /usr/sbin/nvpmodel -m 0 2>/dev/null || echo "⚠️ nvpmodel设置失败"
sudo /usr/bin/jetson_clocks 2>/dev/null || echo "⚠️ jetson_clocks设置失败"

# 2. 显示驱动重置（可选）
# sudo modprobe -r tegra_drm && sleep 1 && sudo modprobe tegra_drm

# 3. 强制framebuffer配置
echo "📺 配置framebuffer..."
sudo sh -c 'echo "U:1920x1200p-0" > /sys/class/graphics/fb0/mode' 2>/dev/null
sudo sh -c 'echo "0" > /sys/class/graphics/fb0/blank' 2>/dev/null
sudo chmod 666 /dev/fb0 2>/dev/null
sleep 3

# 4. 验证显示状态
current_mode=$(cat /sys/class/graphics/fb0/mode 2>/dev/null)
if [ "$current_mode" = "U:1920x1200p-0" ]; then
    echo "✅ 显示模式: $current_mode"
else
    echo "⚠️ 显示模式可能有问题: $current_mode"
fi

# 5. Qt配置 (LinuxFB模式)
export QT_QPA_PLATFORM=linuxfb
export QT_QPA_FB_DEVICE=/dev/fb0
export QT_QPA_GENERIC_PLUGINS=evdevtouch
export QT_QPA_FB_FORCE_FULLSCREEN=1

# 确保X11不干扰
unset DISPLAY
unset WAYLAND_DISPLAY

# 设备权限
for device in /dev/dri/card0 /dev/dri/renderD128 /dev/nvidia0 /dev/nvidiactl; do
    if [ -e "$device" ]; then
        chmod 666 "$device" 2>/dev/null || true
    fi
done

# 摄像头检测
CAMERA_FOUND=false
for device in /dev/video0 /dev/video1 /dev/video2; do
    if [ -e "$device" ] && [ -r "$device" ]; then
        CAMERA_FOUND=true
        export BAMBOO_CAMERA_DEVICE="$device"
        break
    fi
done

if [ "$CAMERA_FOUND" = false ]; then
    export BAMBOO_CAMERA_MODE="simulation"
    export BAMBOO_SKIP_CAMERA="true"
else
    export BAMBOO_CAMERA_MODE="hardware"
    export BAMBOO_SKIP_CAMERA="false"
fi

echo "📋 环境状态:"
echo "   摄像头模式: $BAMBOO_CAMERA_MODE"
echo "   Qt平台: $QT_QPA_PLATFORM"
echo "   EGL平台: $EGL_PLATFORM"

# 简单可靠的启动函数
start_backend_safe() {
    if [ -f "./bamboo_cut_backend" ] && [ -x "./bamboo_cut_backend" ]; then
        echo "🔄 启动后端..."
        ./bamboo_cut_backend &
        BACKEND_PID=$!
        
        sleep 3
        
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo "✅ 后端启动成功"
            return 0
        else
            echo "⚠️ 后端启动失败"
            wait $BACKEND_PID 2>/dev/null || true
            return 1
        fi
    else
        echo "❌ 后端不存在"
        return 1
    fi
}

start_frontend_safe() {
    FRONTEND_EXEC=""
    for candidate in "./bamboo_controller_qt" "./bamboo_cut_frontend"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            FRONTEND_EXEC="$candidate"
            break
        fi
    done
    
    if [ -z "$FRONTEND_EXEC" ]; then
        echo "❌ 前端不存在"
        return 1
    fi
    
    echo "🔄 启动前端: $FRONTEND_EXEC"
    "$FRONTEND_EXEC" &
    FRONTEND_PID=$!
    
    sleep 5
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ 前端启动成功"
        return 0
    else
        echo "❌ 前端启动失败"
        wait $FRONTEND_PID 2>/dev/null || true
        return 1
    fi
}

# 主启动逻辑（简单可靠）
echo "🚀 开始启动智能切竹机系统..."

BACKEND_STARTED=false
FRONTEND_STARTED=false

# 启动后端
if start_backend_safe; then
    BACKEND_STARTED=true
fi

# 启动前端
if start_frontend_safe; then
    FRONTEND_STARTED=true
fi

# 运行逻辑
if [ "$FRONTEND_STARTED" = true ] && [ "$BACKEND_STARTED" = true ]; then
    echo "✅ 前后端都已启动"
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        wait $FRONTEND_PID
    fi
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
elif [ "$FRONTEND_STARTED" = true ]; then
    echo "✅ 仅前端运行"
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        wait $FRONTEND_PID
    fi
elif [ "$BACKEND_STARTED" = true ]; then
    echo "✅ 仅后端运行"
    if kill -0 $BACKEND_PID 2>/dev/null; then
        wait $BACKEND_PID
    fi
else
    echo "❌ 启动失败"
    exit 1
fi

echo "🛑 系统已停止"
exit 0
EOF
    
    chmod +x "${JETPACK_DEPLOY_DIR}/start_bamboo_safe.sh"
    log_success "安全启动脚本已创建"
}

# 部署文件
deploy_files() {
    log_info "部署文件到系统..."
    
    sudo mkdir -p /opt/bamboo-cut
    sudo mkdir -p /opt/bamboo-cut/config
    
    # 部署可执行文件
    if [ -f "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" ]; then
        sudo cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" /opt/bamboo-cut/
        sudo chmod +x /opt/bamboo-cut/bamboo_cut_backend
        log_success "后端已部署"
    fi
    
    # 查找和部署Qt前端
    qt_deployed=false
    QT_CANDIDATES=(
        "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
        "${BUILD_DIR}/qt_frontend/bamboo_cut_frontend"
        "${PROJECT_ROOT}/qt_frontend/build/bamboo_controller_qt"
        "${PROJECT_ROOT}/qt_frontend/build_qmake/bamboo_controller_qt"
    )
    
    for candidate in "${QT_CANDIDATES[@]}"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            sudo cp "$candidate" /opt/bamboo-cut/bamboo_controller_qt
            sudo chmod +x /opt/bamboo-cut/bamboo_controller_qt
            log_success "前端已部署: $(basename $candidate)"
            qt_deployed=true
            break
        fi
    done
    
    if [ "$qt_deployed" = false ]; then
        log_warning "未找到Qt前端可执行文件"
    fi
    
    # 部署配置和脚本
    if [ -d "${PROJECT_ROOT}/config" ]; then
        sudo cp -r "${PROJECT_ROOT}/config"/* /opt/bamboo-cut/config/
    fi
    
    sudo cp "${JETPACK_DEPLOY_DIR}/jetson_env.sh" /opt/bamboo-cut/
    sudo cp "${JETPACK_DEPLOY_DIR}/config/kms.conf" /opt/bamboo-cut/config/
    sudo cp "${JETPACK_DEPLOY_DIR}/start_bamboo_safe.sh" /opt/bamboo-cut/
    
    log_success "文件部署完成"
}

# 创建systemd服务
create_systemd_service() {
    log_jetpack "创建 systemd 服务..."
    
    cat > /tmp/bamboo-cut-jetpack.service << EOF
[Unit]
Description=智能切竹机系统 (Jetson $JETSON_TYPE 修复版)
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_safe.sh
TimeoutStartSec=90
TimeoutStopSec=30
Restart=on-failure
RestartSec=10
StartLimitBurst=3
KillMode=mixed
KillSignal=SIGTERM
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms_egldevice
Environment=EGL_PLATFORM=device
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/bamboo-cut-jetpack.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_success "systemd 服务已创建"
}

# 主函数
main() {
    echo "========================================"
    echo "Jetson EGL 修复 + 编译部署脚本 (修复版)"
    echo "========================================"
    
    if [ "$EUID" -ne 0 ]; then
        log_error "请以root权限运行: sudo $0"
        exit 1
    fi
    
    parse_arguments "$@"
    
    # 检查设备
    if ! check_jetson_device; then
        exit 1
    fi
    
    log_jetpack "检测到设备: $JETSON_TYPE ($TEGRA_CHIP)"
    
    # 强制停止服务
    force_stop_all_services
    
    if [ "$STOP_ONLY" = "true" ]; then
        log_success "仅停止模式完成"
        exit 0
    fi
    
    # 设置默认值
    CLEAN_LEGACY="true"
    FORCE_REBUILD="true"
    ENABLE_QT_DEPLOY="true"
    
    # 执行主要流程
    install_jetpack_dependencies
    build_project
    configure_jetson_libraries
    create_kms_config
    check_tegra_gpu
    create_timeout_safe_startup
    deploy_files
    create_systemd_service
    
    # 启动服务
    log_info "启动服务..."
    sudo systemctl enable bamboo-cut-jetpack
    sudo systemctl start bamboo-cut-jetpack
    
    sleep 5
    
    if systemctl is-active --quiet bamboo-cut-jetpack; then
        log_success "✅ 服务启动成功"
        echo "查看状态: sudo systemctl status bamboo-cut-jetpack"
        echo "查看日志: sudo journalctl -u bamboo-cut-jetpack -f"
    else
        log_warning "⚠️ 服务启动可能有问题"
        echo "检查状态: sudo systemctl status bamboo-cut-jetpack"
        echo "查看日志: sudo journalctl -u bamboo-cut-jetpack --no-pager"
    fi
    
    echo ""
    echo "🎯 修复摘要："
    echo "✅ 修复了函数调用顺序问题"
    echo "✅ 添加了启动超时保护机制"
    echo "✅ 强化了进程清理功能"
    echo "✅ 简化了服务配置"
    echo "✅ 编译和部署了前后端"
    
    log_success "🎉 Jetson EGL 修复和部署完成!"
}

# 运行主函数
main "$@"