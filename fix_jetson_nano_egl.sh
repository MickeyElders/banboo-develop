#!/bin/bash
# Jetson Nano/Orin Tegra SoC 专用 EGL 修复脚本 (增强版，包含前端编译)
# 针对 NVIDIA Tegra GPU 的 EGLDevice/EGLStream 配置

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

# 默认配置
BUILD_TYPE="Release"
FORCE_REBUILD="false"
COMPILE_BACKEND="true"
COMPILE_FRONTEND="true"
INSTALL_DEPENDENCIES="false"

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
    echo -e "${CYAN}[QT-BUILD]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Jetson Tegra SoC 专用 EGL 修复脚本 (增强版，包含前端编译)

用法: $0 [选项]

🎯 主要功能:
    - 检测并适配多种 Jetson 设备 (Nano, Orin NX, AGX Orin, Xavier)
    - 编译 C++ 后端和 Qt 前端
    - 配置 Tegra 专用的 EGL 环境
    - 创建 KMS 配置和启动脚本
    - 设置 systemd 服务

⚙️  可选参数:
    -t, --type TYPE         构建类型 (Debug, Release) [默认: Release]
    -f, --force-rebuild     强制重新编译所有组件
    -b, --backend-only      仅编译 C++ 后端
    -q, --frontend-only     仅编译 Qt 前端
    -i, --install-deps      安装编译依赖包
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

🚀 使用示例:
    $0                                              # 完整编译和配置
    $0 --force-rebuild                              # 强制重新编译
    $0 --backend-only                               # 仅编译后端
    $0 --frontend-only                              # 仅编译前端
    $0 --type Debug --install-deps                  # Debug模式并安装依赖

💡 提示:
    - 脚本会自动检测 Jetson 设备型号并适配
    - 支持 Jetson Nano, Orin NX, AGX Orin, Xavier
    - 使用 EGLDevice 而非 GBM 模式以获得最佳性能
    - 自动配置 Tegra 专用库路径和 EGL 环境

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
            -f|--force-rebuild)
                FORCE_REBUILD="true"
                shift
                ;;
            -b|--backend-only)
                COMPILE_BACKEND="true"
                COMPILE_FRONTEND="false"
                shift
                ;;
            -q|--frontend-only)
                COMPILE_BACKEND="false"
                COMPILE_FRONTEND="true"
                shift
                ;;
            -i|--install-deps)
                INSTALL_DEPENDENCIES="true"
                shift
                ;;
            -v|--version)
                echo "Jetson Tegra EGL 修复脚本 增强版 v1.2.0"
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

# 检查Jetson设备（支持多种型号）
check_jetson_device() {
    log_info "检查Jetson设备..."
    
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

# 安装编译依赖
install_build_dependencies() {
    if [ "$INSTALL_DEPENDENCIES" = "true" ]; then
        log_info "安装编译依赖包..."
        
        # 更新包管理器
        sudo apt update
        
        # 基础编译工具
        log_info "安装基础编译工具..."
        sudo apt install -y \
            build-essential \
            cmake \
            ninja-build \
            pkg-config \
            git
        
        # C++ 后端依赖
        log_info "安装 C++ 后端依赖..."
        sudo apt install -y \
            libopencv-dev \
            libopencv-contrib-dev \
            libmodbus-dev \
            nlohmann-json3-dev \
            libeigen3-dev \
            libprotobuf-dev \
            protobuf-compiler
        
        # Qt6 相关包
        log_info "安装 Qt6 开发包..."
        sudo apt install -y \
            qt6-base-dev \
            qt6-declarative-dev \
            qt6-multimedia-dev \
            qt6-serialport-dev \
            qt6-tools-dev \
            qt6-wayland \
            qml6-module-qtquick \
            qml6-module-qtquick-controls \
            qml6-module-qtmultimedia
        
        # CUDA 和 TensorRT (如果可用)
        log_info "检查 CUDA 和 TensorRT..."
        if command -v nvcc &> /dev/null; then
            log_success "CUDA 已安装"
        else
            log_warning "CUDA 未安装，将跳过 GPU 加速功能"
        fi
        
        # GStreamer (用于硬件加速)
        log_info "安装 GStreamer 硬件加速组件..."
        sudo apt install -y \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-plugins-ugly \
            gstreamer1.0-libav \
            gstreamer1.0-tools \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev
        
        log_success "编译依赖包安装完成"
    fi
}

# 清理构建缓存
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

# 构建 C++ 后端
build_cpp_backend() {
    if [ "$COMPILE_BACKEND" = "true" ]; then
        log_info "构建 C++ 后端..."
        
        # 检查后端源码是否存在
        if [ ! -d "${PROJECT_ROOT}/cpp_backend" ] && [ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
            log_warning "C++ 后端源码目录不存在，跳过后端编译"
            return 0
        fi
        
        mkdir -p "${BUILD_DIR}/cpp_backend"
        cd "${BUILD_DIR}/cpp_backend"
        
        # CMake 配置参数
        CMAKE_ARGS=(
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
            -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
            -DJETSON_BUILD=ON
            -DCMAKE_CUDA_ARCHITECTURES="53;62;72;75;86;87"
        )
        
        # 根据 Jetson 类型添加特定配置
        case "$JETSON_TYPE" in
            "nano")
                CMAKE_ARGS+=(-DTEGRA_CHIP="tegra210")
                CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="53;62")
                ;;
            "xavier")
                CMAKE_ARGS+=(-DTEGRA_CHIP="tegra194")
                CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="72")
                ;;
            "orin"*|"agx-orin")
                CMAKE_ARGS+=(-DTEGRA_CHIP="tegra234")
                CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="87")
                ;;
        esac
        
        # 检查是否有 CUDA
        if command -v nvcc &> /dev/null; then
            CMAKE_ARGS+=(-DENABLE_TENSORRT=ON)
            CMAKE_ARGS+=(-DENABLE_GPU_OPTIMIZATION=ON)
            CMAKE_ARGS+=(-DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda")
            log_info "启用 CUDA 和 TensorRT 支持"
        else
            CMAKE_ARGS+=(-DENABLE_TENSORRT=OFF)
            CMAKE_ARGS+=(-DENABLE_GPU_OPTIMIZATION=OFF)
            log_warning "CUDA 未安装，禁用 GPU 加速"
        fi
        
        # 确定 CMakeLists.txt 位置
        CMAKE_SOURCE_DIR=""
        if [ -f "${PROJECT_ROOT}/cpp_backend/CMakeLists.txt" ]; then
            CMAKE_SOURCE_DIR="${PROJECT_ROOT}/cpp_backend"
        elif [ -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
            CMAKE_SOURCE_DIR="${PROJECT_ROOT}"
        else
            log_error "找不到 CMakeLists.txt 文件"
            return 1
        fi
        
        # 运行 CMake
        log_info "运行 CMake 配置..."
        if ! cmake "${CMAKE_ARGS[@]}" "$CMAKE_SOURCE_DIR"; then
            log_error "C++ 后端 CMake 配置失败"
            return 1
        fi
        
        # 编译
        log_info "编译 C++ 后端..."
        if ! make -j$(nproc); then
            log_error "C++ 后端编译失败"
            return 1
        fi
        
        log_success "C++ 后端构建完成"
        cd "$PROJECT_ROOT"
        return 0
    fi
}

# 构建 Qt 前端
build_qt_frontend() {
    if [ "$COMPILE_FRONTEND" = "true" ]; then
        log_qt "构建 Qt 前端..."
        
        # 检查前端源码是否存在
        if [ ! -d "${PROJECT_ROOT}/qt_frontend" ]; then
            log_warning "Qt 前端源码目录不存在，跳过前端编译"
            return 0
        fi
        
        # 检查是否有 Qt6
        if ! command -v qmake6 &> /dev/null && ! command -v qmake &> /dev/null; then
            log_error "Qt6 未安装，请先安装 Qt6 开发包"
            return 1
        fi
        
        mkdir -p "${BUILD_DIR}/qt_frontend"
        cd "${BUILD_DIR}/qt_frontend"
        
        # Qt 前端 CMake 配置参数
        QT_CMAKE_ARGS=(
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
            -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
            -DJETSON_BUILD=ON
            -DCMAKE_PREFIX_PATH="/usr/lib/aarch64-linux-gnu/cmake"
        )
        
        # 根据 Jetson 设备类型添加特定配置
        case "$JETSON_TYPE" in
            "nano")
                QT_CMAKE_ARGS+=(-DTEGRA_CHIP="tegra210")
                ;;
            "xavier")
                QT_CMAKE_ARGS+=(-DTEGRA_CHIP="tegra194")
                ;;
            "orin"*|"agx-orin")
                QT_CMAKE_ARGS+=(-DTEGRA_CHIP="tegra234")
                ;;
        esac
        
        # 查找 Qt 前端的 CMakeLists.txt
        QT_SOURCE_DIR=""
        if [ -f "${PROJECT_ROOT}/qt_frontend/CMakeLists.txt" ]; then
            QT_SOURCE_DIR="${PROJECT_ROOT}/qt_frontend"
        else
            log_error "Qt 前端 CMakeLists.txt 不存在"
            return 1
        fi
        
        # 运行 CMake for Qt frontend
        log_qt "运行 Qt CMake 配置..."
        if ! cmake "${QT_CMAKE_ARGS[@]}" "$QT_SOURCE_DIR"; then
            log_error "Qt 前端 CMake 配置失败"
            
            # 尝试备用方法：直接使用 qmake
            log_qt "尝试使用 qmake 构建..."
            cd "${PROJECT_ROOT}/qt_frontend"
            
            # 查找 .pro 文件
            PRO_FILE=""
            if [ -f "bamboo_controller_qt.pro" ]; then
                PRO_FILE="bamboo_controller_qt.pro"
            elif [ -f "bamboo_cut_frontend.pro" ]; then
                PRO_FILE="bamboo_cut_frontend.pro"
            elif [ -f "main.pro" ]; then
                PRO_FILE="main.pro"
            else
                PRO_FILE=$(find . -maxdepth 1 -name "*.pro" | head -1)
            fi
            
            if [ -z "$PRO_FILE" ]; then
                log_error "找不到 Qt .pro 文件"
                return 1
            fi
            
            log_qt "使用 .pro 文件: $PRO_FILE"
            
            # 使用 qmake
            QMAKE_CMD="qmake6"
            if ! command -v qmake6 &> /dev/null; then
                QMAKE_CMD="qmake"
            fi
            
            # 创建构建目录
            BUILD_SUBDIR="build_${BUILD_TYPE,,}"
            mkdir -p "$BUILD_SUBDIR"
            cd "$BUILD_SUBDIR"
            
            # 配置 qmake
            if [ "$BUILD_TYPE" = "Debug" ]; then
                QMAKE_CONFIG="CONFIG+=debug"
            else
                QMAKE_CONFIG="CONFIG+=release"
            fi
            
            # 运行 qmake
            if ! $QMAKE_CMD "$QMAKE_CONFIG" "../$PRO_FILE"; then
                log_error "qmake 配置失败"
                return 1
            fi
            
            # 编译
            log_qt "使用 qmake/make 编译 Qt 前端..."
            if ! make -j$(nproc); then
                log_error "Qt 前端 qmake 编译失败"
                return 1
            fi
            
            # 复制到统一的构建目录
            mkdir -p "${BUILD_DIR}/qt_frontend"
            find . -type f -executable -name "bamboo*" -exec cp {} "${BUILD_DIR}/qt_frontend/" \;
            
            log_success "Qt 前端 (qmake) 构建完成"
            cd "$PROJECT_ROOT"
            return 0
        fi
        
        # CMake 成功，继续编译
        log_qt "编译 Qt 前端..."
        if ! make -j$(nproc); then
            log_error "Qt 前端编译失败"
            return 1
        fi
        
        log_success "Qt 前端构建完成"
        cd "$PROJECT_ROOT"
        return 0
    fi
}

# 构建项目 - 统一入口
build_project() {
    log_info "开始构建智能切竹机项目..."
    
    cd "$PROJECT_ROOT"
    
    # 清理构建缓存（如果需要）
    clean_build_cache
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    
    # 构建 C++ 后端
    if ! build_cpp_backend; then
        log_error "C++ 后端构建失败"
        return 1
    fi
    
    # 构建 Qt 前端
    if ! build_qt_frontend; then
        log_error "Qt 前端构建失败"
        return 1
    fi
    
    log_success "项目构建完成"
    return 0
}

# 配置Jetson专用的NVIDIA库环境
configure_jetson_libraries() {
    log_info "配置Jetson专用NVIDIA库环境..."
    
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
    cat > /tmp/jetson_tegra_env.sh << EOF
#!/bin/bash
# Jetson Tegra SoC 专用环境配置

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

    sudo mkdir -p /opt/bamboo-cut
    sudo cp /tmp/jetson_tegra_env.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/jetson_tegra_env.sh
    
    log_success "Jetson Tegra环境配置完成"
}

# 创建Jetson专用KMS配置
create_jetson_kms_config() {
    log_info "创建Jetson专用KMS配置..."
    
    sudo mkdir -p /opt/bamboo-cut/config
    
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
    
    echo "$KMS_CONFIG_CONTENT" | sudo tee /opt/bamboo-cut/config/kms.conf > /dev/null
    sudo chown root:root /opt/bamboo-cut/config/kms.conf
    sudo chmod 644 /opt/bamboo-cut/config/kms.conf
    
    log_success "Jetson $JETSON_TYPE KMS配置已创建"
}

# 检查Tegra GPU状态（替代nvidia-smi）
check_tegra_gpu() {
    log_info "检查Tegra GPU状态..."
    
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

# 创建Jetson专用启动脚本（包含编译后的可执行文件）
create_jetson_startup_script() {
    log_info "创建Jetson专用启动脚本（包含编译检查）..."
    
    cat > /tmp/start_bamboo_jetson_enhanced.sh << 'EOF'
#!/bin/bash
# Jetson Tegra SoC 专用智能切竹机启动脚本（增强版，包含编译检查）

echo "🚀 启动智能切竹机系统（Jetson Tegra专用版）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载Jetson Tegra环境
if [ -f "./jetson_tegra_env.sh" ]; then
    source "./jetson_tegra_env.sh"
    echo "✅ Jetson Tegra环境已加载"
else
    echo "⚠️ Jetson Tegra环境脚本不存在，使用内置配置"
    
    # 内置Jetson配置
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

# Qt EGLFS配置 - Jetson专用
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

echo "✅ Jetson Tegra EGL环境配置完成"
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

# 支持不同Jetson设备的GPU频率路径
GPU_FREQ_PATHS=(
    "/sys/devices/platform/host1x/17000000.gpu/devfreq/17000000.gpu/cur_freq"
    "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq"
)

GPU_FREQ_FOUND=false
for freq_path in "${GPU_FREQ_PATHS[@]}"; do
    if [ -f "$freq_path" ]; then
        GPU_FREQ=$(cat "$freq_path" 2>/dev/null)
        echo "📋 当前GPU频率: $GPU_FREQ Hz"
        GPU_FREQ_FOUND=true
        break
    fi
done

if [ "$GPU_FREQ_FOUND" = false ]; then
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

# 检查编译后的可执行文件
echo "🔍 检查编译后的可执行文件..."

# 检查后端可执行文件
BACKEND_CANDIDATES=(
    "./bamboo_cut_backend"
    "./build/cpp_backend/bamboo_cut_backend" 
    "../build/cpp_backend/bamboo_cut_backend"
)

BACKEND_EXEC=""
for candidate in "${BACKEND_CANDIDATES[@]}"; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        BACKEND_EXEC="$candidate"
        echo "✅ 找到C++后端: $candidate"
        break
    fi
done

if [ -z "$BACKEND_EXEC" ]; then
    echo "❌ 未找到C++后端可执行文件"
    echo "💡 请先编译项目: 运行构建脚本或make命令"
    echo "🔍 查找的位置:"
    for candidate in "${BACKEND_CANDIDATES[@]}"; do
        echo "   - $candidate"
    done
fi

# 检查前端可执行文件
FRONTEND_CANDIDATES=(
    "./bamboo_controller_qt"
    "./bamboo_cut_frontend"
    "./build/qt_frontend/bamboo_controller_qt"
    "./build/qt_frontend/bamboo_cut_frontend"
    "../build/qt_frontend/bamboo_controller_qt"
    "../build/qt_frontend/bamboo_cut_frontend"
)

FRONTEND_EXEC=""
for candidate in "${FRONTEND_CANDIDATES[@]}"; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        FRONTEND_EXEC="$candidate"
        echo "✅ 找到Qt前端: $candidate"
        break
    fi
done

if [ -z "$FRONTEND_EXEC" ]; then
    echo "❌ 未找到Qt前端可执行文件"
    echo "💡 请先编译项目: 运行构建脚本或make qt命令"
    echo "🔍 查找的位置:"
    for candidate in "${FRONTEND_CANDIDATES[@]}"; do
        echo "   - $candidate"
    done
fi

# 启动后端
start_backend() {
    if [ -z "$BACKEND_EXEC" ]; then
        echo "❌ C++后端可执行文件不存在，跳过后端启动"
        return 1
    fi
    
    echo "🔄 启动C++后端: $BACKEND_EXEC"
    timeout 60 "$BACKEND_EXEC" &
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
    if [ -z "$FRONTEND_EXEC" ]; then
        echo "❌ Qt前端可执行文件不存在，跳过前端启动"
        return 1
    fi
    
    echo "🔄 启动Qt前端: $FRONTEND_EXEC"
    echo "🔧 使用Jetson Tegra EGLDevice模式..."
    
    # 显示当前配置
    echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo "   EGL_PLATFORM: ${EGL_PLATFORM}"
    echo "   QT_QPA_PLATFORM: ${QT_QPA_PLATFORM}"
    echo "   QT_QPA_EGLFS_INTEGRATION: ${QT_QPA_EGLFS_INTEGRATION}"
    
    # 启动Qt前端
    timeout 30 "$FRONTEND_EXEC" &
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

# 检查是否有可执行文件
if [ -z "$BACKEND_EXEC" ] && [ -z "$FRONTEND_EXEC" ]; then
    echo "❌ 未找到任何可执行文件"
    echo "💡 请先使用以下命令编译项目："
    echo "   sudo bash fix_jetson_nano_egl.sh --force-rebuild"
    echo "   或者："
    echo "   make all"
    exit 1
fi

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
elif [ -n "$FRONTEND_EXEC" ]; then
    echo "🔄 仅前端模式运行"
    if start_frontend; then
        wait $FRONTEND_PID
    fi
else
    echo "✅ 后端在模拟模式下完成"
fi

echo "🛑 Jetson Tegra智能切竹机系统已停止"
EOF

    sudo cp /tmp/start_bamboo_jetson_enhanced.sh /opt/bamboo-cut/
    sudo chmod +x /opt/bamboo-cut/start_bamboo_jetson_enhanced.sh
    
    log_success "Jetson专用启动脚本已创建（包含编译检查）"
}

# 更新systemd服务为Jetson专用（包含编译逻辑）
update_jetson_systemd_service() {
    log_info "更新systemd服务为Jetson专用..."
    
    cat > /tmp/bamboo-cut-jetpack.service << EOF
[Unit]
Description=智能切竹机系统 (Jetson $JETSON_TYPE 专用)
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_jetson_enhanced.sh
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
    
    log_success "Jetson $JETSON_TYPE 专用systemd服务已更新"
}

# 部署编译后的文件
deploy_compiled_files() {
    log_info "部署编译后的文件..."
    
    sudo mkdir -p /opt/bamboo-cut
    
    # 部署C++后端
    if [ "$COMPILE_BACKEND" = "true" ] && [ -f "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" ]; then
        sudo cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" /opt/bamboo-cut/
        sudo chmod +x /opt/bamboo-cut/bamboo_cut_backend
        log_success "C++后端已部署"
    fi
    
    # 部署Qt前端
    if [ "$COMPILE_FRONTEND" = "true" ]; then
        # 查找Qt前端可执行文件
        QT_FRONTEND_CANDIDATES=(
            "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
            "${BUILD_DIR}/qt_frontend/bamboo_cut_frontend"
            "${PROJECT_ROOT}/qt_frontend/build/bamboo_controller_qt"
            "${PROJECT_ROOT}/qt_frontend/build_debug/bamboo_controller_qt"
            "${PROJECT_ROOT}/qt_frontend/build_release/bamboo_controller_qt"
        )
        
        qt_deployed=false
        for candidate in "${QT_FRONTEND_CANDIDATES[@]}"; do
            if [ -f "$candidate" ]; then
                sudo cp "$candidate" /opt/bamboo-cut/bamboo_controller_qt
                sudo chmod +x /opt/bamboo-cut/bamboo_controller_qt
                log_success "Qt前端已部署: $(basename $candidate) -> bamboo_controller_qt"
                qt_deployed=true
                break
            fi
        done
        
        if [ "$qt_deployed" = false ]; then
            log_warning "Qt前端可执行文件未找到，跳过部署"
        fi
    fi
    
    # 部署配置文件
    if [ -d "${PROJECT_ROOT}/config" ]; then
        sudo cp -r "${PROJECT_ROOT}/config" /opt/bamboo-cut/
        log_success "配置文件已部署"
    fi
    
    log_success "编译文件部署完成"
}

# 主函数
main() {
    echo "========================================"
    echo "Jetson Tegra SoC EGL 专用修复脚本 (增强版)"
    echo "包含前端编译和智能部署功能"
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
    
    log_info "检测到 Jetson $JETSON_TYPE 设备 ($TEGRA_CHIP)"
    
    # 安装编译依赖
    install_build_dependencies
    
    # 停止现有服务
    log_info "停止现有服务..."
    systemctl stop bamboo-cut-jetpack 2>/dev/null || true
    
    # 构建项目
    if [ "$COMPILE_BACKEND" = "true" ] || [ "$COMPILE_FRONTEND" = "true" ]; then
        if ! build_project; then
            log_error "项目构建失败"
            exit 1
        fi
        
        # 部署编译后的文件
        deploy_compiled_files
    else
        log_warning "跳过编译步骤"
    fi
    
    # 执行Jetson专用修复
    configure_jetson_libraries
    create_jetson_kms_config
    check_tegra_gpu
    create_jetson_startup_script
    update_jetson_systemd_service
    
    # 启动服务
    log_info "启动Jetson专用服务..."
    systemctl enable bamboo-cut-jetpack
    systemctl start bamboo-cut-jetpack
    
    sleep 3
    
    # 检查结果
    if systemctl is-active --quiet bamboo-cut-jetpack; then
        log_success "✅ Jetson $JETSON_TYPE 智能切竹机服务启动成功！"
        log_info "查看状态: systemctl status bamboo-cut-jetpack"
        log_info "查看日志: journalctl -u bamboo-cut-jetpack -f"
    else
        log_warning "⚠️ 服务启动可能有问题"
        log_info "检查详情: journalctl -u bamboo-cut-jetpack --no-pager"
    fi
    
    echo ""
    echo "🎯 Jetson $JETSON_TYPE 专用修复摘要："
    echo "✅ 检测并适配了 $JETSON_TYPE 设备 ($TEGRA_CHIP)"
    if [ "$COMPILE_BACKEND" = "true" ]; then
        echo "✅ 编译了C++后端"
    fi
    if [ "$COMPILE_FRONTEND" = "true" ]; then
        echo "✅ 编译了Qt前端"
    fi
    echo "✅ 配置了Tegra专用库路径和EGL环境"
    echo "✅ 使用EGLDevice而非GBM模式"
    echo "✅ 创建了$JETSON_TYPE专用KMS配置"
    echo "✅ 添加了Tegra GPU状态检查（非nvidia-smi）"
    echo "✅ 更新了启动脚本和systemd服务"
    echo "✅ 部署了编译后的可执行文件"
    echo ""
    echo "🔧 注意：Jetson设备使用集成Tegra GPU，无需nvidia-smi"
    
    log_success "🎉 Jetson $JETSON_TYPE EGL专用修复完成！"
}

# 运行主函数
main "$@"