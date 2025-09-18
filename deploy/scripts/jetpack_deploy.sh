#!/bin/bash

# 智能切竹机 JetPack SDK 专用部署脚本
# 针对 Jetson Nano Super 硬件平台优化
# 集成 windeployqt 类似功能和 JetPack 性能调优

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本信息
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
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
DEPLOY_TARGET=""
CREATE_PACKAGE="true"
OPTIMIZE_PERFORMANCE="true"

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
智能切竹机 JetPack SDK 专用部署脚本

用法: $0 [选项]

选项:
    -t, --type TYPE         构建类型 (Debug, Release) [默认: Release]
    -d, --deploy TARGET     部署目标 (local, jetson, remote:IP)
    -i, --install-deps      安装 JetPack SDK 依赖包
    -g, --gpu-opt           启用 GPU 内存和计算优化
    -p, --power-opt         启用功耗管理和性能调优
    -m, --models            自动配置和部署 AI 模型文件
    -q, --qt-deploy         启用 Qt 依赖自动收集和部署
    -c, --create-package    创建完整部署包
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

示例:
    $0 --install-deps --gpu-opt --power-opt        # 安装依赖并启用全部优化
    $0 --deploy jetson --models --qt-deploy        # 部署到 Jetson 并配置模型
    $0 --deploy remote:192.168.1.100 --create-package    # 创建包并部署到远程设备

JetPack SDK 版本: ${JETPACK_VERSION}
CUDA 版本: ${CUDA_VERSION}
TensorRT 版本: ${TENSORRT_VERSION}
OpenCV 版本: ${OPENCV_VERSION}

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
            -d|--deploy)
                DEPLOY_TARGET="$2"
                shift 2
                ;;
            -i|--install-deps)
                INSTALL_DEPENDENCIES="true"
                shift
                ;;
            -g|--gpu-opt)
                ENABLE_GPU_OPTIMIZATION="ON"
                shift
                ;;
            -p|--power-opt)
                ENABLE_POWER_OPTIMIZATION="ON"
                shift
                ;;
            -m|--models)
                DEPLOY_MODELS="true"
                shift
                ;;
            -q|--qt-deploy)
                ENABLE_QT_DEPLOY="true"
                shift
                ;;
            -c|--create-package)
                CREATE_PACKAGE="true"
                shift
                ;;
            -v|--version)
                echo "智能切竹机 JetPack SDK 部署脚本 版本 ${VERSION}"
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

# 检测 JetPack SDK 环境
detect_jetpack_environment() {
    log_jetpack "检测 JetPack SDK 环境..."
    
    # 检查是否在 Jetson 设备上运行
    if [ -f "/proc/device-tree/model" ] && grep -q "Jetson" /proc/device-tree/model; then
        JETSON_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        log_jetpack "检测到 Jetson 设备: ${JETSON_MODEL}"
        
        # 检测 JetPack 版本
        if [ -f "/etc/nv_tegra_release" ]; then
            JETPACK_INFO=$(cat /etc/nv_tegra_release)
            log_jetpack "JetPack 信息: ${JETPACK_INFO}"
        fi
        
        # 检测 CUDA 版本
        if command -v nvcc &> /dev/null; then
            CUDA_INFO=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            log_jetpack "CUDA 版本: ${CUDA_INFO}"
        fi
        
        # 检测 TensorRT 版本
        if [ -f "/usr/include/NvInfer.h" ]; then
            TRT_INFO=$(grep "NV_TENSORRT_MAJOR" /usr/include/NvInfer.h | awk '{print $3}')
            log_jetpack "TensorRT 主版本: ${TRT_INFO}"
        fi
        
        ENABLE_TENSORRT="ON"
        JETSON_DETECTED="true"
    else
        log_warning "未检测到 Jetson 设备，将进行交叉编译配置"
        JETSON_DETECTED="false"
    fi
    
    # 检测系统架构
    HOST_ARCH="$(uname -m)"
    log_info "系统架构: ${HOST_ARCH}"
    log_info "构建类型: ${BUILD_TYPE}"
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
            libnvinfer-plugin-dev
        
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
            
            # 创建功耗配置文件
            cat > "${JETPACK_DEPLOY_DIR}/power_config.sh" << 'EOF'
#!/bin/bash
# JetPack SDK 功耗管理配置

# 设置 CPU 调度器为性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 优化内存管理
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
echo 80 | sudo tee /proc/sys/vm/swappiness

# GPU 功耗管理
echo 1 | sudo tee /sys/devices/platform/host1x/57000000.gpu/power/autosuspend_delay_ms

# 网络优化
echo 1 | sudo tee /proc/sys/net/core/netdev_max_backlog
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
# Qt 环境设置脚本

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="\${SCRIPT_DIR}:\${LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="\${SCRIPT_DIR}"
export QML2_IMPORT_PATH="\${SCRIPT_DIR}/qml"
export QT_QPA_PLATFORM_PLUGIN_PATH="\${SCRIPT_DIR}/platforms"

# JetPack SDK 特定环境变量
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf

echo "Qt 环境已配置完成"
EOF
        chmod +x "${QT_DEPLOY_DIR}/setup_qt_env.sh"
        
        log_success "Qt 依赖部署完成"
    fi
}

# 配置和部署 AI 模型文件
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
        
        # 更新模型配置文件以适配 JetPack SDK 路径
        log_jetpack "更新模型配置文件..."
        
        # 更新 AI 优化配置
        sed -i 's|model_path:.*|model_path: "/opt/bamboo-cut/models/bamboo_detector.onnx"|g' \
            "${PROJECT_ROOT}/config/ai_optimization.yaml" 2>/dev/null || true
        
        # 创建 TensorRT 模型优化脚本
        cat > "${MODELS_DIR}/optimize_models.sh" << 'EOF'
#!/bin/bash
# TensorRT 模型优化脚本

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${MODELS_DIR}/onnx"
TRT_DIR="${MODELS_DIR}/tensorrt"

echo "开始 TensorRT 模型优化..."

# 检查 trtexec 工具
if ! command -v trtexec &> /dev/null; then
    echo "错误: trtexec 未找到，请确保已安装 TensorRT"
    exit 1
fi

# 优化 ONNX 模型为 TensorRT 引擎
for onnx_file in "${ONNX_DIR}"/*.onnx; do
    if [ -f "$onnx_file" ]; then
        filename=$(basename "$onnx_file" .onnx)
        echo "优化模型: $filename"
        
        trtexec \
            --onnx="$onnx_file" \
            --saveEngine="${TRT_DIR}/${filename}.trt" \
            --fp16 \
            --workspace=1024 \
            --minShapes=input:1x3x640x640 \
            --optShapes=input:1x3x640x640 \
            --maxShapes=input:4x3x640x640
    fi
done

echo "TensorRT 模型优化完成"
EOF
        chmod +x "${MODELS_DIR}/optimize_models.sh"
        
        log_success "AI 模型配置和部署完成"
    fi
}

# 构建项目
build_project() {
    log_info "构建智能切竹机项目..."
    
    cd "$PROJECT_ROOT"
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
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
    
    # 运行 CMake
    cmake "${CMAKE_ARGS[@]}" "$PROJECT_ROOT"
    
    # 编译
    make -j$(nproc)
    
    log_success "项目构建完成"
}

# 创建 JetPack SDK 部署包
create_jetpack_package() {
    if [ "$CREATE_PACKAGE" = "true" ]; then
        log_jetpack "创建 JetPack SDK 部署包..."
        
        PACKAGE_DIR="${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}"
        mkdir -p "$PACKAGE_DIR"
        
        # 复制可执行文件
        cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" "$PACKAGE_DIR/" 2>/dev/null || true
        cp "${BUILD_DIR}/qt_frontend/bamboo_cut_frontend" "$PACKAGE_DIR/" 2>/dev/null || true
        
        # 复制配置文件
        cp -r "${PROJECT_ROOT}/config" "$PACKAGE_DIR/"
        
        # 复制 Qt 依赖 (如果启用)
        if [ "$ENABLE_QT_DEPLOY" = "true" ]; then
            cp -r "${JETPACK_DEPLOY_DIR}/qt_libs" "$PACKAGE_DIR/"
        fi
        
        # 复制模型文件 (如果启用)
        if [ "$DEPLOY_MODELS" = "true" ]; then
            cp -r "${JETPACK_DEPLOY_DIR}/models" "$PACKAGE_DIR/"
        fi
        
        # 复制性能优化脚本
        if [ -f "${JETPACK_DEPLOY_DIR}/power_config.sh" ]; then
            cp "${JETPACK_DEPLOY_DIR}/power_config.sh" "$PACKAGE_DIR/"
        fi
        
        # 创建 JetPack 启动脚本
        cat > "$PACKAGE_DIR/start_bamboo_cut_jetpack.sh" << 'EOF'
#!/bin/bash
# 智能切竹机 JetPack SDK 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 Qt 环境 (如果存在)
if [ -f "./qt_libs/setup_qt_env.sh" ]; then
    source "./qt_libs/setup_qt_env.sh"
fi

# 应用性能优化 (如果存在)
if [ -f "./power_config.sh" ]; then
    ./power_config.sh
fi

# 优化模型 (如果存在且需要)
if [ -f "./models/optimize_models.sh" ] && [ ! -f "./models/tensorrt/optimized.flag" ]; then
    echo "首次运行，正在优化 AI 模型..."
    ./models/optimize_models.sh
    touch "./models/tensorrt/optimized.flag"
fi

# 设置环境变量
export LD_LIBRARY_PATH="./qt_libs:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# 启动后端
echo "启动 C++ 后端..."
./bamboo_cut_backend &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "启动 Qt 前端..."
./bamboo_cut_frontend &
FRONTEND_PID=$!

# 等待进程
wait $FRONTEND_PID
kill $BACKEND_PID 2>/dev/null || true

echo "智能切竹机已停止"
EOF
        chmod +x "$PACKAGE_DIR/start_bamboo_cut_jetpack.sh"
        
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
sudo chown -R bamboo-cut:bamboo-cut /opt/bamboo-cut
sudo chown -R bamboo-cut:bamboo-cut /var/log/bamboo-cut
sudo chmod +x /opt/bamboo-cut/*.sh

# 创建 systemd 服务
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK)
After=network.target

[Service]
Type=simple
User=bamboo-cut
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack.sh
Restart=always
RestartSec=10
Environment=DISPLAY=:0

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

# 主函数
main() {
    log_jetpack "=== 智能切竹机 JetPack SDK 专用部署脚本 ==="
    log_jetpack "版本: $VERSION"
    log_jetpack "JetPack SDK: $JETPACK_VERSION"
    
    parse_arguments "$@"
    
    # 创建部署目录
    mkdir -p "$JETPACK_DEPLOY_DIR"
    
    detect_jetpack_environment
    install_jetpack_dependencies
    configure_jetpack_performance
    
    if [ "$ENABLE_QT_DEPLOY" = "true" ]; then
        deploy_qt_dependencies
    fi
    
    if [ "$DEPLOY_MODELS" = "true" ]; then
        deploy_ai_models
    fi
    
    build_project
    create_jetpack_package
    deploy_to_target
    
    log_success "JetPack SDK 部署完成!"
    log_info "部署包位置: ${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}.tar.gz"
    
    if [ "$JETSON_DETECTED" = "true" ]; then
        log_jetpack "运行性能测试: sudo jetson_stats"
        log_jetpack "监控 GPU 使用: sudo tegrastats"
    fi
}

# 运行主函数
main "$@"