#!/bin/bash

# 智能切竹机 C++/Flutter 构建部署脚本
# 支持 Jetson (ARM64) 和 x86_64 平台

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本信息
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="${PROJECT_ROOT}/build"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"

# 默认配置
TARGET_ARCH="$(uname -m)"
BUILD_TYPE="Release"
ENABLE_TENSORRT="OFF"
INSTALL_DEPENDENCIES="false"
DEPLOY_TARGET=""
CROSS_COMPILE="false"

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

# 显示帮助信息
show_help() {
    cat << EOF
智能切竹机 C++/Flutter 构建部署脚本

用法: $0 [选项]

选项:
    -a, --arch ARCH         目标架构 (x86_64, aarch64, arm64)
    -t, --type TYPE         构建类型 (Debug, Release) [默认: Release]
    -d, --deploy TARGET     部署目标 (local, jetson, remote:IP)
    -c, --cross-compile     启用交叉编译
    -r, --tensorrt          启用TensorRT支持
    -i, --install-deps      安装依赖包
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

示例:
    $0 --arch aarch64 --cross-compile --tensorrt          # 交叉编译ARM64版本
    $0 --deploy jetson                                     # 本地构建并部署到Jetson
    $0 --deploy remote:192.168.1.100                      # 部署到远程设备
    $0 --install-deps --arch x86_64                       # 安装依赖并构建x86版本

EOF
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--arch)
                TARGET_ARCH="$2"
                shift 2
                ;;
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -d|--deploy)
                DEPLOY_TARGET="$2"
                shift 2
                ;;
            -c|--cross-compile)
                CROSS_COMPILE="true"
                shift
                ;;
            -r|--tensorrt)
                ENABLE_TENSORRT="ON"
                shift
                ;;
            -i|--install-deps)
                INSTALL_DEPENDENCIES="true"
                shift
                ;;
            -v|--version)
                echo "智能切竹机构建脚本 版本 ${VERSION}"
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

# 检测系统环境
detect_environment() {
    log_info "检测系统环境..."
    
    HOST_ARCH="$(uname -m)"
    HOST_OS="$(uname -s)"
    
    log_info "主机架构: ${HOST_ARCH}"
    log_info "主机操作系统: ${HOST_OS}"
    log_info "目标架构: ${TARGET_ARCH}"
    log_info "构建类型: ${BUILD_TYPE}"
    
    # 检查是否需要交叉编译
    if [ "$HOST_ARCH" != "$TARGET_ARCH" ]; then
        CROSS_COMPILE="true"
        log_warning "目标架构与主机架构不同，将启用交叉编译"
    fi
    
    # 检测Jetson设备
    if [ -f "/proc/device-tree/model" ] && grep -q "Jetson" /proc/device-tree/model; then
        log_info "检测到Jetson设备"
        ENABLE_TENSORRT="ON"
    fi
}

# 安装依赖包
install_dependencies() {
    if [ "$INSTALL_DEPENDENCIES" = "true" ]; then
        log_info "安装依赖包..."
        
        # 更新包管理器
        sudo apt update
        
        # 基础开发工具
        sudo apt install -y \
            build-essential \
            cmake \
            git \
            pkg-config \
            ninja-build
        
        # OpenCV
        sudo apt install -y \
            libopencv-dev \
            libopencv-contrib-dev
        
        # GStreamer
        sudo apt install -y \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev \
            libgstreamer-plugins-bad1.0-dev \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-plugins-ugly \
            gstreamer1.0-libav
        
        # Modbus库
        sudo apt install -y libmodbus-dev
        
        # JSON库
        sudo apt install -y nlohmann-json3-dev
        
        # 交叉编译工具链
        if [ "$CROSS_COMPILE" = "true" ] && [ "$TARGET_ARCH" = "aarch64" ]; then
            sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
        fi
        
        # Flutter依赖
        sudo apt install -y \
            curl \
            file \
            git \
            unzip \
            xz-utils \
            zip \
            libglu1-mesa
        
        log_success "依赖包安装完成"
    fi
}

# 设置交叉编译工具链
setup_cross_compile() {
    if [ "$CROSS_COMPILE" = "true" ]; then
        log_info "设置交叉编译环境..."
        
        case "$TARGET_ARCH" in
            aarch64|arm64)
                export CC=aarch64-linux-gnu-gcc
                export CXX=aarch64-linux-gnu-g++
                export AR=aarch64-linux-gnu-ar
                export STRIP=aarch64-linux-gnu-strip
                CMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain-aarch64.cmake"
                ;;
            *)
                log_error "不支持的交叉编译架构: $TARGET_ARCH"
                exit 1
                ;;
        esac
        
        log_info "交叉编译工具链设置完成"
    fi
}

# 构建C++后端
build_cpp_backend() {
    log_info "构建C++后端..."
    
    cd "$PROJECT_ROOT"
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR/cpp_backend"
    cd "$BUILD_DIR/cpp_backend"
    
    # CMake配置参数
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_INSTALL_PREFIX="/opt/bamboo-cut"
    )
    
    if [ "$ENABLE_TENSORRT" = "ON" ]; then
        CMAKE_ARGS+=(-DENABLE_TENSORRT=ON)
        log_info "启用TensorRT支持"
    fi
    
    if [ "$CROSS_COMPILE" = "true" ] && [ -n "$CMAKE_TOOLCHAIN_FILE" ]; then
        CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE")
        log_info "使用交叉编译工具链"
    fi
    
    # 运行CMake
    cmake "${CMAKE_ARGS[@]}" "${PROJECT_ROOT}/cpp_backend"
    
    # 编译
    make -j$(nproc)
    
    log_success "C++后端构建完成"
}

# 构建Flutter前端
build_flutter_frontend() {
    log_info "构建Flutter前端..."
    
    cd "${PROJECT_ROOT}/flutter_frontend"
    
    # 检查Flutter是否已安装
    if ! command -v flutter &> /dev/null; then
        log_warning "Flutter未安装，正在下载..."
        install_flutter
    fi
    
    # 获取依赖
    flutter pub get
    
    # 构建Linux版本
    flutter build linux --release
    
    log_success "Flutter前端构建完成"
}

# 安装Flutter
install_flutter() {
    log_info "安装Flutter..."
    
    FLUTTER_DIR="/opt/flutter"
    
    # 下载Flutter
    if [ ! -d "$FLUTTER_DIR" ]; then
        sudo mkdir -p "$FLUTTER_DIR"
        cd /tmp
        
        # 选择合适的Flutter版本
        case "$TARGET_ARCH" in
            x86_64)
                FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.16.0-stable.tar.xz"
                ;;
            aarch64|arm64)
                FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_arm64_3.16.0-stable.tar.xz"
                ;;
            *)
                log_error "不支持的Flutter架构: $TARGET_ARCH"
                exit 1
                ;;
        esac
        
        wget "$FLUTTER_URL" -O flutter.tar.xz
        sudo tar xf flutter.tar.xz -C /opt/
        sudo chown -R $USER:$USER "$FLUTTER_DIR"
    fi
    
    # 添加到PATH
    export PATH="$FLUTTER_DIR/bin:$PATH"
    
    # Flutter配置
    flutter config --no-analytics
    flutter doctor
    
    log_success "Flutter安装完成"
}

# 创建部署包
create_deployment_package() {
    log_info "创建部署包..."
    
    PACKAGE_DIR="${DEPLOY_DIR}/packages/bamboo-cut-${VERSION}-${TARGET_ARCH}"
    mkdir -p "$PACKAGE_DIR"
    
    # 复制C++后端
    cp "${BUILD_DIR}/cpp_backend/bamboo_cut_backend" "$PACKAGE_DIR/"
    
    # 复制Flutter前端
    cp -r "${PROJECT_ROOT}/flutter_frontend/build/linux/x64/release/bundle" "$PACKAGE_DIR/frontend"
    
    # 复制配置文件
    cp -r "${PROJECT_ROOT}/config" "$PACKAGE_DIR/"
    
    # 复制启动脚本
    cp "${SCRIPT_DIR}/start_bamboo_cut.sh" "$PACKAGE_DIR/"
    chmod +x "$PACKAGE_DIR/start_bamboo_cut.sh"
    
    # 创建systemd服务文件
    cat > "$PACKAGE_DIR/bamboo-cut.service" << 'EOF'
[Unit]
Description=智能切竹机系统
After=network.target

[Service]
Type=simple
User=bamboo-cut
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # 创建安装脚本
    cat > "$PACKAGE_DIR/install.sh" << 'EOF'
#!/bin/bash
set -e

echo "安装智能切竹机系统..."

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

# 安装systemd服务
sudo cp bamboo-cut.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bamboo-cut

echo "安装完成! 使用以下命令启动服务:"
echo "sudo systemctl start bamboo-cut"
EOF
    chmod +x "$PACKAGE_DIR/install.sh"
    
    # 创建tar包
    cd "${DEPLOY_DIR}/packages"
    tar czf "bamboo-cut-${VERSION}-${TARGET_ARCH}.tar.gz" "bamboo-cut-${VERSION}-${TARGET_ARCH}"
    
    log_success "部署包创建完成: bamboo-cut-${VERSION}-${TARGET_ARCH}.tar.gz"
}

# 部署到目标设备
deploy_to_target() {
    if [ -n "$DEPLOY_TARGET" ]; then
        log_info "部署到目标设备: $DEPLOY_TARGET"
        
        case "$DEPLOY_TARGET" in
            local)
                # 本地安装
                cd "${DEPLOY_DIR}/packages/bamboo-cut-${VERSION}-${TARGET_ARCH}"
                sudo ./install.sh
                ;;
            jetson)
                # 部署到Jetson (假设通过SSH)
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
    local PACKAGE_FILE="${DEPLOY_DIR}/packages/bamboo-cut-${VERSION}-${TARGET_ARCH}.tar.gz"
    
    log_info "部署到远程设备: $TARGET_HOST"
    
    # 传输文件
    scp "$PACKAGE_FILE" "${TARGET_HOST}:/tmp/"
    
    # 远程安装
    ssh "$TARGET_HOST" << EOF
cd /tmp
tar xzf bamboo-cut-${VERSION}-${TARGET_ARCH}.tar.gz
cd bamboo-cut-${VERSION}-${TARGET_ARCH}
sudo ./install.sh
sudo systemctl start bamboo-cut
sudo systemctl status bamboo-cut
EOF
    
    log_success "远程部署完成"
}

# 主函数
main() {
    log_info "=== 智能切竹机 C++/Flutter 构建部署脚本 ==="
    log_info "版本: $VERSION"
    
    parse_arguments "$@"
    detect_environment
    install_dependencies
    setup_cross_compile
    
    build_cpp_backend
    build_flutter_frontend
    create_deployment_package
    deploy_to_target
    
    log_success "构建部署完成!"
    log_info "部署包位置: ${DEPLOY_DIR}/packages/bamboo-cut-${VERSION}-${TARGET_ARCH}.tar.gz"
}

# 运行主函数
main "$@" 