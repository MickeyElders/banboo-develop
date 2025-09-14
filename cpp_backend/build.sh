#!/bin/bash

# 智能切竹机后端构建脚本
# 支持自动检测和安装依赖库

set -e  # 遇到错误立即退出

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

echo "=== 智能切竹机后端构建脚本 ==="
echo "目标平台: Ubuntu Linux (支持Jetson设备)"
echo "系统架构: $(uname -m)"
echo ""

# 检查是否为root权限运行
if [ "$EUID" -eq 0 ]; then
    log_warning "不建议使用root权限运行此脚本"
fi

# 更新包管理器
log_info "更新包管理器..."
sudo apt update

# 检查并安装基础构建工具
log_info "检查基础构建工具..."
BASIC_DEPS=(cmake build-essential git pkg-config)
MISSING_DEPS=()

for dep in "${BASIC_DEPS[@]}"; do
    if ! dpkg -l | grep -q "^ii.*$dep"; then
        MISSING_DEPS+=($dep)
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log_info "安装缺失的基础依赖: ${MISSING_DEPS[*]}"
    sudo apt install -y "${MISSING_DEPS[@]}"
fi

# 智能检测OpenCV
log_info "检测OpenCV..."
OPENCV_FOUND=false

# 方法1: pkg-config检查
if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    log_success "找到OpenCV4: $OPENCV_VERSION (pkg-config)"
    OPENCV_FOUND=true
elif pkg-config --exists opencv 2>/dev/null; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    log_success "找到OpenCV: $OPENCV_VERSION (pkg-config)"
    OPENCV_FOUND=true
# 方法2: 头文件检查
elif [ -f /usr/include/opencv4/opencv2/opencv.hpp ]; then
    log_success "找到OpenCV4头文件: /usr/include/opencv4/"
    OPENCV_FOUND=true
elif [ -f /usr/include/opencv2/opencv.hpp ]; then
    log_success "找到OpenCV头文件: /usr/include/"
    OPENCV_FOUND=true
elif [ -f /usr/local/include/opencv4/opencv2/opencv.hpp ]; then
    log_success "找到OpenCV4头文件: /usr/local/include/opencv4/"
    OPENCV_FOUND=true
elif [ -f /usr/local/include/opencv2/opencv.hpp ]; then
    log_success "找到OpenCV头文件: /usr/local/include/"
    OPENCV_FOUND=true
fi

# 如果未找到OpenCV，尝试安装
if [ "$OPENCV_FOUND" = false ]; then
    log_warning "未找到OpenCV，尝试安装..."
    
    # 检测是否为Jetson设备
    if [ "$(uname -m)" = "aarch64" ] && [ -d "/usr/local/cuda" ]; then
        log_info "检测到Jetson设备，安装Jetson优化的OpenCV包"
        sudo apt install -y \
            libopencv-dev \
            libopencv-contrib-dev \
            libopencv-imgproc-dev \
            libopencv-imgcodecs-dev \
            libopencv-videoio-dev \
            libopencv-calib3d-dev \
            libopencv-features2d-dev \
            libopencv-objdetect-dev \
            libopencv-dnn-dev
    else
        log_info "安装标准OpenCV开发包（包含扩展模块）"
        sudo apt install -y libopencv-dev libopencv-contrib-dev
    fi
    
    # 验证安装
    if pkg-config --exists opencv4 || pkg-config --exists opencv || [ -f /usr/include/opencv2/opencv.hpp ]; then
        log_success "OpenCV安装成功"
    else
        log_error "OpenCV安装失败，请手动安装或检查系统配置"
        exit 1
    fi
fi

# 检查并安装GStreamer
log_info "检查GStreamer..."
if ! pkg-config --exists gstreamer-1.0 2>/dev/null; then
    log_info "安装GStreamer开发包..."
    sudo apt install -y \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools
else
    GSTREAMER_VERSION=$(pkg-config --modversion gstreamer-1.0)
    log_success "找到GStreamer: $GSTREAMER_VERSION"
fi

# 检查其他有用的依赖
log_info "检查其他依赖包..."
OTHER_DEPS=(
    libv4l-dev          # V4L2摄像头支持
    libjpeg-dev         # JPEG支持
    libpng-dev          # PNG支持
    libtiff-dev         # TIFF支持
    libavcodec-dev      # 音视频编码
    libavformat-dev     # 音视频格式
    libswscale-dev      # 视频缩放
)

INSTALL_OTHER=()
for dep in "${OTHER_DEPS[@]}"; do
    if ! dpkg -l | grep -q "^ii.*$(echo $dep | sed 's/-dev$//')" 2>/dev/null; then
        INSTALL_OTHER+=($dep)
    fi
done

if [ ${#INSTALL_OTHER[@]} -ne 0 ]; then
    log_info "安装附加依赖: ${INSTALL_OTHER[*]}"
    sudo apt install -y "${INSTALL_OTHER[@]}" 2>/dev/null || log_warning "部分附加依赖安装失败，但不影响主要功能"
fi

# 检查CUDA (可选)
log_info "检查CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    log_success "检测到CUDA: $CUDA_VERSION，启用GPU加速"
    export ENABLE_CUDA=ON
    
    # 检查TensorRT (Jetson设备常有)
    if [ -f /usr/include/NvInfer.h ] || [ -f /usr/local/include/NvInfer.h ]; then
        log_success "检测到TensorRT，启用推理加速"
        export ENABLE_TENSORRT=ON
    fi
else
    log_info "未检测到CUDA，使用CPU模式"
    export ENABLE_CUDA=OFF
fi

# 检查系统资源
log_info "检查系统资源..."
TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
CPU_CORES=$(nproc)
log_info "系统内存: ${TOTAL_MEM}MB，CPU核心: ${CPU_CORES}"

# 根据系统资源调整编译参数
if [ "$TOTAL_MEM" -lt 2048 ]; then
    MAKE_JOBS=1
    log_warning "内存较少(<2GB)，使用单线程编译以避免内存不足"
elif [ "$TOTAL_MEM" -lt 4096 ]; then
    MAKE_JOBS=$((CPU_CORES / 2))
    log_info "内存适中(<4GB)，使用${MAKE_JOBS}线程编译"
else
    MAKE_JOBS=$CPU_CORES
    log_info "内存充足(>=4GB)，使用${MAKE_JOBS}线程编译"
fi

# 创建构建目录
log_info "准备构建环境..."
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
else
    log_info "清理旧的构建文件..."
    rm -rf "$BUILD_DIR"/*
fi

cd "$BUILD_DIR"

echo ""
log_info "配置CMake项目..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# 添加CUDA支持
if [ "$ENABLE_CUDA" = "ON" ]; then
    CMAKE_ARGS+=(-DENABLE_CUDA=ON)
fi

# 添加TensorRT支持
if [ "$ENABLE_TENSORRT" = "ON" ]; then
    CMAKE_ARGS+=(-DENABLE_TENSORRT=ON)
fi

# 执行CMake配置
if cmake .. "${CMAKE_ARGS[@]}"; then
    log_success "CMake配置成功"
else
    log_error "CMake配置失败"
    exit 1
fi

echo ""
log_info "开始编译项目..."
log_info "使用${MAKE_JOBS}个并行任务"

if make -j${MAKE_JOBS}; then
    log_success "编译完成"
else
    log_error "编译失败"
    exit 1
fi

echo ""
log_success "=== 构建完成 ==="
log_info "可执行文件位置: $(pwd)/bamboo_cut_backend"

# 验证可执行文件
if [ -f "bamboo_cut_backend" ]; then
    FILE_SIZE=$(ls -lh bamboo_cut_backend | awk '{print $5}')
    log_success "可执行文件大小: $FILE_SIZE"
    
    # 显示依赖信息
    log_info "依赖库检查:"
    ldd bamboo_cut_backend | head -5
    echo "  ... (更多依赖)"
else
    log_error "未找到可执行文件!"
    exit 1
fi

echo ""
log_info "运行快速测试..."
if make test 2>/dev/null; then
    log_success "测试通过"
else
    log_warning "测试未配置或失败（这通常是正常的）"
fi

echo ""
log_info "构建统计信息:"
echo "  - 系统: $(uname -s) $(uname -m)"
echo "  - 编译器: $(gcc --version | head -1)"
echo "  - CMake: $(cmake --version | head -1)"
echo "  - 构建时间: $(date)"

echo ""
log_info "后续操作:"
echo "  运行程序: cd $(pwd) && ./bamboo_cut_backend"
echo "  安装到系统: sudo make install"
echo "  清理构建: make clean"
echo "  重新构建: cd .. && ./build.sh"

echo ""
log_success "🎉 智能切竹机后端构建成功！"