#!/bin/bash

# 智能切竹机后端构建脚本
# 支持自动下载和构建依赖库

set -e  # 遇到错误立即退出

echo "=== 智能切竹机后端构建脚本 ==="
echo "目标平台: Ubuntu Linux"
echo ""

# 检查必要的系统依赖
echo "检查系统依赖..."
if ! command -v cmake &> /dev/null; then
    echo "安装 CMake..."
    sudo apt update
    sudo apt install -y cmake build-essential
fi

if ! command -v git &> /dev/null; then
    echo "安装 Git..."
    sudo apt install -y git
fi

# 检查 OpenCV
if ! pkg-config --exists opencv4; then
    echo "安装 OpenCV..."
    sudo apt install -y libopencv-dev
fi

# 检查 GStreamer
if ! pkg-config --exists gstreamer-1.0; then
    echo "安装 GStreamer..."
    sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                       gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                       gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
                       gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x \
                       gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
                       gstreamer1.0-qt5 gstreamer1.0-pulseaudio
fi

# 检查 CUDA (可选)
if command -v nvcc &> /dev/null; then
    echo "检测到 CUDA，启用 GPU 加速"
    export ENABLE_CUDA=ON
else
    echo "未检测到 CUDA，使用 CPU 模式"
    export ENABLE_CUDA=OFF
fi

# 创建构建目录
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo ""
echo "配置 CMake 项目..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=$ENABLE_CUDA \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "编译项目..."
make -j$(nproc)

echo ""
echo "=== 构建完成 ==="
echo "可执行文件位置: $BUILD_DIR/bamboo_cut_backend"
echo ""
echo "运行测试..."
make test

echo ""
echo "安装到系统 (可选)..."
echo "运行: sudo make install"
echo ""
echo "清理构建文件..."
echo "运行: make clean" 