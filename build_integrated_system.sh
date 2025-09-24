#!/bin/bash

# 竹子识别系统一体化构建脚本
# 实际整合现有的cpp_backend和lvgl_frontend代码

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build_integrated"
INSTALL_PREFIX="/opt/bamboo-cut"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}\n"
}

# 检查必要文件
check_prerequisites() {
    print_header "检查整合前提条件"
    
    local missing_files=()
    
    # 检查关键文件是否存在
    if [[ ! -f "$PROJECT_ROOT/integrated_main.cpp" ]]; then
        missing_files+=("integrated_main.cpp")
    fi
    
    if [[ ! -f "$PROJECT_ROOT/CMakeLists_integrated.txt" ]]; then
        missing_files+=("CMakeLists_integrated.txt")
    fi
    
    if [[ ! -d "$PROJECT_ROOT/cpp_backend/src" ]]; then
        missing_files+=("cpp_backend/src/")
    fi
    
    if [[ ! -d "$PROJECT_ROOT/lvgl_frontend/src" ]]; then
        missing_files+=("lvgl_frontend/src/")
    fi
    
    if [[ ! -d "$PROJECT_ROOT/cpp_backend/include" ]]; then
        missing_files+=("cpp_backend/include/")
    fi
    
    if [[ ! -d "$PROJECT_ROOT/lvgl_frontend/include" ]]; then
        missing_files+=("lvgl_frontend/include/")
    fi
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "缺少以下关键文件:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        echo
        print_error "请确保所有源文件都存在"
        exit 1
    fi
    
    print_success "前提条件检查完成"
}

# 检查依赖
check_dependencies() {
    print_header "检查构建依赖"
    
    local missing_deps=()
    
    # 基础工具
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi
    
    if ! command -v g++ &> /dev/null; then
        missing_deps+=("g++")
    fi
    
    # OpenCV检查
    if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
        missing_deps+=("libopencv-dev")
    fi
    
    # GStreamer检查
    if ! pkg-config --exists gstreamer-1.0; then
        missing_deps+=("libgstreamer1.0-dev")
    fi
    
    if ! pkg-config --exists gstreamer-app-1.0; then
        missing_deps+=("libgstreamer-plugins-base1.0-dev")
    fi
    
    # LVGL库检查
    if [[ ! -d "$PROJECT_ROOT/lvgl_frontend/lvgl" ]]; then
        print_warning "LVGL库目录不存在，将尝试下载..."
        download_lvgl
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "缺少以下依赖包:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo
        print_info "Ubuntu/Debian安装命令:"
        echo "sudo apt update"
        echo "sudo apt install ${missing_deps[*]}"
        echo
        exit 1
    fi
    
    print_success "依赖检查完成"
}

# 下载LVGL库
download_lvgl() {
    local lvgl_dir="$PROJECT_ROOT/lvgl_frontend/lvgl"
    
    if [[ ! -d "$lvgl_dir" ]]; then
        print_info "下载LVGL库..."
        
        if command -v git &> /dev/null; then
            cd "$PROJECT_ROOT/lvgl_frontend"
            git clone --depth 1 --branch release/v8.3 https://github.com/lvgl/lvgl.git
            cd "$PROJECT_ROOT"
            print_success "LVGL库下载完成"
        else
            print_error "需要git命令来下载LVGL库"
            exit 1
        fi
    fi
}

# 创建构建目录
setup_build_directory() {
    print_header "设置构建环境"
    
    if [[ -d "$BUILD_DIR" ]]; then
        print_info "清理旧的构建目录: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    print_success "构建目录创建完成: $BUILD_DIR"
}

# 配置构建
configure_build() {
    print_header "配置CMake构建"
    
    cd "$BUILD_DIR"
    
    # CMake配置参数
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )
    
    # 检测架构
    local arch=$(uname -m)
    if [[ "$arch" == "aarch64" ]]; then
        cmake_args+=("-DTARGET_JETSON=ON")
        print_info "检测到ARM64架构 (Jetson平台)"
    else
        print_info "检测到x86_64架构"
    fi
    
    # 检查CUDA
    if command -v nvcc &> /dev/null; then
        cmake_args+=("-DCUDA_FOUND=ON")
        print_info "CUDA支持: 启用"
    else
        print_info "CUDA支持: 未找到"
    fi
    
    print_info "CMake配置参数:"
    for arg in "${cmake_args[@]}"; do
        echo "  $arg"
    done
    echo
    
    # 执行CMake配置
    cmake "${cmake_args[@]}" -f "$PROJECT_ROOT/CMakeLists_integrated.txt" "$PROJECT_ROOT"
    
    print_success "CMake配置完成"
}

# 执行构建
build_project() {
    print_header "编译一体化系统"
    
    cd "$BUILD_DIR"
    
    # 检测CPU核心数
    local cpu_cores=$(nproc)
    local make_jobs=$((cpu_cores > 4 ? cpu_cores - 1 : cpu_cores))
    
    print_info "使用 $make_jobs 个并行任务进行编译"
    print_info "这可能需要几分钟时间..."
    
    # 执行构建
    make -j"$make_jobs"
    
    print_success "编译完成"
}

# 测试构建结果
test_build() {
    print_header "测试构建结果"
    
    cd "$BUILD_DIR"
    
    if [[ ! -f "./bamboo_integrated" ]]; then
        print_error "可执行文件未找到"
        exit 1
    fi
    
    print_info "检查可执行文件信息:"
    file ./bamboo_integrated
    
    print_info "检查动态链接库依赖:"
    ldd ./bamboo_integrated | head -15
    
    print_info "检查文件大小:"
    ls -lh ./bamboo_integrated
    
    print_success "构建测试通过"
}

# 创建启动脚本
create_startup_script() {
    print_header "创建启动脚本"
    
    cat > "$BUILD_DIR/start_integrated.sh" << 'EOF'
#!/bin/bash

# 竹子识别系统一体化启动脚本

# 设置环境变量
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda/bin:$PATH"

# 切换到程序目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查设备权限
echo "检查设备权限..."
for device in /dev/video0 /dev/video1 /dev/fb0 /dev/input/event*; do
    if [[ -e "$device" ]]; then
        if [[ -r "$device" && -w "$device" ]]; then
            echo "✓ $device 权限正常"
        else
            echo "⚠ $device 权限不足，请运行: sudo chmod 666 $device"
        fi
    fi
done

echo "启动一体化系统..."
echo "按 Ctrl+C 退出"
echo

# 启动程序
./bamboo_integrated "$@"
EOF
    
    chmod +x "$BUILD_DIR/start_integrated.sh"
    print_success "启动脚本创建完成"
}

# 创建部署包
create_deployment_package() {
    print_header "创建部署包"
    
    cd "$BUILD_DIR"
    
    local arch=$(uname -m)
    local package_name="bamboo-integrated-v3.0.0-${arch}"
    local package_dir="$BUILD_DIR/$package_name"
    
    mkdir -p "$package_dir"/{bin,config,scripts,docs}
    
    # 复制文件
    cp ./bamboo_integrated "$package_dir/bin/"
    cp ./start_integrated.sh "$package_dir/scripts/"
    
    # 复制配置文件
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config"/* "$package_dir/config/" 2>/dev/null || true
    fi
    
    # 创建说明文档
    cat > "$package_dir/README.md" << EOF
# 竹子识别系统一体化版本

## 版本信息
- 版本: 3.0.0
- 架构: $arch
- 构建时间: $(date)
- 构建主机: $(hostname)

## 系统要求
- Ubuntu 18.04+ 或 JetPack 4.6+
- OpenCV 4.2+
- GStreamer 1.14+
- 至少2GB内存
- 支持的摄像头设备

## 快速启动
\`\`\`bash
# 进入程序目录
cd bin/

# 直接启动
./bamboo_integrated

# 或使用启动脚本 (推荐)
../scripts/start_integrated.sh
\`\`\`

## 配置文件
- 系统配置: config/integrated_system_config.yaml
- 摄像头标定: config/stereo_calibration.xml
- AI优化: config/ai_optimization.yaml

## 设备权限
如果遇到设备权限问题，请运行:
\`\`\`bash
sudo chmod 666 /dev/video0 /dev/video1 /dev/fb0 /dev/input/event*
\`\`\`

## 系统服务 (可选)
可以将系统配置为自动启动服务。
EOF
    
    # 打包
    tar -czf "${package_name}.tar.gz" "$package_name"
    
    print_success "部署包创建完成: ${package_name}.tar.gz"
}

# 显示完成信息
show_completion_info() {
    print_header "整合完成"
    
    local arch=$(uname -m)
    local package_name="bamboo-integrated-v3.0.0-${arch}"
    
    print_success "🎉 竹子识别系统一体化整合成功!"
    echo
    print_info "📁 构建输出:"
    echo "  可执行文件: $BUILD_DIR/bamboo_integrated"
    echo "  启动脚本: $BUILD_DIR/start_integrated.sh"
    echo "  部署包: $BUILD_DIR/${package_name}.tar.gz"
    echo
    print_info "🚀 快速启动:"
    echo "  cd $BUILD_DIR"
    echo "  ./start_integrated.sh"
    echo
    print_info "📋 系统特性:"
    echo "  ✅ 完全整合的单一进程"
    echo "  ✅ 线程安全的数据交换"
    echo "  ✅ 复用所有现有代码"
    echo "  ✅ 性能优化和稳定性提升"
    echo "  ✅ 支持优雅关闭"
    echo
    print_info "🔧 下一步:"
    echo "  1. 测试运行: $BUILD_DIR/start_integrated.sh"
    echo "  2. 检查日志输出确认功能正常"
    echo "  3. 根据需要调整配置文件"
    echo "  4. 部署到生产环境"
    echo
    print_warning "⚠️ 注意事项:"
    echo "  - 确保摄像头设备权限正确"
    echo "  - 检查显示和触摸设备可用性"
    echo "  - 首次运行建议在终端中启动以观察日志"
}

# 主函数
main() {
    print_header "竹子识别系统一体化构建"
    
    # 执行构建流程
    check_prerequisites
    check_dependencies
    setup_build_directory
    configure_build
    build_project
    test_build
    create_startup_script
    create_deployment_package
    show_completion_info
    
    print_success "✨ 整合过程完全完成!"
}

# 运行主函数
main "$@"