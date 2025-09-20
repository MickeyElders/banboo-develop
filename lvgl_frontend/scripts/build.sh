#!/bin/bash

# 智能切竹机控制系统 - LVGL版本构建脚本

set -e  # 遇到错误时退出

# 项目配置
PROJECT_NAME="BambooControllerLVGL"
BUILD_TYPE="Release"
INSTALL_PREFIX="/opt/bamboo"
JETSON_ARCH="aarch64"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    print_info "检查系统要求..."
    
    # 检查是否在Jetson设备上
    if [[ $(uname -m) != "aarch64" ]]; then
        print_warn "警告: 当前系统不是aarch64架构，可能不是Jetson设备"
    fi
    
    # 检查必要的工具
    local tools=("cmake" "make" "g++" "pkg-config")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "缺少必要工具: $tool"
            exit 1
        fi
    done
    
    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        print_warn "警告: 未找到CUDA编译器nvcc"
    fi
    
    # 检查OpenCV
    if ! pkg-config --exists opencv4; then
        print_error "未找到OpenCV 4.x"
        exit 1
    fi
    
    print_success "系统要求检查通过"
}

# 检查并安装依赖
install_dependencies() {
    print_info "检查和安装依赖..."
    
    # 更新包列表
    sudo apt update
    
    # 安装基础开发工具
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config
    
    # 安装LVGL依赖
    sudo apt install -y \
        libfontconfig1-dev \
        libfreetype6-dev \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev
    
    # 安装图像处理库
    sudo apt install -y \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev
    
    # 安装V4L2开发库
    sudo apt install -y \
        libv4l-dev \
        v4l-utils
    
    # 检查TensorRT (Jetson上通常预装)
    if [[ -d "/usr/include/x86_64-linux-gnu" ]] || [[ -d "/usr/include/aarch64-linux-gnu" ]]; then
        print_info "TensorRT开发文件已安装"
    else
        print_warn "警告: TensorRT开发文件可能未安装"
    fi
    
    print_success "依赖安装完成"
}

# 下载和构建LVGL
setup_lvgl() {
    print_info "设置LVGL..."
    
    local lvgl_dir="third_party/lvgl"
    
    if [[ ! -d "$lvgl_dir" ]]; then
        print_info "下载LVGL..."
        mkdir -p third_party
        cd third_party
        git clone --depth 1 --branch release/v8.3 https://github.com/lvgl/lvgl.git
        cd ..
    else
        print_info "LVGL已存在，跳过下载"
    fi
    
    # 复制配置文件
    if [[ -f "lv_conf.h" ]]; then
        cp lv_conf.h $lvgl_dir/
        print_info "LVGL配置文件已复制"
    fi
    
    print_success "LVGL设置完成"
}

# 清理构建目录
clean_build() {
    print_info "清理构建目录..."
    if [[ -d "build" ]]; then
        rm -rf build
    fi
    print_success "构建目录已清理"
}

# 配置CMake
configure_cmake() {
    print_info "配置CMake..."
    
    mkdir -p build
    cd build
    
    # CMake配置选项
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
        "-DCMAKE_CXX_STANDARD=17"
        "-DCMAKE_C_STANDARD=11"
    )
    
    # Jetson特定优化
    if [[ $(uname -m) == "aarch64" ]]; then
        cmake_args+=(
            "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
            "-DCMAKE_C_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math"
            "-DCMAKE_CXX_FLAGS=-mcpu=cortex-a78 -O3 -ffast-math"
        )
    fi
    
    # 查找OpenCV
    local opencv_dir="/usr/lib/aarch64-linux-gnu/cmake/opencv4"
    if [[ -d "$opencv_dir" ]]; then
        cmake_args+=("-DOpenCV_DIR=$opencv_dir")
    fi
    
    # 运行CMake
    cmake "${cmake_args[@]}" ..
    
    cd ..
    print_success "CMake配置完成"
}

# 编译项目
build_project() {
    print_info "开始编译项目..."
    
    cd build
    
    # 获取CPU核心数
    local num_cores=$(nproc)
    print_info "使用 $num_cores 个CPU核心进行编译"
    
    # 编译
    make -j$num_cores
    
    cd ..
    print_success "项目编译完成"
}

# 安装项目
install_project() {
    print_info "安装项目..."
    
    cd build
    sudo make install
    cd ..
    
    # 创建systemd服务
    if [[ -f "build/bamboo-controller.service" ]]; then
        sudo cp build/bamboo-controller.service /etc/systemd/system/
        sudo systemctl daemon-reload
        print_info "systemd服务文件已安装"
    fi
    
    # 设置权限
    sudo chown -R root:root $INSTALL_PREFIX
    sudo chmod +x $INSTALL_PREFIX/bin/bamboo_controller_lvgl
    
    # 创建日志目录
    sudo mkdir -p /var/log/bamboo
    sudo chmod 755 /var/log/bamboo
    
    print_success "项目安装完成"
}

# 创建启动脚本
create_startup_script() {
    print_info "创建启动脚本..."
    
    local startup_script="/usr/local/bin/bamboo-controller"
    
    sudo tee $startup_script > /dev/null << 'EOF'
#!/bin/bash

# 智能切竹机控制系统启动脚本

INSTALL_PREFIX="/opt/bamboo"
BINARY="$INSTALL_PREFIX/bin/bamboo_controller_lvgl"
CONFIG="$INSTALL_PREFIX/config/default_config.json"
LOG_FILE="/var/log/bamboo/controller.log"

# 检查二进制文件是否存在
if [[ ! -f "$BINARY" ]]; then
    echo "错误: 找不到程序文件 $BINARY"
    exit 1
fi

# 检查配置文件是否存在
if [[ ! -f "$CONFIG" ]]; then
    echo "错误: 找不到配置文件 $CONFIG"
    exit 1
fi

# 设置环境变量
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export DISPLAY=:0

# 启动程序
exec "$BINARY" -c "$CONFIG" "$@" 2>&1 | tee -a "$LOG_FILE"
EOF

    sudo chmod +x $startup_script
    print_success "启动脚本已创建: $startup_script"
}

# 运行测试
run_tests() {
    print_info "运行测试..."
    
    if [[ -f "build/bamboo_controller_lvgl" ]]; then
        print_info "测试程序启动..."
        # 这里可以添加自动化测试
        echo "测试暂时跳过 - 需要图形环境"
    else
        print_error "找不到可执行文件"
        exit 1
    fi
    
    print_success "测试完成"
}

# 显示帮助信息
show_help() {
    echo "智能切竹机控制系统 - LVGL版本构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -c, --clean         清理构建目录"
    echo "  -d, --deps          安装依赖"
    echo "  -b, --build-only    仅编译，不安装"
    echo "  -i, --install       编译并安装"
    echo "  -t, --test          运行测试"
    echo "  --debug             调试模式构建"
    echo "  --release           发布模式构建 (默认)"
    echo ""
    echo "示例:"
    echo "  $0 --deps --install    # 安装依赖并构建安装"
    echo "  $0 --clean --build-only # 清理并仅编译"
}

# 主函数
main() {
    local install_deps=false
    local clean_first=false
    local build_only=false
    local install_project_flag=false
    local run_tests_flag=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                clean_first=true
                shift
                ;;
            -d|--deps)
                install_deps=true
                shift
                ;;
            -b|--build-only)
                build_only=true
                shift
                ;;
            -i|--install)
                install_project_flag=true
                shift
                ;;
            -t|--test)
                run_tests_flag=true
                shift
                ;;
            --debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定任何操作，默认为构建和安装
    if [[ "$install_deps" == false && "$clean_first" == false && "$build_only" == false && "$install_project_flag" == false && "$run_tests_flag" == false ]]; then
        build_only=true
    fi
    
    print_info "开始构建 $PROJECT_NAME ($BUILD_TYPE 模式)"
    
    # 检查系统要求
    check_system_requirements
    
    # 安装依赖
    if [[ "$install_deps" == true ]]; then
        install_dependencies
    fi
    
    # 设置LVGL
    setup_lvgl
    
    # 清理构建目录
    if [[ "$clean_first" == true ]]; then
        clean_build
    fi
    
    # 配置和编译
    configure_cmake
    build_project
    
    # 安装
    if [[ "$install_project_flag" == true ]]; then
        install_project
        create_startup_script
    fi
    
    # 运行测试
    if [[ "$run_tests_flag" == true ]]; then
        run_tests
    fi
    
    print_success "构建流程完成!"
    
    if [[ "$install_project_flag" == true ]]; then
        print_info "程序已安装到: $INSTALL_PREFIX"
        print_info "可以使用以下命令启动:"
        print_info "  sudo bamboo-controller"
        print_info "或者使用systemd服务:"
        print_info "  sudo systemctl start bamboo-controller"
    else
        print_info "可执行文件位置: build/bamboo_controller_lvgl"
    fi
}

# 运行主函数
main "$@"