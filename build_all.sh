#!/bin/bash

# 智能切竹机统一编译脚本
# 支持同时编译C++后端和Qt前端

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo "智能切竹机统一编译脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -c, --cpp-only          仅编译C++后端"
    echo "  -q, --qt-only           仅编译Qt前端"
    echo "  -a, --all              编译所有组件 (默认)"
    echo "  -d, --debug            调试模式编译"
    echo "  -r, --release          发布模式编译 (默认)"
    echo "  -j, --jobs N           并行编译任务数 (默认: 4)"
    echo "  -p, --platform PLAT    目标平台 (linux, windows, android)"
    echo "  -t, --target TARGET    编译目标 (desktop, embedded, mobile)"
    echo ""
    echo "示例:"
    echo "  $0                     # 编译所有组件 (发布模式)"
    echo "  $0 -d                  # 调试模式编译所有组件"
    echo "  $0 -c -d               # 仅编译C++后端 (调试模式)"
    echo "  $0 -q                  # 仅编译Qt前端"
    echo "  $0 -a -j 8 -t embedded # 编译所有组件 (8线程, 嵌入式目标)"
}

# 检查依赖
check_dependencies() {
    log_info "检查编译依赖..."
    
    # 检查基本工具
    local missing_deps=()
    
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi
    
    if ! command -v g++ &> /dev/null; then
        missing_deps+=("g++")
    fi
    
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    # 检查Qt (如果需要编译前端)
    if [[ "$COMPILE_QT" == "true" ]]; then
        if ! command -v qmake &> /dev/null && ! command -v qmake6 &> /dev/null; then
            missing_deps+=("qt6-base-dev")
        fi
        
        # 检查Qt6依赖
        if ! pkg-config --exists Qt6Core &> /dev/null; then
            log_warning "Qt6开发包未找到，请安装qt6-base-dev"
        fi
    fi
    
    # 检查OpenCV
    if ! pkg-config --exists opencv4; then
        if ! pkg-config --exists opencv; then
            log_warning "OpenCV未找到，C++后端编译可能失败"
        fi
    fi
    
    # 检查GStreamer
    if ! pkg-config --exists gstreamer-1.0; then
        log_warning "GStreamer未找到，某些功能可能不可用"
    fi
    
    # 检查CUDA (可选)
    if command -v nvcc &> /dev/null; then
        log_info "CUDA编译器已找到"
    else
        log_warning "CUDA未找到，GPU加速功能将不可用"
    fi
    
    # 报告缺失的依赖
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "缺失以下依赖: ${missing_deps[*]}"
        echo "请安装缺失的依赖后重试"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 编译C++后端
compile_cpp_backend() {
    log_info "开始编译C++后端..."
    
    cd cpp_backend
    
    # 创建构建目录
    if [[ "$BUILD_TYPE" == "debug" ]]; then
        mkdir -p build_debug
        cd build_debug
        CMAKE_BUILD_TYPE="Debug"
    else
        mkdir -p build
        cd build
        CMAKE_BUILD_TYPE="Release"
    fi
    
    # 配置CMake
    log_info "配置CMake (${CMAKE_BUILD_TYPE}模式)..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_FLAGS="-std=c++17" \
        -DCMAKE_C_FLAGS="-std=c99"
    
    # 编译
    log_info "编译C++后端 (使用${JOBS}个线程)..."
    make -j${JOBS}
    
    # 检查编译结果
    if [[ -f "bamboo_cut_backend" ]]; then
        log_success "C++后端编译成功"
        # 显示文件信息
        ls -lh bamboo_cut_backend
    else
        log_error "C++后端编译失败"
        exit 1
    fi
    
    cd ../..
}

# 编译Qt前端
compile_qt_frontend() {
    log_info "开始编译Qt前端..."
    
    cd qt_frontend
    
    # 创建构建目录
    if [[ "$BUILD_TYPE" == "debug" ]]; then
        mkdir -p build_debug
        cd build_debug
        CMAKE_BUILD_TYPE="Debug"
    else
        mkdir -p build
        cd build
        CMAKE_BUILD_TYPE="Release"
    fi
    
    # 配置CMake
    log_info "配置Qt CMake (${CMAKE_BUILD_TYPE}模式)..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_PREFIX_PATH="/usr/lib/qt6"
    
    # 编译Qt前端
    log_info "编译Qt前端 (使用${JOBS}个线程)..."
    make -j${JOBS}
    
    # 检查编译结果
    if [[ -f "bamboo_controller_qt" ]]; then
        log_success "Qt前端编译成功"
        ls -lh bamboo_controller_qt
    else
        log_error "Qt前端编译失败"
        exit 1
    fi
    
    cd ../..
}

# 编译嵌入式版本
compile_embedded() {
    log_info "编译嵌入式版本..."
    
    # 编译C++后端 (嵌入式配置)
    cd cpp_backend
    
    mkdir -p build_embedded
    cd build_embedded
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DTARGET_ARCH=aarch64 \
        -DENABLE_EMBEDDED=ON \
        -DENABLE_GUI=OFF \
        -DENABLE_NETWORK=ON
    
    make -j${JOBS}
    
    cd ../..
    
    # 编译Qt嵌入式版本
    cd qt_frontend
    
    mkdir -p build_embedded
    cd build_embedded
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DTARGET_ARCH=aarch64 \
        -DENABLE_EMBEDDED=ON \
        -DCMAKE_PREFIX_PATH="/usr/lib/qt6"
    
    make -j${JOBS}
    
    cd ../..
}

# 创建部署包
create_deployment_package() {
    log_info "创建部署包..."
    
    local package_name="bamboo_cut_${BUILD_TYPE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p ${package_name}
    
    # 复制C++后端
    if [[ "$COMPILE_CPP" == "true" ]]; then
        mkdir -p ${package_name}/backend
        if [[ "$BUILD_TYPE" == "debug" ]]; then
            cp cpp_backend/build_debug/bamboo_cut_backend ${package_name}/backend/
        else
            cp cpp_backend/build/bamboo_cut_backend ${package_name}/backend/
        fi
        cp cpp_backend/config/*.yaml ${package_name}/backend/ 2>/dev/null || true
    fi
    
    # 复制Qt前端
    if [[ "$COMPILE_QT" == "true" ]]; then
        mkdir -p ${package_name}/frontend
        if [[ "$BUILD_TYPE" == "debug" ]]; then
            cp qt_frontend/build_debug/bamboo_controller_qt ${package_name}/frontend/ 2>/dev/null || true
        else
            cp qt_frontend/build/bamboo_controller_qt ${package_name}/frontend/ 2>/dev/null || true
        fi
        # 复制QML文件和资源
        cp -r qt_frontend/qml ${package_name}/frontend/ 2>/dev/null || true
        cp -r qt_frontend/resources ${package_name}/frontend/ 2>/dev/null || true
    fi
    
    # 复制文档和脚本
    cp README.md ${package_name}/ 2>/dev/null || true
    cp *.md ${package_name}/ 2>/dev/null || true
    cp scripts/*.sh ${package_name}/ 2>/dev/null || true
    
    # 创建启动脚本
    cat > ${package_name}/start.sh << 'EOF'
#!/bin/bash
# 智能切竹机启动脚本

echo "启动智能切竹机系统..."

# 启动后端服务
if [[ -f "./backend/bamboo_cut_backend" ]]; then
    echo "启动C++后端服务..."
    ./backend/bamboo_cut_backend &
    BACKEND_PID=$!
    echo "后端服务PID: $BACKEND_PID"
fi

# 启动前端界面
if [[ -d "./frontend" ]]; then
    echo "启动Qt前端..."
    cd frontend
    ./bamboo_controller_qt &
    FRONTEND_PID=$!
    echo "前端服务PID: $FRONTEND_PID"
    cd ..
fi

echo "系统启动完成"
echo "按Ctrl+C停止服务"

# 等待用户中断
trap 'echo "正在停止服务..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
EOF
    
    chmod +x ${package_name}/start.sh
    
    # 创建压缩包
    tar -czf ${package_name}.tar.gz ${package_name}
    
    log_success "部署包创建完成: ${package_name}.tar.gz"
    rm -rf ${package_name}
}

# 主函数
main() {
    # 默认参数
    COMPILE_CPP="true"
    COMPILE_QT="true"
    BUILD_TYPE="release"
    JOBS=4
    TARGET_PLATFORM="linux"
    TARGET_TYPE="desktop"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--cpp-only)
                COMPILE_CPP="true"
                COMPILE_FLUTTER="false"
                shift
                ;;
            -q|--qt-only)
                COMPILE_CPP="false"
                COMPILE_QT="true"
                shift
                ;;
            -a|--all)
                COMPILE_CPP="true"
                COMPILE_QT="true"
                shift
                ;;
            -d|--debug)
                BUILD_TYPE="debug"
                shift
                ;;
            -r|--release)
                BUILD_TYPE="release"
                shift
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            -p|--platform)
                TARGET_PLATFORM="$2"
                shift 2
                ;;
            -t|--target)
                TARGET_TYPE="$2"
                shift 2
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示编译配置
    log_info "编译配置:"
    echo "  C++后端: $COMPILE_CPP"
    echo "  Qt前端: $COMPILE_QT"
    echo "  编译模式: $BUILD_TYPE"
    echo "  并行任务: $JOBS"
    echo "  目标平台: $TARGET_PLATFORM"
    echo "  目标类型: $TARGET_TYPE"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 根据目标类型选择编译方式
    if [[ "$TARGET_TYPE" == "embedded" ]]; then
        compile_embedded
    else
        # 编译C++后端
        if [[ "$COMPILE_CPP" == "true" ]]; then
            compile_cpp_backend
        fi
        
        # 编译Qt前端
        if [[ "$COMPILE_QT" == "true" ]]; then
            compile_qt_frontend
        fi
    fi
    
    # 创建部署包
    create_deployment_package
    
    # 计算编译时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "编译完成！总耗时: ${duration}秒"
    
    # 显示结果
    echo ""
    log_info "编译结果:"
    if [[ "$COMPILE_CPP" == "true" ]]; then
        if [[ "$BUILD_TYPE" == "debug" ]]; then
            ls -lh cpp_backend/build_debug/bamboo_cut_backend 2>/dev/null || echo "C++后端文件未找到"
        else
            ls -lh cpp_backend/build/bamboo_cut_backend 2>/dev/null || echo "C++后端文件未找到"
        fi
    fi
    
    if [[ "$COMPILE_QT" == "true" ]]; then
        if [[ "$BUILD_TYPE" == "debug" ]]; then
            ls -lh qt_frontend/build_debug/bamboo_controller_qt 2>/dev/null || echo "Qt前端文件未找到"
        else
            ls -lh qt_frontend/build/bamboo_controller_qt 2>/dev/null || echo "Qt前端文件未找到"
        fi
    fi
}

# 运行主函数
main "$@" 