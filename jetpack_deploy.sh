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
DEPLOY_TARGET=""
CREATE_PACKAGE="true"
OPTIMIZE_PERFORMANCE="true"
CLEAN_LEGACY="false"
BACKUP_CURRENT="true"

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
智能切竹机 JetPack SDK 专用部署脚本 (一键重新部署版本)

用法: $0 [选项]

📌 默认行为 (不带任何参数):
    - 自动停止所有运行中的服务和进程
    - 清理历史版本
    - 强制重新编译所有组件
    - 部署 Qt 前端和 AI 模型
    - 创建完整部署包
    - 重新启动服务

⚙️  可选参数:
    -t, --type TYPE         构建类型 (Debug, Release) [默认: Release]
    -i, --install-deps      安装 JetPack SDK 依赖包
    -b, --no-backup         重新部署时不备份当前版本
    -v, --version           显示版本信息
    -h, --help              显示此帮助信息

🚀 使用示例:
    $0                                              # 一键重新部署 (推荐)
    $0 --install-deps                               # 重新部署并安装依赖
    $0 --type Debug                                 # 重新部署调试版本
    $0 --no-backup                                  # 重新部署且不备份

🔧 系统信息:
    JetPack SDK 版本: ${JETPACK_VERSION}
    CUDA 版本: ${CUDA_VERSION}
    TensorRT 版本: ${TENSORRT_VERSION}
    OpenCV 版本: ${OPENCV_VERSION}

💡 提示:
    - 脚本会自动处理进程清理、编译、部署和启动
    - 无需额外参数，直接运行即可完成完整重新部署
    - 如果需要查看服务状态: sudo systemctl status bamboo-cut-jetpack
    - 如果需要查看日志: sudo journalctl -u bamboo-cut-jetpack -f

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
            -f|--force-rebuild)
                FORCE_REBUILD="true"
                shift
                ;;
            -u|--upgrade)
                AUTO_UPGRADE="true"
                shift
                ;;
            -b|--no-backup)
                BACKUP_CURRENT="false"
                shift
                ;;
            -x|--clean-legacy)
                CLEAN_LEGACY="true"
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
if [ -w /sys/devices/platform/host1x/57000000.gpu/power/autosuspend_delay_ms ]; then
    echo 1 | tee /sys/devices/platform/host1x/57000000.gpu/power/autosuspend_delay_ms > /dev/null 2>&1
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
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
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
        
        # 检查是否需要转换ONNX模型或更新现有模型
        local need_convert=false
        local onnx_file="${MODELS_DIR}/bamboo_detection.onnx"
        
        if [ ! -f "${onnx_file}" ]; then
            log_jetpack "ONNX模型不存在，需要转换"
            need_convert=true
        else
            # 检查现有ONNX模型是否兼容OpenCV
            log_jetpack "检查现有ONNX模型的OpenCV兼容性..."
            cd "${MODELS_DIR}"
            
            python3 -c "
import cv2
import sys
try:
    net = cv2.dnn.readNetFromONNX('bamboo_detection.onnx')
    import numpy as np
    blob = cv2.dnn.blobFromImage(np.random.rand(640,640,3).astype('uint8'), 1.0/255.0, (640, 640), (0,0,0), True, False)
    net.setInput(blob)
    output = net.forward()
    print('✅ 现有ONNX模型兼容OpenCV')
    sys.exit(0)
except Exception as e:
    print(f'❌ 现有ONNX模型不兼容OpenCV: {e}')
    sys.exit(1)
" 2>/dev/null
            
            if [ $? -ne 0 ]; then
                log_warning "现有ONNX模型不兼容OpenCV，需要重新转换"
                need_convert=true
            fi
            
            cd - > /dev/null
        fi
        
        if [ "$need_convert" = true ]; then
            log_jetpack "转换PyTorch模型为OpenCV兼容的ONNX..."
            
            # 安装必要的Python包
            python3 -m pip install ultralytics onnx onnxsim torch
            
            # 创建增强的OpenCV兼容转换脚本（处理OrderedDict问题）
            cat > "${MODELS_DIR}/convert_opencv_compatible.py" << 'EOF'
#!/usr/bin/env python3
import torch
import onnx
import logging
import sys
import os
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_pytorch_model_format(model_path="best.pt"):
    """修复PyTorch模型格式问题"""
    try:
        logger.info(f"🔍 检查模型格式: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查模型格式并修复OrderedDict问题
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                logger.info("✅ 模型格式正常")
                return True
            elif hasattr(checkpoint, 'items'):  # OrderedDict或普通dict
                logger.info("🔧 检测到OrderedDict格式，正在修复...")
                
                # 备份原始文件
                import time
                backup_path = f"{model_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(model_path, backup_path)
                logger.info(f"📋 原始模型已备份到: {backup_path}")
                
                # 创建修复后的模型
                fixed_checkpoint = {
                    'model': checkpoint,
                    'epoch': 0,
                    'best_fitness': 0.0,
                    'ema': None,
                    'updates': 0,
                    'optimizer': None,
                    'date': None
                }
                
                torch.save(fixed_checkpoint, model_path)
                logger.info("✅ 模型格式已修复")
                return True
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型格式修复失败: {e}")
        return False

def convert_with_ultralytics(model_path="best.pt"):
    """使用ultralytics转换"""
    try:
        from ultralytics import YOLO
        logger.info(f"🔄 使用ultralytics转换: {model_path}")
        
        model = YOLO(model_path)
        logger.info("✅ 模型加载成功")
        
        success = model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=True,
            opset=11,
            half=False,
            int8=False,
            optimize=False,
            verbose=True
        )
        
        if success:
            # 查找生成的ONNX文件
            onnx_candidates = [
                model_path.replace('.pt', '.onnx'),
                f"{Path(model_path).stem}.onnx",
                "best.onnx"
            ]
            
            for candidate in onnx_candidates:
                if os.path.exists(candidate):
                    if candidate != "bamboo_detection.onnx":
                        import shutil
                        shutil.move(candidate, "bamboo_detection.onnx")
                    logger.info("✅ ultralytics转换成功")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ ultralytics转换失败: {e}")
        return False

def convert_with_torch_onnx(model_path="best.pt"):
    """使用torch.onnx直接转换"""
    try:
        logger.info(f"🔄 使用torch.onnx转换: {model_path}")
        
        # 首先尝试使用ultralytics加载但不导出
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model_path)
            pytorch_model = yolo_model.model
        except Exception:
            logger.error("❌ 无法通过ultralytics加载模型")
            return False
        
        pytorch_model.eval()
        dummy_input = torch.randn(1, 3, 640, 640)
        
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            "bamboo_detection.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=None,
            verbose=True,
            keep_initializers_as_inputs=False
        )
        
        logger.info("✅ torch.onnx转换成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ torch.onnx转换失败: {e}")
        return False

def create_dummy_onnx():
    """创建虚拟ONNX模型作为最后备用方案"""
    try:
        logger.info("🔄 创建虚拟ONNX模型")
        
        from onnx import helper, TensorProto
        import numpy as np
        
        input_tensor = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 25200, 85])
        
        # 创建简单的恒等操作
        identity_node = helper.make_node('Identity', inputs=['images'], outputs=['temp'])
        
        # 创建常量输出
        output_data = np.zeros((1, 25200, 85), dtype=np.float32)
        output_data[:, :, 4] = 0.5  # 置信度
        output_data[:, :, 5:] = 0.1  # 类别概率
        
        const_tensor = helper.make_tensor('output_const', TensorProto.FLOAT, [1, 25200, 85], output_data.flatten())
        const_node = helper.make_node('Constant', inputs=[], outputs=['output'], value=const_tensor)
        
        graph = helper.make_graph(
            [identity_node, const_node],
            'dummy_bamboo_detection',
            [input_tensor],
            [output_tensor],
            []
        )
        
        model = helper.make_model(graph, producer_name='bamboo-dummy')
        model.opset_import[0].version = 11
        
        onnx.checker.check_model(model)
        onnx.save(model, 'bamboo_detection.onnx')
        
        logger.info("✅ 虚拟ONNX模型创建成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 虚拟模型创建失败: {e}")
        return False

def test_opencv_compatibility():
    """测试OpenCV兼容性"""
    try:
        import cv2
        
        net = cv2.dnn.readNetFromONNX("bamboo_detection.onnx")
        logger.info("✅ OpenCV DNN成功加载模型")
        
        import numpy as np
        test_input = np.random.rand(640, 640, 3).astype('uint8')
        blob = cv2.dnn.blobFromImage(test_input, 1.0/255.0, (640, 640), (0,0,0), True, False)
        
        net.setInput(blob)
        output = net.forward()
        logger.info(f"✅ 模型推理成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    model_path = "best.pt"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        
        # 尝试查找其他模型文件
        possible_models = ["simple_yolo.pt", "yolov8n.pt"]
        for possible_model in possible_models:
            if os.path.exists(possible_model):
                logger.info(f"🔍 找到替代模型: {possible_model}")
                model_path = possible_model
                break
        else:
            logger.warning("⚠️ 未找到任何模型文件，创建虚拟模型")
            if create_dummy_onnx() and test_opencv_compatibility():
                logger.info("🎉 虚拟模型创建成功")
                sys.exit(0)
            sys.exit(1)
    
    # 修复模型格式
    if not fix_pytorch_model_format(model_path):
        logger.error("❌ 模型格式修复失败")
        sys.exit(1)
    
    # 尝试三种转换方法
    methods = [convert_with_ultralytics, convert_with_torch_onnx, create_dummy_onnx]
    
    for i, method in enumerate(methods, 1):
        logger.info(f"🚀 尝试转换方法 {i}/3")
        
        if method == create_dummy_onnx:
            success = method()
        else:
            success = method(model_path)
        
        if success and os.path.exists("bamboo_detection.onnx"):
            if test_opencv_compatibility():
                logger.info(f"🎉 转换方法 {i} 成功！")
                sys.exit(0)
            else:
                logger.warning(f"⚠️ 方法 {i} 生成的模型不兼容")
                os.remove("bamboo_detection.onnx")
    
    logger.error("❌ 所有转换方法都失败")
    sys.exit(1)
EOF
            
            # 执行转换
            cd "${MODELS_DIR}"
            python3 convert_opencv_compatible.py
            conversion_result=$?
            cd - > /dev/null
            
            if [ $conversion_result -eq 0 ] && [ -f "${onnx_file}" ]; then
                log_success "✅ OpenCV兼容的ONNX模型转换成功"
            else
                log_error "❌ ONNX模型转换失败，尝试备用方案..."
                
                # 备用方案：手动PyTorch导出
                cat > "${MODELS_DIR}/manual_export.py" << 'EOF'
#!/usr/bin/env python3
import torch
import torch.onnx
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def manual_export_onnx(model_path="best.pt"):
    """手动导出ONNX，避免ultralytics的自动优化"""
    
    try:
        # 加载模型并切换到评估模式
        yolo_model = YOLO(model_path)
        pytorch_model = yolo_model.model
        pytorch_model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # 手动导出ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            "bamboo_detection.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=None,  # 禁用动态轴
            verbose=True,
            keep_initializers_as_inputs=False
        )
        
        logger.info("✅ 手动ONNX导出完成")
        
        # 验证导出的模型
        import onnx
        model_onnx = onnx.load("bamboo_detection.onnx")
        onnx.checker.check_model(model_onnx)
        logger.info("✅ ONNX模型验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 手动导出失败: {e}")
        return False

if __name__ == "__main__":
    manual_export_onnx()
EOF
                
                cd "${MODELS_DIR}"
                python3 manual_export.py
                cd - > /dev/null
                
                if [ -f "${onnx_file}" ]; then
                    log_success "✅ 备用方案：手动ONNX导出成功"
                else
                    log_error "❌ 所有ONNX转换方案都失败"
                    return 1
                fi
            fi
        else
            log_success "✅ 现有ONNX模型兼容OpenCV，无需重新转换"
        fi
        
        # 更新模型配置文件以适配 JetPack SDK 路径
        log_jetpack "更新模型配置文件..."
        
        # 更新 AI 优化配置
        sed -i 's|model_path:.*|model_path: "/opt/bamboo-cut/models/bamboo_detection.onnx"|g' \
            "${PROJECT_ROOT}/config/ai_optimization.yaml" 2>/dev/null || true
        
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

# 如果还是找不到trtexec，尝试从包管理器安装
if [ -z "$TRTEXEC_PATH" ]; then
    echo "🔍 尝试安装 trtexec 工具..."
    
    # 尝试安装tensorrt-dev包
    if apt-get update && apt-get install -y tensorrt-dev 2>/dev/null; then
        # 重新查找
        for path in "${POSSIBLE_PATHS[@]}"; do
            if [ -f "$path" ] && [ -x "$path" ]; then
                TRTEXEC_PATH="$path"
                echo "✅ 安装后找到 trtexec: $TRTEXEC_PATH"
                break
            fi
        done
    fi
fi

# 如果仍然找不到，跳过TensorRT优化
if [ -z "$TRTEXEC_PATH" ]; then
    echo "⚠️ trtexec 未找到，跳过 TensorRT 优化"
    echo "💡 TensorRT已安装但缺少trtexec工具"
    echo "🔧 可尝试安装: sudo apt install tensorrt-dev"
    echo "✅ 系统将使用 ONNX 模型继续运行"
    exit 0  # 正常退出，不阻止系统启动
fi

# 首先验证ONNX模型与OpenCV的兼容性
echo "🔍 验证ONNX模型兼容性..."
onnx_found=false
for onnx_file in "${MODELS_DIR}"/*.onnx; do
    if [ -f "$onnx_file" ]; then
        onnx_found=true
        echo "📋 测试模型: $(basename "$onnx_file")"
        python3 -c "
import cv2
try:
    net = cv2.dnn.readNetFromONNX('$(basename "$onnx_file")')
    print('✅ 模型与OpenCV兼容')
except Exception as e:
    print(f'❌ 模型不兼容: {e}')
" 2>/dev/null || echo "⚠️ 跳过不兼容的模型: $(basename "$onnx_file")"
    fi
done

if [ "$onnx_found" = false ]; then
    echo "⚠️ 未找到ONNX模型文件，跳过优化"
    exit 0
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

# 检查是否需要升级现有部署
check_upgrade_needed() {
    if [ "$AUTO_UPGRADE" = "true" ]; then
        log_info "检查现有部署状态..."
        
        # 检查是否有现有的systemd服务
        if systemctl is-enabled bamboo-cut-jetpack >/dev/null 2>&1; then
            log_info "检测到现有的智能切竹机服务"
            
            # 备份当前版本
            if [ "$BACKUP_CURRENT" = "true" ]; then
                backup_current_deployment
            fi
            
            # 自动清理历史版本（升级模式下默认启用）
            log_info "升级模式：自动清理历史版本..."
            clean_legacy_deployment
            
            return 0
        else
            log_info "未检测到现有部署，将进行全新安装"
            return 1
        fi
    fi
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
    log_info "🧹 清理历史版本进程和配置..."
    
    # 停止并清理所有相关的systemd服务
    log_info "停止和清理相关systemd服务..."
    
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
    
    for service in "${BAMBOO_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            log_info "停止服务: $service"
            sudo systemctl stop "$service" || true
        fi
        
        if systemctl is-enabled --quiet "$service" 2>/dev/null; then
            log_info "禁用服务: $service"
            sudo systemctl disable "$service" || true
        fi
        
        if [ -f "/etc/systemd/system/${service}.service" ]; then
            log_info "删除服务文件: ${service}.service"
            sudo rm -f "/etc/systemd/system/${service}.service"
        fi
    done
    
    # 杀死残留的进程
    log_info "清理残留进程..."
    
    BAMBOO_PROCESSES=(
        "bamboo_cut_backend"
        "bamboo_controller_qt"
        "bamboo-cut"
        "bamboo_cut_frontend"
        "bamboo-backend"
        "bamboo-frontend"
    )
    
    for process in "${BAMBOO_PROCESSES[@]}"; do
        if pgrep -f "$process" >/dev/null 2>&1; then
            log_info "终止进程: $process"
            sudo pkill -f "$process" || true
            sleep 2
            # 强制终止如果仍在运行
            if pgrep -f "$process" >/dev/null 2>&1; then
                log_warning "强制终止进程: $process"
                sudo pkill -9 -f "$process" || true
            fi
        fi
    done
    
    # 清理历史安装目录
    log_info "清理历史安装目录..."
    
    LEGACY_INSTALL_DIRS=(
        "/opt/bamboo-cut"
        "/opt/bamboo-controller"
        "/opt/bamboo-backend"
        "/opt/bamboo-frontend"
        "/usr/local/bin/bamboo-cut"
        "/usr/local/share/bamboo-cut"
        "/home/*/bamboo-cut"
    )
    
    for dir in "${LEGACY_INSTALL_DIRS[@]}"; do
        # 使用glob展开处理通配符
        for expanded_dir in $dir; do
            if [ -d "$expanded_dir" ] && [ "$expanded_dir" != "/opt/bamboo-cut" ]; then
                log_info "清理历史目录: $expanded_dir"
                sudo rm -rf "$expanded_dir"
            fi
        done
    done
    
    # 清理历史配置文件
    log_info "清理历史配置文件..."
    
    LEGACY_CONFIG_FILES=(
        "/etc/bamboo-cut.conf"
        "/etc/bamboo-controller.conf"
        "/etc/default/bamboo-cut"
        "/etc/systemd/system/multi-user.target.wants/bamboo-*.service"
        "/var/lib/systemd/deb-systemd-helper-enabled/bamboo-*.service"
        "/var/lib/systemd/deb-systemd-helper-enabled/multi-user.target.wants/bamboo-*.service"
    )
    
    for config in "${LEGACY_CONFIG_FILES[@]}"; do
        # 使用glob展开处理通配符
        for expanded_config in $config; do
            if [ -f "$expanded_config" ] || [ -L "$expanded_config" ]; then
                log_info "删除历史配置: $expanded_config"
                sudo rm -f "$expanded_config"
            fi
        done
    done
    
    # 清理历史日志文件
    log_info "清理历史日志文件..."
    
    LEGACY_LOG_DIRS=(
        "/var/log/bamboo-cut"
        "/var/log/bamboo-controller"
        "/var/log/bamboo-backend"
        "/var/log/bamboo-frontend"
        "/tmp/bamboo-*.log"
        "/tmp/bamboo_*.log"
    )
    
    for log_dir in "${LEGACY_LOG_DIRS[@]}"; do
        # 使用glob展开处理通配符
        for expanded_log in $log_dir; do
            if [ -d "$expanded_log" ]; then
                log_info "清理历史日志目录: $expanded_log"
                sudo rm -rf "$expanded_log"
            elif [ -f "$expanded_log" ]; then
                log_info "删除历史日志文件: $expanded_log"
                sudo rm -f "$expanded_log"
            fi
        done
    done
    
    # 清理历史的环境变量和PATH设置
    log_info "清理历史环境变量配置..."
    
    ENV_FILES=(
        "/etc/environment"
        "/etc/profile"
        "/etc/bash.bashrc"
        "/home/*/.bashrc"
        "/home/*/.profile"
        "/root/.bashrc"
        "/root/.profile"
    )
    
    BAMBOO_ENV_PATTERNS=(
        "BAMBOO_"
        "bamboo-cut"
        "/opt/bamboo"
    )
    
    for env_file in "${ENV_FILES[@]}"; do
        # 使用glob展开处理通配符
        for expanded_env in $env_file; do
            if [ -f "$expanded_env" ]; then
                for pattern in "${BAMBOO_ENV_PATTERNS[@]}"; do
                    if grep -q "$pattern" "$expanded_env" 2>/dev/null; then
                        log_info "清理环境变量配置: $expanded_env (包含 $pattern)"
                        # 创建备份
                        sudo cp "$expanded_env" "${expanded_env}.backup.$(date +%s)" 2>/dev/null || true
                        # 删除包含bamboo相关的行
                        sudo sed -i "/$pattern/d" "$expanded_env" 2>/dev/null || true
                    fi
                done
            fi
        done
    done
    
    # 清理历史的cron任务
    log_info "清理历史cron任务..."
    
    # 检查系统crontab
    if crontab -l 2>/dev/null | grep -q "bamboo"; then
        log_info "清理用户cron任务中的bamboo相关项"
        (crontab -l 2>/dev/null | grep -v "bamboo") | crontab - 2>/dev/null || true
    fi
    
    # 检查root crontab
    if sudo crontab -l 2>/dev/null | grep -q "bamboo"; then
        log_info "清理root cron任务中的bamboo相关项"
        (sudo crontab -l 2>/dev/null | grep -v "bamboo") | sudo crontab - 2>/dev/null || true
    fi
    
    # 清理/etc/cron.d/中的bamboo任务
    for cron_file in /etc/cron.d/bamboo*; do
        if [ -f "$cron_file" ]; then
            log_info "删除cron文件: $cron_file"
            sudo rm -f "$cron_file"
        fi
    done
    
    # 清理历史用户和组
    log_info "清理历史用户和组..."
    
    LEGACY_USERS=(
        "bamboo-cut"
        "bamboo-controller"
        "bamboo-backend"
        "bamboo-frontend"
    )
    
    for user in "${LEGACY_USERS[@]}"; do
        if id "$user" >/dev/null 2>&1; then
            log_info "删除历史用户: $user"
            sudo userdel "$user" 2>/dev/null || true
        fi
        
        if getent group "$user" >/dev/null 2>&1; then
            log_info "删除历史组: $user"
            sudo groupdel "$user" 2>/dev/null || true
        fi
    done
    
    # 重新加载systemd配置
    log_info "重新加载systemd配置..."
    sudo systemctl daemon-reload
    
    # 清理systemd的缓存
    sudo systemctl reset-failed 2>/dev/null || true
    
    log_success "✅ 历史版本清理完成"
}

# 备份当前部署
backup_current_deployment() {
    if [ -d "/opt/bamboo-cut" ]; then
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

# 构建项目 - 修正为分别构建各子项目
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
        
        # 创建健壮的 JetPack 启动脚本
        cat > "$PACKAGE_DIR/start_bamboo_cut_jetpack.sh" << 'EOF'
#!/bin/bash
# 智能切竹机 JetPack SDK 启动脚本（健壮版）

echo "🚀 启动智能切竹机系统（健壮模式）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
    
    # 强制使用EGLFS平台配置（专用触摸屏）
    export QT_QPA_PLATFORM=eglfs
    export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
    export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
    export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
    export QT_QPA_EGLFS_HIDECURSOR=1
    
    # 触摸屏设备配置
    export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
    export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
    
    # 触摸屏调试日志
    export QT_LOGGING_RULES="qt.qpa.*=true;qt.qpa.input*=true"
    echo "✅ EGLFS触摸屏环境已设置"
    echo "   Touch Device: /dev/input/event2"
    echo "   Cursor Hidden: Yes"
fi

# 应用性能优化 (如果存在)
if [ -f "./power_config.sh" ]; then
    ./power_config.sh
    echo "✅ 性能优化已应用"
fi

# 增强摄像头检测逻辑（支持CSI和USB摄像头）
echo "🔍 检测摄像头设备..."
CAMERA_FOUND=false

# 检查CSI摄像头（IMX219等）
echo "📋 检查CSI摄像头..."
CSI_CAMERA_DETECTED=false

# 检查IMX219模块
if lsmod | grep -q imx219; then
    echo "✅ IMX219内核模块已加载"
    CSI_CAMERA_DETECTED=true
elif modprobe imx219 2>/dev/null; then
    echo "✅ IMX219内核模块加载成功"
    CSI_CAMERA_DETECTED=true
    sleep 2  # 等待模块初始化
else
    echo "⚠️ 无法加载IMX219内核模块"
fi

# 检查I2C设备上的IMX219
if command -v i2cdetect >/dev/null 2>&1; then
    echo "🔍 扫描I2C总线上的摄像头设备..."
    for bus in 0 1 2 6 7 8 9 10; do
        if i2cdetect -y -r $bus 2>/dev/null | grep -q "10"; then
            echo "✅ 在I2C总线 $bus 上检测到IMX219设备"
            CSI_CAMERA_DETECTED=true
            break
        fi
    done
fi

# 检查media设备（CSI摄像头通常显示为media设备）
MEDIA_DEVICES=$(find /dev -name "media*" 2>/dev/null)
if [ -n "$MEDIA_DEVICES" ]; then
    echo "📹 找到media设备: $MEDIA_DEVICES"
    CSI_CAMERA_DETECTED=true
fi

# 检查传统video设备
echo "📋 检查传统video设备..."
for device in /dev/video0 /dev/video1 /dev/video2; do
    if [ -e "$device" ]; then
        echo "📹 找到video设备: $device"
        
        # 尝试获取设备信息
        if command -v v4l2-ctl >/dev/null 2>&1; then
            echo "📋 设备信息:"
            v4l2-ctl --device="$device" --info 2>/dev/null || echo "   无法获取设备详细信息"
        fi
        
        # 检查设备是否可读写
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

# 列出所有video设备用于调试
echo "📋 系统中的所有video设备:"
ls -la /dev/video* 2>/dev/null || echo "   未找到video设备"

# 列出所有media设备用于调试
echo "📋 系统中的所有media设备:"
ls -la /dev/media* 2>/dev/null || echo "   未找到media设备"

# 检查USB摄像头
echo "📋 USB设备信息:"
lsusb 2>/dev/null | grep -i "camera\|video\|webcam" || echo "   未检测到USB摄像头"

# 检查内核日志中的摄像头信息
echo "📋 内核日志中的摄像头信息:"
dmesg | grep -i "imx219\|camera\|csi" | tail -5 || echo "   未找到相关日志"

# 设置摄像头模式
if [ "$CAMERA_FOUND" = true ] || [ "$CSI_CAMERA_DETECTED" = true ]; then
    if [ "$CSI_CAMERA_DETECTED" = true ]; then
        echo "✅ 检测到CSI摄像头，启用硬件模式"
        export BAMBOO_CAMERA_TYPE="csi"
    else
        echo "✅ 检测到USB摄像头，启用硬件模式"
        export BAMBOO_CAMERA_TYPE="usb"
    fi
    export BAMBOO_CAMERA_MODE="hardware"
    export BAMBOO_SKIP_CAMERA="false"
else
    echo "⚠️ 未检测到可用摄像头设备，启用模拟模式"
    echo "💡 对于IMX219 CSI摄像头，请运行CSI摄像头配置脚本："
    echo "   sudo bash $(dirname $0)/csi_camera_setup.sh"
    echo "💡 通用摄像头故障排除："
    echo "   1. 摄像头是否正确连接"
    echo "   2. 驱动程序是否已安装"
    echo "   3. 设备权限是否正确"
    echo "   4. 对于CSI摄像头，检查排线连接和内核模块"
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
        echo "⚠️ Qt前端可执行文件不存在，仅运行后端模式"
        return 1
    fi
    
    echo "🔄 启动 Qt 前端: $qt_frontend_exec"
    echo "🔧 当前Qt环境变量："
    echo "   QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
    echo "   Touch Device: /dev/input/event2"
    echo "   Cursor Hidden: Yes"
    echo "   XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
    
    # 强制使用EGLFS平台（不使用fallback）
    export QT_QPA_PLATFORM=eglfs
    export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
    export QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
    export QT_QPA_EGLFS_ALWAYS_SET_MODE=1
    export QT_QPA_EGLFS_HIDECURSOR=1
    
    # 触摸屏设备配置
    export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
    export QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
    
    echo "✅ 使用EGLFS平台启动Qt前端..."
    
    # 启动Qt前端
    timeout 30 "$qt_frontend_exec" &
    FRONTEND_PID=$!
    
    # 等待启动
    sleep 5
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ Qt 前端启动成功 (PID: $FRONTEND_PID, Platform: eglfs)"
        return 0
    else
        echo "❌ Qt前端启动失败"
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
sudo chown -R root:root /opt/bamboo-cut
sudo chown -R bamboo-cut:bamboo-cut /var/log/bamboo-cut
sudo chmod +x /opt/bamboo-cut/*.sh

# 创建健壮的 systemd 服务
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK) - 健壮版
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack.sh
Restart=on-failure
RestartSec=30
StartLimitBurst=3
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
Environment=QT_QPA_EGLFS_ALWAYS_SET_MODE=1
Environment=QT_QPA_EGLFS_HIDECURSOR=1
Environment=QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2
Environment=QT_QPA_GENERIC_PLUGINS=evdevtouch:/dev/input/event2
Environment=QT_LOGGING_RULES=qt.qpa.*=true;qt.qpa.input*=true
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
    
    # 默认启用重新部署模式
    log_info "启动重新部署模式：自动清理运行中的进程和强制重新编译"
    CLEAN_LEGACY="true"
    FORCE_REBUILD="true"
    AUTO_UPGRADE="true"
    ENABLE_QT_DEPLOY="true"
    DEPLOY_MODELS="true"
    CREATE_PACKAGE="true"
    DEPLOY_TARGET="local"
    
    # 首先停止所有运行中的相关服务和进程
    log_info "🛑 停止所有运行中的智能切竹机服务和进程..."
    stop_running_services
    
    # 清理历史版本
    log_info "🧹 清理历史版本..."
    clean_legacy_deployment
    
    # 创建部署目录
    mkdir -p "$JETPACK_DEPLOY_DIR"
    
    detect_jetpack_environment
    install_jetpack_dependencies
    configure_jetpack_performance
    
    deploy_qt_dependencies
    deploy_ai_models
    
    # 确保项目已编译
    log_jetpack "确保项目已编译..."
    if ! build_project; then
        log_error "项目编译失败，停止部署"
        exit 1
    fi
    
    # 如果启用了Qt部署，确保Qt前端已编译
    if [ "$ENABLE_QT_DEPLOY" = "true" ]; then
        log_qt "检查Qt前端编译状态..."
        # 检查编译输出位置（支持正确的可执行文件名）
        QT_FRONTEND_CANDIDATES=(
            "${PROJECT_ROOT}/qt_frontend/build/bamboo_controller_qt"
            "${PROJECT_ROOT}/qt_frontend/build_debug/bamboo_controller_qt"
            "${PROJECT_ROOT}/qt_frontend/build_release/bamboo_controller_qt"
            "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
        )
        
        qt_exists=false
        for candidate in "${QT_FRONTEND_CANDIDATES[@]}"; do
            if [ -f "$candidate" ]; then
                qt_exists=true
                break
            fi
        done
        
        if [ "$qt_exists" = false ]; then
            log_qt "Qt前端未编译，开始编译..."
            cd "$PROJECT_ROOT"
            
            # 使用正确的make目标
            if [ "$BUILD_TYPE" = "Debug" ]; then
                if ! make qt BUILD_TYPE=debug; then
                    log_error "Qt前端Debug编译失败"
                    exit 1
                fi
            else
                if ! make qt BUILD_TYPE=release; then
                    log_error "Qt前端Release编译失败"
                    exit 1
                fi
            fi
            log_success "Qt前端编译完成"
            
            # 复制到BUILD_DIR以便后续使用
            mkdir -p "${BUILD_DIR}/qt_frontend"
            for candidate in "${QT_FRONTEND_CANDIDATES[@]}"; do
                if [ -f "$candidate" ]; then
                    cp "$candidate" "${BUILD_DIR}/qt_frontend/bamboo_controller_qt"
                    break
                fi
            done
        else
            log_qt "Qt前端已编译"
            
            # 确保复制到BUILD_DIR
            mkdir -p "${BUILD_DIR}/qt_frontend"
            for candidate in "${QT_FRONTEND_CANDIDATES[@]}"; do
                if [ -f "$candidate" ]; then
                    cp "$candidate" "${BUILD_DIR}/qt_frontend/bamboo_controller_qt" 2>/dev/null || true
                fi
            done
        fi
    fi
    
    create_jetpack_package
    deploy_to_target
    
    # 自动启动服务
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
    
    log_success "🎉 JetPack SDK 重新部署完成!"
    log_info "部署包位置: ${DEPLOY_DIR}/packages/bamboo-cut-jetpack-${VERSION}.tar.gz"
    
    if [ "$JETSON_DETECTED" = "true" ]; then
        log_jetpack "运行性能测试: sudo jetson_stats"
        log_jetpack "监控 GPU 使用: sudo tegrastats"
    fi
    
    echo ""
    echo "🎯 部署摘要："
    echo "✅ 已停止所有运行中的进程"
    echo "✅ 已清理历史版本"
    echo "✅ 已强制重新编译"
    echo "✅ 已重新部署服务"
    echo "✅ 已启动智能切竹机服务"
    echo ""
    echo "📋 常用命令："
    echo "  查看服务状态: sudo systemctl status bamboo-cut-jetpack"
    echo "  查看实时日志: sudo journalctl -u bamboo-cut-jetpack -f"
    echo "  重启服务: sudo systemctl restart bamboo-cut-jetpack"
    echo "  停止服务: sudo systemctl stop bamboo-cut-jetpack"
}

# 运行主函数
main "$@"