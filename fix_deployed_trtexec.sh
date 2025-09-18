#!/bin/bash

# 修复已部署系统中的 TensorRT trtexec 检测问题

echo "🔧 修复已部署系统的 TensorRT 检测问题..."

# 检查是否存在已部署的模型优化脚本
DEPLOYED_SCRIPT="/opt/bamboo-cut/models/optimize_models.sh"

if [ ! -f "$DEPLOYED_SCRIPT" ]; then
    echo "❌ 未找到已部署的模型优化脚本: $DEPLOYED_SCRIPT"
    echo "💡 请确认系统已正确部署"
    exit 1
fi

echo "✅ 找到已部署的脚本: $DEPLOYED_SCRIPT"
echo "🔄 备份并更新脚本..."

# 备份原始脚本
sudo cp "$DEPLOYED_SCRIPT" "${DEPLOYED_SCRIPT}.backup.$(date +%s)"

# 创建修复后的脚本
sudo tee "$DEPLOYED_SCRIPT" > /dev/null << 'EOF'
#!/bin/bash
# TensorRT 模型优化脚本（修复版，增强 trtexec 检测）

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${MODELS_DIR}/onnx"
TRT_DIR="${MODELS_DIR}/tensorrt"

echo "🚀 开始 TensorRT 模型优化..."

# 查找 trtexec 工具 - 增强版检测
TRTEXEC_PATH=""
POSSIBLE_PATHS=(
    "/usr/bin/trtexec"
    "/usr/local/bin/trtexec"
    "/usr/src/tensorrt/bin/trtexec"
    "/usr/src/tensorrt/samples/trtexec"
    "/opt/tensorrt/bin/trtexec"
)

# 首先检查预定义路径
echo "🔍 检查预定义路径..."
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

# 设置执行权限
sudo chmod +x "$DEPLOYED_SCRIPT"

echo "✅ 脚本修复完成"
echo "🔄 现在重启服务以使用修复后的脚本..."

# 重启服务
sudo systemctl restart bamboo-cut-jetpack

echo "🎉 修复完成！"
echo "📋 检查服务状态:"
sudo systemctl status bamboo-cut-jetpack --no-pager -l

echo ""
echo "📋 查看最新日志:"
echo "sudo journalctl -u bamboo-cut-jetpack -f"