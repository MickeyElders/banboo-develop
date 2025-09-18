# JetPack SDK 智能切竹机系统升级指南

## 🔄 升级概述

本升级方案解决了ONNX模型与OpenCV DNN兼容性问题，并集成了自动模型转换和验证功能。

## 📋 升级内容

### 新增功能
- ✅ **OpenCV兼容性检测**：自动检测现有ONNX模型是否兼容OpenCV DNN
- ✅ **智能模型转换**：自动转换不兼容的模型为OpenCV兼容格式
- ✅ **多级备用方案**：提供多种ONNX导出方法确保成功转换
- ✅ **实时兼容性验证**：转换后立即验证模型可用性
- ✅ **增强的TensorRT优化**：支持兼容性验证的TensorRT引擎生成

### 修复问题
- 🔧 **Reshape节点错误**：彻底解决OpenCV DNN中的Reshape节点兼容性问题
- 🔧 **动态尺寸问题**：禁用动态尺寸避免运行时错误
- 🔧 **半精度问题**：禁用可能导致兼容性问题的半精度优化
- 🔧 **模型验证缺失**：增加转换后的模型完整性验证

## 🚀 升级方法

### 方法1：完整重新部署（推荐）

```bash
# 1. 备份现有安装
sudo systemctl stop bamboo-cut-jetpack
sudo cp -r /opt/bamboo-cut /opt/bamboo-cut.backup.$(date +%Y%m%d)

# 2. 使用新的部署脚本重新部署
cd /path/to/bamboo-cut-project
./jetpack_deploy.sh --models --qt-deploy --deploy local

# 3. 验证升级结果
sudo systemctl status bamboo-cut-jetpack
sudo journalctl -u bamboo-cut-jetpack -f
```

### 方法2：就地升级现有系统

```bash
cd /opt/bamboo-cut

# 创建升级脚本
cat > upgrade_onnx_compatibility.sh << 'EOF'
#!/bin/bash
set -e

echo "🔄 开始智能切竹机ONNX兼容性升级..."

# 1. 安装必要的Python包
python3 -m pip install ultralytics onnx onnxsim torch

# 2. 备份现有模型
if [ -f "models/bamboo_detection.onnx" ]; then
    mv models/bamboo_detection.onnx models/bamboo_detection.onnx.backup.$(date +%Y%m%d%H%M%S)
    echo "✅ 已备份现有ONNX模型"
fi

# 3. 检查PyTorch模型是否存在
if [ ! -f "models/best.pt" ]; then
    echo "❌ 未找到PyTorch模型文件 models/best.pt"
    exit 1
fi

# 4. 创建OpenCV兼容的转换脚本
cat > models/convert_opencv_compatible.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import torch
import onnx
from ultralytics import YOLO
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pytorch_to_opencv_onnx(model_path="best.pt"):
    """转换PyTorch模型为OpenCV DNN兼容的ONNX格式"""
    
    try:
        # 加载YOLO模型
        model = YOLO(model_path)
        logger.info(f"已加载模型: {model_path}")
        
        # 导出为ONNX，使用OpenCV兼容参数
        success = model.export(
            format="onnx",
            imgsz=640,           # 固定输入尺寸
            dynamic=False,       # 禁用动态尺寸，避免Reshape问题
            simplify=True,       # 简化模型
            opset=11,           # 使用OpenCV支持良好的opset版本
            half=False,         # 禁用半精度，避免精度问题
            int8=False,         # 暂时禁用int8
            optimize=False,     # 禁用额外优化，避免引入不兼容节点
            verbose=True
        )
        
        if success:
            logger.info("✅ ONNX模型导出成功")
            
            # 验证模型
            onnx_path = model_path.replace('.pt', '.onnx')
            if os.path.exists(onnx_path):
                model_onnx = onnx.load(onnx_path)
                onnx.checker.check_model(model_onnx)
                logger.info("✅ ONNX模型验证通过")
                
                # 重命名为标准名称
                import shutil
                shutil.move(onnx_path, "bamboo_detection.onnx")
                logger.info("✅ 模型已保存为 bamboo_detection.onnx")
            
            return True
        else:
            logger.error("❌ ONNX模型导出失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 转换过程出错: {e}")
        return False

def test_opencv_compatibility():
    """测试模型与OpenCV DNN的兼容性"""
    try:
        import cv2
        
        # 尝试加载模型
        net = cv2.dnn.readNetFromONNX("bamboo_detection.onnx")
        logger.info("✅ OpenCV DNN成功加载模型")
        
        # 创建测试输入
        import numpy as np
        test_input = np.random.rand(640, 640, 3).astype('uint8')
        blob = cv2.dnn.blobFromImage(test_input, 1.0/255.0, (640, 640), (0,0,0), True, False)
        
        # 设置输入并执行前向传播
        net.setInput(blob)
        output = net.forward()
        logger.info(f"✅ 模型推理成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    # 执行转换
    if convert_pytorch_to_opencv_onnx():
        # 测试兼容性
        if test_opencv_compatibility():
            logger.info("🎉 模型转换和兼容性验证完成")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(1)
PYTHON_EOF

# 5. 执行模型转换
cd models
python3 convert_opencv_compatible.py
conversion_result=$?

if [ $conversion_result -eq 0 ]; then
    echo "✅ OpenCV兼容的ONNX模型转换成功"
else
    echo "❌ 转换失败，尝试备用方案..."
    
    # 备用方案：手动PyTorch导出
    cat > manual_export.py << 'PYTHON_EOF2'
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
PYTHON_EOF2
    
    python3 manual_export.py
    
    if [ -f "bamboo_detection.onnx" ]; then
        echo "✅ 备用方案：手动ONNX导出成功"
    else
        echo "❌ 所有转换方案都失败"
        exit 1
    fi
fi

cd ..

# 6. 验证新模型
echo "🔍 验证新模型兼容性..."
python3 -c "
import cv2
try:
    net = cv2.dnn.readNetFromONNX('models/bamboo_detection.onnx')
    import numpy as np
    blob = cv2.dnn.blobFromImage(np.random.rand(640,640,3).astype('uint8'), 1.0/255.0, (640, 640), (0,0,0), True, False)
    net.setInput(blob)
    output = net.forward()
    print('✅ 新模型与OpenCV完全兼容')
except Exception as e:
    print(f'❌ 新模型仍有问题: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 升级成功！模型现在与OpenCV DNN完全兼容"
else
    echo "❌ 升级失败，请检查日志"
    exit 1
fi

echo "✅ ONNX兼容性升级完成"
EOF

chmod +x upgrade_onnx_compatibility.sh

# 执行升级
sudo -u bamboo-cut ./upgrade_onnx_compatibility.sh

# 重启服务验证
sudo systemctl restart bamboo-cut-jetpack
sleep 5
sudo systemctl status bamboo-cut-jetpack
```

### 方法3：手动升级指定文件

```bash
# 1. 仅更新模型转换功能
cd /opt/bamboo-cut/models
sudo -u bamboo-cut wget https://raw.githubusercontent.com/.../convert_opencv_compatible.py
sudo -u bamboo-cut python3 convert_opencv_compatible.py

# 2. 重启服务
sudo systemctl restart bamboo-cut-jetpack
```

## 📊 升级验证

升级完成后，使用以下命令验证：

```bash
# 1. 检查服务状态
sudo systemctl status bamboo-cut-jetpack

# 2. 查看启动日志
sudo journalctl -u bamboo-cut-jetpack -f

# 3. 验证模型兼容性
cd /opt/bamboo-cut/models
python3 -c "
import cv2
net = cv2.dnn.readNetFromONNX('bamboo_detection.onnx')
print('✅ 模型加载成功')
import numpy as np
blob = cv2.dnn.blobFromImage(np.random.rand(640,640,3).astype('uint8'), 1.0/255.0, (640, 640), (0,0,0), True, False)
net.setInput(blob)
output = net.forward()
print(f'✅ 推理成功，输出形状: {output.shape}')
"

# 4. 测试系统启动
sudo systemctl restart bamboo-cut-jetpack
sleep 10
sudo systemctl is-active bamboo-cut-jetpack
```

## 🔧 故障排除

### 如果升级失败

```bash
# 恢复备份
sudo systemctl stop bamboo-cut-jetpack
sudo rm -rf /opt/bamboo-cut
sudo mv /opt/bamboo-cut.backup.* /opt/bamboo-cut
sudo systemctl start bamboo-cut-jetpack
```

### 如果模型转换失败

```bash
# 检查PyTorch模型文件
ls -la /opt/bamboo-cut/models/best.pt

# 检查Python环境
python3 -c "import ultralytics, onnx, torch; print('✅ 依赖包正常')"

# 手动执行转换
cd /opt/bamboo-cut/models
sudo -u bamboo-cut python3 convert_opencv_compatible.py
```

### 如果服务启动失败

```bash
# 查看详细日志
sudo journalctl -u bamboo-cut-jetpack --no-pager -l

# 检查权限
sudo chown -R bamboo-cut:bamboo-cut /opt/bamboo-cut

# 手动测试启动
cd /opt/bamboo-cut
sudo -u bamboo-cut ./start_bamboo_cut_jetpack.sh
```

## 📝 升级后配置

升级完成后，系统将自动：
- ✅ 检测现有ONNX模型兼容性
- ✅ 自动转换不兼容的模型
- ✅ 验证转换后的模型可用性
- ✅ 生成兼容的TensorRT引擎（如果支持）

## 🎯 预期效果

升级后应该看到：
- 🚫 不再出现"ERROR during processing node [Reshape]"错误
- ✅ C++后端成功初始化BambooDetector
- ✅ 模型加载过程顺利完成
- ✅ 系统日志显示"视觉系统初始化成功"

## 📞 技术支持

如果在升级过程中遇到问题，请提供：
1. 升级方法（完整重新部署/就地升级/手动升级）
2. 错误日志（`journalctl -u bamboo-cut-jetpack`）
3. 模型文件信息（`ls -la /opt/bamboo-cut/models/`）
4. 系统信息（JetPack版本、CUDA版本、OpenCV版本）