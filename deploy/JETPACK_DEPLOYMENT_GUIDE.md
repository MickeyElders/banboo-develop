# 智能切竹机 JetPack SDK 部署指南

## 概述

本指南描述如何使用专用的 JetPack SDK 部署脚本来部署智能切竹机系统到 Jetson Nano Super 硬件平台。该部署脚本集成了类似 windeployqt 的功能，提供自动依赖管理、性能调优和完整的 JetPack SDK 环境配置。

## 功能特性

### 🚀 核心功能
- **自动 JetPack SDK 环境检测**：自动识别 Jetson 设备型号、JetPack 版本、CUDA 和 TensorRT 版本
- **Qt 依赖自动收集**：类似 windeployqt 功能，自动收集和部署 Qt6 库、插件和 QML 模块
- **AI 模型配置管理**：自动配置 ONNX 和 TensorRT 模型文件路径，支持模型优化
- **性能调优集成**：GPU 内存优化、功耗管理、CUDA 加速配置
- **完整部署包创建**：生成包含所有依赖的独立部署包

### 🔧 技术栈集成
- **JetPack SDK 5.1.1**：完整的 Jetson 开发环境
- **CUDA 11.4**：GPU 计算加速
- **TensorRT 8.5.2**：AI 推理优化
- **OpenCV 4.8.0**：计算机视觉处理
- **Qt6**：现代 UI 框架

## 使用方法

### 前置要求
```bash
# 确保脚本具有执行权限
chmod +x deploy/scripts/jetpack_deploy.sh
```

### 基本部署命令

1. **完整安装和部署**（推荐）
```bash
./deploy/scripts/jetpack_deploy.sh \
    --install-deps \
    --gpu-opt \
    --power-opt \
    --models \
    --qt-deploy \
    --create-package
```

2. **仅构建和创建部署包**
```bash
./deploy/scripts/jetpack_deploy.sh \
    --qt-deploy \
    --models \
    --create-package
```

3. **本地安装**
```bash
./deploy/scripts/jetpack_deploy.sh \
    --deploy local \
    --qt-deploy \
    --models
```

4. **远程部署到 Jetson 设备**
```bash
./deploy/scripts/jetpack_deploy.sh \
    --deploy remote:192.168.1.100 \
    --create-package
```

### 命令行参数详解

| 参数 | 描述 | 示例 |
|------|------|------|
| `-t, --type TYPE` | 构建类型 (Debug/Release) | `--type Release` |
| `-d, --deploy TARGET` | 部署目标 | `--deploy jetson` |
| `-i, --install-deps` | 安装 JetPack SDK 依赖包 | `--install-deps` |
| `-g, --gpu-opt` | 启用 GPU 内存和计算优化 | `--gpu-opt` |
| `-p, --power-opt` | 启用功耗管理和性能调优 | `--power-opt` |
| `-m, --models` | 自动配置和部署 AI 模型文件 | `--models` |
| `-q, --qt-deploy` | 启用 Qt 依赖自动收集和部署 | `--qt-deploy` |
| `-c, --create-package` | 创建完整部署包 | `--create-package` |

## 部署流程详解

### 1. 环境检测阶段
- 自动识别 Jetson 设备型号
- 检测 JetPack SDK 版本信息
- 验证 CUDA 和 TensorRT 安装
- 确定系统架构和构建配置

### 2. 依赖安装阶段（--install-deps）
```bash
# JetPack SDK 核心组件
nvidia-jetpack
cuda-toolkit-11-4
tensorrt
libnvinfer-dev

# Qt6 完整环境
qt6-base-dev
qt6-declarative-dev
qt6-multimedia-dev
qt6-serialport-dev

# GStreamer 硬件加速
gstreamer1.0-plugins-*
libgstreamer1.0-dev
```

### 3. 性能优化配置
#### GPU 优化（--gpu-opt）
- CUDA 环境变量配置
- GPU 缓存优化
- 最大性能模式设置
- jetson_clocks 配置

#### 功耗管理（--power-opt）
- CPU 调度器性能模式
- 内存管理优化
- GPU 功耗控制
- 网络性能优化

### 4. Qt 依赖部署（--qt-deploy）
自动收集和部署：
- **核心库**：Qt6Core, Qt6Gui, Qt6Widgets, Qt6Quick, Qt6Qml
- **多媒体库**：Qt6Multimedia, Qt6SerialPort, Qt6Network
- **平台插件**：EGLFS, Wayland, LinuxFB 支持
- **QML 模块**：QtQuick, QtMultimedia 模块
- **环境配置**：自动生成 Qt 环境设置脚本

### 5. AI 模型配置（--models）
- 创建标准模型目录结构
- 配置 ONNX 和 TensorRT 模型路径
- 生成 TensorRT 优化脚本
- 更新系统配置文件

### 6. 部署包创建（--create-package）
生成完整的独立部署包：
```
bamboo-cut-jetpack-1.0.0/
├── bamboo_cut_backend           # C++ 后端可执行文件
├── bamboo_cut_frontend          # Qt 前端可执行文件
├── config/                      # 配置文件目录
├── qt_libs/                     # Qt 依赖库
├── models/                      # AI 模型文件
├── start_bamboo_cut_jetpack.sh  # JetPack 启动脚本
├── install_jetpack.sh           # 安装脚本
└── power_config.sh              # 性能优化脚本
```

## 配置文件适配

### 系统配置更新
JetPack SDK 部署会自动更新以下配置：

1. **system_config.yaml**
```yaml
ai:
  model_path: "/opt/bamboo-cut/models/bamboo_detector.onnx"
  tensorrt_engine_path: "/opt/bamboo-cut/models/tensorrt/bamboo_detector.trt"
  enable_tensorrt: true
  enable_jetpack_optimization: true
```

2. **kms.conf**（EGL/KMS 显示配置）
```json
{
  "device": "/dev/dri/card0",
  "hwcursor": false,
  "pbuffers": true,
  "outputs": [
    {
      "name": "HDMI1",
      "mode": "1920x1080"
    }
  ]
}
```

### AI 优化配置
自动启用 JetPack SDK 特定优化：
- TensorRT FP16 推理加速
- NAM 注意力机制
- GhostConv 卷积优化
- SAHI 切片推理
- 硬件加速支持

## 部署后操作

### 1. 系统服务安装
```bash
# 安装 systemd 服务
sudo systemctl enable bamboo-cut-jetpack
sudo systemctl start bamboo-cut-jetpack

# 查看服务状态
sudo systemctl status bamboo-cut-jetpack
```

### 2. 性能监控
```bash
# JetPack SDK 性能统计
sudo jetson_stats

# GPU 使用监控
sudo tegrastats

# 系统资源监控
htop
```

### 3. 日志查看
```bash
# 系统日志
sudo journalctl -u bamboo-cut-jetpack -f

# 应用日志
tail -f /opt/bamboo-cut/logs/system.log
```

## 故障排除

### 常见问题

1. **Qt 平台插件错误**
```bash
# 检查 Qt 环境变量
source /opt/bamboo-cut/qt_libs/setup_qt_env.sh
echo $QT_QPA_PLATFORM
```

2. **CUDA/TensorRT 问题**
```bash
# 验证 CUDA 安装
nvcc --version
nvidia-smi

# 检查 TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

3. **模型文件缺失**
```bash
# 检查模型文件
ls -la /opt/bamboo-cut/models/
/opt/bamboo-cut/models/optimize_models.sh
```

### 性能优化建议

1. **首次运行优化**
   - 自动执行 TensorRT 模型优化
   - 应用最佳性能配置
   - 预热 GPU 和 CUDA 上下文

2. **监控要点**
   - GPU 利用率应保持在 70-90%
   - 内存使用率不超过 80%
   - CPU 温度控制在 70°C 以下

3. **调优参数**
   - 根据具体硬件调整批处理大小
   - 优化推理线程数量
   - 配置合适的功耗模式

## 技术支持

### 环境信息收集
```bash
# 收集系统信息
./deploy/scripts/jetpack_deploy.sh --version
cat /etc/nv_tegra_release
uname -a
```

### 开发和调试
- 使用 `--type Debug` 构建调试版本
- 启用详细日志记录
- 使用 gdb 进行程序调试

---

**注意**：此部署脚本专门针对 Jetson Nano Super 和 JetPack SDK 5.1.1 环境优化。在其他平台上使用可能需要调整配置参数。