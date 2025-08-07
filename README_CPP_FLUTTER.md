# 智能切竹机系统 v2.0 - C++/Flutter重构版

## 📋 项目概述

智能切竹机系统v2.0是基于C++后端和Flutter前端的全新架构重构版本，专为工业环境设计，支持嵌入式Linux系统和跨平台部署。

### 🎯 核心特性

- **🚀 高性能C++后端**: 基于TensorRT的GPU加速AI推理
- **📱 Flutter嵌入式前端**: 工业级触屏界面，适配LXDE环境
- **🔧 硬件加速**: GStreamer + DeepStream 摄像头处理
- **⚡ 模型优化**: 集成GhostConv、GSConv、VoV-GSCSP、NAM技术
- **🌐 跨平台支持**: x86_64 和 ARM64 (Jetson) 双架构
- **📡 Modbus TCP通信**: 与PLC的实时数据交换

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter 前端界面                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   实时视频显示   │ │   AI检测结果     │ │   系统控制面板   │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │ FFI/Socket
┌─────────────────────────────────────────────────────────────┐
│                      C++ 后端引擎                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │  TensorRT引擎   │ │  GStreamer管道   │ │  Modbus服务器   │  │
│  │  AI推理加速     │ │  摄像头硬件加速   │ │  PLC通信协议    │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      硬件层                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │     摄像头       │ │      PLC       │ │   切割设备       │  │
│  │   V4L2/CSI      │ │  Modbus TCP    │ │   电机控制       │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 项目结构

```
bamboo-cut-develop/
├── cpp_backend/                    # C++后端源码
│   ├── src/                       # 源文件
│   │   ├── core/                  # 核心组件
│   │   ├── vision/                # 视觉处理
│   │   ├── communication/         # 通信模块
│   │   └── main.cpp              # 主程序入口
│   ├── include/                   # 头文件
│   │   └── bamboo_cut/           # 项目头文件
│   ├── tests/                     # 单元测试
│   └── CMakeLists.txt            # CMake构建配置
├── flutter_frontend/              # Flutter前端
│   ├── lib/                      # Dart源码
│   │   ├── screens/              # 界面页面
│   │   ├── widgets/              # UI组件
│   │   ├── services/             # 服务层
│   │   ├── providers/            # 状态管理
│   │   └── theme/                # 主题配置
│   ├── assets/                   # 资源文件
│   └── pubspec.yaml             # Flutter配置
├── deploy/                       # 部署相关
│   ├── scripts/                  # 部署脚本
│   ├── config/                   # 配置文件
│   └── packages/                 # 构建产物
├── docs/                         # 文档
└── config/                       # 系统配置
```

## 🛠️ 技术栈

### C++ 后端
- **推理引擎**: TensorRT 8.x, OpenCV DNN (fallback)
- **摄像头处理**: GStreamer 1.0, DeepStream SDK (Jetson)
- **通信协议**: libmodbus, TCP/IP sockets
- **构建系统**: CMake 3.16+, GCC 9+
- **模型优化**: ONNX, TensorRT Engine, FP16/INT8 量化

### Flutter 前端
- **UI框架**: Flutter 3.16+ (Embedded Linux)
- **状态管理**: Provider + BLoC
- **通信**: FFI (C++ 绑定), WebSocket
- **图表**: fl_chart, charts_flutter
- **平台**: Linux Desktop/Embedded

### 部署平台
- **x86_64**: Ubuntu 20.04+, 通用Linux发行版
- **ARM64**: NVIDIA Jetson (Nano, Xavier, Orin), 树莓派4+
- **显示**: LXDE, X11, Wayland支持

## 🚀 快速开始

### 1. 环境准备

#### Ubuntu/Debian系统
```bash
# 安装基础依赖
sudo apt update
sudo apt install -y build-essential cmake git pkg-config

# 安装OpenCV
sudo apt install -y libopencv-dev libopencv-contrib-dev

# 安装GStreamer
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# 安装其他依赖
sudo apt install -y libmodbus-dev nlohmann-json3-dev
```

#### Jetson设备额外配置
```bash
# 安装JetPack SDK (包含TensorRT, DeepStream)
sudo apt install nvidia-jetpack

# 验证CUDA和TensorRT
nvcc --version
dpkg -l | grep TensorRT
```

### 2. 克隆和构建

```bash
# 克隆项目
git clone <repository-url> bamboo-cut-develop
cd bamboo-cut-develop

# 使用自动化脚本构建
chmod +x deploy/scripts/build_and_deploy.sh

# 选项1: 本地x86_64构建
./deploy/scripts/build_and_deploy.sh --install-deps --arch x86_64

# 选项2: Jetson ARM64构建 (在Jetson设备上)
./deploy/scripts/build_and_deploy.sh --install-deps --tensorrt --arch aarch64

# 选项3: 交叉编译ARM64 (在x86主机上)
./deploy/scripts/build_and_deploy.sh --cross-compile --arch aarch64
```

### 3. 部署安装

```bash
# 本地安装
./deploy/scripts/build_and_deploy.sh --deploy local

# 部署到Jetson设备
./deploy/scripts/build_and_deploy.sh --deploy jetson

# 部署到远程设备
./deploy/scripts/build_and_deploy.sh --deploy remote:192.168.1.100
```

## ⚙️ 配置说明

### 系统配置 (`config/system_config.yaml`)

```yaml
# 硬件配置
hardware:
  camera:
    device_id: "/dev/video0"
    width: 1920
    height: 1080
    framerate: 30

# 通信配置
communication:
  server_ip: "192.168.1.10"
  server_port: 502
  timeout: 5000

# AI配置
ai:
  model_path: "/opt/bamboo-cut/models/bamboo_detection.onnx"
  confidence_threshold: 0.5
  use_tensorrt: true
```

### 性能配置 (`config/performance_config.yaml`)

```yaml
# 推理配置
inference:
  batch_size: 1
  max_threads: 4
  use_fp16: true

# 摄像头配置
camera:
  buffer_size: 10
  use_hardware_acceleration: true

# 模型优化
optimization:
  enable_ghostconv: true
  enable_gsconv: true
  enable_vov_gscsp: true
  enable_nam: true
```

## 🎮 使用说明

### 1. 启动系统

```bash
# 方式1: 通过systemd服务
sudo systemctl start bamboo-cut
sudo systemctl status bamboo-cut

# 方式2: 直接运行
cd /opt/bamboo-cut
./start_bamboo_cut.sh

# 方式3: 调试模式
./bamboo_cut_backend --debug
```

### 2. 界面操作

- **主界面**: 实时视频流 + AI检测结果
- **控制面板**: 系统状态、参数调节
- **设置页面**: 摄像头配置、AI参数
- **日志查看**: 系统运行日志

### 3. PLC通信

系统作为Modbus TCP服务器，PLC作为客户端连接：

```
服务器地址: 192.168.1.10:502
寄存器映射:
- 40001: PLC心跳 (R/W)
- 40002: 视觉系统状态 (R)
- 40101: 坐标数量 (R)
- 40102+: 坐标数据 (R)
```

## 🔧 开发指南

### 添加新的检测算法

1. 在 `cpp_backend/src/vision/` 中实现新的检测器类
2. 继承 `BambooDetector` 基类
3. 实现 `detect()` 和 `initialize()` 方法
4. 在CMakeLists.txt中添加源文件

### 扩展Flutter界面

1. 在 `flutter_frontend/lib/screens/` 中创建新页面
2. 使用Provider模式管理状态
3. 遵循工业设计主题 (`theme/industrial_theme.dart`)
4. 适配触屏操作 (最小触摸目标48dp)

### 性能优化

1. **C++后端优化**:
   - 使用TensorRT FP16模式
   - 启用GStreamer硬件解码
   - 调整线程池大小

2. **Flutter前端优化**:
   - 减少重绘区域
   - 使用const构造函数
   - 合理使用Provider.selector

## 📊 性能基准

### Jetson Xavier NX

| 组件 | 指标 | 性能 |
|------|------|------|
| AI推理 | FPS | 30+ |
| AI推理 | 延迟 | <33ms |
| 摄像头 | FPS | 30 |
| 摄像头 | 延迟 | <16ms |
| 总体 | CPU使用率 | <30% |
| 总体 | 内存使用 | <1GB |

### x86_64 工控机

| 组件 | 指标 | 性能 |
|------|------|------|
| AI推理 | FPS | 15+ |
| AI推理 | 延迟 | <66ms |
| 摄像头 | FPS | 30 |
| 总体 | CPU使用率 | <50% |
| 总体 | 内存使用 | <2GB |

## 🐛 故障排除

### 常见问题

1. **TensorRT模型加载失败**
   ```bash
   # 检查CUDA和TensorRT安装
   nvcc --version
   dpkg -l | grep tensorrt
   
   # 重新生成TensorRT引擎
   trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
   ```

2. **摄像头连接失败**
   ```bash
   # 检查摄像头设备
   v4l2-ctl --list-devices
   
   # 测试GStreamer pipeline
   gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink
   ```

3. **Flutter编译错误**
   ```bash
   # 清理缓存
   flutter clean
   flutter pub get
   
   # 重新构建
   flutter build linux --release
   ```

### 日志查看

```bash
# 系统日志
journalctl -u bamboo-cut -f

# 应用日志
tail -f /var/log/bamboo-cut/backend.log

# Flutter日志
tail -f /var/log/bamboo-cut/frontend.log
```

## 📈 版本历史

- **v2.0.0** (当前版本)
  - C++/Flutter全新架构
  - TensorRT GPU加速
  - 工业级嵌入式界面
  - 跨平台部署支持

- **v1.x.x** (Python版本)
  - Python + GTK4界面
  - OpenCV CPU推理
  - 基础Modbus通信

## 🤝 贡献指南

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

- 📧 邮箱: support@bamboo-cut.com
- 📱 技术支持: +86-xxx-xxxx-xxxx
- 🌐 官网: https://www.bamboo-cut.com
- 📚 文档: https://docs.bamboo-cut.com

---

**智能切竹机 v2.0 - 工业4.0智能制造解决方案** 🏭✨ 