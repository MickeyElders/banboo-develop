# 智能切竹机系统 v2.0 - C++/Flutter重构版

基于C++后端和Flutter前端的高性能工业级竹材切割系统，支持TensorRT GPU加速和嵌入式Linux部署。

## 📋 项目概述

智能切竹机系统v2.0是从Python全面重构为C++/Flutter的工业级解决方案，专为高性能生产环境设计。

### 🎯 核心特性

- **🚀 高性能C++后端**: TensorRT GPU加速AI推理，提升3-5倍性能
- **📱 Flutter嵌入式前端**: 工业级触屏界面，完美适配LXDE环境
- **🔧 硬件加速**: GStreamer + DeepStream摄像头硬件加速
- **⚡ 模型优化**: 集成GhostConv、GSConv、VoV-GSCSP、NAM轻量化技术
- **🌐 跨平台支持**: x86_64 和 ARM64 (Jetson) 双架构部署
- **📡 高效通信**: C++实现的Modbus TCP服务器，支持并发连接

## 🏗️ 系统架构与数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter 前端界面                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   实时视频显示   │ │   AI检测结果     │ │   系统控制面板   │  │
│  │    30 FPS      │ │   坐标标注      │ │   状态监控      │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │ FFI/Socket 通信
┌─────────────────────────────────────────────────────────────┐
│                      C++ 后端引擎                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │  TensorRT引擎   │ │  GStreamer管道   │ │  Modbus服务器   │  │
│  │  AI推理加速     │ │  摄像头硬件加速   │ │  PLC通信协议    │  │
│  │  <33ms延迟     │ │  多线程处理     │ │  <10ms响应     │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
│          │                      │                      │      │
│      坐标数据                视频流                 寄存器数据   │
│          │                      │                      │      │
└─────────────────────────────────────────────────────────────┘
             │                      │                      │
             ▼                      ▼                      ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   AI模型优化     │    │   硬件加速       │    │   PLC设备        │
   │ • GhostConv     │    │ • DeepStream    │    │ • Modbus TCP    │
   │ • GSConv        │    │ • V4L2/CSI      │    │ • 寄存器映射     │
   │ • VoV-GSCSP     │    │ • 零拷贝处理     │    │ • 心跳监控       │
   │ • NAM注意力     │    │ • 帧缓冲管理     │    │ • 指令响应       │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔄 数据流向说明

1. **📹 视频采集**: GStreamer硬件加速管道采集1920x1080@30fps视频流
2. **🧠 AI推理**: TensorRT引擎实时检测竹节位置，输出坐标数据  
3. **📡 数据推送**: Modbus服务器将坐标写入寄存器，设置就绪标志
4. **🔄 PLC轮询**: PLC定期读取寄存器，获取最新坐标数据
5. **📱 界面显示**: Flutter前端通过FFI获取实时视频和检测结果

## 🛠️ 技术栈

### C++ 后端
- **推理引擎**: TensorRT 8.x, OpenCV DNN (fallback)
- **摄像头处理**: GStreamer 1.0, DeepStream SDK (Jetson)
- **通信协议**: Modbus TCP (基于IEEE 802.3以太网标准)
- **网络架构**: 客户端-服务器模式，支持请求-响应和发布-订阅模式
- **构建系统**: CMake 3.16+, GCC 9+
- **标准兼容**: 遵循工业以太网和Modbus Application Protocol规范

### Flutter 前端
- **UI框架**: Flutter 3.16+ (Embedded Linux)
- **状态管理**: Provider + BLoC
- **通信**: FFI (C++ 绑定), WebSocket
- **平台**: Linux Desktop/Embedded

## 🚀 快速开始

### 方法一：使用统一编译系统（推荐）

```bash
# 克隆项目
git clone <repository-url> bamboo-cut-develop
cd bamboo-cut-develop

# Makefile编译（跨平台推荐）
make                    # 编译所有组件（发布模式）
make debug             # 调试模式编译
make embedded          # 嵌入式版本编译
make clean             # 清理编译文件

# 或使用脚本编译
chmod +x build_all.sh
./build_all.sh         # Linux/macOS
# 或
./build_all.ps1        # Windows PowerShell
```

### 方法二：自动化部署脚本

```bash
# 一键构建 (x86_64)
./deploy/scripts/build_and_deploy.sh --install-deps --arch x86_64

# Jetson ARM64构建
./deploy/scripts/build_and_deploy.sh --install-deps --tensorrt --arch aarch64

# 交叉编译ARM64
./deploy/scripts/build_and_deploy.sh --cross-compile --arch aarch64
```

### 部署安装

```bash
# 本地安装
./deploy/scripts/build_and_deploy.sh --deploy local

# 部署到Jetson
./deploy/scripts/build_and_deploy.sh --deploy jetson

# 部署到远程设备
./deploy/scripts/build_and_deploy.sh --deploy remote:192.168.1.100
```

### 启动系统

```bash
# systemd服务
sudo systemctl start bamboo-cut
sudo systemctl status bamboo-cut

# 直接运行
cd /opt/bamboo-cut
./start_bamboo_cut.sh
```

> 📖 **详细构建指南**: 查看 [构建系统使用指南](docs/BUILD_GUIDE.md) 了解更多编译选项和高级配置

## ⚙️ 配置说明

### 系统配置 (`config/system_config.yaml`)

```yaml
# 硬件配置
hardware:
  camera:
    device_id: "/dev/video0"
    width: 1920
    height: 1080

# 通信配置
communication:
  server_ip: "192.168.1.10"
  server_port: 502

# AI配置
ai:
  model_path: "/opt/bamboo-cut/models/bamboo_detection.onnx"
  use_tensorrt: true
```

## 📊 性能基准

### Jetson Xavier NX
- **AI推理**: 30+ FPS, <33ms延迟
- **摄像头**: 30 FPS, <16ms延迟
- **系统负载**: CPU <30%, 内存 <1GB

### x86_64 工控机
- **AI推理**: 15+ FPS, <66ms延迟
- **系统负载**: CPU <50%, 内存 <2GB

## 📡 PLC通信协议

基于**Modbus TCP**的高性能工业通信，采用"视觉推送，PLC轮询"的数据交换模式。

### 🔗 通信架构

```
┌─────────────────┐     Modbus TCP      ┌─────────────────┐
│       PLC       │ ←─────────────────→ │   视觉系统       │
│   (客户端)       │    192.168.1.10     │   (C++服务端)    │
│                 │      端口:502       │                 │
└─────────────────┘                     └─────────────────┘
```

### ⏱️ 通信时序流程

#### 1. 连接建立与心跳维持
```
PLC → 视觉系统: TCP连接 (192.168.1.10:502)
PLC → 视觉系统: 心跳写入 (40001寄存器, 每2秒)
视觉系统 → 内部: 监控心跳变化，更新连接状态
```

#### 2. 数据推送与轮询
```
摄像头 → AI引擎: 实时视频流
AI引擎 → Modbus服务: 推送检测结果
Modbus服务: 更新坐标寄存器 (40101-40148)
Modbus服务: 设置就绪标志 (40148=1)

循环轮询:
PLC → 视觉系统: 读取就绪标志 (40148)
如果 就绪标志=1:
    PLC → 视觉系统: 读取坐标数据 (40101-40147)
    PLC → 视觉系统: 清除就绪标志 (40148=0)
```

#### 3. PLC指令处理
```
PLC → 视觉系统: 写入指令码 (40003寄存器)
视觉系统: 读取并处理指令
视觉系统: 清除指令码 (40003=0)
视觉系统: 执行相应动作 (送料/切割/急停等)
```

### 📊 核心寄存器映射

| 寄存器 | 功能 | 类型 | 说明 |
|--------|------|------|------|
| 40001 | PLC心跳计数器 | UINT16 | PLC每2秒递增 |
| 40002 | 视觉系统状态 | UINT16 | 0=初始化, 100=就绪, 200+=错误 |
| 40003 | PLC指令寄存器 | UINT16 | 1=送料, 3=切割, 99=急停 |
| 40101 | 切点数量 | UINT16 | 检测到的切点数量(0-10) |
| 40102+ | 坐标数据 | FLOAT32 | IEEE754格式，单位:mm |
| 40148 | 坐标就绪标志 | UINT16 | 1=新数据, PLC读取后清零 |

### 🚀 性能特性

- **响应延迟**: < 10ms (局域网环境)
- **并发支持**: 10+ PLC同时连接
- **数据吞吐**: 1000+ 坐标/秒
- **可靠性**: 心跳监控 + 自动重连

**详细文档:**
- [C++版PLC通信协议文档](docs/cpp_plc_communication_protocol.md) - 完整协议规范
- [工业通信标准技术规范](docs/industrial_communication_standards.md) - 技术实现细节

## 🔧 开发指南

### 构建环境

```bash
# Ubuntu/Debian依赖
sudo apt install build-essential cmake git pkg-config
sudo apt install libopencv-dev libgstreamer1.0-dev libmodbus-dev

# Jetson额外依赖
sudo apt install nvidia-jetpack  # TensorRT + DeepStream
```

### 项目结构

```
bamboo-cut-develop/
├── README.md                 # 项目主文档
├── VERSION                   # 版本标识文件 (2.0.0)
├── Makefile                  # 跨平台统一编译配置
├── build_all.sh              # Linux/macOS 编译脚本
├── build_all.ps1             # Windows PowerShell 编译脚本
├── .vscode/                  # VS Code 配置
├── config/                   # 系统配置文件
│   ├── system_config.yaml    # 硬件和系统配置
│   ├── ai_optimization.yaml  # AI模型优化配置
│   └── performance_config.yaml # 性能调优配置
├── models/                   # AI模型文件目录 ⭐
│   ├── README.md             # 模型使用说明
│   ├── bamboo_detection.onnx # 竹节检测ONNX模型
│   └── bamboo_detection.trt  # TensorRT优化引擎
├── cpp_backend/              # C++ 后端源码
│   ├── CMakeLists.txt        # CMake 构建配置
│   ├── include/              # 头文件目录
│   ├── src/                  # 源文件目录
│   ├── tools/                # 开发工具
│   └── examples/             # 示例代码
├── flutter_frontend/         # Flutter 前端源码
│   ├── pubspec.yaml          # Flutter 项目配置
│   └── lib/                  # Dart 源文件
├── deploy/                   # 部署相关
│   └── scripts/              # 部署脚本
└── docs/                     # 技术文档
    ├── BUILD_GUIDE.md        # 详细编译指南
    ├── cpp_plc_communication_protocol.md  # PLC通信协议
    └── ... (其他技术文档)
```

## 📈 版本历史

- **v2.0.0** (当前版本) - C++/Flutter全新架构重构
- **v1.x.x** - Python版本 (已废弃)

## 🤝 贡献指南

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 技术支持

- 📧 邮箱: support@bamboo-cut.com
- 📚 文档: https://docs.bamboo-cut.com
- 🐛 问题反馈: GitHub Issues

---

**智能切竹机 v2.0 - 工业4.0智能制造解决方案** 🏭⚡ 