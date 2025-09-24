# 智能竹材切割系统 v3.0 - 一体化架构版

基于C++和LVGL的高性能工业级竹材切割系统，采用单进程一体化架构，支持TensorRT GPU加速和Jetson Orin NX部署。

## 🚀 v3.0 重大更新

### 一体化架构 (NEW!)
- **单进程设计**: 前后端完全整合，消除IPC通信开销
- **线程安全数据桥接**: 高效的推理线程与UI线程数据交换
- **零拷贝优化**: 视频帧和检测结果的内存零拷贝传输
- **统一构建系统**: 一键构建和部署，简化运维复杂度

### 🎯 核心特性

- **🚀 高性能C++后端**: TensorRT GPU加速AI推理，提升3-5倍性能
- **📱 LVGL嵌入式前端**: 原生嵌入式GUI，完美适配触屏和显示器
- **🔧 硬件加速**: GStreamer + DeepStream摄像头硬件加速
- **⚡ 模型优化**: 集成GhostConv、GSConv、VoV-GSCSP、NAM轻量化技术
- **🌐 跨平台支持**: x86_64 和 ARM64 (Jetson) 双架构部署
- **📡 高效通信**: Modbus TCP服务器，支持PLC并发连接
- **🔗 一体化部署**: 单一可执行文件，自动依赖管理

## 📊 性能基准

### Jetson Orin NX (一体化架构)
- **AI推理**: 35+ FPS, <28ms延迟
- **摄像头**: 30 FPS, <12ms延迟  
- **UI响应**: 60 FPS, <16ms延迟
- **系统负载**: CPU <25%, 内存 <800MB
- **启动时间**: <5秒 (vs 原15秒双进程)

### x86_64 工控机
- **AI推理**: 20+ FPS, <50ms延迟
- **系统负载**: CPU <40%, 内存 <1.5GB

## 🏗️ 系统架构

### 一体化进程架构
```
┌─────────────────────────────────────────────────────────────┐
│                bamboo_integrated (单进程)                    │
├─────────────────────┬─────────────────────┬─────────────────┤
│   推理工作线程        │    数据桥接器         │   LVGL UI线程    │
│                    │                     │                │
│  ┌─────────────────┐ │  ┌─────────────────┐ │ ┌─────────────┐ │
│  │ CameraManager   │ │  │ VideoData       │ │ │ VideoView   │ │
│  │ BambooDetector  │ │  │ DetectionData   │ │ │ StatusBar   │ │
│  │ StereoVision    │◄┼─►│ SystemStats     │◄┼─►│ ControlPanel│ │
│  │ ModbusServer    │ │  │ (线程安全)       │ │ │ SettingsPage│ │
│  └─────────────────┘ │  └─────────────────┘ │ └─────────────┘ │
└─────────────────────┴─────────────────────┴─────────────────┘
```

### 数据流架构
```
摄像头 → AI引擎 → 数据桥接器 → LVGL界面
   ↓       ↓         ↓          ↑
立体视觉 → 检测结果 → 线程同步 → 用户交互
   ↓       ↓         ↓          ↑
坐标计算 → Modbus → 状态更新 → 触摸控制
```

## 📡 PLC通信协议

基于**Modbus TCP**的高性能工业通信，采用"视觉推送，PLC轮询"的数据交换模式。

### 🔗 通信架构

```
┌─────────────────┐     Modbus TCP      ┌─────────────────┐
│       PLC       │ ←─────────────────→ │   视觉系统       │
│   (客户端)       │    192.168.1.10     │ (一体化服务端)    │
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

- **响应延迟**: < 8ms (一体化优化)
- **并发支持**: 10+ PLC同时连接
- **数据吞吐**: 1500+ 坐标/秒
- **可靠性**: 心跳监控 + 自动重连

## 🛠️ 快速开始

### 环境要求

#### 基础依赖
```bash
# Ubuntu/Debian依赖
sudo apt update
sudo apt install build-essential cmake git pkg-config
sudo apt install libopencv-dev libgstreamer1.0-dev libmodbus-dev
sudo apt install nlohmann-json3-dev
```

#### Jetson Orin NX额外依赖
```bash
# Jetson特定依赖 
sudo apt install nvidia-jetpack  # TensorRT + DeepStream
sudo apt install libcuda1 nvidia-cuda-toolkit
```

### 一键部署

```bash
# 1. 克隆项目
git clone <repository-url>
cd banboo-develop

# 2. 一键构建和部署
make deploy

# 3. 验证系统状态
make status
make logs
```

### 手动构建

```bash
# 1. 创建构建目录
mkdir build_integrated && cd build_integrated

# 2. 配置CMake (Jetson Orin NX)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON \
      -DENABLE_TENSORRT=ON \
      -DCMAKE_CXX_FLAGS="-mcpu=cortex-a78 -O3" \
      ..

# 3. 编译
make -j$(nproc)

# 4. 安装
sudo make install
```

## 📁 项目结构

```
banboo-develop/
├── integrated_main.cpp          # 一体化主程序
├── CMakeLists.txt               # 统一构建配置
├── Makefile                     # 一体化部署脚本
├── config/                      # 系统配置文件
│   ├── system_config.yaml       # 主配置文件
│   ├── stereo_calibration.xml   # 立体视觉标定
│   └── performance_config.yaml  # 性能优化配置
├── cpp_backend/                 # C++后端组件
│   ├── include/bamboo_cut/      # 头文件
│   │   ├── vision/              # 视觉模块
│   │   ├── communication/       # 通信模块
│   │   └── core/                # 核心模块
│   └── src/                     # 源文件实现
├── lvgl_frontend/               # LVGL前端组件
│   ├── include/                 # 前端头文件
│   │   ├── gui/                 # GUI组件
│   │   ├── display/             # 显示驱动
│   │   └── input/               # 输入驱动
│   └── src/                     # 前端实现
├── models/                      # AI模型文件
│   ├── best.pt                  # YOLOv8模型
│   └── convert_model.py         # 模型转换脚本
└── docs/                        # 技术文档
```

## 🔧 配置说明

### 系统配置 (config/system_config.yaml)
```yaml
camera:
  width: 1920
  height: 1080
  fps: 30
  device: "/dev/video0"

ai:
  model_path: "./models/best.pt"
  confidence_threshold: 0.5
  use_tensorrt: true

modbus:
  ip: "0.0.0.0"
  port: 502
```

### 性能调优 (config/performance_config.yaml)
```yaml
threads:
  inference_priority: 10
  ui_priority: 5
  
memory:
  enable_memory_pool: true
  max_frame_buffer: 10

gpu:
  enable_zero_copy: true
  memory_fraction: 0.8
```

## 📋 运维管理

### 系统服务

```bash
# 启动服务
sudo systemctl start bamboo-integrated
sudo systemctl enable bamboo-integrated

# 检查状态
sudo systemctl status bamboo-integrated

# 查看日志
sudo journalctl -u bamboo-integrated -f
```

### 监控命令

```bash
# 实时性能监控
make monitor

# 系统状态检查
make status

# 日志查看
make logs

# 重启服务
make restart
```

## 🐛 故障排除

### 常见问题

1. **CUDA库找不到**
   ```bash
   # 检查CUDA安装
   ls /usr/local/cuda*/
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **TensorRT初始化失败**
   ```bash
   # 检查TensorRT版本
   dpkg -l | grep tensorrt
   # 重新生成模型文件
   python models/convert_model.py
   ```

3. **摄像头无法访问**
   ```bash
   # 检查设备权限
   ls -la /dev/video*
   sudo usermod -a -G video $USER
   ```

4. **LVGL显示问题**
   ```bash
   # 检查显示设备
   export DISPLAY=:0
   # 检查触摸设备
   ls /dev/input/event*
   ```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📞 技术支持

如有技术问题，请通过以下方式联系：

- 🐛 **Bug 报告**: 在 GitHub Issues 中提交
- 💬 **技术讨论**: 在 Discussions 中发起话题
- 📧 **商业支持**: 通过邮件联系技术团队

---

**版本**: v3.0.0-integrated  
**更新日期**: 2024年9月  
**兼容平台**: Jetson Orin NX, x86_64 Linux