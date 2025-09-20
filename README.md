# 智能切竹机系统 v2.0 - C++/LVGl重构版

基于C++后端和LVGL前端的高性能工业级竹材切割系统，支持TensorRT GPU加速和嵌入式Linux部署。



### 🎯 核心特性

- **🚀 高性能C++后端**: TensorRT GPU加速AI推理，提升3-5倍性能
- **📱 Flutter嵌入式前端**: 工业级触屏界面，完美适配LXDE环境
- **🔧 硬件加速**: GStreamer + DeepStream摄像头硬件加速
- **⚡ 模型优化**: 集成GhostConv、GSConv、VoV-GSCSP、NAM轻量化技术
- **🌐 跨平台支持**: x86_64 和 ARM64 (Jetson) 双架构部署
- **📡 高效通信**: C++实现的Modbus TCP服务器，支持并发连接



## 📊 性能基准

### Jetson orin NX
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