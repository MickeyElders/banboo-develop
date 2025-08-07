# 智能切竹机后端 - C++ 实现

## 概述

这是智能切竹机系统的 C++ 后端实现，包含立体视觉识别、Modbus TCP 通信、PLC 集成等功能。

## 特性

- **立体视觉系统**: 双摄像头深度检测和 3D 坐标计算
- **Modbus TCP 通信**: 与 PLC 系统的实时数据交换
- **高性能 AI 推理**: 基于 OpenCV 和 TensorRT 的目标检测
- **自动依赖管理**: 使用 CMake FetchContent 自动下载和构建依赖库
- **跨平台支持**: 支持 x86_64 和 ARM64 (Jetson) 架构

## 依赖打包

本项目使用 **CMake FetchContent** 自动管理依赖库，无需手动安装：

### 自动下载的依赖

1. **libmodbus v3.1.10**: Modbus TCP/RTU 通信库
2. **nlohmann/json v3.11.3**: JSON 数据处理库

### 系统依赖

以下依赖需要系统安装：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio
```

## 构建

### 快速构建

```bash
# 进入后端目录
cd cpp_backend

# 运行构建脚本 (Ubuntu)
./build.sh
```

### 手动构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 运行测试
make test

# 安装 (可选)
sudo make install
```

## 配置

### 立体视觉配置

1. **相机设备**: 默认使用 `/dev/video0` (左) 和 `/dev/video1` (右)
2. **分辨率**: 1280x720 @ 30fps
3. **标定文件**: `stereo_calibration.xml`

### Modbus TCP 配置

- **服务器 IP**: 192.168.1.10
- **端口**: 502
- **心跳间隔**: 100ms
- **响应超时**: 1000ms

## 使用

### 运行主程序

```bash
# 从构建目录运行
./bamboo_cut_backend

# 或从安装目录运行
bamboo_cut_backend
```

### 立体相机标定

```bash
# 运行标定工具
./tools/stereo_calibration_tool
```

## 项目结构

```
cpp_backend/
├── include/                 # 头文件
│   └── bamboo_cut/
│       ├── communication/   # 通信模块
│       ├── vision/         # 视觉模块
│       └── core/           # 核心模块
├── src/                    # 源代码
│   ├── communication/      # Modbus TCP 实现
│   ├── vision/            # 立体视觉实现
│   └── main.cpp           # 主程序
├── tools/                  # 工具程序
│   └── stereo_calibration_tool.cpp
├── tests/                  # 单元测试
├── examples/               # 示例代码
├── CMakeLists.txt         # CMake 配置
└── build.sh               # 构建脚本
```

## 故障排除

### 依赖下载失败

如果自动下载依赖失败，系统会回退到查找系统安装的版本：

```bash
# 手动安装 libmodbus
sudo apt install libmodbus-dev

# 或从源码编译
git clone https://github.com/stephane/libmodbus.git
cd libmodbus
./autogen.sh
./configure
make
sudo make install
```

### 相机访问权限

```bash
# 添加用户到 video 组
sudo usermod -a -G video $USER

# 重新登录或重启
sudo reboot
```

### CUDA 支持

如果检测到 CUDA，会自动启用 GPU 加速：

```bash
# 检查 CUDA 版本
nvcc --version

# 手动启用/禁用 CUDA
cmake .. -DENABLE_CUDA=ON
```

## 开发

### 添加新依赖

在 `CMakeLists.txt` 中添加新的 FetchContent 配置：

```cmake
FetchContent_Declare(
    library_name
    GIT_REPOSITORY https://github.com/user/repo.git
    GIT_TAG v1.0.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(library_name)
```

### 代码风格

- 使用 C++17 标准
- 遵循 Google C++ 风格指南
- 所有注释使用中文
- 使用 `LOG_*` 宏进行日志记录

## 许可证

本项目采用 MIT 许可证。 