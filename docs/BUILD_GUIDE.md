# 智能切竹机统一编译系统使用指南

## 概述

智能切竹机项目采用C++后端 + Flutter前端的架构，为了简化编译流程，我们提供了三种统一的编译方式：

1. **Bash脚本** (`build_all.sh`) - Linux/macOS
2. **PowerShell脚本** (`build_all.ps1`) - Windows
3. **Makefile** (`Makefile`) - 跨平台

## 快速开始

### Linux/macOS 用户

```bash
# 给脚本执行权限
chmod +x build_all.sh

# 编译所有组件 (发布模式)
./build_all.sh

# 调试模式编译
./build_all.sh -d

# 仅编译C++后端
./build_all.sh -c

# 仅编译Flutter前端
./build_all.sh -f

# 编译嵌入式版本
./build_all.sh -t embedded
```

### Windows 用户

```powershell
# 编译所有组件 (发布模式)
.\build_all.ps1

# 调试模式编译
.\build_all.ps1 -Debug

# 仅编译C++后端
.\build_all.ps1 -CppOnly

# 仅编译Flutter前端
.\build_all.ps1 -FlutterOnly

# 编译Android版本
.\build_all.ps1 -Platform android
```

### 使用Makefile (推荐)

```bash
# 编译所有组件 (发布模式)
make

# 调试模式编译
make debug

# 仅编译C++后端
make cpp

# 仅编译Flutter前端
make flutter

# 编译嵌入式版本
make embedded

# 快速编译 (仅编译修改的文件)
make quick

# 清理编译文件
make clean

# 检查编译状态
make status
```

## 详细选项说明

### 编译模式

- **release** (默认): 发布模式，优化性能，减小文件大小
- **debug**: 调试模式，包含调试信息，便于调试

### 目标平台

- **linux**: Linux桌面版本
- **windows**: Windows桌面版本
- **android**: Android移动版本
- **web**: Web浏览器版本

### 目标类型

- **desktop** (默认): 桌面应用
- **embedded**: 嵌入式系统 (Jetson等)
- **mobile**: 移动应用

### 并行编译

- **JOBS**: 并行编译任务数 (默认: 4)
- 建议设置为CPU核心数

## 编译配置示例

### 基础编译

```bash
# 发布模式编译所有组件
make

# 调试模式编译所有组件
make BUILD_TYPE=debug

# 指定并行任务数
make JOBS=8
```

### 平台特定编译

```bash
# Linux桌面版本
make PLATFORM=linux

# Windows桌面版本
make PLATFORM=windows

# Android版本
make PLATFORM=android
```

### 嵌入式编译

```bash
# 编译嵌入式版本 (Jetson等)
make embedded

# 嵌入式版本 + 调试模式
make embedded BUILD_TYPE=debug
```

## 依赖管理

### 自动安装依赖 (Ubuntu/Debian)

```bash
# 安装所有依赖
make install-deps-ubuntu

# 重新加载环境变量
source ~/.bashrc
```

### 手动安装依赖

#### Linux 依赖

```bash
# 基础工具
sudo apt install cmake build-essential git pkg-config

# OpenCV
sudo apt install libopencv-dev

# GStreamer
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Modbus
sudo apt install libmodbus-dev

# JSON库
sudo apt install nlohmann-json3-dev

# Flutter
curl -fsSL https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.16.5-stable.tar.xz | sudo tar -xJ -C /opt
echo 'export PATH="/opt/flutter/bin:$PATH"' >> ~/.bashrc
```

#### Windows 依赖

1. **Visual Studio Build Tools 2019+**
2. **CMake**: https://cmake.org/download/
3. **Git**: https://git-scm.com/download/win
4. **Flutter**: https://flutter.dev/docs/get-started/install/windows
5. **OpenCV**: https://opencv.org/releases/

### 检查依赖

```bash
# 检查所有依赖
make check-deps

# 或使用脚本
./build_all.sh --help
```

## 编译输出

### 文件结构

```
项目根目录/
├── cpp_backend/
│   ├── build/                    # 发布版本构建目录
│   ├── build_debug/              # 调试版本构建目录
│   └── build_embedded/           # 嵌入式版本构建目录
├── flutter_frontend/
│   └── build/                    # Flutter构建输出
└── bamboo_cut_*.tar.gz          # 部署包
```

### 可执行文件

- **C++后端**: `cpp_backend/build/bamboo_cut_backend`
- **Flutter前端**: `flutter_frontend/build/linux/x64/release/bundle/`

## 部署包

编译完成后会自动创建部署包：

```bash
# 创建部署包
make package

# 或使用脚本
./build_all.sh
```

部署包包含：
- C++后端可执行文件
- Flutter前端应用
- 配置文件
- 启动脚本
- 文档

## 常见问题

### 编译错误

1. **依赖缺失**
   ```bash
   make check-deps
   make install-deps-ubuntu  # Ubuntu/Debian
   ```

2. **权限问题**
   ```bash
   chmod +x build_all.sh
   ```

3. **Flutter环境问题**
   ```bash
   flutter doctor
   flutter pub get
   ```

### 性能优化

1. **增加并行任务数**
   ```bash
   make JOBS=8
   ```

2. **使用快速编译**
   ```bash
   make quick  # 仅编译修改的文件
   ```

3. **清理缓存**
   ```bash
   make clean
   flutter clean
   ```

### 调试技巧

1. **查看编译状态**
   ```bash
   make status
   ```

2. **查看编译信息**
   ```bash
   make info
   ```

3. **运行测试**
   ```bash
   make test
   ```

## 高级用法

### 自定义编译配置

```bash
# 自定义CMake参数
cd cpp_backend/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# 自定义Flutter参数
cd flutter_frontend
flutter build linux --release --dart-define=ENABLE_DEBUG=false
```

### 持续集成

```bash
# CI/CD脚本示例
#!/bin/bash
set -e

# 安装依赖
make install-deps-ubuntu

# 编译
make BUILD_TYPE=release JOBS=4

# 测试
make test

# 打包
make package
```

### 多平台编译

```bash
# 编译多个平台版本
for platform in linux windows android; do
    make PLATFORM=$platform BUILD_TYPE=release
done
```

## 技术支持

如果遇到编译问题，请：

1. 检查依赖是否完整安装
2. 查看错误日志
3. 尝试清理后重新编译
4. 提交Issue到项目仓库

## 更新日志

- **v1.0.0**: 初始版本，支持基础编译功能
- **v1.1.0**: 添加嵌入式编译支持
- **v1.2.0**: 添加Windows PowerShell脚本
- **v1.3.0**: 优化编译性能和错误处理 