# 智能切竹机系统 DEB 包构建指南

本指南说明如何将智能切竹机系统打包为 DEB 安装包，以便在 Ubuntu/Debian 系统上快速部署。

## 环境要求

### 操作系统
- Ubuntu 18.04+ 或 Debian 10+
- 支持 x86_64 和 ARM64 架构（如 Jetson Nano）

### 必要软件包
```bash
sudo apt update
sudo apt install -y \
    dpkg-dev \
    build-essential \
    python3 \
    python3-pip \
    imagemagick \
    git
```

## 构建步骤

### 1. 准备构建环境
```bash
# 克隆项目（如果尚未克隆）
git clone https://github.com/bamboo-cut/bamboo-cut-develop.git
cd bamboo-cut-develop

# 确保所有必要文件都存在
ls -la main.py src/ models/ config/ package/
```

### 2. 设置文件权限（仅在 Linux 环境下）
```bash
chmod +x build_deb.py
chmod +x package/DEBIAN/postinst
chmod +x package/DEBIAN/prerm
chmod +x package/bamboo-cut
chmod +x package/bamboo-cut-kiosk
```

### 3. （可选）创建自定义图标
```bash
# 方法1：使用 ImageMagick 创建简单图标
convert -size 512x512 xc:'#4CAF50' \
        -font DejaVu-Sans-Bold -pointsize 80 \
        -fill white -gravity center \
        -annotate +0+0 '竹切' \
        package/bamboo-cut.png

# 方法2：从网络下载或使用设计软件创建
# 将图标文件保存为 package/bamboo-cut.png
```

### 4. 构建 DEB 包
```bash
# 运行构建脚本
./build_deb.py
```

### 5. 验证构建结果
```bash
# 检查生成的 DEB 包
ls -la build/bamboo-cut-intelligent-system_1.0.0_all.deb

# 查看包信息
dpkg --info build/bamboo-cut-intelligent-system_1.0.0_all.deb

# 查看包内容
dpkg --contents build/bamboo-cut-intelligent-system_1.0.0_all.deb
```

## 安装和测试

### 安装 DEB 包
```bash
# 安装包（将自动处理依赖）
sudo dpkg -i build/bamboo-cut-intelligent-system_1.0.0_all.deb

# 如果有依赖问题，修复依赖
sudo apt install -f
```

### 测试安装结果
```bash
# 检查服务状态
sudo systemctl status bamboo-cut
sudo systemctl status bamboo-cut-kiosk

# 测试命令行工具
bamboo-cut --help
bamboo-cut-demo --help
bamboo-cut-test

# 检查配置文件
cat /etc/bamboo-cut/system_config.yaml

# 查看日志
tail -f /opt/bamboo-cut/logs/system.log
```

### 启用服务
```bash
# 启动主服务
sudo systemctl start bamboo-cut
sudo systemctl enable bamboo-cut

# 或启用 Kiosk 模式（全屏触摸界面）
sudo systemctl start bamboo-cut-kiosk
sudo systemctl enable bamboo-cut-kiosk
```

## 卸载

### 完全卸载系统
```bash
# 停止所有服务
sudo systemctl stop bamboo-cut bamboo-cut-kiosk
sudo systemctl disable bamboo-cut bamboo-cut-kiosk

# 卸载包
sudo dpkg -r bamboo-cut-intelligent-system

# 清理残留文件（可选）
sudo rm -rf /opt/bamboo-cut
sudo rm -rf /etc/bamboo-cut
sudo userdel bamboo-cut
```

## 构建脚本说明

### build_deb.py 功能
- **自动化构建**: 一键创建完整的 DEB 包
- **目录结构**: 自动创建符合 Debian 标准的包结构
- **文件复制**: 智能复制源代码、配置、文档等文件
- **权限设置**: 自动设置正确的文件权限
- **大小计算**: 自动计算并更新安装后大小
- **依赖检查**: 验证构建环境和工具

### 包结构
```
bamboo-cut-intelligent-system_1.0.0_all.deb
├── DEBIAN/
│   ├── control          # 包元数据
│   ├── postinst         # 安装后脚本
│   └── prerm            # 卸载前脚本
├── opt/bamboo-cut/      # 主程序目录
│   ├── src/             # 源代码
│   ├── models/          # AI模型
│   ├── main.py          # 主程序
│   └── ...
├── etc/bamboo-cut/      # 配置文件
├── etc/systemd/system/  # 系统服务
├── usr/bin/             # 可执行脚本
├── usr/share/applications/ # 桌面文件
└── usr/share/doc/bamboo-cut/ # 文档
```

## 跨平台构建

### 在 Docker 中构建
```bash
# 创建 Ubuntu 构建环境
docker run -it --rm \
    -v $(pwd):/workspace \
    ubuntu:20.04 bash

# 在容器内执行构建
apt update && apt install -y dpkg-dev python3 imagemagick
cd /workspace
./build_deb.py
```

### 针对 ARM64 (Jetson Nano) 构建
```bash
# 在 ARM64 设备上直接构建
./build_deb.py

# 或使用交叉编译（在 x86_64 上为 ARM64 构建）
# 修改 control 文件中的 Architecture: arm64
sed -i 's/Architecture: all/Architecture: arm64/' package/DEBIAN/control
./build_deb.py
```

## 常见问题

### Q: 构建失败，提示缺少依赖
A: 确保安装了所有构建依赖：
```bash
sudo apt install dpkg-dev build-essential python3-dev
```

### Q: 图标显示不正确
A: 创建正确的 PNG 图标文件：
```bash
# 确保图标文件存在且为有效的 PNG 格式
file package/bamboo-cut.png
```

### Q: 服务无法启动
A: 检查权限和依赖：
```bash
# 检查 Python 依赖
pip3 install -r requirements.txt

# 检查服务文件权限
sudo systemctl daemon-reload
```

### Q: 在 Windows 上构建
A: DEB 包必须在 Linux 环境下构建，建议使用：
- WSL2 (Windows Subsystem for Linux)
- Docker Desktop
- 虚拟机（VirtualBox/VMware）

## 自定义构建

### 修改包信息
编辑 `build_deb.py` 中的配置：
```python
PROJECT_NAME = "your-custom-name"
VERSION = "1.0.1" 
ARCHITECTURE = "arm64"  # 或 "amd64"
```

### 添加额外文件
在 `copy_application_files()` 函数中添加：
```python
# 复制自定义文件
if os.path.exists("your_custom_file"):
    shutil.copy2("your_custom_file", f"{DEB_DIR}/opt/bamboo-cut/")
```

### 修改依赖关系
编辑 `package/DEBIAN/control` 文件中的 `Depends` 行。

## 发布和分发

### 创建软件仓库
```bash
# 使用 reprepro 创建 APT 仓库
sudo apt install reprepro

# 配置仓库
mkdir -p ~/deb-repo/conf
echo "Codename: bamboo-cut" > ~/deb-repo/conf/distributions
echo "Architectures: amd64 arm64 all" >> ~/deb-repo/conf/distributions
echo "Components: main" >> ~/deb-repo/conf/distributions

# 添加包到仓库
reprepro -b ~/deb-repo includedeb bamboo-cut \
    build/bamboo-cut-intelligent-system_1.0.0_all.deb
```

### 用户安装命令
```bash
# 添加仓库密钥和源
curl -s https://your-domain.com/deb-repo/key.gpg | sudo apt-key add -
echo "deb https://your-domain.com/deb-repo bamboo-cut main" | \
    sudo tee /etc/apt/sources.list.d/bamboo-cut.list

# 安装
sudo apt update
sudo apt install bamboo-cut-intelligent-system
```

这样用户就可以通过标准的 APT 包管理器安装和更新智能切竹机系统了。 