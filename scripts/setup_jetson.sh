#!/bin/bash
# 智能切竹机 - Jetson Nano系统设置脚本
# 自动配置Jetson Nano开发环境

set -e  # 遇到错误时退出

echo "========================================="
echo "智能切竹机 Jetson Nano 系统设置"
echo "========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否是Jetson设备
check_jetson() {
    log_info "检查Jetson设备类型..."
    
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_VERSION=$(cat /etc/nv_tegra_release)
        log_info "检测到Jetson设备: $JETSON_VERSION"
        return 0
    else
        log_warn "未检测到Jetson设备，继续在通用Linux环境下安装"
        return 1
    fi
}

# 更新系统
update_system() {
    log_info "更新系统包..."
    
    sudo apt update
    sudo apt upgrade -y
    
    log_info "安装基础工具..."
    sudo apt install -y \
        curl \
        wget \
        git \
        vim \
        htop \
        tree \
        unzip \
        build-essential \
        cmake \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools
}

# 设置网络配置
setup_network() {
    log_info "配置网络设置..."
    
    # 备份原始配置
    if [ -f /etc/netplan/01-netcfg.yaml ]; then
        sudo cp /etc/netplan/01-netcfg.yaml /etc/netplan/01-netcfg.yaml.bak
    fi
    
    # 创建静态IP配置
    read -p "是否配置静态IP? (y/n): " SETUP_STATIC_IP
    
    if [ "$SETUP_STATIC_IP" = "y" ]; then
        read -p "请输入IP地址 (默认: 192.168.1.10): " IP_ADDRESS
        IP_ADDRESS=${IP_ADDRESS:-192.168.1.10}
        
        read -p "请输入网关地址 (默认: 192.168.1.1): " GATEWAY
        GATEWAY=${GATEWAY:-192.168.1.1}
        
        read -p "请输入子网掩码 (默认: 24): " NETMASK
        NETMASK=${NETMASK:-24}
        
        log_info "配置静态IP: $IP_ADDRESS/$NETMASK"
        
        sudo tee /etc/netplan/01-netcfg.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses: [$IP_ADDRESS/$NETMASK]
      gateway4: $GATEWAY
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
EOF
        
        sudo netplan apply
        log_info "网络配置已更新"
    fi
}

# 安装Python依赖
install_python_deps() {
    log_info "安装Python依赖包..."
    
    # 升级pip
    python3 -m pip install --upgrade pip
    
    # 安装项目依赖
    if [ -f requirements.txt ]; then
        log_info "从requirements.txt安装依赖..."
        pip3 install -r requirements.txt
    else
        log_info "安装核心依赖包..."
        pip3 install \
            numpy \
            opencv-python \
            pymodbus \
            PyYAML \
            psutil \
            pyserial \
            pillow \
            matplotlib
    fi
    
    # 如果是Jetson，安装Jetson特定包
    if check_jetson; then
        log_info "安装Jetson特定的包..."
        
        # 安装JetPack组件（如果需要）
        # sudo apt install -y nvidia-jetpack
        
        # 安装Jetson GPIO
        pip3 install Jetson.GPIO
    fi
}

# 安装OpenCV (针对Jetson优化)
install_opencv() {
    log_info "安装OpenCV..."
    
    if check_jetson; then
        log_info "在Jetson设备上安装OpenCV..."
        
        # 对于Jetson Nano，建议使用预编译版本
        sudo apt install -y python3-opencv
        
        # 或者如果需要完整功能的OpenCV
        # pip3 install opencv-contrib-python
    else
        log_info "在通用设备上安装OpenCV..."
        pip3 install opencv-python opencv-contrib-python
    fi
    
    # 验证OpenCV安装
    python3 -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
}

# 设置摄像头权限
setup_camera_permissions() {
    log_info "设置摄像头权限..."
    
    # 将用户添加到video组
    sudo usermod -a -G video $USER
    
    # 设置udev规则
    sudo tee /etc/udev/rules.d/99-camera.rules > /dev/null <<EOF
# 摄像头设备权限
SUBSYSTEM=="video4linux", GROUP="video", MODE="0664"
SUBSYSTEM=="usb", ATTRS{idVendor}=="*", ATTRS{idProduct}=="*", GROUP="video", MODE="0664"
EOF
    
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    log_info "摄像头权限设置完成，需要重新登录生效"
}

# 配置GPIO权限（仅Jetson）
setup_gpio_permissions() {
    if check_jetson; then
        log_info "设置GPIO权限..."
        
        # 将用户添加到gpio组
        sudo groupadd -f gpio
        sudo usermod -a -G gpio $USER
        
        # 设置GPIO设备权限
        sudo tee /etc/udev/rules.d/99-gpio.rules > /dev/null <<EOF
# GPIO权限设置
SUBSYSTEM=="gpio", GROUP="gpio", MODE="0664"
KERNEL=="gpiochip[0-9]*", GROUP="gpio", MODE="0664"
EOF
        
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        
        log_info "GPIO权限设置完成"
    fi
}

# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    # 安装代码编辑器
    read -p "是否安装VS Code? (y/n): " INSTALL_VSCODE
    
    if [ "$INSTALL_VSCODE" = "y" ]; then
        log_info "安装VS Code..."
        
        # 添加微软GPG密钥
        wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
        sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
        
        # 添加VS Code仓库
        sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
        
        sudo apt update
        sudo apt install -y code
    fi
    
    # 安装其他有用工具
    sudo apt install -y \
        terminator \
        gparted \
        synaptic \
        software-properties-common
}

# 优化系统性能
optimize_performance() {
    if check_jetson; then
        log_info "优化Jetson Nano性能..."
        
        # 设置电源模式为最大性能
        sudo nvpmodel -m 0
        
        # 设置CPU和GPU频率
        sudo jetson_clocks
        
        # 增加swap空间
        if [ ! -f /swapfile ]; then
            log_info "创建swap文件..."
            sudo fallocate -l 4G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            
            # 添加到fstab
            echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
        fi
        
        log_info "Jetson Nano性能优化完成"
    else
        log_info "在通用设备上进行基础优化..."
        
        # 设置系统参数
        echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    fi
}

# 创建项目目录结构
setup_project_structure() {
    log_info "创建项目目录结构..."
    
    # 创建工作目录
    mkdir -p ~/bamboo-cutting/{data,logs,backup,temp}
    
    # 设置日志目录权限
    chmod 755 ~/bamboo-cutting/logs
    
    # 创建配置文件模板
    if [ ! -f ~/bamboo-cutting/config.yaml ]; then
        tee ~/bamboo-cutting/config.yaml > /dev/null <<EOF
# 智能切竹机配置文件
system:
  name: "智能切竹机测试平台"
  version: "1.0"
  
network:
  plc_host: "192.168.1.20"
  plc_port: 502
  
camera:
  device_id: 0
  resolution: [2048, 1536]
  fps: 30
  
logging:
  level: "INFO"
  file: "~/bamboo-cutting/logs/system.log"
EOF
    fi
    
    log_info "项目目录结构创建完成"
}

# 设置自动启动脚本
setup_autostart() {
    read -p "是否设置系统自动启动? (y/n): " SETUP_AUTOSTART
    
    if [ "$SETUP_AUTOSTART" = "y" ]; then
        log_info "配置自动启动..."
        
        # 创建systemd服务文件
        sudo tee /etc/systemd/system/bamboo-cutting.service > /dev/null <<EOF
[Unit]
Description=智能切竹机系统
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/bamboo-cutting
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        # 启用服务
        sudo systemctl enable bamboo-cutting.service
        
        log_info "自动启动服务已配置"
    fi
}

# 运行测试
run_tests() {
    log_info "运行系统测试..."
    
    # 测试Python环境
    python3 -c "
import sys
print(f'Python版本: {sys.version}')

try:
    import cv2
    print(f'OpenCV版本: {cv2.__version__}')
except ImportError:
    print('OpenCV未安装')

try:
    import numpy as np
    print(f'NumPy版本: {np.__version__}')
except ImportError:
    print('NumPy未安装')

try:
    import yaml
    print('PyYAML已安装')
except ImportError:
    print('PyYAML未安装')
"
    
    # 测试摄像头
    log_info "测试摄像头连接..."
    python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('摄像头连接正常')
    cap.release()
else:
    print('摄像头连接失败')
"
    
    # 测试网络
    log_info "测试网络连接..."
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        log_info "网络连接正常"
    else
        log_warn "网络连接异常"
    fi
}

# 生成安装报告
generate_report() {
    log_info "生成安装报告..."
    
    REPORT_FILE="jetson_setup_report.txt"
    
    tee $REPORT_FILE > /dev/null <<EOF
========================================
智能切竹机 Jetson Nano 安装报告
========================================
安装时间: $(date)
用户: $USER
主机名: $(hostname)

系统信息:
$(uname -a)

Python版本:
$(python3 --version)

安装的包:
$(pip3 list | grep -E "(opencv|numpy|yaml|modbus)")

网络配置:
$(ip addr show)

磁盘使用:
$(df -h)

内存信息:
$(free -h)

========================================
EOF
    
    log_info "安装报告已保存到: $REPORT_FILE"
}

# 主安装流程
main() {
    log_info "开始Jetson Nano系统设置..."
    
    # 检查运行权限
    if [ "$EUID" -eq 0 ]; then
        log_error "请不要以root用户运行此脚本"
        exit 1
    fi
    
    # 执行安装步骤
    check_jetson
    update_system
    setup_network
    install_python_deps
    install_opencv
    setup_camera_permissions
    setup_gpio_permissions
    install_dev_tools
    optimize_performance
    setup_project_structure
    setup_autostart
    run_tests
    generate_report
    
    log_info "Jetson Nano系统设置完成！"
    log_warn "请重新启动系统以确保所有设置生效"
    
    echo ""
    echo "下一步操作："
    echo "1. 重启系统: sudo reboot"
    echo "2. 连接PLC设备并配置网络"
    echo "3. 连接摄像头并测试"
    echo "4. 运行硬件测试: python3 test/test_hardware.py"
}

# 运行主程序
main "$@" 