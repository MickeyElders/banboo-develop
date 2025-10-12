#!/bin/bash

# NVIDIA-DRM 驱动启用脚本
# 用于将系统从tegra_drm迁移到nvidia-drm

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== NVIDIA-DRM 驱动启用脚本 ===${NC}"
echo "将系统从tegra_drm迁移到nvidia-drm驱动"
echo

# 检查当前状态
echo -e "${YELLOW}[检查]${NC} 当前DRM驱动状态:"
if lsmod | grep -q "tegra_drm"; then
    echo -e "${RED}  - tegra_drm 模块已加载${NC}"
    TEGRA_LOADED=true
else
    echo -e "${GREEN}  - tegra_drm 模块未加载${NC}"
    TEGRA_LOADED=false
fi

if lsmod | grep -q "nvidia.*drm"; then
    echo -e "${GREEN}  - nvidia-drm 模块已加载${NC}"
    NVIDIA_LOADED=true
else
    echo -e "${RED}  - nvidia-drm 模块未加载${NC}"
    NVIDIA_LOADED=false
fi

echo

# 停止相关服务
echo -e "${YELLOW}[停止]${NC} 停止相关服务..."
systemctl stop bamboo-cpp-lvgl.service 2>/dev/null || true
systemctl stop display-manager 2>/dev/null || true
systemctl stop gdm 2>/dev/null || true
systemctl stop lightdm 2>/dev/null || true

# 卸载tegra_drm模块 - 增强版本
if [ "$TEGRA_LOADED" = true ]; then
    echo -e "${YELLOW}[卸载]${NC} 强制卸载tegra_drm模块..."
    
    # 停止所有可能使用DRM的进程
    echo "  停止可能使用DRM的进程..."
    pkill -f "Xorg" 2>/dev/null || true
    pkill -f "wayland" 2>/dev/null || true
    pkill -f "weston" 2>/dev/null || true
    pkill -f "lvgl" 2>/dev/null || true
    pkill -f "bamboo" 2>/dev/null || true
    sleep 2
    
    # 强制卸载所有依赖模块
    echo "  强制卸载依赖模块..."
    MODULES_TO_REMOVE="cec drm_kms_helper nvhwpm host1x"
    for module in $MODULES_TO_REMOVE; do
        if lsmod | grep -q "^$module "; then
            echo "    强制卸载: $module"
            rmmod -f "$module" 2>/dev/null || true
        fi
    done
    
    # 多次尝试卸载tegra_drm
    echo "  强制卸载 tegra_drm..."
    for i in {1..5}; do
        if lsmod | grep -q "^tegra_drm "; then
            echo "    尝试 $i/5: 卸载 tegra_drm"
            rmmod -f tegra_drm 2>/dev/null || true
            sleep 1
        else
            echo "    tegra_drm 已成功卸载"
            break
        fi
    done
    
    # 永久禁用tegra_drm
    echo "  永久禁用tegra_drm..."
    cat > /etc/modprobe.d/blacklist-tegra-drm.conf << EOF
# 禁用 tegra_drm 模块
blacklist tegra_drm
install tegra_drm /bin/true
EOF
    
    echo -e "${GREEN}  tegra_drm 强制卸载完成${NC}"
fi

# 配置NVIDIA驱动参数 - 增强版本
echo -e "${YELLOW}[配置]${NC} 配置nvidia-drm模块参数..."

# 移除旧的nvidia模块
echo "  移除旧的nvidia模块..."
modprobe -r nvidia_drm 2>/dev/null || true
modprobe -r nvidia_modeset 2>/dev/null || true
modprobe -r nvidia_uvm 2>/dev/null || true
modprobe -r nvidia 2>/dev/null || true

# 创建强化的nvidia-drm配置文件
cat > /etc/modprobe.d/nvidia-drm.conf << EOF
# NVIDIA DRM 模块配置 - 强化版本
options nvidia-drm modeset=1 fbdev=1
options nvidia NVreg_DeviceFileUID=0 NVreg_DeviceFileGID=44 NVreg_DeviceFileMode=0664
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_EnableMSI=1
options nvidia NVreg_UsePageAttributeTable=1

# 确保nvidia-drm优先加载
softdep drm pre: nvidia-drm
EOF

# 同时更新modules文件确保启动时加载
echo "  更新模块加载配置..."
cat > /etc/modules-load.d/nvidia-drm.conf << EOF
# NVIDIA DRM 模块自动加载
nvidia
nvidia_uvm
nvidia_drm
EOF

echo -e "${GREEN}  nvidia-drm 配置完成${NC}"

# 加载nvidia-drm模块
echo -e "${YELLOW}[加载]${NC} 加载nvidia-drm模块..."

# 确保nvidia模块先加载
if ! lsmod | grep -q "^nvidia "; then
    echo "  加载 nvidia 基础模块"
    modprobe nvidia
fi

# 加载nvidia-drm模块
echo "  加载 nvidia-drm 模块"
modprobe nvidia-drm modeset=1

# 验证加载状态
if lsmod | grep -q "nvidia.*drm"; then
    echo -e "${GREEN}  nvidia-drm 模块加载成功${NC}"
else
    echo -e "${RED}  nvidia-drm 模块加载失败${NC}"
    exit 1
fi

# 检查DRM设备
echo -e "${YELLOW}[验证]${NC} 检查DRM设备..."
ls -la /dev/dri/

# 检查nvidia-drm设备
if [ -e "/dev/dri/card0" ]; then
    CARD_INFO=$(cat /sys/class/drm/card0/device/vendor 2>/dev/null || echo "unknown")
    if [ "$CARD_INFO" = "0x10de" ]; then
        echo -e "${GREEN}  nvidia-drm 设备正常 (/dev/dri/card0)${NC}"
    else
        echo -e "${YELLOW}  警告: card0 可能不是NVIDIA设备${NC}"
    fi
else
    echo -e "${RED}  错误: 未找到DRM设备${NC}"
    exit 1
fi

# 更新initramfs
echo -e "${YELLOW}[更新]${NC} 更新initramfs..."
update-initramfs -u

# 设置环境变量
echo -e "${YELLOW}[环境]${NC} 配置环境变量..."
cat > /etc/environment << EOF
# NVIDIA DRM 环境配置
EGL_PLATFORM=drm
__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
GBM_BACKEND=nvidia-drm
KMSDRM_DEVICE=/dev/dri/card0
LIBVA_DRIVER_NAME=nvidia
EOF

echo -e "${GREEN}  环境变量配置完成${NC}"

# 配置udev规则
echo -e "${YELLOW}[udev]${NC} 配置设备权限..."
cat > /etc/udev/rules.d/99-nvidia-drm.rules << EOF
# NVIDIA DRM 设备权限配置
KERNEL=="card*", SUBSYSTEM=="drm", DRIVERS=="nvidia-drm", MODE="0666"
KERNEL=="controlD*", SUBSYSTEM=="drm", DRIVERS=="nvidia-drm", MODE="0666"
KERNEL=="renderD*", SUBSYSTEM=="drm", DRIVERS=="nvidia-drm", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger

echo -e "${GREEN}  设备权限配置完成${NC}"

# 最终验证
echo
echo -e "${BLUE}=== 最终验证 ===${NC}"
echo -e "${YELLOW}模块状态:${NC}"
lsmod | grep -E "nvidia|tegra.*drm" | head -10

echo -e "${YELLOW}DRM设备:${NC}"
ls -la /dev/dri/

echo -e "${YELLOW}当前显示驱动:${NC}"
for card in /dev/dri/card*; do
    if [ -e "$card" ]; then
        card_num=$(basename "$card" | sed 's/card//')
        driver=$(cat /sys/class/drm/card${card_num}/device/driver/module/name 2>/dev/null || echo "unknown")
        echo "  $card: $driver"
    fi
done

echo
echo -e "${GREEN}=== NVIDIA-DRM 驱动启用完成 ===${NC}"
echo -e "${YELLOW}请重启系统或重新启动应用程序以使用新驱动${NC}"
echo
echo "验证命令:"
echo "  lsmod | grep nvidia"
echo "  ls -la /dev/dri/"
echo "  make nvidia-drm-test"