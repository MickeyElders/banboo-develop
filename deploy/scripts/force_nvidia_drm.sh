#!/bin/bash

# 强制NVIDIA-DRM驱动迁移脚本
# 用于在Jetson设备上强制从tegra_drm切换到nvidia-drm

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 强制NVIDIA-DRM驱动迁移脚本 ===${NC}"
echo "适用于Jetson设备的强制驱动切换"
echo

# 检查是否为root权限
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}错误: 此脚本必须以root权限运行${NC}"
   exit 1
fi

# 备份当前状态
echo -e "${YELLOW}[备份]${NC} 备份当前驱动状态..."
mkdir -p /root/drm_migration_backup
lsmod > /root/drm_migration_backup/lsmod_before.txt
ls -la /dev/dri/ > /root/drm_migration_backup/dri_before.txt 2>/dev/null || echo "无DRI设备" > /root/drm_migration_backup/dri_before.txt

# 强制进入文本模式
echo -e "${YELLOW}[模式]${NC} 切换到文本模式..."
systemctl set-default multi-user.target
systemctl stop display-manager 2>/dev/null || true
systemctl stop gdm 2>/dev/null || true
systemctl stop lightdm 2>/dev/null || true
systemctl stop bamboo-cpp-lvgl.service 2>/dev/null || true

# 杀死所有图形进程
echo -e "${YELLOW}[清理]${NC} 终止所有图形相关进程..."
pkill -f "Xorg" 2>/dev/null || true
pkill -f "weston" 2>/dev/null || true
pkill -f "wayland" 2>/dev/null || true
pkill -f "lvgl" 2>/dev/null || true
pkill -f "bamboo" 2>/dev/null || true
pkill -f "gstreamer" 2>/dev/null || true
sleep 3

# 卸载所有DRM相关模块
echo -e "${YELLOW}[卸载]${NC} 强制卸载所有DRM模块..."

# 按依赖关系逆序卸载
MODULES_TO_REMOVE=(
    "cec"
    "drm_kms_helper" 
    "nvhwpm"
    "host1x"
    "tegra_drm"
    "nvidia_drm"
    "nvidia_modeset"
    "nvidia_uvm"
    "nvidia"
)

for module in "${MODULES_TO_REMOVE[@]}"; do
    if lsmod | grep -q "^$module "; then
        echo "  强制卸载: $module"
        rmmod -f "$module" 2>/dev/null || true
        sleep 0.5
    fi
done

# 确保tegra_drm完全移除
for i in {1..10}; do
    if lsmod | grep -q "tegra_drm"; then
        echo "  第${i}次尝试移除tegra_drm..."
        rmmod -f tegra_drm 2>/dev/null || true
        sleep 1
    else
        echo -e "${GREEN}  tegra_drm已完全移除${NC}"
        break
    fi
done

# 永久黑名单tegra相关模块
echo -e "${YELLOW}[黑名单]${NC} 添加tegra模块到黑名单..."
cat > /etc/modprobe.d/blacklist-tegra.conf << 'EOF'
# 禁用Tegra DRM模块
blacklist tegra_drm
blacklist tegra_wmark
install tegra_drm /bin/true
install tegra_wmark /bin/true
EOF

# 强制配置nvidia-drm
echo -e "${YELLOW}[配置]${NC} 强制配置nvidia-drm..."
cat > /etc/modprobe.d/nvidia-force.conf << 'EOF'
# 强制NVIDIA DRM配置
options nvidia-drm modeset=1 fbdev=1
options nvidia NVreg_DeviceFileUID=0 NVreg_DeviceFileGID=44 NVreg_DeviceFileMode=0664
options nvidia NVreg_PreserveVideoMemoryAllocations=0
options nvidia NVreg_EnableMSI=1
options nvidia NVreg_UsePageAttributeTable=1
options nvidia NVreg_InitializeSystemMemoryAllocations=0

# 确保nvidia-drm优先
softdep drm pre: nvidia-drm
alias drm nvidia-drm
EOF

# 配置模块自动加载
cat > /etc/modules-load.d/nvidia-force.conf << 'EOF'
# 强制加载NVIDIA模块
nvidia
nvidia_uvm
nvidia_drm
EOF

# 强制加载nvidia模块
echo -e "${YELLOW}[加载]${NC} 强制加载nvidia-drm模块..."

# 按正确顺序加载
modprobe nvidia || {
    echo -e "${RED}错误: 无法加载nvidia模块${NC}"
    echo "请检查NVIDIA驱动是否正确安装"
    exit 1
}

modprobe nvidia_uvm || echo "nvidia_uvm加载失败（可选）"

modprobe nvidia_drm modeset=1 || {
    echo -e "${RED}错误: 无法加载nvidia-drm模块${NC}"
    exit 1
}

# 验证加载结果
echo -e "${YELLOW}[验证]${NC} 验证nvidia-drm加载状态..."
sleep 2

if lsmod | grep -q "nvidia.*drm"; then
    echo -e "${GREEN}  ✅ nvidia-drm模块加载成功${NC}"
else
    echo -e "${RED}  ❌ nvidia-drm模块加载失败${NC}"
    exit 1
fi

if lsmod | grep -q "tegra_drm"; then
    echo -e "${RED}  ❌ tegra_drm仍然存在${NC}"
    exit 1
else
    echo -e "${GREEN}  ✅ tegra_drm已完全移除${NC}"
fi

# 检查DRM设备
echo -e "${YELLOW}[设备]${NC} 检查DRM设备..."
ls -la /dev/dri/ || echo "暂无DRM设备"

# 更新initramfs
echo -e "${YELLOW}[更新]${NC} 更新initramfs..."
update-initramfs -u

# 配置环境变量
echo -e "${YELLOW}[环境]${NC} 配置环境变量..."
cat > /etc/environment << 'EOF'
# NVIDIA DRM 强制环境配置
EGL_PLATFORM=drm
__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
GBM_BACKEND=nvidia-drm
KMSDRM_DEVICE=/dev/dri/card0
LIBVA_DRIVER_NAME=nvidia
EOF

# 保存最终状态
echo -e "${YELLOW}[保存]${NC} 保存迁移后状态..."
lsmod > /root/drm_migration_backup/lsmod_after.txt
ls -la /dev/dri/ > /root/drm_migration_backup/dri_after.txt 2>/dev/null || echo "无DRI设备" > /root/drm_migration_backup/dri_after.txt

# 最终报告
echo
echo -e "${BLUE}=== 强制迁移完成 ===${NC}"
echo -e "${YELLOW}当前DRM模块状态:${NC}"
lsmod | grep -E "drm|nvidia" | head -10

echo
echo -e "${GREEN}=== 迁移成功完成 ===${NC}"
echo -e "${YELLOW}请重启系统以确保配置生效${NC}"
echo
echo "备份文件位置: /root/drm_migration_backup/"
echo "验证命令: lsmod | grep nvidia"
echo "          ls -la /dev/dri/"