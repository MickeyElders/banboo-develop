#!/bin/bash
# scripts/check_drm_state.sh
# 简化的DRM状态检查脚本

echo "========================================="
echo "DRM状态检查工具"
echo "========================================="

# 检查DRM设备
echo "1. DRM设备检查:"
if [ -e "/dev/dri/card0" ]; then
    echo "✅ /dev/dri/card0 存在"
    ls -la /dev/dri/
else
    echo "❌ /dev/dri/card0 不存在"
fi

echo ""

# 检查nvidia-drm驱动
echo "2. nvidia-drm驱动检查:"
if lsmod | grep -q nvidia_drm; then
    echo "✅ nvidia-drm驱动已加载"
    
    modeset_status=$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null || echo "N")
    echo "   modeset参数: $modeset_status"
    
    if [ "$modeset_status" = "Y" ]; then
        echo "✅ modeset已启用"
    else
        echo "⚠️  modeset未启用"
    fi
else
    echo "❌ nvidia-drm驱动未加载"
fi

echo ""

# 检查DRM调试信息
echo "3. DRM状态信息:"
if [ -e "/sys/kernel/debug/dri/0/state" ]; then
    echo "✅ DRM调试信息可用"
    echo "--- 当前Plane使用情况 ---"
    sudo cat /sys/kernel/debug/dri/0/state | grep -E "(crtc|plane)" | head -10
    echo "-------------------------"
else
    echo "⚠️  DRM调试信息不可用"
fi

echo ""

# 使用modetest检查资源
echo "4. modetest资源检查:"
if command -v modetest >/dev/null 2>&1; then
    echo "✅ modetest工具可用"
    echo "--- 连接器和CRTC信息 ---"
    sudo modetest -M nvidia-drm -c 2>/dev/null | head -20
    echo "--- Plane信息 ---"
    sudo modetest -M nvidia-drm -p 2>/dev/null | head -20
    echo "-------------------------"
else
    echo "⚠️  modetest工具不可用"
    echo "   安装命令: sudo apt install libdrm-tests"
fi

echo ""

# 检查显示管理器状态
echo "5. 显示管理器状态:"
if systemctl is-active --quiet gdm3; then
    echo "⚠️  gdm3正在运行"
elif systemctl is-active --quiet lightdm; then
    echo "⚠️  lightdm正在运行"
elif systemctl is-active --quiet sddm; then
    echo "⚠️  sddm正在运行"
else
    echo "✅ 无显示管理器运行（适合DRM直接访问）"
fi

echo ""

# 检查内存使用
echo "6. 系统资源检查:"
free_mem=$(free -m | grep 'Mem:' | awk '{printf "%.1f", ($7/$2)*100}')
echo "可用内存: ${free_mem}%"

if (( $(echo "$free_mem > 20" | bc -l) )); then
    echo "✅ 系统内存充足"
else
    echo "⚠️  系统内存不足"
fi

echo ""
echo "========================================="
echo "DRM状态检查完成"
echo "========================================="