#!/bin/bash

echo "=== Jetson Orin NX 触摸设备测试脚本 ==="
echo

# 检查输入设备
echo "1. 检查输入设备..."
ls -la /dev/input/

echo
echo "2. 检查触摸设备信息..."

# 优先检查 event2
for event_dev in event2 event1 event0 event3 event4; do
    if [ -e "/dev/input/$event_dev" ]; then
        echo "--- /dev/input/$event_dev ---"
        
        # 获取设备名称
        if command -v evtest >/dev/null 2>&1; then
            timeout 2 evtest /dev/input/$event_dev 2>/dev/null | head -5
        else
            echo "未安装 evtest，使用基本检查"
            file /dev/input/$event_dev
        fi
        
        # 检查设备权限
        if [ -r "/dev/input/$event_dev" ]; then
            echo "✓ 设备可读"
        else
            echo "✗ 设备不可读，需要权限"
        fi
        
        echo
    fi
done

echo "3. 检查GPU状态..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q | grep -E "(GPU|Temperature|Power)"
else
    echo "NVIDIA驱动未安装"
fi

echo
echo "4. 检查CPU状态..."
cat /proc/cpuinfo | grep "model name" | head -1
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo
echo "5. 检查framebuffer..."
if [ -e "/dev/fb0" ]; then
    echo "✓ Framebuffer 设备存在"
    if command -v fbset >/dev/null 2>&1; then
        fbset | grep -E "(mode|geometry)"
    fi
else
    echo "✗ Framebuffer 设备不存在"
fi

echo
echo "6. 系统优化建议..."
echo "运行以下命令优化触摸性能:"
echo "sudo chmod 666 /dev/input/event*"
echo "sudo echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

echo
echo "=== 测试完成 ==="