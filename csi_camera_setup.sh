#!/bin/bash
# IMX219 CSI摄像头安装和配置脚本
# 适用于Jetson Nano Super和IMX219模块

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== IMX219 CSI摄像头配置脚本 ===${NC}"

# 检查是否为Jetson设备
if [ ! -f "/proc/device-tree/model" ] || ! grep -q "Jetson" /proc/device-tree/model; then
    echo -e "${RED}❌ 此脚本仅适用于Jetson设备${NC}"
    exit 1
fi

JETSON_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
echo -e "${GREEN}检测到设备: ${JETSON_MODEL}${NC}"

# 1. 检查内核模块
echo -e "\n${BLUE}📋 检查摄像头相关内核模块...${NC}"

REQUIRED_MODULES=("imx219" "v4l2_common" "videobuf2_core" "videobuf2_v4l2" "videobuf2_dma_contig")
MISSING_MODULES=()

for module in "${REQUIRED_MODULES[@]}"; do
    if lsmod | grep -q "^$module"; then
        echo -e "${GREEN}✅ $module 模块已加载${NC}"
    else
        echo -e "${YELLOW}⚠️ $module 模块未加载${NC}"
        MISSING_MODULES+=("$module")
    fi
done

# 2. 检查设备树配置
echo -e "\n${BLUE}📋 检查设备树配置...${NC}"

if [ -d "/proc/device-tree/cam_i2cmux" ]; then
    echo -e "${GREEN}✅ 摄像头I2C复用器配置存在${NC}"
else
    echo -e "${YELLOW}⚠️ 摄像头I2C复用器配置可能缺失${NC}"
fi

# 检查CSI端口配置
CSI_PORTS=("/proc/device-tree/host1x/nvcsi@15a00000/channel@0" "/proc/device-tree/host1x/nvcsi@15a00000/channel@1")
for port in "${CSI_PORTS[@]}"; do
    if [ -d "$port" ]; then
        echo -e "${GREEN}✅ CSI端口配置存在: $(basename $port)${NC}"
    fi
done

# 3. 检查CSI摄像头连接
echo -e "\n${BLUE}📋 检查CSI摄像头连接...${NC}"

# 检查I2C设备
echo "🔍 扫描I2C总线上的IMX219设备..."
I2C_BUSES=(0 1 2 6 7 8 9 10)  # Jetson Nano常见的I2C总线

IMX219_FOUND=false
for bus in "${I2C_BUSES[@]}"; do
    if i2cdetect -y -r $bus 2>/dev/null | grep -q "10"; then
        echo -e "${GREEN}✅ 在I2C总线 $bus 上检测到IMX219设备 (地址: 0x10)${NC}"
        IMX219_FOUND=true
        IMX219_I2C_BUS=$bus
        break
    fi
done

if [ "$IMX219_FOUND" = false ]; then
    echo -e "${RED}❌ 未在任何I2C总线上检测到IMX219设备${NC}"
    echo -e "${YELLOW}💡 请检查：${NC}"
    echo "   1. 摄像头模块是否正确连接到CSI端口"
    echo "   2. 排线是否插紧，金手指方向是否正确"
    echo "   3. 摄像头模块是否损坏"
fi

# 4. 尝试加载缺失的模块
if [ ${#MISSING_MODULES[@]} -gt 0 ]; then
    echo -e "\n${BLUE}🔧 尝试加载缺失的内核模块...${NC}"
    for module in "${MISSING_MODULES[@]}"; do
        echo "加载模块: $module"
        if modprobe "$module" 2>/dev/null; then
            echo -e "${GREEN}✅ $module 模块加载成功${NC}"
        else
            echo -e "${YELLOW}⚠️ 无法加载 $module 模块${NC}"
        fi
    done
fi

# 5. 创建摄像头设备节点
echo -e "\n${BLUE}🔧 配置摄像头设备节点...${NC}"

# 检查media设备
MEDIA_DEVICES=$(find /dev -name "media*" 2>/dev/null)
if [ -n "$MEDIA_DEVICES" ]; then
    echo -e "${GREEN}✅ 找到media设备:${NC}"
    for device in $MEDIA_DEVICES; do
        echo "   $device"
        if command -v media-ctl >/dev/null 2>&1; then
            echo "     拓扑: $(media-ctl -d $device -p 2>/dev/null | grep -E "entity|source|sink" | head -3 || echo "无法获取")"
        fi
    done
else
    echo -e "${YELLOW}⚠️ 未找到media设备${NC}"
fi

# 6. 配置GStreamer管道
echo -e "\n${BLUE}🔧 配置GStreamer管道...${NC}"

# 创建测试脚本
cat > /tmp/test_imx219.sh << 'EOF'
#!/bin/bash
echo "🔍 测试IMX219摄像头..."

# 测试不同的GStreamer管道
PIPELINES=(
    "nvarguscamerasrc sensor-id=0 ! nvvidconv ! xvimagesink"
    "nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink"
    "nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! nvvidconv ! xvimagesink"
)

for i in "${!PIPELINES[@]}"; do
    echo "测试管道 $((i+1)): ${PIPELINES[$i]}"
    timeout 5 gst-launch-1.0 ${PIPELINES[$i]} 2>/dev/null && echo "✅ 管道 $((i+1)) 工作正常" || echo "❌ 管道 $((i+1)) 失败"
done
EOF

chmod +x /tmp/test_imx219.sh

# 7. 创建udev规则
echo -e "\n${BLUE}🔧 创建udev规则...${NC}"

sudo tee /etc/udev/rules.d/99-jetson-camera.rules > /dev/null << 'EOF'
# Jetson CSI摄像头udev规则
KERNEL=="video*", ATTRS{name}=="vi-output*", GROUP="video", MODE="0664"
KERNEL=="media*", GROUP="video", MODE="0664"
SUBSYSTEM=="media", GROUP="video", MODE="0664"
EOF

# 8. 配置系统服务
echo -e "\n${BLUE}🔧 配置系统服务...${NC}"

# 创建摄像头初始化服务
sudo tee /etc/systemd/system/jetson-camera-init.service > /dev/null << 'EOF'
[Unit]
Description=Jetson CSI Camera Initialization
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'modprobe imx219 && sleep 2'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jetson-camera-init

# 9. 重新加载udev规则
echo -e "\n${BLUE}🔧 重新加载udev规则...${NC}"
sudo udevadm control --reload-rules
sudo udevadm trigger

# 10. 最终检查
echo -e "\n${BLUE}📋 最终检查...${NC}"

sleep 3  # 等待设备初始化

# 检查video设备
VIDEO_DEVICES=$(find /dev -name "video*" 2>/dev/null)
if [ -n "$VIDEO_DEVICES" ]; then
    echo -e "${GREEN}✅ 找到video设备:${NC}"
    for device in $VIDEO_DEVICES; do
        echo "   $device"
        if command -v v4l2-ctl >/dev/null 2>&1; then
            v4l2-ctl --device="$device" --info 2>/dev/null | head -3 || echo "     无法获取设备信息"
        fi
    done
else
    echo -e "${RED}❌ 仍未找到video设备${NC}"
fi

# 提供解决方案
echo -e "\n${BLUE}💡 下一步操作:${NC}"
echo "1. 重启系统以确保所有更改生效："
echo "   sudo reboot"
echo ""
echo "2. 重启后测试摄像头："
echo "   /tmp/test_imx219.sh"
echo ""
echo "3. 检查设备："
echo "   ls -la /dev/video*"
echo "   dmesg | grep imx219"
echo ""
echo "4. 如果仍有问题，请检查："
echo "   - 摄像头排线连接"
echo "   - JetPack版本兼容性"
echo "   - 设备树配置"

echo -e "\n${GREEN}🎉 CSI摄像头配置完成！${NC}"
echo -e "${YELLOW}⚠️ 请重启系统后测试摄像头功能${NC}"