#!/bin/bash

echo "=== Jetson 摄像头问题修复脚本 ==="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 检查当前运行的相关进程
print_status "检查运行中的摄像头相关进程..."
CAMERA_PROCS=$(pgrep -f "nvargus|gst-launch|argus_daemon" 2>/dev/null)
if [ ! -z "$CAMERA_PROCS" ]; then
    print_warning "发现运行中的摄像头进程:"
    ps aux | grep -E "nvargus|gst-launch|argus_daemon" | grep -v grep
    
    read -p "是否终止这些进程? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "终止冲突进程..."
        sudo pkill -f nvargus-daemon 2>/dev/null || true
        sudo pkill -f gst-launch-1.0 2>/dev/null || true
        sudo pkill -f argus_daemon 2>/dev/null || true
        sleep 2
    fi
fi

# 2. 检查设备文件
print_status "检查摄像头设备文件..."
DEVICE_ISSUES=0

for device in /dev/video* /dev/nvhost-vi /dev/nvhost-isp /dev/nvhost-nvcsi; do
    if [ -e "$device" ]; then
        PERMS=$(stat -c "%a" "$device" 2>/dev/null)
        if [ "$PERMS" != "666" ] && [ "$PERMS" != "664" ]; then
            print_warning "$device 权限为 $PERMS，建议设置为666"
            DEVICE_ISSUES=1
        fi
    else
        print_error "$device 不存在"
        DEVICE_ISSUES=1
    fi
done

if [ $DEVICE_ISSUES -eq 1 ]; then
    read -p "是否修复设备权限? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "修复设备权限..."
        sudo chmod 666 /dev/video* 2>/dev/null || true
        sudo chmod 666 /dev/nvhost-* 2>/dev/null || true
    fi
fi

# 3. 检查用户组
print_status "检查用户组..."
CURRENT_USER=$(whoami)
USER_GROUPS=$(groups $CURRENT_USER)

GROUP_ISSUES=0
for group in video camera; do
    if ! echo "$USER_GROUPS" | grep -q "\b$group\b"; then
        print_warning "用户 $CURRENT_USER 不在 $group 组中"
        GROUP_ISSUES=1
    fi
done

if [ $GROUP_ISSUES -eq 1 ]; then
    read -p "是否添加用户到相关组? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "添加用户到相关组..."
        sudo usermod -a -G video $CURRENT_USER
        sudo usermod -a -G camera $CURRENT_USER 2>/dev/null || true
        print_warning "需要重新登录以生效组更改"
    fi
fi

# 4. 检查并重启 nvargus-daemon
print_status "检查 nvargus-daemon 服务..."
if systemctl is-active --quiet nvargus-daemon; then
    print_status "nvargus-daemon 正在运行，重启以清理状态..."
    sudo systemctl restart nvargus-daemon
    sleep 3
else
    print_warning "nvargus-daemon 未运行，尝试启动..."
    sudo systemctl start nvargus-daemon
    sleep 2
    
    if ! systemctl is-active --quiet nvargus-daemon; then
        print_error "nvargus-daemon 启动失败"
        print_status "检查服务状态:"
        systemctl status nvargus-daemon --no-pager -l
    fi
fi

# 5. 设置环境变量
print_status "设置EGL环境变量..."
export EGL_PLATFORM=drm
export __EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl

# 检查环境变量文件
ENV_FILE="$HOME/.camera_env"
cat > "$ENV_FILE" << 'EOF'
# Jetson 摄像头环境变量
export EGL_PLATFORM=drm
export __EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl
EOF

if ! grep -q "source.*\.camera_env" ~/.bashrc; then
    echo "source $ENV_FILE" >> ~/.bashrc
    print_status "已添加环境变量到 ~/.bashrc"
fi

# 6. 测试摄像头访问
print_status "测试摄像头访问..."
for sensor_id in 0 1; do
    print_status "测试 sensor-id=$sensor_id..."
    
    timeout 10 gst-launch-1.0 nvarguscamerasrc sensor-id=$sensor_id bufapi-version=1 ! \
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=15/1" ! \
        fakesink 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_status "sensor-id=$sensor_id 测试成功!"
        WORKING_SENSOR=$sensor_id
        break
    else
        print_warning "sensor-id=$sensor_id 测试失败"
    fi
done

# 7. 生成诊断报告
print_status "生成诊断报告..."
REPORT_FILE="/tmp/camera_diagnostic_report.txt"
cat > "$REPORT_FILE" << EOF
=== Jetson 摄像头诊断报告 ===
时间: $(date)
用户: $(whoami)
系统: $(uname -a)

=== 设备检查 ===
$(ls -la /dev/video* 2>/dev/null || echo "未找到 /dev/video* 设备")

$(ls -la /dev/nvhost-* 2>/dev/null || echo "未找到 nvhost 设备")

=== 服务状态 ===
$(systemctl status nvargus-daemon --no-pager -l 2>/dev/null || echo "nvargus-daemon 服务不可用")

=== 用户组 ===
$(groups)

=== 环境变量 ===
EGL_PLATFORM=$EGL_PLATFORM
__EGL_VENDOR_LIBRARY_DIRS=$__EGL_VENDOR_LIBRARY_DIRS

=== 运行进程 ===
$(ps aux | grep -E "nvargus|gst-launch|argus" | grep -v grep || echo "未找到相关进程")

=== 测试结果 ===
可用的sensor-id: ${WORKING_SENSOR:-"未找到"}
EOF

print_status "诊断报告已保存到: $REPORT_FILE"

# 8. 最终建议
echo
print_status "=== 修复完成 ==="
echo -e "1. 运行 ${GREEN}source ~/.camera_env${NC} 加载环境变量"
echo -e "2. 如果修改了用户组，请${YELLOW}重新登录${NC}系统"
if [ ! -z "$WORKING_SENSOR" ]; then
    echo -e "3. 使用 ${GREEN}sensor-id=$WORKING_SENSOR${NC} 作为摄像头参数"
else
    echo -e "3. ${RED}未检测到可用摄像头${NC}，请检查硬件连接"
fi
echo -e "4. 查看完整诊断报告: ${GREEN}cat $REPORT_FILE${NC}"

echo
print_status "建议的测试命令:"
echo "gst-launch-1.0 nvarguscamerasrc sensor-id=${WORKING_SENSOR:-0} bufapi-version=1 ! 'video/x-raw(memory:NVMM), width=640, height=480, framerate=15/1' ! nvvidconv ! xvimagesink"