#!/bin/bash

# GTK4应用 Cage启动脚本
# 在无X11环境下使用Cage compositor运行GTK4应用

set -e

# 配置变量
USER_ID=${1:-$(id -u)}
APP_PATH=${2:-"/opt/bamboo-cut/venv/bin/python /opt/bamboo-cut/python_core/ai_bamboo_system.py"}
RUNTIME_DIR="/run/user/$USER_ID"

echo "=== 启动GTK4应用 (Cage Compositor) ==="
echo "用户ID: $USER_ID"
echo "应用路径: $APP_PATH"
echo "运行时目录: $RUNTIME_DIR"

# 创建用户运行时目录
echo "创建运行时目录..."
sudo mkdir -p "$RUNTIME_DIR"
sudo chown "$USER_ID:$USER_ID" "$RUNTIME_DIR"
sudo chmod 700 "$RUNTIME_DIR"

# 设置环境变量
export XDG_RUNTIME_DIR="$RUNTIME_DIR"
export WAYLAND_DISPLAY=wayland-0
export WLR_BACKENDS=headless
export WLR_RENDERER=gles2
export GTK_THEME=Adwaita:dark
export NO_AT_BRIDGE=1
export GI_TYPELIB_PATH="/usr/lib/aarch64-linux-gnu/girepository-1.0:/usr/lib/girepository-1.0"

# 检查Cage是否可用
if ! which cage >/dev/null 2>&1; then
    echo "错误: Cage未安装"
    echo "请运行: sudo apt install cage"
    echo "或者: make install-cage-from-source"
    exit 1
fi

# 检查应用是否存在
if [[ "$APP_PATH" == *"python"* ]]; then
    PYTHON_PATH=$(echo "$APP_PATH" | cut -d' ' -f1)
    SCRIPT_PATH=$(echo "$APP_PATH" | cut -d' ' -f2-)
    
    if [ ! -f "$PYTHON_PATH" ]; then
        echo "错误: Python解释器不存在: $PYTHON_PATH"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "错误: Python脚本不存在: $SCRIPT_PATH"
        exit 1
    fi
fi

echo "启动Cage compositor..."

# 启动Cage和GTK4应用
# Cage将创建一个全屏的Wayland会话
exec cage -- $APP_PATH