#!/bin/bash
# 生成xdg-shell协议头文件

set -e

echo "正在生成xdg-shell协议头文件..."

# 检查wayland-scanner是否存在
if ! command -v wayland-scanner &> /dev/null; then
    echo "错误: wayland-scanner未找到"
    echo "请安装: sudo apt install wayland-protocols libwayland-dev"
    exit 1
fi

# 查找xdg-shell.xml协议文件
PROTOCOL_DIR="/usr/share/wayland-protocols"
XDG_SHELL_XML="$PROTOCOL_DIR/stable/xdg-shell/xdg-shell.xml"

if [ ! -f "$XDG_SHELL_XML" ]; then
    echo "错误: xdg-shell.xml协议文件未找到"
    echo "预期位置: $XDG_SHELL_XML"
    echo "请安装: sudo apt install wayland-protocols"
    exit 1
fi

# 创建协议头文件目录
mkdir -p cpp_backend/include/wayland-protocols

# 生成客户端头文件
echo "生成xdg-shell客户端头文件..."
wayland-scanner client-header "$XDG_SHELL_XML" \
    cpp_backend/include/wayland-protocols/xdg-shell-client-protocol.h

# 生成私有代码文件
echo "生成xdg-shell私有代码文件..."
wayland-scanner private-code "$XDG_SHELL_XML" \
    cpp_backend/src/wayland-protocols/xdg-shell-protocol.c

# 创建源文件目录
mkdir -p cpp_backend/src/wayland-protocols

echo "✅ xdg-shell协议文件生成完成"
echo "   头文件: cpp_backend/include/wayland-protocols/xdg-shell-client-protocol.h"
echo "   源文件: cpp_backend/src/wayland-protocols/xdg-shell-protocol.c"