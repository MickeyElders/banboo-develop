#!/bin/bash

# GTK4环境修复脚本

echo "=== GTK4环境修复脚本 ==="

# 1. 停止服务
echo "1. 停止现有服务..."
sudo systemctl stop bamboo-python-gtk4.service 2>/dev/null || true

# 2. 安装完整的GTK4依赖
echo "2. 安装完整GTK4依赖..."
sudo apt update
sudo apt install -y \
    python3-gi python3-gi-dev python3-gi-cairo \
    gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-glib-2.0 \
    libgtk-4-dev libadwaita-1-dev libgirepository1.0-dev \
    pkg-config build-essential meson ninja-build \
    libglib2.0-dev libgobject-introspection-1.0-dev

# 3. 重新创建虚拟环境
echo "3. 重新创建虚拟环境..."
if [ -d "/opt/bamboo-cut/venv" ]; then
    sudo rm -rf /opt/bamboo-cut/venv
fi

cd /opt/bamboo-cut
sudo -u $(whoami) python3 -m venv venv --system-site-packages

# 4. 测试虚拟环境
echo "4. 测试虚拟环境GTK4..."
/opt/bamboo-cut/venv/bin/python -c "
import gi
gi.require_version('Gtk', '4.0')  
from gi.repository import Gtk
print('GTK4测试成功，版本:', Gtk.get_major_version(), '.', Gtk.get_minor_version())
"

if [ $? -eq 0 ]; then
    echo "✓ GTK4环境修复成功"
else
    echo "✗ GTK4环境仍有问题，请查看错误信息"
    exit 1
fi

# 5. 重新安装Python依赖
echo "5. 重新安装Python依赖..."
/opt/bamboo-cut/venv/bin/pip install -r /home/$(whoami)/banboo-develop/requirements.txt

# 6. 设置权限
echo "6. 设置正确权限..."
sudo chown -R $(whoami):$(whoami) /opt/bamboo-cut/venv

# 7. 重新启动服务
echo "7. 重新启动服务..."
sudo systemctl daemon-reload
sudo systemctl start bamboo-python-gtk4.service

# 8. 检查服务状态
echo "8. 检查服务状态..."
sleep 3
if sudo systemctl is-active --quiet bamboo-python-gtk4.service; then
    echo "✓ 服务启动成功"
    sudo systemctl status bamboo-python-gtk4.service --no-pager
else
    echo "✗ 服务启动失败"
    echo "错误日志:"
    sudo journalctl -u bamboo-python-gtk4.service --no-pager -n 10
fi

echo "=== 修复完成 ==="