#!/bin/bash
# 智能切竹机 - 一键Kiosk模式部署脚本

set -e

echo "========================================"
echo "智能切竹机 Kiosk模式一键部署"
echo "========================================"

# 检查权限
if [ "$EUID" -ne 0 ]; then
    echo "请使用root权限运行: sudo $0"
    exit 1
fi

# 确认部署
read -p "确认部署Kiosk模式？这将修改系统配置 (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "部署已取消"
    exit 0
fi

echo "开始部署..."

# 1. 配置Kiosk自启动
echo "[1/3] 配置Kiosk自启动..."
./scripts/create_kiosk_startup.sh

# 2. 配置自定义启动动画
echo "[2/3] 配置自定义启动动画..."
./scripts/setup_custom_splash.sh

# 3. 设置执行权限
echo "[3/3] 设置权限..."
chmod +x src/gui/touch_interface.py
chmod +x scripts/*.sh

echo "========================================"
echo "部署完成！"
echo "========================================"
echo
echo "配置内容："
echo "✓ 自动登录已启用"
echo "✓ 触摸界面开机自启动"
echo "✓ 自定义启动动画已应用"
echo "✓ 系统性能已优化"
echo
echo "重启系统以查看效果："
echo "sudo reboot"
echo
echo "如需卸载，运行："
echo "./scripts/uninstall_kiosk.sh"
echo "./scripts/uninstall_splash.sh" 