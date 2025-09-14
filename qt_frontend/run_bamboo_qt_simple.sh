#!/bin/bash

# =====================================================================
# 智能切竹机Qt前端简易启动脚本
# 快速启动脚本，适用于已配置好的Jetson环境
# =====================================================================

# 设置执行权限并切换到脚本目录
cd "$(dirname "${BASH_SOURCE[0]}")"

# 检查编译产物
if [ ! -f "build/bamboo_controller_qt" ]; then
    echo "❌ 可执行文件不存在，请先编译项目"
    echo "   cd build && cmake .. && make -j4"
    exit 1
fi

echo "🚀 启动智能切竹机Qt前端..."

# 设置基本环境变量
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
export QT_OPENGL=es2
export QT_QPA_EGLFS_ALWAYS_SET_MODE=1

# 设置设备权限（忽略错误）
sudo chmod 666 /dev/video* 2>/dev/null || true
sudo chmod 666 /dev/fb* 2>/dev/null || true

# 进入构建目录并运行
cd build

echo "🎯 使用EGLFS平台启动应用程序..."
echo "   平台: $QT_QPA_PLATFORM"
echo "   按 Ctrl+C 退出"
echo ""

# 启动应用程序
exec ./bamboo_controller_qt