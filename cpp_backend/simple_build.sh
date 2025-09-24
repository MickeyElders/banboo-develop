#!/bin/bash
# 简化的竹子切割系统编译脚本

echo "编译简化的竹子切割系统..."

# 编译命令
g++ -std=c++17 -O2 bamboo_system.cpp -o bamboo_system \
    `pkg-config --cflags --libs opencv4` \
    -pthread

if [ $? -eq 0 ]; then
    echo "编译成功！可执行文件: bamboo_system"
    echo ""
    echo "使用方法："
    echo "  直接运行: ./bamboo_system"
    echo "  SystemD: systemctl start bamboo-backend"
else
    echo "编译失败！请检查依赖项："
    echo "  sudo apt install libopencv-dev pkg-config build-essential"
fi