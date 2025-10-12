#!/bin/bash

echo "======================================="
echo "DRM资源协调器编译测试"
echo "======================================="

cd /home/lip/banboo-develop

echo ""
echo "步骤1: 清理之前的构建..."
rm -rf build
mkdir -p build
cd build

echo ""
echo "步骤2: 配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败"
    exit 1
fi

echo ""
echo "步骤3: 编译测试..."
make -j4 2>&1 | tee /tmp/build.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 编译成功！"
    echo ""
    echo "检查生成的可执行文件..."
    ls -la ./bamboo_integrated
    echo ""
    echo "编译日志已保存到: /tmp/build.log"
else
    echo ""
    echo "❌ 编译失败"
    echo ""
    echo "显示最后的错误信息..."
    tail -30 /tmp/build.log
    echo ""
    echo "完整编译日志: /tmp/build.log"
    exit 1
fi

echo ""
echo "======================================="
echo "编译测试完成"
echo "======================================="