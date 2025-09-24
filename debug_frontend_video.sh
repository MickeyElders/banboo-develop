#!/bin/bash

echo "=== 前端视频接收调试脚本 ==="
echo ""

# 设置调试环境
export GST_DEBUG=3
export GST_DEBUG_NO_COLOR=1

echo "🔍 检查前端是否运行..."
if pgrep -f "bamboo.*frontend" > /dev/null; then
    echo "✅ 前端进程已运行"
    echo "进程信息："
    ps aux | grep -E "(bamboo|frontend)" | grep -v grep
else
    echo "⚠️  前端进程未运行"
fi

echo ""
echo "🔍 检查后端是否运行..."
if pgrep -f "bamboo.*backend" > /dev/null; then
    echo "✅ 后端进程已运行"
    echo "进程信息："
    ps aux | grep -E "(bamboo|backend)" | grep -v grep
else
    echo "⚠️  后端进程未运行"
fi

echo ""
echo "🔍 检查TCP连接状态..."
echo "前端到后端的TCP连接 (127.0.0.1:8888):"
netstat -t -n | grep ":8888" || echo "无TCP连接"

echo ""
echo "🔍 检查UDP流状态..."
echo "UDP视频流 (127.0.0.1:5000):"
netstat -u -n | grep ":5000" || echo "无UDP监听"

echo ""
echo "🎥 测试前端GStreamer接收能力..."
echo "模拟前端接收管道（5秒测试）："

timeout 5 gst-launch-1.0 \
    udpsrc uri=udp://127.0.0.1:5000 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay \
    ! avdec_h264 \
    ! videoconvert \
    ! video/x-raw,format=BGR \
    ! fakesink dump=true \
    2>&1 | tail -10

echo ""
echo "=== 调试完成 ==="
echo ""
echo "💡 调试建议："
echo "1. 确保后端和前端都在运行"
echo "2. 检查TCP连接是否建立 (8888端口)"
echo "3. 检查UDP流是否发送 (5000端口)"
echo "4. 查看系统日志了解详细错误信息"