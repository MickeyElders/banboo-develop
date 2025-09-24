#!/bin/bash

echo "=== 智能切竹机视频流测试脚本 ==="
echo ""

# 检查GStreamer是否安装
echo "🔍 检查GStreamer安装..."
if ! command -v gst-launch-1.0 &> /dev/null; then
    echo "❌ GStreamer未安装"
    exit 1
fi
echo "✅ GStreamer已安装"

# 检查端口5000是否被占用
echo ""
echo "🔍 检查UDP端口5000状态..."
if netstat -u -l -n 2>/dev/null | grep -q ":5000 "; then
    echo "✅ 端口5000有UDP监听"
else
    echo "⚠️  端口5000无UDP监听（后端可能未运行）"
fi

# 测试接收视频流
echo ""
echo "🎥 测试接收视频流（10秒）..."
echo "如果看到彩色窗口，说明视频流正常"

timeout 10 gst-launch-1.0 \
    udpsrc uri=udp://127.0.0.1:5000 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay \
    ! avdec_h264 \
    ! videoconvert \
    ! autovideosink \
    2>&1 | head -20

echo ""
echo "🔍 检查网络流量..."
echo "预期：每秒应该有30个数据包，每个约几KB"
echo "实际UDP流量："

# 监控UDP流量10秒
timeout 10 tcpdump -i lo -c 50 udp port 5000 2>/dev/null || echo "需要root权限运行tcpdump"

echo ""
echo "=== 测试完成 ==="
echo "如果看到了视频窗口，说明前后端视频流传输正常！"