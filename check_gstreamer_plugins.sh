#!/bin/bash

echo "=== GStreamer插件检测和安装脚本 ==="
echo ""

# 检查GStreamer版本
echo "🔍 检查GStreamer版本..."
gst-launch-1.0 --version 2>/dev/null || echo "❌ GStreamer未安装"

# 检查基础插件
echo ""
echo "🔍 检查基础插件..."
gst-inspect-1.0 appsrc >/dev/null 2>&1 && echo "✅ appsrc" || echo "❌ appsrc"
gst-inspect-1.0 videoconvert >/dev/null 2>&1 && echo "✅ videoconvert" || echo "❌ videoconvert"
gst-inspect-1.0 videoscale >/dev/null 2>&1 && echo "✅ videoscale" || echo "❌ videoscale"
gst-inspect-1.0 udpsink >/dev/null 2>&1 && echo "✅ udpsink" || echo "❌ udpsink"

# 检查编码器
echo ""
echo "🔍 检查H.264编码器..."
gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 && echo "✅ nvv4l2h264enc (NVIDIA V4L2)" || echo "❌ nvv4l2h264enc"
gst-inspect-1.0 omxh264enc >/dev/null 2>&1 && echo "✅ omxh264enc (OpenMAX)" || echo "❌ omxh264enc"
gst-inspect-1.0 nvh264enc >/dev/null 2>&1 && echo "✅ nvh264enc (NVIDIA)" || echo "❌ nvh264enc"
gst-inspect-1.0 x264enc >/dev/null 2>&1 && echo "✅ x264enc (软件)" || echo "❌ x264enc"
gst-inspect-1.0 avenc_h264 >/dev/null 2>&1 && echo "✅ avenc_h264 (FFmpeg)" || echo "❌ avenc_h264"

# 检查JPEG编码器（备用方案）
echo ""
echo "🔍 检查JPEG编码器..."
gst-inspect-1.0 jpegenc >/dev/null 2>&1 && echo "✅ jpegenc" || echo "❌ jpegenc"
gst-inspect-1.0 rtpjpegpay >/dev/null 2>&1 && echo "✅ rtpjpegpay" || echo "❌ rtpjpegpay"

# 检查RTP负载器
echo ""
echo "🔍 检查RTP负载器..."
gst-inspect-1.0 rtph264pay >/dev/null 2>&1 && echo "✅ rtph264pay" || echo "❌ rtph264pay"
gst-inspect-1.0 h264parse >/dev/null 2>&1 && echo "✅ h264parse" || echo "❌ h264parse"

# 建议安装命令
echo ""
echo "💡 如果有插件缺失，请运行以下命令安装："
echo ""
echo "# 基础插件包"
echo "sudo apt update"
echo "sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base"
echo "sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
echo "sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-libav"
echo ""
echo "# Jetson专用插件"
echo "sudo apt install -y nvidia-l4t-gstreamer"
echo "sudo apt install -y gstreamer1.0-omx gstreamer1.0-omx-generic"
echo ""
echo "# 重启后检查"
echo "sudo reboot"

echo ""
echo "🎥 测试基础视频管道..."
timeout 5 gst-launch-1.0 videotestsrc ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5001 2>/dev/null && echo "✅ 基础管道工作正常" || echo "❌ 基础管道失败"