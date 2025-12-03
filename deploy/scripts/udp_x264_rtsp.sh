#!/usr/bin/env bash
# Soft-encode raw UDP frames to RTSP using x264 (no NVENC required).
# Env vars: UDP_PORT (default 5600), WIDTH (1280), HEIGHT (720), FPS (30), RTSP_URL (rtsp://127.0.0.1:8554/live)
set -e
UDP_PORT=${UDP_PORT:-5600}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
FPS=${FPS:-30}
RTSP_URL=${RTSP_URL:-rtsp://127.0.0.1:8554/live}

gst-launch-1.0 -v \
  udpsrc port=${UDP_PORT} caps="video/x-raw,format=I420,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" ! \
  queue ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast key-int-max=30 ! \
  rtph264pay config-interval=1 pt=96 ! \
  rtspclientsink location=${RTSP_URL}
