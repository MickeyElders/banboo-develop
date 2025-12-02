#!/bin/bash
set -e

OUT_DIR="/opt/bamboo-qt/www"
SRC_URL="rtsp://127.0.0.1:8554/deepstream"
PLAYLIST_ROOT=""

mkdir -p "${OUT_DIR}"

exec gst-launch-1.0 -e -v \
  rtspsrc location="${SRC_URL}" latency=200 protocols=tcp do-retransmission=false ! \
  rtph264depay ! queue ! h264parse ! mpegtsmux ! \
  hlssink max-files=4 target-duration=2 \
    playlist-location="${OUT_DIR}/stream.m3u8" \
    location="${OUT_DIR}/segment_%05d.ts"
