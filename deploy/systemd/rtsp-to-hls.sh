#!/bin/bash
# Simple RTSP -> HLS gateway using GStreamer + python http.server
# Input: RTSP_URL (default rtsp://127.0.0.1:8554/deepstream)
# Output: HLS served on HTTP_PORT (default 8080) from /opt/bamboo-qt/hls

set -euo pipefail

RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/deepstream}"
HTTP_PORT="${HTTP_PORT:-8080}"
HLS_DIR="${HLS_DIR:-/opt/bamboo-qt/hls}"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra"
export GST_PLUGIN_SYSTEM_PATH="${GST_PLUGIN_PATH}"
export GST_PLUGIN_SCANNER="/usr/lib/aarch64-linux-gnu/gstreamer-1.0/gst-plugin-scanner"

mkdir -p "${HLS_DIR}"

cleanup() {
    if [[ -n "${GST_PID:-}" ]]; then
        kill "${GST_PID}" 2>/dev/null || true
        wait "${GST_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[hls] starting RTSP->HLS: ${RTSP_URL} -> ${HLS_DIR} (http ${HTTP_PORT})"
gst-launch-1.0 -e \
  rtspsrc location="${RTSP_URL}" latency=200 protocols=tcp ! \
  rtph264depay ! h264parse ! \
  mpegtsmux ! \
  hlssink max-files=5 target-duration=2 \
    playlist-location="${HLS_DIR}/stream.m3u8" \
    location="${HLS_DIR}/segment%05d.ts" \
    playlist-root="http://localhost:${HTTP_PORT}" \
    playlist-length=5 \
  >/var/log/bamboo-hls-gst.log 2>&1 &
GST_PID=$!

echo "[hls] gst-launch pid=${GST_PID}, starting http server on ${HTTP_PORT}"
cd "${HLS_DIR}"
exec python3 -m http.server "${HTTP_PORT}"
