#!/bin/bash
set -euo pipefail

# Default RTSP input (from main app)
RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/deepstream}"
# Default listen port for websocket/HTTP (DeepStream webrtc demo). Override via WEBRTC_PORT.
WEBRTC_PORT="${WEBRTC_PORT:-8080}"
# Path to DeepStream WebRTC demo binary
WEBRTC_BIN="${WEBRTC_BIN:-/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-webrtc-demo/build/deepstream-webrtc-app}"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra"
export GST_PLUGIN_SYSTEM_PATH="${GST_PLUGIN_PATH}"
export GST_PLUGIN_SCANNER="/usr/lib/aarch64-linux-gnu/gstreamer-1.0/gst-plugin-scanner"

if [[ ! -x "${WEBRTC_BIN}" ]]; then
  echo "[webrtc] Binary not found: ${WEBRTC_BIN}"
  echo "[webrtc] Please build DeepStream webrtc demo (dswebrtc) first:"
  echo "        cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-webrtc-demo && \\"
  echo "        mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
  exit 1
fi

echo "[webrtc] starting WebRTC gateway: ${WEBRTC_BIN} -i ${RTSP_URL} -p ${WEBRTC_PORT}"
exec "${WEBRTC_BIN}" -i "${RTSP_URL}" -p "${WEBRTC_PORT}"
