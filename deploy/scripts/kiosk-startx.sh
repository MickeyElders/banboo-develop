#!/usr/bin/env bash
set -e

SESSION_SCRIPT="${SESSION_SCRIPT:-/opt/bamboo-vision/bin/kiosk-session.sh}"
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

# Ensure XDG runtime dir exists for the user
if [ -z "${XDG_RUNTIME_DIR}" ]; then
  XDG_RUNTIME_DIR="/run/user/$(id -u)"
  export XDG_RUNTIME_DIR
fi
mkdir -p "${XDG_RUNTIME_DIR}"

# Launch a minimal X server with the kiosk session
exec startx "${SESSION_SCRIPT}" -- -nocursor
