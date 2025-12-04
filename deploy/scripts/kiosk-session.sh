#!/usr/bin/env bash
set -e

# Lightweight kiosk session: disable power saving, start openbox, then launch Chromium fullscreen.
URL="${KIOSK_URL:-http://localhost:8080/bamboo.html}"
CHROME_FLAGS="--noerrdialogs --disable-infobars --kiosk --start-fullscreen --incognito --autoplay-policy=no-user-gesture-required --disable-features=TranslateUI"

# Power saving off for HDMI panels
xset -dpms
xset s off
xset s noblank

openbox-session &
sleep 1

chromium-browser $CHROME_FLAGS "$URL"
