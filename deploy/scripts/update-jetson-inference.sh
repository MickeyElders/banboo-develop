#!/bin/bash
set -euo pipefail
PREFIX="/opt/bamboo-vision"
JI_DIR="$PREFIX/jetson-inference"

sudo mkdir -p "$PREFIX"

if [ -d "$JI_DIR/.git" ]; then
  echo "Updating jetson-inference in $JI_DIR (master)..."
  sudo git -C "$JI_DIR" fetch origin master --depth=1
  sudo git -C "$JI_DIR" reset --hard origin/master
  sudo git -C "$JI_DIR" submodule update --init --recursive --depth=1
else
  echo "Cloning jetson-inference into $JI_DIR ..."
  sudo rm -rf "$JI_DIR"
  sudo git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference "$JI_DIR"
fi

echo "jetson-inference source ready at $JI_DIR"
