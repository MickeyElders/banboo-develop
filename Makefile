SHELL := /bin/bash

PY ?= python3
PIP ?= $(PY) -m pip
PREFIX ?= /opt/bamboo-vision
SERVICE ?= bamboo-vision.service
JETSON_PY ?= /usr/local/python
# Default TensorRT CLI path (JetPack install). Override with TRTEXEC=/path/to/trtexec if different.
TRTEXEC ?= /usr/src/tensorrt/bin/trtexec
TRT_WORKSPACE := 2048  # MiB workspace for TensorRT engine build

.PHONY: deps check-jetson run install service service-restart service-stop service-status logs clean-install redeploy

deps:
	$(PIP) install -r requirements.txt
	$(MAKE) check-jetson
	$(MAKE) check-gst
	$(MAKE) build-engine

check-jetson:
	@if PYTHONPATH="$(JETSON_PY):$$PYTHONPATH" $(PY) -c "import jetson.inference, jetson.utils" >/dev/null 2>&1; then \
		echo "jetson-inference bindings OK"; \
	else \
		echo "jetson-inference Python bindings not found; attempting source install..." ; \
		$(MAKE) install-jetson; \
	fi

install-jetson:
	@set -e; \
	missing=""; \
	for pkg in python3-dev python3-numpy; do \
		dpkg -s $$pkg >/dev/null 2>&1 || missing="$$missing $$pkg"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "Installing missing build deps:$$missing"; \
		sudo apt-get update && sudo apt-get install -y $$missing; \
	else \
		echo "Build deps OK (python3-dev/python3-numpy)"; \
	fi; \
	$(PY) -m pip install --upgrade "numpy<2"; \
	NUMPY_INC=$$(python3 -c "import numpy, os; print(numpy.get_include())" 2>/dev/null || true); \
	NPYMATH=$$(find /usr -name libnpymath.a -print -quit 2>/dev/null || true); \
	if [ -z "$$NUMPY_INC" ]; then echo "numpy not found, please install python3-numpy"; exit 1; fi; \
	if [ -z "$$NPYMATH" ]; then echo "libnpymath.a not found; try reinstalling python3-numpy"; exit 1; fi; \
	NUMPY_LIBDIR=$$(dirname "$$NPYMATH"); \
	for dst in /usr/lib/libnpymath.a /usr/lib/aarch64-linux-gnu/libnpymath.a; do \
		if [ ! -f $$dst ]; then sudo ln -sf "$$NPYMATH" $$dst; fi; \
	done; \
	tmpdir=$$(mktemp -d); \
	echo "Cloning jetson-inference into $$tmpdir"; \
	cd $$tmpdir; \
	git clone --recursive https://github.com/dusty-nv/jetson-inference.git; \
	cd jetson-inference; \
	mkdir -p build && cd build; \
	cmake .. -DENABLE_PYTHON=ON -DNUMPY_INCLUDE_DIRS="$$NUMPY_INC" -DNUMPY_LIBRARIES="$$NPYMATH"; \
	make -j$$(nproc); \
	sudo make install; \
	sudo ldconfig; \
	rm -rf "$$tmpdir"; \
	echo "jetson-inference install completed"

check-gst:
	@set -e; \
	export GST_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra"; \
	export GST_PLUGIN_SCANNER="/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner"; \
	export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:$$LD_LIBRARY_PATH"; \
	if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then \
		echo "GStreamer: nvv4l2h264enc OK"; \
	else \
		echo "Installing GStreamer encoder deps (nvv4l2h264enc/x264enc)..."; \
		miss=""; \
		for pkg in nvidia-l4t-gstreamer nvidia-l4t-multimedia nvidia-l4t-multimedia-utils nvidia-l4t-jetson-multimedia-api gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools; do \
			dpkg -s $$pkg >/dev/null 2>&1 || miss="$$miss $$pkg"; \
		done; \
		if [ -n "$$miss" ]; then sudo apt-get update && sudo apt-get install -y $$miss; fi; \
		if [ -f /usr/lib/aarch64-linux-gnu/tegra/libnvv4l2h264enc.so ] && [ ! -f /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libnvv4l2h264enc.so ]; then \
			echo "Linking libnvv4l2h264enc.so into gstreamer-1.0 path"; \
			sudo ln -sf /usr/lib/aarch64-linux-gnu/tegra/libnvv4l2h264enc.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libnvv4l2h264enc.so; \
		fi; \
		if ! gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then \
			echo "Warning: nvv4l2h264enc still missing after install; RTSP will fall back to x264enc"; \
		fi; \
	fi; \
	if ! gst-inspect-1.0 x264enc >/dev/null 2>&1; then \
		echo "Installing x264 encoder support"; \
		dpkg -s libx264-dev >/dev/null 2>&1 || (sudo apt-get update && sudo apt-get install -y libx264-dev); \
	fi

build-engine:
	@if [ -f models/best.engine ]; then \
		echo "TensorRT engine already exists (models/best.engine)"; \
	else \
		echo "Building TensorRT engine from models/best.onnx (FP16)"; \
		if [ ! -x "$(TRTEXEC)" ]; then \
			echo "ERROR: trtexec not found at $(TRTEXEC). Set TRTEXEC=/path/to/trtexec"; \
			exit 1; \
		fi; \
		"$(TRTEXEC)" --onnx=models/best.onnx --saveEngine=models/best.engine --fp16 --memPoolSize=workspace:$(TRT_WORKSPACE) || true; \
		if [ ! -f models/best.engine ]; then echo "WARNING: trtexec failed to generate engine; runtime will fall back to ONNX"; fi; \
	fi

run:
	@nohup $(PY) -m bamboo_vision.app --config config/runtime.yaml > /tmp/bamboo-vision.run.log 2>&1 & echo $$! > /tmp/bamboo-vision.run.pid; \
	echo "bamboo-vision started in background (PID $$(cat /tmp/bamboo-vision.run.pid)), logs: /tmp/bamboo-vision.run.log"

install: deps
	sudo mkdir -p "$(PREFIX)"
	sudo cp -r bamboo_vision.py bamboo_vision config models bamboo.html requirements.txt RUNNING.md "$(PREFIX)"
	sudo install -D -m644 deploy/systemd/$(SERVICE) /etc/systemd/system/$(SERVICE)
	sudo systemctl daemon-reload
	sudo systemctl enable --now $(SERVICE)

service: install

service-restart:
	sudo systemctl restart $(SERVICE)

service-stop:
	sudo systemctl stop $(SERVICE)

service-status:
	sudo systemctl status $(SERVICE) --no-pager

logs:
	sudo journalctl -u $(SERVICE) -n 200 -f

clean-install:
	sudo systemctl stop $(SERVICE) 2>/dev/null || true
	sudo systemctl disable $(SERVICE) 2>/dev/null || true
	sudo rm -f /etc/systemd/system/$(SERVICE)
	sudo systemctl daemon-reload
	sudo rm -rf "$(PREFIX)"

redeploy: clean-install install
