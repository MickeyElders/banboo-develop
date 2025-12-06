SHELL := /bin/bash

PY ?= python3
PIP ?= $(PY) -m pip
PREFIX ?= /opt/bamboo-vision
SERVICE ?= bamboo-vision.service
KIOSK_USER ?= ubuntu
KIOSK_URL ?= http://localhost:8080/bamboo.html
JETSON_PY ?= /usr/local/python
JI_SRC ?= ../jetson-inference
JI_SRC_ABS := $(abspath $(JI_SRC))
JI_BUILD ?= $(JI_SRC_ABS)/build
JI_PY ?= $(JI_BUILD)/python
JI_LIB ?= $(JI_BUILD)/lib
# Default TensorRT CLI path (JetPack install). Override with TRTEXEC=/path/to/trtexec if different.
TRTEXEC ?= /usr/src/tensorrt/bin/trtexec
TRT_WORKSPACE := 2048  # MiB workspace for TensorRT engine build

.PHONY: deps check-jetson run install service service-restart service-stop service-status logs clean-install redeploy check-ji-source kiosk-deps kiosk-service stop restart update-ji-source install-jetson-if-missing

stop: service-stop
restart: service-restart
deps: check-ji-source
	@set -e; if ! $(PY) -m pip --version >/dev/null 2>&1; then \
		echo "pip not found; installing python3-pip"; \
		sudo apt-get update && sudo apt-get install -y python3-pip; \
	fi
	$(PIP) install -r requirements.txt
	$(MAKE) update-ji-source
	$(MAKE) install-jetson-if-missing
	$(MAKE) check-gst
	$(MAKE) build-engine

check-ji-source:
	@if [ ! -f "$(JI_SRC_ABS)/CMakeLists.txt" ]; then \
		echo "jetson-inference source missing at $(JI_SRC_ABS), cloning..."; \
		$(MAKE) update-ji-source; \
	fi

check-jetson:
	@if PYTHONPATH="$(JI_PY):$(JETSON_PY):$$PYTHONPATH" LD_LIBRARY_PATH="$(JI_LIB):$$LD_LIBRARY_PATH" $(PY) -c "import jetson.inference, jetson.utils" >/dev/null 2>&1; then \
		echo "jetson-inference bindings OK"; \
	else \
		echo "jetson-inference Python bindings not found; attempting source install..." ; \
		$(MAKE) install-jetson; \
	fi

install-jetson:
	@set -e; \
	missing=""; \
	for pkg in python3-dev python3-numpy libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev libsoup2.4-dev libjson-glib-dev libglew-dev freeglut3-dev libgl1-mesa-dev cmake build-essential git pkg-config; do \
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
	if [ ! -f "$(JI_SRC_ABS)/CMakeLists.txt" ]; then \
		echo "ERROR: $(JI_SRC_ABS) does not contain CMakeLists.txt"; \
		ls -l "$(JI_SRC_ABS)"; \
		exit 1; \
	fi; \
	# Clean old build and rebuild in $(JI_BUILD) (matches manual working steps)
	sudo rm -rf "$(JI_BUILD)"; \
	sudo mkdir -p "$(JI_BUILD)"; \
	cd "$(JI_BUILD)" && sudo cmake .. -DENABLE_PYTHON=ON -DENABLE_GSTREAMER=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DNUMPY_INCLUDE_DIRS="$$NUMPY_INC" -DNUMPY_LIBRARIES="$$NPYMATH"; \
	cd "$(JI_BUILD)" && sudo make -j$$(nproc); \
	cd "$(JI_BUILD)" && sudo make install; \
	sudo ldconfig; \
	if ! find "$(JI_BUILD)" -name 'jetson_utils*.so' | grep -q .; then \
		echo "ERROR: jetson_utils python binding not found under $(JI_BUILD) after build"; \
		exit 1; \
	fi; \
	if ! find "$(JI_BUILD)" -name 'jetson_inference*so' | grep -q .; then \
		echo "ERROR: jetson_inference python binding not found under $(JI_BUILD) after build"; \
		exit 1; \
	fi; \
	echo "jetson-inference build completed under $(JI_BUILD)"

install-jetson-if-missing:
	@set -e; \
	if python3 -c "import jetson_inference, jetson_utils" >/dev/null 2>&1; then \
		echo "jetson-inference already available in system python, skipping build/install"; \
	else \
		$(MAKE) install-jetson; \
	fi

update-ji-source:
	@set -e; \
	if [ -d "$(JI_SRC_ABS)/.git" ]; then \
		echo "Updating jetson-inference at $(JI_SRC_ABS)"; \
		sudo git -C "$(JI_SRC_ABS)" fetch origin master --depth=1; \
		sudo git -C "$(JI_SRC_ABS)" reset --hard origin/master; \
		sudo git -C "$(JI_SRC_ABS)" submodule update --init --recursive --depth=1; \
	else \
		echo "Cloning jetson-inference into $(JI_SRC_ABS)"; \
		sudo rm -rf "$(JI_SRC_ABS)"; \
		sudo git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference "$(JI_SRC_ABS)"; \
	fi

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
	@ENGINE_PATH="models/best.engine"; \
	if [ -f "$$ENGINE_PATH" ]; then \
		echo "TensorRT engine already exists ($$ENGINE_PATH); skip build"; \
	else \
		echo "Building TensorRT engine from models/best.onnx (FP16)"; \
		if [ ! -x "$(TRTEXEC)" ]; then \
			echo "ERROR: trtexec not found at $(TRTEXEC). Set TRTEXEC=/path/to/trtexec"; \
			exit 1; \
		fi; \
		"$(TRTEXEC)" --onnx=models/best.onnx --saveEngine=models/best.engine --shapes=images:1x3x960x960 --minShapes=images:1x3x960x960 --optShapes=images:1x3x960x960 --maxShapes=images:1x3x960x960 --fp16 --best --noBuilderCache --tacticSources=+CUBLAS,+CUDNN,-JIT_CONVOLUTIONS --maxAuxStreams=1 --builderOptimizationLevel=5 --verbose --memPoolSize=workspace:$(TRT_WORKSPACE) || true; \
		if [ ! -f models/best.engine ]; then echo "WARNING: trtexec failed to generate engine; runtime will fall back to ONNX"; fi; \
	fi

run:
	@PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3/dist-packages:/usr/local/python:$$PYTHONPATH" LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:$$LD_LIBRARY_PATH" nohup $(PY) -m bamboo_vision.app --config config/runtime.yaml > /tmp/bamboo-vision.run.log 2>&1 & echo $$! > /tmp/bamboo-vision.run.pid; \
	echo "bamboo-vision started in background (PID $$(cat /tmp/bamboo-vision.run.pid)), logs: /tmp/bamboo-vision.run.log"

install: deps
	-$(MAKE) service-stop
	sudo mkdir -p "$(PREFIX)"
	sudo cp -r bamboo_vision.py bamboo_vision config models bamboo.html requirements.txt RUNNING.md "$(PREFIX)"
	$(MAKE) check-ji-binaries
	sudo install -D -m644 deploy/systemd/$(SERVICE) /etc/systemd/system/$(SERVICE)
	sudo systemctl daemon-reload
	sudo systemctl enable --now $(SERVICE)

service: install

service-restart:
	sudo systemctl restart $(SERVICE)

service-stop:
	sudo systemctl stop $(SERVICE) 2>/dev/null || true
	sudo systemctl reset-failed $(SERVICE) 2>/dev/null || true

service-status:
	sudo systemctl status $(SERVICE) --no-pager

logs:
	sudo journalctl -u $(SERVICE) -n 200 -f

clean-install:
	# Stop/disable service first (no force-kill to avoid terminating this shell)
	sudo systemctl stop $(SERVICE) 2>/dev/null || true
	sudo systemctl disable $(SERVICE) 2>/dev/null || true
	sudo systemctl reset-failed $(SERVICE) 2>/dev/null || true
	# Remove unit file and reload
	sudo rm -f /etc/systemd/system/$(SERVICE)
	sudo systemctl daemon-reload
	# Remove install prefix
	sudo rm -rf "$(PREFIX)"

redeploy: service-stop clean-install install

# verify bindings exist both in source build and install prefix
check-ji-binaries:
	@if ! find "$(JI_SRC_ABS)/build" -name 'jetson_utils*.so' | grep -q .; then \
		echo "ERROR: jetson_utils binding missing in source build $(JI_SRC_ABS)/build"; \
		exit 1; \
	fi; \
	if [ -d "$(PREFIX)/jetson-inference" ]; then \
		if ! find "$(PREFIX)/jetson-inference/build" -name 'jetson_utils*.so' | grep -q .; then \
			echo "ERROR: jetson_utils binding missing in install prefix $(PREFIX)/jetson-inference/build"; \
			exit 1; \
		fi; \
		if ! find "$(PREFIX)/jetson-inference/build" -name 'jetson_inference*.so' | grep -q .; then \
			echo "ERROR: jetson_inference binding missing in install prefix $(PREFIX)/jetson-inference/build"; \
			exit 1; \
		fi; \
	else \
		echo "WARN: install prefix $(PREFIX)/jetson-inference not found; deploy may be incomplete"; \
	fi

kiosk-deps:
	@set -e; \
	missing=""; \
	for pkg in xserver-xorg xinit openbox chromium-browser x11-xserver-utils; do \
		dpkg -s $$pkg >/dev/null 2>&1 || missing="$$missing $$pkg"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "Installing kiosk dependencies:$$missing"; \
		sudo apt-get update && sudo apt-get install -y $$missing; \
	else \
		echo "Kiosk dependencies OK"; \
	fi

kiosk-service: kiosk-deps install
	sudo install -D -m755 deploy/scripts/kiosk-session.sh "$(PREFIX)/bin/kiosk-session.sh"
	sudo install -D -m755 deploy/scripts/kiosk-startx.sh "$(PREFIX)/bin/kiosk-startx.sh"
	sudo sh -c "sed -e 's|@USER@|$(KIOSK_USER)|g' -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@URL@|$(KIOSK_URL)|g' deploy/systemd/bamboo-kiosk.service > /etc/systemd/system/bamboo-kiosk.service"
	sudo systemctl daemon-reload
	sudo systemctl enable --now bamboo-kiosk.service
