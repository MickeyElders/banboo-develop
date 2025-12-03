PREFIX ?= /opt/bamboo-qt
BUILD_DIR ?= build
SERVICE_NAME ?= bamboo-qt-ui.service
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release -DENABLE_GSTREAMER=ON -DENABLE_MODBUS=ON

# Required Debian packages (fail if missing)
MANDATORY_DEPS ?= libmodbus-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-tools \
	libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
	libgstrtspserver-1.0-dev gstreamer1.0-rtsp gstreamer1.0-nice libnice10 \
	gstreamer1.0-x libx264-dev nvidia-l4t-gstreamer nvidia-l4t-multimedia nvidia-l4t-multimedia-utils \
	qml6-module-qtquick qml6-module-qtquick-controls qml6-module-qtquick-layouts qml6-module-qtquick-window qml6-module-qtquick-dialogs \
	qml6-module-qtmultimedia qml6-module-qtqml qml6-module-qtqml-workerscript qml6-module-qtquick-templates \
	qt6-multimedia-dev libqt6websockets6-dev
# Qt6 toolchain from distro (must install if available)
QT6_DEPS ?= qt6-base-dev qt6-base-dev-tools qt6-declarative-dev qt6-multimedia-dev
# Optional packages (best effort, ignore failures)
OPTIONAL_DEPS ?= libgstrtspserver-1.0-dev gstreamer1.0-rtsp gstreamer1.0-libav qt6-shadertools-dev

# Minimal acceptable Qt6 version (LTS, apt friendly)
QT_MIN_VER ?= 6.2.4
# Optional source upgrade to 6.6.3 when you need GBM/eglfs improvements
QT6_SOURCE_URL ?= https://download.qt.io/official_releases/qt/6.6/6.6.3/single/qt-everywhere-src-6.6.3.tar.xz
QT6_PREFIX ?= /opt/qt6
QT6_TARBALL ?=
# Optional mirror list (space separated)
QT6_MIRRORS ?= https://mirrors.cloud.tencent.com/qt/official_releases/qt/6.6/6.6.3/single/qt-everywhere-src-6.6.3.tar.xz \
 https://mirrors.tuna.tsinghua.edu.cn/qt/official_releases/qt/6.6/6.6.3/single/qt-everywhere-src-6.6.3.tar.xz \
 https://mirrors.ustc.edu.cn/qtproject/official_releases/qt/6.6/6.6.3/single/qt-everywhere-src-6.6.3.tar.xz

# HLS repack services (RTSP->HLS) for browser playback (optional)
HLS_SERVICES ?= deploy/systemd/bamboo-rtsp-hls.service deploy/systemd/bamboo-hls-http.service
HLS_SCRIPT ?= deploy/scripts/hls-repack.sh
# MediaMTX gateway (RTSP->HLS/WebRTC)
MEDIAMTX_SERVICE ?= deploy/systemd/mediamtx.service
MEDIAMTX_CONFIG ?= deploy/mediamtx.yml
MEDIAMTX_CONFIG_DEST ?= /etc/mediamtx/mediamtx.yml
# MediaMTX prebuilt binary
MEDIAMTX_BIN ?= /usr/local/bin/mediamtx
MEDIAMTX_VER ?= 1.8.1
MEDIAMTX_URL ?= https://github.com/bluenviron/mediamtx/releases/download/v$(MEDIAMTX_VER)/mediamtx_linux_arm64.tar.gz

.PHONY: all deps configure build run install install-config install-models service hls-services mediamtx-service start stop restart status logs deploy redeploy clean distclean
.PHONY: mediamtx

all: build

deps:
	@missing=""; \
	for pkg in $(MANDATORY_DEPS); do \
		dpkg -s $$pkg >/dev/null 2>&1 || missing="$$missing $$pkg"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "Installing missing packages:$$missing"; \
		sudo apt-get update && sudo apt-get install -y $$missing; \
	else \
		echo "All mandatory packages already installed."; \
	fi
	@qt_missing=""; \
	for pkg in $(QT6_DEPS); do \
		dpkg -s $$pkg >/dev/null 2>&1 || qt_missing="$$qt_missing $$pkg"; \
	done; \
	if [ -n "$$qt_missing" ]; then \
		echo "Installing Qt6 toolchain:$$qt_missing"; \
		sudo apt-get update && sudo apt-get install -y $$qt_missing; \
	else \
		echo "Qt6 toolchain already installed."; \
	fi
	@opt_missing=""; \
	for pkg in $(OPTIONAL_DEPS); do \
		dpkg -s $$pkg >/dev/null 2>&1 || opt_missing="$$opt_missing $$pkg"; \
	done; \
	if [ -n "$$opt_missing" ]; then \
		echo "Attempting to install optional packages (ignored if unavailable):$$opt_missing"; \
		sudo apt-get install -y $$opt_missing || true; \
	else \
		echo "All optional packages already installed."; \
	fi
	@$(MAKE) --no-print-directory mediamtx
	@$(MAKE) --no-print-directory check_qt6

mediamtx:
	@if [ -x "$(MEDIAMTX_BIN)" ]; then \
		echo "mediamtx already installed at $(MEDIAMTX_BIN)"; \
	else \
		tmpdir=$$(mktemp -d); \
		echo "Installing mediamtx $(MEDIAMTX_VER) from $(MEDIAMTX_URL)"; \
		cd $$tmpdir && wget -q "$(MEDIAMTX_URL)" -O mediamtx.tar.gz && \
			tar xf mediamtx.tar.gz && \
			sudo install -m 755 mediamtx "$(MEDIAMTX_BIN)"; \
		rm -rf $$tmpdir; \
	fi

hls-services: install
	@for svc in $(HLS_SERVICES); do \
		echo "Installing $$svc to /etc/systemd/system/$${svc##*/}"; \
		sudo install -D -m644 $$svc /etc/systemd/system/$${svc##*/}; \
	done
	@sudo install -D -m755 $(HLS_SCRIPT) /opt/bamboo-qt/bin/hls-repack.sh
	@sudo systemctl daemon-reload
	@sudo systemctl enable --now bamboo-hls-http.service
	@sudo systemctl restart bamboo-hls-http.service
	@sudo systemctl enable --now bamboo-rtsp-hls.service
	@sudo systemctl restart bamboo-rtsp-hls.service

mediamtx-service: mediamtx
	@echo "Installing $(MEDIAMTX_SERVICE) to /etc/systemd/system/$${MEDIAMTX_SERVICE##*/}"
	@sudo install -D -m644 $(MEDIAMTX_SERVICE) /etc/systemd/system/$${MEDIAMTX_SERVICE##*/}
	@echo "Installing MediaMTX config to $(MEDIAMTX_CONFIG_DEST)"
	@sudo install -D -m644 $(MEDIAMTX_CONFIG) $(MEDIAMTX_CONFIG_DEST)
	@sudo systemctl daemon-reload
	@sudo systemctl enable --now mediamtx.service
	@sudo systemctl restart mediamtx.service

jsmpeg-service: install
	@echo "Installing $(JSMPEG_SERVICE) to /etc/systemd/system/$${JSMPEG_SERVICE##*/}"
	@sudo install -D -m644 $(JSMPEG_SERVICE) /etc/systemd/system/$${JSMPEG_SERVICE##*/}
	@sudo systemctl daemon-reload
	@sudo systemctl enable --now bamboo-jsmpeg.service
	@sudo systemctl restart bamboo-jsmpeg.service

configure: deps
	@if [ ! -d "$(BUILD_DIR)" ]; then mkdir -p "$(BUILD_DIR)"; fi
	cd "$(BUILD_DIR)" && cmake $(CMAKE_FLAGS) ..

build: configure
	cd "$(BUILD_DIR)" && cmake --build . --config Release -j$$(nproc 2>/dev/null || echo 4)

run: build
	cd "$(BUILD_DIR)" && ./bamboo_qt_ui

install: build install-config install-models
	cd "$(BUILD_DIR)" && cmake --install . --prefix "$(PREFIX)"
	@# install web assets
	@install -m 644 bamboo.html "$(PREFIX)/" 2>/dev/null || true

install-config:
	@mkdir -p "$(PREFIX)/config"
	@cp -r config/* "$(PREFIX)/config/" || true

install-models:
	@mkdir -p "$(PREFIX)/models"
	@cp -f models/best.onnx "$(PREFIX)/models/" 2>/dev/null || true

service: install
	@$(MAKE) --no-print-directory install-qt-kms-config
	@sudo install -D -m 644 deploy/systemd/bamboo-qt-ui.service /etc/systemd/system/$(SERVICE_NAME)
	@sudo systemctl daemon-reload
	@sudo systemctl enable $(SERVICE_NAME)
	@sudo systemctl restart $(SERVICE_NAME)


start:
	@sudo systemctl start $(SERVICE_NAME)

stop:
	@sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true

restart:
	@sudo systemctl restart $(SERVICE_NAME)

status:
	@sudo systemctl status $(SERVICE_NAME) --no-pager

logs:
	@sudo journalctl -u $(SERVICE_NAME) -n 200 -f

# Default deploy: only UI service (RTSP output). HLS is optional via `make hls-services`.
deploy: service mediamtx-service

# Full redeploy: stop -> clean -> build/install -> restart UI + MediaMTX gateway
redeploy: stop clean service mediamtx-service

clean:
	@rm -rf "$(BUILD_DIR)"

distclean: clean
	@rm -f CMakeCache.txt

.PHONY: check_qt6 qt6-source-install install-qt-kms-config

check_qt6:
	@qtver=""; \
	if command -v qtpaths6 >/dev/null 2>&1; then \
		qtver=$$(qtpaths6 --qt-version); \
	elif command -v qmake6 >/dev/null 2>&1; then \
		qtver=$$(qmake6 -query QT_VERSION); \
	fi; \
	if [ -z "$$qtver" ]; then \
		echo "Qt6 not found. Install Qt >= $(QT_MIN_VER) via apt (qt6-base-dev qt6-declarative-dev qt6-multimedia-dev) or run 'make qt6-source-install' to build 6.6.3 manually."; \
		exit 1; \
	elif dpkg --compare-versions "$$qtver" ge "$(QT_MIN_VER)"; then \
		echo "Qt6 $$qtver OK (>= $(QT_MIN_VER))."; \
		if dpkg --compare-versions "$$qtver" lt "6.6.3"; then \
			echo "Note: For eglfs_kms/GBM 零拷贝增强，可选手动升级到 6.6.3：make qt6-source-install"; \
		fi; \
	else \
		echo "Qt6 $$qtver is too old (< $(QT_MIN_VER)). Install via apt or run 'make qt6-source-install' to build 6.6.3."; \
		exit 1; \
	fi

install-qt-kms-config:
	@echo "KMS config skipped (offscreen mode)."

qt6-source-install:
	@set -e; \
	tmpdir=$$(mktemp -d); \
	tarball="$(QT6_TARBALL)"; \
	url="$(QT6_SOURCE_URL)"; \
	mirrors="$(QT6_MIRRORS)"; \
	cd $$tmpdir; \
	if [ -n "$$tarball" ]; then \
		if [ ! -f "$$tarball" ]; then echo "QT6_TARBALL specified but file not found: $$tarball"; exit 1; fi; \
		cp "$$tarball" ./qt-everywhere-src-6.6.3.tar.xz; \
		echo "Using local Qt6 source tarball $$tarball"; \
	else \
		ok=0; \
		for u in $$url $$mirrors; do \
			echo "Downloading Qt6 from $$u"; \
			if command -v wget >/dev/null 2>&1; then \
				wget -q --show-progress -O qt-everywhere-src-6.6.3.tar.xz "$$u" && ok=1 && break; \
			elif command -v curl >/dev/null 2>&1; then \
				curl -L --progress-bar -o qt-everywhere-src-6.6.3.tar.xz "$$u" && ok=1 && break; \
			else \
				echo "Need wget or curl to fetch Qt6 sources. Install one and retry."; \
				exit 1; \
			fi; \
			echo "Download failed from $$u, trying next mirror..."; \
		done; \
		if [ $$ok -ne 1 ]; then \
			echo "All Qt6 download attempts failed. Check network/URL or set QT6_TARBALL to a local qt-everywhere-src-6.6.3.tar.xz."; \
			exit 1; \
		fi; \
	fi; \
	tar xf qt-everywhere-src-6.6.3.tar.xz; \
	cd qt-everywhere-src-6.6.3; \
	mkdir -p build && cd build; \
	echo "Configuring Qt6.6.3 to $(QT6_PREFIX) (this will take time)..."; \
	../configure -prefix $(QT6_PREFIX) -opensource -confirm-license -release \
		-opengl es2 -eglfs -kms -gbm -no-xcb -nomake tests -nomake examples \
		-skip qt3d -skip qtwebengine -skip qtvirtualkeyboard -skip qtdoc; \
	cmake --build . --parallel $$(nproc); \
	sudo cmake --install .; \
	echo "Qt6.6.3 installed to $(QT6_PREFIX). Add -DCMAKE_PREFIX_PATH=$(QT6_PREFIX)/lib/cmake when configuring."
