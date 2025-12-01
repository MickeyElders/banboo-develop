PREFIX ?= /opt/bamboo-qt
BUILD_DIR ?= build
SERVICE_NAME ?= bamboo-qt-ui.service
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release -DENABLE_GSTREAMER=ON -DENABLE_MODBUS=ON

# Required Debian packages (fail if missing)
MANDATORY_DEPS ?= libmodbus-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-tools
# Qt6 toolchain from distro (must install if available)
QT6_DEPS ?= qt6-base-dev qt6-base-dev-tools qt6-declarative-dev qt6-multimedia-dev
# Optional packages (best effort, ignore failures)
OPTIONAL_DEPS ?= libgstrtspserver-1.0-dev gstreamer1.0-rtsp gstreamer1.0-libav qt6-shadertools-dev

# Minimal acceptable Qt6 version and source installer settings
QT_MIN_VER ?= 6.6.3
QT6_SOURCE_URL ?= https://download.qt.io/official_releases/qt/6.6/6.6.3/single/qt-everywhere-src-6.6.3.tar.xz
QT6_PREFIX ?= /opt/qt6

.PHONY: all deps configure build run install install-config install-models service start stop restart status logs deploy redeploy clean distclean

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
	@$(MAKE) --no-print-directory check_qt6

configure: deps
	@if [ ! -d "$(BUILD_DIR)" ]; then mkdir -p "$(BUILD_DIR)"; fi
	cd "$(BUILD_DIR)" && cmake $(CMAKE_FLAGS) ..

build: configure
	cd "$(BUILD_DIR)" && cmake --build . --config Release -j$$(nproc 2>/dev/null || echo 4)

run: build
	cd "$(BUILD_DIR)" && ./bamboo_qt_ui

install: build install-config install-models
	cd "$(BUILD_DIR)" && cmake --install . --prefix "$(PREFIX)"

install-config:
	@mkdir -p "$(PREFIX)/config"
	@cp -r config/* "$(PREFIX)/config/" || true

install-models:
	@mkdir -p "$(PREFIX)/models"
	@cp -f models/best.onnx "$(PREFIX)/models/" 2>/dev/null || true

service: install
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

deploy: service

# Full redeploy: stop -> clean -> build/install -> restart service
redeploy: stop clean service

clean:
	@rm -rf "$(BUILD_DIR)"

distclean: clean
	@rm -f CMakeCache.txt

.PHONY: check_qt6 qt6-source-install

check_qt6:
	@qtver=""; \
	if command -v qtpaths6 >/dev/null 2>&1; then \
		qtver=$$(qtpaths6 --qt-version); \
	elif command -v qmake6 >/dev/null 2>&1; then \
		qtver=$$(qmake6 -query QT_VERSION); \
	fi; \
	if [ -z "$$qtver" ]; then \
		echo "Qt6 not found. Will build and install Qt $(QT_MIN_VER) from source."; \
		$(MAKE) --no-print-directory qt6-source-install; \
	elif dpkg --compare-versions "$$qtver" ge "$(QT_MIN_VER)"; then \
		echo "Qt6 $$qtver OK (>= $(QT_MIN_VER))."; \
	else \
		echo "Qt6 $$qtver is too old (< $(QT_MIN_VER)). Building Qt $(QT_MIN_VER) from source..."; \
		$(MAKE) --no-print-directory qt6-source-install; \
	fi

qt6-source-install:
	@set -e; \
	echo "Downloading Qt6 from $(QT6_SOURCE_URL)"; \
	tmpdir=$$(mktemp -d); \
	cd $$tmpdir; \
	wget -q --show-progress $(QT6_SOURCE_URL); \
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
