PREFIX ?= /opt/bamboo-qt
BUILD_DIR ?= build
SERVICE_NAME ?= bamboo-qt-ui.service
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release -DENABLE_GSTREAMER=ON -DENABLE_MODBUS=ON

# 必须依赖（缺少就中断）
MANDATORY_DEPS ?= libmodbus-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-tools qtbase5-dev qtdeclarative5-dev qml-module-qtquick-controls2 qml-module-qtquick-layouts qtmultimedia5-dev
# 可选依赖（尝试安装，失败不影响构建），RTSP 服务器包在不同发行版有不同命名
OPTIONAL_DEPS ?= gstreamer1.0-rtsp gstreamer1.0-rtsp-server gstreamer1.0-plugins-rtsp gstreamer1.0-libav

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
		echo "All Debian dependencies already installed."; \
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
	@sudo journalctl -u $(SERVICE_NAME) -f

deploy: service

# 完整重新部署：停服→清理→重建→安装并重启服务
redeploy: stop clean service

clean:
	@rm -rf "$(BUILD_DIR)"

distclean: clean
	@rm -f CMakeCache.txt
