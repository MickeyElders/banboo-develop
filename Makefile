PREFIX ?= /opt/bamboo-qt
BUILD_DIR ?= build
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release -DENABLE_GSTREAMER=ON

.PHONY: all configure build run install service clean distclean

all: build

configure:
	@if [ ! -d "$(BUILD_DIR)" ]; then mkdir -p "$(BUILD_DIR)"; fi
	cd "$(BUILD_DIR)" && cmake $(CMAKE_FLAGS) ..

build: configure
	cd "$(BUILD_DIR)" && cmake --build . --config Release -j$$(nproc 2>/dev/null || echo 4)

run: build
	cd "$(BUILD_DIR)" && ./bamboo_qt_ui

install: build
	cd "$(BUILD_DIR)" && cmake --install . --prefix "$(PREFIX)"

service: install
	@sudo install -D -m 644 deploy/systemd/bamboo-qt-ui.service /etc/systemd/system/bamboo-qt-ui.service
	@sudo systemctl daemon-reload
	@sudo systemctl enable bamboo-qt-ui.service
	@sudo systemctl restart bamboo-qt-ui.service

clean:
	@rm -rf "$(BUILD_DIR)"

distclean: clean
	@rm -f CMakeCache.txt
