# AI竹子识别系统 - C++ LVGL一体化构建和部署脚�?
# 版本: 5.0.0 (C++ LVGL Integrated Architecture)
# C++推理后端 + LVGL界面 + Modbus通信的完整一体化系统

.PHONY: all install clean test deploy start stop restart status logs \
        install-deps install-system-deps install-lvgl build-lvgl-from-source \
        install-service enable-service disable-service \
        check-system check-wayland build-system install-system setup-config \
        start-mutter stop-mutter mutter-status mutter-logs check-mutter setup-mutter \
        start-weston stop-weston weston-status auto-setup-environment \
        check-weston-version backup-current-weston uninstall-current-weston \
        install-weston12-build-deps download-weston12 compile-weston12 \
        install-weston12 configure-weston12 setup-weston12-service \
        start-weston12 stop-weston12 weston12-status weston12-logs \
        downgrade-to-weston12 test-weston12 \
        build-debug test-system backup

# === 系统配置 ===
PROJECT_NAME := bamboo-recognition-system
VERSION := 5.0.0
INSTALL_DIR := /opt/bamboo-cut
SERVICE_NAME := bamboo-cpp-lvgl
BINARY_NAME := bamboo_integrated

# === C++ LVGL一体化构建配置 ===
BUILD_DIR := build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
               -DENABLE_AI_OPTIMIZATION=ON \
               -DENABLE_TENSORRT=ON \
               -DENABLE_CUDA=ON \
               -DENABLE_MODBUS=ON \
               -DENABLE_LVGL=ON \
               -DENABLE_CPP_INFERENCE=ON \
               -DENABLE_HARDWARE_ACCELERATION=ON

# === 颜色定义 ===
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# === 主要目标 ===

all: check-system auto-setup-environment install-deps build-system
	@echo "$(CYAN)=== AI竹子识别系统构建完成 (v$(VERSION)) ===$(NC)"
	@echo "$(GREEN)C++ LVGL Wayland一体化工业级嵌入式架构$(NC)"
	@echo "使用 'make deploy' 完成系统部署"

install: all install-system install-service
	@echo "$(GREEN)=== 系统安装完成 ===$(NC)"
	@echo "服务名称: $(SERVICE_NAME)"
	@echo "安装目录: $(INSTALL_DIR)"
	@echo "可执行文�? $(INSTALL_DIR)/bin/$(BINARY_NAME)"
	@echo "Wayland环境: 已自动配�?
	@echo "使用 'make start' 启动系统"

deploy: auto-setup-environment install enable-service start
	@echo "$(GREEN)[SUCCESS]$(NC) 系统部署完成�?
	@echo "Wayland环境（Weston 12）已自动配置并启�?

help:
	@echo "$(CYAN)===============================================$(NC)"
	@echo "$(CYAN)   AI竹子识别系统 C++ LVGL构建系统 v$(VERSION)$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)快速部署命�?$(NC)"
	@echo "  deploy           - 首次完整部署(构建+安装+启动服务)"
	@echo "  redeploy         - 代码修改后快速重新部�?跳过依赖检�?"
	@echo "  smart-redeploy   - 智能重新部署(仅在必要时安装依�?"
	@echo "  full-redeploy    - 完整重新部署(包括依赖检�?"
	@echo "  backup           - 创建当前系统备份"
	@echo "  test-system      - 测试模式运行系统"
	@echo ""
	@echo "$(GREEN)构建命令:$(NC)"
	@echo "  all              - 检查系�?安装依赖+构建系统"
	@echo "  build-system     - 构建C++ LVGL系统"
	@echo "  build-debug      - 构建调试版本"
	@echo "  clean            - 清理构建目录"
	@echo ""
	@echo "$(GREEN)单进程统一架构:$(NC)"
	@echo "  unified          - 构建单进程LVGL+GStreamer统一架构"
	@echo "  unified-run      - 运行统一架构"
	@echo "  unified-test     - 测试统一架构和环�?
	@echo "  unified-clean    - 清理统一架构构建文件"
	@echo ""
	@echo "$(GREEN)摄像头诊断工�?$(NC)"
	@echo "  camera-diag      - 运行完整摄像头诊�?
	@echo "  camera-test      - 测试摄像头访�?(使用 SENSOR_ID=X 指定sensor)"
	@echo "  camera-fix       - 运行综合交互式摄像头修复脚本"
	@echo "  camera-fix-quick - 应用快速非交互式摄像头修复"
	@echo "  camera-fix-test  - 测试摄像头修复后功能 (使用 SENSOR_ID=X)"
	@echo ""
	@echo "$(GREEN)NVIDIA-DRM 迁移验证:$(NC)"
	@echo "  enable-nvidia-drm- 启用NVIDIA-DRM驱动（替换tegra_drm�?
	@echo "  force-nvidia-drm - 强制迁移nvidia-drm（处理顽固的tegra_drm�?
	@echo "  nvidia-drm-test  - 运行完整的NVIDIA-DRM迁移验证测试"
	@echo "  nvidia-drm-report- 生成NVIDIA-DRM迁移状态报�?
	@echo "  nvidia-drm-complete - 运行完整的迁移验证流�?
	@echo ""
	@echo "$(GREEN)安装命令:$(NC)"
	@echo "  install          - 完整安装系统"
	@echo "  install-deps     - 安装所有依�?系统+Wayland+LVGL)"
	@echo "  install-system-deps - 仅安装系统依�?
	@echo "  install-wayland-deps - 安装Wayland环境和Sway"
	@echo "  install-lvgl     - 检查并安装LVGL"
	@echo "  install-system   - 安装编译好的系统"
	@echo "  install-service  - 安装systemd服务"
	@echo ""
	@echo "$(GREEN)服务管理:$(NC)"
	@echo "  start            - 启动服务"
	@echo "  stop             - 停止服务"
	@echo "  restart          - 重启服务"
	@echo "  status           - 查看服务状�?
	@echo "  logs             - 查看服务日志"
	@echo "  enable-service   - 启用开机自�?
	@echo "  disable-service  - 禁用开机自�?
	@echo ""
	@echo "$(GREEN)Wayland环境管理（Sway�?$(NC)"
	@echo "  setup-wayland    - 配置Wayland环境和Sway服务"
	@echo "  start-sway       - 启动Sway合成器（自动安装+配置+启动�?
	@echo "  stop-sway        - 停止Sway合成�?
	@echo "  sway-status      - 查看Sway状态和触摸设备"
	@echo "  sway-logs        - 查看Sway日志"
	@echo "  check-wayland    - 检查Wayland环境完整�?
	@echo ""
	@echo "$(GREEN)Weston 12 降级支持（解�?Weston 13 bug�?$(NC)"
	@echo "  downgrade-to-weston12    - 🚀 一键降级到 Weston 12（推荐）"
	@echo "  check-weston-version     - 检查当�?Weston 版本"
	@echo "  start-weston12           - 启动 Weston 12 服务"
	@echo "  stop-weston12            - 停止 Weston 12 服务"
	@echo "  weston12-status          - 查看 Weston 12 状�?
	@echo "  weston12-logs            - 查看 Weston 12 日志"
	@echo "  test-weston12            - 测试 Weston 12 环境"
	@echo ""
	@echo "$(GREEN)维护命令:$(NC)"
	@echo "  check-system     - 检查系统环�?
	@echo "  check-wayland    - 检查Wayland环境"
	@echo "  setup-config     - 设置配置文件"
	@echo "  test             - 运行系统测试"
	@echo "  backup           - 备份当前系统"
	@echo ""
	@echo "$(YELLOW)特性说�?$(NC)"
	@echo "  �?C++17高性能推理引擎"
	@echo "  �?LVGL工业级触摸界�?
	@echo "  �?YOLOv8+TensorRT加�?
	@echo "  �?Modbus TCP PLC通信"
	@echo "  �?Jetson Orin NX优化"
	@echo "  �?实时视频处理与检�?

# === 系统检�?===
check-system:
	@echo "$(BLUE)[INFO]$(NC) 检查系统环�?.."
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) cmake未安�?; \
		exit 1; \
	fi
	@if ! command -v g++ >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) g++编译器未安装"; \
		exit 1; \
	fi
	@if ! pkg-config --exists opencv4 2>/dev/null; then \
		if ! pkg-config --exists opencv 2>/dev/null; then \
			echo "$(RED)[ERROR]$(NC) OpenCV开发库未安�?; \
			exit 1; \
		fi; \
	fi
	@if ! pkg-config --exists gstreamer-1.0 2>/dev/null; then \
		echo "$(RED)[ERROR]$(NC) GStreamer开发库未安�?; \
		exit 1; \
	fi
	@if [ ! -f "/usr/include/modbus/modbus.h" ] && [ ! -f "/usr/local/include/modbus/modbus.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) libmodbus开发库未找到，将禁用Modbus功能"; \
	fi
	@if [ ! -f "/usr/include/lvgl/lvgl.h" ] && [ ! -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL开发库未找到，将禁用界面功�?; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 系统环境检查通过"

# === 依赖安装 ===
install-deps: install-system-deps install-lvgl9-auto
	@echo "$(GREEN)[SUCCESS]$(NC) 所有依赖安装完�?

# === 自动环境配置 ===
# 使用 Weston 12（推荐用�?Jetson + Nvidia�?
auto-setup-environment:
	@echo "$(BLUE)[INFO]$(NC) 自动检查和配置Wayland环境（使用系�?Weston�?.."
	@# 1. 检�?Weston 是否已安�?
	@if ! command -v weston >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) Weston 未安装，请先安装: sudo apt-get install weston"; \
		exit 1; \
	fi
	@WESTON_VERSION=$$(weston --version 2>&1 | grep -oP 'weston \K[\d.]+' | head -1 || echo "unknown"); \
	echo "$(GREEN)[SUCCESS]$(NC) 检测到系统 Weston $$WESTON_VERSION"
	@# 2. 配置 Nvidia DRM 模块（Jetson 必需�?
	@echo "$(BLUE)[INFO]$(NC) 配置 Nvidia DRM 模块..."
	@sudo modprobe nvidia-drm modeset=1 2>/dev/null || true
	@if ! grep -q "options nvidia-drm modeset=1" /etc/modprobe.d/nvidia-drm.conf 2>/dev/null; then \
		echo "options nvidia-drm modeset=1" | sudo tee /etc/modprobe.d/nvidia-drm.conf >/dev/null; \
		echo "$(GREEN)[SUCCESS]$(NC) Nvidia DRM 模块配置已保�?; \
	fi
	@# 3. 配置用户权限
	@echo "$(BLUE)[INFO]$(NC) 配置 DRM 设备权限..."
	@sudo usermod -a -G video,render,input root 2>/dev/null || true
	@# 4. 配置 Weston 服务
	@if [ ! -f "/etc/systemd/system/weston.service" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) Weston 服务未配置，正在配置..."; \
		$(MAKE) setup-weston-service; \
	fi
	@# 5. 停止其他 Wayland 合成�?
	@if systemctl is-active --quiet sway-wayland.service 2>/dev/null; then \
		echo "$(BLUE)[INFO]$(NC) 停止 Sway 服务..."; \
		sudo systemctl stop sway-wayland.service; \
		sudo systemctl disable sway-wayland.service; \
	fi
	@if systemctl is-active --quiet weston12.service 2>/dev/null; then \
		echo "$(BLUE)[INFO]$(NC) 停止 Weston 12 服务..."; \
		sudo systemctl stop weston12.service; \
		sudo systemctl disable weston12.service; \
	fi
	@# 6. 启动 Weston
	@WESTON_RUNNING=false; \
	if pgrep -x weston >/dev/null 2>&1; then \
		echo "$(GREEN)[INFO]$(NC) 检测到 Weston 进程正在运行"; \
		WESTON_RUNNING=true; \
	elif systemctl is-active --quiet weston.service 2>/dev/null; then \
		echo "$(GREEN)[INFO]$(NC) 检测到 Weston 服务正在运行"; \
		WESTON_RUNNING=true; \
	fi; \
	if [ "$$WESTON_RUNNING" = "false" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) Weston 未运行，正在启动..."; \
		sudo systemctl enable weston.service; \
		sudo systemctl start weston.service; \
		sleep 3; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Weston 已在运行，跳过启�?; \
	fi
	@# 7. 验证 Wayland 环境
	@if ! ls /run/user/0/wayland-* >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Wayland socket 不存在，等待 Weston 完全启动..."; \
		sleep 5; \
	@sudo apt-get install -y \
		sway \
		swaylock \
		swayidle \
		xwayland \
		libinput-tools \
		libwayland-dev \
		libwayland-egl1 \
		wayland-protocols \
		libxkbcommon-dev
	@echo "$(GREEN)[SUCCESS]$(NC) Wayland依赖安装完成（Sway�?

# 检�?Mutter 是否已安�?
check-mutter:
	@echo "$(BLUE)[INFO]$(NC) 检查Mutter合成�?.."
	@if ! command -v mutter >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Mutter未安装，正在安装..."; \
		sudo apt-get update && sudo apt-get install -y mutter gnome-session dbus-x11; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Mutter已安�? $$(mutter --version 2>&1 | head -n1)"; \
	fi

# 配置 Mutter 服务
setup-mutter:
	@echo "$(BLUE)[INFO]$(NC) 配置Mutter Wayland服务..."
	@sudo mkdir -p /etc/systemd/system
	@echo "[Unit]" | sudo tee /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Description=Mutter Wayland Compositor" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "After=multi-user.target" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "[Service]" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Type=simple" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "User=root" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "ExecStartPre=/bin/sh -c 'mkdir -p /run/user/0 && chmod 700 /run/user/0'" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "ExecStartPre=/bin/sh -c 'pkill -9 dbus-daemon || true; dbus-daemon --session --address=unix:path=/run/user/0/bus --nofork --nopidfile --syslog &'" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "ExecStartPre=/bin/sleep 2" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "ExecStart=/usr/bin/mutter --wayland --no-x11 --display-server" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=XDG_RUNTIME_DIR=/run/user/0" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=WAYLAND_DISPLAY=wayland-0" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=EGL_PLATFORM=wayland" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/0/bus" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=XDG_SESSION_TYPE=wayland" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Environment=GDK_BACKEND=wayland" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "Restart=always" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "RestartSec=3" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "[Install]" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@echo "WantedBy=multi-user.target" | sudo tee -a /etc/systemd/system/mutter-wayland.service > /dev/null
	@sudo systemctl daemon-reload
	@echo "$(GREEN)[SUCCESS]$(NC) Mutter服务配置完成（已包含D-Bus启动�?

# 启动 Mutter
start-mutter: check-mutter setup-mutter
	@echo "$(BLUE)[INFO]$(NC) 启动Mutter Wayland合成�?.."
	@sudo systemctl enable mutter-wayland
	@sudo systemctl start mutter-wayland
	@sleep 2
	@if sudo systemctl is-active --quiet mutter-wayland; then \
		echo "$(GREEN)[SUCCESS]$(NC) Mutter启动成功"; \
		echo "WAYLAND_DISPLAY=$$(ls /run/user/0/wayland-* 2>/dev/null | head -n1 | xargs basename)"; \
	else \
		echo "$(RED)[ERROR]$(NC) Mutter启动失败"; \
		sudo journalctl -u mutter-wayland -n 20 --no-pager; \
		exit 1; \
	fi

# 停止 Mutter
stop-mutter:
	@echo "$(BLUE)[INFO]$(NC) 停止Mutter..."
	@sudo systemctl stop mutter-wayland || true
	@echo "$(GREEN)[SUCCESS]$(NC) Mutter已停�?

# 检�?Mutter 状�?
mutter-status:
	@echo "$(CYAN)=== Mutter状�?===$(NC)"
	@sudo systemctl status mutter-wayland --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "无Wayland socket"

# Mutter 日志
mutter-logs:
	@echo "$(CYAN)=== Mutter日志 ===$(NC)"
	@sudo journalctl -u mutter-wayland -f --no-hostname

# ============================================================================
# Sway Wayland 合成器（推荐用于嵌入式，支持触摸控制�?
# ============================================================================

# 检�?Sway 是否已安�?
check-sway:
	@echo "$(BLUE)[INFO]$(NC) 检查Sway合成�?.."
	@if ! command -v sway >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Sway未安装，正在安装..."; \
		sudo apt-get update && sudo apt-get install -y sway swaylock swayidle xwayland libinput-tools; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Sway已安�? $$(sway --version 2>&1 | head -n1)"; \
	fi

# 创建 Sway 配置文件（支持触摸控制）
setup-sway-config:
	@echo "$(BLUE)[INFO]$(NC) 创建Sway配置文件（支持触摸控制）..."
	@sudo mkdir -p /root/.config/sway
	@echo "# Sway配置 - Bamboo识别系统（触摸控制优化）" | sudo tee /root/.config/sway/config > /dev/null
	@echo "# 自动生成，请勿手动编�? | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 基础设置" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "set \$$mod Mod4" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 输出配置（根据实际屏幕调整）" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "output * bg #000000 solid_color" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "output * mode 1920x1080@60Hz" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 触摸�?触摸板支持配�? | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "input type:touchscreen {" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    tap enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    drag enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    events enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "}" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "input type:touchpad {" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    tap enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    natural_scroll enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    dwt enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "    drag enabled" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "}" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 禁用屏幕锁定和电源管理（工业应用�? | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "exec swayidle -w timeout 0 'echo disabled' before-sleep 'echo disabled'" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 禁用窗口装饰（全屏模式）" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "default_border none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "default_floating_border none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "titlebar_border_thickness 0" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "titlebar_padding 0" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 焦点配置" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "focus_follows_mouse no" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "mouse_warping none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 自动全屏应用" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "for_window [title=\".*\"] fullscreen enable" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@sudo chmod 644 /root/.config/sway/config
	@echo "$(GREEN)[SUCCESS]$(NC) Sway配置文件创建完成"

# 配置 Sway 服务
setup-sway:
	@echo "$(BLUE)[INFO]$(NC) 配置Sway Wayland服务..."
	@sudo mkdir -p /etc/systemd/system
	@echo "[Unit]" | sudo tee /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Description=Sway Wayland Compositor (Touch-Enabled)" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "After=multi-user.target" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "[Service]" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Type=simple" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "User=root" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "ExecStartPre=/bin/sh -c 'mkdir -p /run/user/0 && chmod 700 /run/user/0'" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "ExecStart=/usr/bin/sway --unsupported-gpu" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=XDG_RUNTIME_DIR=/run/user/0" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=WAYLAND_DISPLAY=wayland-1" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=XDG_SESSION_TYPE=wayland" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=EGL_PLATFORM=wayland" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=WLR_NO_HARDWARE_CURSORS=1" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=WLR_RENDERER=gles2" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Environment=LIBINPUT_DEFAULT_TOUCH_ENABLED=1" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "Restart=always" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "RestartSec=3" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "[Install]" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@echo "WantedBy=multi-user.target" | sudo tee -a /etc/systemd/system/sway-wayland.service > /dev/null
	@sudo systemctl daemon-reload
	@echo "$(GREEN)[SUCCESS]$(NC) Sway服务配置完成（已启用触摸支持�?

# 启动 Sway
start-sway: check-sway setup-sway-config setup-sway
	@echo "$(BLUE)[INFO]$(NC) 启动Sway Wayland合成�?.."
	@sudo systemctl enable sway-wayland
	@sudo systemctl start sway-wayland
	@sleep 3
	@if sudo systemctl is-active --quiet sway-wayland; then \
		echo "$(GREEN)[SUCCESS]$(NC) Sway启动成功"; \
		echo "WAYLAND_DISPLAY=$$(ls /run/user/0/wayland-* 2>/dev/null | head -n1 | xargs basename)"; \
		echo "触摸控制: 已启�?; \
	else \
		echo "$(RED)[ERROR]$(NC) Sway启动失败"; \
		sudo journalctl -u sway-wayland -n 30 --no-pager; \
		exit 1; \
	fi

# 停止 Sway
stop-sway:
	@echo "$(BLUE)[INFO]$(NC) 停止Sway..."
	@sudo systemctl stop sway-wayland || true
	@echo "$(GREEN)[SUCCESS]$(NC) Sway已停�?

# 检�?Sway 状�?
sway-status:
	@echo "$(CYAN)=== Sway状�?===$(NC)"
	@sudo systemctl status sway-wayland --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "无Wayland socket"
	@echo ""
	@echo "$(CYAN)=== 触摸设备 ===$(NC)"
	@libinput list-devices 2>/dev/null | grep -A 5 "Capabilities.*touch" || echo "未检测到触摸设备"

# Sway 日志
sway-logs:
	@echo "$(CYAN)=== Sway日志 ===$(NC)"
	@sudo journalctl -u sway-wayland -f --no-hostname

setup-wayland: start-sway
	@echo "$(GREEN)[SUCCESS]$(NC) Wayland环境配置完成（Sway + 触摸控制�?

# ============================================================================
# Weston 12 降级支持（解�?Weston 13 �?xdg_positioner bug�?
# ============================================================================

.PHONY: check-weston-version backup-current-weston uninstall-current-weston \
        install-weston12-build-deps download-weston12 compile-weston12 \
        install-weston12 configure-weston12 setup-weston12-service \
        start-weston12 stop-weston12 weston12-status weston12-logs \
        downgrade-to-weston12 test-weston12

# 检查当�?Weston 版本
check-weston-version:
	@echo "$(BLUE)[INFO]$(NC) 检�?Weston 版本..."
	@if command -v weston >/dev/null 2>&1; then \
		WESTON_VERSION=$$(weston --version 2>&1 | grep -oP 'weston \K\d+\.\d+' | head -1 || echo "未知"); \
		echo "$(CYAN)当前 Weston 版本: $$WESTON_VERSION$(NC)"; \
		WESTON_MAJOR=$$(echo $$WESTON_VERSION | cut -d. -f1); \
		if [ "$$WESTON_MAJOR" = "12" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) �?Weston 12 已安�?; \
		elif [ "$$WESTON_MAJOR" = "13" ]; then \
			echo "$(YELLOW)[WARNING]$(NC) �?Weston 13 存在已知 xdg_positioner bug，建议降�?; \
		elif [ "$$WESTON_MAJOR" = "9" ] || [ "$$WESTON_MAJOR" = "10" ]; then \
			echo "$(YELLOW)[WARNING]$(NC) �?Weston 版本较旧 ($$WESTON_VERSION)，建议升级到 12"; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) �?未知 Weston 版本: $$WESTON_VERSION"; \
		fi; \
		which weston; \
		ls -lh $$(which weston); \
	else \
		echo "$(RED)[ERROR]$(NC) �?Weston 未安�?; \
	fi
	@echo ""
	@echo "$(CYAN)DRM 设备状�?$(NC)"
	@ls -la /dev/dri/ 2>/dev/null || echo "$(YELLOW)DRM 设备不存�?(NC)"

# 备份当前 Weston 配置
backup-current-weston:
	@echo "$(BLUE)[INFO]$(NC) 备份当前 Weston 配置..."
	@BACKUP_DATE=$$(date +%Y%m%d_%H%M%S); \
	sudo mkdir -p /opt/backup/weston; \
	if [ -d "/etc/xdg/weston" ]; then \
		sudo cp -r /etc/xdg/weston /opt/backup/weston/weston-etc-$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 配置已备�? /opt/backup/weston/weston-etc-$$BACKUP_DATE"; \
	fi; \
	if [ -f "/root/.config/weston.ini" ]; then \
		sudo cp /root/.config/weston.ini /opt/backup/weston/weston.ini.$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 用户配置已备�?; \
	fi; \
	if [ -f "/etc/systemd/system/weston.service" ]; then \
		sudo cp /etc/systemd/system/weston.service /opt/backup/weston/weston.service.$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 服务文件已备�?; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 备份完成"

# 卸载当前 Weston
uninstall-current-weston:
	@echo "$(BLUE)[INFO]$(NC) 停止并卸载当�?Weston..."
	@# 停止所有服�?
	@sudo systemctl stop bamboo-cpp-lvgl 2>/dev/null || true
	@sudo systemctl stop weston.service 2>/dev/null || true
	@sudo systemctl stop weston 2>/dev/null || true
	@sudo pkill -9 weston 2>/dev/null || true
	@sleep 2
	@# 卸载 Weston（如果是 APT 安装�?
	@if dpkg -l | grep -q "^ii.*weston"; then \
		echo "$(BLUE)[INFO]$(NC) 卸载 APT 安装�?Weston..."; \
		sudo apt-get remove --purge -y weston libweston-* 2>/dev/null || true; \
		sudo apt-get autoremove -y; \
	fi
	@# 删除手动编译�?Weston 文件
	@echo "$(BLUE)[INFO]$(NC) 删除手动编译�?Weston 文件..."
	@sudo rm -f /usr/bin/weston* 2>/dev/null || true
	@sudo rm -f /usr/local/bin/weston* 2>/dev/null || true
	@sudo rm -rf /usr/lib/weston 2>/dev/null || true
	@sudo rm -rf /usr/local/lib/weston 2>/dev/null || true
	@sudo rm -rf /usr/lib/aarch64-linux-gnu/weston 2>/dev/null || true
	@sudo rm -rf /usr/share/weston 2>/dev/null || true
	@sudo rm -rf /usr/local/share/weston 2>/dev/null || true
	@sudo rm -f /usr/lib/aarch64-linux-gnu/libweston-*.so* 2>/dev/null || true
	@sudo rm -f /usr/local/lib/libweston-*.so* 2>/dev/null || true
	@sudo ldconfig
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 已卸�?

# 安装 Weston 12 编译依赖
install-weston12-build-deps:
	@echo "$(BLUE)[INFO]$(NC) 安装 Weston 12 编译依赖..."
	@sudo apt-get update
	@sudo apt-get install -y \
		build-essential \
		meson \
		ninja-build \
		pkg-config \
		libwayland-dev \
		wayland-protocols \
		libxkbcommon-dev \
		libpixman-1-dev \
		libdrm-dev \
		libgbm-dev \
		libinput-dev \
		libudev-dev \
		libjpeg-dev \
		libpng-dev \
		libwebp-dev \
		libegl1-mesa-dev \
		libgles2-mesa-dev \
		libxcb1-dev \
		libxcb-composite0-dev \
		libxcb-xfixes0-dev \
		libdbus-1-dev \
		libsystemd-dev \
		libpam0g-dev \
		liblcms2-dev \
		libcolord-dev \
		libxml2-dev \
		libcairo2-dev \
		libpango1.0-dev
	@echo "$(GREEN)[SUCCESS]$(NC) 依赖安装完成"

# 下载 Weston 12.0.0 源码
download-weston12:
	@echo "$(BLUE)[INFO]$(NC) 下载 Weston 12.0.0 源码..."
	@sudo mkdir -p /tmp/weston12-build
	@cd /tmp/weston12-build && \
		if [ ! -f "weston-12.0.0.tar.xz" ]; then \
			wget -q --show-progress https://wayland.freedesktop.org/releases/weston-12.0.0.tar.xz || \
			wget -q --show-progress https://gitlab.freedesktop.org/wayland/weston/-/archive/12.0.0/weston-12.0.0.tar.gz -O weston-12.0.0.tar.xz; \
		fi
	@echo "$(BLUE)[INFO]$(NC) 解压源码..."
	@cd /tmp/weston12-build && \
		rm -rf weston-12.0.0 && \
		tar -xf weston-12.0.0.tar.xz
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12.0.0 源码已准�?

# 编译 Weston 12
compile-weston12: install-weston12-build-deps download-weston12
	@echo "$(CYAN)[COMPILE]$(NC) 开始编�?Weston 12.0.0 (预计 15-30 分钟)..."
	@cd /tmp/weston12-build/weston-12.0.0 && \
		echo "$(BLUE)[INFO]$(NC) 配置 Meson..." && \
		rm -rf build && \
		meson setup build \
			--prefix=/usr \
			--libexecdir=/usr/lib/weston \
			--buildtype=release \
			-Dbackend-drm=true \
			-Dbackend-wayland=false \
			-Dbackend-x11=false \
			-Dbackend-rdp=false \
			-Dbackend-pipewire=false \
			-Drenderer-gl=true \
			-Dxwayland=false \
			-Dshell-desktop=true \
			-Dshell-fullscreen=true \
			-Dshell-ivi=false \
			-Dshell-kiosk=true \
			-Dcolor-management-lcms=false \
			-Dsystemd=true \
			-Dremoting=false \
			-Dpipewire=false \
			-Dbackend-drm-screencast-vaapi=false \
			-Ddemo-clients=false \
			-Dsimple-clients=[] \
			-Dresize-pool=true && \
		echo "$(BLUE)[INFO]$(NC) 开始编�?(使用 $(shell nproc) 个核�?..." && \
		meson compile -C build -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12.0.0 编译完成"

# 安装 Weston 12
install-weston12: compile-weston12
	@echo "$(BLUE)[INFO]$(NC) 安装 Weston 12..."
	@cd /tmp/weston12-build/weston-12.0.0/build && \
		sudo meson install
	@sudo ldconfig
	@echo "$(BLUE)[INFO]$(NC) 验证安装..."
	@weston --version
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 安装完成"

# 配置 Weston 12
configure-weston12:
	@echo "$(BLUE)[INFO]$(NC) 配置 Weston 12..."
	@sudo mkdir -p /etc/xdg/weston
	@echo "[core]" | sudo tee /etc/xdg/weston/weston.ini > /dev/null
	@echo "backend=drm-backend.so" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "idle-time=0" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "require-input=false" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "use-pixman=true" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "[shell]" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "locking=false" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "panel-position=none" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "background-color=0xff000000" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "[output]" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "name=all" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "mode=preferred" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "transform=normal" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "[libinput]" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "enable-tap=true" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "touchscreen_calibrator=true" | sudo tee -a /etc/xdg/weston/weston.ini > /dev/null
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 配置文件已创�? /etc/xdg/weston/weston.ini"

# 创建系统 Weston systemd 服务（使�?Nvidia Weston 13�?
setup-weston-service:
	@echo "$(BLUE)[INFO]$(NC) 创建系统 Weston systemd 服务..."
	@echo "[Unit]" | sudo tee /etc/systemd/system/weston.service > /dev/null
	@echo "Description=Weston Wayland Compositor (Nvidia Jetson)" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Documentation=man:weston(1) man:weston.ini(5)" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "After=systemd-user-sessions.service multi-user.target" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "[Service]" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Type=simple" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "User=root" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Environment=\"XDG_RUNTIME_DIR=/run/user/0\"" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Environment=\"WAYLAND_DISPLAY=wayland-1\"" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Environment=\"XDG_SESSION_TYPE=wayland\"" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Environment=\"EGL_PLATFORM=wayland\"" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Environment=\"__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl\"" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "ExecStartPre=/bin/mkdir -p /run/user/0" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "ExecStartPre=/bin/chmod 0700 /run/user/0" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "ExecStartPre=/bin/sh -c 'rm -f /run/user/0/wayland-*'" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "ExecStartPre=/sbin/modprobe nvidia-drm modeset=1" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "ExecStart=/usr/bin/weston \\" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "    --backend=drm-backend.so \\" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "    --idle-time=0 \\" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "    --use-pixman \\" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "    --log=/var/log/weston.log" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "Restart=always" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "RestartSec=3" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "TimeoutStartSec=60" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "StandardOutput=journal" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "StandardError=journal" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "SyslogIdentifier=weston" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "[Install]" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@echo "WantedBy=multi-user.target" | sudo tee -a /etc/systemd/system/weston.service > /dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable weston.service
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 服务已配置并启用"

# 创建 Weston 12 systemd 服务
setup-weston12-service:
	@echo "$(BLUE)[INFO]$(NC) 创建 Weston 12 systemd 服务..."
	@echo "[Unit]" | sudo tee /etc/systemd/system/weston12.service > /dev/null
	@echo "Description=Weston 12 Wayland Compositor (Jetson Optimized)" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Documentation=man:weston(1) man:weston.ini(5)" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "After=systemd-user-sessions.service multi-user.target" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "[Service]" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Type=simple" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "User=root" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Environment=\"XDG_RUNTIME_DIR=/run/user/0\"" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Environment=\"WAYLAND_DISPLAY=wayland-0\"" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Environment=\"XDG_SESSION_TYPE=wayland\"" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Environment=\"EGL_PLATFORM=wayland\"" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Environment=\"__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl\"" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "ExecStartPre=/bin/mkdir -p /run/user/0" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "ExecStartPre=/bin/chmod 0700 /run/user/0" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "ExecStartPre=/bin/sh -c 'rm -f /run/user/0/wayland-*'" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "ExecStart=/usr/bin/weston \\" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "    --backend=drm-backend.so \\" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "    --idle-time=0 \\" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "    --use-pixman \\" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "    --log=/var/log/weston12.log" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "Restart=always" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "RestartSec=3" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "TimeoutStartSec=60" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "StandardOutput=journal" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "StandardError=journal" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "SyslogIdentifier=weston12" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "[Install]" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@echo "WantedBy=multi-user.target" | sudo tee -a /etc/systemd/system/weston12.service > /dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable weston12.service
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 服务已配置并启用"

# 启动 Weston 12
start-weston12:
	@echo "$(BLUE)[INFO]$(NC) 启动 Weston 12..."
	@sudo systemctl start weston12.service
	@sleep 3
	@if sudo systemctl is-active --quiet weston12.service; then \
		echo "$(GREEN)[SUCCESS]$(NC) �?Weston 12 启动成功"; \
		echo "$(CYAN)Wayland Socket:$(NC)"; \
		ls -la /run/user/0/wayland-* 2>/dev/null || echo "$(YELLOW)等待 socket 创建...$(NC)"; \
		sleep 2; \
		ls -la /run/user/0/wayland-* 2>/dev/null || echo "$(RED)Socket 未创建，查看日志$(NC)"; \
	else \
		echo "$(RED)[ERROR]$(NC) �?Weston 12 启动失败"; \
		echo "$(CYAN)查看最�?30 行日�?$(NC)"; \
		sudo journalctl -u weston12.service -n 30 --no-pager; \
		exit 1; \
	fi

# 停止 Weston 12
stop-weston12:
	@echo "$(BLUE)[INFO]$(NC) 停止 Weston 12..."
	@sudo systemctl stop weston12.service
	@sudo pkill -9 weston 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 已停�?

# 查看 Weston 12 状�?
weston12-status:
	@echo "$(CYAN)=== Weston 12 服务状�?===$(NC)"
	@sudo systemctl status weston12.service --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Weston 进程 ===$(NC)"
	@ps aux | grep weston | grep -v grep || echo "�?Weston 进程"
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "�?Wayland socket"
	@echo ""
	@echo "$(CYAN)=== DRM 设备 ===$(NC)"
	@ls -la /dev/dri/ 2>/dev/null || echo "DRM 设备不存�?

# 查看 Weston 12 日志
weston12-logs:
	@echo "$(CYAN)=== Weston 12 systemd 日志 (最�?100 �? ===$(NC)"
	@sudo journalctl -u weston12.service -n 100 --no-pager
	@echo ""
	@echo "$(CYAN)=== Weston 12 运行日志 ===$(NC)"
	@if [ -f /var/log/weston12.log ]; then \
		sudo tail -100 /var/log/weston12.log; \
	else \
		echo "日志文件 /var/log/weston12.log 不存�?; \
	fi

# 测试 Weston 12
test-weston12:
	@echo "$(BLUE)[INFO]$(NC) 测试 Weston 12..."
	@echo "$(CYAN)1. 检查版�?$(NC)"
	@weston --version
	@echo ""
	@echo "$(CYAN)2. 检查服务状�?$(NC)"
	@sudo systemctl is-active weston12.service && echo "$(GREEN)�?服务运行�?(NC)" || echo "$(RED)�?服务未运�?(NC)"
	@echo ""
	@echo "$(CYAN)3. 检�?Wayland socket:$(NC)"
	@ls -la /run/user/0/wayland-* 2>/dev/null && echo "$(GREEN)�?Socket 存在$(NC)" || echo "$(RED)�?Socket 不存�?(NC)"
	@echo ""
	@echo "$(CYAN)4. 检查配置文�?$(NC)"
	@if [ -f /etc/xdg/weston/weston.ini ]; then \
		echo "$(GREEN)�?配置文件存在$(NC)"; \
		echo "内容预览:"; \
		head -20 /etc/xdg/weston/weston.ini; \
	else \
		echo "$(RED)�?配置文件不存�?(NC)"; \
	fi

# 🚀 一键降级到 Weston 12（推荐使用）
downgrade-to-weston12:
	@echo "$(CYAN)======================================$(NC)"
	@echo "$(CYAN)  Weston 12 完整降级流程$(NC)"
	@echo "$(CYAN)======================================$(NC)"
	@echo ""
	@echo "$(BLUE)[步骤 1/9]$(NC) 检查当前版�?.."
	@$(MAKE) check-weston-version
	@echo ""
	@echo "$(BLUE)[步骤 2/9]$(NC) 备份当前配置..."
	@$(MAKE) backup-current-weston
	@echo ""
	@echo "$(BLUE)[步骤 3/9]$(NC) 卸载当前 Weston..."
	@$(MAKE) uninstall-current-weston
	@echo ""
	@echo "$(BLUE)[步骤 4/9]$(NC) 编译 Weston 12 (需�?15-30 分钟)..."
	@$(MAKE) install-weston12
	@echo ""
	@echo "$(BLUE)[步骤 5/9]$(NC) 配置 Weston 12..."
	@$(MAKE) configure-weston12
	@echo ""
	@echo "$(BLUE)[步骤 6/9]$(NC) 创建 systemd 服务..."
	@$(MAKE) setup-weston12-service
	@echo ""
	@echo "$(BLUE)[步骤 7/9]$(NC) 启动 Weston 12..."
	@$(MAKE) start-weston12
	@echo ""
	@echo "$(BLUE)[步骤 8/9]$(NC) 测试 Weston 12..."
	@$(MAKE) test-weston12
	@echo ""
	@echo "$(BLUE)[步骤 9/9]$(NC) 清理临时文件..."
	@sudo rm -rf /tmp/weston12-build
	@echo ""
	@echo "$(GREEN)======================================$(NC)"
	@echo "$(GREEN)  ✓✓�?Weston 12 降级完成�?(NC)"
	@echo "$(GREEN)======================================$(NC)"
	@echo ""
	@echo "$(CYAN)下一步操�?$(NC)"
	@echo "  1. 查看 Weston 12 状�? $(YELLOW)make weston12-status$(NC)"
	@echo "  2. 查看 Weston 12 日志: $(YELLOW)make weston12-logs$(NC)"
	@echo "  3. 重新部署应用: $(YELLOW)make redeploy$(NC)"
	@echo "  4. 查看应用日志: $(YELLOW)sudo journalctl -u bamboo-cpp-lvgl -f$(NC)"
	@echo ""
	@echo "$(CYAN)如果遇到问题:$(NC)"
	@echo "  - 查看服务状�? $(YELLOW)make weston12-status$(NC)"
	@echo "  - 重启 Weston 12: $(YELLOW)sudo systemctl restart weston12$(NC)"
	@echo "  - 恢复备份: 查看 /opt/backup/weston/"
	@echo ""

# 兼容性别名（更新为使�?Weston 12�?
start-weston: start-weston12
stop-weston: stop-weston12
weston-status: weston12-status

check-wayland:
	@echo "$(BLUE)[INFO]$(NC) 检查Wayland环境（Sway�?.."
	@echo -n "Sway服务状�? "
	@sudo systemctl is-active sway-wayland.service 2>/dev/null || echo "未运�?
	@echo -n "Wayland socket: "
	@ls /run/user/0/wayland-* 2>/dev/null && echo "存在" || echo "不存�?
	@echo -n "Wayland�? "
	@pkg-config --exists wayland-client && echo "已安�? || echo "未安�?
	@echo -n "EGL�? "
	@ldconfig -p | grep -q "libEGL" && echo "已安�? || echo "未安�?

install-system-deps:
	@echo "$(BLUE)[INFO]$(NC) 安装系统依赖..."
	@sudo apt-get update
	@sudo apt-get install -y \
		build-essential \
		cmake \
		pkg-config \
		git \
		wget \
		libopencv-dev \
		libgstreamer1.0-dev \
		libgstreamer-plugins-base1.0-dev \
		libmodbus-dev \
		libcurl4-openssl-dev \
		libjson-c-dev \
		libsystemd-dev \
		libsdl2-dev \
		libfreetype6-dev \
		libpng-dev \
		libjpeg-dev \
		libfontconfig1-dev \
		libharfbuzz-dev \
		libdrm-dev \
		libgbm-dev \
		libegl1-mesa-dev \
		libgles2-mesa-dev \
		mesa-common-dev \
		mutter \
		gnome-session \
		libwayland-dev \
		libwayland-egl1 \
		wayland-protocols \
		libxkbcommon-dev \
		dbus-x11
	@if lspci | grep -i nvidia >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) 检测到NVIDIA GPU，检查CUDA环境..."; \
		if [ -d "/usr/local/cuda" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) CUDA环境已安�?; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) CUDA环境未安装，请手动安装CUDA和TensorRT"; \
		fi \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 系统依赖安装完成"

install-lvgl:
	@echo "$(CYAN)[LVGL]$(NC) 检查LVGL v9安装状�?.."
	@LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null || echo "not_found"); \
	if [ "$$LVGL_VERSION" = "not_found" ] || [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" != "9" ]; then \
		echo "$(BLUE)[INFO]$(NC) LVGL v9未找�?(当前版本: $$LVGL_VERSION)，开始从源码编译安装..."; \
		$(MAKE) build-lvgl-from-source; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL v9已安�?(版本: $$LVGL_VERSION)"; \
	fi

build-lvgl-from-source:
	@echo "$(CYAN)[LVGL]$(NC) === 完全手动安装LVGL v9.1 ==="
	@echo "$(BLUE)[INFO]$(NC) [1/8] 清理旧文�?.."
	@sudo rm -rf /usr/local/include/lvgl 2>/dev/null || true
	@sudo rm -rf /usr/local/lib/liblvgl* 2>/dev/null || true
	@sudo rm -rf /usr/local/lib/pkgconfig/lvgl.pc 2>/dev/null || true
	@sudo rm -rf /tmp/lvgl 2>/dev/null || true
	@sudo ldconfig 2>/dev/null || true
	@echo "$(BLUE)[INFO]$(NC) [2/8] 安装依赖..."
	@sudo apt-get update -qq
	@sudo apt-get install -y git cmake build-essential
	@echo "$(BLUE)[INFO]$(NC) [3/8] 下载LVGL v9.1..."
	@cd /tmp && rm -rf lvgl && git clone --depth 1 --branch release/v9.1 https://github.com/lvgl/lvgl.git
	@echo "$(BLUE)[INFO]$(NC) [4/8] 创建配置文件..."
	@cd /tmp/lvgl && \
	echo "#ifndef LV_CONF_H" > lv_conf.h && \
	echo "#define LV_CONF_H" >> lv_conf.h && \
	echo "#define LV_COLOR_DEPTH 32" >> lv_conf.h && \
	echo "#define LV_FONT_MONTSERRAT_12 1" >> lv_conf.h && \
	echo "#define LV_FONT_MONTSERRAT_14 1" >> lv_conf.h && \
	echo "#define LV_FONT_MONTSERRAT_16 1" >> lv_conf.h && \
	echo "#define LV_FONT_MONTSERRAT_20 1" >> lv_conf.h && \
	echo "#define LV_FONT_MONTSERRAT_24 1" >> lv_conf.h && \
	echo "#define LV_USE_FREETYPE 0" >> lv_conf.h && \
	echo "#define LV_USE_LIBPNG 0" >> lv_conf.h && \
	echo "#define LV_USE_LIBJPEG_TURBO 0" >> lv_conf.h && \
	echo "#endif" >> lv_conf.h
	@echo "$(BLUE)[INFO]$(NC) [5/8] 配置CMake..."
	@cd /tmp/lvgl && mkdir -p build && cd build && \
	cmake .. \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DLV_CONF_PATH=../lv_conf.h \
		-DBUILD_SHARED_LIBS=ON \
		-DLV_USE_FREETYPE=OFF
	@echo "$(BLUE)[INFO]$(NC) [6/8] 编译LVGL..."
	@cd /tmp/lvgl/build && make -j4
	@echo "$(BLUE)[INFO]$(NC) [7/8] 安装文件..."
	@cd /tmp/lvgl/build && sudo make install
	@echo "$(BLUE)[INFO]$(NC) 手动确保头文件安�?.."
	@sudo mkdir -p /usr/local/include/lvgl
	@cd /tmp/lvgl && sudo cp -r src/* /usr/local/include/lvgl/
	@cd /tmp/lvgl && sudo cp lvgl.h /usr/local/include/lvgl/
	@cd /tmp/lvgl && sudo cp lv_conf.h /usr/local/include/
	@echo "$(BLUE)[INFO]$(NC) [8/8] 创建pkg-config文件..."
	@echo "prefix=/usr/local" | sudo tee /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "exec_prefix=\$${prefix}" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "libdir=\$${exec_prefix}/lib" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "includedir=\$${prefix}/include" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "Name: LVGL" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "Description: Light and Versatile Graphics Library" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "Version: 9.1.0" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "Libs: -L\$${libdir} -llvgl" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@echo "Cflags: -I\$${includedir}/lvgl -I\$${includedir}" | sudo tee -a /usr/local/lib/pkgconfig/lvgl.pc > /dev/null
	@sudo ldconfig
	@echo ""
	@echo "$(CYAN)[VERIFY]$(NC) === 验证安装 ==="
	@echo -n "$(BLUE)[INFO]$(NC) 头文�? "
	@ls /usr/local/include/lvgl/lvgl.h >/dev/null 2>&1 && echo "$(GREEN)�?(NC)" || (echo "$(RED)�?失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) 库文�? "
	@ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1 && echo "$(GREEN)�?(NC)" || (echo "$(RED)�?失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) pkg-config: "
	@PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl && echo "$(GREEN)�?(NC)" || (echo "$(RED)�?失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) v9 API: "
	@grep -q "lv_display_create" /usr/local/include/lvgl/lvgl.h && echo "$(GREEN)�?(NC)" || echo "$(YELLOW)�?未检测到但可能正�?(NC)"
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) === LVGL v9.1安装完成 ==="
	@rm -rf /tmp/lvgl

# 安装LVGL v9的快速命�?
install-lvgl9: build-lvgl-from-source
	@echo "$(GREEN)[SUCCESS]$(NC) LVGL v9.3安装完成，系统已准备就绪"

# 自动检查和安装LVGL v9（编译前自动执行�?
install-lvgl9-auto:
	@echo "$(CYAN)[AUTO-INSTALL]$(NC) === 智能检测LVGL v9安装状�?==="
	@echo "$(BLUE)[INFO]$(NC) 正在检测LVGL v9安装状�?.."
	@LVGL_INSTALLED=false; \
	LVGL_VERSION_OK=false; \
	LVGL_API_OK=false; \
	if PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null); \
		echo "$(BLUE)[INFO]$(NC) 发现已安装的LVGL版本: $$LVGL_VERSION"; \
		if [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" = "9" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL主版本为v9 �?; \
			LVGL_VERSION_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) LVGL版本不是v9 (当前: $$LVGL_VERSION)"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) pkg-config未找到LVGL"; \
	fi; \
	if [ -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL头文件存�?�?; \
		if grep -q "lv_display_create\|lv_disp_create" /usr/local/include/lvgl/lvgl.h 2>/dev/null; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL v9 API可用 �?; \
			LVGL_API_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) 未检测到LVGL v9 API"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) LVGL头文件不存在"; \
	fi; \
	if ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL库文件存�?�?; \
		LVGL_INSTALLED=true; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) LVGL库文件不存在"; \
	fi; \
	if [ "$$LVGL_INSTALLED" = "true" ] && [ "$$LVGL_VERSION_OK" = "true" ] && [ "$$LVGL_API_OK" = "true" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) === LVGL v9已正确安装，跳过安装步骤 ==="; \
	else \
		echo "$(CYAN)[INSTALL]$(NC) === 需要安装LVGL v9.1 ==="; \
		$(MAKE) build-lvgl-from-source; \
	fi

# === C++系统构建 ===
build-system:
	@echo "$(CYAN)[C++ LVGL]$(NC) 开始构建C++ LVGL一体化系统..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) C++ LVGL系统构建完成"

build-debug:
	@echo "$(CYAN)[C++ LVGL]$(NC) 构建调试版本..."
	@mkdir -p $(BUILD_DIR)_debug
	@cd $(BUILD_DIR)_debug && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DENABLE_AI_OPTIMIZATION=ON \
		-DENABLE_MODBUS=ON \
		-DENABLE_LVGL=ON
	@cd $(BUILD_DIR)_debug && make -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) 调试版本构建完成"

# 🔧 新增：编译自定义YOLO解析�?
compile-yolo-lib:
	@echo "$(BLUE)[INFO]$(NC) 🔧 编译自定义YOLO解析�?.."
	@sudo mkdir -p $(INSTALL_DIR)/lib
	@g++ -shared -fPIC \
		-I/opt/nvidia/deepstream/deepstream/sources/includes \
		-I/usr/local/cuda/include \
		cpp_backend/src/deepstream/nvdsinfer_yolo_bamboo.cpp \
		-o libnvdsinfer_yolo_bamboo.so
	@sudo cp libnvdsinfer_yolo_bamboo.so $(INSTALL_DIR)/lib/
	@sudo chmod 755 $(INSTALL_DIR)/lib/libnvdsinfer_yolo_bamboo.so
	@echo "$(GREEN)[SUCCESS]$(NC) �?YOLO解析库编译部署完�?

# === 系统安装 ===
install-system: compile-yolo-lib
	@echo "$(BLUE)[INFO]$(NC) 安装C++ LVGL系统�?(INSTALL_DIR)..."
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "$(RED)[ERROR]$(NC) 构建目录不存在，请先运行 make build-system"; \
		exit 1; \
	fi
	@sudo mkdir -p $(INSTALL_DIR)
	@cd $(BUILD_DIR) && sudo make install
	@sudo mkdir -p $(INSTALL_DIR)/logs
	@sudo mkdir -p $(INSTALL_DIR)/backup
	@echo "$(BLUE)[INFO]$(NC) 复制配置文件..."
	@sudo mkdir -p $(INSTALL_DIR)/config
	@sudo cp -r config/* $(INSTALL_DIR)/config/ 2>/dev/null || true
	@echo "$(BLUE)[INFO]$(NC) 确保nvinfer配置文件和标签文件存�?.."
	@if [ -f "config/nvinfer_config.txt" ]; then \
		sudo cp config/nvinfer_config.txt $(INSTALL_DIR)/config/; \
		echo "$(GREEN)[SUCCESS]$(NC) nvinfer配置文件已复�?; \
	fi
	@if [ -f "config/labels.txt" ]; then \
		sudo cp config/labels.txt $(INSTALL_DIR)/config/; \
		echo "$(GREEN)[SUCCESS]$(NC) 标签文件已复�?; \
	fi
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/logs
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/backup
	@echo "$(GREEN)[SUCCESS]$(NC) 系统安装完成"

# === 配置设置 ===
setup-config:
	@echo "$(BLUE)[INFO]$(NC) 设置系统配置..."
	@sudo mkdir -p $(INSTALL_DIR)/etc/bamboo-recognition
	@if [ ! -f "$(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml" ]; then \
		sudo cp config/system_config.yaml $(INSTALL_DIR)/etc/bamboo-recognition/ 2>/dev/null || \
		echo "# C++ LVGL一体化系统配置" | sudo tee $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml >/dev/null; \
	fi
	@sudo chmod 644 $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml
	@echo "$(GREEN)[SUCCESS]$(NC) 配置设置完成"

# === 服务管理 ===
install-service: setup-config
	@echo "$(BLUE)[INFO]$(NC) 安装systemd服务..."
	@if [ -f "$(BUILD_DIR)/bamboo-cpp-lvgl.service" ]; then \
		sudo cp $(BUILD_DIR)/bamboo-cpp-lvgl.service /etc/systemd/system/; \
	else \
		echo "$(RED)[ERROR]$(NC) 服务文件未生成，请检查CMake构建"; \
		exit 1; \
	fi
	@sudo systemctl daemon-reload
	@echo "$(GREEN)[SUCCESS]$(NC) 服务安装完成"

enable-service:
	@sudo systemctl enable $(SERVICE_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) 服务已启用开机自�?

disable-service:
	@sudo systemctl disable $(SERVICE_NAME)
	@echo "$(BLUE)[INFO]$(NC) 服务已禁用开机自�?

start:
	@echo "$(BLUE)[INFO]$(NC) 启动$(SERVICE_NAME)服务..."
	@sudo systemctl start $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) 服务启动成功"; \
	else \
		echo "$(RED)[ERROR]$(NC) 服务启动失败，请查看日志"; \
		exit 1; \
	fi

stop:
	@echo "$(BLUE)[INFO]$(NC) 停止$(SERVICE_NAME)服务..."
	@sudo systemctl stop $(SERVICE_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) 服务已停�?

restart:
	@echo "$(BLUE)[INFO]$(NC) 重启$(SERVICE_NAME)服务..."
	@sudo systemctl restart $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) 服务重启成功"; \
	else \
		echo "$(RED)[ERROR]$(NC) 服务重启失败，请查看日志"; \
	fi

status:
	@echo "$(CYAN)=== 服务状�?===$(NC)"
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l
	@echo ""
	@echo "$(CYAN)=== 系统资源 ===$(NC)"
	@ps aux | grep $(BINARY_NAME) | grep -v grep || echo "进程未运�?

logs:
	@echo "$(CYAN)=== 实时日志 (按Ctrl+C退�? ===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) -f --no-hostname

logs-recent:
	@echo "$(CYAN)=== 最近日�?===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) --no-hostname -n 50

# === 测试和维�?===
test-system:
	@echo "$(BLUE)[INFO]$(NC) 测试模式运行系统..."
	@if [ ! -f "$(INSTALL_DIR)/bin/$(BINARY_NAME)" ]; then \
		echo "$(RED)[ERROR]$(NC) 系统未安装，请先运行 make install"; \
		exit 1; \
	fi
	@cd $(INSTALL_DIR)/bin && sudo ./$(BINARY_NAME) --test --verbose --config $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml

test:
	@echo "$(BLUE)[INFO]$(NC) 运行系统测试..."
	@if [ -f "cpp_backend/tests/run_tests.sh" ]; then \
		cd cpp_backend && bash tests/run_tests.sh; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) 测试脚本不存�?; \
	fi

backup:
	@echo "$(BLUE)[INFO]$(NC) 创建系统备份..."
	@BACKUP_NAME="bamboo-system-backup-$$(date +%Y%m%d-%H%M%S)"; \
	sudo mkdir -p /opt/backup; \
	sudo tar -czf /opt/backup/$$BACKUP_NAME.tar.gz \
		-C $(INSTALL_DIR) . \
		--exclude=logs \
		--exclude=backup; \
	echo "$(GREEN)[SUCCESS]$(NC) 备份已创�? /opt/backup/$$BACKUP_NAME.tar.gz"

# 快速重新部署（跳过依赖检查）
redeploy: stop clean build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 系统重新部署完成�?

# 完整重新部署（包括依赖检查）
full-redeploy: stop install-deps build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 系统完整重新部署完成�?

# 智能重新部署（仅在必要时安装依赖�?
smart-redeploy: stop check-deps-if-needed build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 智能重新部署完成�?

# 检查依赖是否需要重新安�?
check-deps-if-needed:
	@echo "$(BLUE)[INFO]$(NC) 检查是否需要重新安装依�?.."
	@NEED_DEPS=false; \
	if ! PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL未找到，需要安装依�?; \
		NEED_DEPS=true; \
	elif [ "$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null | cut -d. -f1)" != "9" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL版本不是v9，需要更�?; \
		NEED_DEPS=true; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) 依赖已满足，跳过安装步骤"; \
	fi; \
	if [ "$$NEED_DEPS" = "true" ]; then \
		echo "$(CYAN)[INSTALL]$(NC) 安装缺失的依�?.."; \
		$(MAKE) install-deps; \
	fi

# === 清理 ===
clean:
	@echo "$(BLUE)[INFO]$(NC) 清理构建目录..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_DIR)_debug
	@echo "$(GREEN)[SUCCESS]$(NC) 清理完成"

uninstall:
	@echo "$(BLUE)[INFO]$(NC) 卸载系统..."
	@sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true
	@sudo systemctl disable $(SERVICE_NAME) 2>/dev/null || true
	@sudo rm -f /etc/systemd/system/$(SERVICE_NAME).service
	@sudo systemctl daemon-reload
	@sudo rm -rf $(INSTALL_DIR)
	@echo "$(GREEN)[SUCCESS]$(NC) 系统已卸�?

# === 单进程统一架构 ===
unified: unified-build
	@echo "$(GREEN)[SUCCESS]$(NC) 单进程统一架构构建完成"

unified-build:
	@echo "$(CYAN)[UNIFIED]$(NC) 构建单进程LVGL+GStreamer统一架构..."
	@PKG_CONFIG_PATH=/usr/local/lib/pkgconfig g++ -o simple_unified_main \
		simple_unified_main.cpp \
		$$(pkg-config --cflags --libs lvgl) \
		$$(pkg-config --cflags --libs gstreamer-1.0) \
		$$(pkg-config --cflags --libs gstreamer-app-1.0) \
		-lEGL -lpthread \
		-std=c++17 -O2 -DENABLE_LVGL=1
	@echo "$(GREEN)[SUCCESS]$(NC) 统一架构编译完成: ./simple_unified_main"

unified-run:
	@echo "$(BLUE)[INFO]$(NC) 运行单进程统一架构..."
	@if [ ! -f "./simple_unified_main" ]; then \
		echo "$(RED)[ERROR]$(NC) 统一架构可执行文件不存在，请先运�?make unified"; \
		exit 1; \
	fi
	@sudo ./simple_unified_main

unified-test:
	@echo "$(BLUE)[INFO]$(NC) 测试单进程统一架构..."
	@echo "$(CYAN)检查EGL环境...$(NC)"
	@if command -v eglinfo >/dev/null 2>&1; then \
		eglinfo | head -10; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) eglinfo未安装，跳过EGL检�?; \
	fi
	@echo "$(CYAN)检查DRM设备...$(NC)"
	@ls -la /dev/dri/ || echo "$(YELLOW)[WARNING]$(NC) DRM设备不可�?
	@echo "$(CYAN)检查摄像头设备...$(NC)"
	@ls -la /dev/video* || echo "$(YELLOW)[WARNING]$(NC) 摄像头设备不可用"
	@echo "$(GREEN)[SUCCESS]$(NC) 环境检查完成，运行统一架构..."
	@$(MAKE) unified-run

unified-clean:
	@echo "$(BLUE)[INFO]$(NC) 清理统一架构构建文件..."
	@rm -f simple_unified_main
	@echo "$(GREEN)[SUCCESS]$(NC) 清理完成"

# === 摄像头诊断工�?===
GSTREAMER_LIBS := $(shell pkg-config --cflags --libs gstreamer-1.0)
EGL_LIBS := -lEGL
PTHREAD_LIBS := -lpthread

camera-diag: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头诊断工�?.."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[RUNNING]$(NC) 运行摄像头诊�?.."
	sudo ./camera_diagnostics

camera-test: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头测试工�?.."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 测试摄像头访�?(sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_diagnostics test $(or $(SENSOR_ID),0)

camera-fix:
	@echo "$(CYAN)[FIXING]$(NC) 运行综合摄像头修复脚�?.."
	./deploy/scripts/camera_fix.sh

camera-fix-quick:
	@echo "$(BLUE)[INFO]$(NC) 应用快速摄像头修复..."
	@echo "1. 停止冲突进程..."
	-sudo pkill nvargus-daemon 2>/dev/null || true
	-sudo pkill gst-launch-1.0 2>/dev/null || true
	@echo "2. 重启nvargus-daemon..."
	-sudo systemctl restart nvargus-daemon 2>/dev/null || true
	@echo "3. 设置设备权限..."
	sudo chmod 666 /dev/video* 2>/dev/null || true
	sudo chmod 666 /dev/nvhost-* 2>/dev/null || true
	@echo "4. 设置EGL环境..."
	@echo "export EGL_PLATFORM=drm" >> ~/.bashrc
	@echo "export __EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl" >> ~/.bashrc
	@echo "$(GREEN)[SUCCESS]$(NC) 快速修复已应用，请运行 'source ~/.bashrc' 并重�?

camera-fix-test: test_camera_fix.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头修复测试工�?.."
	$(CXX) $(CXXFLAGS) -o camera_fix_test test_camera_fix.cpp $(GSTREAMER_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 运行摄像头修复测�?(sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_fix_test $(or $(SENSOR_ID),0)

# NVIDIA-DRM Migration and Validation
enable-nvidia-drm:
	@echo "$(BLUE)[INFO]$(NC) 启用NVIDIA-DRM驱动..."
	@chmod +x deploy/scripts/enable_nvidia_drm.sh
	@echo "$(YELLOW)[WARNING]$(NC) 此操作将修改系统驱动配置，请确认继续..."
	@read -p "继续启用NVIDIA-DRM? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	sudo deploy/scripts/enable_nvidia_drm.sh

force-nvidia-drm:
	@echo "$(BLUE)[INFO]$(NC) 强制迁移到NVIDIA-DRM驱动..."
	@chmod +x deploy/scripts/force_nvidia_drm.sh
	@echo "$(RED)[DANGER]$(NC) 此操作将强制修改系统驱动，可能影响图形显�?
	@echo "$(YELLOW)[WARNING]$(NC) 建议先备份重要数据，操作需要重启系�?
	@read -p "确认强制迁移到NVIDIA-DRM? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	sudo deploy/scripts/force_nvidia_drm.sh

nvidia-drm-test: nvidia_drm_migration_test.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建NVIDIA-DRM迁移验证工具..."
	$(CXX) $(CXXFLAGS) -o nvidia_drm_migration_test nvidia_drm_migration_test.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS) -ldrm -lm
	@echo "$(CYAN)[TESTING]$(NC) 运行NVIDIA-DRM迁移完整验证..."
	sudo ./nvidia_drm_migration_test

nvidia-drm-report:
	@echo "$(CYAN)[REPORT]$(NC) 生成NVIDIA-DRM迁移状态报�?.."
	@echo "=== NVIDIA-DRM 迁移状态报�?===" > nvidia_drm_status.txt
	@echo "生成时间: $$(date)" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 驱动模块状�?===" >> nvidia_drm_status.txt
	@lsmod | grep -E "nvidia|tegra|drm" >> nvidia_drm_status.txt 2>/dev/null || echo "未找到相关模�? >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== DRM设备状�?===" >> nvidia_drm_status.txt
	@ls -la /dev/dri/ >> nvidia_drm_status.txt 2>/dev/null || echo "DRM设备不存�? >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== EGL环境 ===" >> nvidia_drm_status.txt
	@echo "EGL_PLATFORM=$$EGL_PLATFORM" >> nvidia_drm_status.txt
	@echo "__EGL_VENDOR_LIBRARY_DIRS=$$__EGL_VENDOR_LIBRARY_DIRS" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 系统信息 ===" >> nvidia_drm_status.txt
	@uname -a >> nvidia_drm_status.txt
	@echo "$(GREEN)[SUCCESS]$(NC) 状态报告已保存�? nvidia_drm_status.txt"
	@cat nvidia_drm_status.txt

nvidia-drm-complete: nvidia-drm-test nvidia-drm-report
	@echo "$(GREEN)[COMPLETE]$(NC) NVIDIA-DRM迁移验证全部完成�?
	@echo "查看完整报告:"
	@echo "  验证报告: nvidia_drm_migration_report.txt"
	@echo "  状态报�? nvidia_drm_status.txt"

.PHONY: camera-diag camera-test camera-fix camera-fix-quick camera-fix-test enable-nvidia-drm force-nvidia-drm nvidia-drm-test nvidia-drm-report nvidia-drm-complete

# === 开发辅�?===
dev-run:
	@echo "$(BLUE)[INFO]$(NC) 开发模式直接运�?.."
	@if [ ! -f "$(BUILD_DIR)/bamboo_integrated" ]; then \
		echo "$(RED)[ERROR]$(NC) 可执行文件不存在，请先构建系�?; \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && sudo ./bamboo_integrated --verbose --config ../config/system_config.yaml

monitor:
	@echo "$(CYAN)=== 系统监控 (按Ctrl+C退�? ===$(NC)"
	@while true; do \
		clear; \
		echo "$(GREEN)时间: $$(date)$(NC)"; \
		echo "$(CYAN)服务状�?$(NC)"; \
		systemctl is-active $(SERVICE_NAME) 2>/dev/null || echo "未运�?; \
		echo "$(CYAN)系统资源:$(NC)"; \
		ps aux | grep $(BINARY_NAME) | grep -v grep | head -5 || echo "进程未运�?; \
		echo "$(CYAN)内存使用:$(NC)"; \
		free -h | head -2; \
		echo "$(CYAN)磁盘使用:$(NC)"; \
		df -h / | tail -1; \
		sleep 5; \
	done

# 确保依赖关系
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

.DEFAULT_GOAL := help
