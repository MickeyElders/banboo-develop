# AI绔瑰瓙璇嗗埆绯荤粺 - C++ LVGL涓€浣撳寲鏋勫缓鍜岄儴缃茶剼鏈?
# 鐗堟湰: 5.0.0 (C++ LVGL Integrated Architecture)
# C++鎺ㄧ悊鍚庣 + LVGL鐣岄潰 + Modbus閫氫俊鐨勫畬鏁翠竴浣撳寲绯荤粺

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

# === 绯荤粺閰嶇疆 ===
PROJECT_NAME := bamboo-recognition-system
VERSION := 5.0.0
INSTALL_DIR := /opt/bamboo-cut
SERVICE_NAME := bamboo-cpp-lvgl
BINARY_NAME := bamboo_integrated

# === C++ LVGL涓€浣撳寲鏋勫缓閰嶇疆 ===
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

# === 棰滆壊瀹氫箟 ===
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# === 涓昏鐩爣 ===

all: check-system auto-setup-environment install-deps build-system
	@echo "$(CYAN)=== AI绔瑰瓙璇嗗埆绯荤粺鏋勫缓瀹屾垚 (v$(VERSION)) ===$(NC)"
	@echo "$(GREEN)C++ LVGL Wayland涓€浣撳寲宸ヤ笟绾у祵鍏ュ紡鏋舵瀯$(NC)"
	@echo "浣跨敤 'make deploy' 瀹屾垚绯荤粺閮ㄧ讲"

install: all install-system install-service
	@echo "$(GREEN)=== 绯荤粺瀹夎瀹屾垚 ===$(NC)"
	@echo "鏈嶅姟鍚嶇О: $(SERVICE_NAME)"
	@echo "瀹夎鐩綍: $(INSTALL_DIR)"
	@echo "鍙墽琛屾枃浠? $(INSTALL_DIR)/bin/$(BINARY_NAME)"
	@echo "Wayland鐜: 宸茶嚜鍔ㄩ厤缃?
	@echo "浣跨敤 'make start' 鍚姩绯荤粺"

deploy: auto-setup-environment install enable-service start
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺閮ㄧ讲瀹屾垚锛?
	@echo "Wayland鐜锛圵eston 12锛夊凡鑷姩閰嶇疆骞跺惎鍔?

help:
	@echo "$(CYAN)===============================================$(NC)"
	@echo "$(CYAN)   AI绔瑰瓙璇嗗埆绯荤粺 C++ LVGL鏋勫缓绯荤粺 v$(VERSION)$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)蹇€熼儴缃插懡浠?$(NC)"
	@echo "  deploy           - 棣栨瀹屾暣閮ㄧ讲(鏋勫缓+瀹夎+鍚姩鏈嶅姟)"
	@echo "  redeploy         - 浠ｇ爜淇敼鍚庡揩閫熼噸鏂伴儴缃?璺宠繃渚濊禆妫€鏌?"
	@echo "  smart-redeploy   - 鏅鸿兘閲嶆柊閮ㄧ讲(浠呭湪蹇呰鏃跺畨瑁呬緷璧?"
	@echo "  full-redeploy    - 瀹屾暣閲嶆柊閮ㄧ讲(鍖呮嫭渚濊禆妫€鏌?"
	@echo "  backup           - 鍒涘缓褰撳墠绯荤粺澶囦唤"
	@echo "  test-system      - 娴嬭瘯妯″紡杩愯绯荤粺"
	@echo ""
	@echo "$(GREEN)鏋勫缓鍛戒护:$(NC)"
	@echo "  all              - 妫€鏌ョ郴缁?瀹夎渚濊禆+鏋勫缓绯荤粺"
	@echo "  build-system     - 鏋勫缓C++ LVGL绯荤粺"
	@echo "  build-debug      - 鏋勫缓璋冭瘯鐗堟湰"
	@echo "  clean            - 娓呯悊鏋勫缓鐩綍"
	@echo ""
	@echo "$(GREEN)鍗曡繘绋嬬粺涓€鏋舵瀯:$(NC)"
	@echo "  unified          - 鏋勫缓鍗曡繘绋婰VGL+GStreamer缁熶竴鏋舵瀯"
	@echo "  unified-run      - 杩愯缁熶竴鏋舵瀯"
	@echo "  unified-test     - 娴嬭瘯缁熶竴鏋舵瀯鍜岀幆澧?
	@echo "  unified-clean    - 娓呯悊缁熶竴鏋舵瀯鏋勫缓鏂囦欢"
	@echo ""
	@echo "$(GREEN)鎽勫儚澶磋瘖鏂伐鍏?$(NC)"
	@echo "  camera-diag      - 杩愯瀹屾暣鎽勫儚澶磋瘖鏂?
	@echo "  camera-test      - 娴嬭瘯鎽勫儚澶磋闂?(浣跨敤 SENSOR_ID=X 鎸囧畾sensor)"
	@echo "  camera-fix       - 杩愯缁煎悎浜や簰寮忔憚鍍忓ご淇鑴氭湰"
	@echo "  camera-fix-quick - 搴旂敤蹇€熼潪浜や簰寮忔憚鍍忓ご淇"
	@echo "  camera-fix-test  - 娴嬭瘯鎽勫儚澶翠慨澶嶅悗鍔熻兘 (浣跨敤 SENSOR_ID=X)"
	@echo ""
	@echo "$(GREEN)NVIDIA-DRM 杩佺Щ楠岃瘉:$(NC)"
	@echo "  enable-nvidia-drm- 鍚敤NVIDIA-DRM椹卞姩锛堟浛鎹egra_drm锛?
	@echo "  force-nvidia-drm - 寮哄埗杩佺Щnvidia-drm锛堝鐞嗛〗鍥虹殑tegra_drm锛?
	@echo "  nvidia-drm-test  - 杩愯瀹屾暣鐨凬VIDIA-DRM杩佺Щ楠岃瘉娴嬭瘯"
	@echo "  nvidia-drm-report- 鐢熸垚NVIDIA-DRM杩佺Щ鐘舵€佹姤鍛?
	@echo "  nvidia-drm-complete - 杩愯瀹屾暣鐨勮縼绉婚獙璇佹祦绋?
	@echo ""
	@echo "$(GREEN)瀹夎鍛戒护:$(NC)"
	@echo "  install          - 瀹屾暣瀹夎绯荤粺"
	@echo "  install-deps     - 瀹夎鎵€鏈変緷璧?绯荤粺+Wayland+LVGL)"
	@echo "  install-system-deps - 浠呭畨瑁呯郴缁熶緷璧?
	@echo "  install-wayland-deps - 瀹夎Wayland鐜鍜孲way"
	@echo "  install-lvgl     - 妫€鏌ュ苟瀹夎LVGL"
	@echo "  install-system   - 瀹夎缂栬瘧濂界殑绯荤粺"
	@echo "  install-service  - 瀹夎systemd鏈嶅姟"
	@echo ""
	@echo "$(GREEN)鏈嶅姟绠＄悊:$(NC)"
	@echo "  start            - 鍚姩鏈嶅姟"
	@echo "  stop             - 鍋滄鏈嶅姟"
	@echo "  restart          - 閲嶅惎鏈嶅姟"
	@echo "  status           - 鏌ョ湅鏈嶅姟鐘舵€?
	@echo "  logs             - 鏌ョ湅鏈嶅姟鏃ュ織"
	@echo "  enable-service   - 鍚敤寮€鏈鸿嚜鍚?
	@echo "  disable-service  - 绂佺敤寮€鏈鸿嚜鍚?
	@echo ""
	@echo "$(GREEN)Wayland鐜绠＄悊锛圫way锛?$(NC)"
	@echo "  setup-wayland    - 閰嶇疆Wayland鐜鍜孲way鏈嶅姟"
	@echo "  start-sway       - 鍚姩Sway鍚堟垚鍣紙鑷姩瀹夎+閰嶇疆+鍚姩锛?
	@echo "  stop-sway        - 鍋滄Sway鍚堟垚鍣?
	@echo "  sway-status      - 鏌ョ湅Sway鐘舵€佸拰瑙︽懜璁惧"
	@echo "  sway-logs        - 鏌ョ湅Sway鏃ュ織"
	@echo "  check-wayland    - 妫€鏌ayland鐜瀹屾暣鎬?
	@echo ""
	@echo "$(GREEN)Weston 12 闄嶇骇鏀寔锛堣В鍐?Weston 13 bug锛?$(NC)"
	@echo "  downgrade-to-weston12    - 馃殌 涓€閿檷绾у埌 Weston 12锛堟帹鑽愶級"
	@echo "  check-weston-version     - 妫€鏌ュ綋鍓?Weston 鐗堟湰"
	@echo "  start-weston12           - 鍚姩 Weston 12 鏈嶅姟"
	@echo "  stop-weston12            - 鍋滄 Weston 12 鏈嶅姟"
	@echo "  weston12-status          - 鏌ョ湅 Weston 12 鐘舵€?
	@echo "  weston12-logs            - 鏌ョ湅 Weston 12 鏃ュ織"
	@echo "  test-weston12            - 娴嬭瘯 Weston 12 鐜"
	@echo ""
	@echo "$(GREEN)缁存姢鍛戒护:$(NC)"
	@echo "  check-system     - 妫€鏌ョ郴缁熺幆澧?
	@echo "  check-wayland    - 妫€鏌ayland鐜"
	@echo "  setup-config     - 璁剧疆閰嶇疆鏂囦欢"
	@echo "  test             - 杩愯绯荤粺娴嬭瘯"
	@echo "  backup           - 澶囦唤褰撳墠绯荤粺"
	@echo ""
	@echo "$(YELLOW)鐗规€ц鏄?$(NC)"
	@echo "  鉁?C++17楂樻€ц兘鎺ㄧ悊寮曟搸"
	@echo "  鉁?LVGL宸ヤ笟绾цЕ鎽哥晫闈?
	@echo "  鉁?YOLOv8+TensorRT鍔犻€?
	@echo "  鉁?Modbus TCP PLC閫氫俊"
	@echo "  鉁?Jetson Orin NX浼樺寲"
	@echo "  鉁?瀹炴椂瑙嗛澶勭悊涓庢娴?

# === 绯荤粺妫€鏌?===
check-system:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌ョ郴缁熺幆澧?.."
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) cmake鏈畨瑁?; \
		exit 1; \
	fi
	@if ! command -v g++ >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) g++缂栬瘧鍣ㄦ湭瀹夎"; \
		exit 1; \
	fi
	@if ! pkg-config --exists opencv4 2>/dev/null; then \
		if ! pkg-config --exists opencv 2>/dev/null; then \
			echo "$(RED)[ERROR]$(NC) OpenCV寮€鍙戝簱鏈畨瑁?; \
			exit 1; \
		fi; \
	fi
	@if ! pkg-config --exists gstreamer-1.0 2>/dev/null; then \
		echo "$(RED)[ERROR]$(NC) GStreamer寮€鍙戝簱鏈畨瑁?; \
		exit 1; \
	fi
	@if [ ! -f "/usr/include/modbus/modbus.h" ] && [ ! -f "/usr/local/include/modbus/modbus.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) libmodbus寮€鍙戝簱鏈壘鍒帮紝灏嗙鐢∕odbus鍔熻兘"; \
	fi
	@if [ ! -f "/usr/include/lvgl/lvgl.h" ] && [ ! -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL寮€鍙戝簱鏈壘鍒帮紝灏嗙鐢ㄧ晫闈㈠姛鑳?; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺鐜妫€鏌ラ€氳繃"

# === 渚濊禆瀹夎 ===
install-deps: install-system-deps install-lvgl9-auto
	@echo "$(GREEN)[SUCCESS]$(NC) 鎵€鏈変緷璧栧畨瑁呭畬鎴?

# === 鑷姩鐜閰嶇疆 ===
# 浣跨敤 Weston 12锛堟帹鑽愮敤浜?Jetson + Nvidia锛?
auto-setup-environment:
	@echo "$(BLUE)[INFO]$(NC) 鑷姩妫€鏌ュ拰閰嶇疆Wayland鐜锛堜娇鐢ㄧ郴缁?Weston锛?.."
	@# 1. 妫€鏌?Weston 鏄惁宸插畨瑁?
	@if ! command -v weston >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) Weston 鏈畨瑁咃紝璇峰厛瀹夎: sudo apt-get install weston"; \
		exit 1; \
	fi
	@WESTON_VERSION=$$(weston --version 2>&1 | grep -oP 'weston \K[\d.]+' | head -1 || echo "unknown"); \
	echo "$(GREEN)[SUCCESS]$(NC) 妫€娴嬪埌绯荤粺 Weston $$WESTON_VERSION"
	@# 2. 閰嶇疆 Nvidia DRM 妯″潡锛圝etson 蹇呴渶锛?
	@echo "$(BLUE)[INFO]$(NC) 閰嶇疆 Nvidia DRM 妯″潡..."
	@sudo modprobe nvidia-drm modeset=1 2>/dev/null || true
	@if ! grep -q "options nvidia-drm modeset=1" /etc/modprobe.d/nvidia-drm.conf 2>/dev/null; then \
		echo "options nvidia-drm modeset=1" | sudo tee /etc/modprobe.d/nvidia-drm.conf >/dev/null; \
		echo "$(GREEN)[SUCCESS]$(NC) Nvidia DRM 妯″潡閰嶇疆宸蹭繚瀛?; \
	fi
	@# 3. 閰嶇疆鐢ㄦ埛鏉冮檺
	@echo "$(BLUE)[INFO]$(NC) 閰嶇疆 DRM 璁惧鏉冮檺..."
	@sudo usermod -a -G video,render,input root 2>/dev/null || true
	@# 4. 閰嶇疆 Weston 鏈嶅姟
	@if [ ! -f "/etc/systemd/system/weston.service" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) Weston 鏈嶅姟鏈厤缃紝姝ｅ湪閰嶇疆..."; \
		$(MAKE) setup-weston-service; \
	fi
	@# 5. 鍋滄鍏朵粬 Wayland 鍚堟垚鍣?
	@if systemctl is-active --quiet sway-wayland.service 2>/dev/null; then \
		echo "$(BLUE)[INFO]$(NC) 鍋滄 Sway 鏈嶅姟..."; \
		sudo systemctl stop sway-wayland.service; \
		sudo systemctl disable sway-wayland.service; \
	fi
	@if systemctl is-active --quiet weston12.service 2>/dev/null; then \
		echo "$(BLUE)[INFO]$(NC) 鍋滄 Weston 12 鏈嶅姟..."; \
		sudo systemctl stop weston12.service; \
		sudo systemctl disable weston12.service; \
	fi
	@# 6. 鍚姩 Weston
	@WESTON_RUNNING=false; \
	if pgrep -x weston >/dev/null 2>&1; then \
		echo "$(GREEN)[INFO]$(NC) 妫€娴嬪埌 Weston 杩涚▼姝ｅ湪杩愯"; \
		WESTON_RUNNING=true; \
	elif systemctl is-active --quiet weston.service 2>/dev/null; then \
		echo "$(GREEN)[INFO]$(NC) 妫€娴嬪埌 Weston 鏈嶅姟姝ｅ湪杩愯"; \
		WESTON_RUNNING=true; \
	fi; \
	if [ "$$WESTON_RUNNING" = "false" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) Weston 鏈繍琛岋紝姝ｅ湪鍚姩..."; \
		sudo systemctl enable weston.service; \
		sudo systemctl start weston.service; \
		sleep 3; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Weston 宸插湪杩愯锛岃烦杩囧惎鍔?; \
	fi
	@# 7. 楠岃瘉 Wayland 鐜
	@if ! ls /run/user/0/wayland-* >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Wayland socket 涓嶅瓨鍦紝绛夊緟 Weston 瀹屽叏鍚姩..."; \
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
	@echo "$(GREEN)[SUCCESS]$(NC) Wayland渚濊禆瀹夎瀹屾垚锛圫way锛?

# 妫€鏌?Mutter 鏄惁宸插畨瑁?
check-mutter:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌utter鍚堟垚鍣?.."
	@if ! command -v mutter >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Mutter鏈畨瑁咃紝姝ｅ湪瀹夎..."; \
		sudo apt-get update && sudo apt-get install -y mutter gnome-session dbus-x11; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Mutter宸插畨瑁? $$(mutter --version 2>&1 | head -n1)"; \
	fi

# 閰嶇疆 Mutter 鏈嶅姟
setup-mutter:
	@echo "$(BLUE)[INFO]$(NC) 閰嶇疆Mutter Wayland鏈嶅姟..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Mutter鏈嶅姟閰嶇疆瀹屾垚锛堝凡鍖呭惈D-Bus鍚姩锛?

# 鍚姩 Mutter
start-mutter: check-mutter setup-mutter
	@echo "$(BLUE)[INFO]$(NC) 鍚姩Mutter Wayland鍚堟垚鍣?.."
	@sudo systemctl enable mutter-wayland
	@sudo systemctl start mutter-wayland
	@sleep 2
	@if sudo systemctl is-active --quiet mutter-wayland; then \
		echo "$(GREEN)[SUCCESS]$(NC) Mutter鍚姩鎴愬姛"; \
		echo "WAYLAND_DISPLAY=$$(ls /run/user/0/wayland-* 2>/dev/null | head -n1 | xargs basename)"; \
	else \
		echo "$(RED)[ERROR]$(NC) Mutter鍚姩澶辫触"; \
		sudo journalctl -u mutter-wayland -n 20 --no-pager; \
		exit 1; \
	fi

# 鍋滄 Mutter
stop-mutter:
	@echo "$(BLUE)[INFO]$(NC) 鍋滄Mutter..."
	@sudo systemctl stop mutter-wayland || true
	@echo "$(GREEN)[SUCCESS]$(NC) Mutter宸插仠姝?

# 妫€鏌?Mutter 鐘舵€?
mutter-status:
	@echo "$(CYAN)=== Mutter鐘舵€?===$(NC)"
	@sudo systemctl status mutter-wayland --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "鏃燱ayland socket"

# Mutter 鏃ュ織
mutter-logs:
	@echo "$(CYAN)=== Mutter鏃ュ織 ===$(NC)"
	@sudo journalctl -u mutter-wayland -f --no-hostname

# ============================================================================
# Sway Wayland 鍚堟垚鍣紙鎺ㄨ崘鐢ㄤ簬宓屽叆寮忥紝鏀寔瑙︽懜鎺у埗锛?
# ============================================================================

# 妫€鏌?Sway 鏄惁宸插畨瑁?
check-sway:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌way鍚堟垚鍣?.."
	@if ! command -v sway >/dev/null 2>&1; then \
		echo "$(YELLOW)[WARNING]$(NC) Sway鏈畨瑁咃紝姝ｅ湪瀹夎..."; \
		sudo apt-get update && sudo apt-get install -y sway swaylock swayidle xwayland libinput-tools; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) Sway宸插畨瑁? $$(sway --version 2>&1 | head -n1)"; \
	fi

# 鍒涘缓 Sway 閰嶇疆鏂囦欢锛堟敮鎸佽Е鎽告帶鍒讹級
setup-sway-config:
	@echo "$(BLUE)[INFO]$(NC) 鍒涘缓Sway閰嶇疆鏂囦欢锛堟敮鎸佽Е鎽告帶鍒讹級..."
	@sudo mkdir -p /root/.config/sway
	@echo "# Sway閰嶇疆 - Bamboo璇嗗埆绯荤粺锛堣Е鎽告帶鍒朵紭鍖栵級" | sudo tee /root/.config/sway/config > /dev/null
	@echo "# 鑷姩鐢熸垚锛岃鍕挎墜鍔ㄧ紪杈? | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 鍩虹璁剧疆" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "set \$$mod Mod4" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 杈撳嚭閰嶇疆锛堟牴鎹疄闄呭睆骞曡皟鏁达級" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "output * bg #000000 solid_color" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "output * mode 1920x1080@60Hz" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 瑙︽懜灞?瑙︽懜鏉挎敮鎸侀厤缃? | sudo tee -a /root/.config/sway/config > /dev/null
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
	@echo "# 绂佺敤灞忓箷閿佸畾鍜岀數婧愮鐞嗭紙宸ヤ笟搴旂敤锛? | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "exec swayidle -w timeout 0 'echo disabled' before-sleep 'echo disabled'" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 绂佺敤绐楀彛瑁呴グ锛堝叏灞忔ā寮忥級" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "default_border none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "default_floating_border none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "titlebar_border_thickness 0" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "titlebar_padding 0" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 鐒︾偣閰嶇疆" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "focus_follows_mouse no" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "mouse_warping none" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "# 鑷姩鍏ㄥ睆搴旂敤" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "for_window [title=\".*\"] fullscreen enable" | sudo tee -a /root/.config/sway/config > /dev/null
	@echo "" | sudo tee -a /root/.config/sway/config > /dev/null
	@sudo chmod 644 /root/.config/sway/config
	@echo "$(GREEN)[SUCCESS]$(NC) Sway閰嶇疆鏂囦欢鍒涘缓瀹屾垚"

# 閰嶇疆 Sway 鏈嶅姟
setup-sway:
	@echo "$(BLUE)[INFO]$(NC) 閰嶇疆Sway Wayland鏈嶅姟..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Sway鏈嶅姟閰嶇疆瀹屾垚锛堝凡鍚敤瑙︽懜鏀寔锛?

# 鍚姩 Sway
start-sway: check-sway setup-sway-config setup-sway
	@echo "$(BLUE)[INFO]$(NC) 鍚姩Sway Wayland鍚堟垚鍣?.."
	@sudo systemctl enable sway-wayland
	@sudo systemctl start sway-wayland
	@sleep 3
	@if sudo systemctl is-active --quiet sway-wayland; then \
		echo "$(GREEN)[SUCCESS]$(NC) Sway鍚姩鎴愬姛"; \
		echo "WAYLAND_DISPLAY=$$(ls /run/user/0/wayland-* 2>/dev/null | head -n1 | xargs basename)"; \
		echo "瑙︽懜鎺у埗: 宸插惎鐢?; \
	else \
		echo "$(RED)[ERROR]$(NC) Sway鍚姩澶辫触"; \
		sudo journalctl -u sway-wayland -n 30 --no-pager; \
		exit 1; \
	fi

# 鍋滄 Sway
stop-sway:
	@echo "$(BLUE)[INFO]$(NC) 鍋滄Sway..."
	@sudo systemctl stop sway-wayland || true
	@echo "$(GREEN)[SUCCESS]$(NC) Sway宸插仠姝?

# 妫€鏌?Sway 鐘舵€?
sway-status:
	@echo "$(CYAN)=== Sway鐘舵€?===$(NC)"
	@sudo systemctl status sway-wayland --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "鏃燱ayland socket"
	@echo ""
	@echo "$(CYAN)=== 瑙︽懜璁惧 ===$(NC)"
	@libinput list-devices 2>/dev/null | grep -A 5 "Capabilities.*touch" || echo "鏈娴嬪埌瑙︽懜璁惧"

# Sway 鏃ュ織
sway-logs:
	@echo "$(CYAN)=== Sway鏃ュ織 ===$(NC)"
	@sudo journalctl -u sway-wayland -f --no-hostname

setup-wayland: start-sway
	@echo "$(GREEN)[SUCCESS]$(NC) Wayland鐜閰嶇疆瀹屾垚锛圫way + 瑙︽懜鎺у埗锛?

# ============================================================================
# Weston 12 闄嶇骇鏀寔锛堣В鍐?Weston 13 鐨?xdg_positioner bug锛?
# ============================================================================

.PHONY: check-weston-version backup-current-weston uninstall-current-weston \
        install-weston12-build-deps download-weston12 compile-weston12 \
        install-weston12 configure-weston12 setup-weston12-service \
        start-weston12 stop-weston12 weston12-status weston12-logs \
        downgrade-to-weston12 test-weston12

# 妫€鏌ュ綋鍓?Weston 鐗堟湰
check-weston-version:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌?Weston 鐗堟湰..."
	@if command -v weston >/dev/null 2>&1; then \
		WESTON_VERSION=$$(weston --version 2>&1 | grep -oP 'weston \K\d+\.\d+' | head -1 || echo "鏈煡"); \
		echo "$(CYAN)褰撳墠 Weston 鐗堟湰: $$WESTON_VERSION$(NC)"; \
		WESTON_MAJOR=$$(echo $$WESTON_VERSION | cut -d. -f1); \
		if [ "$$WESTON_MAJOR" = "12" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) 鉁?Weston 12 宸插畨瑁?; \
		elif [ "$$WESTON_MAJOR" = "13" ]; then \
			echo "$(YELLOW)[WARNING]$(NC) 鈿?Weston 13 瀛樺湪宸茬煡 xdg_positioner bug锛屽缓璁檷绾?; \
		elif [ "$$WESTON_MAJOR" = "9" ] || [ "$$WESTON_MAJOR" = "10" ]; then \
			echo "$(YELLOW)[WARNING]$(NC) 鈿?Weston 鐗堟湰杈冩棫 ($$WESTON_VERSION)锛屽缓璁崌绾у埌 12"; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) 鈿?鏈煡 Weston 鐗堟湰: $$WESTON_VERSION"; \
		fi; \
		which weston; \
		ls -lh $$(which weston); \
	else \
		echo "$(RED)[ERROR]$(NC) 鉁?Weston 鏈畨瑁?; \
	fi
	@echo ""
	@echo "$(CYAN)DRM 璁惧鐘舵€?$(NC)"
	@ls -la /dev/dri/ 2>/dev/null || echo "$(YELLOW)DRM 璁惧涓嶅瓨鍦?(NC)"

# 澶囦唤褰撳墠 Weston 閰嶇疆
backup-current-weston:
	@echo "$(BLUE)[INFO]$(NC) 澶囦唤褰撳墠 Weston 閰嶇疆..."
	@BACKUP_DATE=$$(date +%Y%m%d_%H%M%S); \
	sudo mkdir -p /opt/backup/weston; \
	if [ -d "/etc/xdg/weston" ]; then \
		sudo cp -r /etc/xdg/weston /opt/backup/weston/weston-etc-$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 閰嶇疆宸插浠? /opt/backup/weston/weston-etc-$$BACKUP_DATE"; \
	fi; \
	if [ -f "/root/.config/weston.ini" ]; then \
		sudo cp /root/.config/weston.ini /opt/backup/weston/weston.ini.$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 鐢ㄦ埛閰嶇疆宸插浠?; \
	fi; \
	if [ -f "/etc/systemd/system/weston.service" ]; then \
		sudo cp /etc/systemd/system/weston.service /opt/backup/weston/weston.service.$$BACKUP_DATE; \
		echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟鏂囦欢宸插浠?; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 澶囦唤瀹屾垚"

# 鍗歌浇褰撳墠 Weston
uninstall-current-weston:
	@echo "$(BLUE)[INFO]$(NC) 鍋滄骞跺嵏杞藉綋鍓?Weston..."
	@# 鍋滄鎵€鏈夋湇鍔?
	@sudo systemctl stop bamboo-cpp-lvgl 2>/dev/null || true
	@sudo systemctl stop weston.service 2>/dev/null || true
	@sudo systemctl stop weston 2>/dev/null || true
	@sudo pkill -9 weston 2>/dev/null || true
	@sleep 2
	@# 鍗歌浇 Weston锛堝鏋滄槸 APT 瀹夎锛?
	@if dpkg -l | grep -q "^ii.*weston"; then \
		echo "$(BLUE)[INFO]$(NC) 鍗歌浇 APT 瀹夎鐨?Weston..."; \
		sudo apt-get remove --purge -y weston libweston-* 2>/dev/null || true; \
		sudo apt-get autoremove -y; \
	fi
	@# 鍒犻櫎鎵嬪姩缂栬瘧鐨?Weston 鏂囦欢
	@echo "$(BLUE)[INFO]$(NC) 鍒犻櫎鎵嬪姩缂栬瘧鐨?Weston 鏂囦欢..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 宸插嵏杞?

# 瀹夎 Weston 12 缂栬瘧渚濊禆
install-weston12-build-deps:
	@echo "$(BLUE)[INFO]$(NC) 瀹夎 Weston 12 缂栬瘧渚濊禆..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) 渚濊禆瀹夎瀹屾垚"

# 涓嬭浇 Weston 12.0.0 婧愮爜
download-weston12:
	@echo "$(BLUE)[INFO]$(NC) 涓嬭浇 Weston 12.0.0 婧愮爜..."
	@sudo mkdir -p /tmp/weston12-build
	@cd /tmp/weston12-build && \
		if [ ! -f "weston-12.0.0.tar.xz" ]; then \
			wget -q --show-progress https://wayland.freedesktop.org/releases/weston-12.0.0.tar.xz || \
			wget -q --show-progress https://gitlab.freedesktop.org/wayland/weston/-/archive/12.0.0/weston-12.0.0.tar.gz -O weston-12.0.0.tar.xz; \
		fi
	@echo "$(BLUE)[INFO]$(NC) 瑙ｅ帇婧愮爜..."
	@cd /tmp/weston12-build && \
		rm -rf weston-12.0.0 && \
		tar -xf weston-12.0.0.tar.xz
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12.0.0 婧愮爜宸插噯澶?

# 缂栬瘧 Weston 12
compile-weston12: install-weston12-build-deps download-weston12
	@echo "$(CYAN)[COMPILE]$(NC) 寮€濮嬬紪璇?Weston 12.0.0 (棰勮 15-30 鍒嗛挓)..."
	@cd /tmp/weston12-build/weston-12.0.0 && \
		echo "$(BLUE)[INFO]$(NC) 閰嶇疆 Meson..." && \
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
		echo "$(BLUE)[INFO]$(NC) 寮€濮嬬紪璇?(浣跨敤 $(shell nproc) 涓牳蹇?..." && \
		meson compile -C build -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12.0.0 缂栬瘧瀹屾垚"

# 瀹夎 Weston 12
install-weston12: compile-weston12
	@echo "$(BLUE)[INFO]$(NC) 瀹夎 Weston 12..."
	@cd /tmp/weston12-build/weston-12.0.0/build && \
		sudo meson install
	@sudo ldconfig
	@echo "$(BLUE)[INFO]$(NC) 楠岃瘉瀹夎..."
	@weston --version
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 瀹夎瀹屾垚"

# 閰嶇疆 Weston 12
configure-weston12:
	@echo "$(BLUE)[INFO]$(NC) 閰嶇疆 Weston 12..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 閰嶇疆鏂囦欢宸插垱寤? /etc/xdg/weston/weston.ini"

# 鍒涘缓绯荤粺 Weston systemd 鏈嶅姟锛堜娇鐢?Nvidia Weston 13锛?
setup-weston-service:
	@echo "$(BLUE)[INFO]$(NC) 鍒涘缓绯荤粺 Weston systemd 鏈嶅姟..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 鏈嶅姟宸查厤缃苟鍚敤"

# 鍒涘缓 Weston 12 systemd 鏈嶅姟
setup-weston12-service:
	@echo "$(BLUE)[INFO]$(NC) 鍒涘缓 Weston 12 systemd 鏈嶅姟..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 鏈嶅姟宸查厤缃苟鍚敤"

# 鍚姩 Weston 12
start-weston12:
	@echo "$(BLUE)[INFO]$(NC) 鍚姩 Weston 12..."
	@sudo systemctl start weston12.service
	@sleep 3
	@if sudo systemctl is-active --quiet weston12.service; then \
		echo "$(GREEN)[SUCCESS]$(NC) 鉁?Weston 12 鍚姩鎴愬姛"; \
		echo "$(CYAN)Wayland Socket:$(NC)"; \
		ls -la /run/user/0/wayland-* 2>/dev/null || echo "$(YELLOW)绛夊緟 socket 鍒涘缓...$(NC)"; \
		sleep 2; \
		ls -la /run/user/0/wayland-* 2>/dev/null || echo "$(RED)Socket 鏈垱寤猴紝鏌ョ湅鏃ュ織$(NC)"; \
	else \
		echo "$(RED)[ERROR]$(NC) 鉁?Weston 12 鍚姩澶辫触"; \
		echo "$(CYAN)鏌ョ湅鏈€杩?30 琛屾棩蹇?$(NC)"; \
		sudo journalctl -u weston12.service -n 30 --no-pager; \
		exit 1; \
	fi

# 鍋滄 Weston 12
stop-weston12:
	@echo "$(BLUE)[INFO]$(NC) 鍋滄 Weston 12..."
	@sudo systemctl stop weston12.service
	@sudo pkill -9 weston 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) Weston 12 宸插仠姝?

# 鏌ョ湅 Weston 12 鐘舵€?
weston12-status:
	@echo "$(CYAN)=== Weston 12 鏈嶅姟鐘舵€?===$(NC)"
	@sudo systemctl status weston12.service --no-pager -l || true
	@echo ""
	@echo "$(CYAN)=== Weston 杩涚▼ ===$(NC)"
	@ps aux | grep weston | grep -v grep || echo "鏃?Weston 杩涚▼"
	@echo ""
	@echo "$(CYAN)=== Wayland Socket ===$(NC)"
	@ls -lah /run/user/0/wayland-* 2>/dev/null || echo "鏃?Wayland socket"
	@echo ""
	@echo "$(CYAN)=== DRM 璁惧 ===$(NC)"
	@ls -la /dev/dri/ 2>/dev/null || echo "DRM 璁惧涓嶅瓨鍦?

# 鏌ョ湅 Weston 12 鏃ュ織
weston12-logs:
	@echo "$(CYAN)=== Weston 12 systemd 鏃ュ織 (鏈€杩?100 琛? ===$(NC)"
	@sudo journalctl -u weston12.service -n 100 --no-pager
	@echo ""
	@echo "$(CYAN)=== Weston 12 杩愯鏃ュ織 ===$(NC)"
	@if [ -f /var/log/weston12.log ]; then \
		sudo tail -100 /var/log/weston12.log; \
	else \
		echo "鏃ュ織鏂囦欢 /var/log/weston12.log 涓嶅瓨鍦?; \
	fi

# 娴嬭瘯 Weston 12
test-weston12:
	@echo "$(BLUE)[INFO]$(NC) 娴嬭瘯 Weston 12..."
	@echo "$(CYAN)1. 妫€鏌ョ増鏈?$(NC)"
	@weston --version
	@echo ""
	@echo "$(CYAN)2. 妫€鏌ユ湇鍔＄姸鎬?$(NC)"
	@sudo systemctl is-active weston12.service && echo "$(GREEN)鉁?鏈嶅姟杩愯涓?(NC)" || echo "$(RED)鉁?鏈嶅姟鏈繍琛?(NC)"
	@echo ""
	@echo "$(CYAN)3. 妫€鏌?Wayland socket:$(NC)"
	@ls -la /run/user/0/wayland-* 2>/dev/null && echo "$(GREEN)鉁?Socket 瀛樺湪$(NC)" || echo "$(RED)鉁?Socket 涓嶅瓨鍦?(NC)"
	@echo ""
	@echo "$(CYAN)4. 妫€鏌ラ厤缃枃浠?$(NC)"
	@if [ -f /etc/xdg/weston/weston.ini ]; then \
		echo "$(GREEN)鉁?閰嶇疆鏂囦欢瀛樺湪$(NC)"; \
		echo "鍐呭棰勮:"; \
		head -20 /etc/xdg/weston/weston.ini; \
	else \
		echo "$(RED)鉁?閰嶇疆鏂囦欢涓嶅瓨鍦?(NC)"; \
	fi

# 馃殌 涓€閿檷绾у埌 Weston 12锛堟帹鑽愪娇鐢級
downgrade-to-weston12:
	@echo "$(CYAN)======================================$(NC)"
	@echo "$(CYAN)  Weston 12 瀹屾暣闄嶇骇娴佺▼$(NC)"
	@echo "$(CYAN)======================================$(NC)"
	@echo ""
	@echo "$(BLUE)[姝ラ 1/9]$(NC) 妫€鏌ュ綋鍓嶇増鏈?.."
	@$(MAKE) check-weston-version
	@echo ""
	@echo "$(BLUE)[姝ラ 2/9]$(NC) 澶囦唤褰撳墠閰嶇疆..."
	@$(MAKE) backup-current-weston
	@echo ""
	@echo "$(BLUE)[姝ラ 3/9]$(NC) 鍗歌浇褰撳墠 Weston..."
	@$(MAKE) uninstall-current-weston
	@echo ""
	@echo "$(BLUE)[姝ラ 4/9]$(NC) 缂栬瘧 Weston 12 (闇€瑕?15-30 鍒嗛挓)..."
	@$(MAKE) install-weston12
	@echo ""
	@echo "$(BLUE)[姝ラ 5/9]$(NC) 閰嶇疆 Weston 12..."
	@$(MAKE) configure-weston12
	@echo ""
	@echo "$(BLUE)[姝ラ 6/9]$(NC) 鍒涘缓 systemd 鏈嶅姟..."
	@$(MAKE) setup-weston12-service
	@echo ""
	@echo "$(BLUE)[姝ラ 7/9]$(NC) 鍚姩 Weston 12..."
	@$(MAKE) start-weston12
	@echo ""
	@echo "$(BLUE)[姝ラ 8/9]$(NC) 娴嬭瘯 Weston 12..."
	@$(MAKE) test-weston12
	@echo ""
	@echo "$(BLUE)[姝ラ 9/9]$(NC) 娓呯悊涓存椂鏂囦欢..."
	@sudo rm -rf /tmp/weston12-build
	@echo ""
	@echo "$(GREEN)======================================$(NC)"
	@echo "$(GREEN)  鉁撯湏鉁?Weston 12 闄嶇骇瀹屾垚锛?(NC)"
	@echo "$(GREEN)======================================$(NC)"
	@echo ""
	@echo "$(CYAN)涓嬩竴姝ユ搷浣?$(NC)"
	@echo "  1. 鏌ョ湅 Weston 12 鐘舵€? $(YELLOW)make weston12-status$(NC)"
	@echo "  2. 鏌ョ湅 Weston 12 鏃ュ織: $(YELLOW)make weston12-logs$(NC)"
	@echo "  3. 閲嶆柊閮ㄧ讲搴旂敤: $(YELLOW)make redeploy$(NC)"
	@echo "  4. 鏌ョ湅搴旂敤鏃ュ織: $(YELLOW)sudo journalctl -u bamboo-cpp-lvgl -f$(NC)"
	@echo ""
	@echo "$(CYAN)濡傛灉閬囧埌闂:$(NC)"
	@echo "  - 鏌ョ湅鏈嶅姟鐘舵€? $(YELLOW)make weston12-status$(NC)"
	@echo "  - 閲嶅惎 Weston 12: $(YELLOW)sudo systemctl restart weston12$(NC)"
	@echo "  - 鎭㈠澶囦唤: 鏌ョ湅 /opt/backup/weston/"
	@echo ""

# 鍏煎鎬у埆鍚嶏紙鏇存柊涓轰娇鐢?Weston 12锛?
start-weston: start-weston12
stop-weston: stop-weston12
weston-status: weston12-status

check-wayland:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌ayland鐜锛圫way锛?.."
	@echo -n "Sway鏈嶅姟鐘舵€? "
	@sudo systemctl is-active sway-wayland.service 2>/dev/null || echo "鏈繍琛?
	@echo -n "Wayland socket: "
	@ls /run/user/0/wayland-* 2>/dev/null && echo "瀛樺湪" || echo "涓嶅瓨鍦?
	@echo -n "Wayland搴? "
	@pkg-config --exists wayland-client && echo "宸插畨瑁? || echo "鏈畨瑁?
	@echo -n "EGL搴? "
	@ldconfig -p | grep -q "libEGL" && echo "宸插畨瑁? || echo "鏈畨瑁?

install-system-deps:
	@echo "$(BLUE)[INFO]$(NC) 瀹夎绯荤粺渚濊禆..."
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
		echo "$(BLUE)[INFO]$(NC) 妫€娴嬪埌NVIDIA GPU锛屾鏌UDA鐜..."; \
		if [ -d "/usr/local/cuda" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) CUDA鐜宸插畨瑁?; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) CUDA鐜鏈畨瑁咃紝璇锋墜鍔ㄥ畨瑁匔UDA鍜孴ensorRT"; \
		fi \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺渚濊禆瀹夎瀹屾垚"

install-lvgl:
	@echo "$(CYAN)[LVGL]$(NC) 妫€鏌VGL v9瀹夎鐘舵€?.."
	@LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null || echo "not_found"); \
	if [ "$$LVGL_VERSION" = "not_found" ] || [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" != "9" ]; then \
		echo "$(BLUE)[INFO]$(NC) LVGL v9鏈壘鍒?(褰撳墠鐗堟湰: $$LVGL_VERSION)锛屽紑濮嬩粠婧愮爜缂栬瘧瀹夎..."; \
		$(MAKE) build-lvgl-from-source; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL v9宸插畨瑁?(鐗堟湰: $$LVGL_VERSION)"; \
	fi

build-lvgl-from-source:
	@echo "$(CYAN)[LVGL]$(NC) === 瀹屽叏鎵嬪姩瀹夎LVGL v9.1 ==="
	@echo "$(BLUE)[INFO]$(NC) [1/8] 娓呯悊鏃ф枃浠?.."
	@sudo rm -rf /usr/local/include/lvgl 2>/dev/null || true
	@sudo rm -rf /usr/local/lib/liblvgl* 2>/dev/null || true
	@sudo rm -rf /usr/local/lib/pkgconfig/lvgl.pc 2>/dev/null || true
	@sudo rm -rf /tmp/lvgl 2>/dev/null || true
	@sudo ldconfig 2>/dev/null || true
	@echo "$(BLUE)[INFO]$(NC) [2/8] 瀹夎渚濊禆..."
	@sudo apt-get update -qq
	@sudo apt-get install -y git cmake build-essential
	@echo "$(BLUE)[INFO]$(NC) [3/8] 涓嬭浇LVGL v9.1..."
	@cd /tmp && rm -rf lvgl && git clone --depth 1 --branch release/v9.1 https://github.com/lvgl/lvgl.git
	@echo "$(BLUE)[INFO]$(NC) [4/8] 鍒涘缓閰嶇疆鏂囦欢..."
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
	@echo "$(BLUE)[INFO]$(NC) [5/8] 閰嶇疆CMake..."
	@cd /tmp/lvgl && mkdir -p build && cd build && \
	cmake .. \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DLV_CONF_PATH=../lv_conf.h \
		-DBUILD_SHARED_LIBS=ON \
		-DLV_USE_FREETYPE=OFF
	@echo "$(BLUE)[INFO]$(NC) [6/8] 缂栬瘧LVGL..."
	@cd /tmp/lvgl/build && make -j4
	@echo "$(BLUE)[INFO]$(NC) [7/8] 瀹夎鏂囦欢..."
	@cd /tmp/lvgl/build && sudo make install
	@echo "$(BLUE)[INFO]$(NC) 鎵嬪姩纭繚澶存枃浠跺畨瑁?.."
	@sudo mkdir -p /usr/local/include/lvgl
	@cd /tmp/lvgl && sudo cp -r src/* /usr/local/include/lvgl/
	@cd /tmp/lvgl && sudo cp lvgl.h /usr/local/include/lvgl/
	@cd /tmp/lvgl && sudo cp lv_conf.h /usr/local/include/
	@echo "$(BLUE)[INFO]$(NC) [8/8] 鍒涘缓pkg-config鏂囦欢..."
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
	@echo "$(CYAN)[VERIFY]$(NC) === 楠岃瘉瀹夎 ==="
	@echo -n "$(BLUE)[INFO]$(NC) 澶存枃浠? "
	@ls /usr/local/include/lvgl/lvgl.h >/dev/null 2>&1 && echo "$(GREEN)鉁?(NC)" || (echo "$(RED)鉁?澶辫触$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) 搴撴枃浠? "
	@ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1 && echo "$(GREEN)鉁?(NC)" || (echo "$(RED)鉁?澶辫触$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) pkg-config: "
	@PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl && echo "$(GREEN)鉁?(NC)" || (echo "$(RED)鉁?澶辫触$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) v9 API: "
	@grep -q "lv_display_create" /usr/local/include/lvgl/lvgl.h && echo "$(GREEN)鉁?(NC)" || echo "$(YELLOW)鈿?鏈娴嬪埌浣嗗彲鑳芥甯?(NC)"
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) === LVGL v9.1瀹夎瀹屾垚 ==="
	@rm -rf /tmp/lvgl

# 瀹夎LVGL v9鐨勫揩閫熷懡浠?
install-lvgl9: build-lvgl-from-source
	@echo "$(GREEN)[SUCCESS]$(NC) LVGL v9.3瀹夎瀹屾垚锛岀郴缁熷凡鍑嗗灏辩华"

# 鑷姩妫€鏌ュ拰瀹夎LVGL v9锛堢紪璇戝墠鑷姩鎵ц锛?
install-lvgl9-auto:
	@echo "$(CYAN)[AUTO-INSTALL]$(NC) === 鏅鸿兘妫€娴婰VGL v9瀹夎鐘舵€?==="
	@echo "$(BLUE)[INFO]$(NC) 姝ｅ湪妫€娴婰VGL v9瀹夎鐘舵€?.."
	@LVGL_INSTALLED=false; \
	LVGL_VERSION_OK=false; \
	LVGL_API_OK=false; \
	if PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null); \
		echo "$(BLUE)[INFO]$(NC) 鍙戠幇宸插畨瑁呯殑LVGL鐗堟湰: $$LVGL_VERSION"; \
		if [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" = "9" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL涓荤増鏈负v9 鉁?; \
			LVGL_VERSION_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) LVGL鐗堟湰涓嶆槸v9 (褰撳墠: $$LVGL_VERSION)"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) pkg-config鏈壘鍒癓VGL"; \
	fi; \
	if [ -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL澶存枃浠跺瓨鍦?鉁?; \
		if grep -q "lv_display_create\|lv_disp_create" /usr/local/include/lvgl/lvgl.h 2>/dev/null; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL v9 API鍙敤 鉁?; \
			LVGL_API_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) 鏈娴嬪埌LVGL v9 API"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) LVGL澶存枃浠朵笉瀛樺湪"; \
	fi; \
	if ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL搴撴枃浠跺瓨鍦?鉁?; \
		LVGL_INSTALLED=true; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) LVGL搴撴枃浠朵笉瀛樺湪"; \
	fi; \
	if [ "$$LVGL_INSTALLED" = "true" ] && [ "$$LVGL_VERSION_OK" = "true" ] && [ "$$LVGL_API_OK" = "true" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) === LVGL v9宸叉纭畨瑁咃紝璺宠繃瀹夎姝ラ ==="; \
	else \
		echo "$(CYAN)[INSTALL]$(NC) === 闇€瑕佸畨瑁匧VGL v9.1 ==="; \
		$(MAKE) build-lvgl-from-source; \
	fi

# === C++绯荤粺鏋勫缓 ===
build-system:
	@echo "$(CYAN)[C++ LVGL]$(NC) 寮€濮嬫瀯寤篊++ LVGL涓€浣撳寲绯荤粺..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) C++ LVGL绯荤粺鏋勫缓瀹屾垚"

build-debug:
	@echo "$(CYAN)[C++ LVGL]$(NC) 鏋勫缓璋冭瘯鐗堟湰..."
	@mkdir -p $(BUILD_DIR)_debug
	@cd $(BUILD_DIR)_debug && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DENABLE_AI_OPTIMIZATION=ON \
		-DENABLE_MODBUS=ON \
		-DENABLE_LVGL=ON
	@cd $(BUILD_DIR)_debug && make -j$(shell nproc)
	@echo "$(GREEN)[SUCCESS]$(NC) 璋冭瘯鐗堟湰鏋勫缓瀹屾垚"

# 馃敡 鏂板锛氱紪璇戣嚜瀹氫箟YOLO瑙ｆ瀽搴?
compile-yolo-lib:
	@echo "$(BLUE)[INFO]$(NC) 馃敡 缂栬瘧鑷畾涔塝OLO瑙ｆ瀽搴?.."
	@sudo mkdir -p $(INSTALL_DIR)/lib
	@g++ -shared -fPIC \
		-I/opt/nvidia/deepstream/deepstream/sources/includes \
		-I/usr/local/cuda/include \
		cpp_backend/src/deepstream/nvdsinfer_yolo_bamboo.cpp \
		-o libnvdsinfer_yolo_bamboo.so
	@sudo cp libnvdsinfer_yolo_bamboo.so $(INSTALL_DIR)/lib/
	@sudo chmod 755 $(INSTALL_DIR)/lib/libnvdsinfer_yolo_bamboo.so
	@echo "$(GREEN)[SUCCESS]$(NC) 鉁?YOLO瑙ｆ瀽搴撶紪璇戦儴缃插畬鎴?

# === 绯荤粺瀹夎 ===
install-system: compile-yolo-lib
	@echo "$(BLUE)[INFO]$(NC) 瀹夎C++ LVGL绯荤粺鍒?(INSTALL_DIR)..."
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "$(RED)[ERROR]$(NC) 鏋勫缓鐩綍涓嶅瓨鍦紝璇峰厛杩愯 make build-system"; \
		exit 1; \
	fi
	@sudo mkdir -p $(INSTALL_DIR)
	@cd $(BUILD_DIR) && sudo make install
	@sudo mkdir -p $(INSTALL_DIR)/logs
	@sudo mkdir -p $(INSTALL_DIR)/backup
	@echo "$(BLUE)[INFO]$(NC) 澶嶅埗閰嶇疆鏂囦欢..."
	@sudo mkdir -p $(INSTALL_DIR)/config
	@sudo cp -r config/* $(INSTALL_DIR)/config/ 2>/dev/null || true
	@echo "$(BLUE)[INFO]$(NC) 纭繚nvinfer閰嶇疆鏂囦欢鍜屾爣绛炬枃浠跺瓨鍦?.."
	@if [ -f "config/nvinfer_config.txt" ]; then \
		sudo cp config/nvinfer_config.txt $(INSTALL_DIR)/config/; \
		echo "$(GREEN)[SUCCESS]$(NC) nvinfer閰嶇疆鏂囦欢宸插鍒?; \
	fi
	@if [ -f "config/labels.txt" ]; then \
		sudo cp config/labels.txt $(INSTALL_DIR)/config/; \
		echo "$(GREEN)[SUCCESS]$(NC) 鏍囩鏂囦欢宸插鍒?; \
	fi
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/logs
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/backup
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺瀹夎瀹屾垚"

# === 閰嶇疆璁剧疆 ===
setup-config:
	@echo "$(BLUE)[INFO]$(NC) 璁剧疆绯荤粺閰嶇疆..."
	@sudo mkdir -p $(INSTALL_DIR)/etc/bamboo-recognition
	@if [ ! -f "$(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml" ]; then \
		sudo cp config/system_config.yaml $(INSTALL_DIR)/etc/bamboo-recognition/ 2>/dev/null || \
		echo "# C++ LVGL涓€浣撳寲绯荤粺閰嶇疆" | sudo tee $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml >/dev/null; \
	fi
	@sudo chmod 644 $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml
	@echo "$(GREEN)[SUCCESS]$(NC) 閰嶇疆璁剧疆瀹屾垚"

# === 鏈嶅姟绠＄悊 ===
install-service: setup-config
	@echo "$(BLUE)[INFO]$(NC) 瀹夎systemd鏈嶅姟..."
	@if [ -f "$(BUILD_DIR)/bamboo-cpp-lvgl.service" ]; then \
		sudo cp $(BUILD_DIR)/bamboo-cpp-lvgl.service /etc/systemd/system/; \
	else \
		echo "$(RED)[ERROR]$(NC) 鏈嶅姟鏂囦欢鏈敓鎴愶紝璇锋鏌Make鏋勫缓"; \
		exit 1; \
	fi
	@sudo systemctl daemon-reload
	@echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟瀹夎瀹屾垚"

enable-service:
	@sudo systemctl enable $(SERVICE_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟宸插惎鐢ㄥ紑鏈鸿嚜鍚?

disable-service:
	@sudo systemctl disable $(SERVICE_NAME)
	@echo "$(BLUE)[INFO]$(NC) 鏈嶅姟宸茬鐢ㄥ紑鏈鸿嚜鍚?

start:
	@echo "$(BLUE)[INFO]$(NC) 鍚姩$(SERVICE_NAME)鏈嶅姟..."
	@sudo systemctl start $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟鍚姩鎴愬姛"; \
	else \
		echo "$(RED)[ERROR]$(NC) 鏈嶅姟鍚姩澶辫触锛岃鏌ョ湅鏃ュ織"; \
		exit 1; \
	fi

stop:
	@echo "$(BLUE)[INFO]$(NC) 鍋滄$(SERVICE_NAME)鏈嶅姟..."
	@sudo systemctl stop $(SERVICE_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟宸插仠姝?

restart:
	@echo "$(BLUE)[INFO]$(NC) 閲嶅惎$(SERVICE_NAME)鏈嶅姟..."
	@sudo systemctl restart $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) 鏈嶅姟閲嶅惎鎴愬姛"; \
	else \
		echo "$(RED)[ERROR]$(NC) 鏈嶅姟閲嶅惎澶辫触锛岃鏌ョ湅鏃ュ織"; \
	fi

status:
	@echo "$(CYAN)=== 鏈嶅姟鐘舵€?===$(NC)"
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l
	@echo ""
	@echo "$(CYAN)=== 绯荤粺璧勬簮 ===$(NC)"
	@ps aux | grep $(BINARY_NAME) | grep -v grep || echo "杩涚▼鏈繍琛?

logs:
	@echo "$(CYAN)=== 瀹炴椂鏃ュ織 (鎸塁trl+C閫€鍑? ===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) -f --no-hostname

logs-recent:
	@echo "$(CYAN)=== 鏈€杩戞棩蹇?===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) --no-hostname -n 50

# === 娴嬭瘯鍜岀淮鎶?===
test-system:
	@echo "$(BLUE)[INFO]$(NC) 娴嬭瘯妯″紡杩愯绯荤粺..."
	@if [ ! -f "$(INSTALL_DIR)/bin/$(BINARY_NAME)" ]; then \
		echo "$(RED)[ERROR]$(NC) 绯荤粺鏈畨瑁咃紝璇峰厛杩愯 make install"; \
		exit 1; \
	fi
	@cd $(INSTALL_DIR)/bin && sudo ./$(BINARY_NAME) --test --verbose --config $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml

test:
	@echo "$(BLUE)[INFO]$(NC) 杩愯绯荤粺娴嬭瘯..."
	@if [ -f "cpp_backend/tests/run_tests.sh" ]; then \
		cd cpp_backend && bash tests/run_tests.sh; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) 娴嬭瘯鑴氭湰涓嶅瓨鍦?; \
	fi

backup:
	@echo "$(BLUE)[INFO]$(NC) 鍒涘缓绯荤粺澶囦唤..."
	@BACKUP_NAME="bamboo-system-backup-$$(date +%Y%m%d-%H%M%S)"; \
	sudo mkdir -p /opt/backup; \
	sudo tar -czf /opt/backup/$$BACKUP_NAME.tar.gz \
		-C $(INSTALL_DIR) . \
		--exclude=logs \
		--exclude=backup; \
	echo "$(GREEN)[SUCCESS]$(NC) 澶囦唤宸插垱寤? /opt/backup/$$BACKUP_NAME.tar.gz"

# 蹇€熼噸鏂伴儴缃诧紙璺宠繃渚濊禆妫€鏌ワ級
redeploy: stop clean build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺閲嶆柊閮ㄧ讲瀹屾垚锛?

# 瀹屾暣閲嶆柊閮ㄧ讲锛堝寘鎷緷璧栨鏌ワ級
full-redeploy: stop install-deps build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺瀹屾暣閲嶆柊閮ㄧ讲瀹屾垚锛?

# 鏅鸿兘閲嶆柊閮ㄧ讲锛堜粎鍦ㄥ繀瑕佹椂瀹夎渚濊禆锛?
smart-redeploy: stop check-deps-if-needed build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 鏅鸿兘閲嶆柊閮ㄧ讲瀹屾垚锛?

# 妫€鏌ヤ緷璧栨槸鍚﹂渶瑕侀噸鏂板畨瑁?
check-deps-if-needed:
	@echo "$(BLUE)[INFO]$(NC) 妫€鏌ユ槸鍚﹂渶瑕侀噸鏂板畨瑁呬緷璧?.."
	@NEED_DEPS=false; \
	if ! PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL鏈壘鍒帮紝闇€瑕佸畨瑁呬緷璧?; \
		NEED_DEPS=true; \
	elif [ "$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null | cut -d. -f1)" != "9" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL鐗堟湰涓嶆槸v9锛岄渶瑕佹洿鏂?; \
		NEED_DEPS=true; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) 渚濊禆宸叉弧瓒筹紝璺宠繃瀹夎姝ラ"; \
	fi; \
	if [ "$$NEED_DEPS" = "true" ]; then \
		echo "$(CYAN)[INSTALL]$(NC) 瀹夎缂哄け鐨勪緷璧?.."; \
		$(MAKE) install-deps; \
	fi

# === 娓呯悊 ===
clean:
	@echo "$(BLUE)[INFO]$(NC) 娓呯悊鏋勫缓鐩綍..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_DIR)_debug
	@echo "$(GREEN)[SUCCESS]$(NC) 娓呯悊瀹屾垚"

uninstall:
	@echo "$(BLUE)[INFO]$(NC) 鍗歌浇绯荤粺..."
	@sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true
	@sudo systemctl disable $(SERVICE_NAME) 2>/dev/null || true
	@sudo rm -f /etc/systemd/system/$(SERVICE_NAME).service
	@sudo systemctl daemon-reload
	@sudo rm -rf $(INSTALL_DIR)
	@echo "$(GREEN)[SUCCESS]$(NC) 绯荤粺宸插嵏杞?

# === 鍗曡繘绋嬬粺涓€鏋舵瀯 ===
unified: unified-build
	@echo "$(GREEN)[SUCCESS]$(NC) 鍗曡繘绋嬬粺涓€鏋舵瀯鏋勫缓瀹屾垚"

unified-build:
	@echo "$(CYAN)[UNIFIED]$(NC) 鏋勫缓鍗曡繘绋婰VGL+GStreamer缁熶竴鏋舵瀯..."
	@PKG_CONFIG_PATH=/usr/local/lib/pkgconfig g++ -o simple_unified_main \
		simple_unified_main.cpp \
		$$(pkg-config --cflags --libs lvgl) \
		$$(pkg-config --cflags --libs gstreamer-1.0) \
		$$(pkg-config --cflags --libs gstreamer-app-1.0) \
		-lEGL -lpthread \
		-std=c++17 -O2 -DENABLE_LVGL=1
	@echo "$(GREEN)[SUCCESS]$(NC) 缁熶竴鏋舵瀯缂栬瘧瀹屾垚: ./simple_unified_main"

unified-run:
	@echo "$(BLUE)[INFO]$(NC) 杩愯鍗曡繘绋嬬粺涓€鏋舵瀯..."
	@if [ ! -f "./simple_unified_main" ]; then \
		echo "$(RED)[ERROR]$(NC) 缁熶竴鏋舵瀯鍙墽琛屾枃浠朵笉瀛樺湪锛岃鍏堣繍琛?make unified"; \
		exit 1; \
	fi
	@sudo ./simple_unified_main

unified-test:
	@echo "$(BLUE)[INFO]$(NC) 娴嬭瘯鍗曡繘绋嬬粺涓€鏋舵瀯..."
	@echo "$(CYAN)妫€鏌GL鐜...$(NC)"
	@if command -v eglinfo >/dev/null 2>&1; then \
		eglinfo | head -10; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) eglinfo鏈畨瑁咃紝璺宠繃EGL妫€鏌?; \
	fi
	@echo "$(CYAN)妫€鏌RM璁惧...$(NC)"
	@ls -la /dev/dri/ || echo "$(YELLOW)[WARNING]$(NC) DRM璁惧涓嶅彲鐢?
	@echo "$(CYAN)妫€鏌ユ憚鍍忓ご璁惧...$(NC)"
	@ls -la /dev/video* || echo "$(YELLOW)[WARNING]$(NC) 鎽勫儚澶磋澶囦笉鍙敤"
	@echo "$(GREEN)[SUCCESS]$(NC) 鐜妫€鏌ュ畬鎴愶紝杩愯缁熶竴鏋舵瀯..."
	@$(MAKE) unified-run

unified-clean:
	@echo "$(BLUE)[INFO]$(NC) 娓呯悊缁熶竴鏋舵瀯鏋勫缓鏂囦欢..."
	@rm -f simple_unified_main
	@echo "$(GREEN)[SUCCESS]$(NC) 娓呯悊瀹屾垚"

# === 鎽勫儚澶磋瘖鏂伐鍏?===
GSTREAMER_LIBS := $(shell pkg-config --cflags --libs gstreamer-1.0)
EGL_LIBS := -lEGL
PTHREAD_LIBS := -lpthread

camera-diag: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 鏋勫缓鎽勫儚澶磋瘖鏂伐鍏?.."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[RUNNING]$(NC) 杩愯鎽勫儚澶磋瘖鏂?.."
	sudo ./camera_diagnostics

camera-test: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 鏋勫缓鎽勫儚澶存祴璇曞伐鍏?.."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 娴嬭瘯鎽勫儚澶磋闂?(sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_diagnostics test $(or $(SENSOR_ID),0)

camera-fix:
	@echo "$(CYAN)[FIXING]$(NC) 杩愯缁煎悎鎽勫儚澶翠慨澶嶈剼鏈?.."
	./deploy/scripts/camera_fix.sh

camera-fix-quick:
	@echo "$(BLUE)[INFO]$(NC) 搴旂敤蹇€熸憚鍍忓ご淇..."
	@echo "1. 鍋滄鍐茬獊杩涚▼..."
	-sudo pkill nvargus-daemon 2>/dev/null || true
	-sudo pkill gst-launch-1.0 2>/dev/null || true
	@echo "2. 閲嶅惎nvargus-daemon..."
	-sudo systemctl restart nvargus-daemon 2>/dev/null || true
	@echo "3. 璁剧疆璁惧鏉冮檺..."
	sudo chmod 666 /dev/video* 2>/dev/null || true
	sudo chmod 666 /dev/nvhost-* 2>/dev/null || true
	@echo "4. 璁剧疆EGL鐜..."
	@echo "export EGL_PLATFORM=drm" >> ~/.bashrc
	@echo "export __EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl" >> ~/.bashrc
	@echo "$(GREEN)[SUCCESS]$(NC) 蹇€熶慨澶嶅凡搴旂敤锛岃杩愯 'source ~/.bashrc' 骞堕噸璇?

camera-fix-test: test_camera_fix.cpp
	@echo "$(BLUE)[INFO]$(NC) 鏋勫缓鎽勫儚澶翠慨澶嶆祴璇曞伐鍏?.."
	$(CXX) $(CXXFLAGS) -o camera_fix_test test_camera_fix.cpp $(GSTREAMER_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 杩愯鎽勫儚澶翠慨澶嶆祴璇?(sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_fix_test $(or $(SENSOR_ID),0)

# NVIDIA-DRM Migration and Validation
enable-nvidia-drm:
	@echo "$(BLUE)[INFO]$(NC) 鍚敤NVIDIA-DRM椹卞姩..."
	@chmod +x deploy/scripts/enable_nvidia_drm.sh
	@echo "$(YELLOW)[WARNING]$(NC) 姝ゆ搷浣滃皢淇敼绯荤粺椹卞姩閰嶇疆锛岃纭缁х画..."
	@read -p "缁х画鍚敤NVIDIA-DRM? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	sudo deploy/scripts/enable_nvidia_drm.sh

force-nvidia-drm:
	@echo "$(BLUE)[INFO]$(NC) 寮哄埗杩佺Щ鍒癗VIDIA-DRM椹卞姩..."
	@chmod +x deploy/scripts/force_nvidia_drm.sh
	@echo "$(RED)[DANGER]$(NC) 姝ゆ搷浣滃皢寮哄埗淇敼绯荤粺椹卞姩锛屽彲鑳藉奖鍝嶅浘褰㈡樉绀?
	@echo "$(YELLOW)[WARNING]$(NC) 寤鸿鍏堝浠介噸瑕佹暟鎹紝鎿嶄綔闇€瑕侀噸鍚郴缁?
	@read -p "纭寮哄埗杩佺Щ鍒癗VIDIA-DRM? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	sudo deploy/scripts/force_nvidia_drm.sh

nvidia-drm-test: nvidia_drm_migration_test.cpp
	@echo "$(BLUE)[INFO]$(NC) 鏋勫缓NVIDIA-DRM杩佺Щ楠岃瘉宸ュ叿..."
	$(CXX) $(CXXFLAGS) -o nvidia_drm_migration_test nvidia_drm_migration_test.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS) -ldrm -lm
	@echo "$(CYAN)[TESTING]$(NC) 杩愯NVIDIA-DRM杩佺Щ瀹屾暣楠岃瘉..."
	sudo ./nvidia_drm_migration_test

nvidia-drm-report:
	@echo "$(CYAN)[REPORT]$(NC) 鐢熸垚NVIDIA-DRM杩佺Щ鐘舵€佹姤鍛?.."
	@echo "=== NVIDIA-DRM 杩佺Щ鐘舵€佹姤鍛?===" > nvidia_drm_status.txt
	@echo "鐢熸垚鏃堕棿: $$(date)" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 椹卞姩妯″潡鐘舵€?===" >> nvidia_drm_status.txt
	@lsmod | grep -E "nvidia|tegra|drm" >> nvidia_drm_status.txt 2>/dev/null || echo "鏈壘鍒扮浉鍏虫ā鍧? >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== DRM璁惧鐘舵€?===" >> nvidia_drm_status.txt
	@ls -la /dev/dri/ >> nvidia_drm_status.txt 2>/dev/null || echo "DRM璁惧涓嶅瓨鍦? >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== EGL鐜 ===" >> nvidia_drm_status.txt
	@echo "EGL_PLATFORM=$$EGL_PLATFORM" >> nvidia_drm_status.txt
	@echo "__EGL_VENDOR_LIBRARY_DIRS=$$__EGL_VENDOR_LIBRARY_DIRS" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 绯荤粺淇℃伅 ===" >> nvidia_drm_status.txt
	@uname -a >> nvidia_drm_status.txt
	@echo "$(GREEN)[SUCCESS]$(NC) 鐘舵€佹姤鍛婂凡淇濆瓨鍒? nvidia_drm_status.txt"
	@cat nvidia_drm_status.txt

nvidia-drm-complete: nvidia-drm-test nvidia-drm-report
	@echo "$(GREEN)[COMPLETE]$(NC) NVIDIA-DRM杩佺Щ楠岃瘉鍏ㄩ儴瀹屾垚锛?
	@echo "鏌ョ湅瀹屾暣鎶ュ憡:"
	@echo "  楠岃瘉鎶ュ憡: nvidia_drm_migration_report.txt"
	@echo "  鐘舵€佹姤鍛? nvidia_drm_status.txt"

.PHONY: camera-diag camera-test camera-fix camera-fix-quick camera-fix-test enable-nvidia-drm force-nvidia-drm nvidia-drm-test nvidia-drm-report nvidia-drm-complete

# === 寮€鍙戣緟鍔?===
dev-run:
	@echo "$(BLUE)[INFO]$(NC) 寮€鍙戞ā寮忕洿鎺ヨ繍琛?.."
	@if [ ! -f "$(BUILD_DIR)/bamboo_integrated" ]; then \
		echo "$(RED)[ERROR]$(NC) 鍙墽琛屾枃浠朵笉瀛樺湪锛岃鍏堟瀯寤虹郴缁?; \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && sudo ./bamboo_integrated --verbose --config ../config/system_config.yaml

monitor:
	@echo "$(CYAN)=== 绯荤粺鐩戞帶 (鎸塁trl+C閫€鍑? ===$(NC)"
	@while true; do \
		clear; \
		echo "$(GREEN)鏃堕棿: $$(date)$(NC)"; \
		echo "$(CYAN)鏈嶅姟鐘舵€?$(NC)"; \
		systemctl is-active $(SERVICE_NAME) 2>/dev/null || echo "鏈繍琛?; \
		echo "$(CYAN)绯荤粺璧勬簮:$(NC)"; \
		ps aux | grep $(BINARY_NAME) | grep -v grep | head -5 || echo "杩涚▼鏈繍琛?; \
		echo "$(CYAN)鍐呭瓨浣跨敤:$(NC)"; \
		free -h | head -2; \
		echo "$(CYAN)纾佺洏浣跨敤:$(NC)"; \
		df -h / | tail -1; \
		sleep 5; \
	done

# 纭繚渚濊禆鍏崇郴
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

.DEFAULT_GOAL := help

