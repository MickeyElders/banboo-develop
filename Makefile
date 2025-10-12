# AI竹子识别系统 - C++ LVGL一体化构建和部署脚本
# 版本: 5.0.0 (C++ LVGL Integrated Architecture)
# C++推理后端 + LVGL界面 + Modbus通信的完整一体化系统

.PHONY: all install clean test deploy start stop restart status logs \
        install-deps install-system-deps install-lvgl build-lvgl-from-source \
        install-service enable-service disable-service \
        check-system build-system install-system setup-config \
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

all: check-system install-deps build-system
	@echo "$(CYAN)=== AI竹子识别系统构建完成 (v$(VERSION)) ===$(NC)"
	@echo "$(GREEN)C++ LVGL一体化工业级嵌入式架构$(NC)"
	@echo "使用 'make deploy' 完成系统部署"

install: all install-system install-service
	@echo "$(GREEN)=== 系统安装完成 ===$(NC)"
	@echo "服务名称: $(SERVICE_NAME)"
	@echo "安装目录: $(INSTALL_DIR)"
	@echo "可执行文件: $(INSTALL_DIR)/bin/$(BINARY_NAME)"
	@echo "使用 'make start' 启动系统"

deploy: install enable-service start
	@echo "$(GREEN)[SUCCESS]$(NC) 系统部署完成！"

help:
	@echo "$(CYAN)===============================================$(NC)"
	@echo "$(CYAN)   AI竹子识别系统 C++ LVGL构建系统 v$(VERSION)$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)快速部署命令:$(NC)"
	@echo "  deploy           - 首次完整部署(构建+安装+启动服务)"
	@echo "  redeploy         - 代码修改后快速重新部署(跳过依赖检查)"
	@echo "  smart-redeploy   - 智能重新部署(仅在必要时安装依赖)"
	@echo "  full-redeploy    - 完整重新部署(包括依赖检查)"
	@echo "  backup           - 创建当前系统备份"
	@echo "  test-system      - 测试模式运行系统"
	@echo ""
	@echo "$(GREEN)构建命令:$(NC)"
	@echo "  all              - 检查系统+安装依赖+构建系统"
	@echo "  build-system     - 构建C++ LVGL系统"
	@echo "  build-debug      - 构建调试版本"
	@echo "  clean            - 清理构建目录"
	@echo ""
	@echo "$(GREEN)单进程统一架构:$(NC)"
	@echo "  unified          - 构建单进程LVGL+GStreamer统一架构"
	@echo "  unified-run      - 运行统一架构"
	@echo "  unified-test     - 测试统一架构和环境"
	@echo "  unified-clean    - 清理统一架构构建文件"
	@echo ""
	@echo "$(GREEN)摄像头诊断工具:$(NC)"
	@echo "  camera-diag      - 运行完整摄像头诊断"
	@echo "  camera-test      - 测试摄像头访问 (使用 SENSOR_ID=X 指定sensor)"
	@echo "  camera-fix       - 运行综合交互式摄像头修复脚本"
	@echo "  camera-fix-quick - 应用快速非交互式摄像头修复"
	@echo "  camera-fix-test  - 测试摄像头修复后功能 (使用 SENSOR_ID=X)"
	@echo ""
	@echo "$(GREEN)NVIDIA-DRM 迁移验证:$(NC)"
	@echo "  enable-nvidia-drm- 启用NVIDIA-DRM驱动（替换tegra_drm）"
	@echo "  nvidia-drm-test  - 运行完整的NVIDIA-DRM迁移验证测试"
	@echo "  nvidia-drm-report- 生成NVIDIA-DRM迁移状态报告"
	@echo "  nvidia-drm-complete - 运行完整的迁移验证流程"
	@echo ""
	@echo "$(GREEN)安装命令:$(NC)"
	@echo "  install          - 完整安装系统"
	@echo "  install-deps     - 安装所有依赖(系统+LVGL)"
	@echo "  install-system-deps - 仅安装系统依赖"
	@echo "  install-lvgl     - 检查并安装LVGL"
	@echo "  install-system   - 安装编译好的系统"
	@echo "  install-service  - 安装systemd服务"
	@echo ""
	@echo "$(GREEN)服务管理:$(NC)"
	@echo "  start            - 启动服务"
	@echo "  stop             - 停止服务"
	@echo "  restart          - 重启服务"
	@echo "  status           - 查看服务状态"
	@echo "  logs             - 查看服务日志"
	@echo "  enable-service   - 启用开机自启"
	@echo "  disable-service  - 禁用开机自启"
	@echo ""
	@echo "$(GREEN)维护命令:$(NC)"
	@echo "  check-system     - 检查系统环境"
	@echo "  setup-config     - 设置配置文件"
	@echo "  test             - 运行系统测试"
	@echo "  backup           - 备份当前系统"
	@echo ""
	@echo "$(YELLOW)特性说明:$(NC)"
	@echo "  ✓ C++17高性能推理引擎"
	@echo "  ✓ LVGL工业级触摸界面"
	@echo "  ✓ YOLOv8+TensorRT加速"
	@echo "  ✓ Modbus TCP PLC通信"
	@echo "  ✓ Jetson Orin NX优化"
	@echo "  ✓ 实时视频处理与检测"

# === 系统检查 ===
check-system:
	@echo "$(BLUE)[INFO]$(NC) 检查系统环境..."
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) cmake未安装"; \
		exit 1; \
	fi
	@if ! command -v g++ >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(NC) g++编译器未安装"; \
		exit 1; \
	fi
	@if ! pkg-config --exists opencv4 2>/dev/null; then \
		if ! pkg-config --exists opencv 2>/dev/null; then \
			echo "$(RED)[ERROR]$(NC) OpenCV开发库未安装"; \
			exit 1; \
		fi; \
	fi
	@if ! pkg-config --exists gstreamer-1.0 2>/dev/null; then \
		echo "$(RED)[ERROR]$(NC) GStreamer开发库未安装"; \
		exit 1; \
	fi
	@if [ ! -f "/usr/include/modbus/modbus.h" ] && [ ! -f "/usr/local/include/modbus/modbus.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) libmodbus开发库未找到，将禁用Modbus功能"; \
	fi
	@if [ ! -f "/usr/include/lvgl/lvgl.h" ] && [ ! -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL开发库未找到，将禁用界面功能"; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 系统环境检查通过"

# === 依赖安装 ===
install-deps: install-system-deps install-lvgl9-auto
	@echo "$(GREEN)[SUCCESS]$(NC) 所有依赖安装完成"

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
		libdrm-dev
	@if lspci | grep -i nvidia >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) 检测到NVIDIA GPU，检查CUDA环境..."; \
		if [ -d "/usr/local/cuda" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) CUDA环境已安装"; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) CUDA环境未安装，请手动安装CUDA和TensorRT"; \
		fi \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) 系统依赖安装完成"

install-lvgl:
	@echo "$(CYAN)[LVGL]$(NC) 检查LVGL v9安装状态..."
	@LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null || echo "not_found"); \
	if [ "$$LVGL_VERSION" = "not_found" ] || [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" != "9" ]; then \
		echo "$(BLUE)[INFO]$(NC) LVGL v9未找到 (当前版本: $$LVGL_VERSION)，开始从源码编译安装..."; \
		$(MAKE) build-lvgl-from-source; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL v9已安装 (版本: $$LVGL_VERSION)"; \
	fi

build-lvgl-from-source:
	@echo "$(CYAN)[LVGL]$(NC) === 完全手动安装LVGL v9.1 ==="
	@echo "$(BLUE)[INFO]$(NC) [1/8] 清理旧文件..."
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
	@echo "$(BLUE)[INFO]$(NC) 手动确保头文件安装..."
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
	@echo -n "$(BLUE)[INFO]$(NC) 头文件: "
	@ls /usr/local/include/lvgl/lvgl.h >/dev/null 2>&1 && echo "$(GREEN)✓$(NC)" || (echo "$(RED)✗ 失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) 库文件: "
	@ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1 && echo "$(GREEN)✓$(NC)" || (echo "$(RED)✗ 失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) pkg-config: "
	@PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl && echo "$(GREEN)✓$(NC)" || (echo "$(RED)✗ 失败$(NC)" && exit 1)
	@echo -n "$(BLUE)[INFO]$(NC) v9 API: "
	@grep -q "lv_display_create" /usr/local/include/lvgl/lvgl.h && echo "$(GREEN)✓$(NC)" || echo "$(YELLOW)⚠ 未检测到但可能正常$(NC)"
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) === LVGL v9.1安装完成 ==="
	@rm -rf /tmp/lvgl

# 安装LVGL v9的快速命令
install-lvgl9: build-lvgl-from-source
	@echo "$(GREEN)[SUCCESS]$(NC) LVGL v9.3安装完成，系统已准备就绪"

# 自动检查和安装LVGL v9（编译前自动执行）
install-lvgl9-auto:
	@echo "$(CYAN)[AUTO-INSTALL]$(NC) === 智能检测LVGL v9安装状态 ==="
	@echo "$(BLUE)[INFO]$(NC) 正在检测LVGL v9安装状态..."
	@LVGL_INSTALLED=false; \
	LVGL_VERSION_OK=false; \
	LVGL_API_OK=false; \
	if PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		LVGL_VERSION=$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null); \
		echo "$(BLUE)[INFO]$(NC) 发现已安装的LVGL版本: $$LVGL_VERSION"; \
		if [ "$$(echo $$LVGL_VERSION | cut -d. -f1)" = "9" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL主版本为v9 ✓"; \
			LVGL_VERSION_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) LVGL版本不是v9 (当前: $$LVGL_VERSION)"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) pkg-config未找到LVGL"; \
	fi; \
	if [ -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL头文件存在 ✓"; \
		if grep -q "lv_display_create\|lv_disp_create" /usr/local/include/lvgl/lvgl.h 2>/dev/null; then \
			echo "$(GREEN)[SUCCESS]$(NC) LVGL v9 API可用 ✓"; \
			LVGL_API_OK=true; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) 未检测到LVGL v9 API"; \
		fi; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) LVGL头文件不存在"; \
	fi; \
	if ls /usr/local/lib/liblvgl.so* >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) LVGL库文件存在 ✓"; \
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

# === 系统安装 ===
install-system:
	@echo "$(BLUE)[INFO]$(NC) 安装C++ LVGL系统到$(INSTALL_DIR)..."
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "$(RED)[ERROR]$(NC) 构建目录不存在，请先运行 make build-system"; \
		exit 1; \
	fi
	@sudo mkdir -p $(INSTALL_DIR)
	@cd $(BUILD_DIR) && sudo make install
	@sudo mkdir -p $(INSTALL_DIR)/logs
	@sudo mkdir -p $(INSTALL_DIR)/backup
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
	@echo "$(GREEN)[SUCCESS]$(NC) 服务已启用开机自启"

disable-service:
	@sudo systemctl disable $(SERVICE_NAME)
	@echo "$(BLUE)[INFO]$(NC) 服务已禁用开机自启"

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
	@echo "$(GREEN)[SUCCESS]$(NC) 服务已停止"

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
	@echo "$(CYAN)=== 服务状态 ===$(NC)"
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l
	@echo ""
	@echo "$(CYAN)=== 系统资源 ===$(NC)"
	@ps aux | grep $(BINARY_NAME) | grep -v grep || echo "进程未运行"

logs:
	@echo "$(CYAN)=== 实时日志 (按Ctrl+C退出) ===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) -f --no-hostname

logs-recent:
	@echo "$(CYAN)=== 最近日志 ===$(NC)"
	@sudo journalctl -u $(SERVICE_NAME) --no-hostname -n 50

# === 测试和维护 ===
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
		echo "$(YELLOW)[WARNING]$(NC) 测试脚本不存在"; \
	fi

backup:
	@echo "$(BLUE)[INFO]$(NC) 创建系统备份..."
	@BACKUP_NAME="bamboo-system-backup-$$(date +%Y%m%d-%H%M%S)"; \
	sudo mkdir -p /opt/backup; \
	sudo tar -czf /opt/backup/$$BACKUP_NAME.tar.gz \
		-C $(INSTALL_DIR) . \
		--exclude=logs \
		--exclude=backup; \
	echo "$(GREEN)[SUCCESS]$(NC) 备份已创建: /opt/backup/$$BACKUP_NAME.tar.gz"

# 快速重新部署（跳过依赖检查）
redeploy: stop clean build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 系统重新部署完成！"

# 完整重新部署（包括依赖检查）
full-redeploy: stop install-deps build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 系统完整重新部署完成！"

# 智能重新部署（仅在必要时安装依赖）
smart-redeploy: stop check-deps-if-needed build-system install-system restart
	@echo "$(GREEN)[SUCCESS]$(NC) 智能重新部署完成！"

# 检查依赖是否需要重新安装
check-deps-if-needed:
	@echo "$(BLUE)[INFO]$(NC) 检查是否需要重新安装依赖..."
	@NEED_DEPS=false; \
	if ! PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --exists lvgl 2>/dev/null; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL未找到，需要安装依赖"; \
		NEED_DEPS=true; \
	elif [ "$$(PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion lvgl 2>/dev/null | cut -d. -f1)" != "9" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) LVGL版本不是v9，需要更新"; \
		NEED_DEPS=true; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) 依赖已满足，跳过安装步骤"; \
	fi; \
	if [ "$$NEED_DEPS" = "true" ]; then \
		echo "$(CYAN)[INSTALL]$(NC) 安装缺失的依赖..."; \
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
	@echo "$(GREEN)[SUCCESS]$(NC) 系统已卸载"

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
		echo "$(RED)[ERROR]$(NC) 统一架构可执行文件不存在，请先运行 make unified"; \
		exit 1; \
	fi
	@sudo ./simple_unified_main

unified-test:
	@echo "$(BLUE)[INFO]$(NC) 测试单进程统一架构..."
	@echo "$(CYAN)检查EGL环境...$(NC)"
	@if command -v eglinfo >/dev/null 2>&1; then \
		eglinfo | head -10; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) eglinfo未安装，跳过EGL检查"; \
	fi
	@echo "$(CYAN)检查DRM设备...$(NC)"
	@ls -la /dev/dri/ || echo "$(YELLOW)[WARNING]$(NC) DRM设备不可用"
	@echo "$(CYAN)检查摄像头设备...$(NC)"
	@ls -la /dev/video* || echo "$(YELLOW)[WARNING]$(NC) 摄像头设备不可用"
	@echo "$(GREEN)[SUCCESS]$(NC) 环境检查完成，运行统一架构..."
	@$(MAKE) unified-run

unified-clean:
	@echo "$(BLUE)[INFO]$(NC) 清理统一架构构建文件..."
	@rm -f simple_unified_main
	@echo "$(GREEN)[SUCCESS]$(NC) 清理完成"

# === 摄像头诊断工具 ===
GSTREAMER_LIBS := $(shell pkg-config --cflags --libs gstreamer-1.0)
EGL_LIBS := -lEGL
PTHREAD_LIBS := -lpthread

camera-diag: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头诊断工具..."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[RUNNING]$(NC) 运行摄像头诊断..."
	sudo ./camera_diagnostics

camera-test: cpp_backend/src/utils/camera_diagnostics.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头测试工具..."
	$(CXX) $(CXXFLAGS) -o camera_diagnostics \
		cpp_backend/src/utils/camera_diagnostics.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 测试摄像头访问 (sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_diagnostics test $(or $(SENSOR_ID),0)

camera-fix:
	@echo "$(CYAN)[FIXING]$(NC) 运行综合摄像头修复脚本..."
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
	@echo "$(GREEN)[SUCCESS]$(NC) 快速修复已应用，请运行 'source ~/.bashrc' 并重试"

camera-fix-test: test_camera_fix.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建摄像头修复测试工具..."
	$(CXX) $(CXXFLAGS) -o camera_fix_test test_camera_fix.cpp $(GSTREAMER_LIBS)
	@echo "$(CYAN)[TESTING]$(NC) 运行摄像头修复测试 (sensor-id=$(or $(SENSOR_ID),0))..."
	sudo ./camera_fix_test $(or $(SENSOR_ID),0)

# NVIDIA-DRM Migration and Validation
enable-nvidia-drm:
	@echo "$(BLUE)[INFO]$(NC) 启用NVIDIA-DRM驱动..."
	@chmod +x deploy/scripts/enable_nvidia_drm.sh
	@echo "$(YELLOW)[WARNING]$(NC) 此操作将修改系统驱动配置，请确认继续..."
	@read -p "继续启用NVIDIA-DRM? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	sudo deploy/scripts/enable_nvidia_drm.sh

nvidia-drm-test: nvidia_drm_migration_test.cpp
	@echo "$(BLUE)[INFO]$(NC) 构建NVIDIA-DRM迁移验证工具..."
	$(CXX) $(CXXFLAGS) -o nvidia_drm_migration_test nvidia_drm_migration_test.cpp \
		$(GSTREAMER_LIBS) $(EGL_LIBS) $(PTHREAD_LIBS) -ldrm -lm
	@echo "$(CYAN)[TESTING]$(NC) 运行NVIDIA-DRM迁移完整验证..."
	sudo ./nvidia_drm_migration_test

nvidia-drm-report:
	@echo "$(CYAN)[REPORT]$(NC) 生成NVIDIA-DRM迁移状态报告..."
	@echo "=== NVIDIA-DRM 迁移状态报告 ===" > nvidia_drm_status.txt
	@echo "生成时间: $$(date)" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 驱动模块状态 ===" >> nvidia_drm_status.txt
	@lsmod | grep -E "nvidia|tegra|drm" >> nvidia_drm_status.txt 2>/dev/null || echo "未找到相关模块" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== DRM设备状态 ===" >> nvidia_drm_status.txt
	@ls -la /dev/dri/ >> nvidia_drm_status.txt 2>/dev/null || echo "DRM设备不存在" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== EGL环境 ===" >> nvidia_drm_status.txt
	@echo "EGL_PLATFORM=$$EGL_PLATFORM" >> nvidia_drm_status.txt
	@echo "__EGL_VENDOR_LIBRARY_DIRS=$$__EGL_VENDOR_LIBRARY_DIRS" >> nvidia_drm_status.txt
	@echo "" >> nvidia_drm_status.txt
	@echo "=== 系统信息 ===" >> nvidia_drm_status.txt
	@uname -a >> nvidia_drm_status.txt
	@echo "$(GREEN)[SUCCESS]$(NC) 状态报告已保存到: nvidia_drm_status.txt"
	@cat nvidia_drm_status.txt

nvidia-drm-complete: nvidia-drm-test nvidia-drm-report
	@echo "$(GREEN)[COMPLETE]$(NC) NVIDIA-DRM迁移验证全部完成！"
	@echo "查看完整报告:"
	@echo "  验证报告: nvidia_drm_migration_report.txt"
	@echo "  状态报告: nvidia_drm_status.txt"

.PHONY: camera-diag camera-test camera-fix camera-fix-quick camera-fix-test enable-nvidia-drm nvidia-drm-test nvidia-drm-report nvidia-drm-complete

# === 开发辅助 ===
dev-run:
	@echo "$(BLUE)[INFO]$(NC) 开发模式直接运行..."
	@if [ ! -f "$(BUILD_DIR)/bamboo_integrated" ]; then \
		echo "$(RED)[ERROR]$(NC) 可执行文件不存在，请先构建系统"; \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && sudo ./bamboo_integrated --verbose --config ../config/system_config.yaml

monitor:
	@echo "$(CYAN)=== 系统监控 (按Ctrl+C退出) ===$(NC)"
	@while true; do \
		clear; \
		echo "$(GREEN)时间: $$(date)$(NC)"; \
		echo "$(CYAN)服务状态:$(NC)"; \
		systemctl is-active $(SERVICE_NAME) 2>/dev/null || echo "未运行"; \
		echo "$(CYAN)系统资源:$(NC)"; \
		ps aux | grep $(BINARY_NAME) | grep -v grep | head -5 || echo "进程未运行"; \
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