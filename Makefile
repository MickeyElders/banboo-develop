# AI竹子识别系统 - C++ LVGL一体化构建和部署脚本
# 版本: 5.0.0 (C++ LVGL Integrated Architecture)
# C++推理后端 + LVGL界面 + Modbus通信的完整一体化系统

.PHONY: all install clean test deploy start stop restart status logs \
        install-deps install-service enable-service disable-service \
        check-system build-system install-system setup-config \
        build-debug test-system backup

# === 系统配置 ===
PROJECT_NAME := bamboo-recognition-system
VERSION := 5.0.0
INSTALL_DIR := /opt/bamboo-cut
SERVICE_NAME := bamboo-cpp-lvgl
BINARY_NAME := bamboo-recognition

# === C++ LVGL构建配置 ===
BUILD_DIR := cpp_backend/build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
               -DENABLE_AI_OPTIMIZATION=ON \
               -DENABLE_TENSORRT=ON \
               -DENABLE_CUDA=ON \
               -DENABLE_MODBUS=ON \
               -DENABLE_LVGL=ON \
               -DENABLE_HARDWARE_ACCELERATION=ON

# === 颜色定义 ===
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# === 日志函数 ===
define log_info
	@echo "$(BLUE)[INFO]$(NC) $1"
endef

define log_success
	@echo "$(GREEN)[SUCCESS]$(NC) $1"
endef

define log_warning
	@echo "$(YELLOW)[WARNING]$(NC) $1"
endef

define log_error
	@echo "$(RED)[ERROR]$(NC) $1"
endef

define log_highlight
	@echo "$(CYAN)[C++ LVGL]$(NC) $1"
endef

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
	$(call log_success,"系统部署完成！")

help:
	@echo "$(CYAN)===============================================$(NC)"
	@echo "$(CYAN)   AI竹子识别系统 C++ LVGL构建系统 v$(VERSION)$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)快速部署命令:$(NC)"
	@echo "  deploy           - 首次完整部署(构建+安装+启动服务)"
	@echo "  redeploy         - 代码修改后快速重新部署"
	@echo "  backup           - 创建当前系统备份"
	@echo "  test-system      - 测试模式运行系统"
	@echo ""
	@echo "$(GREEN)构建命令:$(NC)"
	@echo "  all              - 检查系统+安装依赖+构建系统"
	@echo "  build-system     - 构建C++ LVGL系统"
	@echo "  build-debug      - 构建调试版本"
	@echo "  clean            - 清理构建目录"
	@echo ""
	@echo "$(GREEN)安装命令:$(NC)"
	@echo "  install          - 完整安装系统"
	@echo "  install-deps     - 安装系统依赖"
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
	$(call log_info,"检查系统环境...")
	@if ! command -v cmake >/dev/null 2>&1; then \
		$(call log_error,"cmake未安装"); \
		exit 1; \
	fi
	@if ! command -v g++ >/dev/null 2>&1; then \
		$(call log_error,"g++编译器未安装"); \
		exit 1; \
	fi
	@if ! pkg-config --exists opencv4 2>/dev/null; then \
		if ! pkg-config --exists opencv 2>/dev/null; then \
			$(call log_error,"OpenCV开发库未安装"); \
			exit 1; \
		fi \
	fi
	@if ! pkg-config --exists gstreamer-1.0 2>/dev/null; then \
		$(call log_error,"GStreamer开发库未安装"); \
		exit 1; \
	fi
	@if [ ! -f "/usr/include/modbus/modbus.h" ] && [ ! -f "/usr/local/include/modbus/modbus.h" ]; then \
		$(call log_warning,"libmodbus开发库未找到，将禁用Modbus功能"); \
	fi
	@if [ ! -f "/usr/include/lvgl/lvgl.h" ] && [ ! -f "/usr/local/include/lvgl/lvgl.h" ]; then \
		$(call log_warning,"LVGL开发库未找到，将禁用界面功能"); \
	fi
	$(call log_success,"系统环境检查通过")

# === 依赖安装 ===
install-deps:
	$(call log_info,"安装系统依赖...")
	@sudo apt-get update
	@sudo apt-get install -y \
		build-essential \
		cmake \
		pkg-config \
		libopencv-dev \
		libgstreamer1.0-dev \
		libgstreamer-plugins-base1.0-dev \
		libmodbus-dev \
		liblvgl-dev \
		libcurl4-openssl-dev \
		libjson-c-dev \
		libsystemd-dev
	@if lspci | grep -i nvidia >/dev/null 2>&1; then \
		$(call log_info,"检测到NVIDIA GPU，检查CUDA环境..."); \
		if [ -d "/usr/local/cuda" ]; then \
			$(call log_success,"CUDA环境已安装"); \
		else \
			$(call log_warning,"CUDA环境未安装，请手动安装CUDA和TensorRT"); \
		fi \
	fi
	$(call log_success,"系统依赖安装完成")

# === C++系统构建 ===
build-system:
	$(call log_highlight,"开始构建C++ LVGL一体化系统...")
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	$(call log_success,"C++ LVGL系统构建完成")

build-debug:
	$(call log_highlight,"构建调试版本...")
	@mkdir -p $(BUILD_DIR)_debug
	@cd $(BUILD_DIR)_debug && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DENABLE_AI_OPTIMIZATION=ON \
		-DENABLE_MODBUS=ON \
		-DENABLE_LVGL=ON
	@cd $(BUILD_DIR)_debug && make -j$(shell nproc)
	$(call log_success,"调试版本构建完成")

# === 系统安装 ===
install-system:
	$(call log_info,"安装C++ LVGL系统到$(INSTALL_DIR)...")
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		$(call log_error,"构建目录不存在，请先运行 make build-system"); \
		exit 1; \
	fi
	@sudo mkdir -p $(INSTALL_DIR)
	@cd $(BUILD_DIR) && sudo make install
	@sudo mkdir -p $(INSTALL_DIR)/logs
	@sudo mkdir -p $(INSTALL_DIR)/backup
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/logs
	@sudo chown -R $(USER):$(USER) $(INSTALL_DIR)/backup
	$(call log_success,"系统安装完成")

# === 配置设置 ===
setup-config:
	$(call log_info,"设置系统配置...")
	@sudo mkdir -p $(INSTALL_DIR)/etc/bamboo-recognition
	@if [ ! -f "$(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml" ]; then \
		sudo cp config/system_config.yaml $(INSTALL_DIR)/etc/bamboo-recognition/ 2>/dev/null || \
		echo "# C++ LVGL一体化系统配置" | sudo tee $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml >/dev/null; \
	fi
	@sudo chmod 644 $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml
	$(call log_success,"配置设置完成")

# === 服务管理 ===
install-service: setup-config
	$(call log_info,"安装systemd服务...")
	@if [ -f "$(BUILD_DIR)/bamboo-cpp-lvgl.service" ]; then \
		sudo cp $(BUILD_DIR)/bamboo-cpp-lvgl.service /etc/systemd/system/; \
	else \
		$(call log_error,"服务文件未生成，请检查CMake构建"); \
		exit 1; \
	fi
	@sudo systemctl daemon-reload
	$(call log_success,"服务安装完成")

enable-service:
	@sudo systemctl enable $(SERVICE_NAME)
	$(call log_success,"服务已启用开机自启")

disable-service:
	@sudo systemctl disable $(SERVICE_NAME)
	$(call log_info,"服务已禁用开机自启")

start:
	$(call log_info,"启动$(SERVICE_NAME)服务...")
	@sudo systemctl start $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		$(call log_success,"服务启动成功"); \
	else \
		$(call log_error,"服务启动失败，请查看日志"); \
		exit 1; \
	fi

stop:
	$(call log_info,"停止$(SERVICE_NAME)服务...")
	@sudo systemctl stop $(SERVICE_NAME)
	$(call log_success,"服务已停止")

restart:
	$(call log_info,"重启$(SERVICE_NAME)服务...")
	@sudo systemctl restart $(SERVICE_NAME)
	@sleep 3
	@if sudo systemctl is-active --quiet $(SERVICE_NAME); then \
		$(call log_success,"服务重启成功"); \
	else \
		$(call log_error,"服务重启失败，请查看日志"); \
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
	$(call log_info,"测试模式运行系统...")
	@if [ ! -f "$(INSTALL_DIR)/bin/$(BINARY_NAME)" ]; then \
		$(call log_error,"系统未安装，请先运行 make install"); \
		exit 1; \
	fi
	@cd $(INSTALL_DIR)/bin && sudo ./$(BINARY_NAME) --test --verbose --config $(INSTALL_DIR)/etc/bamboo-recognition/system_config.yaml

test:
	$(call log_info,"运行系统测试...")
	@if [ -f "cpp_backend/tests/run_tests.sh" ]; then \
		cd cpp_backend && bash tests/run_tests.sh; \
	else \
		$(call log_warning,"测试脚本不存在"); \
	fi

backup:
	$(call log_info,"创建系统备份...")
	@BACKUP_NAME="bamboo-system-backup-$$(date +%Y%m%d-%H%M%S)"; \
	sudo mkdir -p /opt/backup; \
	sudo tar -czf /opt/backup/$$BACKUP_NAME.tar.gz \
		-C $(INSTALL_DIR) . \
		--exclude=logs \
		--exclude=backup; \
	$(call log_success,"备份已创建: /opt/backup/$$BACKUP_NAME.tar.gz")

redeploy: stop build-system install-system restart
	$(call log_success,"系统重新部署完成！")

# === 清理 ===
clean:
	$(call log_info,"清理构建目录...")
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_DIR)_debug
	$(call log_success,"清理完成")

uninstall:
	$(call log_info,"卸载系统...")
	@sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true
	@sudo systemctl disable $(SERVICE_NAME) 2>/dev/null || true
	@sudo rm -f /etc/systemd/system/$(SERVICE_NAME).service
	@sudo systemctl daemon-reload
	@sudo rm -rf $(INSTALL_DIR)
	$(call log_success,"系统已卸载")

# === 开发辅助 ===
dev-run:
	$(call log_info,"开发模式直接运行...")
	@if [ ! -f "$(BUILD_DIR)/bamboo_recognition" ]; then \
		$(call log_error,"可执行文件不存在，请先构建系统"); \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && sudo ./bamboo_recognition --verbose --config ../config/system_config.yaml

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