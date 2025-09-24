# 竹子识别系统一体化构建文件
# 版本: 3.0.0 (Integrated Architecture)
# 将LVGL前端和C++后端整合为单一高性能进程

.PHONY: all clean build-integrated install install-service uninstall start stop restart status logs logs-follow help check-deps deploy redeploy backup

# 项目配置
PROJECT_NAME = bamboo-cutting-integrated
SOURCE_DIR = .
BUILD_DIR = build_integrated
BUILD_TYPE ?= Release
INSTALL_PREFIX = /opt/bamboo
SYSTEMD_DIR = /etc/systemd/system
LOG_DIR = /var/log/bamboo
CONFIG_DIR = /etc/bamboo
JOBS ?= $(shell nproc)

# 服务配置 - 单一服务替代原有的前后端分离服务
INTEGRATED_SERVICE = bamboo-integrated.service

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m

# 日志函数
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
	@echo "$(CYAN)[INTEGRATED]$(NC) $1"
endef

# 默认目标
all: build-integrated

help:
	@echo "$(CYAN)======================================$(NC)"
	@echo "$(CYAN)    竹子识别系统一体化构建系统$(NC)"
	@echo "$(CYAN)======================================$(NC)"
	@echo ""
	@echo "$(GREEN)快速部署命令:$(NC)"
	@echo "  deploy           - 首次完整部署(构建+安装+启动服务)"
	@echo "  redeploy         - 代码修改后快速重新部署"
	@echo "  backup           - 创建当前系统备份"
	@echo ""
	@echo "$(GREEN)构建目标:$(NC)"
	@echo "  all              - 构建一体化系统"
	@echo "  build-integrated - 构建一体化程序(LVGL+C++后端)"
	@echo "  clean            - 清理构建文件"
	@echo ""
	@echo "$(GREEN)部署目标:$(NC)"
	@echo "  install          - 安装到系统"
	@echo "  install-service  - 安装systemd服务"
	@echo "  uninstall        - 从系统卸载"
	@echo ""
	@echo "$(GREEN)服务管理:$(NC)"
	@echo "  start            - 启动一体化服务"
	@echo "  stop             - 停止一体化服务"
	@echo "  restart          - 重启一体化服务"
	@echo "  status           - 查看服务状态"
	@echo "  logs             - 查看服务日志"
	@echo "  logs-follow      - 实时查看服务日志"
	@echo ""
	@echo "  help             - 显示此帮助信息"
	@echo ""
	@echo "$(GREEN)变量:$(NC)"
	@echo "  BUILD_TYPE=Debug|Release  构建模式 (默认: Release)"
	@echo "  JOBS=N                   并行任务数 (默认: CPU核心数)"

check-deps:
	$(call log_info,检查一体化系统依赖...)
	@which cmake >/dev/null 2>&1 || ($(call log_error,cmake未安装) && exit 1)
	@which gcc >/dev/null 2>&1 || ($(call log_error,gcc未安装) && exit 1)
	@which g++ >/dev/null 2>&1 || ($(call log_error,g++未安装) && exit 1)
	@pkg-config --exists opencv4 || pkg-config --exists opencv || ($(call log_error,OpenCV未安装) && exit 1)
	@pkg-config --exists libmodbus || ($(call log_warning,libmodbus未安装，将禁用PLC通信功能))
	@pkg-config --exists gstreamer-1.0 || ($(call log_warning,GStreamer未安装，将禁用高级视频功能))
	$(call log_success,依赖检查完成)

# 准备LVGL依赖
prepare-lvgl:
	$(call log_info,准备LVGL库...)
	@mkdir -p $(BUILD_DIR)/third_party
	@if [ ! -d "$(BUILD_DIR)/third_party/lvgl" ]; then \
		$(call log_info,下载LVGL v8.3...); \
		cd $(BUILD_DIR)/third_party && \
		git clone --depth 1 --branch release/v8.3 https://github.com/lvgl/lvgl.git; \
	fi
	@if [ -f "lvgl_frontend/lv_conf.h" ]; then \
		cp lvgl_frontend/lv_conf.h $(BUILD_DIR)/third_party/lvgl/; \
	fi
	$(call log_success,LVGL准备完成)

build-integrated: check-deps check-cmake prepare-lvgl
	$(call log_highlight,构建一体化竹子识别系统...)
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && \
		cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
			-DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
			-DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
			-DENABLE_TENSORRT=ON \
			-DENABLE_CUDA=ON \
			-DENABLE_OPTIMIZATION=ON \
			-DINTEGRATED_BUILD=ON && \
		make -j$(JOBS)
	$(call log_success,一体化系统构建完成)
	$(call log_highlight,二进制文件: $(BUILD_DIR)/bamboo_integrated)

clean:
	$(call log_info,清理构建文件...)
	@rm -rf $(BUILD_DIR)
	@rm -f *.log *.pid
	@rm -f compile_commands.json
	$(call log_success,清理完成)

install: build-integrated
	$(call log_info,安装一体化系统...)
	@sudo mkdir -p $(INSTALL_PREFIX)/bin
	@sudo mkdir -p $(CONFIG_DIR)
	@sudo mkdir -p $(LOG_DIR)
	
	# 安装一体化主程序
	@sudo cp $(BUILD_DIR)/bamboo_integrated $(INSTALL_PREFIX)/bin/
	
	# 安装配置文件
	@if [ -d "config" ]; then sudo cp -r config/* $(CONFIG_DIR)/ 2>/dev/null || true; fi
	
	# 安装模型文件
	@if [ -d "models" ]; then \
		sudo mkdir -p $(INSTALL_PREFIX)/models && \
		sudo cp -r models/* $(INSTALL_PREFIX)/models/ 2>/dev/null || true; \
	fi
	
	# 设置权限
	@sudo chmod +x $(INSTALL_PREFIX)/bin/*
	@sudo chown -R root:root $(INSTALL_PREFIX)
	@sudo chmod 755 $(LOG_DIR)
	@sudo chmod 755 $(CONFIG_DIR)
	
	$(call log_success,一体化系统安装完成)

# 创建systemd服务文件
create-service-file:
	$(call log_info,创建systemd服务文件...)
	@sudo mkdir -p deploy/systemd
	@echo "[Unit]" | sudo tee deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Description=Bamboo Cutting Integrated System" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "After=network.target multi-user.target graphical-session.target" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Wants=network.target" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "[Service]" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Type=simple" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "User=root" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "ExecStart=$(INSTALL_PREFIX)/bin/bamboo_integrated" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "WorkingDirectory=$(INSTALL_PREFIX)" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Restart=always" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "RestartSec=5" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "StandardOutput=journal" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "StandardError=journal" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Environment=\"LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu\"" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Environment=\"DISPLAY=:0\"" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "Environment=\"XDG_RUNTIME_DIR=/run/user/0\"" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "[Install]" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	@echo "WantedBy=multi-user.target" | sudo tee -a deploy/systemd/$(INTEGRATED_SERVICE) > /dev/null
	$(call log_success,systemd服务文件创建完成)

install-service: install create-service-file
	$(call log_info,安装systemd一体化服务...)
	
	# 停止并删除旧的分离式服务
	@sudo systemctl stop bamboo-backend.service bamboo-frontend.service 2>/dev/null || true
	@sudo systemctl disable bamboo-backend.service bamboo-frontend.service 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-backend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-frontend.service
	
	# 安装新的一体化服务
	@sudo cp deploy/systemd/$(INTEGRATED_SERVICE) $(SYSTEMD_DIR)/
	@sudo systemctl daemon-reload
	@sudo systemctl enable $(INTEGRATED_SERVICE)
	
	$(call log_success,一体化systemd服务安装完成)
	$(call log_highlight,旧的分离式服务已清理)

backup:
	$(call log_info,创建系统备份...)
	@BACKUP_DIR="backup_$(shell date +%Y%m%d_%H%M%S)" && \
	mkdir -p $$BACKUP_DIR && \
	if [ -d "$(INSTALL_PREFIX)" ]; then sudo cp -r $(INSTALL_PREFIX) $$BACKUP_DIR/; fi && \
	if [ -d "$(CONFIG_DIR)" ]; then sudo cp -r $(CONFIG_DIR) $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/bamboo-backend.service" ]; then sudo cp $(SYSTEMD_DIR)/bamboo-backend.service $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/bamboo-frontend.service" ]; then sudo cp $(SYSTEMD_DIR)/bamboo-frontend.service $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/$(INTEGRATED_SERVICE)" ]; then sudo cp $(SYSTEMD_DIR)/$(INTEGRATED_SERVICE) $$BACKUP_DIR/; fi && \
	sudo chown -R $(shell whoami):$(shell whoami) $$BACKUP_DIR && \
	echo "$$BACKUP_DIR" > .last_backup
	$(call log_success,备份完成: $(shell cat .last_backup 2>/dev/null || echo "unknown"))

uninstall:
	$(call log_info,停止并卸载一体化服务...)
	
	# 停止所有相关服务
	@sudo systemctl stop $(INTEGRATED_SERVICE) 2>/dev/null || true
	@sudo systemctl stop bamboo-backend.service bamboo-frontend.service 2>/dev/null || true
	
	# 禁用并删除服务文件
	@sudo systemctl disable $(INTEGRATED_SERVICE) 2>/dev/null || true
	@sudo systemctl disable bamboo-backend.service bamboo-frontend.service 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/$(INTEGRATED_SERVICE)
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-backend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-frontend.service
	@sudo systemctl daemon-reload
	
	# 删除安装文件
	@sudo rm -rf $(INSTALL_PREFIX)
	@sudo rm -rf $(CONFIG_DIR)
	@sudo rm -rf $(LOG_DIR)
	
	$(call log_success,一体化系统卸载完成)

start:
	$(call log_info,启动一体化服务...)
	@sudo systemctl start $(INTEGRATED_SERVICE)
	@sleep 2
	$(call log_success,一体化服务启动完成)

stop:
	$(call log_info,停止一体化服务...)
	@sudo systemctl stop $(INTEGRATED_SERVICE) 2>/dev/null || true
	$(call log_success,一体化服务停止完成)

restart:
	$(call log_info,重启一体化服务...)
	@sudo systemctl restart $(INTEGRATED_SERVICE)
	@sleep 2
	$(call log_success,一体化服务重启完成)

status:
	@echo "$(CYAN)=== 一体化服务状态 ===$(NC)"
	@sudo systemctl status $(INTEGRATED_SERVICE) --no-pager || true
	@echo ""
	@echo "$(CYAN)=== 系统资源使用 ===$(NC)"
	@ps aux | grep bamboo_integrated | head -5 || true

logs:
	@echo "$(CYAN)=== 一体化服务日志 (最近50行) ===$(NC)"
	@sudo journalctl -u $(INTEGRATED_SERVICE) --no-pager -n 50 || true

logs-follow:
	$(call log_info,实时查看一体化服务日志 (Ctrl+C退出)...)
	@sudo journalctl -u $(INTEGRATED_SERVICE) -f

# 性能测试
performance-test:
	$(call log_info,启动性能测试...)
	@if sudo systemctl is-active --quiet $(INTEGRATED_SERVICE); then \
		echo "$(GREEN)服务运行中$(NC) - 监控性能指标"; \
		sudo journalctl -u $(INTEGRATED_SERVICE) -f --since "1 minute ago" | grep -E "(fps|延迟|CPU|内存)" || true; \
	else \
		$(call log_error,服务未运行，请先执行 make start); \
	fi

# 首次完整部署
deploy: backup build-integrated install-service start
	$(call log_success,$(CYAN)一体化系统首次部署完成！$(NC))
	@echo ""
	$(call log_highlight,架构信息:)
	@echo "  ✓ 单进程一体化架构"
	@echo "  ✓ LVGL前端 + C++后端整合"
	@echo "  ✓ 线程安全数据桥接"
	@echo "  ✓ 统一配置和日志管理"
	@echo ""
	$(call log_info,服务管理命令:)
	@echo "  make status          - 查看服务状态"
	@echo "  make logs            - 查看服务日志"
	@echo "  make logs-follow     - 实时查看日志"
	@echo "  make restart         - 重启服务"
	@echo "  make stop            - 停止服务"
	@echo "  make performance-test - 性能监控"

# 代码修改后快速重新部署
redeploy: stop build-integrated install start
	$(call log_success,$(CYAN)一体化系统重新部署完成！$(NC))
	@echo ""
	$(call log_info,检查服务状态:)
	@make status

# CMakeLists.txt 检查和生成
check-cmake:
	@if [ ! -f "CMakeLists.txt" ]; then \
		$(call log_warning,CMakeLists.txt不存在，将创建基础版本); \
		$(MAKE) generate-cmake; \
	fi

generate-cmake:
	$(call log_info,生成一体化CMakeLists.txt...)
	@echo "# 一体化竹子识别系统 CMake配置" > CMakeLists.txt
	@echo "cmake_minimum_required(VERSION 3.16)" >> CMakeLists.txt
	@echo "project(BambooIntegratedSystem VERSION 3.0.0 LANGUAGES C CXX)" >> CMakeLists.txt
	@echo "" >> CMakeLists.txt
	@echo "set(CMAKE_CXX_STANDARD 17)" >> CMakeLists.txt
	@echo "set(CMAKE_CXX_STANDARD_REQUIRED ON)" >> CMakeLists.txt
	@echo "" >> CMakeLists.txt
	@echo "# 源文件收集" >> CMakeLists.txt
	@echo "file(GLOB_RECURSE CPP_BACKEND_SOURCES \"cpp_backend/src/*.cpp\")" >> CMakeLists.txt
	@echo "file(GLOB_RECURSE LVGL_FRONTEND_SOURCES \"lvgl_frontend/src/*.cpp\" \"lvgl_frontend/src/*.c\")" >> CMakeLists.txt
	@echo "" >> CMakeLists.txt
	@echo "# 主程序" >> CMakeLists.txt
	@echo "add_executable(bamboo_integrated integrated_main.cpp \${CPP_BACKEND_SOURCES} \${LVGL_FRONTEND_SOURCES})" >> CMakeLists.txt
	$(call log_success,基础CMakeLists.txt生成完成)