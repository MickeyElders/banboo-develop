# 智能切竹机控制系统 - 主构建文件
# 版本: 2.0.0
# 支持: C++后端(libmodbus+UNIX Socket) + LVGL前端 + systemd服务

.PHONY: all clean build-backend build-frontend install install-service uninstall start stop restart status logs help check-deps

# 项目配置
PROJECT_NAME = bamboo-cutting-system
BACKEND_DIR = cpp_backend
FRONTEND_DIR = lvgl_frontend
BUILD_TYPE ?= Release
INSTALL_PREFIX = /opt/bamboo
SYSTEMD_DIR = /etc/systemd/system
LOG_DIR = /var/log/bamboo
CONFIG_DIR = /etc/bamboo
JOBS ?= $(shell nproc)

# 服务配置
BACKEND_SERVICE = bamboo-backend.service
FRONTEND_SERVICE = bamboo-frontend.service

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
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

# 默认目标
all: build-backend build-frontend

help:
	@echo "智能切竹机控制系统 - 构建和部署系统"
	@echo ""
	@echo "构建目标:"
	@echo "  all              - 构建后端和前端"
	@echo "  build-backend    - 构建C++后端(libmodbus+UNIX Socket)"
	@echo "  build-frontend   - 构建LVGL前端"
	@echo "  clean            - 清理构建文件"
	@echo ""
	@echo "部署目标:"
	@echo "  install          - 安装到系统"
	@echo "  install-service  - 安装systemd服务"
	@echo "  uninstall        - 从系统卸载"
	@echo ""
	@echo "服务管理:"
	@echo "  start            - 启动服务"
	@echo "  stop             - 停止服务"
	@echo "  restart          - 重启服务"
	@echo "  status           - 查看服务状态"
	@echo "  logs             - 查看服务日志"
	@echo ""
	@echo "  help             - 显示此帮助信息"
	@echo ""
	@echo "变量:"
	@echo "  BUILD_TYPE=Debug|Release  构建模式 (默认: Release)"
	@echo "  JOBS=N                   并行任务数 (默认: CPU核心数)"

check-deps:
	$(call log_info,检查系统依赖...)
	@which cmake >/dev/null 2>&1 || ($(call log_error,cmake未安装) && exit 1)
	@which gcc >/dev/null 2>&1 || ($(call log_error,gcc未安装) && exit 1)
	@which g++ >/dev/null 2>&1 || ($(call log_error,g++未安装) && exit 1)
	@pkg-config --exists libmodbus || ($(call log_error,libmodbus未安装) && exit 1)
	$(call log_success,依赖检查通过)

build-backend: check-deps
	$(call log_info,构建C++后端(libmodbus+UNIX Socket)...)
	@mkdir -p $(BACKEND_DIR)/build
	@cd $(BACKEND_DIR)/build && \
		cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
		make -j$(JOBS)
	$(call log_success,后端构建完成)

build-frontend: check-deps
	$(call log_info,构建LVGL前端...)
	@mkdir -p $(FRONTEND_DIR)/build
	@cd $(FRONTEND_DIR) && \
		if [ ! -d "third_party/lvgl" ]; then \
			$(call log_info,下载LVGL库...); \
			mkdir -p third_party && \
			cd third_party && \
			git clone --depth 1 --branch release/v8.3 https://github.com/lvgl/lvgl.git && \
			cd ..; \
		fi && \
		if [ -f "lv_conf.h" ]; then cp lv_conf.h third_party/lvgl/; fi && \
		cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
			-DCMAKE_C_FLAGS="-O3 -ffast-math" \
			-DCMAKE_CXX_FLAGS="-O3 -ffast-math" && \
		make -j$(JOBS)
	$(call log_success,前端构建完成)

clean:
	$(call log_info,清理构建文件...)
	@rm -rf $(BACKEND_DIR)/build
	@rm -rf $(FRONTEND_DIR)/build
	@rm -f *.log *.pid
	$(call log_success,清理完成)

install: all
	$(call log_info,安装到系统...)
	@sudo mkdir -p $(INSTALL_PREFIX)/bin
	@sudo mkdir -p $(CONFIG_DIR)
	@sudo mkdir -p $(LOG_DIR)
	@sudo cp $(BACKEND_DIR)/build/bamboo_cut_backend $(INSTALL_PREFIX)/bin/
	@sudo cp $(FRONTEND_DIR)/build/bamboo_controller_lvgl $(INSTALL_PREFIX)/bin/
	@if [ -d "config" ]; then sudo cp -r config/* $(CONFIG_DIR)/ 2>/dev/null || true; fi
	@sudo chmod +x $(INSTALL_PREFIX)/bin/*
	@sudo chown -R root:root $(INSTALL_PREFIX)
	@sudo chmod 755 $(LOG_DIR)
	$(call log_success,二进制文件安装完成)

install-service: install
	$(call log_info,安装systemd服务...)
	@sudo mkdir -p deploy/systemd
	@sudo cp deploy/systemd/$(BACKEND_SERVICE) $(SYSTEMD_DIR)/ 2>/dev/null || \
		$(call log_warning,systemd服务文件不存在，将创建默认服务)
	@sudo cp deploy/systemd/$(FRONTEND_SERVICE) $(SYSTEMD_DIR)/ 2>/dev/null || true
	@sudo systemctl daemon-reload
	@sudo systemctl enable $(BACKEND_SERVICE)
	@sudo systemctl enable $(FRONTEND_SERVICE)
	$(call log_success,systemd服务安装完成)

uninstall:
	$(call log_info,停止并卸载服务...)
	@sudo systemctl stop $(BACKEND_SERVICE) $(FRONTEND_SERVICE) 2>/dev/null || true
	@sudo systemctl disable $(BACKEND_SERVICE) $(FRONTEND_SERVICE) 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/$(BACKEND_SERVICE)
	@sudo rm -f $(SYSTEMD_DIR)/$(FRONTEND_SERVICE)
	@sudo systemctl daemon-reload
	@sudo rm -rf $(INSTALL_PREFIX)
	@sudo rm -rf $(CONFIG_DIR)
	@sudo rm -rf $(LOG_DIR)
	$(call log_success,卸载完成)

start:
	$(call log_info,启动服务...)
	@sudo systemctl start $(BACKEND_SERVICE)
	@sleep 2
	@sudo systemctl start $(FRONTEND_SERVICE)
	$(call log_success,服务启动完成)

stop:
	$(call log_info,停止服务...)
	@sudo systemctl stop $(FRONTEND_SERVICE)
	@sudo systemctl stop $(BACKEND_SERVICE)
	$(call log_success,服务停止完成)

restart:
	$(call log_info,重启服务...)
	@sudo systemctl restart $(BACKEND_SERVICE)
	@sleep 2
	@sudo systemctl restart $(FRONTEND_SERVICE)
	$(call log_success,服务重启完成)

status:
	@echo "=== 后端服务状态 ==="
	@sudo systemctl status $(BACKEND_SERVICE) --no-pager || true
	@echo ""
	@echo "=== 前端服务状态 ==="
	@sudo systemctl status $(FRONTEND_SERVICE) --no-pager || true

logs:
	@echo "=== 后端服务日志 ==="
	@sudo journalctl -u $(BACKEND_SERVICE) --no-pager -n 20 || true
	@echo ""
	@echo "=== 前端服务日志 ==="
	@sudo journalctl -u $(FRONTEND_SERVICE) --no-pager -n 20 || true
	@echo ""
	@echo "=== 实时日志 (Ctrl+C退出) ==="
	@sudo journalctl -u $(BACKEND_SERVICE) -u $(FRONTEND_SERVICE) -f