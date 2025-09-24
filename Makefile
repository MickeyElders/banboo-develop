# 竹子识别系统 Python GTK3 架构构建文件
# 版本: 5.0.0 (Python GTK3 Architecture)
# Python GTK3前端 + C++推理后端的混合架构

.PHONY: all clean build-cpp build-python install-python-deps install install-service uninstall start stop restart status logs logs-follow help check-deps deploy redeploy backup dev-run test-cpp

# 项目配置
PROJECT_NAME = bamboo-cutting-python-gtk3
SOURCE_DIR = .
BUILD_DIR = build_python_gtk3
CPP_BUILD_DIR = cpp_inference
BUILD_TYPE ?= Release
INSTALL_PREFIX = /opt/bamboo-cut
SYSTEMD_DIR = /etc/systemd/system
LOG_DIR = /var/log/bamboo
CONFIG_DIR = /etc/bamboo
JOBS ?= $(shell nproc)

# Python环境配置
PYTHON_VENV = $(INSTALL_PREFIX)/venv
PYTHON_BIN = $(PYTHON_VENV)/bin/python
PIP_BIN = $(PYTHON_VENV)/bin/pip

# 服务配置 - Python GTK4服务
PYTHON_GTK4_SERVICE = bamboo-python-gtk4.service

# AI优化配置文件
AI_CONFIG = config/ai_optimization.yaml

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
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
	@echo "$(CYAN)[PYTHON-GTK4]$(NC) $1"
endef

define log_ai
	@echo "$(MAGENTA)[AI-OPT]$(NC) $1"
endef

# 默认目标
all: build-cpp build-python

help:
	@echo "$(CYAN)===============================================$(NC)"
	@echo "$(CYAN)    竹子识别系统 Python GTK4 构建系统$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)快速部署命令:$(NC)"
	@echo "  deploy           - 首次完整部署(构建+安装+启动服务)"
	@echo "  redeploy         - 代码修改后快速重新部署"
	@echo "  backup           - 创建当前系统备份"
	@echo "  dev-run          - 开发模式直接运行"
	@echo ""
	@echo "$(GREEN)构建目标:$(NC)"
	@echo "  all              - 构建完整系统(C++推理+Python环境)"
	@echo "  build-cpp        - 构建C++推理核心"
	@echo "  build-python     - 准备Python GTK4环境"
	@echo "  install-python-deps - 安装Python依赖"
	@echo "  clean            - 清理构建文件"
	@echo ""
	@echo "$(GREEN)部署目标:$(NC)"
	@echo "  install          - 安装到系统"
	@echo "  install-service  - 安装systemd服务"
	@echo "  uninstall        - 从系统卸载"
	@echo ""
	@echo "$(GREEN)服务管理:$(NC)"
	@echo "  start            - 启动Python GTK4服务"
	@echo "  stop             - 停止Python GTK4服务"
	@echo "  restart          - 重启Python GTK4服务"
	@echo "  status           - 查看服务状态"
	@echo "  logs             - 查看服务日志"
	@echo "  logs-follow      - 实时查看服务日志"
	@echo ""
	@echo "$(GREEN)开发和测试:$(NC)"
	@echo "  test-cpp         - 测试C++推理模块"
	@echo "  dev-run          - 开发模式运行"
	@echo "  help             - 显示此帮助信息"
	@echo ""
	@echo "$(GREEN)变量:$(NC)"
	@echo "  BUILD_TYPE=Debug|Release  构建模式 (默认: Release)"
	@echo "  JOBS=N                   并行任务数 (默认: CPU核心数)"

check-deps:
	$(call log_info,检查Python GTK4系统依赖...)
	@which cmake >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(NC) cmake未安装" && exit 1)
	@which gcc >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(NC) gcc未安装" && exit 1)
	@which g++ >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(NC) g++未安装" && exit 1)
	@which python3 >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(NC) Python3未安装" && exit 1)
	@which pip3 >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(NC) pip3未安装" && exit 1)
	@pkg-config --exists opencv4 || pkg-config --exists opencv || (echo "$(RED)[ERROR]$(NC) OpenCV未安装" && exit 1)
	@pkg-config --exists libmodbus || echo "$(YELLOW)[WARNING]$(NC) libmodbus未安装，将禁用PLC通信功能"
	@pkg-config --exists gstreamer-1.0 || echo "$(YELLOW)[WARNING]$(NC) GStreamer未安装，将禁用高级视频功能"
	@pkg-config --exists gtk4 || echo "$(YELLOW)[WARNING]$(NC) GTK4开发包未安装，请运行: sudo apt install libgtk-4-dev"
	@python3 -c "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1')" 2>/dev/null || echo "$(YELLOW)[WARNING]$(NC) GTK4 Python绑定未安装"
	@python3 -c "import pybind11" 2>/dev/null || echo "$(YELLOW)[WARNING]$(NC) pybind11未安装，将从requirements.txt安装"
	$(call log_success,GTK4依赖检查完成)

# 检查AI优化配置
check-ai-config:
	$(call log_ai,检查AI优化配置...)
	@if [ ! -f "$(AI_CONFIG)" ]; then \
		echo "$(YELLOW)[WARNING]$(NC) AI优化配置文件不存在: $(AI_CONFIG)"; \
	else \
		echo "$(GREEN)[SUCCESS]$(NC) AI优化配置文件存在: $(AI_CONFIG)"; \
		echo "$(MAGENTA)启用的AI优化技术:$(NC)"; \
		grep -E "enable: true" $(AI_CONFIG) | sed 's/^/  /' || true; \
	fi

# 构建C++推理核心(包含AI优化)
build-cpp: check-deps check-ai-config check-cmake
	$(call log_highlight,构建C++推理核心...)
	$(call log_ai,集成AI优化: NAM注意力、GhostConv、VoV-GSCSP、Wise-IoU、SAHI切片推理)
	@rm -rf $(CPP_BUILD_DIR)
	@mkdir -p $(CPP_BUILD_DIR)
	@cd $(CPP_BUILD_DIR) && \
		cmake ../cpp_backend -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
			-DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
			-DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
			-DENABLE_TENSORRT=ON \
			-DENABLE_CUDA=ON \
			-DENABLE_AI_OPTIMIZATION=ON \
			-DENABLE_NAM_ATTENTION=ON \
			-DENABLE_GHOST_CONV=ON \
			-DENABLE_VOV_GSCSP=ON \
			-DENABLE_WISE_IOU=ON \
			-DENABLE_SAHI_SLICING=ON \
			-DENABLE_HARDWARE_ACCELERATION=ON \
			-DAI_CONFIG_PATH="../$(AI_CONFIG)" && \
		make -j$(JOBS)
	$(call log_success,C++推理核心构建完成)
	$(call log_ai,AI优化推理库: $(CPP_BUILD_DIR)/libbamboo_inference.so)

# 准备Python环境
build-python: install-python-deps
	$(call log_highlight,准备Python GTK4环境...)
	@mkdir -p $(BUILD_DIR)
	
	# 检查Python GTK4文件
	@if [ ! -f "python_core/ai_bamboo_system.py" ]; then \
		$(call log_error,Python GTK4主文件不存在: python_core/ai_bamboo_system.py); \
		exit 1; \
	fi
	@if [ ! -f "python_core/display_driver.py" ]; then \
		$(call log_error,显示驱动文件不存在: python_core/display_driver.py); \
		exit 1; \
	fi
	
	$(call log_success,Python GTK4环境准备完成)

# 安装Python依赖
install-python-deps: install-gtk4-python
	$(call log_info,安装Python依赖...)
	@if [ ! -f "requirements.txt" ]; then \
		echo "$(RED)[ERROR]$(NC) requirements.txt文件不存在"; \
		exit 1; \
	fi
	
	# 检查并安装python3-venv
	@if ! dpkg -l | grep -q python3-venv; then \
		echo "$(YELLOW)[WARNING]$(NC) python3-venv未安装，正在安装..."; \
		sudo apt update && sudo apt install -y python3-venv; \
	fi
	
	# 创建虚拟环境(使用系统包)
	@if [ ! -d "venv" ]; then \
		echo "$(BLUE)[INFO]$(NC) 创建Python虚拟环境(访问系统GTK4包)..."; \
		python3 -m venv venv --system-site-packages; \
	fi
	
	# 验证虚拟环境创建成功
	@if [ ! -f "venv/bin/pip" ]; then \
		echo "$(RED)[ERROR]$(NC) 虚拟环境创建失败，正在重新创建..."; \
		rm -rf venv; \
		python3 -m venv venv --without-pip; \
		wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py; \
		./venv/bin/python get-pip.py; \
		rm get-pip.py; \
	fi
	
	# 激活虚拟环境并安装依赖
	@echo "$(BLUE)[INFO]$(NC) 安装Python包..."
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install -r requirements.txt
	
	# 验证关键包安装
	@echo "$(BLUE)[INFO]$(NC) 验证Python包安装..."
	@./venv/bin/python -c "import cv2; print('OpenCV version:', cv2.__version__)" || \
		echo "$(YELLOW)[WARNING]$(NC) OpenCV包验证失败"
	@./venv/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)" || \
		echo "$(YELLOW)[WARNING]$(NC) NumPy包验证失败"
	@./venv/bin/python -c "import pybind11; print('pybind11 version:', pybind11.__version__)" || \
		echo "$(YELLOW)[WARNING]$(NC) pybind11包验证失败"
	
	# 验证GTK4图形库安装
	@echo "$(BLUE)[INFO]$(NC) 验证GTK4图形库安装..."
	@GTK4_OK=false; \
	\
	if ./venv/bin/python -c "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Gtk, Adw; print('GTK4 + Adwaita验证成功')" 2>/dev/null; then \
		$(call log_success,GTK4 Python绑定验证成功); \
		GTK4_OK=true; \
	elif ./venv/bin/python -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; print('GTK4基础库验证成功')" 2>/dev/null; then \
		$(call log_success,GTK4基础库验证成功，但缺少Adwaita); \
		GTK4_OK=true; \
	elif ./venv/bin/python -c "import pygame; print('Pygame version:', pygame.version.ver)" 2>/dev/null; then \
		$(call log_warning,GTK4不可用，使用Pygame作为后备图形库); \
		GTK4_OK=true; \
	fi; \
	\
	if [ "$$GTK4_OK" = "false" ]; then \
		$(call log_warning,无可用图形库，请安装GTK4系统依赖); \
		echo "$(YELLOW)[提示]$(NC) 运行: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"; \
	fi

	$(call log_success,Python依赖安装完成)

# 安装GTK4 Python绑定
install-gtk4-python:
	$(call log_highlight,安装GTK4 Python绑定...)
	
	# 检查系统依赖
	@echo "$(BLUE)[INFO]$(NC) 安装GTK4完整系统依赖..."
	@sudo apt update || true
	
	# 安装GTK4核心和开发依赖
	@sudo apt install -y \
		python3-gi python3-gi-dev python3-gi-cairo \
		gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-glib-2.0 \
		libgtk-4-dev libadwaita-1-dev libgirepository1.0-dev \
		pkg-config build-essential meson ninja-build \
		libglib2.0-dev libgobject-introspection-1.0-dev || true
		
	# 确保gi-docgen可用(Adwaita文档生成)
	@sudo apt install -y gi-docgen || true
	
	# 确保虚拟环境使用系统包
	@if [ ! -d "venv" ]; then \
		echo "$(BLUE)[INFO]$(NC) 创建Python虚拟环境(使用系统包)..."; \
		python3 -m venv venv --system-site-packages; \
	fi
	
	# 验证GTK4安装
	@echo "$(BLUE)[INFO]$(NC) 验证GTK4 Python绑定..."
	@if ./venv/bin/python -c "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Gtk, Adw; print('GTK4 + Adwaita验证成功: Gtk', Gtk.get_major_version(), '.', Gtk.get_minor_version())" 2>/dev/null; then \
		echo "$(GREEN)[SUCCESS]$(NC) GTK4 + Adwaita安装成功"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) GTK4绑定验证失败，尝试安装额外依赖..."; \
		sudo apt install -y python3-dev libffi-dev libcairo2-dev || true; \
		if ./venv/bin/python -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; print('GTK4基础库可用')" 2>/dev/null; then \
			echo "$(GREEN)[SUCCESS]$(NC) GTK4基础库安装成功"; \
		else \
			echo "$(RED)[ERROR]$(NC) GTK4安装失败，请检查系统依赖"; \
		fi; \
	fi
	

# 安装Cage compositor
install-cage:
	$(call log_info,安装Cage Wayland compositor...)
	@echo "$(BLUE)[INFO]$(NC) 安装Cage和相关依赖..."
	@sudo apt update || true
	@sudo apt install -y cage wlroots-dev libwayland-dev || true
	
	# 验证Cage安装
	@if which cage >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) Cage compositor安装成功"; \
		cage --version 2>/dev/null || echo "Cage可用"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) 从包管理器安装失败，尝试从源码编译..."; \
		$(MAKE) build-cage-from-source; \
	fi

# 从源码编译Cage (备选方案)
build-cage-from-source:
	$(call log_warning,从源码编译Cage compositor...)
	@sudo apt install -y git build-essential meson ninja-build \
		libwayland-dev libwlroots-dev libxkbcommon-dev \
		pkg-config || true
	@cd /tmp && \
		rm -rf cage && \
		git clone https://github.com/Hjdskes/cage.git && \
		cd cage && \
		meson build --prefix=/usr/local && \
		ninja -C build && \
		sudo ninja -C build install && \
		echo "$(GREEN)[SUCCESS]$(NC) Cage源码编译安装完成" || \
		echo "$(RED)[ERROR]$(NC) Cage源码编译失败"
	@sudo ldconfig || true

	$(call log_success,GTK4 Python绑定安装完成)

clean:
	$(call log_info,清理构建文件...)
	@rm -rf $(BUILD_DIR)
	@rm -rf $(CPP_BUILD_DIR)
	@rm -rf venv
	@rm -f *.log *.pid
	@rm -f compile_commands.json
	$(call log_success,清理完成)

install: build-cpp build-python
	$(call log_info,安装Python GTK4系统...)
	@sudo mkdir -p $(INSTALL_PREFIX)/bin
	@sudo mkdir -p $(INSTALL_PREFIX)/lib
	@sudo mkdir -p $(INSTALL_PREFIX)/python_core
	@sudo mkdir -p $(CONFIG_DIR)
	@sudo mkdir -p $(LOG_DIR)
	
	# 安装C++推理库
	@if [ -f "$(CPP_BUILD_DIR)/libbamboo_inference.so" ]; then \
		sudo cp $(CPP_BUILD_DIR)/libbamboo_inference.so $(INSTALL_PREFIX)/lib/; \
	fi
	@if [ -f "$(CPP_BUILD_DIR)/bamboo_cut" ]; then \
		sudo cp $(CPP_BUILD_DIR)/bamboo_cut $(INSTALL_PREFIX)/bin/; \
	fi
	@if [ -f "$(CPP_BUILD_DIR)/bamboo_inference_test" ]; then \
		sudo cp $(CPP_BUILD_DIR)/bamboo_inference_test $(INSTALL_PREFIX)/bin/; \
	fi
	
	# 安装Python虚拟环境
	@if [ -d "venv" ]; then \
		sudo cp -r venv $(INSTALL_PREFIX)/; \
	fi
	
	# 安装Python核心文件
	@sudo cp -r python_core/* $(INSTALL_PREFIX)/python_core/
	
	# 安装配置文件
	@if [ -d "config" ]; then sudo cp -r config/* $(CONFIG_DIR)/ 2>/dev/null || true; fi
	
	# 安装模型文件
	@if [ -d "models" ]; then \
		sudo mkdir -p $(INSTALL_PREFIX)/models && \
		sudo cp -r models/* $(INSTALL_PREFIX)/models/ 2>/dev/null || true; \
	fi
	
	# 创建执行脚本
	@echo '#!/bin/bash' | sudo tee $(INSTALL_PREFIX)/bin/bamboo-python-gtk4 > /dev/null
	@echo 'cd $(INSTALL_PREFIX)' | sudo tee -a $(INSTALL_PREFIX)/bin/bamboo-python-gtk4 > /dev/null
	@echo 'export PYTHONPATH=$(INSTALL_PREFIX):$(INSTALL_PREFIX)/python_core' | sudo tee -a $(INSTALL_PREFIX)/bin/bamboo-python-gtk4 > /dev/null
	@echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(INSTALL_PREFIX)/lib' | sudo tee -a $(INSTALL_PREFIX)/bin/bamboo-python-gtk4 > /dev/null
	@echo 'exec $(INSTALL_PREFIX)/venv/bin/python python_core/ai_bamboo_system.py "$$@"' | sudo tee -a $(INSTALL_PREFIX)/bin/bamboo-python-gtk4 > /dev/null
	
	# 设置权限
	@sudo chmod +x $(INSTALL_PREFIX)/bin/*
	@sudo chown -R $(shell whoami):$(shell whoami) $(INSTALL_PREFIX) || true
	@sudo chmod 755 $(LOG_DIR)
	@sudo chmod 755 $(CONFIG_DIR)
	
	$(call log_success,Python GTK4系统安装完成)

install-service: install
	$(call log_info,安装systemd Python GTK4服务...)
	
	# 停止并删除旧服务(包括LVGL遗留服务)
	@sudo systemctl stop bamboo-backend.service bamboo-frontend.service bamboo-integrated.service bamboo-python-lvgl.service 2>/dev/null || true
	@sudo systemctl disable bamboo-backend.service bamboo-frontend.service bamboo-integrated.service bamboo-python-lvgl.service 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-backend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-frontend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-integrated.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-python-lvgl.service  # 清理旧的LVGL服务
	
	# 获取当前用户和UID
	@CURRENT_USER=$$(whoami); \
	CURRENT_UID=$$(id -u); \
	echo "$(BLUE)[INFO]$(NC) 配置服务用户: $$CURRENT_USER (UID: $$CURRENT_UID)"; \
	\
	sed "s/USER_PLACEHOLDER/$$CURRENT_USER/g; s/UID_PLACEHOLDER/$$CURRENT_UID/g" $(PYTHON_GTK4_SERVICE) > /tmp/$(PYTHON_GTK4_SERVICE); \
	\
	sudo cp /tmp/$(PYTHON_GTK4_SERVICE) $(SYSTEMD_DIR)/; \
	rm -f /tmp/$(PYTHON_GTK4_SERVICE); \
	sudo systemctl daemon-reload; \
	sudo systemctl enable $(PYTHON_GTK4_SERVICE)
	
	$(call log_success,Python GTK4 systemd服务安装完成)
	$(call log_highlight,旧服务已清理，新的Python GTK4服务已安装)

backup:
	$(call log_info,创建系统备份...)
	@BACKUP_DIR="backup_$(shell date +%Y%m%d_%H%M%S)" && \
	mkdir -p $$BACKUP_DIR && \
	if [ -d "$(INSTALL_PREFIX)" ]; then sudo cp -r $(INSTALL_PREFIX) $$BACKUP_DIR/; fi && \
	if [ -d "$(CONFIG_DIR)" ]; then sudo cp -r $(CONFIG_DIR) $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/bamboo-backend.service" ]; then sudo cp $(SYSTEMD_DIR)/bamboo-backend.service $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/bamboo-frontend.service" ]; then sudo cp $(SYSTEMD_DIR)/bamboo-frontend.service $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/bamboo-integrated.service" ]; then sudo cp $(SYSTEMD_DIR)/bamboo-integrated.service $$BACKUP_DIR/; fi && \
	if [ -f "$(SYSTEMD_DIR)/$(PYTHON_GTK4_SERVICE)" ]; then sudo cp $(SYSTEMD_DIR)/$(PYTHON_GTK4_SERVICE) $$BACKUP_DIR/; fi && \
	sudo chown -R $(shell whoami):$(shell whoami) $$BACKUP_DIR && \
	echo "$$BACKUP_DIR" > .last_backup
	$(call log_success,备份完成: $(shell cat .last_backup 2>/dev/null || echo "unknown"))

uninstall:
	$(call log_info,停止并卸载Python GTK4服务...)

	# 停止所有相关服务
	@sudo systemctl stop $(PYTHON_GTK4_SERVICE) 2>/dev/null || true
	@sudo systemctl stop bamboo-backend.service bamboo-frontend.service bamboo-integrated.service 2>/dev/null || true

	# 禁用并删除服务文件
	@sudo systemctl disable $(PYTHON_GTK4_SERVICE) 2>/dev/null || true
	@sudo systemctl disable bamboo-backend.service bamboo-frontend.service bamboo-integrated.service 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/$(PYTHON_GTK4_SERVICE)
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-backend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-frontend.service
	@sudo rm -f $(SYSTEMD_DIR)/bamboo-integrated.service
	@sudo systemctl daemon-reload
	
	# 删除安装文件
	@sudo rm -rf $(INSTALL_PREFIX)
	@sudo rm -rf $(CONFIG_DIR)
	@sudo rm -rf $(LOG_DIR)
	
	$(call log_success,Python GTK4系统卸载完成)

start:
	$(call log_info,启动Python GTK4服务...)
	@if ! sudo systemctl list-unit-files | grep -q "$(PYTHON_GTK4_SERVICE)"; then \
		$(call log_warning,Python GTK4服务未安装，正在自动安装...); \
		$(MAKE) install-service; \
	fi
	@sudo systemctl start $(PYTHON_GTK4_SERVICE)
	@sleep 3
	$(call log_success,Python GTK4服务启动完成)

stop:
	$(call log_info,停止Python GTK4服务...)
	@sudo systemctl stop $(PYTHON_GTK4_SERVICE) 2>/dev/null || true
	$(call log_success,Python GTK4服务停止完成)

restart:
	$(call log_info,重启Python GTK4服务...)
	@sudo systemctl restart $(PYTHON_GTK4_SERVICE)
	@sleep 3
	$(call log_success,Python GTK4服务重启完成)

status:
	@echo "$(CYAN)=== Python GTK4服务状态 ===$(NC)"
	@sudo systemctl status $(PYTHON_GTK4_SERVICE) --no-pager || true
	@echo ""
	@echo "$(CYAN)=== 系统资源使用 ===$(NC)"
	@ps aux | grep ai_bamboo_system | head -5 || true

logs:
	@echo "$(CYAN)=== Python GTK4服务日志 (最近50行) ===$(NC)"
	@sudo journalctl -u $(PYTHON_GTK4_SERVICE) --no-pager -n 50 || true

logs-follow:
	$(call log_info,实时查看Python GTK4服务日志 (Ctrl+C退出)...)
	@sudo journalctl -u $(PYTHON_GTK4_SERVICE) -f

# 开发模式运行
dev-run: build-cpp install-python-deps
	$(call log_highlight,开发模式运行Python GTK4系统...)
	@export PYTHONPATH=$(shell pwd):$(shell pwd)/python_core && \
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(shell pwd)/$(CPP_BUILD_DIR) && \
	./venv/bin/python python_core/ai_bamboo_system.py

# 测试C++推理模块
test-cpp: build-cpp
	$(call log_info,测试C++推理模块...)
	@if [ -f "$(CPP_BUILD_DIR)/bamboo_inference_test" ]; then \
		cd $(CPP_BUILD_DIR) && ./bamboo_inference_test; \
	else \
		$(call log_warning,测试程序不存在，跳过测试); \
	fi

# 性能测试
performance-test:
	$(call log_info,启动性能测试...)
	@if sudo systemctl is-active --quiet $(PYTHON_GTK4_SERVICE); then \
		echo "$(GREEN)Python GTK4服务运行中$(NC) - 监控性能指标"; \
		sudo journalctl -u $(PYTHON_GTK4_SERVICE) -f --since "1 minute ago" | grep -E "(fps|延迟|CPU|内存|AI|推理)" || true; \
	else \
		$(call log_error,服务未运行，请先执行 make start); \
	fi

# 首次完整部署
deploy: backup build-cpp build-python install-cage install-service start
	$(call log_success,$(CYAN)Python GTK4系统首次部署完成！$(NC))
	@echo ""
	$(call log_highlight,架构信息:)
	@echo "  ✓ Python GTK4/无头模式自适应界面"
	@echo "  ✓ C++高性能推理后端"
	@echo "  ✓ AI优化技术集成"
	@echo "  ✓ 混合架构设计"
	@echo "  ✓ systemd服务管理"
	@echo ""
	$(call log_ai,AI优化技术:)
	@echo "  ✓ NAM注意力机制"
	@echo "  ✓ GhostConv卷积优化"
	@echo "  ✓ VoV-GSCSP颈部压缩"
	@echo "  ✓ Wise-IoU损失函数"
	@echo "  ✓ SAHI切片推理"
	@echo "  ✓ 硬件加速配置"
	@echo ""
	$(call log_info,服务管理命令:)
	@echo "  make status          - 查看服务状态"
	@echo "  make logs            - 查看服务日志"
	@echo "  make logs-follow     - 实时查看日志"
	@echo "  make restart         - 重启服务"
	@echo "  make dev-run         - 开发模式运行"
	@echo "  make performance-test - 性能监控"

# 代码修改后快速重新部署
redeploy: stop build-cpp build-python install-service start
	$(call log_success,$(CYAN)Python GTK4系统重新部署完成！$(NC))
	@echo ""
	$(call log_info,检查服务状态:)
	@make status

# CMakeLists.txt 检查和生成
check-cmake:
	@if [ ! -f "CMakeLists.txt" ]; then \
		$(call log_warning,根目录CMakeLists.txt不存在，将使用现有版本); \
	fi

generate-cpp-cmake:
	$(call log_info,生成C++推理核心CMakeLists.txt...)
	@mkdir -p cpp_backend
	@echo "# C++推理核心 CMake配置" > cpp_backend/CMakeLists.txt
	@echo "cmake_minimum_required(VERSION 3.16)" >> cpp_backend/CMakeLists.txt
	@echo "project(BambooInferenceCore VERSION 4.0.0 LANGUAGES C CXX)" >> cpp_backend/CMakeLists.txt
	@echo "" >> cpp_backend/CMakeLists.txt
	@echo "set(CMAKE_CXX_STANDARD 17)" >> cpp_backend/CMakeLists.txt
	@echo "set(CMAKE_CXX_STANDARD_REQUIRED ON)" >> cpp_backend/CMakeLists.txt
	@echo "" >> cpp_backend/CMakeLists.txt
	@echo "# AI优化特性选项" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_AI_OPTIMIZATION \"Enable AI optimizations\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_NAM_ATTENTION \"Enable NAM Attention\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_GHOST_CONV \"Enable GhostConv\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_VOV_GSCSP \"Enable VoV-GSCSP\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_WISE_IOU \"Enable Wise-IoU\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "option(ENABLE_SAHI_SLICING \"Enable SAHI Slicing\" ON)" >> cpp_backend/CMakeLists.txt
	@echo "" >> cpp_backend/CMakeLists.txt
	@echo "# 源文件收集" >> cpp_backend/CMakeLists.txt
	@echo "file(GLOB_RECURSE SOURCES \"src/*.cpp\")" >> cpp_backend/CMakeLists.txt
	@echo "" >> cpp_backend/CMakeLists.txt
	@echo "# 推理库" >> cpp_backend/CMakeLists.txt
	@echo "add_library(bamboo_inference SHARED \$${SOURCES})" >> cpp_backend/CMakeLists.txt
	@echo "add_executable(bamboo_inference_test src/main.cpp)" >> cpp_backend/CMakeLists.txt
	@echo "target_link_libraries(bamboo_inference_test bamboo_inference)" >> cpp_backend/CMakeLists.txt
	$(call log_success,C++推理核心CMakeLists.txt生成完成)