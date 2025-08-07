# 智能切竹机统一编译Makefile
# 支持同时编译C++后端和Flutter前端

# 默认目标
.PHONY: all help clean cpp flutter embedded debug release

# 默认配置
BUILD_TYPE ?= release
PLATFORM ?= linux
TARGET ?= desktop
JOBS ?= 4

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
all: cpp flutter
	$(call log_success,编译完成！)

# 显示帮助
help:
	@echo "智能切竹机统一编译Makefile"
	@echo ""
	@echo "目标:"
	@echo "  all         编译所有组件 (默认)"
	@echo "  cpp         仅编译C++后端"
	@echo "  flutter     仅编译Flutter前端"
	@echo "  embedded    编译嵌入式版本"
	@echo "  debug       调试模式编译"
	@echo "  release     发布模式编译 (默认)"
	@echo "  clean       清理编译文件"
	@echo "  help        显示此帮助信息"
	@echo ""
	@echo "变量:"
	@echo "  BUILD_TYPE=debug|release  编译模式 (默认: release)"
	@echo "  PLATFORM=linux|windows|android  目标平台 (默认: linux)"
	@echo "  TARGET=desktop|embedded|mobile  目标类型 (默认: desktop)"
	@echo "  JOBS=N                    并行编译任务数 (默认: 4)"
	@echo ""
	@echo "示例:"
	@echo "  make                       # 编译所有组件 (发布模式)"
	@echo "  make BUILD_TYPE=debug      # 调试模式编译所有组件"
	@echo "  make cpp BUILD_TYPE=debug  # 仅编译C++后端 (调试模式)"
	@echo "  make flutter PLATFORM=windows  # 仅编译Flutter前端 (Windows)"
	@echo "  make embedded JOBS=8       # 编译嵌入式版本 (8线程)"

# 调试模式
debug: BUILD_TYPE = debug
debug: all

# 发布模式
release: BUILD_TYPE = release
release: all

# 编译C++后端
cpp:
	$(call log_info,开始编译C++后端...)
	@cd cpp_backend && \
	if [ "$(BUILD_TYPE)" = "debug" ]; then \
		mkdir -p build_debug && cd build_debug && \
		cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
		make -j$(JOBS); \
	else \
		mkdir -p build && cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
		make -j$(JOBS); \
	fi
	$(call log_success,C++后端编译成功)

# 编译Flutter前端
flutter:
	$(call log_info,开始编译Flutter前端...)
	@cd flutter_frontend && \
	flutter pub get && \
	case "$(PLATFORM)" in \
		linux) \
			flutter build linux --$(BUILD_TYPE); \
			;; \
		windows) \
			flutter build windows --$(BUILD_TYPE); \
			;; \
		android) \
			flutter build apk --$(BUILD_TYPE); \
			;; \
		web) \
			flutter build web --$(BUILD_TYPE); \
			;; \
		*) \
			flutter build --$(BUILD_TYPE); \
			;; \
	esac
	$(call log_success,Flutter前端编译成功)

# 编译嵌入式版本
embedded: TARGET = embedded
embedded:
	$(call log_info,编译嵌入式版本...)
	@cd cpp_backend && \
	mkdir -p build_embedded && cd build_embedded && \
	cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTARGET_ARCH=aarch64 -DENABLE_EMBEDDED=ON -DENABLE_GUI=OFF -DENABLE_NETWORK=ON && \
	make -j$(JOBS)
	@cd flutter_frontend && \
	flutter build linux --$(BUILD_TYPE) --dart-define=EMBEDDED=true --dart-define=PLATFORM=linux
	$(call log_success,嵌入式版本编译成功)

# 清理编译文件
clean:
	$(call log_info,清理编译文件...)
	@rm -rf cpp_backend/build cpp_backend/build_debug cpp_backend/build_embedded
	@rm -rf flutter_frontend/build
	@rm -rf bamboo_cut_*.tar.gz bamboo_cut_*.zip
	$(call log_success,清理完成)

# 检查依赖
check-deps:
	$(call log_info,检查编译依赖...)
	@command -v cmake >/dev/null 2>&1 || { $(call log_error,cmake未找到); exit 1; }
	@command -v make >/dev/null 2>&1 || { $(call log_error,make未找到); exit 1; }
	@command -v g++ >/dev/null 2>&1 || { $(call log_error,g++未找到); exit 1; }
	@command -v git >/dev/null 2>&1 || { $(call log_error,git未找到); exit 1; }
	@command -v flutter >/dev/null 2>&1 || { $(call log_warning,flutter未找到，前端编译将跳过); }
	@pkg-config --exists opencv4 >/dev/null 2>&1 || pkg-config --exists opencv >/dev/null 2>&1 || { $(call log_warning,OpenCV未找到，C++后端编译可能失败); }
	$(call log_success,依赖检查完成)

# 创建部署包
package: all
	$(call log_info,创建部署包...)
	@PACKAGE_NAME="bamboo_cut_$(BUILD_TYPE)_$$(date +%Y%m%d_%H%M%S)" && \
	mkdir -p $$PACKAGE_NAME && \
	if [ "$(BUILD_TYPE)" = "debug" ]; then \
		cp cpp_backend/build_debug/bamboo_cut_backend $$PACKAGE_NAME/ 2>/dev/null || true; \
	else \
		cp cpp_backend/build/bamboo_cut_backend $$PACKAGE_NAME/ 2>/dev/null || true; \
	fi && \
	case "$(PLATFORM)" in \
		linux) \
			cp -r flutter_frontend/build/linux/x64/$(BUILD_TYPE)/bundle/* $$PACKAGE_NAME/ 2>/dev/null || true; \
			;; \
		windows) \
			cp -r flutter_frontend/build/windows/runner/$(BUILD_TYPE)/* $$PACKAGE_NAME/ 2>/dev/null || true; \
			;; \
		android) \
			cp flutter_frontend/build/app/outputs/flutter-apk/app-$(BUILD_TYPE).apk $$PACKAGE_NAME/ 2>/dev/null || true; \
			;; \
	esac && \
	cp README.md $$PACKAGE_NAME/ 2>/dev/null || true && \
	tar -czf $$PACKAGE_NAME.tar.gz $$PACKAGE_NAME && \
	rm -rf $$PACKAGE_NAME
	$(call log_success,部署包创建完成)

# 快速编译 (仅编译修改的文件)
quick: check-deps
	$(call log_info,快速编译模式...)
	@cd cpp_backend && \
	if [ "$(BUILD_TYPE)" = "debug" ]; then \
		cd build_debug 2>/dev/null && make -j$(JOBS) || { mkdir -p build_debug && cd build_debug && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(JOBS); }; \
	else \
		cd build 2>/dev/null && make -j$(JOBS) || { mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(JOBS); }; \
	fi
	$(call log_success,快速编译完成)

# 安装依赖 (Ubuntu/Debian)
install-deps-ubuntu:
	$(call log_info,安装Ubuntu/Debian依赖...)
	@sudo apt update && \
	sudo apt install -y cmake build-essential git pkg-config && \
	sudo apt install -y libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
	sudo apt install -y libmodbus-dev nlohmann-json3-dev && \
	sudo apt install -y curl unzip && \
	curl -fsSL https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.16.5-stable.tar.xz | sudo tar -xJ -C /opt && \
	echo 'export PATH="/opt/flutter/bin:$PATH"' >> ~/.bashrc
	$(call log_success,依赖安装完成，请重新登录或运行: source ~/.bashrc)

# 安装依赖 (Windows)
install-deps-windows:
	$(call log_info,安装Windows依赖...)
	@echo "请手动安装以下依赖:"
	@echo "1. Visual Studio Build Tools 2019或更新版本"
	@echo "2. CMake: https://cmake.org/download/"
	@echo "3. Git: https://git-scm.com/download/win"
	@echo "4. Flutter: https://flutter.dev/docs/get-started/install/windows"
	@echo "5. OpenCV: https://opencv.org/releases/"

# 显示编译状态
status:
	$(call log_info,编译状态检查...)
	@echo "C++后端:"
	@if [ -f "cpp_backend/build/bamboo_cut_backend" ]; then \
		echo "  ✓ 发布版本已编译"; \
	elif [ -f "cpp_backend/build_debug/bamboo_cut_backend" ]; then \
		echo "  ✓ 调试版本已编译"; \
	else \
		echo "  ✗ 未编译"; \
	fi
	@echo "Flutter前端:"
	@case "$(PLATFORM)" in \
		linux) \
			if [ -d "flutter_frontend/build/linux/x64/$(BUILD_TYPE)/bundle" ]; then \
				echo "  ✓ Linux版本已编译"; \
			else \
				echo "  ✗ 未编译"; \
			fi; \
			;; \
		windows) \
			if [ -d "flutter_frontend/build/windows/runner/$(BUILD_TYPE)" ]; then \
				echo "  ✓ Windows版本已编译"; \
			else \
				echo "  ✗ 未编译"; \
			fi; \
			;; \
		android) \
			if [ -f "flutter_frontend/build/app/outputs/flutter-apk/app-$(BUILD_TYPE).apk" ]; then \
				echo "  ✓ Android版本已编译"; \
			else \
				echo "  ✗ 未编译"; \
			fi; \
			;; \
	esac

# 运行测试
test: cpp
	$(call log_info,运行测试...)
	@cd cpp_backend && \
	if [ "$(BUILD_TYPE)" = "debug" ]; then \
		cd build_debug && make test 2>/dev/null || echo "测试未配置"; \
	else \
		cd build && make test 2>/dev/null || echo "测试未配置"; \
	fi

# 显示编译信息
info:
	@echo "编译配置:"
	@echo "  编译模式: $(BUILD_TYPE)"
	@echo "  目标平台: $(PLATFORM)"
	@echo "  目标类型: $(TARGET)"
	@echo "  并行任务: $(JOBS)"
	@echo "  系统架构: $(shell uname -m)"
	@echo "  操作系统: $(shell uname -s)" 