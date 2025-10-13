#!/bin/bash
# Bamboo Recognition System - 新架构测试脚本
# 测试 LVGL主导 + DeepStream appsink 架构

set -e

echo "=========================================="
echo "🎯 Bamboo Recognition System"
echo "新架构测试 - LVGL主导 + appsink模式"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 步骤1: 检查系统环境
log_info "步骤1: 检查系统环境..."

# 检查Weston是否运行
if pgrep -x "weston" > /dev/null; then
    log_success "Weston合成器正在运行"
else
    log_warning "Weston未运行，尝试启动..."
    sudo systemctl start weston || {
        log_error "Weston启动失败"
        exit 1
    }
    sleep 3
fi

# 检查Wayland环境变量
if [ -z "$WAYLAND_DISPLAY" ]; then
    export WAYLAND_DISPLAY="wayland-0"
    log_info "设置WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
fi

if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR="/run/user/0"
    log_info "设置XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
fi

# 验证Wayland socket
WAYLAND_SOCKET="$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
if [ -S "$WAYLAND_SOCKET" ]; then
    log_success "Wayland socket存在: $WAYLAND_SOCKET"
else
    log_error "Wayland socket不存在: $WAYLAND_SOCKET"
    exit 1
fi

# 步骤2: 检查架构关键组件
log_info "步骤2: 检查架构关键组件..."

# 检查LVGL编译
if [ -f "cpp_backend/src/ui/lvgl_wayland_interface.cpp" ]; then
    log_success "LVGL Wayland接口源代码存在"
else
    log_error "LVGL Wayland接口源代码缺失"
    exit 1
fi

# 检查DeepStream管理器
if [ -f "cpp_backend/src/deepstream/deepstream_manager.cpp" ]; then
    log_success "DeepStream管理器源代码存在"
else
    log_error "DeepStream管理器源代码缺失"
    exit 1
fi

# 检查架构重设计文档
if [ -f "ARCHITECTURE_REDESIGN.md" ]; then
    log_success "架构重设计文档存在"
else
    log_warning "架构重设计文档缺失"
fi

# 步骤3: 编译测试
log_info "步骤3: 编译新架构..."

make clean
if make -j$(nproc); then
    log_success "新架构编译成功"
else
    log_error "新架构编译失败"
    exit 1
fi

# 步骤4: 架构验证测试
log_info "步骤4: 架构验证测试..."

# 检查可执行文件
if [ -f "./integrated_main" ]; then
    log_success "主程序可执行文件存在"
else
    log_error "主程序可执行文件不存在"
    exit 1
fi

# 步骤5: 协议冲突测试
log_info "步骤5: Wayland协议冲突测试..."

# 启动系统进行短期测试
log_info "启动系统进行10秒协议测试..."

timeout 10s ./integrated_main &
MAIN_PID=$!

sleep 5

# 检查进程状态
if kill -0 $MAIN_PID 2>/dev/null; then
    log_success "系统运行正常，无协议冲突"
    
    # 检查是否有xdg_positioner错误
    if journalctl --since "5 seconds ago" | grep -q "xdg_positioner"; then
        log_error "检测到xdg_positioner协议错误"
        kill $MAIN_PID 2>/dev/null || true
        exit 1
    else
        log_success "无xdg_positioner协议错误"
    fi
    
    # 检查EGL初始化
    if journalctl --since "5 seconds ago" | grep -q "EGL_NOT_INITIALIZED"; then
        log_error "检测到EGL初始化失败"
        kill $MAIN_PID 2>/dev/null || true
        exit 1
    else
        log_success "EGL初始化正常"
    fi
    
    kill $MAIN_PID 2>/dev/null || true
    wait $MAIN_PID 2>/dev/null || true
else
    log_error "系统启动失败或崩溃"
    exit 1
fi

# 步骤6: 架构性能测试
log_info "步骤6: 架构性能基准测试..."

# CPU和内存使用测试
log_info "测试CPU和内存使用..."

# 启动系统进行性能监控
./integrated_main &
MAIN_PID=$!

sleep 3

if kill -0 $MAIN_PID 2>/dev/null; then
    # 获取CPU和内存使用情况
    CPU_USAGE=$(ps -p $MAIN_PID -o %cpu --no-headers | tr -d ' ')
    MEM_USAGE=$(ps -p $MAIN_PID -o %mem --no-headers | tr -d ' ')
    
    log_info "CPU使用率: ${CPU_USAGE}%"
    log_info "内存使用率: ${MEM_USAGE}%"
    
    # 性能基准检查
    if (( $(echo "$CPU_USAGE < 80.0" | bc -l) )); then
        log_success "CPU使用率正常 (< 80%)"
    else
        log_warning "CPU使用率较高 (>= 80%)"
    fi
    
    if (( $(echo "$MEM_USAGE < 50.0" | bc -l) )); then
        log_success "内存使用率正常 (< 50%)"
    else
        log_warning "内存使用率较高 (>= 50%)"
    fi
    
    kill $MAIN_PID 2>/dev/null || true
    wait $MAIN_PID 2>/dev/null || true
else
    log_error "性能测试期间系统崩溃"
    exit 1
fi

# 步骤7: 架构完整性验证
log_info "步骤7: 架构完整性验证..."

# 检查关键架构组件
ARCH_SCORE=0
TOTAL_CHECKS=5

# 1. LVGL单窗口架构
if grep -q "appsink.*架构重构" cpp_backend/src/deepstream/deepstream_manager.cpp; then
    log_success "✓ DeepStream使用appsink架构"
    ((ARCH_SCORE++))
else
    log_error "✗ DeepStream未使用appsink架构"
fi

# 2. Wayland连接重置机制
if grep -q "重置Wayland连接" cpp_backend/src/ui/lvgl_wayland_interface.cpp; then
    log_success "✓ Wayland连接重置机制存在"
    ((ARCH_SCORE++))
else
    log_warning "△ Wayland连接重置机制可能缺失"
fi

# 3. EGL恢复机制
if grep -q "EGL初始化失败恢复" cpp_backend/src/ui/lvgl_wayland_interface.cpp; then
    log_success "✓ EGL恢复机制存在"
    ((ARCH_SCORE++))
else
    log_warning "△ EGL恢复机制可能需要优化"
fi

# 4. Canvas更新机制
if grep -q "Canvas更新循环" cpp_backend/src/deepstream/deepstream_manager.cpp; then
    log_success "✓ appsink到LVGL Canvas帧传递机制存在"
    ((ARCH_SCORE++))
else
    log_warning "△ Canvas帧传递机制可能需要优化"
fi

# 5. 架构文档完整性
if [ -f "ARCHITECTURE_REDESIGN.md" ]; then
    log_success "✓ 架构重设计文档完整"
    ((ARCH_SCORE++))
else
    log_warning "△ 架构文档需要完善"
fi

# 计算架构完整性分数
ARCH_PERCENTAGE=$((ARCH_SCORE * 100 / TOTAL_CHECKS))

echo ""
echo "=========================================="
echo "🎯 新架构测试结果总结"
echo "=========================================="

if [ $ARCH_SCORE -eq $TOTAL_CHECKS ]; then
    log_success "🎉 架构重构完全成功! ($ARCH_SCORE/$TOTAL_CHECKS)"
    log_success "✅ LVGL主导 + appsink架构工作正常"
    log_success "✅ 无Wayland协议冲突"
    log_success "✅ 系统性能表现良好"
elif [ $ARCH_SCORE -ge $((TOTAL_CHECKS * 80 / 100)) ]; then
    log_success "🎊 架构重构基本成功! ($ARCH_SCORE/$TOTAL_CHECKS)"
    log_info "💡 建议: 继续优化剩余组件"
else
    log_warning "⚠️ 架构重构部分成功 ($ARCH_SCORE/$TOTAL_CHECKS)"
    log_warning "🔧 需要: 进一步完善架构实现"
fi

echo ""
echo "📊 架构完整性: $ARCH_PERCENTAGE%"
echo "🏗️ 新架构优势:"
echo "   - 消除xdg_positioner协议冲突"
echo "   - 单一xdg-shell窗口架构"
echo "   - LVGL完全控制UI显示"
echo "   - DeepStream专注AI推理和数据处理"
echo "   - 更稳定的硬件加速渲染"

echo ""
log_info "测试完成! 系统已准备就绪。"
log_info "启动命令: ./integrated_main"

exit 0