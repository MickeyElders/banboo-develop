#!/bin/bash

# 智能切竹机完全重新部署脚本
# 彻底停止老服务、清理缓存、删除老版本、重新编译、部署新版本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 脚本信息
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

log_info "🚀 开始智能切竹机完全重新部署流程"
log_info "📂 项目根目录: $PROJECT_ROOT"

# 步骤1: 彻底停止和清理现有服务
log_info "🛑 步骤1: 彻底停止和清理现有服务"

# 停止systemd服务
if systemctl is-active bamboo-cut-jetpack >/dev/null 2>&1; then
    log_info "   停止bambo-cut-jetpack服务..."
    sudo systemctl stop bamboo-cut-jetpack
    sleep 3
    log_success "   服务已停止"
else
    log_info "   服务未运行"
fi

# 杀死所有相关进程
log_info "   杀死所有相关进程..."
sudo pkill -f bamboo_cut_backend || true
sudo pkill -f bamboo_controller_qt || true
sudo pkill -f bamboo_cut_frontend || true

# 清理进程
sleep 2
log_success "   所有相关进程已清理"

# 步骤2: 删除老版本文件
log_info "🗑️ 步骤2: 删除老版本文件"

if [ -d "/opt/bamboo-cut" ]; then
    # 备份配置文件
    if [ -d "/opt/bamboo-cut/config" ]; then
        log_info "   备份用户配置..."
        sudo cp -r /opt/bamboo-cut/config /tmp/bamboo-cut-config-backup-$(date +%s) || true
    fi
    
    # 删除老版本
    log_info "   删除老版本: /opt/bamboo-cut"
    sudo rm -rf /opt/bamboo-cut
    log_success "   老版本已删除"
else
    log_info "   未发现老版本，跳过删除"
fi

# 步骤3: 清理构建缓存
log_info "🧹 步骤3: 清理构建缓存"

cd "$PROJECT_ROOT"

# 清理构建目录
BUILD_DIRS=(
    "build"
    "cpp_backend/build"
    "qt_frontend/build"
    "qt_frontend/build_debug"
    "qt_frontend/build_release"
)

for build_dir in "${BUILD_DIRS[@]}"; do
    if [ -d "$build_dir" ]; then
        log_info "   清理构建目录: $build_dir"
        rm -rf "$build_dir"
    fi
done

log_success "   构建缓存已清理"

# 步骤4: 强制重新编译
log_info "🔨 步骤4: 强制重新编译"

# 执行jetpack部署脚本进行完整重新编译
log_info "   执行完整重新编译和部署..."
if ./jetpack_deploy.sh --force-rebuild --upgrade --deploy local --gpu-opt --power-opt --models --qt-deploy; then
    log_success "   重新编译和部署成功"
else
    log_error "   重新编译和部署失败"
    exit 1
fi

# 步骤5: 验证部署
log_info "🔍 步骤5: 验证部署"

# 检查文件是否存在
if [ -f "/opt/bamboo-cut/bamboo_cut_backend" ]; then
    log_success "   C++后端程序已部署"
else
    log_error "   C++后端程序未找到"
fi

if [ -f "/opt/bamboo-cut/bamboo_controller_qt" ]; then
    log_success "   Qt前端程序已部署"
else
    log_warning "   Qt前端程序未找到"
fi

# 检查服务状态
if systemctl is-enabled bamboo-cut-jetpack >/dev/null 2>&1; then
    log_success "   系统服务已配置"
    
    # 启动服务
    log_info "   启动服务..."
    sudo systemctl start bamboo-cut-jetpack
    sleep 5
    
    # 检查服务状态
    if systemctl is-active bamboo-cut-jetpack >/dev/null 2>&1; then
        log_success "   ✅ 服务启动成功"
    else
        log_error "   ❌ 服务启动失败"
        log_info "   查看服务状态:"
        sudo systemctl status bamboo-cut-jetpack --no-pager || true
    fi
else
    log_error "   系统服务未配置"
fi

# 步骤6: 显示调试信息
log_info "📊 步骤6: 显示调试信息"

echo
log_info "=== 部署完成 ==="
log_info "📋 查看实时日志: sudo journalctl -u bamboo-cut-jetpack -f"
log_info "📋 查看服务状态: sudo systemctl status bamboo-cut-jetpack"
log_info "📋 查看后端日志: sudo tail -f /var/log/bamboo-cut/backend.log"

# 自动显示最近的日志
log_info "🔍 最近5分钟的服务日志:"
sudo journalctl -u bamboo-cut-jetpack --since "5 minutes ago" --no-pager || true

echo
log_success "🎉 完全重新部署流程完成！"

# 如果需要调试摄像头问题
echo
log_info "💡 如果仍有摄像头问题，可以执行以下调试命令:"
echo "   ls -la /dev/video*"
echo "   v4l2-ctl --list-devices"
echo "   lsmod | grep -E '(imx219|uvcvideo)'"
echo "   sudo journalctl -u bamboo-cut-jetpack -f"