#!/bin/bash
# scripts/test_drm_coordination.sh
# DRM资源协调方案验证测试脚本

echo "========================================="
echo "DRM资源协调方案验证测试"
echo "版本: 1.0.0"
echo "平台: Jetson Orin NX + Ubuntu 20.04/22.04"
echo "========================================="

# 脚本配置
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
LOG_DIR="/tmp/bamboo_drm_test"
TEST_LOG="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
APP_NAME="bamboo_integrated"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志记录函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${TEST_LOG}"
}

# 错误处理函数
error_exit() {
    log "❌ 错误: $1"
    exit 1
}

# 成功标记函数
success() {
    log "✅ $1"
}

# 警告标记函数
warning() {
    log "⚠️  $1"
}

# 检查是否为root用户或有sudo权限
check_permissions() {
    log "步骤0: 检查权限..."
    
    if [ "$EUID" -ne 0 ]; then
        if ! sudo -n true 2>/dev/null; then
            error_exit "需要sudo权限来访问DRM设备和系统状态"
        fi
        SUDO_CMD="sudo"
    else
        SUDO_CMD=""
    fi
    
    success "权限检查通过"
}

# 检查系统环境
check_system_environment() {
    log ""
    log "步骤1: 检查系统环境..."
    
    # 检查内核版本
    KERNEL_VERSION=$(uname -r)
    log "内核版本: $KERNEL_VERSION"
    
    # 检查架构
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        warning "当前架构是 $ARCH，但本脚本主要针对 Jetson Orin NX (aarch64)"
    else
        success "检测到Jetson Orin NX架构: $ARCH"
    fi
    
    # 检查nvidia-drm驱动
    if lsmod | grep -q nvidia_drm; then
        success "nvidia-drm驱动已加载"
        NVIDIA_DRM_MODESET=$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null || echo "N")
        log "nvidia-drm modeset参数: $NVIDIA_DRM_MODESET"
        
        if [ "$NVIDIA_DRM_MODESET" = "Y" ]; then
            success "nvidia-drm modeset已启用"
        else
            warning "nvidia-drm modeset未启用，可能影响KMS功能"
        fi
    else
        warning "nvidia-drm驱动未加载"
    fi
    
    # 检查显示管理器状态
    if systemctl is-active --quiet gdm3 || systemctl is-active --quiet lightdm || systemctl is-active --quiet sddm; then
        warning "检测到显示管理器正在运行，可能与DRM直接访问冲突"
        log "建议在测试期间临时停止显示管理器"
    else
        success "显示管理器未运行，适合DRM直接访问测试"
    fi
}

# 检查DRM设备状态
check_drm_devices() {
    log ""
    log "步骤2: 检查DRM设备状态..."
    
    # 检查DRM设备文件
    if [ ! -e "/dev/dri/card0" ]; then
        error_exit "DRM设备 /dev/dri/card0 不存在"
    fi
    
    # 列出所有DRM设备
    log "可用的DRM设备:"
    ls -la /dev/dri/ | tee -a "${TEST_LOG}"
    
    # 检查设备权限
    DRM_PERMISSIONS=$(stat -c "%a" /dev/dri/card0 2>/dev/null || echo "000")
    log "DRM设备权限: $DRM_PERMISSIONS"
    
    if [ "$DRM_PERMISSIONS" -lt "660" ]; then
        warning "DRM设备权限可能不足，当前: $DRM_PERMISSIONS"
    else
        success "DRM设备权限正常: $DRM_PERMISSIONS"
    fi
}

# 检查DRM资源占用情况（启动前）
check_drm_resources_before() {
    log ""
    log "步骤3: 检查DRM资源占用（启动前）..."
    
    # 检查DRM状态
    if [ -e "/sys/kernel/debug/dri/0/state" ]; then
        log "--- DRM状态 (启动前) ---"
        $SUDO_CMD cat /sys/kernel/debug/dri/0/state | grep -E "(crtc|plane)" | head -20 | tee -a "${TEST_LOG}"
        log "-------------------------"
    else
        warning "无法访问 /sys/kernel/debug/dri/0/state，可能需要启用debugfs"
    fi
    
    # 使用modetest检查DRM资源
    if command -v modetest >/dev/null 2>&1; then
        log "--- modetest资源信息 ---"
        $SUDO_CMD modetest -M nvidia-drm 2>/dev/null | head -50 | tee -a "${TEST_LOG}" || {
            warning "modetest执行失败，使用备用方法"
        }
        log "------------------------"
    else
        warning "modetest工具未找到，建议安装 libdrm-tests"
    fi
}

# 编译项目
build_project() {
    log ""
    log "步骤4: 编译项目..."
    
    cd "${PROJECT_ROOT}" || error_exit "无法进入项目根目录: $PROJECT_ROOT"
    
    # 创建build目录
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}" || error_exit "无法进入build目录"
    
    # 配置项目
    log "配置CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | tee -a "${TEST_LOG}"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        error_exit "CMake配置失败"
    fi
    
    # 编译项目
    log "编译项目..."
    make -j$(nproc) 2>&1 | tee -a "${TEST_LOG}"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        error_exit "项目编译失败"
    fi
    
    # 检查可执行文件
    if [ ! -f "${BUILD_DIR}/${APP_NAME}" ]; then
        error_exit "可执行文件未生成: ${BUILD_DIR}/${APP_NAME}"
    fi
    
    success "项目编译成功"
}

# 启动应用程序
start_application() {
    log ""
    log "步骤5: 启动应用程序..."
    
    cd "${BUILD_DIR}" || error_exit "无法进入build目录"
    
    # 设置环境变量
    export DISPLAY=:0
    export GST_DEBUG=2
    export GST_DEBUG_FILE="${LOG_DIR}/gstreamer_debug.log"
    
    # 启动应用（verbose模式）
    log "启动 $APP_NAME (详细模式)..."
    $SUDO_CMD ./${APP_NAME} --verbose 2>&1 | tee "${LOG_DIR}/app_startup.log" &
    APP_PID=$!
    
    log "应用PID: $APP_PID"
    echo $APP_PID > "${LOG_DIR}/app.pid"
    
    # 等待应用初始化
    log "等待应用初始化（10秒）..."
    sleep 10
    
    # 检查进程状态
    if ! kill -0 $APP_PID 2>/dev/null; then
        error_exit "应用进程已退出，检查日志: ${LOG_DIR}/app_startup.log"
    fi
    
    success "应用启动成功，PID: $APP_PID"
}

# 检查DRM资源占用情况（启动后）
check_drm_resources_after() {
    log ""
    log "步骤6: 检查DRM资源占用（启动后）..."
    
    # 等待资源分配稳定
    sleep 3
    
    if [ -e "/sys/kernel/debug/dri/0/state" ]; then
        log "--- DRM状态 (启动后) ---"
        $SUDO_CMD cat /sys/kernel/debug/dri/0/state | grep -E "(crtc|plane)" | head -20 | tee -a "${TEST_LOG}"
        log "-------------------------"
    fi
    
    # 使用modetest检查当前状态
    if command -v modetest >/dev/null 2>&1; then
        log "--- modetest当前状态 ---"
        $SUDO_CMD modetest -M nvidia-drm -p 2>/dev/null | tee -a "${TEST_LOG}" || {
            warning "modetest状态检查失败"
        }
        log "------------------------"
    fi
}

# 分析启动日志
analyze_startup_logs() {
    log ""
    log "步骤7: 分析启动日志..."
    
    STARTUP_LOG="${LOG_DIR}/app_startup.log"
    
    if [ ! -f "$STARTUP_LOG" ]; then
        warning "启动日志文件不存在: $STARTUP_LOG"
        return
    fi
    
    log "--- DRM协调器日志分析 ---"
    
    # 检查DRM协调器初始化
    if grep -q "DRM协调器.*初始化" "$STARTUP_LOG"; then
        success "DRM协调器初始化日志找到"
        grep "DRM协调器" "$STARTUP_LOG" | tail -10 | tee -a "${TEST_LOG}"
    else
        warning "未找到DRM协调器初始化日志"
    fi
    
    # 检查资源分配日志
    if grep -q "Overlay.*分配" "$STARTUP_LOG"; then
        success "Overlay资源分配日志找到"
        grep -E "(Overlay|Primary|Plane)" "$STARTUP_LOG" | tail -10 | tee -a "${TEST_LOG}"
    else
        warning "未找到Overlay资源分配日志"
    fi
    
    # 检查错误信息
    log "--- 错误检查 ---"
    
    if grep -q "No space left on device" "$STARTUP_LOG"; then
        log "❌ 检测到DRM资源冲突错误"
        grep -A 5 "No space left on device" "$STARTUP_LOG" | tee -a "${TEST_LOG}"
        return 1
    else
        success "无DRM资源冲突错误"
    fi
    
    if grep -q "设置primary plane失败" "$STARTUP_LOG"; then
        log "❌ Primary Plane设置失败"
        grep -A 3 "设置primary plane失败" "$STARTUP_LOG" | tee -a "${TEST_LOG}"
        return 1
    else
        success "Primary Plane设置正常"
    fi
    
    if grep -q "KMSSink.*成功" "$STARTUP_LOG"; then
        success "KMSSink管道创建成功"
    elif grep -q "AppSink.*软件合成" "$STARTUP_LOG"; then
        warning "降级到AppSink软件合成模式"
    else
        warning "视频渲染模式不明确"
    fi
    
    log "------------------------"
    return 0
}

# 性能测试
performance_test() {
    log ""
    log "步骤8: 性能测试（30秒采样）..."
    
    if [ ! -f "${LOG_DIR}/app.pid" ]; then
        warning "应用PID文件不存在，跳过性能测试"
        return
    fi
    
    APP_PID=$(cat "${LOG_DIR}/app.pid")
    
    if ! kill -0 $APP_PID 2>/dev/null; then
        warning "应用进程不存在，跳过性能测试"
        return
    fi
    
    log "监控应用性能 (PID: $APP_PID)..."
    
    # 创建性能日志文件
    PERF_LOG="${LOG_DIR}/performance.log"
    echo "时间,CPU%,内存MB,VSZ,RSS" > "$PERF_LOG"
    
    for i in $(seq 1 30); do
        if kill -0 $APP_PID 2>/dev/null; then
            # 获取进程信息
            PROC_INFO=$(ps -p $APP_PID -o pid,pcpu,pmem,vsz,rss --no-headers 2>/dev/null)
            
            if [ -n "$PROC_INFO" ]; then
                read -r pid cpu_pct mem_pct vsz rss <<< "$PROC_INFO"
                mem_mb=$((rss / 1024))
                
                echo "$(date '+%H:%M:%S'),$cpu_pct,$mem_mb,$vsz,$rss" >> "$PERF_LOG"
                log "[$i/30] CPU: ${cpu_pct}%, 内存: ${mem_mb}MB"
            else
                warning "[$i/30] 无法获取进程信息"
            fi
        else
            warning "[$i/30] 应用进程已退出"
            break
        fi
        
        sleep 1
    done
    
    # 分析性能数据
    if [ -f "$PERF_LOG" ] && [ $(wc -l < "$PERF_LOG") -gt 1 ]; then
        log "--- 性能统计 ---"
        
        # 计算平均值（跳过标题行）
        avg_cpu=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.2f", sum/count}' "$PERF_LOG")
        avg_mem=$(awk -F',' 'NR>1 {sum+=$3; count++} END {if(count>0) printf "%.0f", sum/count}' "$PERF_LOG")
        max_cpu=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {printf "%.2f", max}' "$PERF_LOG")
        max_mem=$(awk -F',' 'NR>1 {if($3>max) max=$3} END {printf "%.0f", max}' "$PERF_LOG")
        
        log "平均CPU使用率: ${avg_cpu}%"
        log "平均内存使用: ${avg_mem}MB"
        log "峰值CPU使用率: ${max_cpu}%"
        log "峰值内存使用: ${max_mem}MB"
        
        # 性能评估
        if (( $(echo "$avg_cpu < 30" | bc -l) )); then
            success "CPU使用率正常 (< 30%)"
        else
            warning "CPU使用率较高 (${avg_cpu}%)"
        fi
        
        if (( avg_mem < 500 )); then
            success "内存使用正常 (< 500MB)"
        else
            warning "内存使用较高 (${avg_mem}MB)"
        fi
        
        log "详细性能数据: $PERF_LOG"
        log "----------------"
    fi
}

# 功能验证测试
functionality_test() {
    log ""
    log "步骤9: 功能验证测试..."
    
    if [ ! -f "${LOG_DIR}/app.pid" ]; then
        warning "应用PID文件不存在，跳过功能测试"
        return
    fi
    
    APP_PID=$(cat "${LOG_DIR}/app.pid")
    
    if ! kill -0 $APP_PID 2>/dev/null; then
        warning "应用进程不存在，跳过功能测试"
        return
    fi
    
    # 测试1: 检查LVGL界面响应
    log "测试1: 检查LVGL界面响应..."
    if grep -q "LVGL.*初始化.*成功" "${LOG_DIR}/app_startup.log"; then
        success "LVGL界面初始化成功"
    else
        warning "LVGL界面初始化状态不明"
    fi
    
    # 测试2: 检查DeepStream视频流
    log "测试2: 检查DeepStream视频流..."
    if grep -q "DeepStream.*管道启动成功" "${LOG_DIR}/app_startup.log"; then
        success "DeepStream视频流启动成功"
    elif grep -q "DeepStream.*降级.*AppSink" "${LOG_DIR}/app_startup.log"; then
        warning "DeepStream降级到AppSink模式"
    else
        warning "DeepStream状态不明"
    fi
    
    # 测试3: 检查DRM资源协调
    log "测试3: 检查DRM资源协调..."
    if grep -q "资源分配无冲突" "${LOG_DIR}/app_startup.log"; then
        success "DRM资源协调正常"
    else
        warning "DRM资源协调状态不明"
    fi
    
    # 测试4: 持续运行稳定性
    log "测试4: 持续运行稳定性（10秒）..."
    stable_count=0
    for i in $(seq 1 10); do
        if kill -0 $APP_PID 2>/dev/null; then
            stable_count=$((stable_count + 1))
        fi
        sleep 1
    done
    
    if [ $stable_count -eq 10 ]; then
        success "应用持续运行稳定"
    else
        warning "应用运行不稳定 ($stable_count/10)"
    fi
}

# 清理和停止应用
cleanup_application() {
    log ""
    log "步骤10: 清理应用..."
    
    if [ -f "${LOG_DIR}/app.pid" ]; then
        APP_PID=$(cat "${LOG_DIR}/app.pid")
        
        if kill -0 $APP_PID 2>/dev/null; then
            log "正在停止应用 (PID: $APP_PID)..."
            
            # 先尝试优雅关闭
            kill -TERM $APP_PID
            sleep 3
            
            # 如果还在运行，强制关闭
            if kill -0 $APP_PID 2>/dev/null; then
                warning "优雅关闭失败，强制关闭应用"
                kill -KILL $APP_PID
                sleep 1
            fi
            
            if ! kill -0 $APP_PID 2>/dev/null; then
                success "应用已停止"
            else
                warning "无法停止应用进程"
            fi
        fi
        
        rm -f "${LOG_DIR}/app.pid"
    fi
}

# 生成测试报告
generate_report() {
    log ""
    log "步骤11: 生成测试报告..."
    
    REPORT_FILE="${LOG_DIR}/test_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# DRM资源协调方案测试报告

## 测试信息
- 测试时间: $(date)
- 测试平台: $(uname -a)
- 项目路径: $PROJECT_ROOT
- 日志目录: $LOG_DIR

## 测试结果摘要

### 系统环境
- 内核版本: $(uname -r)
- 架构: $(uname -m)
- nvidia-drm状态: $(lsmod | grep nvidia_drm | wc -l > 0 && echo "已加载" || echo "未加载")

### 功能测试结果
EOF

    # 分析测试结果并添加到报告
    if [ -f "${LOG_DIR}/app_startup.log" ]; then
        echo "" >> "$REPORT_FILE"
        echo "### 应用启动日志分析" >> "$REPORT_FILE"
        
        if grep -q "DRM协调器.*初始化.*成功" "${LOG_DIR}/app_startup.log"; then
            echo "- ✅ DRM协调器初始化成功" >> "$REPORT_FILE"
        else
            echo "- ❌ DRM协调器初始化失败" >> "$REPORT_FILE"
        fi
        
        if ! grep -q "No space left on device" "${LOG_DIR}/app_startup.log"; then
            echo "- ✅ 无DRM资源冲突错误" >> "$REPORT_FILE"
        else
            echo "- ❌ 检测到DRM资源冲突" >> "$REPORT_FILE"
        fi
        
        if grep -q "LVGL.*初始化.*成功" "${LOG_DIR}/app_startup.log"; then
            echo "- ✅ LVGL界面初始化成功" >> "$REPORT_FILE"
        else
            echo "- ⚠️  LVGL界面状态不明" >> "$REPORT_FILE"
        fi
        
        if grep -q "DeepStream.*Overlay.*硬件渲染" "${LOG_DIR}/app_startup.log"; then
            echo "- ✅ DeepStream使用硬件Overlay渲染" >> "$REPORT_FILE"
        elif grep -q "DeepStream.*AppSink.*软件合成" "${LOG_DIR}/app_startup.log"; then
            echo "- ⚠️  DeepStream降级到软件合成模式" >> "$REPORT_FILE"
        else
            echo "- ❓ DeepStream渲染模式不明" >> "$REPORT_FILE"
        fi
    fi
    
    # 添加性能数据
    if [ -f "${LOG_DIR}/performance.log" ]; then
        echo "" >> "$REPORT_FILE"
        echo "### 性能测试结果" >> "$REPORT_FILE"
        
        if [ $(wc -l < "${LOG_DIR}/performance.log") -gt 1 ]; then
            avg_cpu=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.2f", sum/count}' "${LOG_DIR}/performance.log")
            avg_mem=$(awk -F',' 'NR>1 {sum+=$3; count++} END {if(count>0) printf "%.0f", sum/count}' "${LOG_DIR}/performance.log")
            
            echo "- 平均CPU使用率: ${avg_cpu}%" >> "$REPORT_FILE"
            echo "- 平均内存使用: ${avg_mem}MB" >> "$REPORT_FILE"
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "### 相关文件" >> "$REPORT_FILE"
    echo "- 完整测试日志: $TEST_LOG" >> "$REPORT_FILE"
    echo "- 应用启动日志: ${LOG_DIR}/app_startup.log" >> "$REPORT_FILE"
    echo "- 性能测试数据: ${LOG_DIR}/performance.log" >> "$REPORT_FILE"
    echo "- GStreamer调试日志: ${LOG_DIR}/gstreamer_debug.log" >> "$REPORT_FILE"
    
    log "测试报告已生成: $REPORT_FILE"
}

# 主测试流程
main() {
    log "开始DRM资源协调方案验证测试"
    log "测试日志: $TEST_LOG"
    
    # 设置错误处理
    set -e
    trap 'cleanup_application; log "测试被中断"; exit 1' INT TERM
    
    # 执行测试步骤
    check_permissions
    check_system_environment
    check_drm_devices
    check_drm_resources_before
    build_project
    start_application
    
    # 等待应用稳定
    sleep 5
    
    check_drm_resources_after
    
    # 分析日志，如果发现严重错误则提前结束
    if ! analyze_startup_logs; then
        log "检测到严重错误，提前结束测试"
        cleanup_application
        generate_report
        exit 1
    fi
    
    performance_test
    functionality_test
    cleanup_application
    generate_report
    
    log ""
    log "========================================="
    log "测试完成！"
    log "详细日志: $TEST_LOG"
    log "测试报告: ${LOG_DIR}/test_report_*.md"
    log "========================================="
}

# 运行主测试流程
main "$@"