#!/bin/bash

# ========================================
# Bamboo Recognition System - 性能基准测试脚本
# 
# 功能：对比Wayland迁移前后的性能指标
# 版本：1.0.0
# 日期：2024-12-12
# 作者：Bamboo Development Team
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置
BENCHMARK_DURATION=60  # 基准测试持续时间（秒）
SAMPLE_INTERVAL=1      # 采样间隔（秒）
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

log_metric() {
    echo -e "${CYAN}[METRIC]${NC} $1"
}

# 创建结果目录
setup_results_dir() {
    mkdir -p "$RESULTS_DIR"
    log_info "基准测试结果将保存到: $RESULTS_DIR"
}

# ========================================
# 系统信息收集
# ========================================

collect_system_info() {
    log_info "收集系统信息..."
    
    local info_file="$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    
    {
        echo "========== 系统信息 =========="
        echo "时间戳: $(date)"
        echo "主机名: $(hostname)"
        echo "内核版本: $(uname -r)"
        echo "发行版: $(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"
        echo ""
        
        echo "========== 硬件信息 =========="
        echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
        echo "CPU核心数: $(nproc)"
        echo "总内存: $(free -h | awk '/^Mem:/ {print $2}')"
        echo ""
        
        # Jetson特定信息
        if command -v tegrastats > /dev/null 2>&1; then
            echo "========== Jetson信息 =========="
            echo "平台: Jetson (检测到tegrastats)"
            timeout 2s tegrastats 2>/dev/null | head -n 1 || echo "无法获取tegrastats"
            echo ""
        fi
        
        # GPU信息
        if command -v nvidia-smi > /dev/null 2>&1; then
            echo "========== GPU信息 =========="
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
            echo ""
        fi
        
        echo "========== 显示环境 =========="
        echo "DISPLAY: ${DISPLAY:-'未设置'}"
        echo "WAYLAND_DISPLAY: ${WAYLAND_DISPLAY:-'未设置'}"
        echo "XDG_SESSION_TYPE: ${XDG_SESSION_TYPE:-'未设置'}"
        echo ""
        
        echo "========== Weston信息 =========="
        if pgrep weston > /dev/null; then
            echo "Weston状态: 运行中"
            ps aux | grep weston | grep -v grep
        else
            echo "Weston状态: 未运行"
        fi
        echo ""
        
    } > "$info_file"
    
    log_success "系统信息已保存到: $info_file"
}

# ========================================
# CPU性能测试
# ========================================

benchmark_cpu() {
    log_info "开始CPU性能基准测试..."
    
    local cpu_file="$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv"
    echo "timestamp,cpu_usage_percent,load_1min,load_5min,load_15min" > "$cpu_file"
    
    log_info "CPU基准测试运行 $BENCHMARK_DURATION 秒..."
    
    for ((i=0; i<BENCHMARK_DURATION; i+=SAMPLE_INTERVAL)); do
        local timestamp=$(date +%s)
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
        local load_avg=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^ *//')
        local load_1min=$(echo "$load_avg" | cut -d',' -f1 | sed 's/^ *//')
        local load_5min=$(echo "$load_avg" | cut -d',' -f2 | sed 's/^ *//')
        local load_15min=$(echo "$load_avg" | cut -d',' -f3 | sed 's/^ *//')
        
        echo "$timestamp,$cpu_usage,$load_1min,$load_5min,$load_15min" >> "$cpu_file"
        
        if [ $((i % 10)) -eq 0 ]; then
            log_metric "CPU使用率: ${cpu_usage}%, 负载: $load_1min"
        fi
        
        sleep $SAMPLE_INTERVAL
    done
    
    log_success "CPU基准测试完成，结果保存到: $cpu_file"
}

# ========================================
# 内存性能测试
# ========================================

benchmark_memory() {
    log_info "开始内存性能基准测试..."
    
    local mem_file="$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv"
    echo "timestamp,total_mb,used_mb,free_mb,available_mb,usage_percent" > "$mem_file"
    
    log_info "内存基准测试运行 $BENCHMARK_DURATION 秒..."
    
    for ((i=0; i<BENCHMARK_DURATION; i+=SAMPLE_INTERVAL)); do
        local timestamp=$(date +%s)\n        local mem_info=$(free -m | awk '/^Mem:/ {print $2","$3","$4","$7","($3/$2)*100}')\n        \n        echo "$timestamp,$mem_info" >> "$mem_file"\n        \n        if [ $((i % 10)) -eq 0 ]; then\n            local used_mb=$(echo "$mem_info" | cut -d',' -f2)\n            local usage_percent=$(echo "$mem_info" | cut -d',' -f5 | cut -d'.' -f1)\n            log_metric "内存使用: ${used_mb}MB (${usage_percent}%)"\n        fi\n        \n        sleep $SAMPLE_INTERVAL\n    done\n    \n    log_success "内存基准测试完成，结果保存到: $mem_file"\n}\n\n# ========================================\n# GPU性能测试 (Jetson/NVIDIA)\n# ========================================\n\nbenchmark_gpu() {\n    log_info "开始GPU性能基准测试..."\n    \n    local gpu_file="$RESULTS_DIR/gpu_benchmark_$TIMESTAMP.csv"\n    \n    if command -v nvidia-smi > /dev/null 2>&1; then\n        echo "timestamp,gpu_utilization,memory_used_mb,memory_total_mb,temperature_c,power_watts" > "$gpu_file"\n        \n        log_info "NVIDIA GPU基准测试运行 $BENCHMARK_DURATION 秒..."\n        \n        for ((i=0; i<BENCHMARK_DURATION; i+=SAMPLE_INTERVAL)); do\n            local timestamp=$(date +%s)\n            local gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)\n            \n            # 清理数据\n            gpu_stats=$(echo "$gpu_stats" | sed 's/ //g')\n            \n            echo "$timestamp,$gpu_stats" >> "$gpu_file"\n            \n            if [ $((i % 10)) -eq 0 ]; then\n                local gpu_util=$(echo "$gpu_stats" | cut -d',' -f1)\n                local temp=$(echo "$gpu_stats" | cut -d',' -f4)\n                log_metric "GPU使用率: ${gpu_util}%, 温度: ${temp}°C"\n            fi\n            \n            sleep $SAMPLE_INTERVAL\n        done\n        \n        log_success "GPU基准测试完成，结果保存到: $gpu_file"\n        \n    elif command -v tegrastats > /dev/null 2>&1; then\n        echo "timestamp,tegrastats_output" > "$gpu_file"\n        \n        log_info "Jetson GPU基准测试运行 $BENCHMARK_DURATION 秒..."\n        \n        timeout $BENCHMARK_DURATION tegrastats --interval 1000 | while read line; do\n            local timestamp=$(date +%s)\n            echo "$timestamp,\"$line\"" >> "$gpu_file"\n        done\n        \n        log_success "Jetson GPU基准测试完成，结果保存到: $gpu_file"\n    else\n        log_warning "未检测到GPU监控工具，跳过GPU基准测试"\n    fi\n}\n\n# ========================================\n# Wayland/GStreamer性能测试\n# ========================================\n\nbenchmark_wayland_performance() {\n    log_info "开始Wayland性能基准测试..."\n    \n    if [ -z "$WAYLAND_DISPLAY" ]; then\n        log_warning "WAYLAND_DISPLAY未设置，跳过Wayland性能测试"\n        return\n    fi\n    \n    local wayland_file="$RESULTS_DIR/wayland_benchmark_$TIMESTAMP.txt"\n    \n    {\n        echo "========== Wayland性能测试 =========="\n        echo "测试时间: $(date)"\n        echo ""\n        \n        # 测试waylandsink性能\n        echo "waylandsink性能测试:"\n        local start_time=$(date +%s.%N)\n        \n        # 运行30秒的waylandsink测试\n        timeout 30s gst-launch-1.0 \\\n            videotestsrc pattern=ball num-buffers=900 ! \\\n            video/x-raw,width=1920,height=1080,framerate=30/1 ! \\\n            waylandsink sync=false \\\n            > /tmp/waylandsink_test.log 2>&1 || true\n        \n        local end_time=$(date +%s.%N)\n        local duration=$(echo "$end_time - $start_time" | bc)\n        \n        echo "waylandsink测试持续时间: ${duration}秒"\n        \n        # 分析日志中的性能信息\n        if [ -f "/tmp/waylandsink_test.log" ]; then\n            echo "waylandsink日志分析:"\n            grep -i "fps\\|framerate\\|dropped\\|late" /tmp/waylandsink_test.log || echo "无性能信息"\n            rm -f /tmp/waylandsink_test.log\n        fi\n        \n        echo ""\n        \n        # EGL性能测试\n        echo "EGL初始化性能测试:"\n        local egl_start=$(date +%s.%N)\n        \n        # 简单的EGL初始化测试\n        timeout 10s /tmp/egl_wayland_test > /dev/null 2>&1 || echo "EGL测试失败"\n        \n        local egl_end=$(date +%s.%N)\n        local egl_duration=$(echo "$egl_end - $egl_start" | bc)\n        echo "EGL初始化耗时: ${egl_duration}秒"\n        \n    } > "$wayland_file"\n    \n    log_success "Wayland性能测试完成，结果保存到: $wayland_file"\n}\n\n# ========================================\n# 应用程序性能测试\n# ========================================\n\nbenchmark_application() {\n    log_info "开始应用程序性能基准测试..."\n    \n    if [ ! -f "build/bamboo_integrated" ]; then\n        log_warning "应用程序可执行文件不存在，跳过应用程序性能测试"\n        return\n    fi\n    \n    local app_file="$RESULTS_DIR/application_benchmark_$TIMESTAMP.csv"\n    echo "timestamp,cpu_percent,memory_mb,threads,status" > "$app_file"\n    \n    log_info "启动应用程序进行性能测试..."\n    \n    # 启动应用程序\n    (cd build && timeout 60s ./bamboo_integrated > /tmp/bamboo_app.log 2>&1) &\n    local APP_PID=$!\n    \n    log_info "应用程序PID: $APP_PID，监控60秒..."\n    \n    # 监控应用程序性能\n    for ((i=0; i<60; i++)); do\n        if kill -0 $APP_PID 2>/dev/null; then\n            local timestamp=$(date +%s)\n            \n            # 获取进程信息\n            local proc_info=$(ps -o pid,pcpu,rss,nlwp,stat -p $APP_PID --no-headers 2>/dev/null || echo "0 0 0 0 Z")\n            local cpu_percent=$(echo "$proc_info" | awk '{print $2}')\n            local memory_kb=$(echo "$proc_info" | awk '{print $3}')\n            local memory_mb=$((memory_kb / 1024))\n            local threads=$(echo "$proc_info" | awk '{print $4}')\n            local status=$(echo "$proc_info" | awk '{print $5}')\n            \n            echo "$timestamp,$cpu_percent,$memory_mb,$threads,$status" >> "$app_file"\n            \n            if [ $((i % 10)) -eq 0 ]; then\n                log_metric "应用CPU: ${cpu_percent}%, 内存: ${memory_mb}MB, 线程: $threads"\n            fi\n        else\n            log_error "应用程序在${i}秒后退出"\n            break\n        fi\n        \n        sleep 1\n    done\n    \n    # 清理\n    kill $APP_PID 2>/dev/null || true\n    wait $APP_PID 2>/dev/null || true\n    \n    log_success "应用程序性能测试完成，结果保存到: $app_file"\n    \n    # 分析应用程序日志\n    if [ -f "/tmp/bamboo_app.log" ]; then\n        local log_file="$RESULTS_DIR/application_log_$TIMESTAMP.txt"\n        cp /tmp/bamboo_app.log "$log_file"\n        \n        local error_count=$(grep -i "error\\|failed\\|crash" "$log_file" | wc -l)\n        local warning_count=$(grep -i "warning\\|warn" "$log_file" | wc -l)\n        \n        log_info "应用程序日志分析: $error_count 个错误, $warning_count 个警告"\n        rm -f /tmp/bamboo_app.log\n    fi\n}\n\n# ========================================\n# 网络性能测试\n# ========================================\n\nbenchmark_network() {\n    log_info "开始网络性能基准测试..."\n    \n    local network_file="$RESULTS_DIR/network_benchmark_$TIMESTAMP.txt"\n    \n    {\n        echo "========== 网络性能测试 =========="\n        echo "测试时间: $(date)"\n        echo ""\n        \n        # 网络接口信息\n        echo "网络接口信息:"\n        ip link show | grep -E "^[0-9]+:" | awk '{print $2}' | sed 's/:$//' | while read iface; do\n            if [ "$iface" != "lo" ]; then\n                local stats=$(cat /sys/class/net/$iface/statistics/rx_bytes 2>/dev/null || echo "0")\n                local rx_mb=$((stats / 1024 / 1024))\n                local tx_stats=$(cat /sys/class/net/$iface/statistics/tx_bytes 2>/dev/null || echo "0")\n                local tx_mb=$((tx_stats / 1024 / 1024))\n                echo "  $iface: RX ${rx_mb}MB, TX ${tx_mb}MB"\n            fi\n        done\n        echo ""\n        \n        # 连接测试\n        echo "连接测试:"\n        if ping -c 3 8.8.8.8 > /tmp/ping_test.log 2>&1; then\n            local latency=$(grep "avg" /tmp/ping_test.log | awk -F'/' '{print $5}' 2>/dev/null || echo "unknown")\n            echo "  互联网连接: 正常 (延迟: ${latency}ms)"\n        else\n            echo "  互联网连接: 失败"\n        fi\n        rm -f /tmp/ping_test.log\n        \n        # 本地回环测试\n        if ping -c 1 127.0.0.1 > /dev/null 2>&1; then\n            echo "  本地回环: 正常"\n        else\n            echo "  本地回环: 失败"\n        fi\n        \n    } > "$network_file"\n    \n    log_success "网络性能测试完成，结果保存到: $network_file"\n}\n\n# ========================================\n# 生成性能报告\n# ========================================\n\ngenerate_performance_report() {\n    log_info "生成性能基准测试报告..."\n    \n    local report_file="$RESULTS_DIR/performance_report_$TIMESTAMP.html"\n    \n    cat > "$report_file" << EOF\n<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Bamboo Recognition System - 性能基准测试报告</title>\n    <style>\n        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n        h1, h2, h3 { color: #2c3e50; }\n        .metric { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 4px; }\n        .success { color: #27ae60; }\n        .warning { color: #f39c12; }\n        .error { color: #e74c3c; }\n        .info { color: #3498db; }\n        table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n        th { background-color: #34495e; color: white; }\n        .chart-container { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 4px; }\n    </style>\n</head>\n<body>\n    <div class=\"container\">\n        <h1>Bamboo Recognition System</h1>\n        <h2>Wayland架构迁移 - 性能基准测试报告</h2>\n        \n        <div class=\"metric\">\n            <strong>测试时间:</strong> $(date)<br>\n            <strong>测试持续时间:</strong> ${BENCHMARK_DURATION}秒<br>\n            <strong>采样间隔:</strong> ${SAMPLE_INTERVAL}秒\n        </div>\n        \n        <h3>🖥️ 系统信息</h3>\n        <div class=\"chart-container\">\nEOF\n\n    # 添加系统信息\n    if [ -f "$RESULTS_DIR/system_info_$TIMESTAMP.txt" ]; then\n        echo "            <pre>" >> "$report_file"\n        cat "$RESULTS_DIR/system_info_$TIMESTAMP.txt" >> "$report_file"\n        echo "            </pre>" >> "$report_file"\n    fi\n    \n    cat >> "$report_file" << EOF\n        </div>\n        \n        <h3>📊 性能指标摘要</h3>\n        <table>\n            <tr><th>指标</th><th>平均值</th><th>最大值</th><th>状态</th></tr>\nEOF\n\n    # 分析CPU数据\n    if [ -f "$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv" ]; then\n        local avg_cpu=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.1f", sum/count}' "$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv")\n        local max_cpu=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {printf "%.1f", max}' "$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv")\n        local cpu_status="正常"\n        if (( $(echo "$avg_cpu > 80" | bc -l) )); then\n            cpu_status="<span class='warning'>高负载</span>"\n        elif (( $(echo "$avg_cpu > 90" | bc -l) )); then\n            cpu_status="<span class='error'>过载</span>"\n        else\n            cpu_status="<span class='success'>正常</span>"\n        fi\n        echo "            <tr><td>CPU使用率 (%)</td><td>$avg_cpu</td><td>$max_cpu</td><td>$cpu_status</td></tr>" >> "$report_file"\n    fi\n    \n    # 分析内存数据\n    if [ -f "$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv" ]; then\n        local avg_mem=$(awk -F',' 'NR>1 {sum+=$6; count++} END {if(count>0) printf "%.1f", sum/count}' "$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv")\n        local max_mem=$(awk -F',' 'NR>1 {if($6>max) max=$6} END {printf "%.1f", max}' "$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv")\n        local mem_status="正常"\n        if (( $(echo "$avg_mem > 80" | bc -l) )); then\n            mem_status="<span class='warning'>高使用</span>"\n        elif (( $(echo "$avg_mem > 90" | bc -l) )); then\n            mem_status="<span class='error'>临界</span>"\n        else\n            mem_status="<span class='success'>正常</span>"\n        fi\n        echo "            <tr><td>内存使用率 (%)</td><td>$avg_mem</td><td>$max_mem</td><td>$mem_status</td></tr>" >> "$report_file"\n    fi\n    \n    cat >> "$report_file" << EOF\n        </table>\n        \n        <h3>📈 详细性能图表</h3>\n        <div class=\"chart-container\">\n            <p><strong>注意:</strong> 详细的性能数据已保存为CSV文件，可以使用Excel、Python等工具进行进一步分析。</p>\n            <ul>\nEOF\n\n    # 列出所有生成的文件\n    for file in $(ls "$RESULTS_DIR"/*_$TIMESTAMP.* 2>/dev/null | grep -v \\.html); do\n        local filename=$(basename "$file")\n        echo "                <li><code>$filename</code></li>" >> "$report_file"\n    done\n    \n    cat >> "$report_file" << EOF\n            </ul>\n        </div>\n        \n        <h3>🔍 性能分析建议</h3>\n        <div class=\"chart-container\">\n            <h4>优化建议:</h4>\n            <ul>\nEOF\n\n    # 根据性能数据生成建议\n    if [ -f "$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv" ]; then\n        local avg_cpu=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.0f", sum/count}' "$RESULTS_DIR/cpu_benchmark_$TIMESTAMP.csv")\n        if [ \"$avg_cpu\" -gt 70 ]; then\n            echo "                <li class='warning'>CPU使用率较高($avg_cpu%)，建议优化算法或增加硬件资源</li>" >> "$report_file"\n        fi\n    fi\n    \n    if [ -f "$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv" ]; then\n        local avg_mem=$(awk -F',' 'NR>1 {sum+=$6; count++} END {if(count>0) printf "%.0f", sum/count}' "$RESULTS_DIR/memory_benchmark_$TIMESTAMP.csv")\n        if [ \"$avg_mem\" -gt 80 ]; then\n            echo "                <li class='warning'>内存使用率较高($avg_mem%)，建议优化内存管理或增加RAM</li>" >> "$report_file"\n        fi\n    fi\n    \n    echo "                <li class='info'>Wayland架构相比DRM直接访问具有更好的稳定性和兼容性</li>" >> "$report_file"\n    echo "                <li class='success'>建议定期运行此基准测试以监控系统性能</li>" >> "$report_file"\n    \n    cat >> "$report_file" << EOF\n            </ul>\n        </div>\n        \n        <div class=\"metric\">\n            <strong>报告生成时间:</strong> $(date)<br>\n            <strong>测试数据位置:</strong> $RESULTS_DIR\n        </div>\n    </div>\n</body>\n</html>\nEOF\n\n    log_success "性能报告已生成: $report_file"\n    log_info "您可以在浏览器中打开查看详细报告"\n}\n\n# ========================================\n# 主函数\n# ========================================\n\nmain() {\n    log_info "========================================="\n    log_info "Bamboo Recognition System - 性能基准测试"\n    log_info "开始时间: $(date)"\n    log_info "测试持续时间: ${BENCHMARK_DURATION}秒"\n    log_info "========================================="\n    \n    # 检查依赖\n    if ! command -v bc > /dev/null 2>&1; then\n        log_error "bc计算器未安装，请安装: sudo apt install bc"\n        exit 1\n    fi\n    \n    # 设置结果目录\n    setup_results_dir\n    \n    # 收集系统信息\n    collect_system_info\n    \n    # 运行性能基准测试\n    benchmark_cpu &\n    CPU_PID=$!\n    \n    benchmark_memory &\n    MEM_PID=$!\n    \n    benchmark_gpu &\n    GPU_PID=$!\n    \n    # 等待基础监控完成\n    wait $CPU_PID $MEM_PID $GPU_PID\n    \n    # 运行专项测试\n    benchmark_wayland_performance\n    benchmark_application\n    benchmark_network\n    \n    # 生成报告\n    generate_performance_report\n    \n    log_info "========================================="\n    log_success "性能基准测试完成！"\n    log_info "结果保存在: $RESULTS_DIR"\n    log_info "查看HTML报告: $RESULTS_DIR/performance_report_$TIMESTAMP.html"\n    log_info "========================================="\n}\n\n# 运行主函数\nmain "$@"\n