/**
 * @file system_monitor.cpp
 * @brief C++ LVGL一体化系统监控器实现
 * @version 5.0.0
 * @date 2024
 */

#include "bamboo_cut/utils/system_monitor.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdio>

namespace bamboo_cut {
namespace utils {

SystemMonitor& SystemMonitor::getInstance() {
    static SystemMonitor instance;
    return instance;
}

SystemMonitor::SystemMonitor() 
    : running_(false) {
}

SystemMonitor::~SystemMonitor() {
    stop();
}

void SystemMonitor::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    monitor_thread_ = std::thread(&SystemMonitor::monitorLoop, this);
    
    std::cout << "[SystemMonitor] 系统监控已启动" << std::endl;
}

void SystemMonitor::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    std::cout << "[SystemMonitor] 系统监控已停止" << std::endl;
}

SystemStats SystemMonitor::getSystemStats() {
    SystemStats stats;
    
    // 获取CPU使用率
    stats.cpu_usage = getCPUUsage();
    
    // 获取内存使用率
    stats.memory_usage = getMemoryUsage();
    
    // 获取系统温度
    stats.temperature = getTemperature();
    
    return stats;
}

float SystemMonitor::getCPUUsage() {
    // TODO: 实现CPU使用率检测
    return 25.5f; // 示例值
}

float SystemMonitor::getMemoryUsage() {
    // 读取/proc/meminfo获取内存信息
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0.0f;
    }
    
    std::string line;
    long total_memory = 0;
    long available_memory = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %ld kB", &total_memory);
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %ld kB", &available_memory);
        }
    }
    
    if (total_memory > 0) {
        float used_memory = total_memory - available_memory;
        return (used_memory / total_memory) * 100.0f;
    }
    
    return 0.0f;
}

float SystemMonitor::getTemperature() {
    // 尝试读取系统温度
    std::ifstream thermal("/sys/class/thermal/thermal_zone0/temp");
    if (!thermal.is_open()) {
        return 0.0f;
    }
    
    std::string temp_str;
    std::getline(thermal, temp_str);
    
    try {
        int temp_millidegrees = std::stoi(temp_str);
        return temp_millidegrees / 1000.0f;
    } catch (...) {
        return 0.0f;
    }
}

void SystemMonitor::monitorLoop() {
    while (running_) {
        SystemStats stats = getSystemStats();
        
        // 检查系统状态
        if (stats.cpu_usage > 90.0f) {
            std::cout << "[SystemMonitor] 警告: CPU使用率过高: " << stats.cpu_usage << "%" << std::endl;
        }
        
        if (stats.memory_usage > 90.0f) {
            std::cout << "[SystemMonitor] 警告: 内存使用率过高: " << stats.memory_usage << "%" << std::endl;
        }
        
        if (stats.temperature > 80.0f) {
            std::cout << "[SystemMonitor] 警告: 系统温度过高: " << stats.temperature << "°C" << std::endl;
        }
        
        // 5秒检查一次
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

} // namespace utils
} // namespace bamboo_cut