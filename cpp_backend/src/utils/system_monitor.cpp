/**
 * @file system_monitor.cpp
 * @brief C++ LVGL一体化系统监控器实现
 * @version 5.0.0
 * @date 2024
 * 
 * 提供系统资源监控和性能统计功能
 */

#include "bamboo_cut/utils/system_monitor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>

namespace bamboo_cut {
namespace utils {

SystemMonitor& SystemMonitor::getInstance() {
    static SystemMonitor instance;
    return instance;
}

SystemMonitor::SystemMonitor() {
    start_time_ = std::chrono::system_clock::now();
}

SystemMonitor::~SystemMonitor() {
    stop();
}

bool SystemMonitor::initialize() {
    return true;  // 基本初始化完成
}

void SystemMonitor::start() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    monitoring_.store(true);
    
    // 启动监控（这里简化处理）
    updateCPUUsage();
    updateMemoryUsage();
    updateTemperatures();
    updatePowerDraw();
}

void SystemMonitor::stop() {
    monitoring_.store(false);
}

SystemResources SystemMonitor::getSystemResources() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return resources_;
}

NetworkStats SystemMonitor::getNetworkStats() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return network_stats_;
}

std::chrono::seconds SystemMonitor::getUptime() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now - start_time_;
    return std::chrono::duration_cast<std::chrono::seconds>(duration);
}

std::array<float, 3> SystemMonitor::getLoadAverage() const {
    std::array<float, 3> load = {0.0f, 0.0f, 0.0f};
    
    std::ifstream file("/proc/loadavg");
    if (file.is_open()) {
        file >> load[0] >> load[1] >> load[2];
        file.close();
    }
    
    return load;
}

bool SystemMonitor::isOverheating() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return resources_.cpu_temperature > CPU_TEMP_THRESHOLD ||
           resources_.gpu_temperature > GPU_TEMP_THRESHOLD;
}

void SystemMonitor::setMonitorInterval(int interval_ms) {
    monitor_interval_ms_ = interval_ms;
}

void SystemMonitor::updateCPUUsage() {
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        resources_.cpu_usage = 0.0f;
        return;
    }
    
    std::string line;
    std::getline(file, line);
    file.close();
    
    std::istringstream ss(line);
    std::string cpu_label;
    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    
    ss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    
    uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
    uint64_t work_time = total - idle;
    
    if (last_cpu_total_ != 0) {
        uint64_t total_diff = total - last_cpu_total_;
        uint64_t work_diff = work_time - (last_cpu_total_ - last_cpu_idle_);
        
        if (total_diff > 0) {
            resources_.cpu_usage = (static_cast<float>(work_diff) / total_diff) * 100.0f;
        }
    }
    
    last_cpu_total_ = total;
    last_cpu_idle_ = idle;
}

void SystemMonitor::updateMemoryUsage() {
    std::ifstream file("/proc/meminfo");
    if (!file.is_open()) {
        resources_.memory_usage = 0.0f;
        return;
    }
    
    uint64_t total_mem = 0, free_mem = 0, available_mem = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string key;
        uint64_t value;
        
        ss >> key >> value;
        
        if (key == "MemTotal:") {
            total_mem = value;
        } else if (key == "MemFree:") {
            free_mem = value;
        } else if (key == "MemAvailable:") {
            available_mem = value;
            break;
        }
    }
    
    file.close();
    
    if (total_mem > 0) {
        uint64_t used_mem = total_mem - (available_mem > 0 ? available_mem : free_mem);
        resources_.memory_usage = (static_cast<float>(used_mem) / total_mem) * 100.0f;
    }
}

void SystemMonitor::updateTemperatures() {
    // CPU温度
    std::string cpu_temp = readSystemFile("/sys/class/thermal/thermal_zone0/temp");
    if (!cpu_temp.empty()) {
        try {
            float temp = std::stof(cpu_temp) / 1000.0f; // 转换为摄氏度
            resources_.cpu_temperature = temp;
        } catch (const std::exception&) {
            resources_.cpu_temperature = 0.0f;
        }
    }
    
    // GPU温度（Jetson设备）
    std::string gpu_temp = readSystemFile("/sys/devices/virtual/thermal/thermal_zone1/temp");
    if (!gpu_temp.empty()) {
        try {
            float temp = std::stof(gpu_temp) / 1000.0f;
            resources_.gpu_temperature = temp;
        } catch (const std::exception&) {
            resources_.gpu_temperature = 0.0f;
        }
    }
}

void SystemMonitor::updatePowerDraw() {
    // 尝试读取功耗信息（Jetson特定）
    std::string power = readSystemFile("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input");
    if (!power.empty()) {
        try {
            resources_.power_draw = std::stof(power) / 1000.0f; // 转换为瓦特
        } catch (const std::exception&) {
            resources_.power_draw = 0.0f;
        }
    }
}

std::string SystemMonitor::readSystemFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    
    std::string content;
    std::getline(file, content);
    file.close();
    
    return content;
}

} // namespace utils
} // namespace bamboo_cut