/**
 * @file system_monitor.h
 * @brief C++ LVGL一体化系统监控器
 * 提供系统资源监控和性能统计功能
 */

#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <array>

namespace bamboo_cut {
namespace utils {

/**
 * @brief 系统资源信息
 */
struct SystemResources {
    float cpu_usage;        // CPU使用率 (%)
    float memory_usage;     // 内存使用率 (%)
    float disk_usage;       // 磁盘使用率 (%)
    float cpu_temperature;  // CPU温度 (°C)
    float gpu_temperature;  // GPU温度 (°C)
    float power_draw;       // 功耗 (W)
    
    SystemResources() 
        : cpu_usage(0.0f), memory_usage(0.0f), disk_usage(0.0f)
        , cpu_temperature(0.0f), gpu_temperature(0.0f), power_draw(0.0f) {}
};

/**
 * @brief 网络统计信息
 */
struct NetworkStats {
    uint64_t bytes_sent;        // 发送字节数
    uint64_t bytes_received;    // 接收字节数
    uint64_t packets_sent;      // 发送包数
    uint64_t packets_received;  // 接收包数
    
    NetworkStats() 
        : bytes_sent(0), bytes_received(0)
        , packets_sent(0), packets_received(0) {}
};

/**
 * @brief 系统监控器类
 */
class SystemMonitor {
public:
    static SystemMonitor& getInstance();
    
    /**
     * @brief 初始化系统监控
     */
    bool initialize();
    
    /**
     * @brief 开始监控
     */
    void start();
    
    /**
     * @brief 停止监控
     */
    void stop();
    
    /**
     * @brief 获取系统资源信息
     */
    SystemResources getSystemResources() const;
    
    /**
     * @brief 获取网络统计信息
     */
    NetworkStats getNetworkStats() const;
    
    /**
     * @brief 获取系统运行时间
     */
    std::chrono::seconds getUptime() const;
    
    /**
     * @brief 获取系统负载
     */
    std::array<float, 3> getLoadAverage() const;
    
    /**
     * @brief 检查系统是否过热
     */
    bool isOverheating() const;
    
    /**
     * @brief 设置监控间隔
     */
    void setMonitorInterval(int interval_ms);

private:
    SystemMonitor();
    ~SystemMonitor();
    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;
    
    /**
     * @brief 更新CPU使用率
     */
    void updateCPUUsage();
    
    /**
     * @brief 更新内存使用率
     */
    void updateMemoryUsage();
    
    /**
     * @brief 更新温度信息
     */
    void updateTemperatures();
    
    /**
     * @brief 更新功耗信息
     */
    void updatePowerDraw();
    
    /**
     * @brief 读取系统文件
     */
    std::string readSystemFile(const std::string& path);
    
private:
    mutable std::mutex monitor_mutex_;
    SystemResources resources_;
    NetworkStats network_stats_;
    std::chrono::system_clock::time_point start_time_;
    
    std::atomic<bool> monitoring_{false};
    int monitor_interval_ms_{1000};  // 默认1秒间隔
    
    // 上次CPU时间统计
    uint64_t last_cpu_total_{0};
    uint64_t last_cpu_idle_{0};
    
    // 温度阈值
    static constexpr float CPU_TEMP_THRESHOLD = 80.0f;  // °C
    static constexpr float GPU_TEMP_THRESHOLD = 85.0f;  // °C
};

} // namespace utils
} // namespace bamboo_cut