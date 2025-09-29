/**
 * @file jetson_monitor.h
 * @brief Jetson Orin Nano系统监控器
 * 通过tegrastats获取详细的系统状态信息
 */

#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

namespace bamboo_cut {
namespace utils {

/**
 * @brief CPU核心信息
 */
struct CPUCore {
    int usage_percent = 0;
    int frequency_mhz = 0;
};

/**
 * @brief GPU信息
 */
struct GPUInfo {
    int usage_percent = 0;
    int frequency_mhz = 0;
    int gr3d_freq_percent = 0;
};

/**
 * @brief 内存信息
 */
struct MemoryInfo {
    int ram_used_mb = 0;
    int ram_total_mb = 0;
    int lfb_blocks = 0;
    int lfb_size_mb = 0;
    int swap_used_mb = 0;
    int swap_total_mb = 0;
    int swap_cached_mb = 0;
};

/**
 * @brief 温度传感器信息
 */
struct TemperatureInfo {
    float cpu_temp = 0.0f;
    float gpu_temp = 0.0f;
    float thermal_temp = 0.0f;
    float cv_temp = 0.0f;
    float soc_temp = 0.0f;
};

/**
 * @brief 电源信息
 */
struct PowerInfo {
    int vdd_in_current_ma = 0;
    int vdd_in_power_mw = 0;
    int vdd_cpu_gpu_cv_current_ma = 0;
    int vdd_cpu_gpu_cv_power_mw = 0;
    int vdd_soc_current_ma = 0;
    int vdd_soc_power_mw = 0;
};

/**
 * @brief 其他系统信息
 */
struct OtherInfo {
    int emc_freq_percent = 0;
    int emc_freq_mhz = 0;
    int vic_usage_percent = 0;
    int vic_freq_mhz = 0;
    int ape_freq_mhz = 0;
    int fan_rpm = 0;
};

/**
 * @brief 完整的系统状态
 */
struct SystemStats {
    std::vector<CPUCore> cpu_cores;
    GPUInfo gpu;
    MemoryInfo memory;
    TemperatureInfo temperature;
    PowerInfo power;
    OtherInfo other;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Jetson系统监控器
 */
class JetsonMonitor {
public:
    JetsonMonitor();
    ~JetsonMonitor();
    
    /**
     * @brief 启动监控
     */
    bool start();
    
    /**
     * @brief 停止监控
     */
    void stop();
    
    /**
     * @brief 获取最新的系统状态
     */
    SystemStats getLatestStats() const;
    
    /**
     * @brief 检查是否在运行
     */
    bool isRunning() const { return running_.load(); }

private:
    /**
     * @brief 监控线程主循环
     */
    void monitorLoop();
    
    /**
     * @brief 执行tegrastats命令并获取输出
     */
    std::string executeTegrastats();
    
    /**
     * @brief 解析tegrastats输出
     */
    SystemStats parseTegrastatsOutput(const std::string& output);
    
    /**
     * @brief 解析CPU信息
     */
    std::vector<CPUCore> parseCPUInfo(const std::string& cpu_str);
    
    /**
     * @brief 解析GPU信息
     */
    GPUInfo parseGPUInfo(const std::string& gpu_str, const std::string& gr3d_str);
    
    /**
     * @brief 解析内存信息
     */
    MemoryInfo parseMemoryInfo(const std::string& ram_str, const std::string& swap_str);
    
    /**
     * @brief 解析温度信息
     */
    TemperatureInfo parseTemperatureInfo(const std::string& temp_str);
    
    /**
     * @brief 解析电源信息
     */
    PowerInfo parsePowerInfo(const std::string& power_str);
    
    /**
     * @brief 解析其他信息
     */
    OtherInfo parseOtherInfo(const std::string& emc_str, const std::string& vic_str, 
                           const std::string& ape_str, const std::string& fan_str);
    
    /**
     * @brief 使用正则表达式提取数值
     */
    int extractInt(const std::string& str, const std::string& pattern);
    float extractFloat(const std::string& str, const std::string& pattern);

private:
    std::thread monitor_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    mutable std::mutex stats_mutex_;
    SystemStats latest_stats_;
    
    // 监控间隔（秒）
    static constexpr int MONITOR_INTERVAL_SEC = 2;
};

} // namespace utils
} // namespace bamboo_cut