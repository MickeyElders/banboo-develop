/**
 * @file jetson_monitor.cpp
 * @brief Jetson Orin Nano系统监控器实现
 * 通过tegrastats获取详细的系统状态信息
 */

#include "bamboo_cut/utils/jetson_monitor.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <cstdio>
#include <memory>
#include <fstream>

namespace bamboo_cut {
namespace utils {

JetsonMonitor::JetsonMonitor() {
    // 静默初始化，避免干扰推理日志
}

JetsonMonitor::~JetsonMonitor() {
    // 静默析构，避免干扰推理日志
    stop();
}

bool JetsonMonitor::start() {
    if (running_.load()) {
        // 静默处理重复启动，避免干扰推理日志
        return false;
    }
    
    // 静默启动系统监控线程
    should_stop_.store(false);
    running_.store(true);
    
    monitor_thread_ = std::thread(&JetsonMonitor::monitorLoop, this);
    
    return true;
}

void JetsonMonitor::stop() {
    if (!running_.load()) {
        return;
    }
    
    // 静默停止系统监控线程
    should_stop_.store(true);
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    running_.store(false);
}

SystemStats JetsonMonitor::getLatestStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return latest_stats_;
}

void JetsonMonitor::monitorLoop() {
    // 静默运行系统监控主循环，避免干扰推理日志
    
    while (!should_stop_.load()) {
        try {
            // 获取tegrastats输出
            std::string output = executeTegrastats();
            if (!output.empty()) {
                // 解析输出
                SystemStats stats = parseTegrastatsOutput(output);
                stats.timestamp = std::chrono::system_clock::now();
                
                // 更新最新状态
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    latest_stats_ = stats;
                }
            }
        } catch (const std::exception& e) {
            // 保留错误输出，但减少频率
            static int error_count = 0;
            if (++error_count % 10 == 1) { // 每10次错误只输出一次
                std::cerr << "[JetsonMonitor] 监控循环异常: " << e.what() << std::endl;
            }
        } catch (...) {
            // 保留严重未知异常的输出
            static int unknown_error_count = 0;
            if (++unknown_error_count % 10 == 1) { // 每10次错误只输出一次
                std::cerr << "[JetsonMonitor] 监控循环未知异常" << std::endl;
            }
        }
        
        // 等待下次监控
        std::this_thread::sleep_for(std::chrono::seconds(MONITOR_INTERVAL_SEC));
    }
    
    // 静默结束监控循环
}

std::string Jestd::string JetsonMonitor::readSysfsStats() {
    // 轻量回退：从 /proc/stat 和 /proc/meminfo 获取 CPU/Mem（GPU 置 0）
    try {
        auto read_cpu_line = []() {
            std::ifstream f("/proc/stat");
            std::string line;
            if (std::getline(f, line)) return line;
            return std::string();
        };
        auto parse_cpu = [](const std::string& l) {
            std::istringstream iss(l);
            std::string cpu;
            long user, nice, sys, idle, iowait, irq, softirq, steal, guest, guest_nice;
            iss >> cpu >> user >> nice >> sys >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
            long idle_all = idle + iowait;
            long non_idle = user + nice + sys + irq + softirq + steal;
            long total = idle_all + non_idle;
            return std::make_pair(total, idle_all);
        };

        auto l1 = read_cpu_line();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto l2 = read_cpu_line();
        int cpu_percent = 0;
        if (!l1.empty() && !l2.empty()) {
            auto pair1 = parse_cpu(l1);
            auto pair2 = parse_cpu(l2);
            long totald = pair2.first - pair1.first;
            long idled = pair2.second - pair1.second;
            if (totald > 0) cpu_percent = int((totald - idled) * 100 / totald);
        }

        std::ifstream mf("/proc/meminfo");
        long mem_total = 0, mem_avail = 0;
        if (mf) {
            std::string key, unit;
            long val;
            while (mf >> key >> val >> unit) {
                if (key == "MemTotal:") mem_total = val / 1024;
                else if (key == "MemAvailable:") mem_avail = val / 1024;
            }
        }
        long mem_used = mem_total - mem_avail;

        std::ostringstream oss;
        oss << "RAM " << mem_used << "/" << mem_total << "MB (lfb 0x0MB) "
            << "SWAP 0/0MB (cached 0MB) CPU [" << cpu_percent << "%@0] "
            << "GR3D_FREQ 0% EMC_FREQ 0%@0 VIC 0%@0";
        return oss.str();
    } catch (...) {
        return "";
    }
}

SystemStats JetsonMonitor::parseTegrastatsOutput(const std::string& output) {
    SystemStats stats;
    
    try {
        // 示例tegrastats输出：
        // RAM 3347/7467MB (lfb 896x4MB) SWAP 0/3733MB (cached 0MB) CPU [21%@1190,19%@1190,18%@1190,21%@1190,20%@1190,20%@1190] 
        // EMC_FREQ 13%@2133 GR3D_FREQ 61% APE 150 MTS fg 0% bg 0% NVDLA0 OFF NVDLA1 OFF PVA0 OFF VIC 0%@115 
        // MSENC OFF NVENC OFF NVJPG OFF NVDEC1 OFF NVDEC OFF OFA OFF VDD_IN 5336/4970 VDD_CPU_GPU_CV 2617/2336 VDD_SOC 1478/1318 
        // CV0@-256C GPU@49.2C PMIC@50.0C AO@48.5C thermal@50.187C POM_5V_IN 5336/4970 POM_5V_GPU 1040/936 POM_5V_CPU 847/758

        // 静默解析tegrastats输出，避免干扰推理日志
        // DEBUG: std::cout << "[JetsonMonitor] 解析tegrastats输出: " << output.substr(0, 100) << "..." << std::endl;

        // 解析RAM信息
        std::regex ram_regex(R"(RAM\s+(\d+)/(\d+)MB\s+\(lfb\s+(\d+)x(\d+)MB\))");
        std::smatch ram_match;
        if (std::regex_search(output, ram_match, ram_regex)) {
            stats.memory.ram_used_mb = std::stoi(ram_match[1]);
            stats.memory.ram_total_mb = std::stoi(ram_match[2]);
            stats.memory.lfb_blocks = std::stoi(ram_match[3]);
            stats.memory.lfb_size_mb = std::stoi(ram_match[4]);
        }

        // 解析SWAP信息
        std::regex swap_regex(R"(SWAP\s+(\d+)/(\d+)MB\s+\(cached\s+(\d+)MB\))");
        std::smatch swap_match;
        if (std::regex_search(output, swap_match, swap_regex)) {
            stats.memory.swap_used_mb = std::stoi(swap_match[1]);
            stats.memory.swap_total_mb = std::stoi(swap_match[2]);
            stats.memory.swap_cached_mb = std::stoi(swap_match[3]);
        }

        // 解析CPU信息
        std::regex cpu_regex(R"(CPU\s+\[([^\]]+)\])");
        std::smatch cpu_match;
        if (std::regex_search(output, cpu_match, cpu_regex)) {
            stats.cpu_cores = parseCPUInfo(cpu_match[1]);
        }

        // 解析GPU信息
        std::regex gr3d_regex(R"(GR3D_FREQ\s+(\d+)%)");
        std::regex gr3d_detail_regex(R"(GR3D\s+(\d+)%@(\d+))");
        std::smatch gr3d_match, gr3d_detail_match;
        
        std::string gpu_str, gr3d_str;
        if (std::regex_search(output, gr3d_match, gr3d_regex)) {
            gr3d_str = gr3d_match[0];
        }
        if (std::regex_search(output, gr3d_detail_match, gr3d_detail_regex)) {
            gpu_str = gr3d_detail_match[0];
        }
        stats.gpu = parseGPUInfo(gpu_str, gr3d_str);

        // 解析EMC信息
        std::regex emc_regex(R"(EMC_FREQ\s+(\d+)%@(\d+))");
        std::smatch emc_match;
        if (std::regex_search(output, emc_match, emc_regex)) {
            stats.other.emc_freq_percent = std::stoi(emc_match[1]);
            stats.other.emc_freq_mhz = std::stoi(emc_match[2]);
        }

        // 解析VIC信息
        std::regex vic_regex(R"(VIC\s+(\d+)%@(\d+))");
        std::smatch vic_match;
        if (std::regex_search(output, vic_match, vic_regex)) {
            stats.other.vic_usage_percent = std::stoi(vic_match[1]);
            stats.other.vic_freq_mhz = std::stoi(vic_match[2]);
        }

        // 解析APE信息
        std::regex ape_regex(R"(APE\s+(\d+))");
        std::smatch ape_match;
        if (std::regex_search(output, ape_match, ape_regex)) {
            stats.other.ape_freq_mhz = std::stoi(ape_match[1]);
        }

        // 解析温度信息
        std::regex temp_regex(R"((CPU@[\d\.]+C|GPU@[\d\.]+C|thermal@[\d\.]+C|CV\d@[\d\.-]+C|AO@[\d\.]+C))");
        std::sregex_iterator temp_iter(output.begin(), output.end(), temp_regex);
        std::sregex_iterator temp_end;
        
        for (; temp_iter != temp_end; ++temp_iter) {
            std::string temp_str = temp_iter->str();
            if (temp_str.find("CPU@") != std::string::npos) {
                stats.temperature.cpu_temp = extractFloat(temp_str, R"((\d+\.?\d*)C)");
            } else if (temp_str.find("GPU@") != std::string::npos) {
                stats.temperature.gpu_temp = extractFloat(temp_str, R"((\d+\.?\d*)C)");
            } else if (temp_str.find("thermal@") != std::string::npos) {
                stats.temperature.thermal_temp = extractFloat(temp_str, R"((\d+\.?\d*)C)");
            } else if (temp_str.find("CV") != std::string::npos) {
                stats.temperature.cv_temp = extractFloat(temp_str, R"((\-?\d+\.?\d*)C)");
            } else if (temp_str.find("AO@") != std::string::npos) {
                stats.temperature.soc_temp = extractFloat(temp_str, R"((\d+\.?\d*)C)");
            }
        }

        // 解析电源信息
        std::regex vdd_in_regex(R"(VDD_IN\s+(\d+)/(\d+))");
        std::regex vdd_cpu_regex(R"(VDD_CPU_GPU_CV\s+(\d+)/(\d+))");
        std::regex vdd_soc_regex(R"(VDD_SOC\s+(\d+)/(\d+))");
        
        std::smatch vdd_match;
        if (std::regex_search(output, vdd_match, vdd_in_regex)) {
            stats.power.vdd_in_current_ma = std::stoi(vdd_match[1]);
            stats.power.vdd_in_power_mw = std::stoi(vdd_match[2]);
        }
        if (std::regex_search(output, vdd_match, vdd_cpu_regex)) {
            stats.power.vdd_cpu_gpu_cv_current_ma = std::stoi(vdd_match[1]);
            stats.power.vdd_cpu_gpu_cv_power_mw = std::stoi(vdd_match[2]);
        }
        if (std::regex_search(output, vdd_match, vdd_soc_regex)) {
            stats.power.vdd_soc_current_ma = std::stoi(vdd_match[1]);
            stats.power.vdd_soc_power_mw = std::stoi(vdd_match[2]);
        }

        // 风扇信息（如果有）
        std::regex fan_regex(R"(FAN\s+(\d+)RPM)");
        std::smatch fan_match;
        if (std::regex_search(output, fan_match, fan_regex)) {
            stats.other.fan_rpm = std::stoi(fan_match[1]);
        }

    } catch (const std::exception& e) {
        // 静默处理解析异常，避免干扰推理日志
    }
    
    return stats;
}

std::vector<CPUCore> JetsonMonitor::parseCPUInfo(const std::string& cpu_str) {
    std::vector<CPUCore> cores;
    
    try {
        // CPU格式: 21%@1190,19%@1190,18%@1190,21%@1190,20%@1190,20%@1190
        std::regex core_regex(R"((\d+)%@(\d+))");
        std::sregex_iterator core_iter(cpu_str.begin(), cpu_str.end(), core_regex);
        std::sregex_iterator core_end;
        
        for (; core_iter != core_end; ++core_iter) {
            CPUCore core;
            core.usage_percent = std::stoi(core_iter->str(1));
            core.frequency_mhz = std::stoi(core_iter->str(2));
            cores.push_back(core);
        }
    } catch (const std::exception& e) {
        std::cerr << "[JetsonMonitor] 解析CPU信息异常: " << e.what() << std::endl;
    }
    
    return cores;
}

GPUInfo JetsonMonitor::parseGPUInfo(const std::string& gpu_str, const std::string& gr3d_str) {
    GPUInfo gpu;
    
    try {
        // GR3D 61%@624
        if (!gpu_str.empty()) {
            std::regex gpu_regex(R"((\d+)%@(\d+))");
            std::smatch gpu_match;
            if (std::regex_search(gpu_str, gpu_match, gpu_regex)) {
                gpu.usage_percent = std::stoi(gpu_match[1]);
                gpu.frequency_mhz = std::stoi(gpu_match[2]);
            }
        }
        
        // GR3D_FREQ 61%
        if (!gr3d_str.empty()) {
            std::regex gr3d_regex(R"((\d+)%)");
            std::smatch gr3d_match;
            if (std::regex_search(gr3d_str, gr3d_match, gr3d_regex)) {
                gpu.gr3d_freq_percent = std::stoi(gr3d_match[1]);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[JetsonMonitor] 解析GPU信息异常: " << e.what() << std::endl;
    }
    
    return gpu;
}

MemoryInfo JetsonMonitor::parseMemoryInfo(const std::string& ram_str, const std::string& swap_str) {
    // 这个函数在主解析函数中已经实现，这里保留空实现以保持接口完整性
    return MemoryInfo{};
}

TemperatureInfo JetsonMonitor::parseTemperatureInfo(const std::string& temp_str) {
    // 这个函数在主解析函数中已经实现，这里保留空实现以保持接口完整性
    return TemperatureInfo{};
}

PowerInfo JetsonMonitor::parsePowerInfo(const std::string& power_str) {
    // 这个函数在主解析函数中已经实现，这里保留空实现以保持接口完整性
    return PowerInfo{};
}

OtherInfo JetsonMonitor::parseOtherInfo(const std::string& emc_str, const std::string& vic_str, 
                                      const std::string& ape_str, const std::string& fan_str) {
    // 这个函数在主解析函数中已经实现，这里保留空实现以保持接口完整性
    return OtherInfo{};
}

int JetsonMonitor::extractInt(const std::string& str, const std::string& pattern) {
    try {
        std::regex regex(pattern);
        std::smatch match;
        if (std::regex_search(str, match, regex)) {
            return std::stoi(match[1]);
        }
    } catch (const std::exception& e) {
        std::cerr << "[JetsonMonitor] 提取整数异常: " << e.what() << std::endl;
    }
    return 0;
}

float JetsonMonitor::extractFloat(const std::string& str, const std::string& pattern) {
    try {
        std::regex regex(pattern);
        std::smatch match;
        if (std::regex_search(str, match, regex)) {
            return std::stof(match[1]);
        }
    } catch (const std::exception& e) {
        std::cerr << "[JetsonMonitor] 提取浮点数异常: " << e.what() << std::endl;
    }
    return 0.0f;
}

} // namespace utils
} // namespace bamboo_cut
