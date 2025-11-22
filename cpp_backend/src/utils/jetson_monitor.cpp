/**
 * @file jetson_monitor.cpp
 * @brief Jetson Orin Nano绯荤粺鐩戞帶鍣ㄥ疄鐜?
 * 閫氳繃tegrastats鑾峰彇璇︾粏鐨勭郴缁熺姸鎬佷俊鎭?
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
    // 闈欓粯鍒濆鍖栵紝閬垮厤骞叉壈鎺ㄧ悊鏃ュ織
}

JetsonMonitor::~JetsonMonitor() {
    // 闈欓粯鏋愭瀯锛岄伩鍏嶅共鎵版帹鐞嗘棩蹇?
    stop();
}

bool JetsonMonitor::start() {
    if (running_.load()) {
        // 闈欓粯澶勭悊閲嶅鍚姩锛岄伩鍏嶅共鎵版帹鐞嗘棩蹇?
        return false;
    }
    
    // 闈欓粯鍚姩绯荤粺鐩戞帶绾跨▼
    should_stop_.store(false);
    running_.store(true);
    
    monitor_thread_ = std::thread(&JetsonMonitor::monitorLoop, this);
    
    return true;
}

void JetsonMonitor::stop() {
    if (!running_.load()) {
        return;
    }
    
    // 闈欓粯鍋滄绯荤粺鐩戞帶绾跨▼
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
    // 闈欓粯杩愯绯荤粺鐩戞帶涓诲惊鐜紝閬垮厤骞叉壈鎺ㄧ悊鏃ュ織
    
    while (!should_stop_.load()) {
        try {
            // 鑾峰彇tegrastats杈撳嚭
            std::string output = executeTegrastats();
            if (!output.empty()) {
                // 瑙ｆ瀽杈撳嚭
                SystemStats stats = parseTegrastatsOutput(output);
                stats.timestamp = std::chrono::system_clock::now();
                
                // 鏇存柊鏈€鏂扮姸鎬?
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    latest_stats_ = stats;
                }
            }
        } catch (const std::exception& e) {
            // 淇濈暀閿欒杈撳嚭锛屼絾鍑忓皯棰戠巼
            static int error_count = 0;
            if (++error_count % 10 == 1) { // 姣?0娆￠敊璇彧杈撳嚭涓€娆?
                std::cerr << "[JetsonMonitor] 鐩戞帶寰幆寮傚父: " << e.what() << std::endl;
            }
        } catch (...) {
            // 淇濈暀涓ラ噸鏈煡寮傚父鐨勮緭鍑?
            static int unknown_error_count = 0;
            if (++unknown_error_count % 10 == 1) { // 姣?0娆￠敊璇彧杈撳嚭涓€娆?
                std::cerr << "[JetsonMonitor] 鐩戞帶寰幆鏈煡寮傚父" << std::endl;
            }
        }
        
        // 绛夊緟涓嬫鐩戞帶
        std::this_thread::sleep_for(std::chrono::seconds(MONITOR_INTERVAL_SEC));
    }
    
    // 闈欓粯缁撴潫鐩戞帶寰幆
}

std::string JetsonMonitor::executeTegrastats() {
    try {
        // 浼樺厛浣跨敤 tegrastats锛屽け璐ユ垨鏃犺緭鍑哄垯鍥為€€鍒?sysfs 缁熻
        std::unique_ptr<FILE, decltype(&pclose)> pipe(
            popen("timeout 3s tegrastats --interval 1000", "r"), pclose);
        if (!pipe) {
            return readSysfsStats();
        }

        char buffer[1024] = {0};
        if (fgets(buffer, sizeof(buffer), pipe.get()) == nullptr) {
            return readSysfsStats();
        }

        std::string result(buffer);
        if (result.empty()) {
            return readSysfsStats();
        }
        return result;
    } catch (...) {
        return readSysfsStats();
    }
}

std::string JetsonMonitor::readSysfsStats() {
    // 轻量回退：从 /proc/stat 与 /proc/meminfo 获取 CPU/Mem，GPU 填充 0
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
        // 绀轰緥tegrastats杈撳嚭锛?
        // RAM 3347/7467MB (lfb 896x4MB) SWAP 0/3733MB (cached 0MB) CPU [21%@1190,19%@1190,18%@1190,21%@1190,20%@1190,20%@1190] 
        // EMC_FREQ 13%@2133 GR3D_FREQ 61% APE 150 MTS fg 0% bg 0% NVDLA0 OFF NVDLA1 OFF PVA0 OFF VIC 0%@115 
        // MSENC OFF NVENC OFF NVJPG OFF NVDEC1 OFF NVDEC OFF OFA OFF VDD_IN 5336/4970 VDD_CPU_GPU_CV 2617/2336 VDD_SOC 1478/1318 
        // CV0@-256C GPU@49.2C PMIC@50.0C AO@48.5C thermal@50.187C POM_5V_IN 5336/4970 POM_5V_GPU 1040/936 POM_5V_CPU 847/758

        // 闈欓粯瑙ｆ瀽tegrastats杈撳嚭锛岄伩鍏嶅共鎵版帹鐞嗘棩蹇?
        // DEBUG: std::cout << "[JetsonMonitor] 瑙ｆ瀽tegrastats杈撳嚭: " << output.substr(0, 100) << "..." << std::endl;

        // 瑙ｆ瀽RAM淇℃伅
        std::regex ram_regex(R"(RAM\s+(\d+)/(\d+)MB\s+\(lfb\s+(\d+)x(\d+)MB\))");
        std::smatch ram_match;
        if (std::regex_search(output, ram_match, ram_regex)) {
            stats.memory.ram_used_mb = std::stoi(ram_match[1]);
            stats.memory.ram_total_mb = std::stoi(ram_match[2]);
            stats.memory.lfb_blocks = std::stoi(ram_match[3]);
            stats.memory.lfb_size_mb = std::stoi(ram_match[4]);
        }

        // 瑙ｆ瀽SWAP淇℃伅
        std::regex swap_regex(R"(SWAP\s+(\d+)/(\d+)MB\s+\(cached\s+(\d+)MB\))");
        std::smatch swap_match;
        if (std::regex_search(output, swap_match, swap_regex)) {
            stats.memory.swap_used_mb = std::stoi(swap_match[1]);
            stats.memory.swap_total_mb = std::stoi(swap_match[2]);
            stats.memory.swap_cached_mb = std::stoi(swap_match[3]);
        }

        // 瑙ｆ瀽CPU淇℃伅
        std::regex cpu_regex(R"(CPU\s+\[([^\]]+)\])");
        std::smatch cpu_match;
        if (std::regex_search(output, cpu_match, cpu_regex)) {
            stats.cpu_cores = parseCPUInfo(cpu_match[1]);
        }

        // 瑙ｆ瀽GPU淇℃伅
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

        // 瑙ｆ瀽EMC淇℃伅
        std::regex emc_regex(R"(EMC_FREQ\s+(\d+)%@(\d+))");
        std::smatch emc_match;
        if (std::regex_search(output, emc_match, emc_regex)) {
            stats.other.emc_freq_percent = std::stoi(emc_match[1]);
            stats.other.emc_freq_mhz = std::stoi(emc_match[2]);
        }

        // 瑙ｆ瀽VIC淇℃伅
        std::regex vic_regex(R"(VIC\s+(\d+)%@(\d+))");
        std::smatch vic_match;
        if (std::regex_search(output, vic_match, vic_regex)) {
            stats.other.vic_usage_percent = std::stoi(vic_match[1]);
            stats.other.vic_freq_mhz = std::stoi(vic_match[2]);
        }

        // 瑙ｆ瀽APE淇℃伅
        std::regex ape_regex(R"(APE\s+(\d+))");
        std::smatch ape_match;
        if (std::regex_search(output, ape_match, ape_regex)) {
            stats.other.ape_freq_mhz = std::stoi(ape_match[1]);
        }

        // 瑙ｆ瀽娓╁害淇℃伅
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

        // 瑙ｆ瀽鐢垫簮淇℃伅
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

        // 椋庢墖淇℃伅锛堝鏋滄湁锛?
        std::regex fan_regex(R"(FAN\s+(\d+)RPM)");
        std::smatch fan_match;
        if (std::regex_search(output, fan_match, fan_regex)) {
            stats.other.fan_rpm = std::stoi(fan_match[1]);
        }

    } catch (const std::exception& e) {
        // 闈欓粯澶勭悊瑙ｆ瀽寮傚父锛岄伩鍏嶅共鎵版帹鐞嗘棩蹇?
    }
    
    return stats;
}

std::vector<CPUCore> JetsonMonitor::parseCPUInfo(const std::string& cpu_str) {
    std::vector<CPUCore> cores;
    
    try {
        // CPU鏍煎紡: 21%@1190,19%@1190,18%@1190,21%@1190,20%@1190,20%@1190
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
        std::cerr << "[JetsonMonitor] 瑙ｆ瀽CPU淇℃伅寮傚父: " << e.what() << std::endl;
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
        std::cerr << "[JetsonMonitor] 瑙ｆ瀽GPU淇℃伅寮傚父: " << e.what() << std::endl;
    }
    
    return gpu;
}

MemoryInfo JetsonMonitor::parseMemoryInfo(const std::string& ram_str, const std::string& swap_str) {
    // 杩欎釜鍑芥暟鍦ㄤ富瑙ｆ瀽鍑芥暟涓凡缁忓疄鐜帮紝杩欓噷淇濈暀绌哄疄鐜颁互淇濇寔鎺ュ彛瀹屾暣鎬?
    return MemoryInfo{};
}

TemperatureInfo JetsonMonitor::parseTemperatureInfo(const std::string& temp_str) {
    // 杩欎釜鍑芥暟鍦ㄤ富瑙ｆ瀽鍑芥暟涓凡缁忓疄鐜帮紝杩欓噷淇濈暀绌哄疄鐜颁互淇濇寔鎺ュ彛瀹屾暣鎬?
    return TemperatureInfo{};
}

PowerInfo JetsonMonitor::parsePowerInfo(const std::string& power_str) {
    // 杩欎釜鍑芥暟鍦ㄤ富瑙ｆ瀽鍑芥暟涓凡缁忓疄鐜帮紝杩欓噷淇濈暀绌哄疄鐜颁互淇濇寔鎺ュ彛瀹屾暣鎬?
    return PowerInfo{};
}

OtherInfo JetsonMonitor::parseOtherInfo(const std::string& emc_str, const std::string& vic_str, 
                                      const std::string& ape_str, const std::string& fan_str) {
    // 杩欎釜鍑芥暟鍦ㄤ富瑙ｆ瀽鍑芥暟涓凡缁忓疄鐜帮紝杩欓噷淇濈暀绌哄疄鐜颁互淇濇寔鎺ュ彛瀹屾暣鎬?
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
        std::cerr << "[JetsonMonitor] 鎻愬彇鏁存暟寮傚父: " << e.what() << std::endl;
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
        std::cerr << "[JetsonMonitor] 鎻愬彇娴偣鏁板紓甯? " << e.what() << std::endl;
    }
    return 0.0f;
}

} // namespace utils
} // namespace bamboo_cut

