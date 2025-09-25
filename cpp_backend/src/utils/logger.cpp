/**
 * @file logger.cpp
 * @brief C++ LVGL一体化系统日志管理器实现
 * @version 5.0.0
 * @date 2024
 */

#include "bamboo_cut/utils/logger.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace bamboo_cut {
namespace utils {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : log_level_(LogLevel::INFO) {
}

void Logger::setLogLevel(LogLevel level) {
    log_level_ = level;
}

void Logger::setLogFile(const std::string& filename) {
    log_file_ = filename;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < log_level_) {
        return;
    }
    
    std::string timestamp = getCurrentTimestamp();
    std::string level_str = logLevelToString(level);
    std::string formatted_message = "[" + timestamp + "] [" + level_str + "] " + message;
    
    // 输出到控制台
    std::cout << formatted_message << std::endl;
    
    // 输出到文件
    if (!log_file_.empty()) {
        std::ofstream file(log_file_, std::ios::app);
        if (file.is_open()) {
            file << formatted_message << std::endl;
        }
    }
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

} // namespace utils
} // namespace bamboo_cut