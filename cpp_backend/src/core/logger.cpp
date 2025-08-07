#include "bamboo_cut/core/logger.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace bamboo_cut {
namespace core {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::init(const std::string& log_file, LogLevel level) {
    log_file_ = log_file;
    level_ = level;
    initialized_ = true;
}

void Logger::setLevel(LogLevel level) {
    level_ = level;
}

void Logger::debug(const std::string& message) {
    if (level_ <= LogLevel::DEBUG) {
        log(LogLevel::DEBUG, message);
    }
}

void Logger::info(const std::string& message) {
    if (level_ <= LogLevel::INFO) {
        log(LogLevel::INFO, message);
    }
}

void Logger::warn(const std::string& message) {
    if (level_ <= LogLevel::WARN) {
        log(LogLevel::WARN, message);
    }
}

void Logger::error(const std::string& message) {
    if (level_ <= LogLevel::ERROR) {
        log(LogLevel::ERROR, message);
    }
}

void Logger::fatal(const std::string& message) {
    log(LogLevel::FATAL, message);
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!initialized_) {
        // 如果没有初始化，使用默认设置
        init("", LogLevel::INFO);
    }
    
    std::string timestamp = getCurrentTime();
    std::string level_str = levelToString(level);
    std::string log_message = timestamp + " [" + level_str + "] " + message + "\n";
    
    // 输出到控制台
    std::cout << log_message;
    
    // 如果指定了日志文件，也写入文件
    if (!log_file_.empty()) {
        std::ofstream log_stream(log_file_, std::ios::app);
        if (log_stream.is_open()) {
            log_stream << log_message;
            log_stream.close();
        }
    }
}

std::string Logger::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

} // namespace core
} // namespace bamboo_cut 