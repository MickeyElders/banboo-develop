/**
 * @file logger.cpp
 * @brief C++ LVGL一体化系统日志记录器实现
 * @version 5.0.0
 * @date 2024
 * 
 * 提供统一的日志记录接口，支持多级别日志输出
 */

#include "bamboo_cut/utils/logger.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace bamboo_cut {
namespace utils {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : min_level_(LogLevel::INFO), enable_console_(true), initialized_(false) {
    // 构造函数实现
}

Logger::~Logger() {
    if (file_stream_) {
        file_stream_->close();
    }
}

bool Logger::initialize(const std::string& log_file, LogLevel level, bool enable_console) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    min_level_ = level;
    enable_console_ = enable_console;
    
    if (!log_file.empty()) {
        file_stream_ = std::make_unique<std::ofstream>(log_file, std::ios::app);
        if (!file_stream_->is_open()) {
            return false;
        }
    }
    
    initialized_ = true;
    return true;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level < min_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string timestamp = getCurrentTimestamp();
    std::string level_str = levelToString(level);
    
    std::ostringstream log_stream;
    log_stream << "[" << timestamp << "] [" << level_str << "] " << message;
    
    if (file && line > 0) {
        log_stream << " (" << file << ":" << line << ")";
    }
    
    std::string log_line = log_stream.str();
    
    // 输出到控制台
    if (enable_console_) {
        std::cout << log_line << std::endl;
    }
    
    // 输出到文件
    if (file_stream_ && file_stream_->is_open()) {
        *file_stream_ << log_line << std::endl;
        file_stream_->flush();
    }
}

void Logger::trace(const std::string& message, const char* file, int line) {
    log(LogLevel::TRACE, message, file, line);
}

void Logger::debug(const std::string& message, const char* file, int line) {
    log(LogLevel::DEBUG, message, file, line);
}

void Logger::info(const std::string& message, const char* file, int line) {
    log(LogLevel::INFO, message, file, line);
}

void Logger::warn(const std::string& message, const char* file, int line) {
    log(LogLevel::WARN, message, file, line);
}

void Logger::error(const std::string& message, const char* file, int line) {
    log(LogLevel::ERROR, message, file, line);
}

void Logger::fatal(const std::string& message, const char* file, int line) {
    log(LogLevel::FATAL, message, file, line);
}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_level_ = level;
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->flush();
    }
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

} // namespace utils
} // namespace bamboo_cut