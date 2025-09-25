/**
 * @file logger.h
 * @brief C++ LVGL一体化系统日志记录器头文件
 * @version 5.0.0
 * @date 2024
 * 
 * 提供统一的日志记录接口，支持多级别日志输出
 */

#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <mutex>

namespace bamboo_cut {
namespace utils {

/**
 * @brief 日志级别枚举
 */
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

/**
 * @brief 线程安全的日志记录器类
 */
class Logger {
public:
    /**
     * @brief 获取全局日志器实例（单例模式）
     * @return 日志器实例的引用
     */
    static Logger& getInstance();

    /**
     * @brief 初始化日志器
     * @param log_file 日志文件路径
     * @param level 最小日志级别
     * @param enable_console 是否同时输出到控制台
     * @return 初始化是否成功
     */
    bool initialize(const std::string& log_file = "bamboo_system.log", 
                   LogLevel level = LogLevel::INFO,
                   bool enable_console = true);

    /**
     * @brief 记录日志
     * @param level 日志级别
     * @param message 日志消息
     * @param file 源文件名
     * @param line 源文件行号
     */
    void log(LogLevel level, const std::string& message, 
             const char* file = nullptr, int line = 0);

    /**
     * @brief 记录TRACE级别日志
     * @param message 日志消息
     */
    void trace(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 记录DEBUG级别日志
     * @param message 日志消息
     */
    void debug(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 记录INFO级别日志
     * @param message 日志消息
     */
    void info(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 记录WARN级别日志
     * @param message 日志消息
     */
    void warn(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 记录ERROR级别日志
     * @param message 日志消息
     */
    void error(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 记录FATAL级别日志
     * @param message 日志消息
     */
    void fatal(const std::string& message, const char* file = nullptr, int line = 0);

    /**
     * @brief 设置日志级别
     * @param level 新的日志级别
     */
    void setLevel(LogLevel level);

    /**
     * @brief 获取当前日志级别
     * @return 当前日志级别
     */
    LogLevel getLevel() const { return min_level_; }

    /**
     * @brief 刷新日志缓冲区
     */
    void flush();

private:
    Logger();
    ~Logger();

    // 禁用拷贝构造和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief 将日志级别转换为字符串
     * @param level 日志级别
     * @return 对应的字符串
     */
    std::string levelToString(LogLevel level) const;
    

    /**
     * @brief 获取当前时间戳字符串
     * @return 格式化的时间戳
     */
    std::string getCurrentTimestamp() const;

    mutable std::mutex mutex_;          ///< 线程安全互斥锁
    std::unique_ptr<std::ofstream> file_stream_;  ///< 文件输出流
    LogLevel min_level_;                ///< 最小日志级别
    bool enable_console_;               ///< 是否启用控制台输出
    bool initialized_;                  ///< 是否已初始化
};

} // namespace utils
} // namespace bamboo_cut

// 便利宏定义
#define LOG_TRACE(msg) bamboo_cut::utils::Logger::getInstance().trace(msg, __FILE__, __LINE__)
#define LOG_DEBUG(msg) bamboo_cut::utils::Logger::getInstance().debug(msg, __FILE__, __LINE__)
#define LOG_INFO(msg) bamboo_cut::utils::Logger::getInstance().info(msg, __FILE__, __LINE__)
#define LOG_WARN(msg) bamboo_cut::utils::Logger::getInstance().warn(msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) bamboo_cut::utils::Logger::getInstance().error(msg, __FILE__, __LINE__)
#define LOG_FATAL(msg) bamboo_cut::utils::Logger::getInstance().fatal(msg, __FILE__, __LINE__)