#ifndef BAMBOO_CUT_LOGGER_H
#define BAMBOO_CUT_LOGGER_H

#include <string>
#include <memory>
#include <sstream>

namespace bamboo_cut {
namespace core {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
public:
    static Logger& getInstance();
    
    void init(const std::string& log_file = "", LogLevel level = LogLevel::INFO);
    void setLevel(LogLevel level);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);
    
    template<typename... Args>
    void debug(const std::string& format, Args&&... args) {
        if (level_ <= LogLevel::DEBUG) {
            log(LogLevel::DEBUG, formatString(format, std::forward<Args>(args)...));
        }
    }
    
    template<typename... Args>
    void info(const std::string& format, Args&&... args) {
        if (level_ <= LogLevel::INFO) {
            log(LogLevel::INFO, formatString(format, std::forward<Args>(args)...));
        }
    }
    
    template<typename... Args>
    void warn(const std::string& format, Args&&... args) {
        if (level_ <= LogLevel::WARN) {
            log(LogLevel::WARN, formatString(format, std::forward<Args>(args)...));
        }
    }
    
    template<typename... Args>
    void error(const std::string& format, Args&&... args) {
        if (level_ <= LogLevel::ERROR) {
            log(LogLevel::ERROR, formatString(format, std::forward<Args>(args)...));
        }
    }
    
    template<typename... Args>
    void fatal(const std::string& format, Args&&... args) {
        log(LogLevel::FATAL, formatString(format, std::forward<Args>(args)...));
    }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void log(LogLevel level, const std::string& message);
    std::string getCurrentTime();
    std::string levelToString(LogLevel level);
    
    template<typename... Args>
    std::string formatString(const std::string& format, Args&&... args) {
        // 简单的格式化实现，生产环境可以使用fmt库
        std::ostringstream oss;
        formatHelper(oss, format, std::forward<Args>(args)...);
        return oss.str();
    }
    
    void formatHelper(std::ostringstream& oss, const std::string& format) {
        oss << format;
    }
    
    template<typename T, typename... Args>
    void formatHelper(std::ostringstream& oss, const std::string& format, T&& value, Args&&... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << value;
            formatHelper(oss, format.substr(pos + 2), std::forward<Args>(args)...);
        } else {
            oss << format;
        }
    }
    
    LogLevel level_ = LogLevel::INFO;
    std::string log_file_;
    bool initialized_ = false;
};

} // namespace core
} // namespace bamboo_cut

// 便捷的宏定义
#define LOG_DEBUG(...) bamboo_cut::core::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) bamboo_cut::core::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARN(...) bamboo_cut::core::Logger::getInstance().warn(__VA_ARGS__)
#define LOG_ERROR(...) bamboo_cut::core::Logger::getInstance().error(__VA_ARGS__)
#define LOG_FATAL(...) bamboo_cut::core::Logger::getInstance().fatal(__VA_ARGS__)

#endif // BAMBOO_CUT_LOGGER_H 