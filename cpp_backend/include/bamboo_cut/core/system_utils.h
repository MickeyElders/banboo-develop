#ifndef BAMBOO_CUT_SYSTEM_UTILS_H
#define BAMBOO_CUT_SYSTEM_UTILS_H

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <new>

namespace bamboo_cut {
namespace core {

/**
 * @brief 安全的系统工具类
 * 
 * 提供安全的系统命令执行、文件操作等功能，
 * 替代不安全的 std::system 等函数
 */
class SystemUtils {
public:
    /**
     * @brief 命令执行结果
     */
    struct CommandResult {
        int exit_code = -1;
        std::string output;
        std::string error;
        bool success = false;
    };

    /**
     * @brief 安全执行系统命令
     * @param command 要执行的命令
     * @param timeout_ms 超时时间（毫秒），0表示无超时
     * @return 命令执行结果
     */
    static CommandResult executeCommand(const std::string& command, 
                                      int timeout_ms = 5000);

    /**
     * @brief 检查命令是否存在
     * @param command 命令名称
     * @return 是否存在
     */
    static bool commandExists(const std::string& command);

    /**
     * @brief 检查文件是否存在
     * @param path 文件路径
     * @return 是否存在
     */
    static bool fileExists(const std::string& path);

    /**
     * @brief 检查目录是否存在
     * @param path 目录路径
     * @return 是否存在
     */
    static bool directoryExists(const std::string& path);

    /**
     * @brief 安全读取文件内容
     * @param path 文件路径
     * @param max_size 最大读取大小（字节）
     * @return 文件内容，失败则返回空
     */
    static std::optional<std::string> readFile(const std::string& path, 
                                             size_t max_size = 1024 * 1024);

    /**
     * @brief 安全的字符串转整数
     * @param str 字符串
     * @param default_value 转换失败时的默认值
     * @return 转换结果
     */
    static int safeStringToInt(const std::string& str, int default_value = 0);

    /**
     * @brief 安全的字符串转浮点数
     * @param str 字符串
     * @param default_value 转换失败时的默认值
     * @return 转换结果
     */
    static double safeStringToDouble(const std::string& str, double default_value = 0.0);

    /**
     * @brief 检查模块是否已加载
     * @param module_name 模块名称
     * @return 是否已加载
     */
    static bool isModuleLoaded(const std::string& module_name);

    /**
     * @brief 获取环境变量
     * @param name 环境变量名
     * @param default_value 默认值
     * @return 环境变量值
     */
    static std::string getEnvironmentVariable(const std::string& name, 
                                            const std::string& default_value = "");

    /**
     * @brief 创建目录（递归）
     * @param path 目录路径
     * @return 是否成功
     */
    static bool createDirectory(const std::string& path);

    /**
     * @brief 获取可执行文件路径
     * @param command 命令名称
     * @return 完整路径，未找到则返回空
     */
    static std::optional<std::string> getExecutablePath(const std::string& command);

private:
    // 私有构造函数，工具类不允许实例化
    SystemUtils() = default;
    
    // 禁用拷贝和赋值
    SystemUtils(const SystemUtils&) = delete;
    SystemUtils& operator=(const SystemUtils&) = delete;
    
    // 内部工具方法
    static std::vector<std::string> tokenizeCommand(const std::string& command);
    static bool isCommandSafe(const std::string& command);
};

/**
 * @brief 安全的内存管理辅助类
 */
template<typename T>
class SafeArray {
public:
    explicit SafeArray(size_t size) : size_(size), data_(nullptr) {
        if (size > 0 && size <= MAX_SAFE_SIZE) {
            data_ = new(std::nothrow) T[size];
        }
    }
    
    ~SafeArray() {
        delete[] data_;
    }
    
    // 禁用拷贝
    SafeArray(const SafeArray&) = delete;
    SafeArray& operator=(const SafeArray&) = delete;
    
    // 移动构造
    SafeArray(SafeArray&& other) noexcept 
        : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }
    
    T* get() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    
    T& operator[](size_t index) {
        if (index >= size_ || !data_) {
            throw std::out_of_range("SafeArray index out of range");
        }
        return data_[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= size_ || !data_) {
            throw std::out_of_range("SafeArray index out of range");
        }
        return data_[index];
    }

private:
    static constexpr size_t MAX_SAFE_SIZE = 1024 * 1024 * 100; // 100MB限制
    size_t size_;
    T* data_;
};

/**
 * @brief 范围检查的向量访问器
 */
template<typename T>
class SafeVectorAccess {
public:
    explicit SafeVectorAccess(std::vector<T>& vec) : vec_(vec) {}
    
    T& at(size_t index) {
        if (index >= vec_.size()) {
            throw std::out_of_range("SafeVectorAccess index out of range");
        }
        return vec_[index];
    }
    
    const T& at(size_t index) const {
        if (index >= vec_.size()) {
            throw std::out_of_range("SafeVectorAccess index out of range");
        }
        return vec_[index];
    }
    
    T& front() {
        if (vec_.empty()) {
            throw std::runtime_error("SafeVectorAccess: empty vector");
        }
        return vec_.front();
    }
    
    T& back() {
        if (vec_.empty()) {
            throw std::runtime_error("SafeVectorAccess: empty vector");
        }
        return vec_.back();
    }
    
    size_t size() const { return vec_.size(); }
    bool empty() const { return vec_.empty(); }

private:
    std::vector<T>& vec_;
};

} // namespace core
} // namespace bamboo_cut

#endif // BAMBOO_CUT_SYSTEM_UTILS_H