#include "bamboo_cut/core/system_utils.h"
#include "bamboo_cut/core/logger.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <regex>

namespace bamboo_cut {
namespace core {

SystemUtils::CommandResult SystemUtils::executeCommand(const std::string& command, int timeout_ms) {
    CommandResult result;
    
    // 安全性检查
    if (!isCommandSafe(command)) {
        result.error = "不安全的命令被拒绝执行";
        LOG_ERROR("拒绝执行不安全命令: {}", command);
        return result;
    }
    
    try {
        int pipefd[2];
        int error_pipefd[2];
        
        if (pipe(pipefd) == -1 || pipe(error_pipefd) == -1) {
            result.error = "创建管道失败";
            return result;
        }
        
        pid_t pid = fork();
        if (pid == -1) {
            close(pipefd[0]);
            close(pipefd[1]);
            close(error_pipefd[0]);
            close(error_pipefd[1]);
            result.error = "创建子进程失败";
            return result;
        }
        
        if (pid == 0) {
            // 子进程
            close(pipefd[0]);
            close(error_pipefd[0]);
            
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(error_pipefd[1], STDERR_FILENO);
            
            close(pipefd[1]);
            close(error_pipefd[1]);
            
            // 分割命令和参数
            auto tokens = tokenizeCommand(command);
            if (tokens.empty()) {
                _exit(127);
            }
            
            std::vector<char*> args;
            for (auto& token : tokens) {
                args.push_back(const_cast<char*>(token.c_str()));
            }
            args.push_back(nullptr);
            
            execvp(args[0], args.data());
            _exit(127);
        } else {
            // 父进程
            close(pipefd[1]);
            close(error_pipefd[1]);
            
            // 设置非阻塞I/O
            fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
            fcntl(error_pipefd[0], F_SETFL, O_NONBLOCK);
            
            std::string output, error;
            char buffer[4096];
            
            auto start_time = std::chrono::steady_clock::now();
            bool timeout_occurred = false;
            
            while (true) {
                // 检查超时
                if (timeout_ms > 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (elapsed >= timeout_ms) {
                        timeout_occurred = true;
                        kill(pid, SIGTERM);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        kill(pid, SIGKILL);
                        break;
                    }
                }
                
                // 读取输出
                ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
                if (n > 0) {
                    buffer[n] = '\0';
                    output += buffer;
                }
                
                // 读取错误输出
                n = read(error_pipefd[0], buffer, sizeof(buffer) - 1);
                if (n > 0) {
                    buffer[n] = '\0';
                    error += buffer;
                }
                
                // 检查子进程状态
                int status;
                pid_t wait_result = waitpid(pid, &status, WNOHANG);
                if (wait_result == pid) {
                    // 子进程结束
                    if (WIFEXITED(status)) {
                        result.exit_code = WEXITSTATUS(status);
                        result.success = (result.exit_code == 0);
                    }
                    break;
                } else if (wait_result == -1) {
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // 读取剩余输出
            while (true) {
                ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
                if (n <= 0) break;
                buffer[n] = '\0';
                output += buffer;
            }
            
            while (true) {
                ssize_t n = read(error_pipefd[0], buffer, sizeof(buffer) - 1);
                if (n <= 0) break;
                buffer[n] = '\0';
                error += buffer;
            }
            
            close(pipefd[0]);
            close(error_pipefd[0]);
            
            if (timeout_occurred) {
                result.error = "命令执行超时";
                result.exit_code = -1;
                result.success = false;
            }
            
            result.output = output;
            if (!error.empty()) {
                result.error += (result.error.empty() ? "" : "; ") + error;
            }
        }
        
    } catch (const std::exception& e) {
        result.error = std::string("命令执行异常: ") + e.what();
        LOG_ERROR("命令执行异常: {}", e.what());
    }
    
    return result;
}

bool SystemUtils::commandExists(const std::string& command) {
    auto result = executeCommand("which " + command + " 2>/dev/null", 3000);
    return result.success && !result.output.empty();
}

bool SystemUtils::fileExists(const std::string& path) {
    try {
        return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
    } catch (const std::exception&) {
        return false;
    }
}

bool SystemUtils::directoryExists(const std::string& path) {
    try {
        return std::filesystem::exists(path) && std::filesystem::is_directory(path);
    } catch (const std::exception&) {
        return false;
    }
}

std::optional<std::string> SystemUtils::readFile(const std::string& path, size_t max_size) {
    try {
        if (!fileExists(path)) {
            return std::nullopt;
        }
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::nullopt;
        }
        
        // 检查文件大小
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        if (file_size > max_size) {
            LOG_WARN("文件 {} 太大 ({} bytes)，超过限制 ({} bytes)", path, file_size, max_size);
            return std::nullopt;
        }
        
        file.seekg(0, std::ios::beg);
        std::string content(file_size, '\0');
        file.read(&content[0], file_size);
        
        return content;
    } catch (const std::exception& e) {
        LOG_ERROR("读取文件失败 {}: {}", path, e.what());
        return std::nullopt;
    }
}

int SystemUtils::safeStringToInt(const std::string& str, int default_value) {
    try {
        size_t pos;
        int result = std::stoi(str, &pos);
        // 检查是否整个字符串都被转换
        if (pos != str.length()) {
            LOG_WARN("字符串转整数部分成功: '{}', 使用默认值 {}", str, default_value);
            return default_value;
        }
        return result;
    } catch (const std::exception&) {
        LOG_WARN("字符串转整数失败: '{}', 使用默认值 {}", str, default_value);
        return default_value;
    }
}

double SystemUtils::safeStringToDouble(const std::string& str, double default_value) {
    try {
        size_t pos;
        double result = std::stod(str, &pos);
        // 检查是否整个字符串都被转换
        if (pos != str.length()) {
            LOG_WARN("字符串转浮点数部分成功: '{}', 使用默认值 {}", str, default_value);
            return default_value;
        }
        return result;
    } catch (const std::exception&) {
        LOG_WARN("字符串转浮点数失败: '{}', 使用默认值 {}", str, default_value);
        return default_value;
    }
}

bool SystemUtils::isModuleLoaded(const std::string& module_name) {
    auto result = executeCommand("lsmod | grep -q " + module_name, 3000);
    return result.success;
}

std::string SystemUtils::getEnvironmentVariable(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

bool SystemUtils::createDirectory(const std::string& path) {
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        LOG_ERROR("创建目录失败 {}: {}", path, e.what());
        return false;
    }
}

std::optional<std::string> SystemUtils::getExecutablePath(const std::string& command) {
    auto result = executeCommand("which " + command, 3000);
    if (result.success && !result.output.empty()) {
        // 去除换行符
        std::string path = result.output;
        path.erase(std::remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(std::remove(path.begin(), path.end(), '\r'), path.end());
        return path;
    }
    return std::nullopt;
}

std::vector<std::string> SystemUtils::tokenizeCommand(const std::string& command) {
    std::vector<std::string> tokens;
    std::istringstream iss(command);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

bool SystemUtils::isCommandSafe(const std::string& command) {
    // 基本安全检查
    if (command.empty()) {
        return false;
    }
    
    // 检查危险字符和模式
    std::vector<std::string> dangerous_patterns = {
        ";", "&&", "||", "|", ">", "<", ">>", "<<",
        "`", "$(",  "$(", "rm -rf", "dd if=", 
        "mkfs", "fdisk", "parted", "chmod 777"
    };
    
    for (const auto& pattern : dangerous_patterns) {
        if (command.find(pattern) != std::string::npos) {
            return false;
        }
    }
    
    // 只允许特定的安全命令
    std::vector<std::string> allowed_commands = {
        "which", "lsmod", "grep", "ls", "cat", "echo",
        "nvarguscamerasrc", "gst-inspect", "v4l2-ctl",
        "lsusb", "dmesg", "i2cdetect", "find", "modprobe",
        "v4l2-compliance", "media-ctl", "test", "stat"
    };
    
    auto tokens = tokenizeCommand(command);
    if (tokens.empty()) {
        return false;
    }
    
    std::string base_command = tokens[0];
    // 移除路径，只保留命令名
    size_t last_slash = base_command.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_command = base_command.substr(last_slash + 1);
    }
    
    for (const auto& allowed : allowed_commands) {
        if (base_command == allowed) {
            return true;
        }
    }
    
    LOG_WARN("命令不在安全白名单中: {}", base_command);
    return false;
}

} // namespace core
} // namespace bamboo_cut