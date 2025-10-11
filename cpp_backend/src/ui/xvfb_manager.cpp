/**
 * @file xvfb_manager.cpp
 * @brief Xvfb虚拟显示管理器 - 解决nvarguscamerasrc EGL初始化问题
 */

#include "bamboo_cut/ui/xvfb_manager.h"
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <chrono>
#include <thread>

namespace bamboo_cut {
namespace ui {

// 静态成员初始化
pid_t XvfbManager::xvfb_pid_ = 0;
bool XvfbManager::initialized_ = false;
const char* XvfbManager::DISPLAY_NUM = "99";

bool XvfbManager::startXvfb() {
    std::cout << "🔧 检查Xvfb虚拟显示服务状态..." << std::endl;
    
    // 检查是否已有Xvfb进程运行
    int check_result = system("pgrep -x Xvfb > /dev/null 2>&1");
    if (check_result == 0) {
        std::cout << "✅ Xvfb虚拟显示服务已在运行" << std::endl;
        initialized_ = true;
        return true;
    }
    
    std::cout << "🚀 启动Xvfb虚拟显示服务..." << std::endl;
    
    // 清理可能存在的锁文件
    std::string lock_file = std::string("/tmp/.X") + DISPLAY_NUM + "-lock";
    unlink(lock_file.c_str());
    
    // 创建子进程启动Xvfb
    xvfb_pid_ = fork();
    if (xvfb_pid_ == 0) {
        // 子进程：启动Xvfb
        execl("/usr/bin/Xvfb", "Xvfb",
              (std::string(":") + DISPLAY_NUM).c_str(),
              "-screen", "0", "1920x1080x24",
              "-ac", "+extension", "GLX",
              "+render", "-noreset",
              (char*)NULL);
        
        // 如果execl失败，退出子进程
        std::cerr << "❌ 无法启动Xvfb" << std::endl;
        _exit(1);
    } else if (xvfb_pid_ > 0) {
        // 父进程：等待Xvfb启动
        std::cout << "⏳ 等待Xvfb启动（PID: " << xvfb_pid_ << "）..." << std::endl;
        
        // 等待最多5秒让Xvfb启动
        for (int i = 0; i < 50; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // 检查进程是否还存在
            int status;
            pid_t result = waitpid(xvfb_pid_, &status, WNOHANG);
            if (result == xvfb_pid_) {
                std::cerr << "❌ Xvfb启动失败，进程已退出" << std::endl;
                return false;
            }
            
            // 检查是否能连接到显示
            std::string test_cmd = std::string("DISPLAY=:") + DISPLAY_NUM + " xdpyinfo > /dev/null 2>&1";
            if (system(test_cmd.c_str()) == 0) {
                std::cout << "✅ Xvfb虚拟显示服务启动成功！" << std::endl;
                initialized_ = true;
                return true;
            }
        }
        
        std::cerr << "❌ Xvfb启动超时" << std::endl;
        return false;
    } else {
        std::cerr << "❌ 无法创建子进程启动Xvfb" << std::endl;
        return false;
    }
}

void XvfbManager::setupEnvironment() {
    if (!initialized_) {
        if (!startXvfb()) {
            std::cerr << "⚠️ 警告：Xvfb启动失败，可能会遇到EGL初始化问题" << std::endl;
            return;
        }
    }
    
    std::string display = std::string(":") + DISPLAY_NUM;
    setenv("DISPLAY", display.c_str(), 1);
    setenv("XAUTHORITY", "", 1);
    
    std::cout << "🔧 设置环境变量 DISPLAY=" << display << std::endl;
    std::cout << "✅ Xvfb环境配置完成，nvarguscamerasrc EGL初始化应该正常" << std::endl;
}

void XvfbManager::stopXvfb() {
    if (xvfb_pid_ > 0 && initialized_) {
        std::cout << "🛑 停止Xvfb虚拟显示服务（PID: " << xvfb_pid_ << "）..." << std::endl;
        kill(xvfb_pid_, SIGTERM);
        
        // 等待进程结束
        int status;
        waitpid(xvfb_pid_, &status, 0);
        
        xvfb_pid_ = 0;
        initialized_ = false;
        std::cout << "✅ Xvfb虚拟显示服务已停止" << std::endl;
    }
}

bool XvfbManager::isRunning() {
    return initialized_ && (system("pgrep -x Xvfb > /dev/null 2>&1") == 0);
}

private:
    static pid_t xvfb_pid_;
    static bool initialized_;
    static const char* DISPLAY_NUM;
    
public:
    /**
     * @brief 启动Xvfb虚拟显示服务
     * @return true 如果启动成功或已运行
     */
    static bool startXvfb() {
        std::cout << "🔧 检查Xvfb虚拟显示服务状态..." << std::endl;
        
        // 检查是否已有Xvfb进程运行
        int check_result = system("pgrep -x Xvfb > /dev/null 2>&1");
        if (check_result == 0) {
            std::cout << "✅ Xvfb虚拟显示服务已在运行" << std::endl;
            initialized_ = true;
            return true;
        }
        
        std::cout << "🚀 启动Xvfb虚拟显示服务..." << std::endl;
        
        // 清理可能存在的锁文件
        std::string lock_file = std::string("/tmp/.X") + DISPLAY_NUM + "-lock";
        unlink(lock_file.c_str());
        
        // 创建子进程启动Xvfb
        xvfb_pid_ = fork();
        if (xvfb_pid_ == 0) {
            // 子进程：启动Xvfb
            execl("/usr/bin/Xvfb", "Xvfb", 
                  (std::string(":") + DISPLAY_NUM).c_str(),
                  "-screen", "0", "1920x1080x24",
                  "-ac", "+extension", "GLX",
                  "+render", "-noreset",
                  (char*)NULL);
            
            // 如果execl失败，退出子进程
            std::cerr << "❌ 无法启动Xvfb" << std::endl;
            _exit(1);
        } else if (xvfb_pid_ > 0) {
            // 父进程：等待Xvfb启动
            std::cout << "⏳ 等待Xvfb启动（PID: " << xvfb_pid_ << "）..." << std::endl;
            
            // 等待最多5秒让Xvfb启动
            for (int i = 0; i < 50; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // 检查进程是否还存在
                int status;
                pid_t result = waitpid(xvfb_pid_, &status, WNOHANG);
                if (result == xvfb_pid_) {
                    std::cerr << "❌ Xvfb启动失败，进程已退出" << std::endl;
                    return false;
                }
                
                // 检查是否能连接到显示
                std::string test_cmd = std::string("DISPLAY=:") + DISPLAY_NUM + " xdpyinfo > /dev/null 2>&1";
                if (system(test_cmd.c_str()) == 0) {
                    std::cout << "✅ Xvfb虚拟显示服务启动成功！" << std::endl;
                    initialized_ = true;
                    return true;
                }
            }
            
            std::cerr << "❌ Xvfb启动超时" << std::endl;
            return false;
        } else {
            std::cerr << "❌ 无法创建子进程启动Xvfb" << std::endl;
            return false;
        }
    }
    
    /**
     * @brief 设置环境变量以使用Xvfb显示
     */
    static void setupEnvironment() {
        if (!initialized_) {
            if (!startXvfb()) {
                std::cerr << "⚠️ 警告：Xvfb启动失败，可能会遇到EGL初始化问题" << std::endl;
                return;
            }
        }
        
        std::string display = std::string(":") + DISPLAY_NUM;
        setenv("DISPLAY", display.c_str(), 1);
        setenv("XAUTHORITY", "", 1);
        
        std::cout << "🔧 设置环境变量 DISPLAY=" << display << std::endl;
        std::cout << "✅ Xvfb环境配置完成，nvarguscamerasrc EGL初始化应该正常" << std::endl;
    }
    
    /**
     * @brief 停止Xvfb虚拟显示服务
     */
    static void stopXvfb() {
        if (xvfb_pid_ > 0 && initialized_) {
            std::cout << "🛑 停止Xvfb虚拟显示服务（PID: " << xvfb_pid_ << "）..." << std::endl;
            kill(xvfb_pid_, SIGTERM);
            
            // 等待进程结束
            int status;
            waitpid(xvfb_pid_, &status, 0);
            
            xvfb_pid_ = 0;
            initialized_ = false;
            std::cout << "✅ Xvfb虚拟显示服务已停止" << std::endl;
        }
    }
    
    /**
     * @brief 检查Xvfb是否正在运行
     */
    static bool isRunning() {
        return initialized_ && (system("pgrep -x Xvfb > /dev/null 2>&1") == 0);
    }
};

// 静态成员初始化
pid_t XvfbManager::xvfb_pid_ = 0;
bool XvfbManager::initialized_ = false;
const char* XvfbManager::DISPLAY_NUM = "99";

} // namespace ui
} // namespace bamboo_cut