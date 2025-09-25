/**
 * @file main.cpp
 * @brief C++ LVGL一体化竹子识别系统主程序入口
 * AI竹子识别切割系统 - 工业级嵌入式应用
 */

#include <iostream>
#include <signal.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>

#include "bamboo_cut/core/bamboo_system.h"

using namespace bamboo_cut;

// 全局系统实例
std::unique_ptr<core::BambooSystem> g_system;

/**
 * @brief 显示帮助信息
 */
void showHelp(const char* program_name) {
    std::cout << "AI Bamboo Recognition System v2.1 - C++ LVGL Edition\n";
    std::cout << "工业级竹子识别切割系统\n\n";
    std::cout << "用法: " << program_name << " [选项]\n\n";
    std::cout << "选项:\n";
    std::cout << "  -c, --config <file>     指定配置文件路径 (默认: config/system_config.yaml)\n";
    std::cout << "  -d, --daemon           作为守护进程运行\n";
    std::cout << "  -v, --verbose          详细输出模式\n";
    std::cout << "  -t, --test             测试模式（不启动实际硬件）\n";
    std::cout << "  -s, --status           显示系统状态\n";
    std::cout << "  -h, --help             显示此帮助信息\n";
    std::cout << "      --version          显示版本信息\n\n";
    std::cout << "信号处理:\n";
    std::cout << "  SIGINT (Ctrl+C)        优雅关闭系统\n";
    std::cout << "  SIGTERM                终止系统\n";
    std::cout << "  SIGUSR1                重新加载配置\n";
    std::cout << "  SIGUSR2                切换详细模式\n\n";
    std::cout << "示例:\n";
    std::cout << "  " << program_name << " -c /opt/bamboo/config.yaml\n";
    std::cout << "  " << program_name << " --daemon --verbose\n";
    std::cout << "  " << program_name << " --test\n";
}

/**
 * @brief 显示版本信息
 */
void showVersion() {
    auto version = core::getVersionInfo();
    std::cout << "AI Bamboo Recognition System\n";
    std::cout << "版本: " << version.toString() << "\n";
    std::cout << "构建日期: " << version.build_date << "\n";
    std::cout << "Git提交: " << version.git_commit << "\n";
    std::cout << "C++ LVGL一体化架构\n";
    std::cout << "支持特性:\n";
#ifdef ENABLE_TENSORRT
    std::cout << "  - TensorRT加速推理\n";
#endif
#ifdef ENABLE_CUDA
    std::cout << "  - CUDA GPU加速\n";
#endif
#ifdef ENABLE_MODBUS
    std::cout << "  - Modbus TCP通信\n";
#endif
#ifdef JETSON_PLATFORM
    std::cout << "  - Jetson Orin NX优化\n";
#endif
    std::cout << "  - LVGL工业界面\n";
    std::cout << "  - YOLOv8目标检测\n";
    std::cout << "  - 实时视频处理\n";
}

/**
 * @brief 检查系统运行环境
 */
bool checkSystemEnvironment() {
    // 检查是否以root权限运行（可选）
    if (geteuid() != 0) {
        std::cout << "警告: 建议以root权限运行以获得最佳性能\n";
    }
    
    // 检查配置目录
    struct stat st;
    if (stat("config", &st) != 0) {
        std::cerr << "错误: config目录不存在\n";
        return false;
    }
    
    // 检查模型文件
    if (stat("models", &st) != 0) {
        std::cerr << "警告: models目录不存在，AI推理可能无法工作\n";
    }
    
    // 检查触摸设备
    if (access("/dev/input", R_OK) != 0) {
        std::cerr << "警告: 无法访问触摸设备，界面可能无法响应触摸\n";
    }
    
    // 检查framebuffer设备
    if (access("/dev/fb0", R_OK) != 0) {
        std::cerr << "警告: 无法访问framebuffer设备，可能需要配置显示\n";
    }
    
    return true;
}

/**
 * @brief 创建守护进程
 */
bool daemonize() {
    pid_t pid = fork();
    
    if (pid < 0) {
        std::cerr << "错误: 无法创建子进程\n";
        return false;
    }
    
    if (pid > 0) {
        // 父进程退出
        exit(0);
    }
    
    // 子进程继续
    if (setsid() < 0) {
        std::cerr << "错误: 无法创建新会话\n";
        return false;
    }
    
    // 改变工作目录到根目录
    chdir("/");
    
    // 关闭标准输入输出
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    
    return true;
}

/**
 * @brief 信号处理器
 */
void signalHandler(int signal) {
    switch (signal) {
        case SIGINT:
        case SIGTERM:
            std::cout << "\n收到终止信号，正在优雅关闭系统...\n";
            if (g_system) {
                g_system->stop();
            }
            break;
            
        case SIGUSR1:
            std::cout << "收到重载配置信号\n";
            if (g_system) {
                core::SystemConfig config;
                if (config.loadFromFile("config/system_config.yaml")) {
                    g_system->reloadConfig(config);
                    std::cout << "配置重新加载完成\n";
                } else {
                    std::cerr << "配置重新加载失败\n";
                }
            }
            break;
            
        case SIGUSR2:
            std::cout << "收到状态查询信号\n";
            if (g_system) {
                auto info = g_system->getSystemInfo();
                std::cout << "系统状态: " << info.state_name << "\n";
                std::cout << "运行时间: " << info.uptime.count() << "秒\n";
                std::cout << "当前步骤: " << info.current_workflow_step << "\n";
                std::cout << "推理FPS: " << info.performance.inference_fps << "\n";
                std::cout << "界面FPS: " << info.performance.ui_fps << "\n";
            }
            break;
    }
}

/**
 * @brief 安装信号处理器
 */
void setupSignalHandlers() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGUSR1, signalHandler);
    signal(SIGUSR2, signalHandler);
    
    // 忽略SIGPIPE
    signal(SIGPIPE, SIG_IGN);
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 默认配置
    std::string config_file = "config/system_config.yaml";
    bool daemon_mode = false;
    bool verbose_mode = false;
    bool test_mode = false;
    bool show_status = false;
    
    // 解析命令行参数
    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"daemon", no_argument, 0, 'd'},
        {"verbose", no_argument, 0, 'v'},
        {"test", no_argument, 0, 't'},
        {"status", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    
    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "c:dvtsh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'c':
                config_file = optarg;
                break;
            case 'd':
                daemon_mode = true;
                break;
            case 'v':
                verbose_mode = true;
                break;
            case 't':
                test_mode = true;
                break;
            case 's':
                show_status = true;
                break;
            case 'h':
                showHelp(argv[0]);
                return 0;
            case 0:
                if (option_index == 6) { // --version
                    showVersion();
                    return 0;
                }
                break;
            case '?':
                std::cerr << "使用 --help 查看帮助信息\n";
                return 1;
        }
    }
    
    std::cout << "=== AI Bamboo Recognition System v2.1 ===\n";
    std::cout << "C++ LVGL一体化工业级嵌入式应用\n";
    std::cout << "配置文件: " << config_file << "\n";
    
    if (verbose_mode) {
        std::cout << "详细模式: 启用\n";
    }
    
    if (test_mode) {
        std::cout << "测试模式: 启用\n";
    }
    
    // 检查系统环境
    if (!checkSystemEnvironment()) {
        std::cerr << "系统环境检查失败\n";
        return 1;
    }
    
    // 如果是守护进程模式，创建守护进程
    if (daemon_mode && !test_mode) {
        std::cout << "切换到守护进程模式...\n";
        if (!daemonize()) {
            return 1;
        }
    }
    
    // 安装信号处理器
    setupSignalHandlers();
    
    try {
        // 加载系统配置
        core::SystemConfig config;
        if (!config.loadFromFile(config_file)) {
            std::cerr << "警告: 无法加载配置文件，使用默认配置\n";
        }
        
        // 测试模式配置调整
        if (test_mode) {
            config.system_params.enable_modbus_communication = false;
            std::cout << "测试模式: 禁用Modbus通信\n";
        }
        
        // 创建系统实例
        g_system = std::make_unique<core::BambooSystem>();
        
        // 初始化系统
        std::cout << "正在初始化系统...\n";
        if (!g_system->initialize(config)) {
            std::cerr << "系统初始化失败: " << g_system->getLastError() << "\n";
            return 1;
        }
        
        // 启动系统
        std::cout << "正在启动系统...\n";
        if (!g_system->start()) {
            std::cerr << "系统启动失败: " << g_system->getLastError() << "\n";
            return 1;
        }
        
        std::cout << "系统启动完成！\n";
        std::cout << "使用 Ctrl+C 或 kill -TERM <pid> 优雅关闭系统\n";
        std::cout << "使用 kill -USR1 <pid> 重新加载配置\n";
        std::cout << "使用 kill -USR2 <pid> 查看系统状态\n\n";
        
        // 运行系统主循环
        int result = g_system->run();
        
        std::cout << "系统正在关闭...\n";
        g_system->stop();
        g_system.reset();
        
        std::cout << "系统已完全关闭\n";
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "系统异常: " << e.what() << "\n";
        if (g_system) {
            g_system->stop();
            g_system.reset();
        }
        return 1;
    } catch (...) {
        std::cerr << "未知系统异常\n";
        if (g_system) {
            g_system->stop();
            g_system.reset();
        }
        return 1;
    }
}