/**
 * 智能切竹机控制系统 - LVGL版本
 * 主程序入口
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string.h>
#include <sched.h>

#include "lvgl.h"
#include "app/main_app.h"
#include "app/config_manager.h"
#include "system/lv_port_tick.h"
#include "display/framebuffer_driver.h"
#include "display/lvgl_display.h"
#include "input/touch_driver.h"
#include "common/utils.h"

/* 全局变量 */
static MainApp* g_app = nullptr;
static volatile bool g_running = true;

/* 信号处理函数 */
void signal_handler(int sig) {
    printf("接收到信号 %d，开始清理资源...\n", sig);
    g_running = false;
}


/* 主线程循环 */
void* main_loop(void* arg) {
    printf("LVGL主循环线程启动\n");
    
    uint32_t loop_count = 0;
    
    while (g_running) {
        /* 处理LVGL任务 */
        lv_timer_handler();
        
        /* 处理应用程序事件 */
        if (g_app) {
            g_app->process_events();
        }
        
        /* 更新摄像头显示 - 每次循环都更新以保证流畅 */
        update_camera_display();
        
        /* 更新系统状态显示 - 现在有频率限制，可以安全调用 */
        update_system_status_display();
        
        /* 控制帧率 - 60fps，但其他更新有自己的频率限制 */
        usleep(16667); // 约16.67ms，60fps，确保UI响应流畅
        
        loop_count++;
    }
    
    printf("LVGL主循环线程退出 (处理了 %u 次循环)\n", loop_count);
    return nullptr;
}

/* 检查系统要求 */
bool check_system_requirements() {
    printf("检查系统要求...\n");
    
    // 检查framebuffer设备（非阻塞）
    if (access("/dev/fb0", F_OK) != 0) {
        printf("警告: 找不到framebuffer设备 /dev/fb0，可能会影响显示\n");
        // 不返回false，让系统继续尝试启动
    }
    
    // 检查触摸设备 - 优先检查 event2
    bool touch_found = false;
    const char* touch_devices[] = {"/dev/input/event2", "/dev/input/event1", "/dev/input/event0", NULL};
    
    for (int i = 0; touch_devices[i] != NULL; i++) {
        if (access(touch_devices[i], F_OK) == 0) {
            printf("找到触摸设备: %s\n", touch_devices[i]);
            touch_found = true;
            break;
        }
    }
    
    if (!touch_found) {
        printf("警告: 未找到触摸设备\n");
    }
    
    // 检查摄像头设备
    if (access("/dev/video0", F_OK) != 0) {
        printf("警告: 找不到摄像头设备 /dev/video0\n");
    }
    
    // 检查CUDA设备
    if (access("/dev/nvidia0", F_OK) != 0) {
        printf("警告: 找不到NVIDIA GPU设备\n");
    }
    
    printf("系统要求检查完成\n");
    return true;
}

/* 初始化LVGL */
bool initialize_lvgl() {
    printf("初始化LVGL...\n");
    
    /* 初始化LVGL */
    lv_init();
    
    /* 初始化LVGL时钟系统 */
    lv_port_tick_init();
    
    /* 初始化显示驱动 */
    if (!lvgl_display_init()) {
        fprintf(stderr, "错误: LVGL显示驱动初始化失败\n");
        return false;
    }
    
    /* 初始化触摸驱动 */
    if (!touch_driver_init()) {
        printf("警告: 触摸驱动初始化失败，将禁用触摸功能\n");
    } else {
        printf("触摸驱动初始化成功\n");
    }
    
    printf("LVGL初始化完成\n");
    return true;
}

/* 设置实时优先级 */
void set_realtime_priority() {
    struct sched_param param;
    param.sched_priority = 50; // 中等优先级
    
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("警告: 无法设置实时调度优先级");
    } else {
        printf("实时调度优先级设置成功\n");
    }
}

/* 打印系统信息 */
void print_system_info() {
    printf("================================\n");
    printf("智能切竹机控制系统 - LVGL版本\n");
    printf("版本: 2.0.0\n");
    printf("构建时间: %s %s\n", __DATE__, __TIME__);
    printf("================================\n");
    
    // 打印CPU信息
    int ret1 = system("cat /proc/cpuinfo | grep 'model name' | head -1");
    (void)ret1; // 明确忽略返回值
    
    // 打印内存信息
    int ret2 = system("cat /proc/meminfo | grep 'MemTotal'");
    (void)ret2; // 明确忽略返回值
    
    // 打印GPU信息
    int ret3 = system("nvidia-smi -L 2>/dev/null || echo '未检测到NVIDIA GPU'");
    (void)ret3; // 明确忽略返回值
    
    printf("================================\n");
}

/* 命令行参数处理 */
void print_usage(const char* program_name) {
    printf("用法: %s [选项]\n", program_name);
    printf("选项:\n");
    printf("  -c, --config <文件>     指定配置文件路径\n");
    printf("  -d, --debug            启用调试模式\n");
    printf("  -f, --fullscreen       全屏模式\n");
    printf("  -v, --version          显示版本信息\n");
    printf("  -h, --help             显示帮助信息\n");
}

int main(int argc, char* argv[]) {
    const char* config_file = nullptr;
    bool debug_mode = false;
    bool fullscreen = false;
    
    /* 解析命令行参数 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                fprintf(stderr, "错误: -c/--config 需要配置文件路径\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--fullscreen") == 0) {
            fullscreen = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            printf("智能切竹机控制系统 LVGL版本 2.0.0\n");
            return 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "错误: 未知参数 %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    /* 打印系统信息 */
    print_system_info();
    
    /* 检查系统要求（非阻塞模式） */
    check_system_requirements(); // 只检查不强制退出
    
    /* 设置信号处理 */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGQUIT, signal_handler);
    
    /* 设置实时优先级 */
    set_realtime_priority();
    
    /* 初始化LVGL */
    if (!initialize_lvgl()) {
        fprintf(stderr, "错误: LVGL初始化失败\n");
        return 1;
    }
    
    /* 加载配置 */
    ConfigManager config;
    if (!config.load(config_file)) {
        fprintf(stderr, "错误: 配置加载失败\n");
        return 1;
    }
    
    /* 应用调试模式设置 */
    if (debug_mode) {
        config.set_debug_mode(true);
        // 启用触摸驱动调试模式
        touch_driver_enable_debug(true);
    }
    
    /* 创建主应用程序 */
    g_app = new MainApp(config.get_config());
    if (!g_app->initialize()) {
        fprintf(stderr, "错误: 应用程序初始化失败\n");
        delete g_app;
        return 1;
    }
    
    /* 启动应用程序 */
    if (!g_app->start()) {
        fprintf(stderr, "错误: 应用程序启动失败\n");
        delete g_app;
        return 1;
    }
    
    printf("应用程序启动成功，按 Ctrl+C 退出\n");
    
    /* 创建主循环线程 */
    pthread_t main_thread;
    if (pthread_create(&main_thread, nullptr, main_loop, nullptr) != 0) {
        fprintf(stderr, "错误: 无法创建主循环线程\n");
        delete g_app;
        return 1;
    }
    
    /* 等待主循环线程结束 */
    pthread_join(main_thread, nullptr);
    
    /* 清理资源 */
    printf("正在清理资源...\n");
    
    if (g_app) {
        g_app->stop();
        delete g_app;
        g_app = nullptr;
    }
    
    /* 清理LVGL */
    touch_driver_deinit();
    lvgl_display_deinit();
    
    printf("程序正常退出\n");
    return 0;
}