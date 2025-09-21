#include "input/touch_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>

// 位测试宏定义
#ifndef NBITS
#define NBITS(x) ((((x)-1)/(sizeof(long)*8))+1)
#endif
#ifndef NLONGS
#define NLONGS(x) NBITS(x)
#endif
#ifndef test_bit
#define test_bit(bit, array) ((array)[(bit)/(sizeof(long)*8)] & (1L<<((bit)%(sizeof(long)*8))))
#endif

// 全局变量
static int touch_fd = -1;
static touch_driver_config_t touch_config;
static touch_point_t last_point = {0, 0, false, 0};
static bool driver_initialized = false;
static pthread_t touch_thread;
static bool thread_running = false;
static pthread_mutex_t touch_mutex = PTHREAD_MUTEX_INITIALIZER;

// 默认配置
static const touch_driver_config_t default_config = {
    .device_path = "/dev/input/event0",
    .max_x = 4095,
    .max_y = 4095,
    .screen_width = 1920,
    .screen_height = 1080,
    .swap_xy = false,
    .invert_x = false,
    .invert_y = false
};

// USB触摸屏设备检测
bool usb_touch_detect() {
    DIR *input_dir = opendir("/dev/input");
    if (!input_dir) {
        printf("无法打开 /dev/input 目录\n");
        return false;
    }
    
    struct dirent *entry;
    bool found = false;
    
    while ((entry = readdir(input_dir)) != NULL) {
        if (strncmp(entry->d_name, "event", 5) == 0) {
            char device_path[256];
            snprintf(device_path, sizeof(device_path), "/dev/input/%s", entry->d_name);
            
            int fd = open(device_path, O_RDONLY | O_NONBLOCK);
            if (fd >= 0) {
                char device_name[256] = {0};
                if (ioctl(fd, EVIOCGNAME(sizeof(device_name) - 1), device_name) >= 0) {
                    // 检查是否为触摸屏设备
                    if (strstr(device_name, "Touch") || strstr(device_name, "touch") ||
                        strstr(device_name, "Touchscreen") || strstr(device_name, "touchscreen") ||
                        strstr(device_name, "USB") || strstr(device_name, "HID")) {
                        printf("检测到USB触摸屏设备: %s (%s)\n", device_name, device_path);
                        
                        // 更新配置
                        strncpy((char*)touch_config.device_path, device_path, strlen(device_path) + 1);
                        found = true;
                        close(fd);
                        break;
                    }
                }
                close(fd);
            }
        }
    }
    
    closedir(input_dir);
    return found;
}

// 配置USB触摸屏
bool usb_touch_configure() {
    if (touch_fd < 0) return false;
    
    // 获取设备能力
    unsigned long evbit[NLONGS(EV_MAX)];
    if (ioctl(touch_fd, EVIOCGBIT(0, EV_MAX), evbit) < 0) {
        printf("无法获取设备事件类型\n");
        return false;
    }
    
    // 检查是否支持绝对坐标事件
    if (test_bit(EV_ABS, evbit)) {
        struct input_absinfo abs_info;
        
        // 获取X轴信息
        if (ioctl(touch_fd, EVIOCGABS(ABS_X), &abs_info) >= 0) {
            touch_config.max_x = abs_info.maximum;
            printf("X轴范围: %d - %d\n", abs_info.minimum, abs_info.maximum);
        }
        
        // 获取Y轴信息
        if (ioctl(touch_fd, EVIOCGABS(ABS_Y), &abs_info) >= 0) {
            touch_config.max_y = abs_info.maximum;
            printf("Y轴范围: %d - %d\n", abs_info.minimum, abs_info.maximum);
        }
        
        return true;
    }
    
    return false;
}

// 触摸事件读取线程
void* touch_read_thread(void* arg) {
    struct input_event ev;
    int16_t raw_x = 0, raw_y = 0;
    bool touch_down = false;
    
    printf("触摸读取线程已启动\n");
    
    while (thread_running) {
        if (touch_fd < 0) {
            usleep(100000); // 100ms
            continue;
        }
        
        ssize_t bytes = read(touch_fd, &ev, sizeof(ev));
        if (bytes == sizeof(ev)) {
            pthread_mutex_lock(&touch_mutex);
            
            switch (ev.type) {
                case EV_ABS:
                    if (ev.code == ABS_X) {
                        raw_x = ev.value;
                    } else if (ev.code == ABS_Y) {
                        raw_y = ev.value;
                    }
                    break;
                    
                case EV_KEY:
                    if (ev.code == BTN_TOUCH || ev.code == BTN_LEFT) {
                        touch_down = (ev.value == 1);
                    }
                    break;
                    
                case EV_SYN:
                    if (ev.code == SYN_REPORT) {
                        // 坐标转换
                        int16_t screen_x = raw_x;
                        int16_t screen_y = raw_y;
                        
                        // 缩放到屏幕坐标
                        if (touch_config.max_x > 0) {
                            screen_x = (raw_x * touch_config.screen_width) / touch_config.max_x;
                        }
                        if (touch_config.max_y > 0) {
                            screen_y = (raw_y * touch_config.screen_height) / touch_config.max_y;
                        }
                        
                        // 应用变换
                        if (touch_config.swap_xy) {
                            int16_t temp = screen_x;
                            screen_x = screen_y;
                            screen_y = temp;
                        }
                        
                        if (touch_config.invert_x) {
                            screen_x = touch_config.screen_width - screen_x;
                        }
                        
                        if (touch_config.invert_y) {
                            screen_y = touch_config.screen_height - screen_y;
                        }
                        
                        // 更新触摸点数据
                        last_point.x = screen_x;
                        last_point.y = screen_y;
                        last_point.pressed = touch_down;
                        last_point.timestamp = ev.time.tv_sec * 1000 + ev.time.tv_usec / 1000;
                        
                        // 调试输出
                        if (touch_down) {
                            printf("触摸: (%d, %d) -> 屏幕: (%d, %d)\n", raw_x, raw_y, screen_x, screen_y);
                        }
                    }
                    break;
            }
            
            pthread_mutex_unlock(&touch_mutex);
        } else if (bytes < 0 && errno != EAGAIN) {
            printf("读取触摸数据错误: %s\n", strerror(errno));
            usleep(10000); // 10ms
        } else {
            usleep(1000); // 1ms
        }
    }
    
    printf("触摸读取线程已退出\n");
    return NULL;
}

// LVGL触摸读取回调
void touch_driver_read(lv_indev_drv_t* indev_drv, lv_indev_data_t* data) {
    pthread_mutex_lock(&touch_mutex);
    
    data->point.x = last_point.x;
    data->point.y = last_point.y;
    data->state = last_point.pressed ? LV_INDEV_STATE_PRESSED : LV_INDEV_STATE_RELEASED;
    
    pthread_mutex_unlock(&touch_mutex);
}

// 获取USB触摸屏设备路径
const char* usb_touch_get_device_path() {
    return touch_config.device_path;
}

// 设置触摸配置
void touch_driver_set_config(const touch_driver_config_t* config) {
    if (config) {
        memcpy(&touch_config, config, sizeof(touch_driver_config_t));
    }
}

// 检查触摸驱动是否可用
bool touch_driver_is_available() {
    return driver_initialized && touch_fd >= 0;
}

// 初始化触摸驱动
bool touch_driver_init() {
    printf("初始化USB Type-C触摸屏驱动\n");
    
    if (driver_initialized) {
        printf("触摸驱动已初始化\n");
        return true;
    }
    
    // 使用默认配置
    memcpy(&touch_config, &default_config, sizeof(touch_driver_config_t));
    
    // 检测USB触摸屏
    if (!usb_touch_detect()) {
        printf("未检测到USB触摸屏设备，使用默认路径: %s\n", touch_config.device_path);
    }
    
    // 打开触摸设备
    touch_fd = open(touch_config.device_path, O_RDONLY | O_NONBLOCK);
    if (touch_fd < 0) {
        printf("无法打开触摸设备 %s: %s\n", touch_config.device_path, strerror(errno));
        return false;
    }
    
    // 配置触摸设备
    if (!usb_touch_configure()) {
        printf("警告: 无法配置触摸设备，使用默认参数\n");
    }
    
    // 启动读取线程
    thread_running = true;
    if (pthread_create(&touch_thread, NULL, touch_read_thread, NULL) != 0) {
        printf("无法创建触摸读取线程\n");
        close(touch_fd);
        touch_fd = -1;
        return false;
    }
    
    driver_initialized = true;
    printf("USB触摸屏驱动初始化成功\n");
    printf("设备路径: %s\n", touch_config.device_path);
    printf("分辨率: %dx%d -> %dx%d\n",
           touch_config.max_x, touch_config.max_y,
           touch_config.screen_width, touch_config.screen_height);
    
    return true;
}

// 清理触摸驱动
void touch_driver_deinit() {
    printf("清理USB触摸屏驱动\n");
    
    if (thread_running) {
        thread_running = false;
        if (pthread_join(touch_thread, NULL) != 0) {
            printf("等待触摸线程退出失败\n");
        }
    }
    
    if (touch_fd >= 0) {
        close(touch_fd);
        touch_fd = -1;
    }
    
    driver_initialized = false;
    printf("USB触摸屏驱动已清理\n");
}