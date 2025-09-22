#include "input/touch_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>

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

// 前向声明
void* screen_monitor_worker(void* arg);

// 全局变量
static int touch_fd = -1;
static touch_driver_config_t touch_config;
static touch_point_t last_point = {0, 0, false, 0, 0, 0};
static multi_touch_data_t multitouch_data = {0};
static touch_device_info_t device_info = {0};
static screen_info_t screen_info = {0};
static bool driver_initialized = false;
static bool debug_enabled = false;
static pthread_t touch_thread;
static pthread_t screen_monitor_thread;
static bool thread_running = false;
static bool screen_monitor_running = false;
static pthread_mutex_t touch_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t screen_mutex = PTHREAD_MUTEX_INITIALIZER;

// Jetson Orin NX 优化配置
static const touch_driver_config_t jetson_orin_nx_config = {
    .device_path = "/dev/input/event2",
    .max_x = 32767,
    .max_y = 32767,
    .screen_width = 1920,
    .screen_height = 1080,
    .swap_xy = false,
    .invert_x = false,
    .invert_y = false,
    .scale_x = 1.0f,
    .scale_y = 1.0f,
    .offset_x = 0,
    .offset_y = 0,
    .enable_multitouch = true,
    .touch_threshold = 10
};

// 默认配置
static const touch_driver_config_t default_config = {
    .device_path = "/dev/input/event0",
    .max_x = 4095,
    .max_y = 4095,
    .screen_width = 1920,
    .screen_height = 1080,
    .swap_xy = false,
    .invert_x = false,
    .invert_y = false,
    .scale_x = 1.0f,
    .scale_y = 1.0f,
    .offset_x = 0,
    .offset_y = 0,
    .enable_multitouch = false,
    .touch_threshold = 5
};

// 智能触摸设备检测 - 优先使用 /dev/input/event2
bool touch_device_smart_detect() {
    if (debug_enabled) {
        printf("开始智能触摸设备检测...\n");
    }
    
    touch_device_info_t candidates[16];
    int candidate_count = 0;
    
    // 首先检查优先设备 /dev/input/event2
    const char* priority_devices[] = {
        "/dev/input/event2",
        "/dev/input/event1",
        "/dev/input/event0",
        "/dev/input/event3",
        "/dev/input/event4",
        NULL
    };
    
    for (int i = 0; priority_devices[i] != NULL; i++) {
        if (touch_driver_test_device(priority_devices[i])) {
            strncpy(touch_config.device_path, priority_devices[i], sizeof(touch_config.device_path) - 1);
            touch_config.device_path[sizeof(touch_config.device_path) - 1] = '\0';
            
            if (debug_enabled) {
                printf("使用优先触摸设备: %s\n", priority_devices[i]);
            }
            return true;
        }
    }
    
    // 扫描所有输入设备
    DIR *input_dir = opendir("/dev/input");
    if (!input_dir) {
        printf("无法打开 /dev/input 目录\n");
        return false;
    }
    
    struct dirent *entry;
    while ((entry = readdir(input_dir)) != NULL && candidate_count < 16) {
        if (strncmp(entry->d_name, "event", 5) == 0) {
            char device_path[256];
            snprintf(device_path, sizeof(device_path), "/dev/input/%s", entry->d_name);
            
            int fd = open(device_path, O_RDONLY | O_NONBLOCK);
            if (fd >= 0) {
                char device_name[256] = {0};
                struct input_id device_id;
                
                if (ioctl(fd, EVIOCGNAME(sizeof(device_name) - 1), device_name) >= 0 &&
                    ioctl(fd, EVIOCGID, &device_id) >= 0) {
                    
                    // 检查设备能力
                    unsigned long evbit[NLONGS(EV_MAX)];
                    unsigned long absbit[NLONGS(ABS_MAX)];
                    
                    if (ioctl(fd, EVIOCGBIT(0, EV_MAX), evbit) >= 0 &&
                        ioctl(fd, EVIOCGBIT(EV_ABS, ABS_MAX), absbit) >= 0) {
                        
                        // 检查是否为触摸设备
                        if (test_bit(EV_ABS, evbit) &&
                            test_bit(ABS_X, absbit) &&
                            test_bit(ABS_Y, absbit)) {
                            
                            touch_device_info_t* candidate = &candidates[candidate_count];
                            strncpy(candidate->device_path, device_path, sizeof(candidate->device_path) - 1);
                            strncpy(candidate->device_name, device_name, sizeof(candidate->device_name) - 1);
                            candidate->vendor_id = device_id.vendor;
                            candidate->product_id = device_id.product;
                            candidate->is_multitouch = test_bit(ABS_MT_POSITION_X, absbit);
                            candidate->is_direct = true;
                            
                            // 计算优先级
                            candidate->priority = 0;
                            
                            // Jetson Orin NX 相关设备优先级更高
                            if (strstr(device_name, "Goodix") || strstr(device_name, "goodix") ||
                                strstr(device_name, "ELAN") || strstr(device_name, "elan") ||
                                strstr(device_name, "Synaptics") || strstr(device_name, "synaptics")) {
                                candidate->priority += 100;
                            }
                            
                            // 多点触控设备优先级更高
                            if (candidate->is_multitouch) {
                                candidate->priority += 50;
                            }
                            
                            // 触摸屏关键词匹配
                            if (strstr(device_name, "Touch") || strstr(device_name, "touch") ||
                                strstr(device_name, "Touchscreen") || strstr(device_name, "touchscreen")) {
                                candidate->priority += 30;
                            }
                            
                            // event2 获得额外优先级
                            if (strcmp(device_path, "/dev/input/event2") == 0) {
                                candidate->priority += 200;
                            }
                            
                            if (debug_enabled) {
                                printf("发现触摸设备候选: %s (%s) 优先级: %d\n",
                                      device_name, device_path, candidate->priority);
                            }
                            
                            candidate_count++;
                        }
                    }
                }
                close(fd);
            }
        }
    }
    
    closedir(input_dir);
    
    // 选择优先级最高的设备
    if (candidate_count > 0) {
        touch_device_info_t* best_device = &candidates[0];
        for (int i = 1; i < candidate_count; i++) {
            if (candidates[i].priority > best_device->priority) {
                best_device = &candidates[i];
            }
        }
        
        // 复制最佳设备信息
        memcpy(&device_info, best_device, sizeof(touch_device_info_t));
        strncpy(touch_config.device_path, best_device->device_path, sizeof(touch_config.device_path) - 1);
        touch_config.device_path[sizeof(touch_config.device_path) - 1] = '\0';
        
        printf("选择触摸设备: %s (%s) 优先级: %d\n",
               best_device->device_name, best_device->device_path, best_device->priority);
        
        return true;
    }
    
    printf("未找到合适的触摸设备\n");
    return false;
}

// 测试触摸设备是否可用
bool touch_driver_test_device(const char* device_path) {
    int fd = open(device_path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        return false;
    }
    
    // 检查设备能力
    unsigned long evbit[NLONGS(EV_MAX)];
    unsigned long absbit[NLONGS(ABS_MAX)];
    
    bool is_touch_device = false;
    
    if (ioctl(fd, EVIOCGBIT(0, EV_MAX), evbit) >= 0 &&
        ioctl(fd, EVIOCGBIT(EV_ABS, ABS_MAX), absbit) >= 0) {
        
        // 检查是否支持绝对坐标和触摸事件
        if (test_bit(EV_ABS, evbit) &&
            test_bit(ABS_X, absbit) &&
            test_bit(ABS_Y, absbit)) {
            is_touch_device = true;
        }
    }
    
    close(fd);
    return is_touch_device;
}

// 配置触摸设备
bool touch_device_configure() {
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
            if (debug_enabled) {
                printf("X轴范围: %d - %d\n", abs_info.minimum, abs_info.maximum);
            }
        }
        
        // 获取Y轴信息
        if (ioctl(touch_fd, EVIOCGABS(ABS_Y), &abs_info) >= 0) {
            touch_config.max_y = abs_info.maximum;
            if (debug_enabled) {
                printf("Y轴范围: %d - %d\n", abs_info.minimum, abs_info.maximum);
            }
        }
        
        // 检查多点触控支持
        unsigned long absbit[NLONGS(ABS_MAX)];
        if (ioctl(touch_fd, EVIOCGBIT(EV_ABS, ABS_MAX), absbit) >= 0) {
            device_info.is_multitouch = test_bit(ABS_MT_POSITION_X, absbit) &&
                                       test_bit(ABS_MT_POSITION_Y, absbit);
            
            if (device_info.is_multitouch && debug_enabled) {
                printf("设备支持多点触控\n");
            }
        }
        
        return true;
    }
    
    return false;
}

// Jetson Orin NX 平台优化
bool jetson_orin_nx_optimize() {
    printf("应用 Jetson Orin NX 优化配置...\n");
    
    // 应用优化的触摸配置
    memcpy(&touch_config, &jetson_orin_nx_config, sizeof(touch_driver_config_t));
    
    // 设置GPU频率（如果权限允许）
    jetson_orin_nx_set_gpu_frequency();
    
    // 设置CPU调度策略
    jetson_orin_nx_set_cpu_governor();
    
    return true;
}

void jetson_orin_nx_set_gpu_frequency() {
    // 尝试设置GPU最高频率以获得最佳触摸响应
    int ret = system("echo 1 > /sys/class/devfreq/17000000.ga10b/governor 2>/dev/null");
    if (ret == 0 && debug_enabled) {
        printf("GPU频率优化已应用\n");
    }
}

void jetson_orin_nx_set_cpu_governor() {
    // 设置CPU性能模式以获得更好的触摸响应
    int ret = system("echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null");
    if (ret == 0 && debug_enabled) {
        printf("CPU性能模式已应用\n");
    }
}

// 屏幕监控初始化
bool screen_monitor_init() {
    printf("初始化屏幕监控...\n");
    
    // 获取framebuffer信息
    int fb_fd = open("/dev/fb0", O_RDONLY);
    if (fb_fd < 0) {
        printf("无法打开framebuffer设备\n");
        return false;
    }
    
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) == 0 &&
        ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo) == 0) {
        
        screen_info.width = vinfo.xres;
        screen_info.height = vinfo.yres;
        screen_info.bits_per_pixel = vinfo.bits_per_pixel;
        screen_info.line_length = finfo.line_length;
        screen_info.has_changed = false;
        
        // 更新触摸配置中的屏幕尺寸
        touch_config.screen_width = screen_info.width;
        touch_config.screen_height = screen_info.height;
        
        printf("屏幕信息: %dx%d, %d bpp\n",
               screen_info.width, screen_info.height, screen_info.bits_per_pixel);
    }
    
    close(fb_fd);
    
    // 启动屏幕监控线程
    screen_monitor_running = true;
    if (pthread_create(&screen_monitor_thread, NULL, screen_monitor_worker, NULL) != 0) {
        printf("无法创建屏幕监控线程\n");
        screen_monitor_running = false;
        return false;
    }
    
    return true;
}

// 屏幕监控工作线程
void* screen_monitor_worker(void* arg) {
    printf("屏幕监控线程启动\n");
    
    while (screen_monitor_running) {
        pthread_mutex_lock(&screen_mutex);
        
        // 检查屏幕配置变化
        int fb_fd = open("/dev/fb0", O_RDONLY);
        if (fb_fd >= 0) {
            struct fb_var_screeninfo vinfo;
            if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) == 0) {
                if (vinfo.xres != screen_info.width ||
                    vinfo.yres != screen_info.height ||
                    vinfo.bits_per_pixel != screen_info.bits_per_pixel) {
                    
                    screen_info.width = vinfo.xres;
                    screen_info.height = vinfo.yres;
                    screen_info.bits_per_pixel = vinfo.bits_per_pixel;
                    screen_info.has_changed = true;
                    
                    // 更新触摸配置
                    touch_config.screen_width = screen_info.width;
                    touch_config.screen_height = screen_info.height;
                    
                    printf("检测到屏幕配置变化: %dx%d, %d bpp\n",
                           screen_info.width, screen_info.height, screen_info.bits_per_pixel);
                }
            }
            close(fb_fd);
        }
        
        pthread_mutex_unlock(&screen_mutex);
        
        // 每秒检查一次
        sleep(1);
    }
    
    printf("屏幕监控线程退出\n");
    return NULL;
}

void screen_monitor_deinit() {
    if (screen_monitor_running) {
        screen_monitor_running = false;
        if (pthread_join(screen_monitor_thread, NULL) != 0) {
            printf("等待屏幕监控线程退出失败\n");
        }
    }
}

bool screen_monitor_check_changes() {
    pthread_mutex_lock(&screen_mutex);
    bool changed = screen_info.has_changed;
    screen_info.has_changed = false;
    pthread_mutex_unlock(&screen_mutex);
    return changed;
}

screen_info_t* screen_monitor_get_info() {
    return &screen_info;
}

// 多点触控数据处理
void multitouch_process_event(struct input_event* ev) {
    static int current_slot = 0;
    static int16_t mt_x = 0, mt_y = 0;
    static bool mt_tracking = false;
    
    switch (ev->code) {
        case ABS_MT_SLOT:
            current_slot = ev->value;
            if (current_slot >= 0 && current_slot < MAX_TOUCH_POINTS) {
                mt_tracking = true;
            }
            break;
            
        case ABS_MT_POSITION_X:
            if (mt_tracking && current_slot < MAX_TOUCH_POINTS) {
                mt_x = ev->value;
            }
            break;
            
        case ABS_MT_POSITION_Y:
            if (mt_tracking && current_slot < MAX_TOUCH_POINTS) {
                mt_y = ev->value;
            }
            break;
            
        case ABS_MT_TRACKING_ID:
            if (mt_tracking && current_slot < MAX_TOUCH_POINTS) {
                if (ev->value == -1) {
                    // 手指抬起
                    multitouch_data.points[current_slot].pressed = false;
                    multitouch_data.points[current_slot].finger_id = 0;
                } else {
                    // 手指按下
                    multitouch_data.points[current_slot].pressed = true;
                    multitouch_data.points[current_slot].finger_id = ev->value;
                    multitouch_data.points[current_slot].x = mt_x;
                    multitouch_data.points[current_slot].y = mt_y;
                }
            }
            break;
    }
}

// 坐标转换函数
void transform_coordinates(int16_t raw_x, int16_t raw_y, int16_t* screen_x, int16_t* screen_y) {
    // 基本缩放
    float scaled_x = (float)raw_x;
    float scaled_y = (float)raw_y;
    
    if (touch_config.max_x > 0) {
        scaled_x = (scaled_x * touch_config.screen_width) / touch_config.max_x;
    }
    if (touch_config.max_y > 0) {
        scaled_y = (scaled_y * touch_config.screen_height) / touch_config.max_y;
    }
    
    // 应用缩放因子
    scaled_x *= touch_config.scale_x;
    scaled_y *= touch_config.scale_y;
    
    // 应用偏移
    scaled_x += touch_config.offset_x;
    scaled_y += touch_config.offset_y;
    
    // 转换为整数坐标
    int16_t final_x = (int16_t)scaled_x;
    int16_t final_y = (int16_t)scaled_y;
    
    // 应用几何变换
    if (touch_config.swap_xy) {
        int16_t temp = final_x;
        final_x = final_y;
        final_y = temp;
    }
    
    if (touch_config.invert_x) {
        final_x = touch_config.screen_width - final_x;
    }
    
    if (touch_config.invert_y) {
        final_y = touch_config.screen_height - final_y;
    }
    
    // 边界检查
    if (final_x < 0) final_x = 0;
    if (final_y < 0) final_y = 0;
    if (final_x >= touch_config.screen_width) final_x = touch_config.screen_width - 1;
    if (final_y >= touch_config.screen_height) final_y = touch_config.screen_height - 1;
    
    *screen_x = final_x;
    *screen_y = final_y;
}

// 改进的触摸事件读取线程
void* touch_read_thread(void* arg) {
    struct input_event ev;
    int16_t raw_x = 0, raw_y = 0;
    int16_t pressure = 0;
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
                    } else if (ev.code == ABS_PRESSURE) {
                        pressure = ev.value;
                    } else if (touch_config.enable_multitouch) {
                        // 处理多点触控事件
                        multitouch_process_event(&ev);
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
                        int16_t screen_x, screen_y;
                        transform_coordinates(raw_x, raw_y, &screen_x, &screen_y);
                        
                        // 更新触摸点数据
                        last_point.x = screen_x;
                        last_point.y = screen_y;
                        last_point.pressed = touch_down;
                        last_point.pressure = pressure;
                        last_point.timestamp = ev.time.tv_sec * 1000 + ev.time.tv_usec / 1000;
                        
                        // 更新多点触控时间戳
                        if (touch_config.enable_multitouch) {
                            multitouch_data.timestamp = last_point.timestamp;
                            
                            // 统计有效触摸点
                            uint8_t count = 0;
                            for (int i = 0; i < MAX_TOUCH_POINTS; i++) {
                                if (multitouch_data.points[i].pressed) {
                                    count++;
                                }
                            }
                            multitouch_data.point_count = count;
                        }
                        
                        // 调试输出（无论是否为debug模式都输出，用于诊断）
                        if (touch_down) {
                            printf("触摸事件: 原始坐标(%d, %d) -> 屏幕坐标(%d, %d) 压力: %d 按下: %s\n",
                                   raw_x, raw_y, screen_x, screen_y, pressure, touch_down ? "是" : "否");
                        }
                    }
                    break;
            }
            
            pthread_mutex_unlock(&touch_mutex);
        } else if (bytes < 0 && errno != EAGAIN) {
            if (debug_enabled) {
                printf("读取触摸数据错误: %s\n", strerror(errno));
            }
            usleep(10000); // 10ms
        } else {
            usleep(500); // 0.5ms - 更高的响应频率
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
    
    // 临时调试输出 - 检查LVGL是否接收到触摸数据
    static uint32_t last_print_time = 0;
    uint32_t current_time = last_point.timestamp;
    if (last_point.pressed && (current_time - last_print_time > 500)) { // 每500ms打印一次
        printf("LVGL触摸回调: (%d, %d) 状态: %s\\n",
               data->point.x, data->point.y,
               data->state == LV_INDEV_STATE_PRESSED ? "按下" : "释放");
        last_print_time = current_time;
    }
    
    pthread_mutex_unlock(&touch_mutex);
}

// 获取触摸设备路径
const char* touch_device_get_path() {
    return touch_config.device_path;
}

// 获取触摸设备信息
touch_device_info_t* touch_device_get_info() {
    return &device_info;
}

// 设置触摸配置
void touch_driver_set_config(const touch_driver_config_t* config) {
    if (config) {
        pthread_mutex_lock(&touch_mutex);
        memcpy(&touch_config, config, sizeof(touch_driver_config_t));
        pthread_mutex_unlock(&touch_mutex);
    }
}

// 检查触摸驱动是否可用
bool touch_driver_is_available() {
    return driver_initialized && touch_fd >= 0;
}

// 多点触控功能
bool multitouch_is_enabled() {
    return touch_config.enable_multitouch && device_info.is_multitouch;
}

multi_touch_data_t* multitouch_get_data() {
    return &multitouch_data;
}

// 调试功能
void touch_driver_enable_debug(bool enable) {
    debug_enabled = enable;
    if (debug_enabled) {
        printf("触摸驱动调试模式已启用\n");
    }
}

void touch_driver_print_info() {
    printf("=== 触摸驱动信息 ===\n");
    printf("设备路径: %s\n", touch_config.device_path);
    printf("设备名称: %s\n", device_info.device_name);
    printf("厂商ID: 0x%04X\n", device_info.vendor_id);
    printf("产品ID: 0x%04X\n", device_info.product_id);
    printf("多点触控: %s\n", device_info.is_multitouch ? "支持" : "不支持");
    printf("触摸范围: %dx%d\n", touch_config.max_x, touch_config.max_y);
    printf("屏幕尺寸: %dx%d\n", touch_config.screen_width, touch_config.screen_height);
    printf("缩放因子: %.2fx%.2f\n", touch_config.scale_x, touch_config.scale_y);
    printf("偏移: (%d, %d)\n", touch_config.offset_x, touch_config.offset_y);
    printf("变换: 交换XY=%s, 反转X=%s, 反转Y=%s\n",
           touch_config.swap_xy ? "是" : "否",
           touch_config.invert_x ? "是" : "否",
           touch_config.invert_y ? "是" : "否");
    printf("====================\n");
}

// 初始化触摸驱动
bool touch_driver_init() {
    printf("初始化智能触摸驱动 (Jetson Orin NX 优化版)\n");
    
    if (driver_initialized) {
        printf("触摸驱动已初始化\n");
        return true;
    }
    
    // 使用默认配置
    memcpy(&touch_config, &default_config, sizeof(touch_driver_config_t));
    
    // 应用 Jetson Orin NX 优化
    jetson_orin_nx_optimize();
    
    // 初始化屏幕监控
    if (!screen_monitor_init()) {
        printf("警告: 屏幕监控初始化失败\n");
    }
    
    // 智能检测触摸设备
    if (!touch_device_smart_detect()) {
        printf("未检测到触摸设备，使用默认路径: %s\n", touch_config.device_path);
    }
    
    // 打开触摸设备
    touch_fd = open(touch_config.device_path, O_RDONLY | O_NONBLOCK);
    if (touch_fd < 0) {
        printf("无法打开触摸设备 %s: %s\n", touch_config.device_path, strerror(errno));
        return false;
    }
    
    // 配置触摸设备
    if (!touch_device_configure()) {
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
    printf("触摸驱动初始化成功\n");
    
    // 打印详细信息
    if (debug_enabled) {
        touch_driver_print_info();
    } else {
        printf("设备路径: %s\n", touch_config.device_path);
        printf("分辨率: %dx%d -> %dx%d\n",
               touch_config.max_x, touch_config.max_y,
               touch_config.screen_width, touch_config.screen_height);
    }
    
    return true;
}

// 清理触摸驱动
void touch_driver_deinit() {
    printf("清理触摸驱动\n");
    
    if (thread_running) {
        thread_running = false;
        if (pthread_join(touch_thread, NULL) != 0) {
            printf("等待触摸线程退出失败\n");
        }
    }
    
    // 清理屏幕监控
    screen_monitor_deinit();
    
    if (touch_fd >= 0) {
        close(touch_fd);
        touch_fd = -1;
    }
    
    driver_initialized = false;
    printf("触摸驱动已清理\n");
}

// 触摸校准功能占位符
bool touch_calibration_start() {
    printf("开始触摸校准\n");
    // TODO: 实现触摸校准逻辑
    return true;
}

void touch_calibration_stop() {
    printf("停止触摸校准\n");
    // TODO: 实现触摸校准停止逻辑
}

bool touch_calibration_is_active() {
    // TODO: 实现校准状态检查
    return false;
}

void touch_calibration_process_point(int16_t x, int16_t y) {
    // TODO: 实现校准点处理
    if (debug_enabled) {
        printf("校准点: (%d, %d)\n", x, y);
    }
}