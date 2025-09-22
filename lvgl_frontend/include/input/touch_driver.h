#ifndef INPUT_TOUCH_DRIVER_H
#define INPUT_TOUCH_DRIVER_H

#include "lvgl.h"
#include <stdbool.h>
#include <stdint.h>

// 触摸点数据结构
typedef struct {
    int16_t x;
    int16_t y;
    bool pressed;
    uint32_t timestamp;
    int16_t pressure;           // 压力值（如果支持）
    uint8_t finger_id;          // 手指ID（多点触控）
} touch_point_t;

// 多点触控支持
#define MAX_TOUCH_POINTS 10
typedef struct {
    touch_point_t points[MAX_TOUCH_POINTS];
    uint8_t point_count;
    uint32_t timestamp;
} multi_touch_data_t;

// 触摸设备信息
typedef struct {
    char device_path[256];      // 设备路径
    char device_name[256];      // 设备名称
    uint16_t vendor_id;         // 厂商ID
    uint16_t product_id;        // 产品ID
    int priority;               // 优先级（越高越优先）
    bool is_multitouch;         // 是否支持多点触控
    bool is_direct;             // 是否为直接触控设备
} touch_device_info_t;

// 触摸设备配置 (使用不同的名称避免与common/types.h冲突)
typedef struct {
    char device_path[256];      // 设备路径，如 /dev/input/event2
    int16_t max_x;              // 最大X坐标
    int16_t max_y;              // 最大Y坐标
    int16_t screen_width;       // 屏幕宽度
    int16_t screen_height;      // 屏幕高度
    bool swap_xy;               // 是否交换X/Y坐标
    bool invert_x;              // 是否反转X坐标
    bool invert_y;              // 是否反转Y坐标
    float scale_x;              // X轴缩放因子
    float scale_y;              // Y轴缩放因子
    int16_t offset_x;           // X轴偏移
    int16_t offset_y;           // Y轴偏移
    bool enable_multitouch;     // 启用多点触控
    uint8_t touch_threshold;    // 触摸阈值
} touch_driver_config_t;

// 屏幕信息结构
typedef struct {
    uint16_t width;
    uint16_t height;
    uint8_t bits_per_pixel;
    uint32_t line_length;
    bool has_changed;
} screen_info_t;

// 触摸驱动函数
bool touch_driver_init();
void touch_driver_deinit();
void touch_driver_read(lv_indev_drv_t* indev_drv, lv_indev_data_t* data);
void touch_driver_set_config(const touch_driver_config_t* config);
bool touch_driver_is_available();

// 设备检测和配置
bool touch_device_smart_detect();
bool touch_device_configure();
const char* touch_device_get_path();
touch_device_info_t* touch_device_get_info();

// Jetson Orin NX 优化
bool jetson_orin_nx_optimize();
void jetson_orin_nx_set_gpu_frequency();
void jetson_orin_nx_set_cpu_governor();

// 屏幕监控
bool screen_monitor_init();
void screen_monitor_deinit();
bool screen_monitor_check_changes();
screen_info_t* screen_monitor_get_info();

// 多点触控支持
bool multitouch_is_enabled();
multi_touch_data_t* multitouch_get_data();
void multitouch_process_event(struct input_event* ev);

// 触摸校准
bool touch_calibration_start();
void touch_calibration_stop();
bool touch_calibration_is_active();
void touch_calibration_process_point(int16_t x, int16_t y);

// 调试和诊断
void touch_driver_print_info();
void touch_driver_enable_debug(bool enable);
bool touch_driver_test_device(const char* device_path);

#endif