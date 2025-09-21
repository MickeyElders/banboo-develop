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
} touch_point_t;

// 触摸设备配置 (使用不同的名称避免与common/types.h冲突)
typedef struct {
    const char* device_path;     // 设备路径，如 /dev/input/event0
    int16_t max_x;              // 最大X坐标
    int16_t max_y;              // 最大Y坐标
    int16_t screen_width;       // 屏幕宽度
    int16_t screen_height;      // 屏幕高度
    bool swap_xy;               // 是否交换X/Y坐标
    bool invert_x;              // 是否反转X坐标
    bool invert_y;              // 是否反转Y坐标
} touch_driver_config_t;

// 触摸驱动函数
bool touch_driver_init();
void touch_driver_deinit();
void touch_driver_read(lv_indev_drv_t* indev_drv, lv_indev_data_t* data);
void touch_driver_set_config(const touch_driver_config_t* config);
bool touch_driver_is_available();

// USB触摸屏检测和配置
bool usb_touch_detect();
bool usb_touch_configure();
const char* usb_touch_get_device_path();

#endif