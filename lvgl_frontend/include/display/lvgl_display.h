#ifndef DISPLAY_LVGL_DISPLAY_H
#define DISPLAY_LVGL_DISPLAY_H

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif

// LVGL显示驱动初始化和清理
bool lvgl_display_init();
void lvgl_display_deinit();

// LVGL显示刷新回调函数
void lvgl_disp_flush(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_p);

// 测试UI创建函数
void create_test_ui();

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_LVGL_DISPLAY_H