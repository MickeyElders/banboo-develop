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

// 摄像头系统函数
void init_camera_system();
void deinit_camera_system();
void update_camera_display();
bool is_camera_running();
double get_camera_fps();

// 后端系统函数
void init_backend_system();
void deinit_backend_system();
void update_system_status_display();
void* get_backend_client(); // 使用void*避免类型依赖

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_LVGL_DISPLAY_H