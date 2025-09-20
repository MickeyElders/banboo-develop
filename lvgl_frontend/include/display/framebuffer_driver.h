#ifndef DISPLAY_FRAMEBUFFER_DRIVER_H
#define DISPLAY_FRAMEBUFFER_DRIVER_H

#include <linux/fb.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 基础初始化和清理函数
bool framebuffer_driver_init();
void framebuffer_driver_deinit();

// 获取framebuffer信息
struct fb_var_screeninfo* get_fb_vinfo();
struct fb_fix_screeninfo* get_fb_finfo();
char* get_fb_buffer();

// LVGL缓冲区管理
uint32_t* get_lvgl_buffer();
uint32_t* get_lvgl_buffer2();

// 显示刷新函数
void framebuffer_flush(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, const uint32_t* color_map);

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_FRAMEBUFFER_DRIVER_H