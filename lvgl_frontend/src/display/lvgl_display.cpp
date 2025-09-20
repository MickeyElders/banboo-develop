#include "display/lvgl_display.h"
#include "display/framebuffer_driver.h"
#include "lvgl.h"
#include <stdio.h>
#include <string.h>

// LVGL显示驱动变量
static lv_disp_drv_t disp_drv;
static lv_disp_t *disp;
static lv_disp_draw_buf_t disp_buf;

// 显示刷新回调函数
void lvgl_disp_flush(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_p) {
    printf("LVGL刷新显示区域: (%d,%d) -> (%d,%d)\n",
           area->x1, area->y1, area->x2, area->y2);
    
    // 获取framebuffer信息
    struct fb_var_screeninfo *vinfo = get_fb_vinfo();
    if (!vinfo) {
        printf("错误: 无法获取framebuffer信息\n");
        lv_disp_flush_ready(drv);
        return;
    }
    
    // 将LVGL颜色数据转换为32位ARGB格式并刷新到framebuffer
    uint32_t *fb_data = get_lvgl_buffer();
    if (!fb_data) {
        printf("错误: 无法获取LVGL缓冲区\n");
        lv_disp_flush_ready(drv);
        return;
    }
    
    // 复制颜色数据
    int32_t x, y;
    lv_color_t *src = color_p;
    
    for (y = area->y1; y <= area->y2; y++) {
        for (x = area->x1; x <= area->x2; x++) {
            // 将LVGL颜色转换为32位ARGB
            uint32_t color = lv_color_to32(*src);
            fb_data[y * vinfo->xres + x] = color;
            src++;
        }
    }
    
    // 刷新到实际framebuffer
    framebuffer_flush(area->x1, area->y1, area->x2, area->y2, fb_data);
    
    // 通知LVGL刷新完成
    lv_disp_flush_ready(drv);
}

bool lvgl_display_init() {
    printf("初始化 LVGL显示驱动...\n");
    
    // 首先初始化framebuffer驱动
    if (!framebuffer_driver_init()) {
        printf("错误: Framebuffer驱动初始化失败\n");
        return false;
    }
    
    // 获取framebuffer信息
    struct fb_var_screeninfo *vinfo = get_fb_vinfo();
    if (!vinfo) {
        printf("错误: 无法获取framebuffer信息\n");
        return false;
    }
    
    printf("配置LVGL显示驱动:\n");
    printf("  分辨率: %dx%d\n", vinfo->xres, vinfo->yres);
    printf("  色深: %d位\n", vinfo->bits_per_pixel);
    
    // 获取LVGL缓冲区
    uint32_t *buf1 = get_lvgl_buffer();
    uint32_t *buf2 = get_lvgl_buffer2();
    
    if (!buf1 || !buf2) {
        printf("错误: 无法获取LVGL缓冲区\n");
        framebuffer_driver_deinit();
        return false;
    }
    
    // 初始化LVGL显示缓冲区
    lv_disp_draw_buf_init(&disp_buf, buf1, buf2, vinfo->xres * vinfo->yres);
    
    // 初始化显示驱动
    lv_disp_drv_init(&disp_drv);
    disp_drv.draw_buf = &disp_buf;
    disp_drv.flush_cb = lvgl_disp_flush;
    disp_drv.hor_res = vinfo->xres;
    disp_drv.ver_res = vinfo->yres;
    
    // 注册显示驱动
    disp = lv_disp_drv_register(&disp_drv);
    if (!disp) {
        printf("错误: LVGL显示驱动注册失败\n");
        framebuffer_driver_deinit();
        return false;
    }
    
    // 创建测试界面
    create_test_ui();
    
    printf("LVGL显示驱动初始化成功\n");
    return true;
}

void lvgl_display_deinit() {
    printf("清理 LVGL显示驱动\n");
    
    if (disp) {
        lv_obj_clean(lv_scr_act());
    }
    
    framebuffer_driver_deinit();
}

// 创建测试界面
void create_test_ui() {
    printf("创建测试UI界面...\n");
    
    // 获取当前屏幕
    lv_obj_t *scr = lv_scr_act();
    
    // 设置背景颜色
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x003366), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);
    
    // 创建标题标签
    lv_obj_t *title_label = lv_label_create(scr);
    lv_label_set_text(title_label, "智能切竹机控制系统");
    lv_obj_set_style_text_font(title_label, &lv_font_montserrat_24, 0);
    lv_obj_set_style_text_color(title_label, lv_color_white(), 0);
    lv_obj_align(title_label, LV_ALIGN_TOP_MID, 0, 20);
    
    // 创建版本标签
    lv_obj_t *version_label = lv_label_create(scr);
    lv_label_set_text(version_label, "LVGL版本 2.0.0");
    lv_obj_set_style_text_color(version_label, lv_color_hex(0xCCCCCC), 0);
    lv_obj_align(version_label, LV_ALIGN_TOP_MID, 0, 60);
    
    // 创建状态指示器
    lv_obj_t *status_label = lv_label_create(scr);
    lv_label_set_text(status_label, "系统状态: 正常运行");
    lv_obj_set_style_text_color(status_label, lv_color_hex(0x00FF00), 0);
    lv_obj_align(status_label, LV_ALIGN_CENTER, 0, -50);
    
    // 创建按钮
    lv_obj_t *btn = lv_btn_create(scr);
    lv_obj_set_size(btn, 200, 50);
    lv_obj_align(btn, LV_ALIGN_CENTER, 0, 20);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x0066CC), LV_PART_MAIN);
    
    lv_obj_t *btn_label = lv_label_create(btn);
    lv_label_set_text(btn_label, "启动检测");
    lv_obj_set_style_text_color(btn_label, lv_color_white(), 0);
    lv_obj_center(btn_label);
    
    // 创建底部信息
    lv_obj_t *info_label = lv_label_create(scr);
    lv_label_set_text(info_label, "Jetson Orin NX - LVGL驱动测试");
    lv_obj_set_style_text_color(info_label, lv_color_hex(0x999999), 0);
    lv_obj_align(info_label, LV_ALIGN_BOTTOM_MID, 0, -20);
    
    printf("测试UI界面创建完成\n");
}