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

// 创建智能切竹机工业控制界面
void create_test_ui() {
    printf("创建智能切竹机工业控制界面...\n");
    
    // 获取当前屏幕
    lv_obj_t *scr = lv_scr_act();
    
    // 设置工业级深蓝色背景
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x1e3a5f), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);
    
    // ========== 顶部状态栏 ==========
    lv_obj_t *header = lv_obj_create(scr);
    lv_obj_set_size(header, LV_PCT(100), 60);
    lv_obj_align(header, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(header, lv_color_hex(0x2c5282), LV_PART_MAIN);
    lv_obj_set_style_border_width(header, 0, LV_PART_MAIN);
    
    // 系统标题
    lv_obj_t *title_label = lv_label_create(header);
    lv_label_set_text(title_label, "智能切竹机系统 v2.0");
    lv_obj_set_style_text_font(title_label, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(title_label, lv_color_white(), 0);
    lv_obj_align(title_label, LV_ALIGN_LEFT_MID, 20, 0);
    
    // 系统状态指示灯
    lv_obj_t *status_led = lv_led_create(header);
    lv_led_set_color(status_led, lv_color_hex(0x00FF00));
    lv_led_on(status_led);
    lv_obj_align(status_led, LV_ALIGN_RIGHT_MID, -80, 0);
    
    // 状态文本
    lv_obj_t *status_text = lv_label_create(header);
    lv_label_set_text(status_text, "系统就绪");
    lv_obj_set_style_text_color(status_text, lv_color_white(), 0);
    lv_obj_align(status_text, LV_ALIGN_RIGHT_MID, -20, 0);
    
    // ========== 左侧控制面板 ==========
    lv_obj_t *control_panel = lv_obj_create(scr);
    lv_obj_set_size(control_panel, 300, LV_PCT(70));
    lv_obj_align(control_panel, LV_ALIGN_LEFT_MID, 10, 10);
    lv_obj_set_style_bg_color(control_panel, lv_color_hex(0x2d3748), LV_PART_MAIN);
    lv_obj_set_style_border_color(control_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(control_panel, 2, LV_PART_MAIN);
    
    // 控制面板标题
    lv_obj_t *panel_title = lv_label_create(control_panel);
    lv_label_set_text(panel_title, "控制面板");
    lv_obj_set_style_text_color(panel_title, lv_color_hex(0xE2E8F0), 0);
    lv_obj_align(panel_title, LV_ALIGN_TOP_MID, 0, 15);
    
    // AI检测状态
    lv_obj_t *ai_status = lv_label_create(control_panel);
    lv_label_set_text(ai_status, "AI推理引擎: TensorRT");
    lv_obj_set_style_text_color(ai_status, lv_color_hex(0x68D391), 0);
    lv_obj_align(ai_status, LV_ALIGN_TOP_LEFT, 20, 50);
    
    // 摄像头状态
    lv_obj_t *camera_status = lv_label_create(control_panel);
    lv_label_set_text(camera_status, "摄像头: 1920x1080@30fps");
    lv_obj_set_style_text_color(camera_status, lv_color_hex(0x68D391), 0);
    lv_obj_align(camera_status, LV_ALIGN_TOP_LEFT, 20, 80);
    
    // PLC通信状态
    lv_obj_t *plc_status = lv_label_create(control_panel);
    lv_label_set_text(plc_status, "PLC通信: Modbus TCP");
    lv_obj_set_style_text_color(plc_status, lv_color_hex(0x68D391), 0);
    lv_obj_align(plc_status, LV_ALIGN_TOP_LEFT, 20, 110);
    
    // 主要控制按钮
    lv_obj_t *start_btn = lv_btn_create(control_panel);
    lv_obj_set_size(start_btn, 250, 50);
    lv_obj_align(start_btn, LV_ALIGN_TOP_MID, 0, 150);
    lv_obj_set_style_bg_color(start_btn, lv_color_hex(0x38A169), LV_PART_MAIN);
    
    lv_obj_t *start_label = lv_label_create(start_btn);
    lv_label_set_text(start_label, "启动AI检测");
    lv_obj_set_style_text_color(start_label, lv_color_white(), 0);
    lv_obj_center(start_label);
    
    // 停止按钮
    lv_obj_t *stop_btn = lv_btn_create(control_panel);
    lv_obj_set_size(stop_btn, 250, 50);
    lv_obj_align(stop_btn, LV_ALIGN_TOP_MID, 0, 220);
    lv_obj_set_style_bg_color(stop_btn, lv_color_hex(0xE53E3E), LV_PART_MAIN);
    
    lv_obj_t *stop_label = lv_label_create(stop_btn);
    lv_label_set_text(stop_label, "停止检测");
    lv_obj_set_style_text_color(stop_label, lv_color_white(), 0);
    lv_obj_center(stop_label);
    
    // ========== 右侧视频显示区域 ==========
    lv_obj_t *video_panel = lv_obj_create(scr);
    lv_obj_set_size(video_panel, LV_PCT(60), LV_PCT(70));
    lv_obj_align(video_panel, LV_ALIGN_RIGHT_MID, -10, 10);
    lv_obj_set_style_bg_color(video_panel, lv_color_hex(0x000000), LV_PART_MAIN);
    lv_obj_set_style_border_color(video_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(video_panel, 2, LV_PART_MAIN);
    
    // 视频区域标题
    lv_obj_t *video_title = lv_label_create(video_panel);
    lv_label_set_text(video_title, "实时视频监控");
    lv_obj_set_style_text_color(video_title, lv_color_white(), 0);
    lv_obj_align(video_title, LV_ALIGN_TOP_MID, 0, 15);
    
    // 视频占位符
    lv_obj_t *video_placeholder = lv_label_create(video_panel);
    lv_label_set_text(video_placeholder, "摄像头初始化中...\n等待视频流");
    lv_obj_set_style_text_color(video_placeholder, lv_color_hex(0x888888), 0);
    lv_obj_set_style_text_align(video_placeholder, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_center(video_placeholder);
    
    // ========== 底部数据显示区域 ==========
    lv_obj_t *data_panel = lv_obj_create(scr);
    lv_obj_set_size(data_panel, LV_PCT(95), 120);
    lv_obj_align(data_panel, LV_ALIGN_BOTTOM_MID, 0, -10);
    lv_obj_set_style_bg_color(data_panel, lv_color_hex(0x2d3748), LV_PART_MAIN);
    lv_obj_set_style_border_color(data_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(data_panel, 2, LV_PART_MAIN);
    
    // 数据面板标题
    lv_obj_t *data_title = lv_label_create(data_panel);
    lv_label_set_text(data_title, "检测数据");
    lv_obj_set_style_text_color(data_title, lv_color_hex(0xE2E8F0), 0);
    lv_obj_align(data_title, LV_ALIGN_TOP_LEFT, 20, 10);
    
    // 检测结果显示
    lv_obj_t *detection_data = lv_label_create(data_panel);
    lv_label_set_text(detection_data, "切点数量: 0 | 坐标数据: 等待检测结果...");
    lv_obj_set_style_text_color(detection_data, lv_color_hex(0xA0AEC0), 0);
    lv_obj_align(detection_data, LV_ALIGN_TOP_LEFT, 20, 40);
    
    // PLC通信数据
    lv_obj_t *plc_data = lv_label_create(data_panel);
    lv_label_set_text(plc_data, "PLC连接: 192.168.1.10:502 | 心跳: 正常 | 寄存器状态: 就绪");
    lv_obj_set_style_text_color(plc_data, lv_color_hex(0xA0AEC0), 0);
    lv_obj_align(plc_data, LV_ALIGN_TOP_LEFT, 20, 70);
    
    // 性能指标
    lv_obj_t *perf_data = lv_label_create(data_panel);
    lv_label_set_text(perf_data, "推理性能: 30 FPS | 延迟: <33ms | CPU负载: 25%");
    lv_obj_set_style_text_color(perf_data, lv_color_hex(0x68D391), 0);
    lv_obj_align(perf_data, LV_ALIGN_BOTTOM_LEFT, 20, -15);
    
    printf("智能切竹机工业控制界面创建完成\n");
}