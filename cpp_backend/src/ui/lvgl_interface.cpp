/**
 * @file lvgl_interface.cpp
 * @brief C++ LVGL一体化系统界面管理器实现
 * 工业级竹子识别系统LVGL界面 - 现代科技风格版本
 * 适配DRM渲染后端
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>
#include <cerrno>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#endif

namespace bamboo_cut {
namespace ui {

// 静态成员初始化
#ifdef ENABLE_LVGL
lv_color_t* LVGLInterface::disp_buf1_ = nullptr;
lv_color_t* LVGLInterface::disp_buf2_ = nullptr;
lv_draw_buf_t LVGLInterface::draw_buf_;
#else
void* LVGLInterface::disp_buf1_ = nullptr;
void* LVGLInterface::disp_buf2_ = nullptr;
char LVGLInterface::draw_buf_[64] = {0};
#endif

// 全局样式变量
#ifdef ENABLE_LVGL
static lv_style_t style_card;
static lv_style_t style_btn_primary;
static lv_style_t style_btn_success;
static lv_style_t style_btn_warning;
static lv_style_t style_btn_danger;
static lv_style_t style_btn_pressed;
static lv_style_t style_text_title;
static lv_style_t style_text_body;
static lv_style_t style_text_small;
static lv_style_t style_table_header;
static lv_style_t style_table_cell;
#else
// LVGL未启用时的占位符
static char style_card[64];
static char style_btn_primary[64];
static char style_btn_success[64];
static char style_btn_warning[64];
static char style_btn_danger[64];
static char style_btn_pressed[64];
static char style_text_title[64];
static char style_text_body[64];
static char style_text_small[64];
static char style_table_header[64];
static char style_table_cell[64];
#endif

LVGLInterface::LVGLInterface(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(data_bridge)
    , main_screen_(nullptr)
    , header_panel_(nullptr)
    , camera_panel_(nullptr)
    , camera_canvas_(nullptr)
    , control_panel_(nullptr)
    , status_panel_(nullptr)
    , footer_panel_(nullptr)
    , display_(nullptr)
    , input_device_(nullptr)
    , frame_count_(0)
    , ui_fps_(0.0f)
    , current_step_(WorkflowStep::FEED_DETECTION)
    , system_running_(false)
    , emergency_stop_(false)
    , selected_blade_(1)
{
    std::cout << "[LVGLInterface] 构造函数调用" << std::endl;
}

LVGLInterface::~LVGLInterface() {
    std::cout << "[LVGLInterface] 析构函数调用" << std::endl;
    stop();
    
#ifdef ENABLE_LVGL
    if (disp_buf1_) {
        delete[] disp_buf1_;
        disp_buf1_ = nullptr;
    }
    if (disp_buf2_) {
        delete[] disp_buf2_;
        disp_buf2_ = nullptr;
    }
#endif
}

bool LVGLInterface::initialize(const LVGLConfig& config) {
    std::cout << "[LVGLInterface] 初始化界面系统..." << std::endl;
    config_ = config;
    
    // 自动检测显示器分辨率
    int detected_width, detected_height;
    if (detectDisplayResolution(detected_width, detected_height)) {
        std::cout << "[LVGLInterface] 检测到显示器分辨率: "
                  << detected_width << "x" << detected_height << std::endl;
        config_.screen_width = detected_width;
        config_.screen_height = detected_height;
    } else {
        std::cout << "[LVGLInterface] 无法检测显示器分辨率，使用默认值: "
                  << config_.screen_width << "x" << config_.screen_height << std::endl;
    }
    
#ifdef ENABLE_LVGL
    // 初始化LVGL
    lv_init();
    
    // 初始化DRM显示驱动
    if (!initializeDisplay()) {
        std::cerr << "[LVGLInterface] 显示驱动初始化失败" << std::endl;
        return false;
    }
    
    // 初始化输入设备
    if (config_.enable_touch && !initializeInput()) {
        std::cerr << "[LVGLInterface] 输入设备初始化失败" << std::endl;
    }
    
    // 初始化主题
    initializeTheme();
    
    // 创建主界面
    createMainInterface();
    
    std::cout << "[LVGLInterface] 界面系统初始化成功" << std::endl;
    return true;
#else
    std::cerr << "[LVGLInterface] LVGL未启用，无法初始化" << std::endl;
    return false;
#endif
}

bool LVGLInterface::initializeDisplay() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 初始化DRM显示驱动..." << std::endl;
    
    // 分配显示缓冲区
    uint32_t buf_size = config_.screen_width * config_.screen_height;
    disp_buf1_ = new lv_color_t[buf_size];
    disp_buf2_ = new lv_color_t[buf_size];
    
    if (!disp_buf1_ || !disp_buf2_) {
        std::cerr << "[LVGLInterface] 显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    // 初始化显示缓冲区 (LVGL v9 API) - 使用与DRM匹配的格式
    lv_draw_buf_init(&draw_buf_, config_.screen_width, config_.screen_height,
                     LV_COLOR_FORMAT_XRGB8888, config_.screen_width * 4,
                     disp_buf1_, buf_size * sizeof(lv_color_t));
    
    // 创建显示器
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        std::cerr << "[LVGLInterface] 显示器创建失败" << std::endl;
        return false;
    }
    
    // 设置显示缓冲区
    lv_display_set_buffers(display_, disp_buf1_, disp_buf2_, buf_size * sizeof(lv_color_t), LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置刷新回调函数
    lv_display_set_flush_cb(display_, display_flush_cb);
    
    std::cout << "[LVGLInterface] DRM显示驱动初始化成功 ("
              << config_.screen_width << "x" << config_.screen_height << ")" << std::endl;
    return true;
#else
    return false;
#endif
}

bool LVGLInterface::initializeInput() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 初始化触摸输入设备: " << config_.touch_device << std::endl;
    
    // 创建输入设备 (LVGL v9 API)
    input_device_ = lv_indev_create();
    if (!input_device_) {
        std::cerr << "[LVGLInterface] 输入设备创建失败" << std::endl;
        return false;
    }
    
    lv_indev_set_type(input_device_, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(input_device_, input_read_cb);
    
    std::cout << "[LVGLInterface] 触摸输入设备初始化成功" << std::endl;
    return true;
#else
    return false;
#endif
}

void LVGLInterface::initializeTheme() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 初始化现代舒适主题..." << std::endl;
    
    // 优化后的配色方案 - 更柔和，对比度适中
    color_background_ = lv_color_hex(0x1A1F26);   // 温和深色背景
    color_surface_    = lv_color_hex(0x252B35);   // 卡片表面
    color_primary_    = lv_color_hex(0x5B9BD5);   // 柔和蓝色主色
    color_secondary_  = lv_color_hex(0x70A5DB);   // 淡蓝色副色
    color_success_    = lv_color_hex(0x7FB069);   // 柔和绿色
    color_warning_    = lv_color_hex(0xE6A055);   // 温和橙色
    color_error_      = lv_color_hex(0xD67B7B);   // 柔和红色
    
    // 设置全局背景
    lv_obj_set_style_bg_color(lv_scr_act(), color_background_, 0);
    
    // ========== 卡片样式 ==========
    lv_style_init(&style_card);
    lv_style_set_bg_color(&style_card, color_surface_);
    lv_style_set_bg_opa(&style_card, LV_OPA_COVER);
    lv_style_set_radius(&style_card, 12);
    lv_style_set_border_width(&style_card, 2);
    lv_style_set_border_color(&style_card, lv_color_hex(0x2A3441));
    lv_style_set_border_opa(&style_card, LV_OPA_50);
    lv_style_set_shadow_width(&style_card, 20);
    lv_style_set_shadow_color(&style_card, lv_color_black());
    lv_style_set_shadow_opa(&style_card, LV_OPA_40);
    lv_style_set_shadow_spread(&style_card, 2);
    lv_style_set_pad_all(&style_card, 20);
    
    // ========== 优化后的按钮样式 ==========
    // 主按钮（柔和蓝色）
    lv_style_init(&style_btn_primary);
    lv_style_set_radius(&style_btn_primary, 12);
    lv_style_set_bg_color(&style_btn_primary, color_primary_);
    lv_style_set_bg_opa(&style_btn_primary, LV_OPA_COVER);
    lv_style_set_shadow_width(&style_btn_primary, 8);
    lv_style_set_shadow_opa(&style_btn_primary, LV_OPA_10);
    lv_style_set_shadow_color(&style_btn_primary, lv_color_black());
    lv_style_set_text_color(&style_btn_primary, lv_color_white());
    lv_style_set_border_width(&style_btn_primary, 0);
    lv_style_set_pad_all(&style_btn_primary, 16);
    
    // 成功按钮（柔和绿色）
    lv_style_init(&style_btn_success);
    lv_style_set_radius(&style_btn_success, 12);
    lv_style_set_bg_color(&style_btn_success, color_success_);
    lv_style_set_shadow_width(&style_btn_success, 8);
    lv_style_set_shadow_color(&style_btn_success, lv_color_black());
    lv_style_set_shadow_opa(&style_btn_success, LV_OPA_10);
    lv_style_set_text_color(&style_btn_success, lv_color_white());
    lv_style_set_border_width(&style_btn_success, 0);
    lv_style_set_pad_all(&style_btn_success, 16);
    
    // 警告按钮（温和橙色）
    lv_style_init(&style_btn_warning);
    lv_style_set_radius(&style_btn_warning, 12);
    lv_style_set_bg_color(&style_btn_warning, color_warning_);
    lv_style_set_shadow_width(&style_btn_warning, 8);
    lv_style_set_shadow_opa(&style_btn_warning, LV_OPA_10);
    lv_style_set_shadow_color(&style_btn_warning, lv_color_black());
    lv_style_set_text_color(&style_btn_warning, lv_color_white());
    lv_style_set_border_width(&style_btn_warning, 0);
    lv_style_set_pad_all(&style_btn_warning, 16);
    
    // 危险按钮（柔和红色）
    lv_style_init(&style_btn_danger);
    lv_style_set_radius(&style_btn_danger, 12);
    lv_style_set_bg_color(&style_btn_danger, color_error_);
    lv_style_set_shadow_width(&style_btn_danger, 10);
    lv_style_set_shadow_color(&style_btn_danger, lv_color_black());
    lv_style_set_shadow_opa(&style_btn_danger, LV_OPA_20);
    lv_style_set_text_color(&style_btn_danger, lv_color_white());
    lv_style_set_border_width(&style_btn_danger, 0);
    lv_style_set_pad_all(&style_btn_danger, 16);
    
    // 按钮按下效果（更温和）
    lv_style_init(&style_btn_pressed);
    lv_style_set_transform_width(&style_btn_pressed, -2);
    lv_style_set_transform_height(&style_btn_pressed, -2);
    lv_style_set_shadow_width(&style_btn_pressed, 4);
    
    // ========== 文字样式 ==========
    lv_style_init(&style_text_title);
    lv_style_set_text_color(&style_text_title, lv_color_white());
    lv_style_set_text_font(&style_text_title, &lv_font_montserrat_24);
    
    lv_style_init(&style_text_body);
    lv_style_set_text_color(&style_text_body, lv_color_hex(0xB0B8C1));
    lv_style_set_text_font(&style_text_body, &lv_font_montserrat_16);
    
    lv_style_init(&style_text_small);
    lv_style_set_text_color(&style_text_small, lv_color_hex(0x8A92A1));
    lv_style_set_text_font(&style_text_small, &lv_font_montserrat_14);
    
    // ========== 表格样式 ==========
    lv_style_init(&style_table_header);
    lv_style_set_bg_color(&style_table_header, color_primary_);
    lv_style_set_text_color(&style_table_header, lv_color_white());
    lv_style_set_text_font(&style_table_header, &lv_font_montserrat_14);
    lv_style_set_pad_all(&style_table_header, 10);
    lv_style_set_border_color(&style_table_header, lv_color_hex(0x2A3441));
    lv_style_set_border_width(&style_table_header, 1);
    
    lv_style_init(&style_table_cell);
    lv_style_set_bg_color(&style_table_cell, color_surface_);
    lv_style_set_text_color(&style_table_cell, lv_color_hex(0xB0B8C1));
    lv_style_set_text_font(&style_table_cell, &lv_font_montserrat_14);
    lv_style_set_border_color(&style_table_cell, lv_color_hex(0x2A3441));
    lv_style_set_border_width(&style_table_cell, 1);
    lv_style_set_pad_all(&style_table_cell, 8);
    
    std::cout << "[LVGLInterface] 主题初始化完成" << std::endl;
#endif
}

void LVGLInterface::createMainInterface() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 创建主界面..." << std::endl;
    
    main_screen_ = lv_scr_act();
    
    // 创建各个面板
    createHeaderPanel();
    createCameraPanel();
    createControlPanel();
    createStatusPanel();
    createFooterPanel();
    
    std::cout << "[LVGLInterface] 主界面创建完成" << std::endl;
#endif
}

lv_obj_t* LVGLInterface::createHeaderPanel() {
#ifdef ENABLE_LVGL
    header_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(header_panel_, lv_pct(100), 70);  // 使用百分比并减少高度
    lv_obj_align(header_panel_, LV_ALIGN_TOP_MID, 0, 0);
    
    // 简洁的背景样式
    lv_obj_set_style_bg_color(header_panel_, color_surface_, 0);
    lv_obj_set_style_bg_opa(header_panel_, LV_OPA_90, 0);
    lv_obj_set_style_radius(header_panel_, 0, 0);
    lv_obj_set_style_border_width(header_panel_, 1, 0);
    lv_obj_set_style_border_side(header_panel_, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(header_panel_, LV_OPA_50, 0);
    lv_obj_set_style_pad_all(header_panel_, 12, 0);
    lv_obj_clear_flag(header_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // Flex布局
    lv_obj_set_flex_flow(header_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(header_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // System Title
    header_widgets_.system_title = lv_label_create(header_panel_);
    lv_label_set_text(header_widgets_.system_title,
        LV_SYMBOL_IMAGE " Bamboo Intelligent Cutting");
    lv_obj_add_style(header_widgets_.system_title, &style_text_title, 0);
    lv_obj_set_style_text_color(header_widgets_.system_title, color_primary_, 0);
    
    // Compact workflow indicator container
    lv_obj_t* workflow_container = lv_obj_create(header_panel_);
    lv_obj_set_size(workflow_container, 350, 50);  // More compact
    lv_obj_set_style_bg_opa(workflow_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(workflow_container, 0, 0);
    lv_obj_set_style_pad_all(workflow_container, 8, 0);
    lv_obj_set_flex_flow(workflow_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(workflow_container, LV_FLEX_ALIGN_SPACE_AROUND,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_clear_flag(workflow_container, LV_OBJ_FLAG_SCROLLABLE);
    
    const char* steps[] = {"Feed", "Detect", "Transfer", "Prepare", "Cut"};
    const char* icons[] = {
        LV_SYMBOL_UPLOAD, LV_SYMBOL_EYE_OPEN, LV_SYMBOL_GPS,
        LV_SYMBOL_SETTINGS, LV_SYMBOL_CUT
    };
    
    for(int i = 0; i < 5; i++) {
        lv_obj_t* step = lv_obj_create(workflow_container);
        lv_obj_set_size(step, 40, 40);  // 更小更紧凑
        lv_obj_set_style_radius(step, 20, 0);
        lv_obj_set_style_border_width(step, 1, 0);
        lv_obj_set_style_pad_all(step, 0, 0);
        lv_obj_clear_flag(step, LV_OBJ_FLAG_SCROLLABLE);
        
        bool is_active = (i == (int)current_step_ - 1);
        bool is_completed = (i < (int)current_step_ - 1);
        
        if(is_active) {
            lv_obj_set_style_bg_color(step, color_primary_, 0);
            lv_obj_set_style_border_color(step, color_primary_, 0);
            lv_obj_set_style_shadow_width(step, 6, 0);
            lv_obj_set_style_shadow_color(step, color_primary_, 0);
            lv_obj_set_style_shadow_opa(step, LV_OPA_20, 0);
        } else if(is_completed) {
            lv_obj_set_style_bg_color(step, color_success_, 0);
            lv_obj_set_style_border_color(step, color_success_, 0);
            lv_obj_set_style_shadow_width(step, 4, 0);
            lv_obj_set_style_shadow_opa(step, LV_OPA_10, 0);
            lv_obj_set_style_shadow_color(step, lv_color_black(), 0);
        } else {
            lv_obj_set_style_bg_color(step, lv_color_hex(0x3A4048), 0);
            lv_obj_set_style_border_color(step, lv_color_hex(0x4A5058), 0);
            lv_obj_set_style_shadow_width(step, 0, 0);
        }
        
        lv_obj_t* icon = lv_label_create(step);
        lv_label_set_text(icon, icons[i]);
        lv_obj_set_style_text_font(icon, &lv_font_montserrat_16, 0);  // 稍小字体
        lv_obj_set_style_text_color(icon, lv_color_white(), 0);
        lv_obj_center(icon);
        
        header_widgets_.workflow_buttons.push_back(step);
    }
    
    // Heartbeat Status
    lv_obj_t* heartbeat_container = lv_obj_create(header_panel_);
    lv_obj_set_size(heartbeat_container, 160, 50);
    lv_obj_align(heartbeat_container, LV_ALIGN_RIGHT_MID, -180, 0);
    lv_obj_set_style_bg_color(heartbeat_container, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(heartbeat_container, 25, 0);
    lv_obj_set_style_border_width(heartbeat_container, 2, 0);
    lv_obj_set_style_border_color(heartbeat_container, color_success_, 0);
    lv_obj_set_style_pad_all(heartbeat_container, 0, 0);
    lv_obj_clear_flag(heartbeat_container, LV_OBJ_FLAG_SCROLLABLE);
    
    header_widgets_.heartbeat_label = lv_label_create(heartbeat_container);
    lv_label_set_text(header_widgets_.heartbeat_label, LV_SYMBOL_REFRESH " Heartbeat OK");
    lv_obj_set_style_text_color(header_widgets_.heartbeat_label, color_success_, 0);
    lv_obj_set_style_text_font(header_widgets_.heartbeat_label, &lv_font_montserrat_14, 0);
    lv_obj_center(header_widgets_.heartbeat_label);
    
    // Response Time
    header_widgets_.response_label = lv_label_create(header_panel_);
    lv_label_set_text(header_widgets_.response_label, LV_SYMBOL_LOOP " 15ms");
    lv_obj_set_style_text_color(header_widgets_.response_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(header_widgets_.response_label, &lv_font_montserrat_14, 0);
    lv_obj_align(header_widgets_.response_label, LV_ALIGN_RIGHT_MID, -15, 0);
    
    return header_panel_;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLInterface::createCameraPanel() {
#ifdef ENABLE_LVGL
    camera_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(camera_panel_, lv_pct(75), lv_pct(75));  // 调整为75%布局
    lv_obj_align(camera_panel_, LV_ALIGN_TOP_LEFT, lv_pct(2), 80);
    lv_obj_add_style(camera_panel_, &style_card, 0);
    
    // 简洁优雅的边框
    lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0x1E2329), 0);
    lv_obj_set_style_border_width(camera_panel_, 1, 0);
    lv_obj_set_style_border_color(camera_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_60, 0);
    lv_obj_set_style_shadow_width(camera_panel_, 12, 0);
    lv_obj_set_style_shadow_color(camera_panel_, lv_color_black(), 0);
    lv_obj_set_style_shadow_opa(camera_panel_, LV_OPA_10, 0);
    lv_obj_set_style_radius(camera_panel_, 16, 0);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // Canvas画布
    camera_canvas_ = lv_canvas_create(camera_panel_);
    lv_obj_center(camera_canvas_);
    
    // 信息覆盖层
    lv_obj_t* info_overlay = lv_obj_create(camera_panel_);
    lv_obj_set_size(info_overlay, lv_pct(100), 60);
    lv_obj_align(info_overlay, LV_ALIGN_BOTTOM_MID, 0, -5);
    lv_obj_set_style_bg_color(info_overlay, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(info_overlay, LV_OPA_70, 0);
    lv_obj_set_style_border_width(info_overlay, 0, 0);
    lv_obj_set_style_radius(info_overlay, 8, 0);
    lv_obj_set_style_pad_all(info_overlay, 10, 0);
    lv_obj_clear_flag(info_overlay, LV_OBJ_FLAG_SCROLLABLE);
    
    // Coordinate Information
    camera_widgets_.coord_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.coord_value, LV_SYMBOL_GPS " X: 0.00 Y: 0.00 Z: 0.00");
    lv_obj_set_style_text_color(camera_widgets_.coord_value, color_primary_, 0);
    lv_obj_set_style_text_font(camera_widgets_.coord_value, &lv_font_montserrat_14, 0);
    lv_obj_align(camera_widgets_.coord_value, LV_ALIGN_LEFT_MID, 10, 0);
    
    // Quality Score
    camera_widgets_.quality_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.quality_value, LV_SYMBOL_IMAGE " Quality: 95%");
    lv_obj_set_style_text_color(camera_widgets_.quality_value, color_success_, 0);
    lv_obj_set_style_text_font(camera_widgets_.quality_value, &lv_font_montserrat_14, 0);
    lv_obj_align(camera_widgets_.quality_value, LV_ALIGN_CENTER, 0, 0);
    
    // Blade Information
    camera_widgets_.blade_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.blade_value, LV_SYMBOL_SETTINGS " Blade: #3");
    lv_obj_set_style_text_color(camera_widgets_.blade_value, color_warning_, 0);
    lv_obj_set_style_text_font(camera_widgets_.blade_value, &lv_font_montserrat_14, 0);
    lv_obj_align(camera_widgets_.blade_value, LV_ALIGN_RIGHT_MID, -10, 0);
    
    return camera_panel_;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLInterface::createControlPanel() {
#ifdef ENABLE_LVGL
    control_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(control_panel_, lv_pct(25), lv_pct(75));  // 调整为25%宽度×75%高度
    lv_obj_align(control_panel_, LV_ALIGN_TOP_RIGHT, -lv_pct(2), 80);
    lv_obj_add_style(control_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(control_panel_, 20, 0);
    lv_obj_set_style_radius(control_panel_, 16, 0);
    lv_obj_set_flex_flow(control_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(control_panel_, 16, 0);
    
    // Control Panel Title
    lv_obj_t* title = lv_label_create(control_panel_);
    lv_label_set_text(title, LV_SYMBOL_SETTINGS " System Control");
    lv_obj_add_style(title, &style_text_title, 0);
    lv_obj_set_style_text_color(title, color_primary_, 0);
    
    // === Jetson System Monitoring Progress Bar Area ===
    lv_obj_t* jetson_section = lv_obj_create(control_panel_);
    lv_obj_set_size(jetson_section, lv_pct(100), 120);
    lv_obj_set_style_bg_color(jetson_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(jetson_section, 12, 0);
    lv_obj_set_style_border_width(jetson_section, 1, 0);
    lv_obj_set_style_border_color(jetson_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(jetson_section, 12, 0);
    lv_obj_clear_flag(jetson_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* jetson_label = lv_label_create(jetson_section);
    lv_label_set_text(jetson_label, LV_SYMBOL_CHARGE " Jetson System");
    lv_obj_set_style_text_color(jetson_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(jetson_label, &lv_font_montserrat_14, 0);
    lv_obj_align(jetson_label, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // CPU进度条
    control_widgets_.cpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.cpu_bar, lv_pct(100), 18);
    lv_obj_align(control_widgets_.cpu_bar, LV_ALIGN_TOP_LEFT, 0, 25);
    lv_bar_set_range(control_widgets_.cpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.cpu_bar, 45, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_success_, LV_PART_INDICATOR);
    
    control_widgets_.cpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.cpu_label, "CPU: 45%");
    lv_obj_set_style_text_color(control_widgets_.cpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.cpu_label, LV_ALIGN_TOP_LEFT, 0, 48);
    
    // GPU进度条
    control_widgets_.gpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.gpu_bar, lv_pct(100), 18);
    lv_obj_align(control_widgets_.gpu_bar, LV_ALIGN_TOP_LEFT, 0, 65);
    lv_bar_set_range(control_widgets_.gpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.gpu_bar, 72, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
    
    control_widgets_.gpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.gpu_label, "GPU: 72%");
    lv_obj_set_style_text_color(control_widgets_.gpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.gpu_label, LV_ALIGN_TOP_LEFT, 0, 88);
    
    // 内存进度条
    control_widgets_.mem_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.mem_bar, lv_pct(50), 18);
    lv_obj_align(control_widgets_.mem_bar, LV_ALIGN_TOP_RIGHT, 0, 25);
    lv_bar_set_range(control_widgets_.mem_bar, 0, 100);
    lv_bar_set_value(control_widgets_.mem_bar, 58, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_primary_, LV_PART_INDICATOR);
    
    control_widgets_.mem_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.mem_label, "MEM: 4.6/8GB");
    lv_obj_set_style_text_color(control_widgets_.mem_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.mem_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.mem_label, LV_ALIGN_TOP_RIGHT, 0, 48);
    
    // === AI Model Monitoring Area ===
    lv_obj_t* ai_section = lv_obj_create(control_panel_);
    lv_obj_set_size(ai_section, lv_pct(100), 110);
    lv_obj_set_style_bg_color(ai_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(ai_section, 12, 0);
    lv_obj_set_style_border_width(ai_section, 1, 0);
    lv_obj_set_style_border_color(ai_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(ai_section, 12, 0);
    lv_obj_clear_flag(ai_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* ai_title = lv_label_create(ai_section);
    lv_label_set_text(ai_title, LV_SYMBOL_EYE_OPEN " AI Model Monitor");
    lv_obj_set_style_text_color(ai_title, lv_color_hex(0x7FB069), 0);
    lv_obj_set_style_text_font(ai_title, &lv_font_montserrat_14, 0);
    lv_obj_align(ai_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    control_widgets_.ai_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_fps_label, "Inference FPS: 28.5");
    lv_obj_set_style_text_color(control_widgets_.ai_fps_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_fps_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_fps_label, LV_ALIGN_TOP_LEFT, 0, 25);
    
    control_widgets_.ai_confidence_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_confidence_label, "Confidence: 0.94");
    lv_obj_set_style_text_color(control_widgets_.ai_confidence_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_confidence_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_confidence_label, LV_ALIGN_TOP_LEFT, 0, 45);
    
    control_widgets_.ai_latency_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_latency_label, "Latency: 12ms");
    lv_obj_set_style_text_color(control_widgets_.ai_latency_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_latency_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_latency_label, LV_ALIGN_TOP_RIGHT, 0, 25);
    
    control_widgets_.ai_model_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_model_label, "Model: YOLOv8n");
    lv_obj_set_style_text_color(control_widgets_.ai_model_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_model_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_model_label, LV_ALIGN_TOP_RIGHT, 0, 45);
    
    // === Modbus Communication Statistics Area ===
    lv_obj_t* modbus_section = lv_obj_create(control_panel_);
    lv_obj_set_size(modbus_section, lv_pct(100), 110);
    lv_obj_set_style_bg_color(modbus_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(modbus_section, 12, 0);
    lv_obj_set_style_border_width(modbus_section, 1, 0);
    lv_obj_set_style_border_color(modbus_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(modbus_section, 12, 0);
    lv_obj_clear_flag(modbus_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* modbus_title = lv_label_create(modbus_section);
    lv_label_set_text(modbus_title, LV_SYMBOL_WIFI " Modbus Statistics");
    lv_obj_set_style_text_color(modbus_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(modbus_title, &lv_font_montserrat_14, 0);
    lv_obj_align(modbus_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    control_widgets_.modbus_connection_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_connection_label, "Connection: 02:15:32");
    lv_obj_set_style_text_color(control_widgets_.modbus_connection_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_connection_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_connection_label, LV_ALIGN_TOP_LEFT, 0, 25);
    
    control_widgets_.modbus_packets_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_packets_label, "Packets: 1247");
    lv_obj_set_style_text_color(control_widgets_.modbus_packets_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_packets_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_packets_label, LV_ALIGN_TOP_LEFT, 0, 45);
    
    control_widgets_.modbus_errors_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_errors_label, "Error Rate: 0.02%");
    lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_errors_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_errors_label, LV_ALIGN_TOP_RIGHT, 0, 25);
    
    control_widgets_.modbus_heartbeat_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_heartbeat_label, "Heartbeat: OK");
    lv_obj_set_style_text_color(control_widgets_.modbus_heartbeat_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_heartbeat_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_heartbeat_label, LV_ALIGN_TOP_RIGHT, 0, 45);
    
    return control_panel_;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLInterface::createStatusPanel() {
#ifdef ENABLE_LVGL
    status_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(status_panel_, lv_pct(96), lv_pct(20));  // 底部状态面板
    lv_obj_align(status_panel_, LV_ALIGN_BOTTOM_MID, 0, -100);
    lv_obj_add_style(status_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(status_panel_, 16, 0);
    lv_obj_set_style_radius(status_panel_, 12, 0);
    lv_obj_clear_flag(status_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 使用flex布局分为左右两部分
    lv_obj_set_flex_flow(status_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(status_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(status_panel_, 20, 0);
    
    // === 左侧：系统指标3列网格 ===
    lv_obj_t* metrics_container = lv_obj_create(status_panel_);
    lv_obj_set_size(metrics_container, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_opa(metrics_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(metrics_container, 0, 0);
    lv_obj_set_style_pad_all(metrics_container, 0, 0);
    lv_obj_clear_flag(metrics_container, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* metrics_title = lv_label_create(metrics_container);
    lv_label_set_text(metrics_title, LV_SYMBOL_SETTINGS " System Metrics");
    lv_obj_set_style_text_color(metrics_title, color_primary_, 0);
    lv_obj_set_style_text_font(metrics_title, &lv_font_montserrat_14, 0);
    lv_obj_align(metrics_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // 3 Column Grid Container
    lv_obj_t* grid_container = lv_obj_create(metrics_container);
    lv_obj_set_size(grid_container, lv_pct(100), 80);
    lv_obj_align(grid_container, LV_ALIGN_TOP_LEFT, 0, 25);
    lv_obj_set_style_bg_opa(grid_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(grid_container, 0, 0);
    lv_obj_set_style_pad_all(grid_container, 0, 0);
    lv_obj_set_flex_flow(grid_container, LV_FLEX_FLOW_ROW_WRAP);
    lv_obj_set_flex_align(grid_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_clear_flag(grid_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 12 System Metrics Data
    const char* metric_names[] = {
        "CPU Freq", "GPU Freq", "MEM Freq",
        "CPU Temp", "GPU Temp", "SYS Temp",
        "Fan Speed", "Power V", "Power A",
        "Net Ping", "Disk Read", "Disk Write"
    };
    const char* metric_values[] = {
        "1.9GHz", "1.3GHz", "1600MHz",
        "62°C", "58°C", "45°C",
        "3200RPM", "19.2V", "2.1A",
        "12ms", "450MB/s", "380MB/s"
    };
    const lv_color_t metric_colors[] = {
        color_success_, color_warning_, color_primary_,
        color_warning_, color_success_, color_success_,
        color_primary_, color_success_, color_success_,
        color_success_, color_primary_, color_primary_
    };
    
    for(int i = 0; i < 12; i++) {
        lv_obj_t* metric_item = lv_obj_create(grid_container);
        lv_obj_set_size(metric_item, lv_pct(32), 35);  // 3列布局，每列32%宽度
        lv_obj_set_style_bg_color(metric_item, lv_color_hex(0x0F1419), 0);
        lv_obj_set_style_radius(metric_item, 6, 0);
        lv_obj_set_style_border_width(metric_item, 1, 0);
        lv_obj_set_style_border_color(metric_item, lv_color_hex(0x2A3441), 0);
        lv_obj_set_style_pad_all(metric_item, 8, 0);
        lv_obj_clear_flag(metric_item, LV_OBJ_FLAG_SCROLLABLE);
        
        lv_obj_t* name_label = lv_label_create(metric_item);
        lv_label_set_text(name_label, metric_names[i]);
        lv_obj_set_style_text_color(name_label, lv_color_hex(0x8A92A1), 0);
        lv_obj_set_style_text_font(name_label, &lv_font_montserrat_12, 0);
        lv_obj_align(name_label, LV_ALIGN_TOP_LEFT, 0, 0);
        
        lv_obj_t* value_label = lv_label_create(metric_item);
        lv_label_set_text(value_label, metric_values[i]);
        lv_obj_set_style_text_color(value_label, metric_colors[i], 0);
        lv_obj_set_style_text_font(value_label, &lv_font_montserrat_14, 0);
        lv_obj_align(value_label, LV_ALIGN_BOTTOM_RIGHT, 0, 0);
        
        status_widgets_.metric_labels.push_back(value_label);
    }
    
    // === 右侧：版本信息 ===
    lv_obj_t* version_container = lv_obj_create(status_panel_);
    lv_obj_set_size(version_container, lv_pct(28), lv_pct(100));
    lv_obj_set_style_bg_color(version_container, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(version_container, 8, 0);
    lv_obj_set_style_border_width(version_container, 1, 0);
    lv_obj_set_style_border_color(version_container, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(version_container, 12, 0);
    lv_obj_clear_flag(version_container, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* version_title = lv_label_create(version_container);
    lv_label_set_text(version_title, LV_SYMBOL_LIST " System Version");
    lv_obj_set_style_text_color(version_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(version_title, &lv_font_montserrat_14, 0);
    lv_obj_align(version_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // JetPack版本
    status_widgets_.jetpack_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.jetpack_version, "JetPack: 5.1.2");
    lv_obj_set_style_text_color(status_widgets_.jetpack_version, lv_color_white(), 0);
    lv_obj_set_style_text_font(status_widgets_.jetpack_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.jetpack_version, LV_ALIGN_TOP_LEFT, 0, 22);
    
    // CUDA版本
    status_widgets_.cuda_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.cuda_version, "CUDA: 11.4.315");
    lv_obj_set_style_text_color(status_widgets_.cuda_version, color_success_, 0);
    lv_obj_set_style_text_font(status_widgets_.cuda_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.cuda_version, LV_ALIGN_TOP_LEFT, 0, 40);
    
    // TensorRT版本
    status_widgets_.tensorrt_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.tensorrt_version, "TensorRT: 8.5.2");
    lv_obj_set_style_text_color(status_widgets_.tensorrt_version, color_primary_, 0);
    lv_obj_set_style_text_font(status_widgets_.tensorrt_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.tensorrt_version, LV_ALIGN_TOP_LEFT, 0, 58);
    
    // OpenCV版本
    status_widgets_.opencv_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.opencv_version, "OpenCV: 4.8.0");
    lv_obj_set_style_text_color(status_widgets_.opencv_version, color_warning_, 0);
    lv_obj_set_style_text_font(status_widgets_.opencv_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.opencv_version, LV_ALIGN_TOP_LEFT, 0, 76);
    
    // Ubuntu版本
    status_widgets_.ubuntu_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.ubuntu_version, "Ubuntu: 20.04.6");
    lv_obj_set_style_text_color(status_widgets_.ubuntu_version, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(status_widgets_.ubuntu_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.ubuntu_version, LV_ALIGN_TOP_LEFT, 0, 94);
    
    return status_panel_;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLInterface::createFooterPanel() {
#ifdef ENABLE_LVGL
    footer_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(footer_panel_, lv_pct(96), 80);  // 使用百分比，降低高度
    lv_obj_align(footer_panel_, LV_ALIGN_BOTTOM_MID, 0, -12);
    
    // 现代简洁的背景样式
    lv_obj_set_style_bg_color(footer_panel_, color_surface_, 0);
    lv_obj_set_style_radius(footer_panel_, 20, 0);  // 降低圆角
    lv_obj_set_style_border_width(footer_panel_, 1, 0);
    lv_obj_set_style_border_color(footer_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(footer_panel_, LV_OPA_40, 0);
    lv_obj_set_style_shadow_width(footer_panel_, 16, 0);  // 减少阴影
    lv_obj_set_style_shadow_color(footer_panel_, lv_color_black(), 0);
    lv_obj_set_style_shadow_opa(footer_panel_, LV_OPA_20, 0);
    lv_obj_set_style_pad_all(footer_panel_, 16, 0);
    lv_obj_clear_flag(footer_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 使用flex布局，增加间距
    lv_obj_set_flex_flow(footer_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(footer_panel_, LV_FLEX_ALIGN_SPACE_AROUND,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(footer_panel_, 12, 0);
    
    // 主操作区域容器
    lv_obj_t* main_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(main_controls, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_opa(main_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(main_controls, 0, 0);
    lv_obj_set_style_pad_all(main_controls, 0, 0);
    lv_obj_set_flex_flow(main_controls, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(main_controls, LV_FLEX_ALIGN_SPACE_EVENLY,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Start Button
    footer_widgets_.start_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.start_btn, 110, 48);  // More compact
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_success, 0);
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* start_label = lv_label_create(footer_widgets_.start_btn);
    lv_label_set_text(start_label, LV_SYMBOL_PLAY " START");
    lv_obj_set_style_text_font(start_label, &lv_font_montserrat_16, 0);
    lv_obj_center(start_label);
    lv_obj_add_event_cb(footer_widgets_.start_btn, onStartButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // Pause Button
    footer_widgets_.pause_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.pause_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_warning, 0);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* pause_label = lv_label_create(footer_widgets_.pause_btn);
    lv_label_set_text(pause_label, LV_SYMBOL_PAUSE " PAUSE");
    lv_obj_set_style_text_font(pause_label, &lv_font_montserrat_16, 0);
    lv_obj_center(pause_label);
    lv_obj_add_event_cb(footer_widgets_.pause_btn, onPauseButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // Stop Button
    footer_widgets_.stop_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.stop_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_primary, 0);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_pressed, LV_STATE_PRESSED);
    lv_obj_set_style_bg_color(footer_widgets_.stop_btn, lv_color_hex(0x6B7280), 0);
    
    lv_obj_t* stop_label = lv_label_create(footer_widgets_.stop_btn);
    lv_label_set_text(stop_label, LV_SYMBOL_STOP " STOP");
    lv_obj_set_style_text_font(stop_label, &lv_font_montserrat_16, 0);
    lv_obj_center(stop_label);
    lv_obj_add_event_cb(footer_widgets_.stop_btn, onStopButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 危险操作区域
    lv_obj_t* danger_zone = lv_obj_create(footer_panel_);
    lv_obj_set_size(danger_zone, 70, lv_pct(100));
    lv_obj_set_style_bg_opa(danger_zone, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(danger_zone, 0, 0);
    lv_obj_set_style_pad_all(danger_zone, 0, 0);
    
    // 急停按钮（独立突出）
    footer_widgets_.emergency_btn = lv_btn_create(danger_zone);
    lv_obj_set_size(footer_widgets_.emergency_btn, 60, 60);
    lv_obj_add_style(footer_widgets_.emergency_btn, &style_btn_danger, 0);
    lv_obj_add_style(footer_widgets_.emergency_btn, &style_btn_pressed, LV_STATE_PRESSED);
    lv_obj_center(footer_widgets_.emergency_btn);
    
    lv_obj_t* emergency_label = lv_label_create(footer_widgets_.emergency_btn);
    lv_label_set_text(emergency_label, LV_SYMBOL_WARNING);
    lv_obj_set_style_text_font(emergency_label, &lv_font_montserrat_24, 0);
    lv_obj_center(emergency_label);
    lv_obj_add_event_cb(footer_widgets_.emergency_btn, onEmergencyButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 辅助操作区域
    lv_obj_t* aux_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(aux_controls, lv_pct(20), lv_pct(100));
    lv_obj_set_style_bg_opa(aux_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(aux_controls, 0, 0);
    lv_obj_set_style_pad_all(aux_controls, 0, 0);
    lv_obj_set_flex_flow(aux_controls, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(aux_controls, LV_FLEX_ALIGN_CENTER,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 设置按钮
    footer_widgets_.power_btn = lv_btn_create(aux_controls);
    lv_obj_set_size(footer_widgets_.power_btn, 48, 48);
    lv_obj_add_style(footer_widgets_.power_btn, &style_btn_primary, 0);
    lv_obj_add_style(footer_widgets_.power_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* power_label = lv_label_create(footer_widgets_.power_btn);
    lv_label_set_text(power_label, LV_SYMBOL_SETTINGS);
    lv_obj_set_style_text_font(power_label, &lv_font_montserrat_16, 0);
    lv_obj_center(power_label);
    lv_obj_add_event_cb(footer_widgets_.power_btn, onSettingsButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    return footer_panel_;
#else
    return nullptr;
#endif
}

bool LVGLInterface::start() {
#ifdef ENABLE_LVGL
    if (running_.load()) {
        std::cerr << "[LVGLInterface] 界面线程已在运行" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLInterface] 启动界面线程..." << std::endl;
    should_stop_.store(false);
    running_.store(true);
    
    ui_thread_ = std::thread(&LVGLInterface::uiLoop, this);
    
    std::cout << "[LVGLInterface] 界面线程启动成功" << std::endl;
    return true;
#else
    return false;
#endif
}

void LVGLInterface::stop() {
#ifdef ENABLE_LVGL
    if (!running_.load()) {
        return;
    }
    
    std::cout << "[LVGLInterface] 停止界面线程..." << std::endl;
    should_stop_.store(true);
    
    if (ui_thread_.joinable()) {
        ui_thread_.join();
    }
    
    running_.store(false);
    std::cout << "[LVGLInterface] 界面线程已停止" << std::endl;
#endif
}

void LVGLInterface::uiLoop() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] UI主循环开始" << std::endl;
    
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    frame_count_ = 0;
    
    while (!should_stop_.load()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        
        // 处理LVGL任务
        uint32_t time_till_next = lv_timer_handler();
        
        // 更新界面数据
        updateInterface();
        
        // 计算FPS
        frame_count_++;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_frame_time).count();
        
        if (elapsed >= 1000) {
            ui_fps_ = frame_count_ * 1000.0f / elapsed;
            frame_count_ = 0;
            last_frame_time = current_time;
        }
        
        // 控制刷新率
        std::this_thread::sleep_for(std::chrono::milliseconds(
            std::min(time_till_next, 16u))); // 最少16ms (60fps)
    }
    
    std::cout << "[LVGLInterface] UI主循环结束" << std::endl;
#endif
}

void LVGLInterface::updateInterface() {
#ifdef ENABLE_LVGL
    if (!data_bridge_) return;
    
    // 更新系统状态
    updateSystemStats();
    
    // 更新Modbus显示
    updateModbusDisplay();
    
    // 更新工作流程
    updateWorkflowStatus();
    
    // 更新摄像头
    updateCameraView();
#endif
}

void LVGLInterface::updateSystemStats() {
#ifdef ENABLE_LVGL
    // 更新头部响应时间标签
    if (header_widgets_.response_label) {
        static int counter = 0;
        int response_ms = 12 + (counter++ % 10);
        lv_label_set_text_fmt(header_widgets_.response_label,
            LV_SYMBOL_LOOP " %dms", response_ms);
    }
    
    // 更新Jetson系统监控进度条
    if (control_widgets_.cpu_bar) {
        static int cpu_usage = 45;
        cpu_usage = 40 + (rand() % 30);  // 模拟40-70%的CPU使用率
        lv_bar_set_value(control_widgets_.cpu_bar, cpu_usage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.cpu_label, "CPU: %d%%", cpu_usage);
        
        // 根据使用率设置颜色
        if (cpu_usage > 80) {
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_error_, LV_PART_INDICATOR);
        } else if (cpu_usage > 60) {
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
        } else {
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_success_, LV_PART_INDICATOR);
        }
    }
    
    if (control_widgets_.gpu_bar) {
        static int gpu_usage = 72;
        gpu_usage = 60 + (rand() % 25);  // 模拟60-85%的GPU使用率
        lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.gpu_label, "GPU: %d%%", gpu_usage);
        
        if (gpu_usage > 85) {
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_error_, LV_PART_INDICATOR);
        } else if (gpu_usage > 70) {
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
        } else {
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_success_, LV_PART_INDICATOR);
        }
    }
    
    if (control_widgets_.mem_bar) {
        static float mem_used = 4.6f;
        static float mem_total = 8.0f;
        mem_used = 3.8f + ((rand() % 200) / 100.0f);  // 模拟3.8-5.8GB内存使用
        int mem_percentage = (int)((mem_used / mem_total) * 100);
        lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.mem_label, "MEM: %.1f/%.0fGB", mem_used, mem_total);
        
        if (mem_percentage > 85) {
            lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_error_, LV_PART_INDICATOR);
        } else if (mem_percentage > 70) {
            lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
        } else {
            lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_primary_, LV_PART_INDICATOR);
        }
    }
    
    // 更新AI模型监控数据
    if (control_widgets_.ai_fps_label) {
        static float ai_fps = 28.5f;
        ai_fps = 25.0f + ((rand() % 80) / 10.0f);  // 模拟25-33FPS
        lv_label_set_text_fmt(control_widgets_.ai_fps_label, "推理FPS: %.1f", ai_fps);
    }
    
    if (control_widgets_.ai_confidence_label) {
        static float confidence = 0.94f;
        confidence = 0.85f + ((rand() % 15) / 100.0f);  // 模拟0.85-1.00置信度
        lv_label_set_text_fmt(control_widgets_.ai_confidence_label, "置信度: %.2f", confidence);
    }
    
    if (control_widgets_.ai_latency_label) {
        static int latency = 12;
        latency = 8 + (rand() % 8);  // 模拟8-16ms延迟
        lv_label_set_text_fmt(control_widgets_.ai_latency_label, "延迟: %dms", latency);
    }
    
    // 更新Modbus通信统计
    if (control_widgets_.modbus_connection_label) {
        static int hours = 2, minutes = 15, seconds = 32;
        seconds++;
        if (seconds >= 60) { seconds = 0; minutes++; }
        if (minutes >= 60) { minutes = 0; hours++; }
        lv_label_set_text_fmt(control_widgets_.modbus_connection_label,
            "连接时长: %02d:%02d:%02d", hours, minutes, seconds);
    }
    
    if (control_widgets_.modbus_packets_label) {
        static int packets = 1247;
        packets += 1 + (rand() % 3);  // 模拟数据包增长
        lv_label_set_text_fmt(control_widgets_.modbus_packets_label, "数据包: %d", packets);
    }
    
    if (control_widgets_.modbus_errors_label) {
        static float error_rate = 0.02f;
        error_rate = (rand() % 10) / 1000.0f;  // 模拟0.000-0.010%错误率
        lv_label_set_text_fmt(control_widgets_.modbus_errors_label, "错误率: %.3f%%", error_rate);
    }
    
    // 更新系统指标（12项指标的动态数据）
    if (!status_widgets_.metric_labels.empty()) {
        static const char* dynamic_values[][12] = {
            {"1.9GHz", "1.3GHz", "1600MHz", "62°C", "58°C", "45°C", "3200RPM", "19.2V", "2.1A", "12ms", "450MB/s", "380MB/s"},
            {"2.0GHz", "1.4GHz", "1600MHz", "64°C", "60°C", "47°C", "3400RPM", "19.1V", "2.2A", "11ms", "460MB/s", "390MB/s"},
            {"1.8GHz", "1.2GHz", "1533MHz", "59°C", "55°C", "43°C", "3000RPM", "19.3V", "2.0A", "13ms", "440MB/s", "370MB/s"}
        };
        
        static int update_cycle = 0;
        static int cycle_counter = 0;
        
        if (++cycle_counter >= 30) {  // 每30次更新切换一次数据组
            cycle_counter = 0;
            update_cycle = (update_cycle + 1) % 3;
        }
        
        for (size_t i = 0; i < status_widgets_.metric_labels.size() && i < 12; i++) {
            lv_label_set_text(status_widgets_.metric_labels[i], dynamic_values[update_cycle][i]);
        }
    }
#endif
}

void LVGLInterface::updateModbusDisplay() {
#ifdef ENABLE_LVGL
    if (!control_widgets_.modbus_table) return;
    
    // 示例数据更新
    lv_table_set_cell_value(control_widgets_.modbus_table, 1, 0, "40001");
    lv_table_set_cell_value(control_widgets_.modbus_table, 1, 1, "1234");
    lv_table_set_cell_value(control_widgets_.modbus_table, 1, 2, LV_SYMBOL_OK);
    
    lv_table_set_cell_value(control_widgets_.modbus_table, 2, 0, "40002");
    lv_table_set_cell_value(control_widgets_.modbus_table, 2, 1, "5678");
    lv_table_set_cell_value(control_widgets_.modbus_table, 2, 2, LV_SYMBOL_OK);
    
    lv_table_set_cell_value(control_widgets_.modbus_table, 3, 0, "40003");
    lv_table_set_cell_value(control_widgets_.modbus_table, 3, 1, "9012");
    lv_table_set_cell_value(control_widgets_.modbus_table, 3, 2, LV_SYMBOL_OK);
#endif
}

void LVGLInterface::updateWorkflowStatus() {
#ifdef ENABLE_LVGL
    // 更新工作流程步骤指示器
    for(size_t i = 0; i < header_widgets_.workflow_buttons.size(); i++) {
        lv_obj_t* step = header_widgets_.workflow_buttons[i];
        
        bool is_active = (i == (size_t)current_step_ - 1);
        bool is_completed = (i < (size_t)current_step_ - 1);
        
        if(is_active) {
            lv_obj_set_style_bg_color(step, color_primary_, 0);
            lv_obj_set_style_border_color(step, color_primary_, 0);
        } else if(is_completed) {
            lv_obj_set_style_bg_color(step, color_success_, 0);
            lv_obj_set_style_border_color(step, color_success_, 0);
        } else {
            lv_obj_set_style_bg_color(step, lv_color_hex(0x2A3441), 0);
            lv_obj_set_style_border_color(step, lv_color_hex(0x4A5461), 0);
        }
    }
#endif
}

void LVGLInterface::updateCameraView() {
#ifdef ENABLE_LVGL
    // 这里应该更新canvas内容
    // 从DataBridge获取最新图像并绘制到canvas上
    
    // 示例：更新信息标签
    if (camera_widgets_.coord_value) {
        static float x = 0.0f;
        x += 0.1f;
        lv_label_set_text_fmt(camera_widgets_.coord_value,
            LV_SYMBOL_GPS " X: %.2f Y: %.2f Z: %.2f", x, x*0.8f, x*0.5f);
    }
#endif
}

void LVGLInterface::drawDetectionResults(const core::DetectionResult& result) {
#ifdef ENABLE_LVGL
    // 在canvas上绘制检测框和标签
    // 需要根据实际的DetectionResult结构实现
#endif
}

// ==================== 事件处理器 ====================

void LVGLInterface::onStartButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 启动按钮被点击" << std::endl;
        self->system_running_ = true;
        self->showMessageDialog("系统启动", "系统正在启动...");
    }
#endif
}

void LVGLInterface::onStopButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 停止按钮被点击" << std::endl;
        self->system_running_ = false;
        self->showMessageDialog("系统停止", "系统已停止运行");
    }
#endif
}

void LVGLInterface::onPauseButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 暂停按钮被点击" << std::endl;
        self->showMessageDialog("系统暂停", "系统已暂停");
    }
#endif
}

void LVGLInterface::onEmergencyButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 急停按钮被点击" << std::endl;
        self->emergency_stop_ = true;
        self->system_running_ = false;
        self->showMessageDialog("紧急停止", "系统已紧急停止！");
    }
#endif
}

void LVGLInterface::onBladeSelectionChanged(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    lv_obj_t* btn = static_cast<lv_obj_t*>(lv_event_get_target(e));
    
    if (self && btn) {
        // 找到被点击的刀片编号
        for(size_t i = 0; i < self->control_widgets_.blade_buttons.size(); i++) {
            if (self->control_widgets_.blade_buttons[i] == btn) {
                self->selected_blade_ = i + 1;
                std::cout << "[LVGLInterface] 选择刀片 #" << (i+1) << std::endl;
                
                // 更新所有按钮样式
                for(size_t j = 0; j < self->control_widgets_.blade_buttons.size(); j++) {
                    lv_obj_t* b = self->control_widgets_.blade_buttons[j];
                    if (j == i) {
                        lv_obj_set_style_bg_color(b, self->color_success_, 0);
                        lv_obj_set_style_shadow_color(b, self->color_success_, 0);
                    } else {
                        lv_obj_set_style_bg_color(b, self->color_primary_, 0);
                        lv_obj_set_style_shadow_color(b, self->color_primary_, 0);
                    }
                }
                
                // 更新摄像头面板的刀片显示
                if (self->camera_widgets_.blade_value) {
                    lv_label_set_text_fmt(self->camera_widgets_.blade_value,
                        LV_SYMBOL_SETTINGS " 刀片: #%d", self->selected_blade_);
                }
                
                break;
            }
        }
    }
#endif
}

void LVGLInterface::onSettingsButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 设置按钮被点击" << std::endl;
        self->showMessageDialog("系统设置", "设置功能开发中...");
    }
#endif
}

void LVGLInterface::showMessageDialog(const std::string& title, const std::string& message) {
#ifdef ENABLE_LVGL
    lv_obj_t* mbox = lv_msgbox_create(NULL);
    lv_msgbox_add_title(mbox, title.c_str());
    lv_msgbox_add_text(mbox, message.c_str());
    lv_msgbox_add_close_button(mbox);
    lv_obj_center(mbox);
    
    // 添加样式
    lv_obj_set_style_bg_color(mbox, color_surface_, 0);
    lv_obj_set_style_border_color(mbox, color_primary_, 0);
    lv_obj_set_style_border_width(mbox, 2, 0);
    lv_obj_set_style_shadow_width(mbox, 30, 0);
    lv_obj_set_style_shadow_opa(mbox, LV_OPA_50, 0);
#endif
}

void LVGLInterface::setFullscreen(bool fullscreen) {
    // DRM模式默认就是全屏
    std::cout << "[LVGLInterface] 全屏模式: " << (fullscreen ? "启用" : "禁用") << std::endl;
}

bool LVGLInterface::detectDisplayResolution(int& width, int& height) {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 正在检测DRM显示器分辨率..." << std::endl;
    
    // 尝试多个DRM设备路径
    const char* drm_devices[] = {
        "/dev/dri/card1",
        "/dev/dri/card0",
        "/dev/dri/card2"
    };
    
    for (const char* device_path : drm_devices) {
        int fd = open(device_path, O_RDWR);
        if (fd < 0) {
            std::cout << "[LVGLInterface] 无法打开DRM设备: " << device_path << std::endl;
            continue;
        }
        
        std::cout << "[LVGLInterface] 成功打开DRM设备: " << device_path << std::endl;
        
        // 获取DRM资源
        drmModeRes* resources = drmModeGetResources(fd);
        if (!resources) {
            std::cerr << "[LVGLInterface] 无法获取DRM资源" << std::endl;
            close(fd);
            continue;
        }
        
        // 查找连接的显示器
        for (int i = 0; i < resources->count_connectors; i++) {
            drmModeConnector* connector = drmModeGetConnector(fd, resources->connectors[i]);
            if (!connector) continue;
            
            // 检查连接器是否连接了显示器
            if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
                // 获取首选模式（通常是第一个模式）
                drmModeModeInfo* mode = &connector->modes[0];
                width = mode->hdisplay;
                height = mode->vdisplay;
                
                std::cout << "[LVGLInterface] 检测到显示器分辨率: "
                          << width << "x" << height << " @" << mode->vrefresh << "Hz" << std::endl;
                std::cout << "[LVGLInterface] 显示器模式名称: " << mode->name << std::endl;
                
                drmModeFreeConnector(connector);
                drmModeFreeResources(resources);
                close(fd);
                return true;
            }
            
            drmModeFreeConnector(connector);
        }
        
        drmModeFreeResources(resources);
        close(fd);
    }
    
    std::cerr << "[LVGLInterface] 无法检测到连接的显示器" << std::endl;
    return false;
#else
    std::cerr << "[LVGLInterface] LVGL未启用，无法检测显示器分辨率" << std::endl;
    return false;
#endif
}

// ==================== LVGL v9 回调函数 ====================

void display_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
#ifdef ENABLE_LVGL
    // 优化的DRM显示刷新实现 - 修复渲染质量问题
    static int drm_fd = -1;
    static uint32_t fb_id = 0;
    static drmModeCrtc* crtc = nullptr;
    static drmModeConnector* connector = nullptr;
    static uint32_t* framebuffer = nullptr;
    static uint32_t fb_handle = 0;
    static bool drm_initialized = false;
    static uint32_t drm_width = 0;
    static uint32_t drm_height = 0;
    static uint32_t stride = 0;
    static uint32_t buffer_size = 0;
    
    // 初始化DRM设备 (只初始化一次)
    if (!drm_initialized) {
        const char* drm_devices[] = {"/dev/dri/card1", "/dev/dri/card0", "/dev/dri/card2"};
        
        for (const char* device_path : drm_devices) {
            drm_fd = open(device_path, O_RDWR);
            if (drm_fd >= 0) {
                std::cout << "[DRM] 成功打开设备: " << device_path << std::endl;
                
                // 获取DRM资源
                drmModeRes* resources = drmModeGetResources(drm_fd);
                if (resources) {
                    // 查找连接的显示器
                    for (int i = 0; i < resources->count_connectors; i++) {
                        connector = drmModeGetConnector(drm_fd, resources->connectors[i]);
                        if (connector && connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
                            
                            // 选择最佳显示模式 (通常是第一个，也是首选模式)
                            drmModeModeInfo* mode = &connector->modes[0];
                            drm_width = mode->hdisplay;
                            drm_height = mode->vdisplay;
                            
                            std::cout << "[DRM] 显示器模式: " << drm_width << "x" << drm_height
                                      << " @" << mode->vrefresh << "Hz" << std::endl;
                            std::cout << "[DRM] 模式名称: " << mode->name << std::endl;
                            
                            // 查找合适的CRTC
                            for (int j = 0; j < resources->count_crtcs; j++) {
                                // 检查这个CRTC是否可以驱动这个连接器
                                if (connector->encoder_id) {
                                    drmModeEncoder* encoder = drmModeGetEncoder(drm_fd, connector->encoder_id);
                                    if (encoder && (encoder->possible_crtcs & (1 << j))) {
                                        crtc = drmModeGetCrtc(drm_fd, resources->crtcs[j]);
                                        if (crtc) {
                                            std::cout << "[DRM] 找到合适的CRTC: " << crtc->crtc_id << std::endl;
                                            drmModeFreeEncoder(encoder);
                                            break;
                                        }
                                    }
                                    if (encoder) drmModeFreeEncoder(encoder);
                                }
                            }
                            
                            if (crtc) break;
                        }
                    }
                    
                    // 创建优化的dumb buffer
                    if (crtc && connector && drm_width > 0 && drm_height > 0) {
                        struct drm_mode_create_dumb create_req = {};
                        create_req.width = drm_width;
                        create_req.height = drm_height;
                        create_req.bpp = 32; // 确保使用32位颜色深度
                        
                        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) == 0) {
                            fb_handle = create_req.handle;
                            stride = create_req.pitch;
                            buffer_size = create_req.size;
                            
                            std::cout << "[DRM] 创建buffer: " << drm_width << "x" << drm_height
                                      << ", stride: " << stride << ", size: " << buffer_size << std::endl;
                            
                            // 创建framebuffer对象 - 使用正确的格式
                            uint32_t depth = 24; // 颜色深度
                            uint32_t bpp = 32;   // 每像素位数
                            
                            if (drmModeAddFB(drm_fd, drm_width, drm_height, depth, bpp, stride, fb_handle, &fb_id) == 0) {
                                // 映射framebuffer到用户空间
                                struct drm_mode_map_dumb map_req = {};
                                map_req.handle = fb_handle;
                                
                                if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) == 0) {
                                    framebuffer = (uint32_t*)mmap(0, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_req.offset);
                                    if (framebuffer != MAP_FAILED) {
                                        // 设置CRTC使用我们的framebuffer - 使用正确的模式
                                        drmModeModeInfo* best_mode = &connector->modes[0];
                                        int ret = drmModeSetCrtc(drm_fd, crtc->crtc_id, fb_id, 0, 0,
                                                               &connector->connector_id, 1, best_mode);
                                        if (ret == 0) {
                                            drm_initialized = true;
                                            std::cout << "[DRM] DRM framebuffer初始化成功" << std::endl;
                                            
                                            // 清空framebuffer (深色背景)
                                            memset(framebuffer, 0x00, buffer_size);
                                            
                                            // 强制刷新显示
                                            drmModePageFlip(drm_fd, crtc->crtc_id, fb_id, 0, nullptr);
                                        } else {
                                            std::cerr << "[DRM] drmModeSetCrtc失败: " << ret << " (" << strerror(-ret) << ")" << std::endl;
                                        }
                                    } else {
                                        std::cerr << "[DRM] framebuffer映射失败: " << strerror(errno) << std::endl;
                                    }
                                } else {
                                    std::cerr << "[DRM] map dumb buffer失败" << std::endl;
                                }
                            } else {
                                std::cerr << "[DRM] 创建framebuffer失败" << std::endl;
                            }
                        } else {
                            std::cerr << "[DRM] 创建dumb buffer失败" << std::endl;
                        }
                    }
                    drmModeFreeResources(resources);
                }
                
                if (drm_initialized) {
                    break;
                } else {
                    close(drm_fd);
                    drm_fd = -1;
                }
            }
        }
        
        if (!drm_initialized) {
            std::cerr << "[DRM] 无法初始化任何DRM设备" << std::endl;
            drm_initialized = true; // 避免重复尝试
        }
    }
    
    // 优化的像素数据复制 - 修复颜色显示问题
    if (drm_initialized && framebuffer && drm_width > 0 && drm_height > 0 && stride > 0) {
        uint32_t area_width = area->x2 - area->x1 + 1;
        uint32_t area_height = area->y2 - area->y1 + 1;
        
        // 边界检查
        if (area->x1 >= 0 && area->y1 >= 0 &&
            area->x2 < (int32_t)drm_width && area->y2 < (int32_t)drm_height) {
            
            // 按行复制像素数据 - 考虑stride对齐
            for (uint32_t y = 0; y < area_height; y++) {
                uint32_t dst_row = area->y1 + y;
                uint32_t src_row_offset = y * area_width;
                uint32_t dst_row_offset = dst_row * (stride / 4); // stride是字节数，除以4得到uint32_t数量
                
                for (uint32_t x = 0; x < area_width; x++) {
                    uint32_t dst_col = area->x1 + x;
                    uint32_t src_idx = src_row_offset + x;
                    uint32_t dst_idx = dst_row_offset + dst_col;
                    
                    // LVGL v9像素格式转换 - 修复颜色问题
                    // LVGL可能使用RGB565或RGB888格式，需要正确转换
                    lv_color_t src_pixel;
                    
                    // 根据LVGL配置确定像素格式
                    #if LV_COLOR_DEPTH == 32
                        // 32位ARGB8888格式
                        uint32_t* src_pixels = (uint32_t*)px_map;
                        uint32_t src_value = src_pixels[src_idx];
                        src_pixel.red = (src_value >> 16) & 0xFF;
                        src_pixel.green = (src_value >> 8) & 0xFF;
                        src_pixel.blue = src_value & 0xFF;
                    #elif LV_COLOR_DEPTH == 16
                        // 16位RGB565格式
                        uint16_t* src_pixels = (uint16_t*)px_map;
                        uint16_t src_value = src_pixels[src_idx];
                        src_pixel.red = ((src_value >> 11) & 0x1F) * 255 / 31;   // 5位红色
                        src_pixel.green = ((src_value >> 5) & 0x3F) * 255 / 63;   // 6位绿色
                        src_pixel.blue = (src_value & 0x1F) * 255 / 31;          // 5位蓝色
                    #else
                        // 默认24位RGB888格式
                        lv_color_t* src_pixels = (lv_color_t*)px_map;
                        src_pixel = src_pixels[src_idx];
                    #endif
                    
                    // 转换为DRM标准的XRGB8888格式 (X=未使用,R=红色,G=绿色,B=蓝色)
                    uint32_t pixel = 0x00000000 |                    // 最高8位未使用
                                    ((uint32_t)src_pixel.red << 16) |   // 红色通道
                                    ((uint32_t)src_pixel.green << 8) |  // 绿色通道
                                    ((uint32_t)src_pixel.blue);         // 蓝色通道
                    
                    framebuffer[dst_idx] = pixel;
                }
            }
        }
    }
    
    // 通知LVGL刷新完成
    lv_display_flush_ready(disp);
#endif
}

void input_read_cb(lv_indev_t* indev, lv_indev_data_t* data) {
#ifdef ENABLE_LVGL
    // 简单的输入读取实现
    // 在实际项目中，这里应该读取触摸设备或鼠标输入
    data->state = LV_INDEV_STATE_RELEASED;
    data->point.x = 0;
    data->point.y = 0;
#endif
}

} // namespace ui
} // namespace bamboo_cut
