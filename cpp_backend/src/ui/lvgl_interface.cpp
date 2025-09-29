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
    
    // 初始化显示缓冲区 (LVGL v9 API)
    lv_draw_buf_init(&draw_buf_, config_.screen_width, config_.screen_height,
                     LV_COLOR_FORMAT_RGB888, config_.screen_width * 4,
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
    
    // 系统标题
    header_widgets_.system_title = lv_label_create(header_panel_);
    lv_label_set_text(header_widgets_.system_title,
        LV_SYMBOL_IMAGE " 竹子智能切割系统");
    lv_obj_add_style(header_widgets_.system_title, &style_text_title, 0);
    lv_obj_set_style_text_color(header_widgets_.system_title, color_primary_, 0);
    
    // 紧凑的工作流程指示器容器
    lv_obj_t* workflow_container = lv_obj_create(header_panel_);
    lv_obj_set_size(workflow_container, 350, 50);  // 更紧凑
    lv_obj_set_style_bg_opa(workflow_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(workflow_container, 0, 0);
    lv_obj_set_style_pad_all(workflow_container, 8, 0);
    lv_obj_set_flex_flow(workflow_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(workflow_container, LV_FLEX_ALIGN_SPACE_AROUND,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_clear_flag(workflow_container, LV_OBJ_FLAG_SCROLLABLE);
    
    const char* steps[] = {"进料", "识别", "传输", "准备", "切割"};
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
    
    // 心跳状态
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
    lv_label_set_text(header_widgets_.heartbeat_label, LV_SYMBOL_REFRESH " 心跳正常");
    lv_obj_set_style_text_color(header_widgets_.heartbeat_label, color_success_, 0);
    lv_obj_set_style_text_font(header_widgets_.heartbeat_label, &lv_font_montserrat_14, 0);
    lv_obj_center(header_widgets_.heartbeat_label);
    
    // 响应时间
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
    lv_obj_set_size(camera_panel_, lv_pct(60), lv_pct(60));  // 百分比布局
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
    
    // 坐标信息
    camera_widgets_.coord_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.coord_value, LV_SYMBOL_GPS " X: 0.00 Y: 0.00 Z: 0.00");
    lv_obj_set_style_text_color(camera_widgets_.coord_value, color_primary_, 0);
    lv_obj_set_style_text_font(camera_widgets_.coord_value, &lv_font_montserrat_14, 0);
    lv_obj_align(camera_widgets_.coord_value, LV_ALIGN_LEFT_MID, 10, 0);
    
    // 质量评分
    camera_widgets_.quality_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.quality_value, LV_SYMBOL_IMAGE " 质量: 95%");
    lv_obj_set_style_text_color(camera_widgets_.quality_value, color_success_, 0);
    lv_obj_set_style_text_font(camera_widgets_.quality_value, &lv_font_montserrat_14, 0);
    lv_obj_align(camera_widgets_.quality_value, LV_ALIGN_CENTER, 0, 0);
    
    // 刀片信息
    camera_widgets_.blade_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.blade_value, LV_SYMBOL_SETTINGS " 刀片: #3");
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
    lv_obj_set_size(control_panel_, lv_pct(36), lv_pct(60));  // 百分比布局
    lv_obj_align(control_panel_, LV_ALIGN_TOP_RIGHT, -lv_pct(2), 80);
    lv_obj_add_style(control_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(control_panel_, 24, 0);  // 增加内边距，增强呼吸感
    lv_obj_set_style_radius(control_panel_, 16, 0);
    lv_obj_set_flex_flow(control_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START);
    
    // 增加间距，营造呼吸感
    lv_obj_set_style_pad_gap(control_panel_, 20, 0);
    
    // 标题
    lv_obj_t* title = lv_label_create(control_panel_);
    lv_label_set_text(title, LV_SYMBOL_SETTINGS " 控制面板");
    lv_obj_add_style(title, &style_text_title, 0);
    lv_obj_set_style_text_color(title, color_primary_, 0);
    lv_obj_set_style_pad_bottom(title, 8, 0);
    
    // Modbus寄存器表格
    lv_obj_t* modbus_label = lv_label_create(control_panel_);
    lv_label_set_text(modbus_label, "Modbus 寄存器");
    lv_obj_add_style(modbus_label, &style_text_body, 0);
    lv_obj_set_style_pad_top(modbus_label, 15, 0);
    
    control_widgets_.modbus_table = lv_table_create(control_panel_);
    lv_obj_set_size(control_widgets_.modbus_table, lv_pct(100), 120);
    lv_table_set_col_cnt(control_widgets_.modbus_table, 3);
    lv_table_set_row_cnt(control_widgets_.modbus_table, 4);
    
    // 设置列宽
    lv_table_set_col_width(control_widgets_.modbus_table, 0, 150);
    lv_table_set_col_width(control_widgets_.modbus_table, 1, 150);
    lv_table_set_col_width(control_widgets_.modbus_table, 2, 150);
    
    // 表头
    lv_table_set_cell_value(control_widgets_.modbus_table, 0, 0, "寄存器");
    lv_table_set_cell_value(control_widgets_.modbus_table, 0, 1, "数值");
    lv_table_set_cell_value(control_widgets_.modbus_table, 0, 2, "状态");
    
    // 样式
    lv_obj_add_style(control_widgets_.modbus_table, &style_table_cell, LV_PART_ITEMS);
    
    // 刀片选择
    lv_obj_t* blade_label = lv_label_create(control_panel_);
    lv_label_set_text(blade_label, "刀片选择");
    lv_obj_add_style(blade_label, &style_text_body, 0);
    lv_obj_set_style_pad_top(blade_label, 15, 0);
    
    lv_obj_t* blade_container = lv_obj_create(control_panel_);
    lv_obj_set_size(blade_container, lv_pct(100), 80);
    lv_obj_set_style_bg_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(blade_container, 0, 0);
    lv_obj_set_style_pad_all(blade_container, 0, 0);
    lv_obj_set_flex_flow(blade_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(blade_container, LV_FLEX_ALIGN_SPACE_EVENLY, 
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    for(int i = 1; i <= 5; i++) {
        lv_obj_t* btn = lv_btn_create(blade_container);
        lv_obj_set_size(btn, 70, 60);
        lv_obj_add_style(btn, &style_btn_primary, 0);
        lv_obj_add_style(btn, &style_btn_pressed, LV_STATE_PRESSED);
        
        if(i == selected_blade_) {
            lv_obj_set_style_bg_color(btn, color_success_, 0);
            lv_obj_set_style_shadow_color(btn, color_success_, 0);
        }
        
        lv_obj_t* label = lv_label_create(btn);
        lv_label_set_text_fmt(label, "#%d", i);
        lv_obj_set_style_text_font(label, &lv_font_montserrat_16, 0);
        lv_obj_center(label);
        
        lv_obj_add_event_cb(btn, onBladeSelectionChanged, LV_EVENT_CLICKED, this);
        control_widgets_.blade_buttons.push_back(btn);
    }
    
    // 系统信息
    lv_obj_t* stats_label = lv_label_create(control_panel_);
    lv_label_set_text(stats_label, "系统信息");
    lv_obj_add_style(stats_label, &style_text_body, 0);
    lv_obj_set_style_pad_top(stats_label, 15, 0);
    
    lv_obj_t* stats_container = lv_obj_create(control_panel_);
    lv_obj_set_size(stats_container, lv_pct(100), 100);
    lv_obj_set_style_bg_color(stats_container, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(stats_container, 8, 0);
    lv_obj_set_style_border_width(stats_container, 1, 0);
    lv_obj_set_style_border_color(stats_container, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(stats_container, 15, 0);
    
    lv_obj_t* cpu_label = lv_label_create(stats_container);
    lv_label_set_text(cpu_label, LV_SYMBOL_CHARGE " CPU: 45%");
    lv_obj_set_style_text_color(cpu_label, color_success_, 0);
    lv_obj_set_style_text_font(cpu_label, &lv_font_montserrat_14, 0);
    lv_obj_align(cpu_label, LV_ALIGN_TOP_LEFT, 0, 0);
    
    lv_obj_t* mem_label = lv_label_create(stats_container);
    lv_label_set_text(mem_label, LV_SYMBOL_SD_CARD " MEM: 2.1GB");
    lv_obj_set_style_text_color(mem_label, color_warning_, 0);
    lv_obj_set_style_text_font(mem_label, &lv_font_montserrat_14, 0);
    lv_obj_align(mem_label, LV_ALIGN_TOP_LEFT, 0, 25);
    
    lv_obj_t* temp_label = lv_label_create(stats_container);
    lv_label_set_text(temp_label, LV_SYMBOL_WARNING " 温度: 62°C");
    lv_obj_set_style_text_color(temp_label, lv_color_hex(0xFFAB00), 0);
    lv_obj_set_style_text_font(temp_label, &lv_font_montserrat_14, 0);
    lv_obj_align(temp_label, LV_ALIGN_TOP_LEFT, 0, 50);
    
    return control_panel_;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLInterface::createStatusPanel() {
#ifdef ENABLE_LVGL
    // 这个版本中状态信息已集成到控制面板中
    return nullptr;
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
    
    // 启动按钮
    footer_widgets_.start_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.start_btn, 110, 48);  // 更紧凑
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_success, 0);
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* start_label = lv_label_create(footer_widgets_.start_btn);
    lv_label_set_text(start_label, LV_SYMBOL_PLAY " 启动");
    lv_obj_set_style_text_font(start_label, &lv_font_montserrat_16, 0);
    lv_obj_center(start_label);
    lv_obj_add_event_cb(footer_widgets_.start_btn, onStartButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 暂停按钮
    footer_widgets_.pause_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.pause_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_warning, 0);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* pause_label = lv_label_create(footer_widgets_.pause_btn);
    lv_label_set_text(pause_label, LV_SYMBOL_PAUSE " 暂停");
    lv_obj_set_style_text_font(pause_label, &lv_font_montserrat_16, 0);
    lv_obj_center(pause_label);
    lv_obj_add_event_cb(footer_widgets_.pause_btn, onPauseButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 停止按钮
    footer_widgets_.stop_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.stop_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_primary, 0);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_pressed, LV_STATE_PRESSED);
    lv_obj_set_style_bg_color(footer_widgets_.stop_btn, lv_color_hex(0x6B7280), 0);
    
    lv_obj_t* stop_label = lv_label_create(footer_widgets_.stop_btn);
    lv_label_set_text(stop_label, LV_SYMBOL_STOP " 停止");
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
    // 这里应该从DataBridge获取实际数据
    // 示例：更新响应时间标签
    if (header_widgets_.response_label) {
        static int counter = 0;
        int response_ms = 12 + (counter++ % 10);
        lv_label_set_text_fmt(header_widgets_.response_label, 
            LV_SYMBOL_LOOP " %dms", response_ms);
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
    // 实际的DRM显示刷新实现
    static int drm_fd = -1;
    static uint32_t fb_id = 0;
    static drmModeCrtc* crtc = nullptr;
    static drmModeConnector* connector = nullptr;
    static uint32_t* framebuffer = nullptr;
    static uint32_t fb_handle = 0;
    static bool drm_initialized = false;
    static uint32_t drm_width = 0;
    static uint32_t drm_height = 0;
    
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
                            // 找到CRTC
                            if (connector->encoder_id) {
                                drmModeEncoder* encoder = drmModeGetEncoder(drm_fd, connector->encoder_id);
                                if (encoder && encoder->crtc_id) {
                                    crtc = drmModeGetCrtc(drm_fd, encoder->crtc_id);
                                    if (crtc) {
                                        drm_width = crtc->width;
                                        drm_height = crtc->height;
                                        std::cout << "[DRM] 找到CRTC: " << drm_width << "x" << drm_height << std::endl;
                                    }
                                    drmModeFreeEncoder(encoder);
                                    break;
                                }
                            }
                        }
                    }
                    
                    // 创建dumb buffer
                    if (crtc && connector && drm_width > 0 && drm_height > 0) {
                        struct drm_mode_create_dumb create_req = {};
                        create_req.width = drm_width;
                        create_req.height = drm_height;
                        create_req.bpp = 32;
                        
                        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) == 0) {
                            fb_handle = create_req.handle;
                            uint32_t stride = create_req.pitch;
                            uint32_t size = create_req.size;
                            
                            // 创建framebuffer对象
                            if (drmModeAddFB(drm_fd, drm_width, drm_height, 24, 32, stride, fb_handle, &fb_id) == 0) {
                                // 映射framebuffer
                                struct drm_mode_map_dumb map_req = {};
                                map_req.handle = fb_handle;
                                
                                if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) == 0) {
                                    framebuffer = (uint32_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_req.offset);
                                    if (framebuffer != MAP_FAILED) {
                                        // 设置CRTC使用我们的framebuffer
                                        int ret = drmModeSetCrtc(drm_fd, crtc->crtc_id, fb_id, 0, 0, &connector->connector_id, 1, &connector->modes[0]);
                                        if (ret == 0) {
                                            drm_initialized = true;
                                            std::cout << "[DRM] DRM framebuffer初始化成功: " << drm_width << "x" << drm_height << std::endl;
                                            // 清空framebuffer (黑色背景)
                                            memset(framebuffer, 0, size);
                                        } else {
                                            std::cerr << "[DRM] drmModeSetCrtc失败: " << ret << std::endl;
                                        }
                                    } else {
                                        std::cerr << "[DRM] framebuffer映射失败" << std::endl;
                                    }
                                }
                            }
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
    
    // 将LVGL像素数据复制到DRM framebuffer
    if (drm_initialized && framebuffer && drm_width > 0 && drm_height > 0) {
        uint32_t area_width = area->x2 - area->x1 + 1;
        uint32_t area_height = area->y2 - area->y1 + 1;
        
        // 确保坐标在有效范围内
        if (area->x1 >= 0 && area->y1 >= 0 &&
            area->x1 < (int32_t)drm_width && area->y1 < (int32_t)drm_height) {
            
            for (uint32_t y = 0; y < area_height; y++) {
                if ((area->y1 + y) >= (int32_t)drm_height) break;
                
                for (uint32_t x = 0; x < area_width; x++) {
                    if ((area->x1 + x) >= (int32_t)drm_width) break;
                    
                    // 计算源像素和目标像素位置
                    uint32_t src_idx = (y * area_width + x);
                    uint32_t dst_idx = (area->y1 + y) * drm_width + (area->x1 + x);
                    
                    // LVGL使用lv_color_t格式，需要转换为DRM格式
                    lv_color_t* src_color = (lv_color_t*)px_map + src_idx;
                    
                    // 转换为32位ARGB格式 (DRM格式)
                    uint32_t pixel = 0xFF000000 | // Alpha = 255 (不透明)
                                    (src_color->red << 16) |
                                    (src_color->green << 8) |
                                    (src_color->blue);
                    
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
