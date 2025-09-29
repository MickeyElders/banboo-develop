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
    , jetson_monitor_(std::make_shared<utils::JetsonMonitor>())
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
    
    // 启动Jetson系统监控
    if (jetson_monitor_) {
        jetson_monitor_->start();
    }
    
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
    
    // 验证屏幕尺寸
    if (config_.screen_width <= 0 || config_.screen_height <= 0 ||
        config_.screen_width > 4096 || config_.screen_height > 4096) {
        std::cerr << "[LVGLInterface] 无效的屏幕尺寸: " << config_.screen_width
                  << "x" << config_.screen_height << std::endl;
        return false;
    }
    
    // 安全分配显示缓冲区
    uint32_t buf_size = config_.screen_width * config_.screen_height;
    
    // 检查缓冲区大小是否合理（不超过64MB）
    if (buf_size > 16 * 1024 * 1024) {
        std::cerr << "[LVGLInterface] 缓冲区大小过大: " << buf_size << " 像素" << std::endl;
        return false;
    }
    
    try {
        disp_buf1_ = new(std::nothrow) lv_color_t[buf_size];
        disp_buf2_ = new(std::nothrow) lv_color_t[buf_size];
        
        if (!disp_buf1_ || !disp_buf2_) {
            std::cerr << "[LVGLInterface] 显示缓冲区分配失败" << std::endl;
            if (disp_buf1_) { delete[] disp_buf1_; disp_buf1_ = nullptr; }
            if (disp_buf2_) { delete[] disp_buf2_; disp_buf2_ = nullptr; }
            return false;
        }
        
        // 初始化缓冲区为黑色
        memset(disp_buf1_, 0, buf_size * sizeof(lv_color_t));
        memset(disp_buf2_, 0, buf_size * sizeof(lv_color_t));
        
    } catch (const std::bad_alloc& e) {
        std::cerr << "[LVGLInterface] 内存分配异常: " << e.what() << std::endl;
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
    
    // 创建头部面板
    createHeaderPanel();
    
    // 创建中间内容容器（使用Flex布局管理左右面板）
    lv_obj_t* content_container = lv_obj_create(main_screen_);
    lv_obj_set_size(content_container, lv_pct(98), lv_pct(85));
    lv_obj_align(content_container, LV_ALIGN_TOP_MID, 0, 80);
    lv_obj_set_style_bg_opa(content_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(content_container, 0, 0);
    lv_obj_set_style_pad_all(content_container, 5, 0);
    lv_obj_clear_flag(content_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置Flex布局：水平排列，左右分布
    lv_obj_set_flex_flow(content_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(content_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(content_container, 10, 0);
    
    // 在容器内创建左右面板
    createCameraPanel(content_container);
    createControlPanel(content_container);
    
    // 创建底部面板
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

lv_obj_t* LVGLInterface::createCameraPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* container = parent ? parent : main_screen_;
    
    camera_panel_ = lv_obj_create(container);
    lv_obj_set_size(camera_panel_, lv_pct(73), lv_pct(100));  // 在容器内占73%宽度，100%高度
    lv_obj_set_flex_grow(camera_panel_, 3);  // 占据更多空间
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

lv_obj_t* LVGLInterface::createControlPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* container = parent ? parent : main_screen_;
    
    control_panel_ = lv_obj_create(container);
    lv_obj_set_size(control_panel_, lv_pct(25), lv_pct(100));  // 在容器内占25%宽度，100%高度
    lv_obj_set_flex_grow(control_panel_, 1);  // 占据较少空间
    lv_obj_add_style(control_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(control_panel_, 15, 0);  // 减少内边距以容纳更多内容
    lv_obj_set_style_radius(control_panel_, 16, 0);
    lv_obj_set_flex_flow(control_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(control_panel_, 12, 0);  // 减少组件间距
    
    // Control Panel Title
    lv_obj_t* title = lv_label_create(control_panel_);
    lv_label_set_text(title, LV_SYMBOL_SETTINGS " System Control");
    lv_obj_add_style(title, &style_text_title, 0);
    lv_obj_set_style_text_color(title, color_primary_, 0);
    
    // === Jetson System Monitoring Area ===
    lv_obj_t* jetson_section = lv_obj_create(control_panel_);
    lv_obj_set_size(jetson_section, lv_pct(100), 320);  // 增加高度以容纳更多信息
    lv_obj_set_style_bg_color(jetson_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(jetson_section, 12, 0);
    lv_obj_set_style_border_width(jetson_section, 1, 0);
    lv_obj_set_style_border_color(jetson_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(jetson_section, 12, 0);
    lv_obj_clear_flag(jetson_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* jetson_label = lv_label_create(jetson_section);
    lv_label_set_text(jetson_label, LV_SYMBOL_CHARGE " Jetson Orin Nano");
    lv_obj_set_style_text_color(jetson_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(jetson_label, &lv_font_montserrat_14, 0);
    lv_obj_align(jetson_label, LV_ALIGN_TOP_LEFT, 0, 0);
    
    int y_pos = 25;
    
    // === CPU信息区域 - 表格式布局 ===
    // CPU标签行容器
    lv_obj_t* cpu_label_row = lv_obj_create(jetson_section);
    lv_obj_set_size(cpu_label_row, lv_pct(100), 18);
    lv_obj_set_pos(cpu_label_row, 0, y_pos);
    lv_obj_set_style_bg_opa(cpu_label_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cpu_label_row, 0, 0);
    lv_obj_set_style_pad_all(cpu_label_row, 0, 0);
    lv_obj_clear_flag(cpu_label_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(cpu_label_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cpu_label_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // CPU标签 (左对齐)
    lv_obj_t* cpu_name_label = lv_label_create(cpu_label_row);
    lv_label_set_text(cpu_name_label, "CPU:");
    lv_obj_set_style_text_color(cpu_name_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(cpu_name_label, &lv_font_montserrat_12, 0);
    
    // CPU进度条 (右对齐)
    control_widgets_.cpu_bar = lv_bar_create(cpu_label_row);
    lv_obj_set_size(control_widgets_.cpu_bar, 120, 16);
    lv_bar_set_range(control_widgets_.cpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.cpu_bar, 45, LV_ANIM_ON);
    // 设置进度条主体背景（未使用部分）
    lv_obj_set_style_bg_color(control_widgets_.cpu_bar, lv_color_hex(0x2A3441), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(control_widgets_.cpu_bar, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_radius(control_widgets_.cpu_bar, 8, LV_PART_MAIN);
    // 设置进度条指示器（已使用部分）为黄色
    lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
    lv_obj_set_style_bg_opa(control_widgets_.cpu_bar, LV_OPA_COVER, LV_PART_INDICATOR);
    lv_obj_set_style_radius(control_widgets_.cpu_bar, 8, LV_PART_INDICATOR);
    
    // CPU数值行容器
    lv_obj_t* cpu_value_row = lv_obj_create(jetson_section);
    lv_obj_set_size(cpu_value_row, lv_pct(100), 18);
    lv_obj_set_pos(cpu_value_row, 0, y_pos + 22);
    lv_obj_set_style_bg_opa(cpu_value_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cpu_value_row, 0, 0);
    lv_obj_set_style_pad_all(cpu_value_row, 0, 0);
    lv_obj_clear_flag(cpu_value_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(cpu_value_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cpu_value_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // CPU数值 (左侧)
    control_widgets_.cpu_label = lv_label_create(cpu_value_row);
    lv_label_set_text(control_widgets_.cpu_label, "45%@1.9GHz");
    lv_obj_set_style_text_color(control_widgets_.cpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_label, &lv_font_montserrat_12, 0);
    
    // CPU温度 (右侧)
    control_widgets_.cpu_temp_label = lv_label_create(cpu_value_row);
    lv_label_set_text(control_widgets_.cpu_temp_label, "62°C");
    lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_temp_label, &lv_font_montserrat_12, 0);
    
    y_pos += 54;
    
    // === GPU信息区域 - 表格式布局 ===
    // GPU标签行容器
    lv_obj_t* gpu_label_row = lv_obj_create(jetson_section);
    lv_obj_set_size(gpu_label_row, lv_pct(100), 18);
    lv_obj_set_pos(gpu_label_row, 0, y_pos);
    lv_obj_set_style_bg_opa(gpu_label_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(gpu_label_row, 0, 0);
    lv_obj_set_style_pad_all(gpu_label_row, 0, 0);
    lv_obj_clear_flag(gpu_label_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(gpu_label_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(gpu_label_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // GPU标签 (左对齐)
    lv_obj_t* gpu_name_label = lv_label_create(gpu_label_row);
    lv_label_set_text(gpu_name_label, "GPU:");
    lv_obj_set_style_text_color(gpu_name_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(gpu_name_label, &lv_font_montserrat_12, 0);
    
    // GPU进度条 (右对齐)
    control_widgets_.gpu_bar = lv_bar_create(gpu_label_row);
    lv_obj_set_size(control_widgets_.gpu_bar, 120, 16);
    lv_bar_set_range(control_widgets_.gpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.gpu_bar, 72, LV_ANIM_ON);
    // 设置进度条主体背景（未使用部分）
    lv_obj_set_style_bg_color(control_widgets_.gpu_bar, lv_color_hex(0x2A3441), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(control_widgets_.gpu_bar, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_radius(control_widgets_.gpu_bar, 8, LV_PART_MAIN);
    // 设置进度条指示器（已使用部分）为黄色
    lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
    lv_obj_set_style_bg_opa(control_widgets_.gpu_bar, LV_OPA_COVER, LV_PART_INDICATOR);
    lv_obj_set_style_radius(control_widgets_.gpu_bar, 8, LV_PART_INDICATOR);
    
    // GPU数值行容器
    lv_obj_t* gpu_value_row = lv_obj_create(jetson_section);
    lv_obj_set_size(gpu_value_row, lv_pct(100), 18);
    lv_obj_set_pos(gpu_value_row, 0, y_pos + 22);
    lv_obj_set_style_bg_opa(gpu_value_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(gpu_value_row, 0, 0);
    lv_obj_set_style_pad_all(gpu_value_row, 0, 0);
    lv_obj_clear_flag(gpu_value_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(gpu_value_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(gpu_value_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // GPU数值 (左侧)
    control_widgets_.gpu_label = lv_label_create(gpu_value_row);
    lv_label_set_text(control_widgets_.gpu_label, "72%@624MHz");
    lv_obj_set_style_text_color(control_widgets_.gpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_label, &lv_font_montserrat_12, 0);
    
    // GPU温度 (右侧)
    control_widgets_.gpu_temp_label = lv_label_create(gpu_value_row);
    lv_label_set_text(control_widgets_.gpu_temp_label, "58°C");
    lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_temp_label, &lv_font_montserrat_12, 0);
    
    y_pos += 54;
    
    // === 内存信息区域 - 表格式布局 ===
    // 内存标签行容器
    lv_obj_t* mem_label_row = lv_obj_create(jetson_section);
    lv_obj_set_size(mem_label_row, lv_pct(100), 18);
    lv_obj_set_pos(mem_label_row, 0, y_pos);
    lv_obj_set_style_bg_opa(mem_label_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(mem_label_row, 0, 0);
    lv_obj_set_style_pad_all(mem_label_row, 0, 0);
    lv_obj_clear_flag(mem_label_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(mem_label_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(mem_label_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 内存标签 (左对齐)
    lv_obj_t* mem_name_label = lv_label_create(mem_label_row);
    lv_label_set_text(mem_name_label, "RAM:");
    lv_obj_set_style_text_color(mem_name_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(mem_name_label, &lv_font_montserrat_12, 0);
    
    // 内存进度条 (右对齐)
    control_widgets_.mem_bar = lv_bar_create(mem_label_row);
    lv_obj_set_size(control_widgets_.mem_bar, 120, 16);
    lv_bar_set_range(control_widgets_.mem_bar, 0, 100);
    lv_bar_set_value(control_widgets_.mem_bar, 58, LV_ANIM_ON);
    // 设置进度条主体背景（未使用部分）
    lv_obj_set_style_bg_color(control_widgets_.mem_bar, lv_color_hex(0x2A3441), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(control_widgets_.mem_bar, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_radius(control_widgets_.mem_bar, 8, LV_PART_MAIN);
    // 设置进度条指示器（已使用部分）为黄色
    lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
    lv_obj_set_style_bg_opa(control_widgets_.mem_bar, LV_OPA_COVER, LV_PART_INDICATOR);
    lv_obj_set_style_radius(control_widgets_.mem_bar, 8, LV_PART_INDICATOR);
    
    // 内存数值行容器
    lv_obj_t* mem_value_row = lv_obj_create(jetson_section);
    lv_obj_set_size(mem_value_row, lv_pct(100), 18);
    lv_obj_set_pos(mem_value_row, 0, y_pos + 22);
    lv_obj_set_style_bg_opa(mem_value_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(mem_value_row, 0, 0);
    lv_obj_set_style_pad_all(mem_value_row, 0, 0);
    lv_obj_clear_flag(mem_value_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(mem_value_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(mem_value_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 内存数值 (左侧)
    control_widgets_.mem_label = lv_label_create(mem_value_row);
    lv_label_set_text(control_widgets_.mem_label, "3347/7467MB");
    lv_obj_set_style_text_color(control_widgets_.mem_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.mem_label, &lv_font_montserrat_12, 0);
    
    // SWAP使用情况 (右侧)
    control_widgets_.swap_usage_label = lv_label_create(mem_value_row);
    lv_label_set_text(control_widgets_.swap_usage_label, "SWAP: 0/3733MB");
    lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.swap_usage_label, &lv_font_montserrat_12, 0);
    
    y_pos += 54;
    
    // === 温度信息区域 - 表格式布局 ===
    lv_obj_t* temp_row = lv_obj_create(jetson_section);
    lv_obj_set_size(temp_row, lv_pct(100), 18);
    lv_obj_set_pos(temp_row, 0, y_pos);
    lv_obj_set_style_bg_opa(temp_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(temp_row, 0, 0);
    lv_obj_set_style_pad_all(temp_row, 0, 0);
    lv_obj_clear_flag(temp_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(temp_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(temp_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // SOC温度 (左侧)
    control_widgets_.soc_temp_label = lv_label_create(temp_row);
    lv_label_set_text(control_widgets_.soc_temp_label, "SOC: 48.5°C");
    lv_obj_set_style_text_color(control_widgets_.soc_temp_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.soc_temp_label, &lv_font_montserrat_12, 0);
    
    // 热区温度 (右侧)
    control_widgets_.thermal_temp_label = lv_label_create(temp_row);
    lv_label_set_text(control_widgets_.thermal_temp_label, "Thermal: 50.2°C");
    lv_obj_set_style_text_color(control_widgets_.thermal_temp_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.thermal_temp_label, &lv_font_montserrat_12, 0);
    
    y_pos += 25;
    
    // === 电源信息区域 - 表格式布局 ===
    // VDD_IN电源行
    lv_obj_t* power_in_row = lv_obj_create(jetson_section);
    lv_obj_set_size(power_in_row, lv_pct(100), 18);
    lv_obj_set_pos(power_in_row, 0, y_pos);
    lv_obj_set_style_bg_opa(power_in_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(power_in_row, 0, 0);
    lv_obj_set_style_pad_all(power_in_row, 0, 0);
    lv_obj_clear_flag(power_in_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(power_in_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(power_in_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* power_in_label_name = lv_label_create(power_in_row);
    lv_label_set_text(power_in_label_name, "VDD_IN:");
    lv_obj_set_style_text_color(power_in_label_name, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(power_in_label_name, &lv_font_montserrat_12, 0);
    
    control_widgets_.power_in_label = lv_label_create(power_in_row);
    lv_label_set_text(control_widgets_.power_in_label, "5336mA/4970mW");
    lv_obj_set_style_text_color(control_widgets_.power_in_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_in_label, &lv_font_montserrat_12, 0);
    
    y_pos += 20;
    
    // CPU_GPU电源行
    lv_obj_t* power_cpu_gpu_row = lv_obj_create(jetson_section);
    lv_obj_set_size(power_cpu_gpu_row, lv_pct(100), 18);
    lv_obj_set_pos(power_cpu_gpu_row, 0, y_pos);
    lv_obj_set_style_bg_opa(power_cpu_gpu_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(power_cpu_gpu_row, 0, 0);
    lv_obj_set_style_pad_all(power_cpu_gpu_row, 0, 0);
    lv_obj_clear_flag(power_cpu_gpu_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(power_cpu_gpu_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(power_cpu_gpu_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* power_cpu_gpu_label_name = lv_label_create(power_cpu_gpu_row);
    lv_label_set_text(power_cpu_gpu_label_name, "CPU_GPU:");
    lv_obj_set_style_text_color(power_cpu_gpu_label_name, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(power_cpu_gpu_label_name, &lv_font_montserrat_12, 0);
    
    control_widgets_.power_cpu_gpu_label = lv_label_create(power_cpu_gpu_row);
    lv_label_set_text(control_widgets_.power_cpu_gpu_label, "2617mA/2336mW");
    lv_obj_set_style_text_color(control_widgets_.power_cpu_gpu_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_cpu_gpu_label, &lv_font_montserrat_12, 0);
    
    y_pos += 20;
    
    // SOC电源行
    lv_obj_t* power_soc_row = lv_obj_create(jetson_section);
    lv_obj_set_size(power_soc_row, lv_pct(100), 18);
    lv_obj_set_pos(power_soc_row, 0, y_pos);
    lv_obj_set_style_bg_opa(power_soc_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(power_soc_row, 0, 0);
    lv_obj_set_style_pad_all(power_soc_row, 0, 0);
    lv_obj_clear_flag(power_soc_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(power_soc_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(power_soc_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* power_soc_label_name = lv_label_create(power_soc_row);
    lv_label_set_text(power_soc_label_name, "SOC:");
    lv_obj_set_style_text_color(power_soc_label_name, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(power_soc_label_name, &lv_font_montserrat_12, 0);
    
    control_widgets_.power_soc_label = lv_label_create(power_soc_row);
    lv_label_set_text(control_widgets_.power_soc_label, "1478mA/1318mW");
    lv_obj_set_style_text_color(control_widgets_.power_soc_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_soc_label, &lv_font_montserrat_12, 0);
    
    y_pos += 25;
    
    // === 其他系统信息 - 表格式布局 ===
    // EMC和VIC频率行
    lv_obj_t* freq_row = lv_obj_create(jetson_section);
    lv_obj_set_size(freq_row, lv_pct(100), 18);
    lv_obj_set_pos(freq_row, 0, y_pos);
    lv_obj_set_style_bg_opa(freq_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(freq_row, 0, 0);
    lv_obj_set_style_pad_all(freq_row, 0, 0);
    lv_obj_clear_flag(freq_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(freq_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(freq_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // EMC频率 (左侧)
    control_widgets_.emc_freq_label = lv_label_create(freq_row);
    lv_label_set_text(control_widgets_.emc_freq_label, "EMC: 13%@2133MHz");
    lv_obj_set_style_text_color(control_widgets_.emc_freq_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.emc_freq_label, &lv_font_montserrat_12, 0);
    
    // VIC使用率 (右侧)
    control_widgets_.vic_usage_label = lv_label_create(freq_row);
    lv_label_set_text(control_widgets_.vic_usage_label, "VIC: 0%@115MHz");
    lv_obj_set_style_text_color(control_widgets_.vic_usage_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.vic_usage_label, &lv_font_montserrat_12, 0);
    
    y_pos += 20;
    
    // 风扇转速行
    lv_obj_t* fan_row = lv_obj_create(jetson_section);
    lv_obj_set_size(fan_row, lv_pct(100), 18);
    lv_obj_set_pos(fan_row, 0, y_pos);
    lv_obj_set_style_bg_opa(fan_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(fan_row, 0, 0);
    lv_obj_set_style_pad_all(fan_row, 0, 0);
    lv_obj_clear_flag(fan_row, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(fan_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(fan_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* fan_label_name = lv_label_create(fan_row);
    lv_label_set_text(fan_label_name, "FAN:");
    lv_obj_set_style_text_color(fan_label_name, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(fan_label_name, &lv_font_montserrat_12, 0);
    
    control_widgets_.fan_speed_label = lv_label_create(fan_row);
    lv_label_set_text(control_widgets_.fan_speed_label, "N/A");
    lv_obj_set_style_text_color(control_widgets_.fan_speed_label, lv_color_hex(0x8A92A1), 0);
    lv_obj_set_style_text_font(control_widgets_.fan_speed_label, &lv_font_montserrat_12, 0);
    
    // === AI Model Monitoring Area ===
    lv_obj_t* ai_section = lv_obj_create(control_panel_);
    lv_obj_set_size(ai_section, lv_pct(100), 340);  // 增加高度以容纳摄像头状态信息
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
    
    // === 基本AI模型信息 (上半部分) ===
    // 模型版本
    control_widgets_.ai_model_version_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_model_version_label, "模型版本: YOLOv8n");
    lv_obj_set_style_text_color(control_widgets_.ai_model_version_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_model_version_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_model_version_label, LV_ALIGN_TOP_LEFT, 0, 25);
    
    // 推理时间
    control_widgets_.ai_inference_time_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_inference_time_label, "推理时间: 18.3ms");
    lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_inference_time_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_inference_time_label, LV_ALIGN_TOP_LEFT, 0, 45);
    
    // 置信阈值
    control_widgets_.ai_confidence_threshold_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_confidence_threshold_label, "置信阈值: 0.85");
    lv_obj_set_style_text_color(control_widgets_.ai_confidence_threshold_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_confidence_threshold_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_confidence_threshold_label, LV_ALIGN_TOP_LEFT, 0, 65);
    
    // 检测精度
    control_widgets_.ai_detection_accuracy_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_detection_accuracy_label, "检测精度: 94.2%");
    lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_detection_accuracy_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_detection_accuracy_label, LV_ALIGN_TOP_RIGHT, 0, 25);
    
    // 总检测数
    control_widgets_.ai_total_detections_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_total_detections_label, "总检测数: 15,432");
    lv_obj_set_style_text_color(control_widgets_.ai_total_detections_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_total_detections_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_total_detections_label, LV_ALIGN_TOP_RIGHT, 0, 45);
    
    // 今日检测数
    control_widgets_.ai_daily_detections_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_daily_detections_label, "今日检测: 89");
    lv_obj_set_style_text_color(control_widgets_.ai_daily_detections_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_daily_detections_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.ai_daily_detections_label, LV_ALIGN_TOP_RIGHT, 0, 65);
    
    // === 分隔线 ===
    lv_obj_t* separator = lv_obj_create(ai_section);
    lv_obj_set_size(separator, lv_pct(100), 1);
    lv_obj_set_pos(separator, 0, 90);
    lv_obj_set_style_bg_color(separator, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_border_width(separator, 0, 0);
    lv_obj_clear_flag(separator, LV_OBJ_FLAG_SCROLLABLE);
    
    // === 当前竹子检测状态 (下半部分) ===
    lv_obj_t* bamboo_status_title = lv_label_create(ai_section);
    lv_label_set_text(bamboo_status_title, LV_SYMBOL_CHARGE " 当前竹子：");
    lv_obj_set_style_text_color(bamboo_status_title, color_warning_, 0);
    lv_obj_set_style_text_font(bamboo_status_title, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(bamboo_status_title, 0, 100);
    
    // 竹子直径
    control_widgets_.bamboo_diameter_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_diameter_label, "- 直径：45.2mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_diameter_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.bamboo_diameter_label, 0, 120);
    
    // 竹子长度
    control_widgets_.bamboo_length_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_length_label, "- 长度：2850mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_length_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.bamboo_length_label, 0, 140);
    
    // 预切位置
    control_widgets_.bamboo_cut_positions_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "- 预切位置：[250mm, 1450mm, 2650mm]");
    lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_cut_positions_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.bamboo_cut_positions_label, 0, 160);
    
    // 检测置信度和耗时 (同一行)
    control_widgets_.bamboo_confidence_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_confidence_label, "- 置信度：0.96");
    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_confidence_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.bamboo_confidence_label, 0, 180);
    
    control_widgets_.bamboo_detection_time_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_detection_time_label, "- 检测耗时：16.8ms");
    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_detection_time_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.bamboo_detection_time_label, control_widgets_.bamboo_confidence_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    // === 第二分隔线 ===
    lv_obj_t* separator2 = lv_obj_create(ai_section);
    lv_obj_set_size(separator2, lv_pct(100), 1);
    lv_obj_set_pos(separator2, 0, 205);
    lv_obj_set_style_bg_color(separator2, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_border_width(separator2, 0, 0);
    lv_obj_clear_flag(separator2, LV_OBJ_FLAG_SCROLLABLE);
    
    // === 摄像头系统状态 ===
    lv_obj_t* camera_status_title = lv_label_create(ai_section);
    lv_label_set_text(camera_status_title, LV_SYMBOL_EYE_OPEN " 摄像头状态：");
    lv_obj_set_style_text_color(camera_status_title, color_primary_, 0);
    lv_obj_set_style_text_font(camera_status_title, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(camera_status_title, 0, 215);
    
    // === 摄像头-1状态 ===
    control_widgets_.camera1_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_status_label, "摄像头-1：在线 ✓");
    lv_obj_set_style_text_color(control_widgets_.camera1_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_status_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.camera1_status_label, 0, 235);
    
    control_widgets_.camera1_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_fps_label, "  帧率：30 FPS");
    lv_obj_set_style_text_color(control_widgets_.camera1_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_fps_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.camera1_fps_label, 0, 250);
    
    control_widgets_.camera1_resolution_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_resolution_label, "  分辨率：1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera1_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_resolution_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.camera1_resolution_label, 0, 265);
    
    control_widgets_.camera1_exposure_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_exposure_label, "  曝光：自动");
    lv_obj_set_style_text_color(control_widgets_.camera1_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_exposure_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.camera1_exposure_label, 0, 280);
    
    control_widgets_.camera1_lighting_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_lighting_label, "  光照评分：良好");
    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_lighting_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.camera1_lighting_label, 0, 295);
    
    // === 摄像头-2状态 ===
    control_widgets_.camera2_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_status_label, "摄像头-2：在线 ✓");
    lv_obj_set_style_text_color(control_widgets_.camera2_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_status_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.camera2_status_label, control_widgets_.camera1_status_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    control_widgets_.camera2_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_fps_label, "  帧率：30 FPS");
    lv_obj_set_style_text_color(control_widgets_.camera2_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_fps_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.camera2_fps_label, control_widgets_.camera1_fps_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    control_widgets_.camera2_resolution_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_resolution_label, "  分辨率：1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera2_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_resolution_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.camera2_resolution_label, control_widgets_.camera1_resolution_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    control_widgets_.camera2_exposure_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_exposure_label, "  曝光：自动");
    lv_obj_set_style_text_color(control_widgets_.camera2_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_exposure_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.camera2_exposure_label, control_widgets_.camera1_exposure_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    control_widgets_.camera2_lighting_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_lighting_label, "  光照评分：良好");
    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_lighting_label, &lv_font_montserrat_12, 0);
    lv_obj_align_to(control_widgets_.camera2_lighting_label, control_widgets_.camera1_lighting_label, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    
    // === Modbus Communication Statistics Area ===
    lv_obj_t* modbus_section = lv_obj_create(control_panel_);
    lv_obj_set_size(modbus_section, lv_pct(100), 200);  // 增加高度以容纳寄存器信息
    lv_obj_set_style_bg_color(modbus_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(modbus_section, 12, 0);
    lv_obj_set_style_border_width(modbus_section, 1, 0);
    lv_obj_set_style_border_color(modbus_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(modbus_section, 10, 0);
    lv_obj_clear_flag(modbus_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* modbus_title = lv_label_create(modbus_section);
    lv_label_set_text(modbus_title, LV_SYMBOL_WIFI " Modbus通信状态");
    lv_obj_set_style_text_color(modbus_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(modbus_title, &lv_font_montserrat_12, 0);
    lv_obj_align(modbus_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // 连接统计信息（第一行）
    control_widgets_.modbus_connection_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_connection_label, "连接: 02:15:32");
    lv_obj_set_style_text_color(control_widgets_.modbus_connection_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_connection_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_connection_label, LV_ALIGN_TOP_LEFT, 0, 18);
    
    control_widgets_.modbus_packets_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_packets_label, "数据包: 1247");
    lv_obj_set_style_text_color(control_widgets_.modbus_packets_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_packets_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_packets_label, LV_ALIGN_TOP_LEFT, 0, 34);
    
    control_widgets_.modbus_errors_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_errors_label, "错误率: 0.02%");
    lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_errors_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_errors_label, LV_ALIGN_TOP_RIGHT, 0, 18);
    
    control_widgets_.modbus_heartbeat_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_heartbeat_label, "心跳: OK");
    lv_obj_set_style_text_color(control_widgets_.modbus_heartbeat_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_heartbeat_label, &lv_font_montserrat_12, 0);
    lv_obj_align(control_widgets_.modbus_heartbeat_label, LV_ALIGN_TOP_RIGHT, 0, 34);
    
    // === Modbus寄存器状态区域 ===
    int reg_y_start = 55;
    
    // 系统状态 (40001)
    control_widgets_.modbus_system_status_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_system_status_label, "40001 系统状态: 运行");
    lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_system_status_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_system_status_label, 0, reg_y_start);
    
    // PLC命令 (40002)
    control_widgets_.modbus_plc_command_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_plc_command_label, "40002 PLC命令: 无");
    lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_plc_command_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_plc_command_label, 0, reg_y_start + 16);
    
    // 坐标就绪 (40003)
    control_widgets_.modbus_coord_ready_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_coord_ready_label, "40003 坐标就绪: 否");
    lv_obj_set_style_text_color(control_widgets_.modbus_coord_ready_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_coord_ready_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_coord_ready_label, 0, reg_y_start + 32);
    
    // X坐标 (40004-05)
    control_widgets_.modbus_x_coordinate_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_x_coordinate_label, "40004 X坐标: 0.0mm");
    lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_x_coordinate_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_x_coordinate_label, 0, reg_y_start + 48);
    
    // 切割质量 (40006)
    control_widgets_.modbus_cut_quality_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_cut_quality_label, "40006 切割质量: 正常");
    lv_obj_set_style_text_color(control_widgets_.modbus_cut_quality_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_cut_quality_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_cut_quality_label, 0, reg_y_start + 64);
    
    // 刀片编号 (40009)
    control_widgets_.modbus_blade_number_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_blade_number_label, "40009 刀片编号: 3");
    lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_blade_number_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_blade_number_label, 0, reg_y_start + 80);
    
    // 健康状态 (40010)
    control_widgets_.modbus_health_status_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_health_status_label, "40010 健康状态: 正常");
    lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_health_status_label, &lv_font_montserrat_12, 0);
    lv_obj_set_pos(control_widgets_.modbus_health_status_label, 0, reg_y_start + 96);
    
    // === 系统版本信息区域 - 添加到控制面板底部 ===
    lv_obj_t* version_section = lv_obj_create(control_panel_);
    lv_obj_set_size(version_section, lv_pct(100), 120);
    lv_obj_set_style_bg_color(version_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(version_section, 12, 0);
    lv_obj_set_style_border_width(version_section, 1, 0);
    lv_obj_set_style_border_color(version_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(version_section, 10, 0);
    lv_obj_clear_flag(version_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* version_title = lv_label_create(version_section);
    lv_label_set_text(version_title, LV_SYMBOL_LIST " System Version");
    lv_obj_set_style_text_color(version_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(version_title, &lv_font_montserrat_12, 0);
    lv_obj_align(version_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // JetPack版本
    status_widgets_.jetpack_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.jetpack_version, "JetPack: 5.1.2");
    lv_obj_set_style_text_color(status_widgets_.jetpack_version, lv_color_white(), 0);
    lv_obj_set_style_text_font(status_widgets_.jetpack_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.jetpack_version, LV_ALIGN_TOP_LEFT, 0, 18);
    
    // CUDA版本
    status_widgets_.cuda_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.cuda_version, "CUDA: 11.4.315");
    lv_obj_set_style_text_color(status_widgets_.cuda_version, color_success_, 0);
    lv_obj_set_style_text_font(status_widgets_.cuda_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.cuda_version, LV_ALIGN_TOP_LEFT, 0, 35);
    
    // TensorRT版本
    status_widgets_.tensorrt_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.tensorrt_version, "TensorRT: 8.5.2");
    lv_obj_set_style_text_color(status_widgets_.tensorrt_version, color_primary_, 0);
    lv_obj_set_style_text_font(status_widgets_.tensorrt_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.tensorrt_version, LV_ALIGN_TOP_LEFT, 0, 52);
    
    // OpenCV版本
    status_widgets_.opencv_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.opencv_version, "OpenCV: 4.8.0");
    lv_obj_set_style_text_color(status_widgets_.opencv_version, color_warning_, 0);
    lv_obj_set_style_text_font(status_widgets_.opencv_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.opencv_version, LV_ALIGN_TOP_LEFT, 0, 69);
    
    // Ubuntu版本
    status_widgets_.ubuntu_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.ubuntu_version, "Ubuntu: 20.04.6");
    lv_obj_set_style_text_color(status_widgets_.ubuntu_version, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(status_widgets_.ubuntu_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.ubuntu_version, LV_ALIGN_TOP_LEFT, 0, 86);
    
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
    lv_obj_set_size(footer_panel_, lv_pct(96), 70);  // 适应新布局，减小高度
    lv_obj_align(footer_panel_, LV_ALIGN_BOTTOM_MID, 0, -10);
    
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
    
    // 停止Jetson系统监控
    if (jetson_monitor_) {
        jetson_monitor_->stop();
    }
    
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
    
    try {
        while (!should_stop_.load()) {
            auto current_time = std::chrono::high_resolution_clock::now();
            
            try {
                // 处理LVGL任务 - 添加异常保护
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
                
                // 控制刷新率 - 限制最小间隔
                uint32_t sleep_time = std::max(std::min(time_till_next, 16u), 1u);
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                
            } catch (const std::exception& e) {
                std::cerr << "[LVGLInterface] UI循环异常: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 错误恢复延迟
            } catch (...) {
                std::cerr << "[LVGLInterface] UI循环未知异常" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 错误恢复延迟
            }
        }
    } catch (...) {
        std::cerr << "[LVGLInterface] UI主循环致命异常" << std::endl;
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
    
    // 获取真实的Jetson系统状态
    if (jetson_monitor_ && jetson_monitor_->isRunning()) {
        utils::SystemStats stats = jetson_monitor_->getLatestStats();
        
        // 更新CPU信息
        if (control_widgets_.cpu_bar && !stats.cpu_cores.empty()) {
            // 计算平均CPU使用率
            int total_usage = 0;
            int total_freq = 0;
            for (const auto& core : stats.cpu_cores) {
                total_usage += core.usage_percent;
                total_freq += core.frequency_mhz;
            }
            int avg_usage = total_usage / stats.cpu_cores.size();
            int avg_freq = total_freq / stats.cpu_cores.size();
            
            lv_bar_set_value(control_widgets_.cpu_bar, avg_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.cpu_label, "CPU: %d%% @%dMHz", avg_usage, avg_freq);
            
            // 设置进度条主体背景（未使用部分）
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, lv_color_hex(0x2A3441), LV_PART_MAIN);
            lv_obj_set_style_bg_opa(control_widgets_.cpu_bar, LV_OPA_COVER, LV_PART_MAIN);
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
            lv_obj_set_style_bg_opa(control_widgets_.cpu_bar, LV_OPA_COVER, LV_PART_INDICATOR);
        }
        
        // 更新GPU信息
        if (control_widgets_.gpu_bar) {
            int gpu_usage = stats.gpu.usage_percent;
            lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.gpu_label, "GPU: %d%% @%dMHz",
                                 gpu_usage, stats.gpu.frequency_mhz);
            
            // 设置进度条主体背景（未使用部分）
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, lv_color_hex(0x2A3441), LV_PART_MAIN);
            lv_obj_set_style_bg_opa(control_widgets_.gpu_bar, LV_OPA_COVER, LV_PART_MAIN);
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
            lv_obj_set_style_bg_opa(control_widgets_.gpu_bar, LV_OPA_COVER, LV_PART_INDICATOR);
        }
        
        // 更新内存信息
        if (control_widgets_.mem_bar && stats.memory.ram_total_mb > 0) {
            int mem_percentage = (stats.memory.ram_used_mb * 100) / stats.memory.ram_total_mb;
            lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.mem_label, "RAM: %d%% %d/%dMB (LFB:%dx%dMB)",
                                 mem_percentage, stats.memory.ram_used_mb, stats.memory.ram_total_mb,
                                 stats.memory.lfb_blocks, stats.memory.lfb_size_mb);
            
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
        }
        
        // 从DataBridge获取真实的AI模型统计数据
        core::SystemStats databridge_stats = data_bridge_->getStats();
        
        // 更新AI模型版本
        if (control_widgets_.ai_model_version_label) {
            lv_label_set_text_fmt(control_widgets_.ai_model_version_label, "模型版本: %s",
                                  databridge_stats.ai_model.model_version.c_str());
        }
        
        // 更新推理时间
        if (control_widgets_.ai_inference_time_label) {
            lv_label_set_text_fmt(control_widgets_.ai_inference_time_label, "推理时间: %.1fms",
                                  databridge_stats.ai_model.inference_time_ms);
            
            // 根据推理时间设置颜色
            if (databridge_stats.ai_model.inference_time_ms > 30.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_error_, 0);
            } else if (databridge_stats.ai_model.inference_time_ms > 20.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
            }
        }
        
        // 更新置信阈值
        if (control_widgets_.ai_confidence_threshold_label) {
            lv_label_set_text_fmt(control_widgets_.ai_confidence_threshold_label, "置信阈值: %.2f",
                                  databridge_stats.ai_model.confidence_threshold);
        }
        
        // 更新检测精度
        if (control_widgets_.ai_detection_accuracy_label) {
            lv_label_set_text_fmt(control_widgets_.ai_detection_accuracy_label, "检测精度: %.1f%%",
                                  databridge_stats.ai_model.detection_accuracy);
            
            // 根据检测精度设置颜色
            if (databridge_stats.ai_model.detection_accuracy > 90.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
            } else if (databridge_stats.ai_model.detection_accuracy > 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_error_, 0);
            }
        }
        
        // 更新总检测数
        if (control_widgets_.ai_total_detections_label) {
            lv_label_set_text_fmt(control_widgets_.ai_total_detections_label, "总检测数: %d",
                                  databridge_stats.ai_model.total_detections);
        }
        
        // 更新今日检测数
        if (control_widgets_.ai_daily_detections_label) {
            lv_label_set_text_fmt(control_widgets_.ai_daily_detections_label, "今日检测: %d",
                                  databridge_stats.ai_model.daily_detections);
        }
        
        // === 更新当前竹子检测状态 ===
        const auto& bamboo_detection = databridge_stats.ai_model.current_bamboo;
        
        // 更新竹子直径
        if (control_widgets_.bamboo_diameter_label) {
            if (bamboo_detection.has_bamboo) {
                lv_label_set_text_fmt(control_widgets_.bamboo_diameter_label, "- 直径：%.1fmm",
                                     bamboo_detection.diameter_mm);
                
                // 根据直径设置颜色 (合理范围20-80mm)
                if (bamboo_detection.diameter_mm >= 20.0f && bamboo_detection.diameter_mm <= 80.0f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_warning_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.bamboo_diameter_label, "- 直径：无检测");
                lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新竹子长度
        if (control_widgets_.bamboo_length_label) {
            if (bamboo_detection.has_bamboo) {
                lv_label_set_text_fmt(control_widgets_.bamboo_length_label, "- 长度：%.0fmm",
                                     bamboo_detection.length_mm);
                
                // 根据长度设置颜色 (合理范围1000-5000mm)
                if (bamboo_detection.length_mm >= 1000.0f && bamboo_detection.length_mm <= 5000.0f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_warning_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.bamboo_length_label, "- 长度：无检测");
                lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新预切位置
        if (control_widgets_.bamboo_cut_positions_label) {
            if (bamboo_detection.has_bamboo && !bamboo_detection.cut_positions.empty()) {
                // 构建预切位置字符串
                std::string positions_str = "- 预切位置：[";
                for (size_t i = 0; i < bamboo_detection.cut_positions.size(); i++) {
                    if (i > 0) positions_str += ", ";
                    positions_str += std::to_string(static_cast<int>(bamboo_detection.cut_positions[i])) + "mm";
                }
                positions_str += "]";
                
                lv_label_set_text(control_widgets_.bamboo_cut_positions_label, positions_str.c_str());
                lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
            } else {
                lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "- 预切位置：无数据");
                lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新检测置信度
        if (control_widgets_.bamboo_confidence_label) {
            if (bamboo_detection.has_bamboo) {
                lv_label_set_text_fmt(control_widgets_.bamboo_confidence_label, "- 置信度：%.2f",
                                     bamboo_detection.confidence);
                
                // 根据置信度设置颜色
                if (bamboo_detection.confidence >= 0.9f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
                } else if (bamboo_detection.confidence >= 0.7f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_warning_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_error_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.bamboo_confidence_label, "- 置信度：N/A");
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新检测耗时
        if (control_widgets_.bamboo_detection_time_label) {
            if (bamboo_detection.has_bamboo) {
                lv_label_set_text_fmt(control_widgets_.bamboo_detection_time_label, "- 检测耗时：%.1fms",
                                     bamboo_detection.detection_time_ms);
                
                // 根据检测时间设置颜色
                if (bamboo_detection.detection_time_ms <= 20.0f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_success_, 0);
                } else if (bamboo_detection.detection_time_ms <= 35.0f) {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_warning_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_error_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.bamboo_detection_time_label, "- 检测耗时：N/A");
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // === 更新摄像头系统状态 ===
        const auto& camera_system = databridge_stats.ai_model.camera_system;
        
        // 更新摄像头-1状态
        if (control_widgets_.camera1_status_label) {
            if (camera_system.camera1.is_online) {
                lv_label_set_text(control_widgets_.camera1_status_label, "摄像头-1：在线 ✓");
                lv_obj_set_style_text_color(control_widgets_.camera1_status_label, color_success_, 0);
            } else {
                lv_label_set_text(control_widgets_.camera1_status_label, "摄像头-1：未安装 ✗");
                lv_obj_set_style_text_color(control_widgets_.camera1_status_label, color_error_, 0);
            }
        }
        
        if (control_widgets_.camera1_fps_label) {
            if (camera_system.camera1.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera1_fps_label, "  帧率：%.0f FPS", camera_system.camera1.fps);
                lv_obj_set_style_text_color(control_widgets_.camera1_fps_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(control_widgets_.camera1_fps_label, "  帧率：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera1_fps_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera1_resolution_label) {
            if (camera_system.camera1.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera1_resolution_label, "  分辨率：%dx%d",
                                     camera_system.camera1.width, camera_system.camera1.height);
                lv_obj_set_style_text_color(control_widgets_.camera1_resolution_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(control_widgets_.camera1_resolution_label, "  分辨率：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera1_resolution_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera1_exposure_label) {
            if (camera_system.camera1.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera1_exposure_label, "  曝光：%s",
                                     camera_system.camera1.exposure_mode.c_str());
                lv_obj_set_style_text_color(control_widgets_.camera1_exposure_label, color_primary_, 0);
            } else {
                lv_label_set_text(control_widgets_.camera1_exposure_label, "  曝光：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera1_exposure_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera1_lighting_label) {
            if (camera_system.camera1.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera1_lighting_label, "  光照评分：%s",
                                     camera_system.camera1.lighting_quality.c_str());
                
                // 根据光照质量设置颜色
                if (camera_system.camera1.lighting_quality == "良好") {
                    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_success_, 0);
                } else if (camera_system.camera1.lighting_quality == "一般") {
                    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_warning_, 0);
                } else if (camera_system.camera1.lighting_quality == "差") {
                    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_error_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_primary_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.camera1_lighting_label, "  光照评分：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新摄像头-2状态
        if (control_widgets_.camera2_status_label) {
            if (camera_system.camera2.is_online) {
                lv_label_set_text(control_widgets_.camera2_status_label, "摄像头-2：在线 ✓");
                lv_obj_set_style_text_color(control_widgets_.camera2_status_label, color_success_, 0);
            } else {
                lv_label_set_text(control_widgets_.camera2_status_label, "摄像头-2：未安装 ✗");
                lv_obj_set_style_text_color(control_widgets_.camera2_status_label, color_error_, 0);
            }
        }
        
        if (control_widgets_.camera2_fps_label) {
            if (camera_system.camera2.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera2_fps_label, "  帧率：%.0f FPS", camera_system.camera2.fps);
                lv_obj_set_style_text_color(control_widgets_.camera2_fps_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(control_widgets_.camera2_fps_label, "  帧率：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera2_fps_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera2_resolution_label) {
            if (camera_system.camera2.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera2_resolution_label, "  分辨率：%dx%d",
                                     camera_system.camera2.width, camera_system.camera2.height);
                lv_obj_set_style_text_color(control_widgets_.camera2_resolution_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(control_widgets_.camera2_resolution_label, "  分辨率：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera2_resolution_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera2_exposure_label) {
            if (camera_system.camera2.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera2_exposure_label, "  曝光：%s",
                                     camera_system.camera2.exposure_mode.c_str());
                lv_obj_set_style_text_color(control_widgets_.camera2_exposure_label, color_primary_, 0);
            } else {
                lv_label_set_text(control_widgets_.camera2_exposure_label, "  曝光：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera2_exposure_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        if (control_widgets_.camera2_lighting_label) {
            if (camera_system.camera2.is_online) {
                lv_label_set_text_fmt(control_widgets_.camera2_lighting_label, "  光照评分：%s",
                                     camera_system.camera2.lighting_quality.c_str());
                
                // 根据光照质量设置颜色
                if (camera_system.camera2.lighting_quality == "良好") {
                    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_success_, 0);
                } else if (camera_system.camera2.lighting_quality == "一般") {
                    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_warning_, 0);
                } else if (camera_system.camera2.lighting_quality == "差") {
                    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_error_, 0);
                } else {
                    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_primary_, 0);
                }
            } else {
                lv_label_set_text(control_widgets_.camera2_lighting_label, "  光照评分：N/A");
                lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
        // 更新温度信息
        if (control_widgets_.cpu_temp_label) {
            lv_label_set_text_fmt(control_widgets_.cpu_temp_label, "CPU: %.1f°C", stats.temperature.cpu_temp);
            
            // 根据温度设置颜色
            if (stats.temperature.cpu_temp > 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_error_, 0);
            } else if (stats.temperature.cpu_temp > 70.0f) {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_success_, 0);
            }
        }
        
        if (control_widgets_.gpu_temp_label) {
            lv_label_set_text_fmt(control_widgets_.gpu_temp_label, "GPU: %.1f°C", stats.temperature.gpu_temp);
            
            // 根据温度设置颜色
            if (stats.temperature.gpu_temp > 75.0f) {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_error_, 0);
            } else if (stats.temperature.gpu_temp > 65.0f) {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_success_, 0);
            }
        }
        
        if (control_widgets_.soc_temp_label) {
            lv_label_set_text_fmt(control_widgets_.soc_temp_label, "SOC: %.1f°C", stats.temperature.soc_temp);
        }
        
        if (control_widgets_.thermal_temp_label) {
            lv_label_set_text_fmt(control_widgets_.thermal_temp_label, "Thermal: %.1f°C", stats.temperature.thermal_temp);
        }
        
        // 更新电源信息
        if (control_widgets_.power_in_label) {
            lv_label_set_text_fmt(control_widgets_.power_in_label, "VDD_IN: %dmA/%dmW",
                                 stats.power.vdd_in_current_ma, stats.power.vdd_in_power_mw);
        }
        
        if (control_widgets_.power_cpu_gpu_label) {
            lv_label_set_text_fmt(control_widgets_.power_cpu_gpu_label, "CPU_GPU: %dmA/%dmW",
                                 stats.power.vdd_cpu_gpu_cv_current_ma, stats.power.vdd_cpu_gpu_cv_power_mw);
        }
        
        if (control_widgets_.power_soc_label) {
            lv_label_set_text_fmt(control_widgets_.power_soc_label, "SOC: %dmA/%dmW",
                                 stats.power.vdd_soc_current_ma, stats.power.vdd_soc_power_mw);
        }
        
        // 更新SWAP使用情况
        if (control_widgets_.swap_usage_label && stats.memory.swap_total_mb > 0) {
            lv_label_set_text_fmt(control_widgets_.swap_usage_label, "SWAP: %d/%dMB",
                                 stats.memory.swap_used_mb, stats.memory.swap_total_mb);
            
            // 根据SWAP使用率设置颜色
            int swap_percentage = (stats.memory.swap_used_mb * 100) / stats.memory.swap_total_mb;
            if (swap_percentage > 80) {
                lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_error_, 0);
            } else if (swap_percentage > 50) {
                lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_success_, 0);
            }
        }
        
        // 更新EMC频率信息
        if (control_widgets_.emc_freq_label) {
            lv_label_set_text_fmt(control_widgets_.emc_freq_label, "EMC: %d%%@%dMHz",
                                 stats.other.emc_freq_percent, stats.other.emc_freq_mhz);
        }
        
        // 更新VIC使用率
        if (control_widgets_.vic_usage_label) {
            lv_label_set_text_fmt(control_widgets_.vic_usage_label, "VIC: %d%%@%dMHz",
                                 stats.other.vic_usage_percent, stats.other.vic_freq_mhz);
        }
        
        // 更新风扇转速
        if (control_widgets_.fan_speed_label) {
            if (stats.other.fan_rpm > 0) {
                lv_label_set_text_fmt(control_widgets_.fan_speed_label, "FAN: %dRPM", stats.other.fan_rpm);
                lv_obj_set_style_text_color(control_widgets_.fan_speed_label, color_primary_, 0);
            } else {
                lv_label_set_text(control_widgets_.fan_speed_label, "FAN: N/A");
                lv_obj_set_style_text_color(control_widgets_.fan_speed_label, lv_color_hex(0x8A92A1), 0);
            }
        }
        
    } else {
        // 如果Jetson监控不可用，回退到模拟数据
        if (control_widgets_.cpu_bar) {
            static int cpu_usage = 45;
            cpu_usage = 40 + (rand() % 30);  // 模拟40-70%的CPU使用率
            lv_bar_set_value(control_widgets_.cpu_bar, cpu_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.cpu_label, "CPU: %d%% @1.9GHz (模拟)", cpu_usage);
        }
        
        if (control_widgets_.gpu_bar) {
            static int gpu_usage = 72;
            gpu_usage = 60 + (rand() % 25);  // 模拟60-85%的GPU使用率
            lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.gpu_label, "GPU: %d%% @624MHz (模拟)", gpu_usage);
        }
        
        if (control_widgets_.mem_bar) {
            static float mem_used = 4.6f;
            static float mem_total = 8.0f;
            mem_used = 3.8f + ((rand() % 200) / 100.0f);  // 模拟3.8-5.8GB内存使用
            int mem_percentage = (int)((mem_used / mem_total) * 100);
            lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.mem_label, "RAM: %d%% %.1f/%.0fGB (模拟)", mem_percentage, mem_used, mem_total);
        }
    }
    
    // 注意：旧的ai_fps_label等组件已被新的AI模型监控组件替代，在上面已经更新
    
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
    if (!data_bridge_) return;
    
    // 从DataBridge获取真实的Modbus寄存器数据
    core::ModbusRegisters modbus_registers = data_bridge_->getModbusRegisters();
    
    // 更新连接状态信息（使用模拟数据）
    if (control_widgets_.modbus_connection_label) {
        static int hours = 2, minutes = 15, seconds = 32;
        seconds++;
        if (seconds >= 60) { seconds = 0; minutes++; }
        if (minutes >= 60) { minutes = 0; hours++; }
        lv_label_set_text_fmt(control_widgets_.modbus_connection_label,
            "连接: %02d:%02d:%02d", hours, minutes, seconds);
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
        
        // 根据错误率设置颜色
        if (error_rate > 1.0f) {
            lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_error_, 0);
        } else if (error_rate > 0.1f) {
            lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_success_, 0);
        }
    }
    
    if (control_widgets_.modbus_heartbeat_label) {
        bool heartbeat_ok = (modbus_registers.heartbeat > 0);
        lv_label_set_text(control_widgets_.modbus_heartbeat_label,
            heartbeat_ok ? "心跳: OK" : "心跳: 超时");
        lv_obj_set_style_text_color(control_widgets_.modbus_heartbeat_label,
            heartbeat_ok ? color_success_ : color_error_, 0);
    }
    
    // === 更新Modbus寄存器状态信息 ===
    
    // 40001: 系统状态
    if (control_widgets_.modbus_system_status_label) {
        const char* status_names[] = {"停止", "运行", "错误", "暂停", "紧急停止"};
        uint16_t system_status = modbus_registers.system_status;
        const char* status_str = (system_status < 5) ? status_names[system_status] : "未知";
        
        lv_label_set_text_fmt(control_widgets_.modbus_system_status_label,
            "40001 系统状态: %s", status_str);
        
        // 根据状态设置颜色
        if (system_status == 1) { // 运行
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_success_, 0);
        } else if (system_status == 2 || system_status == 4) { // 错误或紧急停止
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_error_, 0);
        } else if (system_status == 3) { // 暂停
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_warning_, 0);
        } else { // 停止或未知
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
    
    // 40002: PLC命令
    if (control_widgets_.modbus_plc_command_label) {
        const char* command_names[] = {"无", "进料检测", "切割准备", "切割完成", "启动送料",
                                       "停止送料", "复位系统", "保持", "刀片选择"};
        uint16_t plc_command = modbus_registers.plc_command;
        const char* command_str = (plc_command < 9) ? command_names[plc_command] : "未知命令";
        
        lv_label_set_text_fmt(control_widgets_.modbus_plc_command_label,
            "40002 PLC命令: %s", command_str);
        
        // 根据命令类型设置颜色
        if (plc_command >= 1 && plc_command <= 5) { // 正常操作命令
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, color_primary_, 0);
        } else if (plc_command == 6) { // 复位系统
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, color_warning_, 0);
        } else { // 无命令或其他
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
    
    // 40003: 坐标就绪标志
    if (control_widgets_.modbus_coord_ready_label) {
        uint16_t coord_ready = modbus_registers.coord_ready;
        const char* ready_str = coord_ready ? "是" : "否";
        
        lv_label_set_text_fmt(control_widgets_.modbus_coord_ready_label,
            "40003 坐标就绪: %s", ready_str);
        
        lv_obj_set_style_text_color(control_widgets_.modbus_coord_ready_label,
            coord_ready ? color_success_ : color_warning_, 0);
    }
    
    // 40004-40005: X坐标 (32位，0.1mm精度)
    if (control_widgets_.modbus_x_coordinate_label) {
        uint32_t x_coord_raw = modbus_registers.x_coordinate;
        float x_coord_mm = static_cast<float>(x_coord_raw) * 0.1f; // 0.1mm精度
        
        lv_label_set_text_fmt(control_widgets_.modbus_x_coordinate_label,
            "40004 X坐标: %.1fmm", x_coord_mm);
        
        // 根据坐标值设置颜色（假设有效范围是0-1000mm）
        if (x_coord_mm >= 0.0f && x_coord_mm <= 1000.0f) {
            lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, color_primary_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, color_warning_, 0);
        }
    }
    
    // 40006: 切割质量
    if (control_widgets_.modbus_cut_quality_label) {
        const char* quality_names[] = {"正常", "异常"};
        uint16_t cut_quality = modbus_registers.cut_quality;
        const char* quality_str = (cut_quality < 2) ? quality_names[cut_quality] : "未知";
        
        lv_label_set_text_fmt(control_widgets_.modbus_cut_quality_label,
            "40006 切割质量: %s", quality_str);
        
        lv_obj_set_style_text_color(control_widgets_.modbus_cut_quality_label,
            (cut_quality == 0) ? color_success_ : color_error_, 0);
    }
    
    // 40009: 刀片编号
    if (control_widgets_.modbus_blade_number_label) {
        const char* blade_names[] = {"无", "刀片1", "刀片2", "双刀片"};
        uint16_t blade_number = modbus_registers.blade_number;
        const char* blade_str = (blade_number < 4) ? blade_names[blade_number] : "未知刀片";
        
        lv_label_set_text_fmt(control_widgets_.modbus_blade_number_label,
            "40009 刀片编号: %s", blade_str);
        
        // 根据刀片状态设置颜色
        if (blade_number >= 1 && blade_number <= 3) { // 有刀片
            lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, color_success_, 0);
        } else { // 无刀片或未知
            lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, color_warning_, 0);
        }
    }
    
    // 40010: 健康状态
    if (control_widgets_.modbus_health_status_label) {
        const char* health_names[] = {"正常", "警告", "错误", "严重"};
        uint16_t health_status = modbus_registers.health_status;
        const char* health_str = (health_status < 4) ? health_names[health_status] : "未知";
        
        lv_label_set_text_fmt(control_widgets_.modbus_health_status_label,
            "40010 健康状态: %s", health_str);
        
        // 根据健康状态设置颜色
        if (health_status == 0) { // 正常
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_success_, 0);
        } else if (health_status == 1) { // 警告
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_warning_, 0);
        } else if (health_status >= 2) { // 错误或严重
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_error_, 0);
        } else { // 未知
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
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
    // 安全的DRM显示刷新实现 - 修复段错误问题
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
    
    // 严格的参数验证
    if (!disp || !area || !px_map) {
        std::cerr << "[DRM] Invalid parameters in display_flush_cb" << std::endl;
        if (disp) lv_display_flush_ready(disp);
        return;
    }
    
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
    
    // 安全的像素数据复制 - 修复段错误和内存访问问题
    if (drm_initialized && framebuffer && drm_width > 0 && drm_height > 0 && stride > 0) {
        // 严格的区域边界检查
        if (area->x1 < 0 || area->y1 < 0 ||
            area->x2 >= (int32_t)drm_width || area->y2 >= (int32_t)drm_height ||
            area->x1 > area->x2 || area->y1 > area->y2) {
            std::cerr << "[DRM] Invalid area bounds: (" << area->x1 << "," << area->y1
                      << ") to (" << area->x2 << "," << area->y2 << ")" << std::endl;
            lv_display_flush_ready(disp);
            return;
        }
        
        uint32_t area_width = area->x2 - area->x1 + 1;
        uint32_t area_height = area->y2 - area->y1 + 1;
        uint32_t pixels_per_row = stride / 4; // stride是字节数，除以4得到uint32_t数量
        
        // 验证缓冲区大小
        uint32_t total_area_pixels = area_width * area_height;
        if (total_area_pixels == 0) {
            lv_display_flush_ready(disp);
            return;
        }
        
        try {
            // 安全的按行复制像素数据
            for (uint32_t y = 0; y < area_height; y++) {
                uint32_t dst_row = area->y1 + y;
                uint32_t dst_row_offset = dst_row * pixels_per_row;
                
                // 检查目标行是否在有效范围内
                if (dst_row >= drm_height || dst_row_offset >= (buffer_size / 4)) {
                    continue;
                }
                
                for (uint32_t x = 0; x < area_width; x++) {
                    uint32_t dst_col = area->x1 + x;
                    uint32_t dst_idx = dst_row_offset + dst_col;
                    uint32_t src_idx = y * area_width + x;
                    
                    // 检查目标和源索引的有效性
                    if (dst_col >= drm_width || dst_idx >= (buffer_size / 4) ||
                        src_idx >= total_area_pixels) {
                        continue;
                    }
                    
                    // 简化的像素格式转换 - 使用32位ARGB8888
                    uint32_t pixel = 0x00000000; // 默认黑色
                    
                    #if LV_COLOR_DEPTH == 32
                        // 32位ARGB8888格式 - 直接复制
                        uint32_t* src_pixels = (uint32_t*)px_map;
                        if (src_idx < total_area_pixels) {
                            pixel = src_pixels[src_idx];
                        }
                    #elif LV_COLOR_DEPTH == 16
                        // 16位RGB565格式转换
                        uint16_t* src_pixels = (uint16_t*)px_map;
                        if (src_idx < total_area_pixels) {
                            uint16_t src_value = src_pixels[src_idx];
                            uint8_t r = ((src_value >> 11) & 0x1F) * 255 / 31;
                            uint8_t g = ((src_value >> 5) & 0x3F) * 255 / 63;
                            uint8_t b = (src_value & 0x1F) * 255 / 31;
                            pixel = (r << 16) | (g << 8) | b;
                        }
                    #else
                        // 24位RGB888格式
                        uint8_t* src_pixels = (uint8_t*)px_map;
                        uint32_t byte_idx = src_idx * 3;
                        if (byte_idx + 2 < total_area_pixels * 3) {
                            uint8_t r = src_pixels[byte_idx + 2]; // BGR -> RGB
                            uint8_t g = src_pixels[byte_idx + 1];
                            uint8_t b = src_pixels[byte_idx + 0];
                            pixel = (r << 16) | (g << 8) | b;
                        }
                    #endif
                    
                    framebuffer[dst_idx] = pixel;
                }
            }
        } catch (...) {
            std::cerr << "[DRM] Exception during pixel copy" << std::endl;
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
