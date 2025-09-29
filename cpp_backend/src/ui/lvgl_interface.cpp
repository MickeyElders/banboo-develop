/**
 * @file lvgl_interface.cpp
 * @brief C++ LVGL一体化系统界面管理器主实现
 * 工业级竹子识别系统LVGL界面 - 现代科技风格版本
 * 适配DRM渲染后端
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <thread>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
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

// 全局样式变量定义
#ifdef ENABLE_LVGL
lv_style_t style_card;
lv_style_t style_btn_primary;
lv_style_t style_btn_success;
lv_style_t style_btn_warning;
lv_style_t style_btn_danger;
lv_style_t style_btn_pressed;
lv_style_t style_text_title;
lv_style_t style_text_body;
lv_style_t style_text_small;
lv_style_t style_table_header;
lv_style_t style_table_cell;
#else
// LVGL未启用时的占位符
char style_card[64];
char style_btn_primary[64];
char style_btn_success[64];
char style_btn_warning[64];
char style_btn_danger[64];
char style_btn_pressed[64];
char style_text_title[64];
char style_text_body[64];
char style_text_small[64];
char style_table_header[64];
char style_table_cell[64];
#endif

// 类的静态成员变量定义
#ifdef ENABLE_LVGL
lv_style_t LVGLInterface::style_card;
lv_style_t LVGLInterface::style_text_title;
#else
char LVGLInterface::style_card[64];
char LVGLInterface::style_text_title[64];
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
    // 修复内存释放：使用与分配匹配的delete方式
    if (disp_buf1_) {
        delete[] reinterpret_cast<uint32_t*>(disp_buf1_);
        disp_buf1_ = nullptr;
    }
    if (disp_buf2_) {
        delete[] reinterpret_cast<uint32_t*>(disp_buf2_);
        disp_buf2_ = nullptr;
    }
    std::cout << "[LVGLInterface] 显示缓冲区内存已释放" << std::endl;
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
    
    // === 完全使用Flex布局 - 修复布局冲突 ===
    
    // 1. 设置主屏幕背景色和Flex布局
    lv_obj_set_style_bg_color(main_screen_, color_background_, 0);
    lv_obj_set_style_bg_opa(main_screen_, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(main_screen_, 0, 0);
    lv_obj_clear_flag(main_screen_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置主屏幕为垂直Flex布局
    lv_obj_set_flex_flow(main_screen_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(main_screen_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(main_screen_, 5, 0);
    std::cout << "[LVGLInterface] 设置主屏幕Flex布局完成" << std::endl;
    
    // 2. 添加可见性测试标签
    lv_obj_t* test_label = lv_label_create(main_screen_);
    lv_label_set_text(test_label, "BAMBOO SYSTEM - Interface Test");
    lv_obj_set_style_text_color(test_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(test_label, &lv_font_montserrat_24, 0);
    lv_obj_set_style_bg_color(test_label, lv_color_hex(0xFF0000), 0);  // 红色背景便于识别
    lv_obj_set_style_bg_opa(test_label, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(test_label, 10, 0);
    std::cout << "[LVGLInterface] 创建测试标签完成" << std::endl;
    
    // 3. 创建头部面板
    std::cout << "[LVGLInterface] 开始创建头部面板..." << std::endl;
    header_panel_ = createHeaderPanel();
    if (header_panel_) {
        lv_obj_set_flex_grow(header_panel_, 0);  // 头部面板不允许增长
        std::cout << "[LVGLInterface] 头部面板创建成功" << std::endl;
    } else {
        std::cerr << "[LVGLInterface] 头部面板创建失败" << std::endl;
    }
    
    // 4. 创建中间内容容器（使用Flex布局管理左右面板）
    std::cout << "[LVGLInterface] 开始创建中间内容容器..." << std::endl;
    lv_obj_t* content_container = lv_obj_create(main_screen_);
    lv_obj_set_width(content_container, lv_pct(100));
    lv_obj_set_flex_grow(content_container, 1);  // 中间容器占据剩余空间
    lv_obj_set_style_bg_opa(content_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(content_container, 0, 0);
    lv_obj_set_style_pad_all(content_container, 5, 0);
    lv_obj_clear_flag(content_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置中间容器为水平Flex布局
    lv_obj_set_flex_flow(content_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(content_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(content_container, 10, 0);
    std::cout << "[LVGLInterface] 中间内容容器创建完成" << std::endl;
    
    // 5. 在容器内创建左右面板
    std::cout << "[LVGLInterface] 开始创建摄像头面板..." << std::endl;
    camera_panel_ = createCameraPanel(content_container);
    if (camera_panel_) {
        std::cout << "[LVGLInterface] 摄像头面板创建成功" << std::endl;
    } else {
        std::cerr << "[LVGLInterface] 摄像头面板创建失败" << std::endl;
    }
    
    std::cout << "[LVGLInterface] 开始创建控制面板..." << std::endl;
    control_panel_ = createControlPanel(content_container);
    if (control_panel_) {
        std::cout << "[LVGLInterface] 控制面板创建成功" << std::endl;
    } else {
        std::cerr << "[LVGLInterface] 控制面板创建失败" << std::endl;
    }
    
    // 6. 创建底部面板
    std::cout << "[LVGLInterface] 开始创建底部面板..." << std::endl;
    footer_panel_ = createFooterPanel();
    if (footer_panel_) {
        lv_obj_set_flex_grow(footer_panel_, 0);  // 底部面板不允许增长
        std::cout << "[LVGLInterface] 底部面板创建成功" << std::endl;
    } else {
        std::cerr << "[LVGLInterface] 底部面板创建失败" << std::endl;
    }
    
    // 7. 强制刷新界面
    lv_obj_invalidate(main_screen_);
    lv_refr_now(NULL);
    std::cout << "[LVGLInterface] 执行强制界面刷新" << std::endl;
    
    std::cout << "[LVGLInterface] 主界面创建完成，使用完全Flex布局" << std::endl;
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
    
    // 添加初始化延迟，确保所有组件就绪
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "[LVGLInterface] UI主循环初始化延迟完成" << std::endl;
    
    try {
        while (!should_stop_.load()) {
            auto current_time = std::chrono::high_resolution_clock::now();
            
            try {
                // 处理LVGL任务 - 添加异常保护和空指针检查
                if (display_ && main_screen_) {
                    uint32_t time_till_next = lv_timer_handler();
                    
                    // 延迟首次数据更新，确保所有控件初始化完成
                    static int update_counter = 0;
                    static int startup_delay = 0;
                    
                    if (++update_counter >= 3) {  // 每3次循环更新一次界面数据
                        update_counter = 0;
                        
                        // 延迟至少100次循环后才开始数据更新，确保界面初始化完成
                        if (++startup_delay > 100) {
                            updateInterface();
                        } else {
                            std::cout << "[LVGLInterface] 延迟数据更新，等待界面初始化完成 (" << startup_delay << "/100)" << std::endl;
                        }
                    }
                    
                    // 计算FPS
                    frame_count_++;
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        current_time - last_frame_time).count();
                    
                    if (elapsed >= 1000) {
                        ui_fps_ = frame_count_ * 1000.0f / elapsed;
                        frame_count_ = 0;
                        last_frame_time = current_time;
                    }
                    
                    // 控制刷新率 - 增加最小延迟以降低CPU占用
                    uint32_t sleep_time = std::max(std::min(time_till_next, 33u), 16u);  // 限制在30-60FPS
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                } else {
                    std::cerr << "[LVGLInterface] 显示器或主屏幕未初始化，跳过本次循环" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[LVGLInterface] UI循环异常: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 增加错误恢复延迟
            } catch (...) {
                std::cerr << "[LVGLInterface] UI循环未知异常" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 增加错误恢复延迟
            }
        }
    } catch (...) {
        std::cerr << "[LVGLInterface] UI主循环致命异常" << std::endl;
    }
    
    std::cout << "[LVGLInterface] UI主循环结束" << std::endl;
#endif
}


} // namespace ui
} // namespace bamboo_cut