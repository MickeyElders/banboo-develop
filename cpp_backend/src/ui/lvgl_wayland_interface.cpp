/**
 * @file lvgl_wayland_interface.cpp
 * @brief LVGL Wayland接口实现 - 替代DRM直接访问的现代方案
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/utils/jetson_monitor.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>

// LVGL和驱动头文件
#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
// 如果lv_drivers可用，包含Wayland驱动
#if __has_include("wayland/wayland.h")
#include "wayland/wayland.h"
#define HAS_WAYLAND_DRIVER 1
#else
#define HAS_WAYLAND_DRIVER 0
#warning "lv_drivers Wayland driver not found, using fallback implementation"
#endif
#else
#define HAS_WAYLAND_DRIVER 0
#endif

namespace bamboo_cut {
namespace ui {

LVGLWaylandInterface::LVGLWaylandInterface(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(data_bridge)
    , jetson_monitor_(std::make_shared<utils::JetsonMonitor>())
    , display_(nullptr)
    , touch_indev_(nullptr)
    , pointer_indev_(nullptr)
    , keyboard_indev_(nullptr)
    , main_screen_(nullptr)
    , header_panel_(nullptr)
    , camera_panel_(nullptr)
    , camera_canvas_(nullptr)
    , control_panel_(nullptr)
    , footer_panel_(nullptr)
    , current_step_(WorkflowStep::FEED_DETECTION)
    , selected_blade_(1)
{
    std::cout << "[LVGLWaylandInterface] 构造函数调用" << std::endl;
    
    // 初始化所有控件指针为nullptr
    std::memset(&header_widgets_, 0, sizeof(header_widgets_));
    std::memset(&camera_widgets_, 0, sizeof(camera_widgets_));
    std::memset(&control_widgets_, 0, sizeof(control_widgets_));
    std::memset(&footer_widgets_, 0, sizeof(footer_widgets_));
    
    std::cout << "[LVGLWaylandInterface] 控件结构体初始化完成" << std::endl;
}

LVGLWaylandInterface::~LVGLWaylandInterface() {
    std::cout << "[LVGLWaylandInterface] 析构函数调用" << std::endl;
    stop();
}

bool LVGLWaylandInterface::initialize(const LVGLWaylandConfig& config) {
    std::cout << "[LVGLWaylandInterface] 初始化Wayland界面系统..." << std::endl;
    config_ = config;
    
    // 检查Wayland环境
    if (!checkWaylandEnvironment()) {
        std::cerr << "[LVGLWaylandInterface] Wayland环境检查失败" << std::endl;
        return false;
    }
    
#ifdef ENABLE_LVGL
    // 初始化LVGL
    lv_init();
    std::cout << "[LVGLWaylandInterface] LVGL核心初始化完成" << std::endl;
    
    // 初始化Wayland显示
    if (!initializeWaylandDisplay()) {
        std::cerr << "[LVGLWaylandInterface] Wayland显示初始化失败" << std::endl;
        return false;
    }
    
    // 初始化输入设备
    if (config_.enable_touch || config_.enable_mouse || config_.enable_keyboard) {
        if (!initializeInput()) {
            std::cerr << "[LVGLWaylandInterface] 输入设备初始化失败" << std::endl;
            // 输入设备失败不是致命错误，继续执行
        }
    }
    
    // 初始化主题
    initializeTheme();
    
    // 创建主界面
    createMainInterface();
    
    // 启动Jetson系统监控
    if (jetson_monitor_) {
        jetson_monitor_->start();
    }
    
    std::cout << "[LVGLWaylandInterface] Wayland界面系统初始化成功" << std::endl;
    return true;
#else
    std::cerr << "[LVGLWaylandInterface] LVGL未启用，无法初始化" << std::endl;
    return false;
#endif
}

bool LVGLWaylandInterface::start() {
#ifdef ENABLE_LVGL
    if (running_.load()) {
        std::cerr << "[LVGLWaylandInterface] 界面线程已在运行" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLWaylandInterface] 启动UI线程..." << std::endl;
    should_stop_.store(false);
    running_.store(true);
    
    ui_thread_ = std::thread(&LVGLWaylandInterface::uiLoop, this);
    
    // 等待初始化完成
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "[LVGLWaylandInterface] UI线程启动成功" << std::endl;
    return true;
#else
    return false;
#endif
}

void LVGLWaylandInterface::stop() {
#ifdef ENABLE_LVGL
    if (!running_.load()) {
        return;
    }
    
    std::cout << "[LVGLWaylandInterface] 停止UI线程..." << std::endl;
    should_stop_.store(true);
    
    // 停止Jetson系统监控
    if (jetson_monitor_) {
        jetson_monitor_->stop();
    }
    
    if (ui_thread_.joinable()) {
        ui_thread_.join();
    }
    
    running_.store(false);
    fully_initialized_.store(false);
    std::cout << "[LVGLWaylandInterface] UI线程已停止" << std::endl;
#endif
}

bool LVGLWaylandInterface::isFullyInitialized() const {
#ifdef ENABLE_LVGL
    // 检查基础组件是否已初始化
    if (!running_.load() || !display_ || !main_screen_) {
        return false;
    }
    
    // 检查主要面板是否已创建
    if (!header_panel_ || !camera_panel_ || !control_panel_ || !footer_panel_) {
        return false;
    }
    
    // 检查关键控件是否已创建
    if (!camera_canvas_) {
        return false;
    }
    
    // 所有检查都通过，认为已完全初始化
    return fully_initialized_.load();
#else
    return false;
#endif
}

bool LVGLWaylandInterface::isRunning() const {
    return running_.load();
}

lv_obj_t* LVGLWaylandInterface::getCameraCanvas() {
    std::lock_guard<std::mutex> lock(ui_mutex_);
    return camera_canvas_;
}

void LVGLWaylandInterface::updateCameraCanvas(const cv::Mat& frame) {
    if (frame.empty() || !camera_canvas_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(canvas_mutex_);
    
    try {
        // 将帧数据保存供UI线程处理
        latest_frame_ = frame.clone();
        new_frame_available_.store(true);
        
    } catch (const std::exception& e) {
        std::cerr << "[LVGLWaylandInterface] Canvas更新异常: " << e.what() << std::endl;
    }
}

void LVGLWaylandInterface::setCurrentStep(WorkflowStep step) {
    current_step_ = step;
    // 触发界面更新
    // 这里可以添加步骤变化的UI反馈
}

bool LVGLWaylandInterface::checkWaylandEnvironment() {
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        // 尝试设置默认值
        setenv("WAYLAND_DISPLAY", config_.wayland_display.c_str(), 0);
        wayland_display = getenv("WAYLAND_DISPLAY");
        std::cout << "[LVGLWaylandInterface] 设置WAYLAND_DISPLAY=" << wayland_display << std::endl;
    } else {
        std::cout << "[LVGLWaylandInterface] 检测到WAYLAND_DISPLAY=" << wayland_display << std::endl;
    }
    
    // 检查XDG_RUNTIME_DIR
    const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
    if (!runtime_dir) {
        std::cerr << "[LVGLWaylandInterface] 警告：XDG_RUNTIME_DIR未设置" << std::endl;
        // 尝试设置默认值
        setenv("XDG_RUNTIME_DIR", "/run/user/1000", 0);
        runtime_dir = getenv("XDG_RUNTIME_DIR");
        std::cout << "[LVGLWaylandInterface] 设置XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
    } else {
        std::cout << "[LVGLWaylandInterface] 检测到XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
    }
    
#if HAS_WAYLAND_DRIVER
    std::cout << "[LVGLWaylandInterface] Wayland驱动可用" << std::endl;
    return true;
#else
    std::cout << "[LVGLWaylandInterface] 警告：Wayland驱动不可用，使用兼容模式" << std::endl;
    return true; // 允许在没有驱动的情况下继续，使用兼容模式
#endif
}

bool LVGLWaylandInterface::initializeWaylandDisplay() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLWaylandInterface] 初始化Wayland显示..." << std::endl;
    
#if HAS_WAYLAND_DRIVER
    // 使用lv_drivers Wayland驱动
    display_ = lv_wayland_create_window(config_.screen_width, config_.screen_height, 
                                       "Bamboo Recognition System", nullptr);
    
    if (!display_) {
        std::cerr << "[LVGLWaylandInterface] Wayland窗口创建失败" << std::endl;
        return false;
    }
    
    // 设置显示器为默认
    lv_display_set_default(display_);
    
    std::cout << "[LVGLWaylandInterface] Wayland显示创建成功 (" 
              << config_.screen_width << "x" << config_.screen_height << ")" << std::endl;
    
#else
    // 降级实现：创建一个虚拟显示
    std::cout << "[LVGLWaylandInterface] 使用虚拟显示（Wayland驱动不可用）" << std::endl;
    
    // 创建显示缓冲区
    size_t buf_size = config_.screen_width * config_.screen_height * sizeof(lv_color_t);
    void* buf1 = malloc(buf_size);
    void* buf2 = malloc(buf_size);
    
    if (!buf1 || !buf2) {
        std::cerr << "[LVGLWaylandInterface] 显示缓冲区内存分配失败" << std::endl;
        return false;
    }
    
    // 创建显示对象
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        std::cerr << "[LVGLWaylandInterface] 显示对象创建失败" << std::endl;
        free(buf1);
        free(buf2);
        return false;
    }
    
    // 设置显示缓冲区
    lv_display_set_buffers(display_, buf1, buf2, buf_size, LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置虚拟刷新回调
    lv_display_set_flush_cb(display_, waylandFlushCallback);
    
    std::cout << "[LVGLWaylandInterface] 虚拟显示创建成功" << std::endl;
#endif
    
    return true;
#else
    return false;
#endif
}

bool LVGLWaylandInterface::initializeInput() {
#ifdef ENABLE_LVGL && HAS_WAYLAND_DRIVER
    std::cout << "[LVGLWaylandInterface] 初始化Wayland输入设备..." << std::endl;
    
    // 获取Wayland输入设备
    if (config_.enable_touch || config_.enable_mouse) {
        pointer_indev_ = lv_wayland_get_pointer(display_);
        if (pointer_indev_) {
            std::cout << "[LVGLWaylandInterface] 指针输入设备初始化成功" << std::endl;
        }
    }
    
    if (config_.enable_keyboard) {
        keyboard_indev_ = lv_wayland_get_keyboard(display_);
        if (keyboard_indev_) {
            std::cout << "[LVGLWaylandInterface] 键盘输入设备初始化成功" << std::endl;
        }
    }
    
    return true;
#else
    std::cout << "[LVGLWaylandInterface] 跳过输入设备初始化（Wayland驱动不可用）" << std::endl;
    return true;
#endif
}

void LVGLWaylandInterface::initializeTheme() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLWaylandInterface] 初始化现代舒适主题..." << std::endl;
    
    // 优化后的配色方案 - 更柔和，对比度适中
    color_background_ = lv_color_hex(0x1A1F26);   // 温和深色背景
    color_surface_    = lv_color_hex(0x252B35);   // 卡片表面
    color_primary_    = lv_color_hex(0x5B9BD5);   // 柔和蓝色主色
    color_secondary_  = lv_color_hex(0x70A5DB);   // 淡蓝色副色
    color_success_    = lv_color_hex(0x7FB069);   // 柔和绿色
    color_warning_    = lv_color_hex(0xE6A055);   // 温和橙色
    color_error_      = lv_color_hex(0xD67B7B);   // 柔和红色
    
    std::cout << "[LVGLWaylandInterface] 主题初始化完成" << std::endl;
#endif
}

void LVGLWaylandInterface::createMainInterface() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLWaylandInterface] 创建主界面..." << std::endl;
    
    main_screen_ = lv_scr_act();
    
    // 设置主屏幕背景色和布局
    lv_obj_set_style_bg_color(main_screen_, color_background_, 0);
    lv_obj_set_style_bg_opa(main_screen_, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(main_screen_, 0, 0);
    lv_obj_clear_flag(main_screen_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置主屏幕为垂直Flex布局
    lv_obj_set_flex_flow(main_screen_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(main_screen_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(main_screen_, 5, 0);
    
    // 添加可见性测试标签
    lv_obj_t* test_label = lv_label_create(main_screen_);
    lv_label_set_text(test_label, "BAMBOO WAYLAND SYSTEM");
    lv_obj_set_style_text_color(test_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(test_label, &lv_font_montserrat_24, 0);
    lv_obj_set_style_bg_color(test_label, lv_color_hex(0x0000CC), 0);
    lv_obj_set_style_bg_opa(test_label, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(test_label, 10, 0);
    
    // 创建头部面板
    header_panel_ = createHeaderPanel();
    if (header_panel_) {
        lv_obj_set_flex_grow(header_panel_, 0);
        std::cout << "[LVGLWaylandInterface] 头部面板创建成功" << std::endl;
    }
    
    // 创建中间内容容器
    lv_obj_t* content_container = lv_obj_create(main_screen_);
    lv_obj_set_width(content_container, lv_pct(100));
    lv_obj_set_flex_grow(content_container, 1);
    lv_obj_set_style_bg_opa(content_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(content_container, 0, 0);
    lv_obj_set_style_pad_all(content_container, 5, 0);
    lv_obj_clear_flag(content_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置为水平布局
    lv_obj_set_flex_flow(content_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(content_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(content_container, 10, 0);
    
    // 创建摄像头面板
    camera_panel_ = createCameraPanel(content_container);
    if (camera_panel_) {
        std::cout << "[LVGLWaylandInterface] 摄像头面板创建成功" << std::endl;
    }
    
    // 创建控制面板
    control_panel_ = createControlPanel(content_container);
    if (control_panel_) {
        std::cout << "[LVGLWaylandInterface] 控制面板创建成功" << std::endl;
    }
    
    // 创建底部面板
    footer_panel_ = createFooterPanel();
    if (footer_panel_) {
        lv_obj_set_flex_grow(footer_panel_, 0);
        std::cout << "[LVGLWaylandInterface] 底部面板创建成功" << std::endl;
    }
    
    // 强制刷新界面
    lv_obj_invalidate(main_screen_);
    lv_refr_now(NULL);
    
    std::cout << "[LVGLWaylandInterface] 主界面创建完成" << std::endl;
#endif
}

lv_obj_t* LVGLWaylandInterface::createHeaderPanel() {
#ifdef ENABLE_LVGL
    lv_obj_t* panel = lv_obj_create(main_screen_);
    lv_obj_set_size(panel, lv_pct(100), 80);
    lv_obj_set_style_bg_color(panel, color_surface_, 0);
    lv_obj_set_style_bg_opa(panel, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(panel, 8, 0);
    lv_obj_set_style_pad_all(panel, 10, 0);
    
    // 标题标签
    header_widgets_.title_label = lv_label_create(panel);
    lv_label_set_text(header_widgets_.title_label, "竹子识别系统 - Wayland版本");
    lv_obj_set_style_text_color(header_widgets_.title_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(header_widgets_.title_label, &lv_font_montserrat_20, 0);
    lv_obj_align(header_widgets_.title_label, LV_ALIGN_LEFT_MID, 0, 0);
    
    // 状态标签
    header_widgets_.status_label = lv_label_create(panel);
    lv_label_set_text(header_widgets_.status_label, "Wayland就绪");
    lv_obj_set_style_text_color(header_widgets_.status_label, color_success_, 0);
    lv_obj_align(header_widgets_.status_label, LV_ALIGN_RIGHT_MID, 0, 0);
    
    return panel;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLWaylandInterface::createCameraPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* panel = lv_obj_create(parent);
    lv_obj_set_size(panel, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_color(panel, color_surface_, 0);
    lv_obj_set_style_bg_opa(panel, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(panel, 8, 0);
    lv_obj_set_style_pad_all(panel, 10, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(panel);
    lv_label_set_text(title, "摄像头画面 (Wayland)");
    lv_obj_set_style_text_color(title, lv_color_white(), 0);
    lv_obj_align(title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // 创建Canvas用于显示摄像头画面
    camera_canvas_ = lv_canvas_create(panel);
    
    // 设置Canvas尺寸 (960x640, ARGB8888格式)
    static lv_color32_t canvas_buffer[960 * 640];
    lv_canvas_set_buffer(camera_canvas_, canvas_buffer, 960, 640, LV_COLOR_FORMAT_ARGB8888);
    
    lv_obj_set_size(camera_canvas_, 960, 640);
    lv_obj_align(camera_canvas_, LV_ALIGN_CENTER, 0, 10);
    
    // 初始填充黑色
    lv_canvas_fill_bg(camera_canvas_, lv_color_black(), LV_OPA_COVER);
    
    // 添加"等待Wayland视频流"文字
    lv_draw_label_dsc_t label_dsc;
    lv_draw_label_dsc_init(&label_dsc);
    label_dsc.color = lv_color_white();
    label_dsc.font = &lv_font_montserrat_24;
    
    lv_point_t text_pos = {480 - 100, 320};
    lv_canvas_draw_text(camera_canvas_, &text_pos, 200, &label_dsc, "等待Wayland视频流");
    
    camera_widgets_.canvas = camera_canvas_;
    
    return panel;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLWaylandInterface::createControlPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* panel = lv_obj_create(parent);
    lv_obj_set_size(panel, lv_pct(30), lv_pct(100));
    lv_obj_set_style_bg_color(panel, color_surface_, 0);
    lv_obj_set_style_bg_opa(panel, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(panel, 8, 0);
    lv_obj_set_style_pad_all(panel, 10, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(panel);
    lv_label_set_text(title, "控制面板");
    lv_obj_set_style_text_color(title, lv_color_white(), 0);
    lv_obj_align(title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // 启动按钮
    control_widgets_.start_btn = lv_btn_create(panel);
    lv_obj_set_size(control_widgets_.start_btn, 200, 50);
    lv_obj_align(control_widgets_.start_btn, LV_ALIGN_TOP_MID, 0, 40);
    lv_obj_set_style_bg_color(control_widgets_.start_btn, color_success_, 0);
    
    lv_obj_t* start_label = lv_label_create(control_widgets_.start_btn);
    lv_label_set_text(start_label, "启动系统");
    lv_obj_center(start_label);
    
    // 停止按钮
    control_widgets_.stop_btn = lv_btn_create(panel);
    lv_obj_set_size(control_widgets_.stop_btn, 200, 50);
    lv_obj_align(control_widgets_.stop_btn, LV_ALIGN_TOP_MID, 0, 100);
    lv_obj_set_style_bg_color(control_widgets_.stop_btn, color_warning_, 0);
    
    lv_obj_t* stop_label = lv_label_create(control_widgets_.stop_btn);
    lv_label_set_text(stop_label, "停止系统");
    lv_obj_center(stop_label);
    
    // 紧急停止按钮
    control_widgets_.emergency_btn = lv_btn_create(panel);
    lv_obj_set_size(control_widgets_.emergency_btn, 200, 50);
    lv_obj_align(control_widgets_.emergency_btn, LV_ALIGN_TOP_MID, 0, 160);
    lv_obj_set_style_bg_color(control_widgets_.emergency_btn, color_error_, 0);
    
    lv_obj_t* emergency_label = lv_label_create(control_widgets_.emergency_btn);
    lv_label_set_text(emergency_label, "紧急停止");
    lv_obj_center(emergency_label);
    
    return panel;
#else
    return nullptr;
#endif
}

lv_obj_t* LVGLWaylandInterface::createFooterPanel() {
#ifdef ENABLE_LVGL
    lv_obj_t* panel = lv_obj_create(main_screen_);
    lv_obj_set_size(panel, lv_pct(100), 60);
    lv_obj_set_style_bg_color(panel, color_surface_, 0);
    lv_obj_set_style_bg_opa(panel, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(panel, 8, 0);
    lv_obj_set_style_pad_all(panel, 10, 0);
    
    // CPU标签
    footer_widgets_.cpu_label = lv_label_create(panel);
    lv_label_set_text(footer_widgets_.cpu_label, "CPU: --%");
    lv_obj_set_style_text_color(footer_widgets_.cpu_label, lv_color_white(), 0);
    lv_obj_align(footer_widgets_.cpu_label, LV_ALIGN_LEFT_MID, 0, 0);
    
    // 内存标签
    footer_widgets_.memory_label = lv_label_create(panel);
    lv_label_set_text(footer_widgets_.memory_label, "Memory: --MB");
    lv_obj_set_style_text_color(footer_widgets_.memory_label, lv_color_white(), 0);
    lv_obj_align(footer_widgets_.memory_label, LV_ALIGN_CENTER, 0, 0);
    
    // 检测计数标签
    footer_widgets_.detection_count_label = lv_label_create(panel);
    lv_label_set_text(footer_widgets_.detection_count_label, "Wayland Mode");
    lv_obj_set_style_text_color(footer_widgets_.detection_count_label, color_primary_, 0);
    lv_obj_align(footer_widgets_.detection_count_label, LV_ALIGN_RIGHT_MID, 0, 0);
    
    return panel;
#else
    return nullptr;
#endif
}

void LVGLWaylandInterface::uiLoop() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLWaylandInterface] UI主循环开始" << std::endl;
    
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    frame_count_ = 0;
    
    // 标记为完全初始化
    fully_initialized_.store(true);
    
    try {
        while (!should_stop_.load()) {
            auto current_time = std::chrono::high_resolution_clock::now();
            
            try {
                // 处理LVGL任务
                if (display_ && main_screen_) {
                    uint32_t time_till_next = lv_timer_handler();
                    
                    // 定期更新界面数据
                    static int update_counter = 0;
                    if (++update_counter >= 5) {  // 每5次循环更新一次
                        update_counter = 0;
                        updateInterface();
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
                    
                    // 控制刷新率
                    uint32_t sleep_time = std::max(std::min(time_till_next, 33u), 16u);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[LVGLWaylandInterface] UI循环异常: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    } catch (...) {
        std::cerr << "[LVGLWaylandInterface] UI主循环致命异常" << std::endl;
    }
    
    fully_initialized_.store(false);
    std::cout << "[LVGLWaylandInterface] UI主循环结束" << std::endl;
#endif
}

void LVGLWaylandInterface::updateInterface() {
#ifdef ENABLE_LVGL
    // 更新Canvas内容
    if (new_frame_available_.load() && camera_canvas_) {
        std::lock_guard<std::mutex> lock(canvas_mutex_);
        
        if (!latest_frame_.empty()) {
            try {
                // 获取Canvas缓冲区信息
                lv_img_dsc_t* canvas_dsc = lv_canvas_get_image(camera_canvas_);
                if (canvas_dsc && canvas_dsc->data) {
                    // 转换OpenCV Mat到LVGL格式
                    cv::Mat display_frame;
                    
                    // 确保帧格式为BGRA
                    if (latest_frame_.channels() == 4) {
                        display_frame = latest_frame_.clone();
                    } else if (latest_frame_.channels() == 3) {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_BGR2BGRA);
                    } else {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_GRAY2BGRA);
                    }
                    
                    // 调整到Canvas尺寸 (960x640)
                    if (display_frame.cols != 960 || display_frame.rows != 640) {
                        cv::resize(display_frame, display_frame, cv::Size(960, 640),
                                   0, 0, cv::INTER_LINEAR);
                    }
                    
                    // 确保数据连续性
                    if (!display_frame.isContinuous()) {
                        display_frame = display_frame.clone();
                    }
                    
                    // 转换像素格式：BGRA -> ARGB8888
                    uint32_t* canvas_buffer = (uint32_t*)canvas_dsc->data;
                    const uint8_t* src_data = display_frame.data;
                    const size_t step = display_frame.step[0];
                    
                    for (int y = 0; y < 640; y++) {
                        const uint8_t* row_ptr = src_data + y * step;
                        uint32_t* canvas_row = canvas_buffer + y * 960;
                        
                        for (int x = 0; x < 960; x++) {
                            const uint8_t* pixel = row_ptr + x * 4;
                            uint8_t b = pixel[0];
                            uint8_t g = pixel[1];
                            uint8_t r = pixel[2];
                            uint8_t a = pixel[3];
                            
                            // LVGL ARGB8888格式：A在最高位
                            canvas_row[x] = (a << 24) | (r << 16) | (g << 8) | b;
                        }
                    }
                    
                    // 刷新Canvas显示
                    lv_obj_invalidate(camera_canvas_);
                }
                
                new_frame_available_.store(false);
                
            } catch (const std::exception& e) {
                std::cerr << "[LVGLWaylandInterface] Canvas更新异常: " << e.what() << std::endl;
                new_frame_available_.store(false);
            }
        }
    }
    
    // 更新系统状态
    if (jetson_monitor_) {
        try {
            // 更新CPU使用率
            if (footer_widgets_.cpu_label) {
                float cpu_usage = 45.0f; // 这里应该从jetson_monitor_获取实际数据
                std::ostringstream cpu_text;
                cpu_text << "CPU: " << std::fixed << std::setprecision(1) << cpu_usage << "%";
                lv_label_set_text(footer_widgets_.cpu_label, cpu_text.str().c_str());
            }
            
            // 更新内存使用情况
            if (footer_widgets_.memory_label) {
                float memory_mb = 1024.0f; // 这里应该从jetson_monitor_获取实际数据
                std::ostringstream mem_text;
                mem_text << "Memory: " << std::fixed << std::setprecision(0) << memory_mb << "MB";
                lv_label_set_text(footer_widgets_.memory_label, mem_text.str().c_str());
            }
            
            // 更新状态信息
            if (header_widgets_.status_label) {
                const char* status_text = fully_initialized_.load() ? "Wayland运行中" : "初始化中";
                lv_label_set_text(header_widgets_.status_label, status_text);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[LVGLWaylandInterface] 状态更新异常: " << e.what() << std::endl;
        }
    }
#endif
}

void LVGLWaylandInterface::waylandFlushCallback(lv_display_t* display, const lv_area_t* area, lv_color_t* color_p) {
    // 虚拟刷新回调 - 在没有真实Wayland驱动时使用
    // 实际部署时，这个函数不会被调用，因为使用真实的Wayland驱动
    LV_UNUSED(display);
    LV_UNUSED(area);
    LV_UNUSED(color_p);
    
    // 告诉LVGL刷新完成
    lv_display_flush_ready(display);
}

} // namespace ui
} // namespace bamboo_cut