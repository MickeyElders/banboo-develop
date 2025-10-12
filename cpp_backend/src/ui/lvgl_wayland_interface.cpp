/**
 * @file lvgl_wayland_interface.cpp
 * @brief LVGL Wayland接口实现 - Weston合成器架构支持
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/utils/logger.h"
#include <lvgl.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <mutex>
#include <chrono>

// 暂时使用fallback实现，等待lv_drivers集成
#define HAS_WAYLAND_DRIVER 0
#warning "Using fallback LVGL implementation without lv_drivers"

namespace bamboo_cut {
namespace ui {

/**
 * @brief LVGL Wayland接口内部实现类
 */
class LVGLWaylandInterface::Impl {
public:
    LVGLWaylandConfig config_;
    
    // LVGL对象
    lv_display_t* display_ = nullptr;
    lv_indev_t* touch_indev_ = nullptr;
    lv_indev_t* pointer_indev_ = nullptr;
    lv_indev_t* keyboard_indev_ = nullptr;
    
    // UI元素
    lv_obj_t* main_screen_ = nullptr;
    lv_obj_t* header_panel_ = nullptr;
    lv_obj_t* camera_panel_ = nullptr;
    lv_obj_t* control_panel_ = nullptr;
    lv_obj_t* footer_panel_ = nullptr;
    lv_obj_t* camera_canvas_ = nullptr;
    
    // 线程同步
    std::mutex ui_mutex_;
    std::mutex canvas_mutex_;
    std::atomic<bool> should_stop_{false};
    
    // Canvas更新
    cv::Mat latest_frame_;
    std::atomic<bool> new_frame_available_{false};
    
    // 初始化状态
    bool wayland_initialized_ = false;
    bool display_initialized_ = false;
    bool input_initialized_ = false;
    
    Impl() = default;
    ~Impl() = default;
    
    bool checkWaylandEnvironment();
    bool initializeWaylandDisplay();
    bool initializeInput();
    void initializeTheme();
    void createMainInterface();
    void updateCanvasFromFrame();
};

LVGLWaylandInterface::LVGLWaylandInterface() 
    : pImpl_(std::make_unique<Impl>()) {
}

LVGLWaylandInterface::~LVGLWaylandInterface() {
    stop();
    cleanup();
}

bool LVGLWaylandInterface::initialize(const LVGLWaylandConfig& config) {
    std::lock_guard<std::mutex> lock(pImpl_->ui_mutex_);
    
    pImpl_->config_ = config;
    
    // 检查Wayland环境
    if (!pImpl_->checkWaylandEnvironment()) {
        std::cerr << "Wayland环境不可用" << std::endl;
        return false;
    }
    
    // 初始化LVGL
    if (!lv_is_initialized()) {
        lv_init();
    }
    
    // 初始化显示
    if (!pImpl_->initializeWaylandDisplay()) {
        std::cerr << "Wayland显示初始化失败" << std::endl;
        return false;
    }
    
    // 初始化输入设备
    if (config.enable_touch) {
        if (!pImpl_->initializeInput()) {
            std::cerr << "输入设备初始化失败" << std::endl;
            return false;
        }
    }
    
    // 初始化主题
    pImpl_->initializeTheme();
    
    // 创建主界面
    pImpl_->createMainInterface();
    
    fully_initialized_.store(true);
    return true;
}

bool LVGLWaylandInterface::start() {
    if (running_.load()) {
        return true;
    }
    
    pImpl_->should_stop_.store(false);
    
    // 启动UI线程
    ui_thread_ = std::thread(&LVGLWaylandInterface::uiThreadLoop, this);
    
    running_.store(true);
    return true;
}

void LVGLWaylandInterface::stop() {
    if (!running_.load()) {
        return;
    }
    
    pImpl_->should_stop_.store(true);
    
    if (ui_thread_.joinable()) {
        ui_thread_.join();
    }
    
    running_.store(false);
    fully_initialized_.store(false);
}

bool LVGLWaylandInterface::isFullyInitialized() const {
    if (!running_.load() || !pImpl_->display_ || !pImpl_->main_screen_) {
        return false;
    }
    
    if (!pImpl_->header_panel_ || !pImpl_->camera_panel_ || 
        !pImpl_->control_panel_ || !pImpl_->footer_panel_) {
        return false;
    }
    
    if (!pImpl_->camera_canvas_) {
        return false;
    }
    
    return fully_initialized_.load();
}

bool LVGLWaylandInterface::isRunning() const {
    return running_.load();
}

lv_obj_t* LVGLWaylandInterface::getCameraCanvas() {
    std::lock_guard<std::mutex> lock(pImpl_->ui_mutex_);
    return pImpl_->camera_canvas_;
}

void LVGLWaylandInterface::updateCameraCanvas(const cv::Mat& frame) {
    if (frame.empty() || !pImpl_->camera_canvas_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->canvas_mutex_);
    
    try {
        pImpl_->latest_frame_ = frame.clone();
        pImpl_->new_frame_available_.store(true);
    } catch (const std::exception& e) {
        std::cerr << "Canvas更新失败: " << e.what() << std::endl;
    }
}

bool LVGLWaylandInterface::isWaylandEnvironmentAvailable() {
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        return false;
    }
    
    // 检查wayland socket是否存在
    std::string socket_path = "/run/user/" + std::to_string(getuid()) + "/" + wayland_display;
    return access(socket_path.c_str(), F_OK) == 0;
}

void LVGLWaylandInterface::uiThreadLoop() {
    auto last_update = std::chrono::steady_clock::now();
    const auto frame_time = std::chrono::milliseconds(1000 / pImpl_->config_.refresh_rate);
    
    while (!pImpl_->should_stop_.load()) {
        auto now = std::chrono::steady_clock::now();
        
        // 处理LVGL任务
        {
            std::lock_guard<std::mutex> lock(pImpl_->ui_mutex_);
            lv_timer_handler();
        }
        
        // 更新Canvas（如果有新帧）
        if (pImpl_->new_frame_available_.load()) {
            pImpl_->updateCanvasFromFrame();
            pImpl_->new_frame_available_.store(false);
        }
        
        // 控制帧率
        auto elapsed = now - last_update;
        if (elapsed < frame_time) {
            std::this_thread::sleep_for(frame_time - elapsed);
        }
        last_update = std::chrono::steady_clock::now();
    }
}

void LVGLWaylandInterface::createUI() {
    pImpl_->createMainInterface();
}

void LVGLWaylandInterface::createHeaderPanel() {
    // 创建头部面板的实现在createMainInterface中
}

void LVGLWaylandInterface::createCameraPanel() {
    // 创建摄像头面板的实现在createMainInterface中
}

void LVGLWaylandInterface::createControlPanel() {
    // 创建控制面板的实现在createMainInterface中
}

void LVGLWaylandInterface::createFooterPanel() {
    // 创建底部面板的实现在createMainInterface中
}

void LVGLWaylandInterface::setupEventHandlers() {
    // 事件处理器设置
}

bool LVGLWaylandInterface::initializeInputDevices() {
    return pImpl_->initializeInput();
}

void LVGLWaylandInterface::cleanup() {
    if (pImpl_) {
        if (pImpl_->display_) {
            lv_display_delete(pImpl_->display_);
            pImpl_->display_ = nullptr;
        }
        
        if (pImpl_->touch_indev_) {
            lv_indev_delete(pImpl_->touch_indev_);
            pImpl_->touch_indev_ = nullptr;
        }
        
        if (pImpl_->pointer_indev_) {
            lv_indev_delete(pImpl_->pointer_indev_);
            pImpl_->pointer_indev_ = nullptr;
        }
        
        if (pImpl_->keyboard_indev_) {
            lv_indev_delete(pImpl_->keyboard_indev_);
            pImpl_->keyboard_indev_ = nullptr;
        }
    }
}

// ========== Impl 类方法实现 ==========

bool LVGLWaylandInterface::Impl::checkWaylandEnvironment() {
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        std::cerr << "WAYLAND_DISPLAY环境变量未设置" << std::endl;
        return false;
    }
    
    // 检查wayland socket是否存在
    std::string socket_path = "/run/user/" + std::to_string(getuid()) + "/" + wayland_display;
    if (access(socket_path.c_str(), F_OK) != 0) {
        std::cerr << "Wayland socket不存在: " << socket_path << std::endl;
        return false;
    }
    
    wayland_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeWaylandDisplay() {
    // Fallback实现：创建基本显示缓冲区
    static lv_color_t* buf1 = nullptr;
    static lv_color_t* buf2 = nullptr;
    
    size_t buf_size = config_.screen_width * config_.screen_height;
    buf1 = (lv_color_t*)malloc(buf_size * sizeof(lv_color_t));
    buf2 = (lv_color_t*)malloc(buf_size * sizeof(lv_color_t));
    
    if (!buf1 || !buf2) {
        std::cerr << "显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        free(buf1);
        free(buf2);
        return false;
    }
    
    lv_display_set_buffers(display_, buf1, buf2, buf_size * sizeof(lv_color_t), LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置刷新回调（空实现，因为没有真实的Wayland驱动）
    lv_display_set_flush_cb(display_, [](lv_display_t* disp, const lv_area_t* area, uint8_t* color_p) {
        lv_display_flush_ready(disp);
    });
    
    display_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeInput() {
    // Fallback实现：创建虚拟输入设备
    touch_indev_ = lv_indev_create();
    if (touch_indev_) {
        lv_indev_set_type(touch_indev_, LV_INDEV_TYPE_POINTER);
        lv_indev_set_read_cb(touch_indev_, [](lv_indev_t* indev, lv_indev_data_t* data) {
            // 空实现，没有真实的触摸输入
            data->state = LV_INDEV_STATE_RELEASED;
        });
    }
    
    input_initialized_ = true;
    return true;
}

void LVGLWaylandInterface::Impl::initializeTheme() {
    // 使用默认主题
    lv_theme_t* theme = lv_theme_default_init(display_, 
                                            lv_palette_main(LV_PALETTE_BLUE), 
                                            lv_palette_main(LV_PALETTE_RED), 
                                            true, 
                                            LV_FONT_DEFAULT);
    lv_display_set_theme(display_, theme);
}

void LVGLWaylandInterface::Impl::createMainInterface() {
    // 创建主屏幕
    main_screen_ = lv_obj_create(nullptr);
    lv_obj_set_size(main_screen_, config_.screen_width, config_.screen_height);
    lv_obj_clear_flag(main_screen_, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(main_screen_, lv_color_hex(0x1E1E1E), 0);
    
    // 创建头部面板 (高度: 60px)
    header_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(header_panel_, config_.screen_width, 60);
    lv_obj_align(header_panel_, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(header_panel_, lv_color_hex(0x2A2A2A), 0);
    lv_obj_clear_flag(header_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 头部标题
    lv_obj_t* title_label = lv_label_create(header_panel_);
    lv_label_set_text(title_label, "Bamboo Recognition System - Wayland Mode");
    lv_obj_set_style_text_color(title_label, lv_color_white(), 0);
    lv_obj_center(title_label);
    
    // 创建主容器 (中间部分)
    lv_obj_t* main_container = lv_obj_create(main_screen_);
    lv_obj_set_size(main_container, config_.screen_width, config_.screen_height - 120); // 减去头部和底部
    lv_obj_align(main_container, LV_ALIGN_CENTER, 0, 0);
    lv_obj_set_style_bg_opa(main_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(main_container, LV_OPA_TRANSP, 0);
    lv_obj_clear_flag(main_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 创建摄像头面板 (左侧，宽度: 60%)
    int camera_width = (int)(config_.screen_width * 0.6);
    camera_panel_ = lv_obj_create(main_container);
    lv_obj_set_size(camera_panel_, camera_width, config_.screen_height - 120);
    lv_obj_align(camera_panel_, LV_ALIGN_LEFT_MID, 10, 0);
    lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0x1A1A1A), 0);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 摄像头Canvas
    camera_canvas_ = lv_canvas_create(camera_panel_);
    lv_obj_set_size(camera_canvas_, camera_width - 20, (config_.screen_height - 120) - 20);
    lv_obj_center(camera_canvas_);
    
    // 为Canvas分配缓冲区
    static lv_color_t* canvas_buf = nullptr;
    size_t canvas_buf_size = (camera_width - 20) * ((config_.screen_height - 120) - 20);
    canvas_buf = (lv_color_t*)malloc(canvas_buf_size * sizeof(lv_color_t));
    if (canvas_buf) {
        lv_canvas_set_buffer(camera_canvas_, canvas_buf, camera_width - 20, (config_.screen_height - 120) - 20, LV_COLOR_FORMAT_RGB888);
        lv_canvas_fill_bg(camera_canvas_, lv_color_hex(0x333333), LV_OPA_COVER);
    }
    
    // 创建控制面板 (右侧，宽度: 35%)
    int control_width = (int)(config_.screen_width * 0.35);
    control_panel_ = lv_obj_create(main_container);
    lv_obj_set_size(control_panel_, control_width, config_.screen_height - 120);
    lv_obj_align(control_panel_, LV_ALIGN_RIGHT_MID, -10, 0);
    lv_obj_set_style_bg_color(control_panel_, lv_color_hex(0x2A2A2A), 0);
    lv_obj_clear_flag(control_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 控制面板标题
    lv_obj_t* control_title = lv_label_create(control_panel_);
    lv_label_set_text(control_title, "控制面板");
    lv_obj_set_style_text_color(control_title, lv_color_white(), 0);
    lv_obj_align(control_title, LV_ALIGN_TOP_MID, 0, 10);
    
    // 创建一些控制按钮
    lv_obj_t* start_btn = lv_btn_create(control_panel_);
    lv_obj_set_size(start_btn, control_width - 40, 40);
    lv_obj_align(start_btn, LV_ALIGN_TOP_MID, 0, 50);
    lv_obj_t* start_label = lv_label_create(start_btn);
    lv_label_set_text(start_label, "开始检测");
    lv_obj_center(start_label);
    
    lv_obj_t* stop_btn = lv_btn_create(control_panel_);
    lv_obj_set_size(stop_btn, control_width - 40, 40);
    lv_obj_align(stop_btn, LV_ALIGN_TOP_MID, 0, 100);
    lv_obj_t* stop_label = lv_label_create(stop_btn);
    lv_label_set_text(stop_label, "停止检测");
    lv_obj_center(stop_label);
    
    // 创建底部面板 (高度: 60px)
    footer_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(footer_panel_, config_.screen_width, 60);
    lv_obj_align(footer_panel_, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_set_style_bg_color(footer_panel_, lv_color_hex(0x2A2A2A), 0);
    lv_obj_clear_flag(footer_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 底部状态信息
    lv_obj_t* status_label = lv_label_create(footer_panel_);
    lv_label_set_text(status_label, "状态: Wayland模式 - 准备就绪");
    lv_obj_set_style_text_color(status_label, lv_color_white(), 0);
    lv_obj_center(status_label);
    
    // 加载主屏幕
    lv_screen_load(main_screen_);
}

void LVGLWaylandInterface::Impl::updateCanvasFromFrame() {
    if (!camera_canvas_ || latest_frame_.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(canvas_mutex_);
    
    try {
        // 获取Canvas尺寸
        lv_coord_t canvas_width = lv_obj_get_width(camera_canvas_);
        lv_coord_t canvas_height = lv_obj_get_height(camera_canvas_);
        
        // 调整图像尺寸
        cv::Mat resized_frame;
        cv::resize(latest_frame_, resized_frame, cv::Size(canvas_width, canvas_height));
        
        // 转换颜色格式 (BGR -> RGB)
        cv::Mat rgb_frame;
        cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
        
        // 更新Canvas缓冲区
        lv_color_t* canvas_buf = (lv_color_t*)lv_canvas_get_buf(camera_canvas_);
        if (canvas_buf) {
            for (int y = 0; y < canvas_height; y++) {
                for (int x = 0; x < canvas_width; x++) {
                    cv::Vec3b pixel = rgb_frame.at<cv::Vec3b>(y, x);
                    lv_color_t color = lv_color_make(pixel[0], pixel[1], pixel[2]);
                    canvas_buf[y * canvas_width + x] = color;
                }
            }
            lv_obj_invalidate(camera_canvas_);
        }
    } catch (const std::exception& e) {
        std::cerr << "Canvas帧更新失败: " << e.what() << std::endl;
    }
}

} // namespace ui
} // namespace bamboo_cut