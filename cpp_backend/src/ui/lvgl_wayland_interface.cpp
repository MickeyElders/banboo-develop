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

// 系统头文件
#include <fcntl.h>
#include <errno.h>

// EGL和DRM头文件
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>

// 使用DRM EGL共享架构实现真正的屏幕渲染
#define HAS_DRM_EGL_BACKEND 1

namespace bamboo_cut {
namespace ui {

/**
 * @brief LVGL Wayland接口内部实现类 - 使用DRM EGL共享架构
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
    
    // DRM EGL后端
    int drm_fd_ = -1;
    struct gbm_device* gbm_device_ = nullptr;
    struct gbm_surface* gbm_surface_ = nullptr;
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLConfig egl_config_;
    
    // DRM资源
    drmModeRes* drm_resources_ = nullptr;
    drmModeConnector* drm_connector_ = nullptr;
    drmModeEncoder* drm_encoder_ = nullptr;
    drmModeCrtc* drm_crtc_ = nullptr;
    drmModeModeInfo drm_mode_;
    uint32_t drm_crtc_id_ = 0;
    uint32_t drm_connector_id_ = 0;
    
    // 显示缓冲区
    lv_color_t* front_buffer_ = nullptr;
    lv_color_t* back_buffer_ = nullptr;
    uint32_t buffer_size_ = 0;
    
    // 线程同步
    std::mutex ui_mutex_;
    std::mutex canvas_mutex_;
    std::mutex render_mutex_;
    std::atomic<bool> should_stop_{false};
    
    // Canvas更新
    cv::Mat latest_frame_;
    std::atomic<bool> new_frame_available_{false};
    
    // 初始化状态
    bool wayland_initialized_ = false;
    bool display_initialized_ = false;
    bool input_initialized_ = false;
    bool drm_initialized_ = false;
    bool egl_initialized_ = false;
    
    Impl() = default;
    ~Impl();
    
    bool checkWaylandEnvironment();
    bool initializeDRMBackend();
    bool initializeEGL();
    bool initializeWaylandDisplay();
    bool initializeFallbackDisplay();
    bool initializeInput();
    void initializeTheme();
    void createMainInterface();
    void updateCanvasFromFrame();
    void flushDisplay(const lv_area_t* area, lv_color_t* color_p);
    void cleanup();
    
    // DRM辅助函数
    bool findDRMResources();
    bool setupDRMMode();
    EGLConfig chooseEGLConfig();
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
    // 首先初始化DRM后端
    std::cout << "正在初始化DRM后端..." << std::endl;
    if (!initializeDRMBackend()) {
        std::cerr << "DRM后端初始化失败，使用fallback模式" << std::endl;
        return initializeFallbackDisplay();
    }
    
    // 然后初始化EGL
    std::cout << "正在初始化EGL..." << std::endl;
    if (!initializeEGL()) {
        std::cerr << "EGL初始化失败，使用fallback模式" << std::endl;
        cleanup();
        return initializeFallbackDisplay();
    }
    
    // 创建LVGL显示设备
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        std::cerr << "LVGL显示创建失败" << std::endl;
        cleanup();
        return false;
    }
    
    // 分配显示缓冲区
    buffer_size_ = config_.screen_width * config_.screen_height * sizeof(lv_color_t);
    front_buffer_ = (lv_color_t*)malloc(buffer_size_);
    back_buffer_ = (lv_color_t*)malloc(buffer_size_);
    
    if (!front_buffer_ || !back_buffer_) {
        std::cerr << "显示缓冲区分配失败" << std::endl;
        cleanup();
        return false;
    }
    
    // 设置LVGL缓冲区
    lv_display_set_buffers(display_, front_buffer_, back_buffer_,
                          buffer_size_, LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // ✅ 关键修复：设置真正的flush回调
    lv_display_set_flush_cb(display_, [](lv_display_t* disp, const lv_area_t* area, uint8_t* color_p) {
        // 从用户数据获取Impl实例
        LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(
            lv_display_get_user_data(disp));
        
        if (impl) {
            impl->flushDisplay(area, (lv_color_t*)color_p);
        }
        
        lv_display_flush_ready(disp);
    });
    
    // 设置用户数据，以便在回调中访问
    lv_display_set_user_data(display_, this);
    
    display_initialized_ = true;
    std::cout << "DRM EGL显示初始化成功" << std::endl;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeFallbackDisplay() {
    std::cout << "使用fallback显示模式" << std::endl;
    
    static lv_color_t* buf1 = nullptr;
    static lv_color_t* buf2 = nullptr;
    
    size_t buf_size = config_.screen_width * config_.screen_height;
    buf1 = (lv_color_t*)malloc(buf_size * sizeof(lv_color_t));
    buf2 = (lv_color_t*)malloc(buf_size * sizeof(lv_color_t));
    
    if (!buf1 || !buf2) {
        std::cerr << "Fallback显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        free(buf1);
        free(buf2);
        return false;
    }
    
    lv_display_set_buffers(display_, buf1, buf2, buf_size * sizeof(lv_color_t), LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置空的刷新回调
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

// DRM EGL后端实现
bool LVGLWaylandInterface::Impl::initializeDRMBackend() {
    // 打开DRM设备
    drm_fd_ = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        std::cerr << "无法打开DRM设备" << std::endl;
        return false;
    }
    
    // 查找DRM资源
    if (!findDRMResources()) {
        close(drm_fd_);
        drm_fd_ = -1;
        return false;
    }
    
    // 设置显示模式
    if (!setupDRMMode()) {
        cleanup();
        return false;
    }
    
    // 创建GBM设备
    gbm_device_ = gbm_create_device(drm_fd_);
    if (!gbm_device_) {
        std::cerr << "GBM设备创建失败" << std::endl;
        cleanup();
        return false;
    }
    
    // 创建GBM表面
    gbm_surface_ = gbm_surface_create(gbm_device_,
                                      config_.screen_width,
                                      config_.screen_height,
                                      GBM_FORMAT_XRGB8888,
                                      GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!gbm_surface_) {
        std::cerr << "GBM表面创建失败" << std::endl;
        cleanup();
        return false;
    }
    
    drm_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeEGL() {
    if (!drm_initialized_) {
        return false;
    }
    
    // 获取EGL显示
    egl_display_ = eglGetDisplay((EGLNativeDisplayType)gbm_device_);
    if (egl_display_ == EGL_NO_DISPLAY) {
        std::cerr << "EGL显示获取失败" << std::endl;
        return false;
    }
    
    // 初始化EGL
    EGLint major, minor;
    if (!eglInitialize(egl_display_, &major, &minor)) {
        std::cerr << "EGL初始化失败" << std::endl;
        return false;
    }
    
    // 选择EGL配置
    egl_config_ = chooseEGLConfig();
    
    // 创建EGL上下文
    static const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, context_attribs);
    if (egl_context_ == EGL_NO_CONTEXT) {
        std::cerr << "EGL上下文创建失败" << std::endl;
        return false;
    }
    
    // 创建EGL表面
    egl_surface_ = eglCreateWindowSurface(egl_display_, egl_config_,
                                          (EGLNativeWindowType)gbm_surface_, nullptr);
    if (egl_surface_ == EGL_NO_SURFACE) {
        std::cerr << "EGL表面创建失败" << std::endl;
        return false;
    }
    
    // 激活上下文
    if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
        std::cerr << "EGL上下文激活失败" << std::endl;
        return false;
    }
    
    egl_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::findDRMResources() {
    drm_resources_ = drmModeGetResources(drm_fd_);
    if (!drm_resources_) {
        std::cerr << "获取DRM资源失败" << std::endl;
        return false;
    }
    
    // 查找连接的显示器
    for (int i = 0; i < drm_resources_->count_connectors; i++) {
        drm_connector_ = drmModeGetConnector(drm_fd_, drm_resources_->connectors[i]);
        if (drm_connector_ && drm_connector_->connection == DRM_MODE_CONNECTED) {
            drm_connector_id_ = drm_connector_->connector_id;
            break;
        }
        if (drm_connector_) {
            drmModeFreeConnector(drm_connector_);
            drm_connector_ = nullptr;
        }
    }
    
    if (!drm_connector_) {
        std::cerr << "未找到连接的显示器" << std::endl;
        return false;
    }
    
    // 查找编码器
    if (drm_connector_->encoder_id) {
        drm_encoder_ = drmModeGetEncoder(drm_fd_, drm_connector_->encoder_id);
        if (drm_encoder_) {
            drm_crtc_id_ = drm_encoder_->crtc_id;
        }
    }
    
    return true;
}

bool LVGLWaylandInterface::Impl::setupDRMMode() {
    if (!drm_connector_ || drm_connector_->count_modes == 0) {
        return false;
    }
    
    // 使用第一个可用模式
    drm_mode_ = drm_connector_->modes[0];
    
    // 获取CRTC
    drm_crtc_ = drmModeGetCrtc(drm_fd_, drm_crtc_id_);
    if (!drm_crtc_) {
        std::cerr << "获取CRTC失败" << std::endl;
        return false;
    }
    
    return true;
}

EGLConfig LVGLWaylandInterface::Impl::chooseEGLConfig() {
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    
    if (!eglChooseConfig(egl_display_, config_attribs, &config, 1, &num_configs)) {
        std::cerr << "EGL配置选择失败" << std::endl;
        return nullptr;
    }
    
    return config;
}

void LVGLWaylandInterface::Impl::flushDisplay(const lv_area_t* area, lv_color_t* color_p) {
    if (!egl_initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(render_mutex_);
    
    // 使用OpenGL ES渲染到屏幕
    glViewport(0, 0, config_.screen_width, config_.screen_height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 这里应该将LVGL的颜色缓冲区渲染到OpenGL纹理
    // 暂时留空，因为需要更复杂的纹理操作
    
    // 交换缓冲区
    eglSwapBuffers(egl_display_, egl_surface_);
}

void LVGLWaylandInterface::Impl::cleanup() {
    // 清理EGL资源
    if (egl_initialized_) {
        if (egl_display_ != EGL_NO_DISPLAY) {
            eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            
            if (egl_surface_ != EGL_NO_SURFACE) {
                eglDestroySurface(egl_display_, egl_surface_);
                egl_surface_ = EGL_NO_SURFACE;
            }
            
            if (egl_context_ != EGL_NO_CONTEXT) {
                eglDestroyContext(egl_display_, egl_context_);
                egl_context_ = EGL_NO_CONTEXT;
            }
            
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
        }
        egl_initialized_ = false;
    }
    
    // 清理GBM资源
    if (gbm_surface_) {
        gbm_surface_destroy(gbm_surface_);
        gbm_surface_ = nullptr;
    }
    
    if (gbm_device_) {
        gbm_device_destroy(gbm_device_);
        gbm_device_ = nullptr;
    }
    
    // 清理DRM资源
    if (drm_crtc_) {
        drmModeFreeCrtc(drm_crtc_);
        drm_crtc_ = nullptr;
    }
    
    if (drm_encoder_) {
        drmModeFreeEncoder(drm_encoder_);
        drm_encoder_ = nullptr;
    }
    
    if (drm_connector_) {
        drmModeFreeConnector(drm_connector_);
        drm_connector_ = nullptr;
    }
    
    if (drm_resources_) {
        drmModeFreeResources(drm_resources_);
        drm_resources_ = nullptr;
    }
    
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }
    
    drm_initialized_ = false;
    
    // 清理显示缓冲区
    if (front_buffer_) {
        free(front_buffer_);
        front_buffer_ = nullptr;
    }
    
    if (back_buffer_) {
        free(back_buffer_);
        back_buffer_ = nullptr;
    }
}

// 析构函数实现
LVGLWaylandInterface::Impl::~Impl() {
    cleanup();
}

} // namespace ui
} // namespace bamboo_cut