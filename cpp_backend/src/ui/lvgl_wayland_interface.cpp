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

// EGL和Wayland头文件
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <vector>

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
    
    // Wayland EGL后端
    struct wl_display* wl_display_ = nullptr;
    struct wl_registry* wl_registry_ = nullptr;
    struct wl_compositor* wl_compositor_ = nullptr;
    struct wl_shell* wl_shell_ = nullptr;
    struct wl_surface* wl_surface_ = nullptr;
    struct wl_shell_surface* wl_shell_surface_ = nullptr;
    struct wl_egl_window* wl_egl_window_ = nullptr;
    
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLConfig egl_config_;
    
    // 显示缓冲区
    lv_color_t* front_buffer_ = nullptr;
    lv_color_t* back_buffer_ = nullptr;
    uint32_t buffer_size_ = 0;
    
    // OpenGL渲染资源
    GLuint shader_program_ = 0;
    GLuint texture_id_ = 0;
    GLuint vbo_ = 0;
    bool gl_resources_initialized_ = false;
    
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
    bool wayland_egl_initialized_ = false;
    bool egl_initialized_ = false;
    
    Impl() = default;
    ~Impl();
    
    bool checkWaylandEnvironment();
    bool initializeWaylandClient();
    bool initializeWaylandEGL();
    bool initializeWaylandDisplay();
    bool initializeFallbackDisplay();
    bool initializeInput();
    void initializeTheme();
    void createMainInterface();
    void updateCanvasFromFrame();
    void flushDisplay(const lv_area_t* area, lv_color_t* color_p);
    void cleanup();
    
    // Wayland辅助函数
    static void registryHandler(void* data, struct wl_registry* registry, uint32_t id, const char* interface, uint32_t version);
    static void registryRemover(void* data, struct wl_registry* registry, uint32_t id);
    EGLConfig chooseEGLConfig();
    
    // OpenGL渲染资源管理
    bool initializeGLResources();
    void cleanupGLResources();
    bool createShaderProgram();
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
    // 首先初始化Wayland客户端
    std::cout << "正在初始化Wayland客户端..." << std::endl;
    if (!initializeWaylandClient()) {
        std::cerr << "Wayland客户端初始化失败，使用fallback模式" << std::endl;
        return initializeFallbackDisplay();
    }
    
    // 然后初始化Wayland EGL
    std::cout << "正在初始化Wayland EGL..." << std::endl;
    if (!initializeWaylandEGL()) {
        std::cerr << "Wayland EGL初始化失败，使用fallback模式" << std::endl;
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
    std::cout << "Wayland EGL显示初始化成功" << std::endl;
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

// Wayland客户端实现
bool LVGLWaylandInterface::Impl::initializeWaylandClient() {
    // 连接到Wayland显示服务器
    wl_display_ = wl_display_connect(nullptr);
    if (!wl_display_) {
        std::cerr << "无法连接到Wayland显示服务器" << std::endl;
        return false;
    }
    
    // 获取registry并绑定全局对象
    wl_registry_ = wl_display_get_registry(wl_display_);
    if (!wl_registry_) {
        std::cerr << "无法获取Wayland registry" << std::endl;
        return false;
    }
    
    static const struct wl_registry_listener registry_listener = {
        registryHandler,
        registryRemover
    };
    
    wl_registry_add_listener(wl_registry_, &registry_listener, this);
    
    // 等待初始的roundtrip来获取所有全局对象
    wl_display_dispatch(wl_display_);
    wl_display_roundtrip(wl_display_);
    
    if (!wl_compositor_) {
        std::cerr << "Wayland compositor不可用" << std::endl;
        return false;
    }
    
    // 创建surface
    wl_surface_ = wl_compositor_create_surface(wl_compositor_);
    if (!wl_surface_) {
        std::cerr << "无法创建Wayland surface" << std::endl;
        return false;
    }
    
    // 如果有shell，创建shell surface
    if (wl_shell_) {
        wl_shell_surface_ = wl_shell_get_shell_surface(wl_shell_, wl_surface_);
        if (wl_shell_surface_) {
            wl_shell_surface_set_toplevel(wl_shell_surface_);
        }
    }
    
    wayland_egl_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeWaylandEGL() {
    if (!wayland_egl_initialized_) {
        return false;
    }
    
    // 创建EGL窗口
    wl_egl_window_ = wl_egl_window_create(wl_surface_, config_.screen_width, config_.screen_height);
    if (!wl_egl_window_) {
        std::cerr << "无法创建Wayland EGL窗口" << std::endl;
        return false;
    }
    
    // 获取EGL显示
    egl_display_ = eglGetDisplay((EGLNativeDisplayType)wl_display_);
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
                                          (EGLNativeWindowType)wl_egl_window_, nullptr);
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

// Wayland registry回调函数
void LVGLWaylandInterface::Impl::registryHandler(void* data, struct wl_registry* registry,
                                                  uint32_t id, const char* interface, uint32_t version) {
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    
    if (strcmp(interface, "wl_compositor") == 0) {
        impl->wl_compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 1));
    } else if (strcmp(interface, "wl_shell") == 0) {
        impl->wl_shell_ = static_cast<struct wl_shell*>(
            wl_registry_bind(registry, id, &wl_shell_interface, 1));
    }
}

void LVGLWaylandInterface::Impl::registryRemover(void* data, struct wl_registry* registry, uint32_t id) {
    // 处理全局对象移除（可选实现）
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
    
    // 初始化OpenGL资源（第一次调用时）
    if (!gl_resources_initialized_) {
        if (!initializeGLResources()) {
            std::cerr << "OpenGL资源初始化失败" << std::endl;
            return;
        }
        gl_resources_initialized_ = true;
    }
    
    // 设置视口
    glViewport(0, 0, config_.screen_width, config_.screen_height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 计算渲染区域
    int32_t x1 = area->x1;
    int32_t y1 = area->y1;
    int32_t w = area->x2 - area->x1 + 1;
    int32_t h = area->y2 - area->y1 + 1;
    
    // 绑定纹理
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    
    // 正确的LVGL颜色格式转换
    std::vector<uint8_t> rgba_data(w * h * 4);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            lv_color_t pixel = color_p[y * w + x];
            
            // 根据LVGL v9.x的颜色结构直接访问
            int idx = (y * w + x) * 4;
            
            #if LV_COLOR_DEPTH == 16
                // RGB565格式
                rgba_data[idx + 0] = (pixel.red & 0x1F) << 3;      // R: 5bit -> 8bit
                rgba_data[idx + 1] = (pixel.green & 0x3F) << 2;    // G: 6bit -> 8bit
                rgba_data[idx + 2] = (pixel.blue & 0x1F) << 3;     // B: 5bit -> 8bit
            #elif LV_COLOR_DEPTH == 32
                // ARGB8888格式
                rgba_data[idx + 0] = pixel.red;
                rgba_data[idx + 1] = pixel.green;
                rgba_data[idx + 2] = pixel.blue;
            #else
                // 默认处理
                rgba_data[idx + 0] = pixel.red << 3;
                rgba_data[idx + 1] = pixel.green << 2;
                rgba_data[idx + 2] = pixel.blue << 3;
            #endif
            
            rgba_data[idx + 3] = 255;  // A: 完全不透明
        }
    }
    
    // 只在第一次或全屏更新时创建完整纹理
    if (x1 == 0 && y1 == 0 && w == config_.screen_width && h == config_.screen_height) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, config_.screen_width, config_.screen_height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_data.data());
    } else {
        // 部分更新：Y坐标需要翻转（OpenGL坐标系）
        int32_t gl_y = config_.screen_height - y1 - h;
        glTexSubImage2D(GL_TEXTURE_2D, 0, x1, gl_y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_data.data());
    }
    
    // 使用shader程序
    glUseProgram(shader_program_);
    
    // 绑定VBO和设置属性
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    
    GLint pos_attr = glGetAttribLocation(shader_program_, "a_position");
    GLint tex_attr = glGetAttribLocation(shader_program_, "a_texcoord");
    GLint tex_uniform = glGetUniformLocation(shader_program_, "u_texture");
    
    glVertexAttribPointer(pos_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(pos_attr);
    
    glVertexAttribPointer(tex_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(tex_attr);
    
    glUniform1i(tex_uniform, 0);
    
    // 渲染四边形
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    // 清理
    glDisableVertexAttribArray(pos_attr);
    glDisableVertexAttribArray(tex_attr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // 交换缓冲区（这会自动处理DRM framebuffer更新）
    eglSwapBuffers(egl_display_, egl_surface_);
    
    // 检查OpenGL错误
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL渲染错误: 0x" << std::hex << error << std::endl;
    }
}

void LVGLWaylandInterface::Impl::cleanup() {
    // 首先清理OpenGL资源（必须在EGL上下文有效时执行）
    if (gl_resources_initialized_ && egl_initialized_) {
        cleanupGLResources();
    }
    
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
    
    // 清理Wayland EGL资源
    if (wl_egl_window_) {
        wl_egl_window_destroy(wl_egl_window_);
        wl_egl_window_ = nullptr;
    }
    
    // 清理Wayland资源
    if (wl_shell_surface_) {
        wl_shell_surface_destroy(wl_shell_surface_);
        wl_shell_surface_ = nullptr;
    }
    
    if (wl_surface_) {
        wl_surface_destroy(wl_surface_);
        wl_surface_ = nullptr;
    }
    
    if (wl_shell_) {
        wl_shell_destroy(wl_shell_);
        wl_shell_ = nullptr;
    }
    
    if (wl_compositor_) {
        wl_compositor_destroy(wl_compositor_);
        wl_compositor_ = nullptr;
    }
    
    if (wl_registry_) {
        wl_registry_destroy(wl_registry_);
        wl_registry_ = nullptr;
    }
    
    if (wl_display_) {
        wl_display_disconnect(wl_display_);
        wl_display_ = nullptr;
    }
    
    wayland_egl_initialized_ = false;
    
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

// OpenGL资源管理实现
bool LVGLWaylandInterface::Impl::initializeGLResources() {
    // 创建shader程序
    if (!createShaderProgram()) {
        return false;
    }
    
    // 创建纹理
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // 创建VBO（顶点缓冲对象）
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    
    // 全屏四边形顶点数据（位置 + 纹理坐标）
    GLfloat vertices[] = {
        // 位置      纹理坐标
        -1.0f, -1.0f,  0.0f, 1.0f,  // 左下
         1.0f, -1.0f,  1.0f, 1.0f,  // 右下
        -1.0f,  1.0f,  0.0f, 0.0f,  // 左上
         1.0f,  1.0f,  1.0f, 0.0f   // 右上
    };
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    return true;
}

void LVGLWaylandInterface::Impl::cleanupGLResources() {
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    
    if (texture_id_ != 0) {
        glDeleteTextures(1, &texture_id_);
        texture_id_ = 0;
    }
    
    if (shader_program_ != 0) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
    
    gl_resources_initialized_ = false;
}

bool LVGLWaylandInterface::Impl::createShaderProgram() {
    const char* vertex_shader_source = R"(
        attribute vec2 a_position;
        attribute vec2 a_texcoord;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texcoord = a_texcoord;
        }
    )";
    
    const char* fragment_shader_source = R"(
        precision mediump float;
        varying vec2 v_texcoord;
        uniform sampler2D u_texture;
        void main() {
            gl_FragColor = texture2D(u_texture, v_texcoord);
        }
    )";
    
    // 编译vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
        std::cerr << "Vertex shader编译失败: " << info_log << std::endl;
        glDeleteShader(vertex_shader);
        return false;
    }
    
    // 编译fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
        std::cerr << "Fragment shader编译失败: " << info_log << std::endl;
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }
    
    // 创建shader程序
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertex_shader);
    glAttachShader(shader_program_, fragment_shader);
    glLinkProgram(shader_program_);
    
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shader_program_, 512, NULL, info_log);
        std::cerr << "Shader程序链接失败: " << info_log << std::endl;
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
        return false;
    }
    
    // 清理shader对象
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    return true;
}

// 析构函数实现
LVGLWaylandInterface::Impl::~Impl() {
    cleanup();
}

} // namespace ui
} // namespace bamboo_cut