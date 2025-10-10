/**
 * @file lvgl_wayland_backend.cpp
 * @brief LVGL Wayland显示后端实现
 * 支持LVGL在Wayland环境下运行
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include "bamboo_cut/ui/display_backend.h"
#include "bamboo_cut/ui/wayland_compositor.h"
#include "bamboo_cut/ui/wayland_compat.h"
#include <iostream>
#include <cstring>
#include <memory>

// 前向声明
void wayland_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map);

namespace bamboo_cut {
namespace ui {
namespace lvgl_wayland_extension {
    bool initializeWaylandBackend(LVGLInterface* interface, const LVGLConfig& config);
    void cleanupWaylandBackend();
    bool isWaylandBackendAvailable();
}
}
}

namespace bamboo_cut {
namespace ui {

/**
 * @brief LVGL Wayland显示后端类
 * 实现LVGL在Wayland环境下的显示功能
 */
class LVGLWaylandBackend {
public:
    LVGLWaylandBackend();
    ~LVGLWaylandBackend();
    
    /**
     * @brief 初始化Wayland后端
     * @param width 屏幕宽度
     * @param height 屏幕高度
     * @return true如果初始化成功
     */
    bool initialize(int width, int height);
    
    /**
     * @brief 启动Wayland后端
     * @return true如果启动成功
     */
    bool start();
    
    /**
     * @brief 停止Wayland后端
     */
    void stop();
    
    /**
     * @brief 检查是否正在运行
     * @return true如果正在运行
     */
    bool isRunning() const { return running_; }
    
    /**
     * @brief 获取LVGL显示器指针
     * @return LVGL显示器指针
     */
    void* getLVGLDisplay() const { return lvgl_display_; }
    
    /**
     * @brief 刷新显示内容
     * @param area 刷新区域
     * @param color_map 像素数据
     */
    void flushDisplay(const void* area, const uint8_t* color_map);

private:
    /**
     * @brief 初始化Wayland连接
     * @return true如果连接成功
     */
    bool initializeWaylandConnection();
    
    /**
     * @brief 创建Wayland surface
     * @return true如果创建成功
     */
    bool createWaylandSurface();
    
    /**
     * @brief 初始化EGL上下文
     * @return true如果初始化成功
     */
    bool initializeEGL();
    
    /**
     * @brief 创建LVGL显示驱动
     * @return true如果创建成功
     */
    bool createLVGLDisplay();
    
    /**
     * @brief 清理资源
     */
    void cleanup();
    
    /**
     * @brief 处理Wayland事件
     */
    void handleWaylandEvents();

private:
    // 状态管理
    bool initialized_;
    bool running_;
    int width_, height_;
    
    // Wayland对象
#ifdef ENABLE_WAYLAND
    struct wl_display* wl_display_;
    struct wl_registry* wl_registry_;
    struct wl_compositor* wl_compositor_;
    struct wl_shell* wl_shell_;
    struct wl_surface* wl_surface_;
    struct wl_shell_surface* wl_shell_surface_;
    struct wl_egl_window* wl_egl_window_;
    
    // EGL对象
    EGLDisplay egl_display_;
    EGLConfig egl_config_;
    EGLContext egl_context_;
    EGLSurface egl_surface_;
#else
    void* wl_display_;
    void* wl_registry_;
    void* wl_compositor_;
    void* wl_shell_;
    void* wl_surface_;
    void* wl_shell_surface_;
    void* wl_egl_window_;
    void* egl_display_;
    void* egl_config_;
    void* egl_context_;
    void* egl_surface_;
#endif
    
    // LVGL对象
#ifdef ENABLE_LVGL
    lv_display_t* lvgl_display_;
    lv_color_t* disp_buf1_;
    lv_color_t* disp_buf2_;
    lv_draw_buf_t draw_buf_;
#else
    void* lvgl_display_;
    void* disp_buf1_;
    void* disp_buf2_;
    char draw_buf_[64];
#endif

    // 静态回调函数
#ifdef ENABLE_WAYLAND
    static void registry_handler(void* data, struct wl_registry* registry,
                                uint32_t id, const char* interface, uint32_t version);
    static void registry_remover(void* data, struct wl_registry* registry, uint32_t id);
    static void shell_surface_ping(void* data, struct wl_shell_surface* shell_surface, uint32_t serial);
    static void shell_surface_configure(void* data, struct wl_shell_surface* shell_surface,
                                       uint32_t edges, int32_t width, int32_t height);
    static void shell_surface_popup_done(void* data, struct wl_shell_surface* shell_surface);
    
    static const struct wl_registry_listener registry_listener_;
    static const struct wl_shell_surface_listener shell_surface_listener_;
#endif
};

// 静态监听器定义
#ifdef ENABLE_WAYLAND
const struct wl_registry_listener LVGLWaylandBackend::registry_listener_ = {
    LVGLWaylandBackend::registry_handler,
    LVGLWaylandBackend::registry_remover
};

const struct wl_shell_surface_listener LVGLWaylandBackend::shell_surface_listener_ = {
    LVGLWaylandBackend::shell_surface_ping,
    LVGLWaylandBackend::shell_surface_configure,
    LVGLWaylandBackend::shell_surface_popup_done
};
#endif

// 全局Wayland后端实例
static std::unique_ptr<LVGLWaylandBackend> g_wayland_backend = nullptr;

LVGLWaylandBackend::LVGLWaylandBackend()
    : initialized_(false)
    , running_(false)
    , width_(0)
    , height_(0)
{
#ifdef ENABLE_WAYLAND
    wl_display_ = nullptr;
    wl_registry_ = nullptr;
    wl_compositor_ = nullptr;
    wl_shell_ = nullptr;
    wl_surface_ = nullptr;
    wl_shell_surface_ = nullptr;
    wl_egl_window_ = nullptr;
    egl_display_ = EGL_NO_DISPLAY;
    egl_config_ = nullptr;
    egl_context_ = EGL_NO_CONTEXT;
    egl_surface_ = EGL_NO_SURFACE;
#else
    wl_display_ = nullptr;
    wl_registry_ = nullptr;
    wl_compositor_ = nullptr;
    wl_shell_ = nullptr;
    wl_surface_ = nullptr;
    wl_shell_surface_ = nullptr;
    wl_egl_window_ = nullptr;
    egl_display_ = nullptr;
    egl_config_ = nullptr;
    egl_context_ = nullptr;
    egl_surface_ = nullptr;
#endif

#ifdef ENABLE_LVGL
    lvgl_display_ = nullptr;
    disp_buf1_ = nullptr;
    disp_buf2_ = nullptr;
    memset(&draw_buf_, 0, sizeof(draw_buf_));
#else
    lvgl_display_ = nullptr;
    disp_buf1_ = nullptr;
    disp_buf2_ = nullptr;
    memset(draw_buf_, 0, sizeof(draw_buf_));
#endif
}

LVGLWaylandBackend::~LVGLWaylandBackend() {
    cleanup();
}

bool LVGLWaylandBackend::initialize(int width, int height) {
    std::cout << "[LVGLWaylandBackend] 初始化Wayland后端: " << width << "x" << height << std::endl;
    
    width_ = width;
    height_ = height;
    
    // 检查Wayland支持
    if (!WaylandDetector::detectWaylandSupport()) {
        std::cerr << "[LVGLWaylandBackend] Wayland环境不可用" << std::endl;
        return false;
    }
    
    // 初始化Wayland连接
    if (!initializeWaylandConnection()) {
        std::cerr << "[LVGLWaylandBackend] Wayland连接初始化失败" << std::endl;
        return false;
    }
    
    // 创建Wayland surface
    if (!createWaylandSurface()) {
        std::cerr << "[LVGLWaylandBackend] Wayland surface创建失败" << std::endl;
        return false;
    }
    
    // 初始化EGL
    if (!initializeEGL()) {
        std::cerr << "[LVGLWaylandBackend] EGL初始化失败" << std::endl;
        return false;
    }
    
    // 创建LVGL显示驱动
    if (!createLVGLDisplay()) {
        std::cerr << "[LVGLWaylandBackend] LVGL显示驱动创建失败" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[LVGLWaylandBackend] Wayland后端初始化成功" << std::endl;
    return true;
}

bool LVGLWaylandBackend::start() {
    if (!initialized_) {
        std::cerr << "[LVGLWaylandBackend] 后端未初始化" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLWaylandBackend] 启动Wayland后端" << std::endl;
    running_ = true;
    return true;
}

void LVGLWaylandBackend::stop() {
    if (!running_) return;
    
    std::cout << "[LVGLWaylandBackend] 停止Wayland后端" << std::endl;
    running_ = false;
}

bool LVGLWaylandBackend::initializeWaylandConnection() {
#ifdef ENABLE_WAYLAND
    std::cout << "[LVGLWaylandBackend] 连接Wayland服务器..." << std::endl;
    
    wl_display_ = wl_display_connect(nullptr);
    if (!wl_display_) {
        std::cerr << "[LVGLWaylandBackend] 无法连接到Wayland display" << std::endl;
        return false;
    }
    
    wl_registry_ = wl_display_get_registry(static_cast<struct wl_display*>(wl_display_));
    if (!wl_registry_) {
        std::cerr << "[LVGLWaylandBackend] 无法获取Wayland registry" << std::endl;
        return false;
    }
    
    wl_registry_add_listener(wl_registry_, &registry_listener_, this);
    wl_display_dispatch(wl_display_);
    wl_display_roundtrip(wl_display_);
    
    if (!wl_compositor_ || !wl_shell_) {
        std::cerr << "[LVGLWaylandBackend] 缺少必要的Wayland协议支持" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLWaylandBackend] Wayland连接建立成功" << std::endl;
    return true;
#else
    std::cerr << "[LVGLWaylandBackend] Wayland支持未编译" << std::endl;
    return false;
#endif
}

bool LVGLWaylandBackend::createWaylandSurface() {
#ifdef ENABLE_WAYLAND
    std::cout << "[LVGLWaylandBackend] 创建Wayland surface..." << std::endl;
    
    wl_surface_ = wl_compositor_create_surface(wl_compositor_);
    if (!wl_surface_) {
        std::cerr << "[LVGLWaylandBackend] 无法创建Wayland surface" << std::endl;
        return false;
    }
    
    wl_shell_surface_ = wl_shell_get_shell_surface(wl_shell_, wl_surface_);
    if (!wl_shell_surface_) {
        std::cerr << "[LVGLWaylandBackend] 无法创建shell surface" << std::endl;
        return false;
    }
    
    wl_shell_surface_add_listener(wl_shell_surface_, &shell_surface_listener_, this);
    wl_shell_surface_set_toplevel(wl_shell_surface_);
    wl_shell_surface_set_title(wl_shell_surface_, "Bamboo Cut System");
    
    wl_egl_window_ = wl_egl_window_create(static_cast<struct wl_surface*>(wl_surface_), width_, height_);
    if (!wl_egl_window_) {
        std::cerr << "[LVGLWaylandBackend] 无法创建EGL窗口" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLWaylandBackend] Wayland surface创建成功" << std::endl;
    return true;
#else
    std::cout << "[LVGLWaylandBackend] Wayland支持未编译，跳过surface创建" << std::endl;
    return false;
#endif
}

bool LVGLWaylandBackend::initializeEGL() {
#ifdef ENABLE_WAYLAND
    std::cout << "[LVGLWaylandBackend] 初始化EGL..." << std::endl;
    
    egl_display_ = eglGetDisplay((EGLNativeDisplayType)wl_display_);
    if (egl_display_ == EGL_NO_DISPLAY) {
        std::cerr << "[LVGLWaylandBackend] 无法获取EGL display" << std::endl;
        return false;
    }
    
    if (!eglInitialize(egl_display_, nullptr, nullptr)) {
        std::cerr << "[LVGLWaylandBackend] EGL初始化失败" << std::endl;
        return false;
    }
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLint num_configs;
    if (!eglChooseConfig(egl_display_, config_attribs, &egl_config_, 1, &num_configs)) {
        std::cerr << "[LVGLWaylandBackend] EGL配置选择失败" << std::endl;
        return false;
    }
    
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, context_attribs);
    if (egl_context_ == EGL_NO_CONTEXT) {
        std::cerr << "[LVGLWaylandBackend] EGL上下文创建失败" << std::endl;
        return false;
    }
    
    egl_surface_ = eglCreateWindowSurface(egl_display_, egl_config_, 
                                         (EGLNativeWindowType)wl_egl_window_, nullptr);
    if (egl_surface_ == EGL_NO_SURFACE) {
        std::cerr << "[LVGLWaylandBackend] EGL surface创建失败" << std::endl;
        return false;
    }
    
    if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
        std::cerr << "[LVGLWaylandBackend] EGL上下文激活失败" << std::endl;
        return false;
    }
    
    std::cout << "[LVGLWaylandBackend] EGL初始化成功" << std::endl;
    return true;
#else
    std::cout << "[LVGLWaylandBackend] EGL支持未编译，跳过EGL初始化" << std::endl;
    return false;
#endif
}

bool LVGLWaylandBackend::createLVGLDisplay() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLWaylandBackend] 创建LVGL显示驱动..." << std::endl;
    
    // 分配显示缓冲区
    uint32_t buf_size = width_ * height_;
    disp_buf1_ = new(std::nothrow) lv_color_t[buf_size];
    disp_buf2_ = new(std::nothrow) lv_color_t[buf_size];
    
    if (!disp_buf1_ || !disp_buf2_) {
        std::cerr << "[LVGLWaylandBackend] 显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    // 初始化显示缓冲区
    lv_draw_buf_init(&draw_buf_, width_, height_,
                     LV_COLOR_FORMAT_XRGB8888, width_ * 4,
                     disp_buf1_, buf_size * sizeof(lv_color_t));
    
    // 创建LVGL显示器
    lvgl_display_ = lv_display_create(width_, height_);
    if (!lvgl_display_) {
        std::cerr << "[LVGLWaylandBackend] LVGL显示器创建失败" << std::endl;
        return false;
    }
    
    // 设置显示缓冲区
    lv_display_set_buffers(lvgl_display_, disp_buf1_, disp_buf2_, 
                          buf_size * sizeof(lv_color_t), LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置刷新回调
    lv_display_set_flush_cb(lvgl_display_, wayland_flush_cb);
    
    std::cout << "[LVGLWaylandBackend] LVGL显示驱动创建成功" << std::endl;
    return true;
#else
    std::cout << "[LVGLWaylandBackend] LVGL支持未编译，跳过显示驱动创建" << std::endl;
    return false;
#endif
}

void LVGLWaylandBackend::flushDisplay(const void* area, const uint8_t* color_map) {
#ifdef ENABLE_WAYLAND
    if (!running_) return;
    
    // 处理Wayland事件
    handleWaylandEvents();
    
    // 使用OpenGL ES渲染像素数据
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 这里可以添加实际的像素数据渲染逻辑
    // 为了简化，暂时只是清空屏幕
    
    // 交换缓冲区
    eglSwapBuffers(egl_display_, egl_surface_);
    
    // 提交Wayland surface
    wl_surface_commit(wl_surface_);
#endif
}

void LVGLWaylandBackend::handleWaylandEvents() {
#ifdef ENABLE_WAYLAND
    if (wl_display_) {
        wl_display_dispatch_pending(wl_display_);
        wl_display_flush(wl_display_);
    }
#endif
}

void LVGLWaylandBackend::cleanup() {
    std::cout << "[LVGLWaylandBackend] 清理资源..." << std::endl;
    
#ifdef ENABLE_LVGL
    // 清理LVGL资源
    if (disp_buf1_) {
        delete[] disp_buf1_;
        disp_buf1_ = nullptr;
    }
    if (disp_buf2_) {
        delete[] disp_buf2_;
        disp_buf2_ = nullptr;
    }
    // LVGL显示器会被LVGL系统自动清理
#endif

#ifdef ENABLE_WAYLAND
    // 清理EGL资源
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_context_ != EGL_NO_CONTEXT) {
            eglDestroyContext(egl_display_, egl_context_);
        }
        if (egl_surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(egl_display_, egl_surface_);
        }
        eglTerminate(egl_display_);
    }
    
    // 清理Wayland资源
    if (wl_egl_window_) {
        wl_egl_window_destroy(wl_egl_window_);
    }
    if (wl_shell_surface_) {
        wl_shell_surface_destroy(wl_shell_surface_);
    }
    if (wl_surface_) {
        wl_surface_destroy(wl_surface_);
    }
    if (wl_shell_) {
        wl_shell_destroy(wl_shell_);
    }
    if (wl_compositor_) {
        wl_compositor_destroy(wl_compositor_);
    }
    if (wl_registry_) {
        wl_registry_destroy(wl_registry_);
    }
    if (wl_display_) {
        wl_display_disconnect(wl_display_);
    }
#endif

    initialized_ = false;
    running_ = false;
    std::cout << "[LVGLWaylandBackend] 资源清理完成" << std::endl;
}

// Wayland回调函数实现
#ifdef ENABLE_WAYLAND
void LVGLWaylandBackend::registry_handler(void* data, struct wl_registry* registry,
                                         uint32_t id, const char* interface, uint32_t version) {
    LVGLWaylandBackend* backend = static_cast<LVGLWaylandBackend*>(data);
    
    std::cout << "[LVGLWaylandBackend] Wayland接口: " << interface << " (版本 " << version << ")" << std::endl;
    
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        backend->wl_compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 1));
    } else if (strcmp(interface, wl_shell_interface.name) == 0) {
        backend->wl_shell_ = static_cast<struct wl_shell*>(
            wl_registry_bind(registry, id, &wl_shell_interface, 1));
    }
}

void LVGLWaylandBackend::registry_remover(void* data, struct wl_registry* registry, uint32_t id) {
    std::cout << "[LVGLWaylandBackend] Wayland接口移除: " << id << std::endl;
}

void LVGLWaylandBackend::shell_surface_ping(void* data, struct wl_shell_surface* shell_surface, uint32_t serial) {
    wl_shell_surface_pong(shell_surface, serial);
}

void LVGLWaylandBackend::shell_surface_configure(void* data, struct wl_shell_surface* shell_surface,
                                                 uint32_t edges, int32_t width, int32_t height) {
    LVGLWaylandBackend* backend = static_cast<LVGLWaylandBackend*>(data);
    
    if (width > 0 && height > 0) {
        std::cout << "[LVGLWaylandBackend] Surface配置更改: " << width << "x" << height << std::endl;
        if (backend->wl_egl_window_) {
            wl_egl_window_resize(backend->wl_egl_window_, width, height, 0, 0);
        }
    }
}

void LVGLWaylandBackend::shell_surface_popup_done(void* data, struct wl_shell_surface* shell_surface) {
    // 弹出窗口完成事件，暂时不处理
}
#endif

// LVGL Wayland刷新回调函数
void wayland_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
#ifdef ENABLE_LVGL
    if (g_wayland_backend && g_wayland_backend->isRunning()) {
        g_wayland_backend->flushDisplay(area, px_map);
    }
    
    lv_display_flush_ready(disp);
#endif
}

// 扩展LVGLInterface类以支持Wayland
namespace lvgl_wayland_extension {

bool initializeWaylandBackend(LVGLInterface* interface, const LVGLConfig& config) {
    std::cout << "[LVGLWaylandExtension] 初始化LVGL Wayland后端..." << std::endl;
    
    if (g_wayland_backend) {
        std::cout << "[LVGLWaylandExtension] Wayland后端已存在，先清理..." << std::endl;
        g_wayland_backend.reset();
    }
    
    g_wayland_backend = std::make_unique<LVGLWaylandBackend>();
    
    if (!g_wayland_backend->initialize(config.screen_width, config.screen_height)) {
        std::cerr << "[LVGLWaylandExtension] Wayland后端初始化失败" << std::endl;
        g_wayland_backend.reset();
        return false;
    }
    
    if (!g_wayland_backend->start()) {
        std::cerr << "[LVGLWaylandExtension] Wayland后端启动失败" << std::endl;
        g_wayland_backend.reset();
        return false;
    }
    
    std::cout << "[LVGLWaylandExtension] LVGL Wayland后端初始化成功" << std::endl;
    return true;
}

void cleanupWaylandBackend() {
    std::cout << "[LVGLWaylandExtension] 清理Wayland后端..." << std::endl;
    if (g_wayland_backend) {
        g_wayland_backend->stop();
        g_wayland_backend.reset();
    }
}

bool isWaylandBackendAvailable() {
    return WaylandDetector::detectWaylandSupport();
}

} // namespace lvgl_wayland_extension

} // namespace ui
} // namespace bamboo_cut