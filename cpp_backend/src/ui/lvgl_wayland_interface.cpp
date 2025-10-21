/**
 * @file lvgl_wayland_interface.cpp
 * @brief LVGL Wayland接口实现 - Weston合成器架构支持
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/utils/logger.h"
#include <lvgl.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <poll.h>   

// 系统头文件
#include <errno.h>

// 🔧 修复：注释EGL头文件，只保留Wayland SHM
// #include <EGL/egl.h>
// #include <EGL/eglext.h>
// #include <GLES2/gl2.h>
// #include <GLES2/gl2ext.h>
#include <wayland-client.h>
// #include <wayland-egl.h>
#include <vector>

#include <wayland-client-protocol.h>
#include "wayland-protocols/xdg-shell-client-protocol.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <string>

// 🔧 修复：禁用EGL，完全使用SHM避免与DeepStream冲突
// #define HAS_DRM_EGL_BACKEND 1
#define HAS_DRM_EGL_BACKEND 0

// 🆕 辅助函数：创建匿名共享内存文件（在Impl类外部定义）
static int createAnonymousFile(size_t size) {
    static const char template_str[] = "/bamboo-cut-XXXXXX";
    const char* path = getenv("XDG_RUNTIME_DIR");
    if (!path) {
        path = "/tmp";
    }
    
    std::string name = std::string(path) + template_str;
    int fd = mkstemp(&name[0]);
    if (fd < 0) {
        return -1;
    }
    
    unlink(name.c_str());
    
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }
    
    return fd;
}

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
    
    // Wayland EGL后端 - 现代xdg-shell协议实现 + Subsurface支持
    struct wl_display* wl_display_ = nullptr;
    struct wl_registry* wl_registry_ = nullptr;
    struct wl_compositor* wl_compositor_ = nullptr;
    struct wl_subcompositor* wl_subcompositor_ = nullptr;  // 🆕 新增：subcompositor支持
    struct wl_shm* wl_shm_ = nullptr;  // 🆕 新增：共享内存接口
    struct xdg_wm_base* xdg_wm_base_ = nullptr;
    struct wl_surface* wl_surface_ = nullptr;
    struct xdg_surface* xdg_surface_ = nullptr;
    struct xdg_toplevel* xdg_toplevel_ = nullptr;
    // 🔧 修复：注释EGL相关成员，避免与DeepStream冲突
    // struct wl_egl_window* wl_egl_window_ = nullptr;
    struct wl_callback* frame_callback_ = nullptr;

    // EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    // EGLContext egl_context_ = EGL_NO_CONTEXT;
    // EGLSurface egl_surface_ = EGL_NO_SURFACE;
    // EGLConfig egl_config_;
    
    // 显示缓冲区
    lv_color_t* front_buffer_ = nullptr;
    lv_color_t* back_buffer_ = nullptr;
    uint32_t buffer_size_ = 0;
    
    // 🔧 修复：注释OpenGL资源，完全使用SHM
    // GLuint shader_program_ = 0;
    // GLuint texture_id_ = 0;
    // GLuint vbo_ = 0;
    // bool gl_resources_initialized_ = false;
    
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
    // 🔧 修复：移除EGL状态标志
    // bool wayland_egl_initialized_ = false;
    // bool egl_initialized_ = false;
    
    Impl() = default;
    ~Impl();
    
    bool checkWaylandEnvironment();
    bool initializeWaylandClient();
    // 🔧 修复：注释EGL方法，只保留SHM相关方法
    // bool initializeWaylandEGL();
    bool initializeWaylandDisplay();
    bool initializeFallbackDisplay();
    bool initializeFallbackDisplayWithWaylandObjects();
    bool initializeInput();
    void initializeTheme();
    void createMainInterface();
    void updateCanvasFromFrame();
    // void flushDisplay(const lv_area_t* area, lv_color_t* color_p);
    void cleanup();
    void flushDisplayViaSHM(const lv_area_t* area, lv_color_t* color_p);
    // Wayland辅助函数 - 现代xdg-shell协议实现
    static void registryHandler(void* data, struct wl_registry* registry, uint32_t id, const char* interface, uint32_t version);
    static void registryRemover(void* data, struct wl_registry* registry, uint32_t id);
    static void xdgWmBasePing(void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial);
    static void xdgSurfaceConfigure(void* data, struct xdg_surface* xdg_surface, uint32_t serial);
    static void xdgToplevelConfigure(void* data, struct xdg_toplevel* xdg_toplevel, int32_t width, int32_t height, struct wl_array* states);
    static void xdgToplevelClose(void* data, struct xdg_toplevel* xdg_toplevel);
    static void frameCallback(void* data, struct wl_callback* callback, uint32_t time);
    // 🔧 修复：注释EGL和OpenGL方法
    // EGLConfig chooseEGLConfig();
    void handleWaylandEvents();
    void requestFrame();
    
    // 🔧 修复：注释OpenGL资源管理方法
    // bool initializeGLResources();
    // void cleanupGLResources();
    // bool createShaderProgram();

     // 🆕 新增：configure事件同步
    std::mutex configure_mutex_;
    std::condition_variable configure_cv_;
    std::atomic<bool> configure_received_{false};

    // 🆕 为 DeepStream 创建 Subsurface
    struct SubsurfaceHandle {
        void* surface;      // wl_surface*
        void* subsurface;   // wl_subsurface*
    };
    
    SubsurfaceHandle createSubsurface(int x, int y, int width, int height);
    void destroySubsurface(SubsurfaceHandle handle);
    
    // 获取 Wayland 对象（供 DeepStream 使用）
    void* getWaylandDisplay();
    void* getWaylandCompositor();
    void* getWaylandSubcompositor();
    void* getWaylandSurface();
};

LVGLWaylandInterface::SubsurfaceHandle 
LVGLWaylandInterface::createSubsurface(int x, int y, int width, int height) {
    SubsurfaceHandle handle = {nullptr, nullptr};
    
    if (!pImpl_->wl_compositor_ || !pImpl_->wl_subcompositor_ || !pImpl_->wl_surface_) {
        std::cerr << "❌ Wayland 对象未初始化" << std::endl;
        return handle;
    }
    
    std::cout << "🎬 为 DeepStream 创建 Subsurface..." << std::endl;
    
    // 创建 DeepStream 的独立 surface
    auto* wl_surface = wl_compositor_create_surface(pImpl_->wl_compositor_);
    if (!wl_surface) {
        std::cerr << "❌ 无法创建 DeepStream surface" << std::endl;
        return handle;
    }
    handle.surface = wl_surface;
    
    // 将其设置为 LVGL 主 surface 的 subsurface
    auto* wl_subsurface = wl_subcompositor_get_subsurface(
        pImpl_->wl_subcompositor_,
        wl_surface,
        pImpl_->wl_surface_  // 父 surface
    );
    
    if (!wl_subsurface) {
        std::cerr << "❌ 无法创建 subsurface" << std::endl;
        wl_surface_destroy(wl_surface);
        handle.surface = nullptr;
        return handle;
    }
    handle.subsurface = wl_subsurface;
    
    // 设置 subsurface 位置
    wl_subsurface_set_position(wl_subsurface, x, y);
    
    // 设置为同步模式
    wl_subsurface_set_sync(wl_subsurface);
    
    // 提交更改
    wl_surface_commit(wl_surface);
    wl_display_flush(pImpl_->wl_display_);
    
    std::cout << "✅ DeepStream Subsurface 创建成功，位置: (" 
              << x << ", " << y << ")" << std::endl;
    
    return handle;
}

void LVGLWaylandInterface::destroySubsurface(SubsurfaceHandle handle) {
    if (handle.subsurface) {
        wl_subsurface_destroy(static_cast<struct wl_subsurface*>(handle.subsurface));
    }
    if (handle.surface) {
        wl_surface_destroy(static_cast<struct wl_surface*>(handle.surface));
    }
}

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
    
    // 🆕 Jetson Orin NX特定：等待Weston完全就绪
    std::cout << "🔧 [Jetson] 等待Weston合成器完全初始化..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
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
    
    std::cout << "🚀 LVGL UI线程启动 (刷新率: " << pImpl_->config_.refresh_rate << "fps)" << std::endl;
    
    int loop_count = 0;
    while (!pImpl_->should_stop_.load()) {
        auto now = std::chrono::steady_clock::now();
        loop_count++;
        
        
        // ✅ 关键修复：处理Wayland事件循环
        pImpl_->handleWaylandEvents();
        
        // 处理LVGL任务
        {
            std::lock_guard<std::mutex> lock(pImpl_->ui_mutex_);
            lv_timer_handler();
        }
        
        // 更新Canvas（如果有新帧）
        if (pImpl_->new_frame_available_.load()) {
            if (loop_count <= 5) {
                std::cout << "🖼️ 更新Canvas帧" << std::endl;
            }
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
    
    std::cout << "🛑 LVGL UI线程停止" << std::endl;
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
    std::cout << "🔍 检查Wayland环境..." << std::endl;
    
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        std::cerr << "❌ WAYLAND_DISPLAY环境变量未设置" << std::endl;
        return false;
    }
    std::cout << "✅ WAYLAND_DISPLAY = " << wayland_display << std::endl;
    
    // 检查wayland socket是否存在
    std::string socket_path = "/run/user/" + std::to_string(getuid()) + "/" + wayland_display;
    if (access(socket_path.c_str(), F_OK) != 0) {
        std::cerr << "❌ Wayland socket不存在: " << socket_path << std::endl;
        return false;
    }
    std::cout << "✅ Wayland socket存在: " << socket_path << std::endl;
    
    wayland_initialized_ = true;
    return true;
}

bool LVGLWaylandInterface::Impl::initializeWaylandDisplay() {
    std::cout << "正在初始化Wayland客户端..." << std::endl;
    
    // 初始化 Wayland 客户端（使用新的 ID 预留策略）
    if (!initializeWaylandClient()) {
        std::cerr << "❌ Wayland客户端初始化失败" << std::endl;
        return false;
    }
    
    // 🔧 关键：完全使用 SHM 软件渲染，避免与 DeepStream 的 EGL 冲突
    std::cout << "📺 LVGL 使用 SHM 软件渲染（避免与 DeepStream 的 EGL 冲突）..." << std::endl;
    std::cout << "🎯 DeepStream 将独占 EGL/DRM 硬件加速" << std::endl;
    
    // 创建 LVGL 显示设备
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        std::cerr << "LVGL显示创建失败" << std::endl;
        return false;
    }
    
    // 分配 SHM 缓冲区
    buffer_size_ = config_.screen_width * config_.screen_height * sizeof(lv_color_t);
    front_buffer_ = (lv_color_t*)malloc(buffer_size_);
    back_buffer_ = (lv_color_t*)malloc(buffer_size_);
    
    if (!front_buffer_ || !back_buffer_) {
        std::cerr << "显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    lv_display_set_buffers(display_, front_buffer_, back_buffer_,
                          buffer_size_, LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 设置 flush 回调（使用 SHM）
    lv_display_set_flush_cb(display_, [](lv_display_t* disp, const lv_area_t* area, uint8_t* color_p) {
        LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(
            lv_display_get_user_data(disp));
        
        if (impl) {
            impl->flushDisplayViaSHM(area, (lv_color_t*)color_p);
        }
        
        lv_display_flush_ready(disp);
    });
    
    lv_display_set_user_data(display_, this);
    
    display_initialized_ = true;
    std::cout << "✅ LVGL Wayland SHM 显示初始化成功（纯软件渲染）" << std::endl;
    std::cout << "🚫 已跳过 EGL 初始化，避免与 DeepStream 冲突" << std::endl;
    std::cout << "🎬 DeepStream 可以独占 EGL/DRM 硬件加速资源" << std::endl;
    return true;
}

// 新增：通过 SHM 刷新显示的方法
void LVGLWaylandInterface::Impl::flushDisplayViaSHM(const lv_area_t* area, lv_color_t* color_p) {
    if (!wl_surface_ || !wl_shm_) return;
    
    // 创建 SHM buffer
    int width = area->x2 - area->x1 + 1;
    int height = area->y2 - area->y1 + 1;
    int stride = width * 4;
    int size = stride * height;
    
    int fd = createAnonymousFile(size);
    if (fd < 0) return;
    
    void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return;
    }
    
    // 复制 LVGL 像素数据到 SHM
    uint32_t* pixels = (uint32_t*)data;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            lv_color_t c = color_p[y * width + x];
            pixels[y * width + x] = (0xFF << 24) | (c.red << 16) | (c.green << 8) | c.blue;
        }
    }
    
    munmap(data, size);
    
    // 创建 Wayland buffer
    struct wl_shm_pool* pool = wl_shm_create_pool(wl_shm_, fd, size);
    struct wl_buffer* buffer = wl_shm_pool_create_buffer(pool, 0, width, height, stride, WL_SHM_FORMAT_ARGB8888);
    wl_shm_pool_destroy(pool);
    close(fd);
    
    // 提交到 Wayland
    wl_surface_attach(wl_surface_, buffer, area->x1, area->y1);
    wl_surface_damage(wl_surface_, area->x1, area->y1, width, height);
    wl_surface_commit(wl_surface_);
    wl_display_flush(wl_display_);
    
    wl_buffer_destroy(buffer);
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

// 🆕 新增：保留Wayland对象的fallback模式
bool LVGLWaylandInterface::Impl::initializeFallbackDisplayWithWaylandObjects() {
   // 🔧 修复：不再保留损坏的对象
    std::cout << "🔄 使用fallback显示模式" << std::endl;
    
    // 检查Wayland对象是否健康
    bool wayland_healthy = false;
    if (wl_display_) {
        int error_code = wl_display_get_error(wl_display_);
        wayland_healthy = (error_code == 0);
    }
    
    if (wayland_healthy) {
        std::cout << "✅ Wayland连接健康，保留对象供DeepStream使用" << std::endl;
        wayland_initialized_ = true;
    } else {
        std::cout << "❌ Wayland连接已损坏，清理对象" << std::endl;
        
        // 清理损坏的对象
        if (xdg_toplevel_) { xdg_toplevel_destroy(xdg_toplevel_); xdg_toplevel_ = nullptr; }
        if (xdg_surface_) { xdg_surface_destroy(xdg_surface_); xdg_surface_ = nullptr; }
        if (wl_surface_) { wl_surface_destroy(wl_surface_); wl_surface_ = nullptr; }
        if (xdg_wm_base_) { xdg_wm_base_destroy(xdg_wm_base_); xdg_wm_base_ = nullptr; }
        if (wl_compositor_) { wl_compositor_destroy(wl_compositor_); wl_compositor_ = nullptr; }
        if (wl_registry_) { wl_registry_destroy(wl_registry_); wl_registry_ = nullptr; }
        if (wl_display_) { wl_display_disconnect(wl_display_); wl_display_ = nullptr; }
        
        wayland_initialized_ = false;
    }
    
    // 创建fallback显示
    return initializeFallbackDisplay();
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


bool LVGLWaylandInterface::Impl::initializeWaylandClient() {
    std::cout << "正在初始化Wayland客户端..." << std::endl;
    
    // 连接 display
    wl_display_ = wl_display_connect(nullptr);
    if (!wl_display_) {
        std::cerr << "❌ 无法连接Wayland display" << std::endl;
        return false;
    }
    std::cout << "✅ Wayland display连接成功" << std::endl;
    
    // 获取 registry 并绑定接口
    wl_registry_ = wl_display_get_registry(wl_display_);
    if (!wl_registry_) {
        std::cerr << "❌ 无法获取registry" << std::endl;
        return false;
    }
    
    static const struct wl_registry_listener registry_listener = {
        registryHandler,
        registryRemover
    };
    wl_registry_add_listener(wl_registry_, &registry_listener, this);
    wl_display_roundtrip(wl_display_);
    std::cout << "✅ Registry同步完成" << std::endl;
    
    // 验证必需接口
    if (!wl_compositor_ || !xdg_wm_base_ || !wl_shm_) {
        std::cerr << "❌ 缺少必需的Wayland接口" << std::endl;
        return false;
    }
    
    // 设置 xdg_wm_base 监听器
    static const struct xdg_wm_base_listener xdg_wm_base_listener = {
        xdgWmBasePing
    };
    xdg_wm_base_add_listener(xdg_wm_base_, &xdg_wm_base_listener, this);
    
    // 创建 surface
    std::cout << "📐 创建主 Surface..." << std::endl;
    wl_surface_ = wl_compositor_create_surface(wl_compositor_);
    if (!wl_surface_) {
        std::cerr << "❌ 无法创建surface" << std::endl;
        return false;
    }
    std::cout << "✅ 主 Surface 创建成功" << std::endl;
    
    // 创建 xdg_surface（toplevel 窗口不需要 positioner）
    std::cout << "🎯 创建 XDG Surface..." << std::endl;
    xdg_surface_ = xdg_wm_base_get_xdg_surface(xdg_wm_base_, wl_surface_);
    if (!xdg_surface_) {
        std::cerr << "❌ 无法创建xdg_surface" << std::endl;
        return false;
    }
    
    static const struct xdg_surface_listener xdg_surface_listener = {
        xdgSurfaceConfigure
    };
    xdg_surface_add_listener(xdg_surface_, &xdg_surface_listener, this);
    std::cout << "✅ XDG Surface 创建成功" << std::endl;
    
    // 创建 toplevel（顶层窗口，不使用 positioner）
    std::cout << "🎯 创建 XDG Toplevel..." << std::endl;
    xdg_toplevel_ = xdg_surface_get_toplevel(xdg_surface_);
    if (!xdg_toplevel_) {
        std::cerr << "❌ 无法创建xdg_toplevel" << std::endl;
        return false;
    }
    
    // 设置窗口标题和应用ID
    xdg_toplevel_set_title(xdg_toplevel_, "Bamboo Recognition System");
    xdg_toplevel_set_app_id(xdg_toplevel_, "bamboo-cut-lvgl");
    
    // 🆕 设置全屏显示
    std::cout << "🖥️  设置全屏显示..." << std::endl;
    xdg_toplevel_set_fullscreen(xdg_toplevel_, nullptr);
    
    static const struct xdg_toplevel_listener xdg_toplevel_listener = {
        xdgToplevelConfigure,
        xdgToplevelClose
    };
    xdg_toplevel_add_listener(xdg_toplevel_, &xdg_toplevel_listener, this);
    std::cout << "✅ XDG Toplevel 创建成功，已设置全屏" << std::endl;
    
    // ... 后面的代码保持不变 ...
    
    std::cout << "🎨 创建初始 SHM buffer..." << std::endl;
    
    // 创建一个全屏尺寸的 buffer（而不是1x1）
    int stride = config_.screen_width * 4;
    int size = stride * config_.screen_height;
    
    int fd = createAnonymousFile(size);
    if (fd < 0) {
        std::cerr << "❌ 无法创建共享内存文件" << std::endl;
        return false;
    }
    
    void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        std::cerr << "❌ mmap 失败" << std::endl;
        return false;
    }
    
    // 填充黑色
    memset(data, 0, size);
    
    // 创建 wl_shm_pool
    struct wl_shm_pool* pool = wl_shm_create_pool(wl_shm_, fd, size);
    
    // 创建全屏 buffer
    struct wl_buffer* buffer = wl_shm_pool_create_buffer(
        pool, 0, config_.screen_width, config_.screen_height, stride, WL_SHM_FORMAT_ARGB8888);
    
    wl_shm_pool_destroy(pool);
    munmap(data, size);
    close(fd);
    
    // ✅ 关键：在 commit 前附加全屏 buffer
    wl_surface_attach(wl_surface_, buffer, 0, 0);
    wl_surface_damage(wl_surface_, 0, 0, config_.screen_width, config_.screen_height);
    
    std::cout << "✅ 全屏 buffer 已附加: " << config_.screen_width << "x" << config_.screen_height << std::endl;
    
    // 提交 surface 并触发 configure 事件
    std::cout << "📝 提交 surface，触发 configure..." << std::endl;
    wl_surface_commit(wl_surface_);
    wl_display_flush(wl_display_);
    
    // 等待 configure 事件
    std::cout << "⏳ 等待 configure 事件..." << std::endl;
    configure_received_.store(false);
    
    int max_attempts = 50;
    int attempts = 0;
    
    while (!configure_received_.load() && attempts < max_attempts) {
        if (wl_display_dispatch(wl_display_) < 0) {
            int error = wl_display_get_error(wl_display_);
            std::cerr << "❌ Wayland dispatch 失败，错误码: " << error << std::endl;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        attempts++;
    }
    
    if (!configure_received_.load()) {
        std::cerr << "❌ 等待 configure 超时" << std::endl;
        return false;
    }
    
    std::cout << "✅ Wayland 客户端初始化完成" << std::endl;
    return true;
}


// 🔧 更新：xdg_toplevel configure回调
void LVGLWaylandInterface::Impl::xdgToplevelConfigure(
    void* data, 
    struct xdg_toplevel* xdg_toplevel,
    int32_t width, 
    int32_t height, 
    struct wl_array* states) {
    
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    std::cout << "📐 XDG toplevel configure: " << width << "x" << height << std::endl;
    
    // 如果合成器建议新尺寸，更新配置
    if (width > 0 && height > 0) {
        impl->config_.screen_width = width;
        impl->config_.screen_height = height;
        std::cout << "📏 更新窗口尺寸为: " << width << "x" << height << std::endl;
    }
    
    // 打印窗口状态
    if (states && states->size > 0) {
        uint32_t* state_data = static_cast<uint32_t*>(states->data);
        size_t num_states = states->size / sizeof(uint32_t);
        
        for (size_t i = 0; i < num_states; i++) {
            switch (state_data[i]) {
                case XDG_TOPLEVEL_STATE_MAXIMIZED:
                    std::cout << "🔲 窗口状态: 最大化" << std::endl;
                    break;
                case XDG_TOPLEVEL_STATE_FULLSCREEN:
                    std::cout << "🔳 窗口状态: 全屏" << std::endl;
                    break;
                case XDG_TOPLEVEL_STATE_ACTIVATED:
                    std::cout << "✨ 窗口状态: 激活" << std::endl;
                    break;
            }
        }
    }
}

// 🔧 修复：注释整个EGL初始化方法，避免与DeepStream冲突
/*
bool LVGLWaylandInterface::Impl::initializeWaylandEGL() {
    std::cout << "🎨 初始化Wayland EGL..." << std::endl;
    // ... EGL 初始化代码已注释，避免与 DeepStream 冲突 ...
    return true;
}
*/

// Wayland registry回调函数 - 支持subcompositor绑定
void LVGLWaylandInterface::Impl::registryHandler(void* data, struct wl_registry* registry,
                                                  uint32_t id, const char* interface, uint32_t version) {
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    
    std::cout << "🔍 发现Wayland接口: " << interface << " (id=" << id << ", version=" << version << ")" << std::endl;
    
    if (strcmp(interface, "wl_compositor") == 0) {
        impl->wl_compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 1));
        std::cout << "✅ 绑定wl_compositor成功" << std::endl;
    }
    else if (strcmp(interface, "wl_subcompositor") == 0) {
        // 🆕 关键：绑定subcompositor接口，用于创建subsurface
        impl->wl_subcompositor_ = static_cast<struct wl_subcompositor*>(
            wl_registry_bind(registry, id, &wl_subcompositor_interface, 1));
        std::cout << "✅ 绑定wl_subcompositor成功（支持Subsurface架构）" << std::endl;
    }
    else if (strcmp(interface, "wl_shm") == 0) {
        // 🆕 新增：绑定共享内存接口，用于创建buffer
        impl->wl_shm_ = static_cast<struct wl_shm*>(
            wl_registry_bind(registry, id, &wl_shm_interface, 1));
        std::cout << "✅ 绑定wl_shm成功" << std::endl;
    }
    else if (strcmp(interface, "xdg_wm_base") == 0) {
    // 🔧 关键修复：使用 version 2 或更高，但不超过服务器支持的版本
    uint32_t bind_version = std::min(version, 2u);  // 使用 v2，兼容性更好
    impl->xdg_wm_base_ = static_cast<struct xdg_wm_base*>(
        wl_registry_bind(registry, id, &xdg_wm_base_interface, bind_version));
    std::cout << "✅ 绑定xdg_wm_base成功 (version " << bind_version << ")" << std::endl;
    }
}

void LVGLWaylandInterface::Impl::registryRemover(void* data, struct wl_registry* registry, uint32_t id) {
    // 处理全局对象移除（可选实现）
}

// ✅ 新增：xdg-shell协议回调函数实现
void LVGLWaylandInterface::Impl::xdgWmBasePing(void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial) {
    std::cout << "🏓 收到xdg_wm_base ping, serial=" << serial << std::endl;
    xdg_wm_base_pong(xdg_wm_base, serial);
    std::cout << "✅ 已回复xdg_wm_base pong" << std::endl;
}

void LVGLWaylandInterface::Impl::xdgSurfaceConfigure(void* data, struct xdg_surface* xdg_surface, uint32_t serial) {
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    std::cout << "📐 收到XDG surface配置, serial=" << serial << std::endl;
    
    // 步骤6: 确认configure（协议要求必须ack）
    xdg_surface_ack_configure(xdg_surface, serial);
    std::cout << "✅ 已确认xdg surface配置" << std::endl;
    
    // 设置标志，通知主线程configure已到达
    impl->configure_received_.store(true);
    impl->configure_cv_.notify_one();
    
    // ⚠️ 注意：不要在这里commit，让主线程在ack后commit
}


void LVGLWaylandInterface::Impl::xdgToplevelClose(void* data, struct xdg_toplevel* xdg_toplevel) {
    std::cout << "❌ XDG toplevel关闭请求" << std::endl;
    // 这里可以处理关闭窗口的逻辑
}

// ✅ 新增：frame回调函数 - 同步渲染
void LVGLWaylandInterface::Impl::frameCallback(void* data, struct wl_callback* callback, uint32_t time) {
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    
    static uint32_t last_time = 0;
    if (last_time > 0) {
        uint32_t delta = time - last_time;
        if (delta > 0) {
            float fps = 1000.0f / delta;
            static int frame_count = 0;
            frame_count++;
            if (frame_count % 60 == 0) { // 每60帧打印一次
                std::cout << "🎬 Wayland帧回调: " << fps << " fps (时间=" << time << "ms)" << std::endl;
            }
        }
    }
    last_time = time;
    
    // 销毁当前回调
    if (callback) {
        wl_callback_destroy(callback);
    }
    impl->frame_callback_ = nullptr;
    
    // 🔧 关键：请求下一帧回调
    impl->requestFrame();
}

// ✅ 新增：请求frame回调函数
void LVGLWaylandInterface::Impl::requestFrame() {
    if (!wl_surface_) {
        return;
    }
    
    // 如果已有回调，先销毁
    if (frame_callback_) {
        wl_callback_destroy(frame_callback_);
    }
    
    // 请求新的frame回调
    frame_callback_ = wl_surface_frame(wl_surface_);
    if (frame_callback_) {
        static const struct wl_callback_listener frame_listener = {
            frameCallback
        };
        wl_callback_add_listener(frame_callback_, &frame_listener, this);
    }
}

// ✅ 新增：Wayland事件处理函数
void LVGLWaylandInterface::Impl::handleWaylandEvents() {
    static int event_count = 0;
    
    if (!wl_display_) {
        return;
    }
    
    // 🔍 详细的事件处理日志
    event_count++;
    if (event_count <= 10 || event_count % 120 == 0) { // 前10次和每2秒（60fps）
        std::cout << "🔄 处理Wayland事件 #" << event_count << std::endl;
    }
    
    // 处理所有待处理的事件，但不阻塞
    int pending_events = 0;
    while (wl_display_prepare_read(wl_display_) != 0) {
        wl_display_dispatch_pending(wl_display_);
        pending_events++;
    }
    
    if (pending_events > 0 && event_count <= 10) {
        std::cout << "📨 处理了 " << pending_events << " 个待处理事件" << std::endl;
    }
    
    // 检查是否有数据可读
    if (wl_display_flush(wl_display_) < 0) {
        if (event_count <= 10) {
            std::cerr << "⚠️  Wayland display flush失败" << std::endl;
        }
    }
    
    // 读取并分发事件（非阻塞）
    if (wl_display_read_events(wl_display_) >= 0) {
        int dispatched = wl_display_dispatch_pending(wl_display_);
        if (dispatched > 0 && event_count <= 10) {
            std::cout << "✅ 分发了 " << dispatched << " 个新事件" << std::endl;
        }
    } else {
        wl_display_cancel_read(wl_display_);
        if (event_count <= 10) {
            std::cout << "❌ Wayland事件读取取消" << std::endl;
        }
    }
}

// 🔧 修复：注释EGL配置选择方法
/*
EGLConfig LVGLWaylandInterface::Impl::chooseEGLConfig() {
    // ... EGL 配置选择代码已注释，避免与 DeepStream 冲突 ...
    return nullptr;
}
*/

// 🔧 修复：注释EGL flush方法，LVGL现在完全使用SHM
/*
void LVGLWaylandInterface::Impl::flushDisplay(const lv_area_t* area, lv_color_t* color_p) {
    // ... EGL 渲染代码已注释，LVGL 现在使用 SHM 软件渲染 ...
}
*/

void LVGLWaylandInterface::Impl::cleanup() {
    // 🔧 修复：完全使用SHM，只清理Wayland资源
    
    // 清理Wayland资源 - xdg-shell实现
    if (frame_callback_) {
        wl_callback_destroy(frame_callback_);
        frame_callback_ = nullptr;
    }
    
    if (xdg_toplevel_) {
        xdg_toplevel_destroy(xdg_toplevel_);
        xdg_toplevel_ = nullptr;
    }
    
    if (xdg_surface_) {
        xdg_surface_destroy(xdg_surface_);
        xdg_surface_ = nullptr;
    }
    
    if (wl_surface_) {
        wl_surface_destroy(wl_surface_);
        wl_surface_ = nullptr;
    }
    
    if (xdg_wm_base_) {
        xdg_wm_base_destroy(xdg_wm_base_);
        xdg_wm_base_ = nullptr;
    }
    
    if (wl_subcompositor_) {
        wl_subcompositor_destroy(wl_subcompositor_);
        wl_subcompositor_ = nullptr;
    }
    
    if (wl_shm_) {
        wl_shm_destroy(wl_shm_);
        wl_shm_ = nullptr;
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

// 🔧 修复：删除所有OpenGL和EGL方法实现，完全使用SHM

// 析构函数实现
LVGLWaylandInterface::Impl::~Impl() {
    cleanup();
}

} // namespace ui
} // namespace bamboo_cut

namespace bamboo_cut {
namespace ui {

// 🆕 实现获取内部实现指针的方法
void* LVGLWaylandInterface::getImpl() {
    return pImpl_.get();
}

// 🆕 实现获取Wayland对象的方法，用于DeepStream Subsurface创建
void* LVGLWaylandInterface::getWaylandDisplay() {
    return pImpl_ ? pImpl_->wl_display_ : nullptr;
}

void* LVGLWaylandInterface::getWaylandCompositor() {
    return pImpl_ ? pImpl_->wl_compositor_ : nullptr;
}

void* LVGLWaylandInterface::getWaylandSubcompositor() {
    return pImpl_ ? pImpl_->wl_subcompositor_ : nullptr;
}

void* LVGLWaylandInterface::getWaylandSurface() {
    return pImpl_ ? pImpl_->wl_surface_ : nullptr;
}

} // namespace ui
} // namespace bamboo_cut