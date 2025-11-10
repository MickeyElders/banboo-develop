/**
 * @file lvgl_wayland_interface.cpp
 * @brief LVGL Wayland接口实现 - 标准 Wayland 协议
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/utils/logger.h"
#include <lvgl.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <thread>
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
// 🔧 修复：使用系统提供的 xdg-shell 协议头文件，而非自定义生成的
// 系统协议库与 Weston 版本完全匹配，避免协议解析错误
#include <xdg-shell-client-protocol.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <string>

// 🔧 修复：使用条件编译避免重定义警告
// HAS_DRM_EGL_BACKEND 在 CMakeLists.txt 中定义
#ifndef HAS_DRM_EGL_BACKEND
#define HAS_DRM_EGL_BACKEND 0
#endif

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
    lv_color_t* front_buffer_ = nullptr;  // LVGL 前缓冲（PARTIAL 模式）
    lv_color_t* back_buffer_ = nullptr;   // LVGL 后缓冲（PARTIAL 模式）
    uint32_t buffer_size_ = 0;
    
    // 🔧 完整帧累积 buffer（用于 PARTIAL 模式累积更新）
    uint32_t* full_frame_buffer_ = nullptr;  // 完整屏幕帧（ARGB8888）
    
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
    
    // 🔧 修复：使用异步模式让视频独立渲染，不受父surface影响
    wl_subsurface_set_desync(wl_subsurface);
    
    // 🔧 修复：显式将视频subsurface放置在父surface之上（避免被LVGL遮挡）
    // 这确保点击 LVGL UI 后视频仍然可见
    wl_subsurface_place_above(wl_subsurface, pImpl_->wl_surface_);
    
    // 提交更改到视频 surface
    wl_surface_commit(wl_surface);
    
    // 同时提交父 surface 以应用 Z-order 变化
    wl_surface_commit(pImpl_->wl_surface_);
    
    wl_display_flush(pImpl_->wl_display_);
    
    std::cout << "✅ DeepStream Subsurface 创建成功（异步模式，Z-order: 在LVGL之上）" << std::endl;
    std::cout << "📍 位置: (" << x << ", " << y << ") 尺寸: " 
              << width << "x" << height << std::endl;
    
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
    
    // 🆕 Jetson Orin NX特定：等待 Wayland 合成器完全就绪
    std::cout << "🔧 [Jetson] 等待 Wayland 合成器完全初始化..." << std::endl;
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
    
    while (!pImpl_->should_stop_.load()) {
        auto now = std::chrono::steady_clock::now();
        
        // ✅ 关键修复：处理Wayland事件循环
        pImpl_->handleWaylandEvents();
        
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
    
    // 🔧 最终架构：PARTIAL 模式 + 非持久化 SHM buffer
    // 
    // 为什么不用 DIRECT 模式？
    // - DIRECT 模式会在初始化时立即触发全屏渲染，导致卡住
    // - 每次渲染 9MB 数据效率低
    //
    // 为什么不用持久化 buffer？
    // - 持久化 buffer + 部分 damage 会导致状态不一致
    // - Wayland 合成器缓存导致渲染伪影
    //
    // 新方案：PARTIAL 模式 + 每次创建新 SHM buffer + 全屏 damage
    // - PARTIAL 模式：小 buffer（1/10 屏幕），渲染快
    // - 非持久化：每次 flush 创建新 SHM buffer
    // - 全屏 damage：标记整个屏幕，避免缓存问题
    // - 累积更新：在内部完整帧 buffer 上累积所有部分更新
    
    size_t partial_buffer_size = (config_.screen_width * config_.screen_height / 10) * sizeof(lv_color_t);
    front_buffer_ = (lv_color_t*)malloc(partial_buffer_size);
    back_buffer_ = (lv_color_t*)malloc(partial_buffer_size);
    
    if (!front_buffer_ || !back_buffer_) {
        std::cerr << "显示缓冲区分配失败" << std::endl;
        return false;
    }
    
    // 分配完整帧 buffer 用于累积更新
    size_t full_frame_size = config_.screen_width * config_.screen_height * sizeof(uint32_t);
    full_frame_buffer_ = (uint32_t*)malloc(full_frame_size);
    if (!full_frame_buffer_) {
        std::cerr << "完整帧缓冲区分配失败" << std::endl;
        return false;
    }
    
    // 初始化为深灰色背景
    uint32_t bg_color = 0xFF1E1E1E;
    for (size_t i = 0; i < (full_frame_size / sizeof(uint32_t)); i++) {
        full_frame_buffer_[i] = bg_color;
    }
    
    lv_display_set_buffers(display_, front_buffer_, back_buffer_,
                          partial_buffer_size, LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    std::cout << "✅ LVGL 使用 PARTIAL 渲染模式 (buffer: " 
              << (partial_buffer_size / 1024) << " KB, 全帧累积: " 
              << (full_frame_size / 1024) << " KB)" << std::endl;
    
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
    if (!wl_surface_ || !wl_shm_ || !full_frame_buffer_) return;
    
    int width = config_.screen_width;
    int height = config_.screen_height;
    
    // 🔧 步骤1：将此次的部分更新累积到完整帧 buffer
    // PARTIAL 模式下，color_p 只包含更新区域的数据
    int area_width = area->x2 - area->x1 + 1;
    int area_height = area->y2 - area->y1 + 1;
    
    #if LV_COLOR_DEPTH == 32
        // 逐行拷贝更新区域到完整帧 buffer
        int color_idx = 0;
        for (int y = area->y1; y <= area->y2; y++) {
            uint32_t* row_start = full_frame_buffer_ + y * width + area->x1;
            memcpy(row_start, &color_p[color_idx], area_width * sizeof(uint32_t));
            color_idx += area_width;
        }
    #else
        #error "Only LV_COLOR_DEPTH=32 is supported"
    #endif
    
    // 🔧 步骤2：创建新的 SHM buffer，拷贝完整帧
    // 这样 Wayland 合成器总是看到完整、一致的帧
    int stride = width * 4;
    size_t size = stride * height;
    
    int fd = createAnonymousFile(size);
    if (fd < 0) {
        std::cerr << "❌ 创建SHM文件失败" << std::endl;
        return;
    }
    
    void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        std::cerr << "❌ mmap失败" << std::endl;
        return;
    }
    
    // 从完整帧 buffer 拷贝到 SHM
    memcpy(data, full_frame_buffer_, size);
    
    // 🔧 步骤3：创建 Wayland buffer 并提交
    struct wl_shm_pool* pool = wl_shm_create_pool(wl_shm_, fd, size);
    struct wl_buffer* buffer = wl_shm_pool_create_buffer(
        pool, 0, width, height, stride, WL_SHM_FORMAT_ARGB8888);
    wl_shm_pool_destroy(pool);
    close(fd);
    
    // 附加 buffer 并标记整个屏幕为 damage
    // 虽然只更新了部分区域，但我们标记全屏，确保合成器看到完整帧
    wl_surface_attach(wl_surface_, buffer, 0, 0);
    wl_surface_damage(wl_surface_, 0, 0, width, height);
    wl_surface_commit(wl_surface_);
    wl_display_flush(wl_display_);
    
    // 设置 buffer 释放回调
    static const struct wl_buffer_listener buffer_listener = {
        [](void* data, struct wl_buffer* buffer) {
            wl_buffer_destroy(buffer);
        }
    };
    wl_buffer_add_listener(buffer, &buffer_listener, nullptr);
    
    // 释放 mmap
    munmap(data, size);
    
    // 调试：前5次 flush 打印信息
    static int flush_count = 0;
    if (++flush_count <= 5) {
        std::cout << "🖼️  LVGL flush #" << flush_count 
                  << " 更新区域: [" << area->x1 << "," << area->y1 
                  << " -> " << area->x2 << "," << area->y2 
                  << "] " << area_width << "x" << area_height 
                  << " → 提交完整帧 " << width << "x" << height << std::endl;
    }
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
    // 🔧 修复：让摄像头面板完全透明，这样DeepStream视频可以显示
    int camera_width = (int)(config_.screen_width * 0.6);
    camera_panel_ = lv_obj_create(main_container);
    lv_obj_set_size(camera_panel_, camera_width, config_.screen_height - 120);
    lv_obj_align(camera_panel_, LV_ALIGN_LEFT_MID, 10, 0);
    lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 🔧 透明背景
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 🔧 透明边框
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 🔧 修复：让摄像头面板不接收点击事件，点击穿透
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_CLICKABLE);  // 禁用点击响应
    lv_obj_add_flag(camera_panel_, LV_OBJ_FLAG_EVENT_BUBBLE);  // 让事件向上传递
    
    // 🔧 修复：不创建Canvas，让视频直接显示
    // DeepStream Subsurface会自动显示在这个位置
    // 如果需要边框，可以创建一个label
    lv_obj_t* video_label = lv_label_create(camera_panel_);
    lv_label_set_text(video_label, "Camera Feed (DeepStream)");
    lv_obj_set_style_text_color(video_label, lv_color_white(), 0);
    lv_obj_align(video_label, LV_ALIGN_TOP_LEFT, 5, 5);
    
    // 保留camera_canvas_指针为nullptr（视频不使用Canvas）
    camera_canvas_ = nullptr;
    
    std::cout << "📺 摄像头区域已设置为透明，DeepStream视频将显示在下方" << std::endl;
    
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
    
    // 🔧 修复：递归标记所有子对象为脏，确保完整刷新
    lv_obj_invalidate(main_screen_);
    lv_obj_invalidate(header_panel_);
    lv_obj_invalidate(camera_panel_);
    lv_obj_invalidate(control_panel_);
    lv_obj_invalidate(footer_panel_);
    
    std::cout << "✅ UI 创建完成，已标记所有面板需要刷新" << std::endl;
}

void LVGLWaylandInterface::Impl::updateCanvasFromFrame() {
    // 🔧 修复：不再使用Canvas，视频由DeepStream Subsurface直接显示
    // 这个方法现在是空操作（no-op），视频渲染由GPU加速的waylandsink处理
    
    // 如果将来需要叠加检测框等信息，可以在这里实现
    return;
}


bool LVGLWaylandInterface::Impl::initializeWaylandClient() {
    std::cout << "正在初始化Wayland客户端..." << std::endl;
    
    // 🔧 修复：禁用 Wayland 协议调试日志
    unsetenv("WAYLAND_DEBUG");
    
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
    // 🔧 修复：新版协议使用 xdg_wm_base_get_xdg_surface 而非 xdg_wm_base_create_xdg_surface
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
    
    // ⚠️ 关键：必须先添加监听器，再设置属性
    static const struct xdg_toplevel_listener xdg_toplevel_listener = {
        xdgToplevelConfigure,
        xdgToplevelClose
    };
    xdg_toplevel_add_listener(xdg_toplevel_, &xdg_toplevel_listener, this);
    std::cout << "✅ XDG Toplevel 监听器已添加" << std::endl;
    
    // 现在可以安全地设置窗口属性
    xdg_toplevel_set_title(xdg_toplevel_, "Bamboo Recognition System");
    xdg_toplevel_set_app_id(xdg_toplevel_, "bamboo-cut-lvgl");
    xdg_toplevel_set_fullscreen(xdg_toplevel_, nullptr);
    std::cout << "✅ XDG Toplevel 创建成功，已设置全屏" << std::endl;
    
    // ⚠️ 关键：第一次 commit 必须是空 commit（不附加 buffer）
    // 这是 xdg-shell 协议的要求，用于触发 configure 事件
    std::cout << "📝 执行空 commit，触发 configure 事件..." << std::endl;
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
    
    std::cout << "✅ 收到 configure 事件" << std::endl;
    
    // ✅ 现在可以创建并附加 buffer（在 configure 之后）
    std::cout << "🎨 创建初始 SHM buffer..." << std::endl;
    
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
    
    // 🔧 修复：填充深灰色背景（与 LVGL 主题一致），而不是黑色
    // 这样即使 LVGL 刷新不完整，屏幕也能显示合理的背景色
    uint32_t* pixels = (uint32_t*)data;
    uint32_t bg_color = 0xFF1E1E1E;  // ARGB: #1E1E1E（深灰色）
    for (int i = 0; i < (size / 4); i++) {
        pixels[i] = bg_color;
    }
    
    // 创建 wl_shm_pool
    struct wl_shm_pool* pool = wl_shm_create_pool(wl_shm_, fd, size);
    
    // 创建全屏 buffer
    struct wl_buffer* buffer = wl_shm_pool_create_buffer(
        pool, 0, config_.screen_width, config_.screen_height, stride, WL_SHM_FORMAT_ARGB8888);
    
    wl_shm_pool_destroy(pool);
    munmap(data, size);
    close(fd);
    
    // 附加 buffer 并提交（第二次 commit）
    wl_surface_attach(wl_surface_, buffer, 0, 0);
    wl_surface_damage(wl_surface_, 0, 0, config_.screen_width, config_.screen_height);
    wl_surface_commit(wl_surface_);
    wl_display_flush(wl_display_);
    
    std::cout << "✅ Buffer 已附加并提交: " << config_.screen_width << "x" << config_.screen_height << std::endl;
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
    
    // 🔧 修复：只在首次 configure 或尺寸变化时打印日志
    static bool first_toplevel_config = true;
    static int32_t last_width = 0;
    static int32_t last_height = 0;
    
    bool size_changed = (width != last_width || height != last_height);
    
    if (first_toplevel_config || size_changed) {
        if (width > 0 && height > 0) {
            std::cout << "📐 窗口尺寸: " << width << "x" << height << std::endl;
            impl->config_.screen_width = width;
            impl->config_.screen_height = height;
            last_width = width;
            last_height = height;
        }
        
        // 只在首次打印窗口状态
        if (first_toplevel_config && states && states->size > 0) {
            uint32_t* state_data = static_cast<uint32_t*>(states->data);
            size_t num_states = states->size / sizeof(uint32_t);
            
            for (size_t i = 0; i < num_states; i++) {
                if (state_data[i] == XDG_TOPLEVEL_STATE_FULLSCREEN) {
                    std::cout << "🔳 窗口模式: 全屏" << std::endl;
                    break;
                }
            }
        }
        
        first_toplevel_config = false;
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
    
    // 🔧 修复：只打印关键接口的绑定信息
    if (strcmp(interface, "wl_compositor") == 0) {
        impl->wl_compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 1));
        std::cout << "✅ 绑定wl_compositor" << std::endl;
    }
    else if (strcmp(interface, "wl_subcompositor") == 0) {
        impl->wl_subcompositor_ = static_cast<struct wl_subcompositor*>(
            wl_registry_bind(registry, id, &wl_subcompositor_interface, 1));
        std::cout << "✅ 绑定wl_subcompositor" << std::endl;
    }
    else if (strcmp(interface, "wl_shm") == 0) {
        impl->wl_shm_ = static_cast<struct wl_shm*>(
            wl_registry_bind(registry, id, &wl_shm_interface, 1));
        std::cout << "✅ 绑定wl_shm" << std::endl;
    }
    else if (strcmp(interface, "xdg_wm_base") == 0) {
    uint32_t use_version = (version < 3) ? version : 3;
    impl->xdg_wm_base_ = static_cast<struct xdg_wm_base*>(
        wl_registry_bind(registry, id, &xdg_wm_base_interface, use_version));
    std::cout << "✅ 绑定xdg_wm_base (v" << use_version << ")" << std::endl;
    }
    // 其他接口静默绑定，不打印日志
}

void LVGLWaylandInterface::Impl::registryRemover(void* data, struct wl_registry* registry, uint32_t id) {
    // 处理全局对象移除（可选实现）
}

// ✅ 新增：xdg-shell协议回调函数实现
void LVGLWaylandInterface::Impl::xdgWmBasePing(void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial) {
    // 🔧 修复：静默处理 ping/pong，不打印日志
    xdg_wm_base_pong(xdg_wm_base, serial);
}

void LVGLWaylandInterface::Impl::xdgSurfaceConfigure(void* data, struct xdg_surface* xdg_surface, uint32_t serial) {
    LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(data);
    
    // 🔧 修复：只在首次 configure 时打印日志
    static bool first_configure = true;
    if (first_configure) {
        std::cout << "📐 收到首次 XDG surface 配置" << std::endl;
        first_configure = false;
    }
    
    // 步骤6: 确认configure（协议要求必须ack）
    xdg_surface_ack_configure(xdg_surface, serial);
    
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
    
    // 🔧 修复：完全禁用帧回调日志，避免日志泛滥
    // 如需调试，可以取消下面的注释
    /*
    static uint32_t last_time = 0;
    if (last_time > 0) {
        uint32_t delta = time - last_time;
        if (delta > 0) {
            float fps = 1000.0f / delta;
            static int frame_count = 0;
            frame_count++;
            if (frame_count % 300 == 0) { // 每5秒（60fps）打印一次
                std::cout << "🎬 Wayland FPS: " << fps << std::endl;
            }
        }
    }
    last_time = time;
    */
    
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
    if (!wl_display_) {
        return;
    }
    
    // 🔧 修复：完全禁用事件处理日志，避免日志泛滥
    // 只在出错时打印错误信息
    
    // 处理所有待处理的事件，但不阻塞
    while (wl_display_prepare_read(wl_display_) != 0) {
        wl_display_dispatch_pending(wl_display_);
    }
    
    // 检查是否有数据可读
    if (wl_display_flush(wl_display_) < 0) {
        static int flush_error_count = 0;
        if (flush_error_count++ < 3) {  // 只打印前3次错误
            std::cerr << "❌ Wayland display flush失败" << std::endl;
        }
    }
    
    // 读取并分发事件（非阻塞）
    if (wl_display_read_events(wl_display_) >= 0) {
        wl_display_dispatch_pending(wl_display_);
    } else {
        wl_display_cancel_read(wl_display_);
        static int read_error_count = 0;
        if (read_error_count++ < 3) {  // 只打印前3次错误
            std::cerr << "❌ Wayland事件读取失败" << std::endl;
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
    // 注意：新架构中不再有持久化 SHM buffer（每次 flush 创建新 buffer）
    
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
    
    if (full_frame_buffer_) {
        free(full_frame_buffer_);
        full_frame_buffer_ = nullptr;
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