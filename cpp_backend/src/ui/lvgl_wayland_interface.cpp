/**
 * @file lvgl_wayland_interface.cpp
 * @brief LVGL Wayland接口实现 - 标准 Wayland 协议
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/ui/lvgl_ui_utils.h"           // 🆕 共享 UI 工具函数
#include "bamboo_cut/utils/logger.h"
#include "bamboo_cut/core/data_bridge.h"           // 🆕 数据桥接器
#include "bamboo_cut/utils/jetson_monitor.h"       // 🆕 Jetson 系统监控
#include <lvgl.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <poll.h>
#include <sstream>      // 🆕 用于字符串格式化
#include <iomanip>      // 🆕 用于格式化输出   

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
#include <algorithm>

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
    
    // 🆕 缓存 camera_panel 的坐标（避免每次 flush 都计算）
    int camera_x1_ = 0, camera_y1_ = 0, camera_x2_ = 0, camera_y2_ = 0;
    
    // 🆕 DeepStream subsurface 指针（用于维护 Z-order）
    struct wl_subsurface* video_subsurface_ = nullptr;
    
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
    uint32_t* full_frame_buffer_ = nullptr;  // 完整帧累积 buffer（ARGB8888）
    
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
    std::atomic<bool> ui_created_{false};  // 🔧 UI 是否已创建完成
    
    // Canvas更新
    cv::Mat latest_frame_;
    std::atomic<bool> new_frame_available_{false};
    
    // 🆕 数据源（与原始 UI 一致）
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge_;
    std::shared_ptr<bamboo_cut::utils::JetsonMonitor> jetson_monitor_;
    
    // 性能监控
    std::chrono::steady_clock::time_point last_update_time_;  // 🔧 使用 steady_clock
    int frame_count_ = 0;
    float ui_fps_ = 0.0f;
    
    // 🆕 UI 控件引用（用于动态更新）
    LVGLControlWidgets control_widgets_;
    LVGLThemeColors theme_colors_;
    
    // 初始化状态
    bool wayland_initialized_ = false;
    bool display_initialized_ = false;
    bool input_initialized_ = false;
    bool debug_camera_panel_opaque_ = false;
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

bool LVGLWaylandInterface::getCameraPanelCoords(int& x, int& y, int& width, int& height) {
    if (!pImpl_ || !pImpl_->camera_panel_) {
        return false;
    }
    
    // 强制更新布局以获取最新坐标
    lv_obj_update_layout(pImpl_->camera_panel_);
    
    lv_area_t area;
    lv_obj_get_coords(pImpl_->camera_panel_, &area);
    
    x = area.x1;
    y = area.y1;
    width = area.x2 - area.x1;
    height = area.y2 - area.y1;
    
    return true;
}

LVGLWaylandInterface::LVGLWaylandInterface(std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge) 
    : pImpl_(std::make_unique<Impl>()) {
    // 🆕 初始化数据源（与原始 UI 一致）
    pImpl_->data_bridge_ = data_bridge;
    pImpl_->jetson_monitor_ = std::make_shared<bamboo_cut::utils::JetsonMonitor>();
    pImpl_->last_update_time_ = std::chrono::steady_clock::now();  // 🔧 使用 steady_clock
    
    // 🆕 初始化主题颜色（使用原始 UI 配色）
    pImpl_->theme_colors_ = ui::LVGLThemeColors();

    // 🧪 调试：允许通过环境变量让摄像头面板保持不透明，方便诊断
    if (const char* env_value = std::getenv("BAMBOO_CAMERA_PANEL_OPAQUE")) {
        std::string value = env_value;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (value == "1" || value == "true" || value == "yes" || value == "on" || value == "debug") {
            pImpl_->debug_camera_panel_opaque_ = true;
            std::cout << "⚠️  [Debug] 摄像头面板使用不透明模式 (BAMBOO_CAMERA_PANEL_OPAQUE=" << env_value << ")" << std::endl;
        }
    }
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
    
    // 🆕 启动 Jetson 系统监控（如果可用）
    if (pImpl_->jetson_monitor_) {
        pImpl_->jetson_monitor_->start();
        std::cout << "✅ Jetson 系统监控已启动" << std::endl;
    }
    
    // 🔧 关键：UI 创建完成后设置标志，启用正常渲染
    std::cout << "✅ UI 创建完成，启用正常渲染..." << std::endl;
    pImpl_->ui_created_.store(true);
    
    // 🔧 强制触发完整刷新，确保UI立即显示
    std::cout << "🔄 强制刷新整个屏幕..." << std::endl;
    if (pImpl_->display_) {
        lv_obj_invalidate(lv_screen_active());  // 标记当前屏幕为脏
        lv_refr_now(pImpl_->display_);          // 立即刷新
        std::cout << "✅ 初始刷新完成" << std::endl;
    }
    
    std::cout << "✅ PARTIAL 模式渲染就绪" << std::endl;
    
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
    
    // 🔧 注意：Wayland 版本使用 subsurface 渲染视频，不需要 camera_canvas_
    // camera_canvas_ 被设置为 nullptr 是正常的
    // if (!pImpl_->camera_canvas_) {
    //     return false;
    // }
    
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
    auto last_data_update = std::chrono::steady_clock::now();
    const auto frame_time = std::chrono::milliseconds(1000 / pImpl_->config_.refresh_rate);
    const auto data_update_interval = std::chrono::milliseconds(500);  // 每 500ms 更新一次数据
    
    std::cout << "🚀 LVGL UI线程启动 (刷新率: " << pImpl_->config_.refresh_rate << "fps)" << std::endl;
    
    // 等待几帧后再开始数据更新（确保 UI 完全初始化）
    int warmup_frames = 0;
    const int warmup_threshold = 10;
    
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
        
        // 🆕 定期更新动态数据（Jetson 监控、AI 统计等）
        warmup_frames++;
        if (warmup_frames > warmup_threshold) {
            auto data_elapsed = now - last_data_update;
            if (data_elapsed >= data_update_interval) {
                std::lock_guard<std::mutex> lock(pImpl_->ui_mutex_);
                
                // 更新 Jetson 系统监控
                ui::updateJetsonMonitoring(pImpl_->control_widgets_, 
                                          pImpl_->jetson_monitor_, 
                                          pImpl_->theme_colors_);
                
                // 更新 AI 模型统计
                ui::updateAIModelStats(pImpl_->control_widgets_, 
                                      pImpl_->data_bridge_);
                
                // 更新摄像头状态
                ui::updateCameraStatus(pImpl_->control_widgets_, 
                                      pImpl_->data_bridge_);
                
                // 更新 Modbus 通信状态
                ui::updateModbusStatus(pImpl_->control_widgets_, 
                                      pImpl_->data_bridge_);
                
                // 计算 UI FPS
                pImpl_->frame_count_++;
                auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - pImpl_->last_update_time_).count();
                if (time_since_last >= 1000) {
                    pImpl_->ui_fps_ = (pImpl_->frame_count_ * 1000.0f) / time_since_last;
                    pImpl_->frame_count_ = 0;
                    pImpl_->last_update_time_ = now;
                    
                    // 更新 UI FPS 标签
                    if (pImpl_->control_widgets_.ui_fps_label) {
                        std::ostringstream fps_text;
                        fps_text << LV_SYMBOL_EYE_OPEN " UI: " << std::fixed << std::setprecision(1) 
                                << pImpl_->ui_fps_ << " fps";
                        lv_label_set_text(pImpl_->control_widgets_.ui_fps_label, fps_text.str().c_str());
                    }
                }
                
                last_data_update = now;
            }
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
    
    // 构建 socket 路径：
    //  1. 如果 WAYLAND_DISPLAY 已经是绝对路径（以 / 开头），直接使用
    //  2. 否则优先使用 XDG_RUNTIME_DIR（nvweston 为 /run/nvidia-wayland）
    //  3. 再退回到 /run/user/<uid>
    std::string socket_path;
    if (wayland_display[0] == '/') {
        socket_path = wayland_display;
    } else {
        const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
        if (runtime_dir && runtime_dir[0] == '/') {
            socket_path = std::string(runtime_dir) + "/" + wayland_display;
        } else {
            socket_path = "/run/user/" + std::to_string(getuid()) + "/" + wayland_display;
        }
    }
    
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
    
    std::cout << "🔧 [DEBUG] 步骤1: 创建 LVGL 显示设备..." << std::endl;
    std::cout << "   屏幕尺寸: " << config_.screen_width << "x" << config_.screen_height << std::endl;
    
    // 创建 LVGL 显示设备
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) {
        std::cerr << "LVGL显示创建失败" << std::endl;
        return false;
    }
    std::cout << "✅ [DEBUG] LVGL显示创建成功" << std::endl;
    
    // 🔧 核心架构：DIRECT 模式解决渲染伪影
    // 
    // 核心目标：同时正确显示 UI 和视频
    // 
    // 为什么必须用 DIRECT 模式？
    // - PARTIAL + 持久化/累积 buffer 会导致状态不一致
    // - Wayland 合成器缓存部分 damage，导致渲染伪影（黑白条纹）
    // - 只有 DIRECT 模式才能确保合成器看到完整、一致的帧
    //
    // DIRECT 模式初始化卡住的解决方案：
    // - 延迟注册 flush 回调到 UI 创建之后
    // - 避免在初始化阶段触发渲染
    
    std::cout << "🔧 [DEBUG] 步骤2: 分配完整帧累积 buffer..." << std::endl;
    // 为 PARTIAL 模式累积完整帧
    size_t full_frame_size = config_.screen_width * config_.screen_height * sizeof(uint32_t);
    full_frame_buffer_ = (uint32_t*)malloc(full_frame_size);
    
    if (!full_frame_buffer_) {
        std::cerr << "完整帧 buffer 分配失败" << std::endl;
        return false;
    }
    
    std::cout << "   完整帧 buffer 大小: " << (full_frame_size / 1024) << " KB" << std::endl;
    
    // 初始化为深灰色背景
    uint32_t bg_color = 0xFF1E1E1E;
    for (size_t i = 0; i < (full_frame_size / sizeof(uint32_t)); i++) {
        full_frame_buffer_[i] = bg_color;
    }
    std::cout << "✅ [DEBUG] 完整帧 buffer 已初始化" << std::endl;
    
    std::cout << "🔧 [DEBUG] 步骤4: 使用 PARTIAL 模式（DIRECT 模式不稳定）..." << std::endl;
    // 🔧 架构决策：DIRECT 模式在 lv_display_set_buffers() 时卡住
    // 根本原因：LVGL DIRECT 模式在这个版本/环境下有问题
    // 
    // 最终方案：PARTIAL 模式 + 完整帧提交
    // - PARTIAL 模式：LVGL 只渲染变化区域到小 buffer
    // - flush 时：提交 LVGL 的完整 display buffer（包含所有累积更新）
    // - 全屏 damage：确保 Wayland 合成器刷新整个屏幕
    
    // 分配 PARTIAL 模式的 buffer（1/10 屏幕大小）
    size_t partial_buffer_size = (config_.screen_width * config_.screen_height / 10) * sizeof(lv_color_t);
    front_buffer_ = (lv_color_t*)malloc(partial_buffer_size);
    back_buffer_ = (lv_color_t*)malloc(partial_buffer_size);
    
    if (!front_buffer_ || !back_buffer_) {
        std::cerr << "PARTIAL 模式 buffer 分配失败" << std::endl;
        return false;
    }
    
    std::cout << "   PARTIAL buffer 大小: " << (partial_buffer_size / 1024) << " KB × 2" << std::endl;
    
    // 设置 PARTIAL 模式（不会立即触发渲染）
    lv_display_set_buffers(display_, front_buffer_, back_buffer_,
                          partial_buffer_size, LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    std::cout << "✅ LVGL 使用 PARTIAL 渲染模式" << std::endl;
    
    // 注册 flush 回调
    lv_display_set_user_data(display_, this);
    lv_display_set_flush_cb(display_, [](lv_display_t* disp, const lv_area_t* area, uint8_t* color_p) {
        LVGLWaylandInterface::Impl* impl = static_cast<LVGLWaylandInterface::Impl*>(
            lv_display_get_user_data(disp));
        
        if (impl && impl->ui_created_) {
            impl->flushDisplayViaSHM(area, (lv_color_t*)color_p);
        }
        
        lv_display_flush_ready(disp);
    });
    
    std::cout << "✅ flush 回调已注册" << std::endl;
    
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
    
    // PARTIAL 模式步骤1: 将增量区域拷贝到完整帧 buffer
    int area_width = area->x2 - area->x1 + 1;
    int area_height = area->y2 - area->y1 + 1;
    
#if LV_COLOR_DEPTH == 32
    const uint32_t* src_pixels = reinterpret_cast<const uint32_t*>(color_p);
    for (int y = area->y1; y <= area->y2; y++) {
        uint32_t* dst_row = full_frame_buffer_ + y * width + area->x1;
        const uint32_t* src_row = src_pixels + (y - area->y1) * area_width;
        memcpy(dst_row, src_row, area_width * sizeof(uint32_t));
    }
#else
#error "Only LV_COLOR_DEPTH=32 is supported"
#endif
    
    // DEBUG: 暂时取消 camera_panel 透明/opaque/input region 优化，
    // 使用完全不透明 UI，由 Subsurface 负责覆盖摄像头区域。
    
    // 步骤2: 创建临时 SHM buffer 并提交完整帧
    int stride = width * 4;
    size_t size = stride * height;
    
    int fd = createAnonymousFile(size);
    if (fd < 0) {
        std::cerr << "创建 SHM 文件失败" << std::endl;
        return;
    }
    
    void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        std::cerr << "mmap 失败" << std::endl;
        return;
    }
    
    memcpy(data, full_frame_buffer_, size);
    
    struct wl_shm_pool* pool = wl_shm_create_pool(wl_shm_, fd, size);
    struct wl_buffer* buffer = wl_shm_pool_create_buffer(
        pool, 0, width, height, stride, WL_SHM_FORMAT_ARGB8888);
    wl_shm_pool_destroy(pool);
    close(fd);
    
    wl_surface_attach(wl_surface_, buffer, 0, 0);
    wl_surface_damage_buffer(wl_surface_, 0, 0, width, height);
    wl_surface_commit(wl_surface_);
    wl_display_flush(wl_display_);
    
    static const struct wl_buffer_listener buffer_listener = {
        [](void* data, struct wl_buffer* buffer) {
            wl_buffer_destroy(buffer);
        }
    };
    wl_buffer_add_listener(buffer, &buffer_listener, nullptr);
    
    munmap(data, size);
    
    static int flush_count = 0;
    if (++flush_count <= 5) {
        std::cout << "🖼️  LVGL flush #" << flush_count
                  << " PARTIAL 更新 [" << area->x1 << "," << area->y1
                  << "-" << area->x2 << "," << area->y2
                  << "] → 提交完整帧 " << width << "x" << height << std::endl;
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
    // 🔧 创建深色主题 - 使用中文字体
    // LV_FONT_DEFAULT 在 lv_conf.h 中定义为支持中文的字体
    lv_theme_t* theme = lv_theme_default_init(display_, 
                                            lv_palette_main(LV_PALETTE_BLUE), 
                                            lv_palette_main(LV_PALETTE_RED), 
                                            true,  // dark mode
                                            LV_FONT_DEFAULT);  // 使用宏，自动解析为中文字体
    lv_display_set_theme(display_, theme);
    
    std::cout << "✅ 主题已初始化（默认字体：Montserrat）" << std::endl;
    std::cout << "⚠️  注意：中文字符可能显示为方框，需要自定义字体支持" << std::endl;
}

void LVGLWaylandInterface::Impl::createMainInterface() {
    // 🎨 使用原始 UI 的配色方案
    lv_color_t color_background = lv_color_hex(0x1A1F26);   // 温和深色背景
    lv_color_t color_surface    = lv_color_hex(0x252B35);   // 卡片表面
    lv_color_t color_primary    = lv_color_hex(0x5B9BD5);   // 柔和蓝色主色
    lv_color_t color_success    = lv_color_hex(0x7FB069);   // 柔和绿色
    lv_color_t color_warning    = lv_color_hex(0xE6A055);   // 温和橙色
    lv_color_t color_error      = lv_color_hex(0xD67B7B);   // 柔和红色
    
    // 创建主屏幕 - 使用 Flex 布局（与原始 UI 一致）
    main_screen_ = lv_obj_create(nullptr);
    lv_obj_set_size(main_screen_, config_.screen_width, config_.screen_height);
    lv_obj_set_style_bg_color(main_screen_, color_background, 0);
    lv_obj_set_style_bg_opa(main_screen_, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(main_screen_, 0, 0);
    lv_obj_clear_flag(main_screen_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 🔧 关键：设置主屏幕为垂直 Flex 布局（与原始 UI 一致）
    lv_obj_set_flex_flow(main_screen_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(main_screen_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(main_screen_, 5, 0);
    
    // === 创建头部面板 === (固定高度，使用 Flex)
    header_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_width(header_panel_, lv_pct(100));
    lv_obj_set_height(header_panel_, 60);
    lv_obj_set_flex_grow(header_panel_, 0);  // 不允许增长
    lv_obj_set_style_bg_color(header_panel_, color_surface, 0);
    lv_obj_set_style_radius(header_panel_, 0, 0);  // 无圆角
    lv_obj_set_style_border_width(header_panel_, 0, 0);
    lv_obj_set_style_pad_all(header_panel_, 10, 0);
    lv_obj_clear_flag(header_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 🆕 设置头部为水平Flex布局
    lv_obj_set_flex_flow(header_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(header_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(header_panel_, 20, 0);
    
    // 🆕 系统标题
    control_widgets_.system_title = lv_label_create(header_panel_);
    lv_label_set_text(control_widgets_.system_title, LV_SYMBOL_HOME " Bamboo Recognition System");
    lv_obj_set_style_text_color(control_widgets_.system_title, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.system_title, &lv_font_montserrat_16, 0);
    
    // 🆕 心跳标签（显示系统运行状态）
    control_widgets_.heartbeat_label = lv_label_create(header_panel_);
    lv_label_set_text(control_widgets_.heartbeat_label, LV_SYMBOL_LOOP " Online");
    lv_obj_set_style_text_color(control_widgets_.heartbeat_label, color_success, 0);
    
    // 🆕 响应时间标签
    control_widgets_.response_label = lv_label_create(header_panel_);
    lv_label_set_text(control_widgets_.response_label, LV_SYMBOL_CHARGE " 12ms");
    lv_obj_set_style_text_color(control_widgets_.response_label, color_primary, 0);
    
    // === 创建中间容器 === (占据剩余空间，使用水平 Flex 布局)
    lv_obj_t* main_container = lv_obj_create(main_screen_);
    lv_obj_set_width(main_container, lv_pct(100));
    lv_obj_set_flex_grow(main_container, 1);  // 占据剩余空间
    lv_obj_set_style_bg_opa(main_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(main_container, 0, 0);
    lv_obj_set_style_pad_all(main_container, 5, 0);
    lv_obj_clear_flag(main_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 🔧 设置为水平 Flex 布局（左右排列摄像头和控制面板）
    lv_obj_set_flex_flow(main_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(main_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(main_container, 10, 0);
    
    std::cout << "📐 [UI] 主容器使用 Flex 布局，水平排列" << std::endl;
    
    // === 创建摄像头面板 === (左侧，占 75% 宽度，使用 Flex)
    camera_panel_ = lv_obj_create(main_container);
    lv_obj_set_height(camera_panel_, lv_pct(100));
    lv_obj_set_flex_grow(camera_panel_, 3);  // 占 3/4 空间

    if (debug_camera_panel_opaque_) {
        lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0x1D2330), 0);
        lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_80, 0);
        lv_obj_set_style_border_color(camera_panel_, lv_color_hex(0x4A90E2), 0);
        lv_obj_set_style_border_opa(camera_panel_, LV_OPA_60, 0);
        lv_obj_set_style_border_width(camera_panel_, 2, 0);
    } else {
        lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 🔧 透明背景
        lv_obj_set_style_border_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 🔧 透明边框
    }

    lv_obj_set_style_pad_all(camera_panel_, 0, 0);
    lv_obj_set_style_radius(camera_panel_, 8, 0);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_CLICKABLE);  // 禁用点击响应
    lv_obj_add_flag(camera_panel_, LV_OBJ_FLAG_EVENT_BUBBLE);  // 让事件向上传递
    
    // 🔧 关键修复：完全透明但保持布局参与
    lv_obj_set_style_opa(camera_panel_, debug_camera_panel_opaque_ ? LV_OPA_COVER : LV_OPA_0, 0);  // 完全透明（包括子对象）
    // ❌ 不能使用 IGNORE_LAYOUT，会破坏 Flex 布局计算！
    
    std::cout << "📐 [UI] 摄像头面板: flex_grow=3 (75% 宽度，完全透明）" << std::endl;
    
    // 🔧 摄像头区域标签（仅用于调试）
    lv_obj_t* video_label = lv_label_create(camera_panel_);
    lv_label_set_text(video_label, LV_SYMBOL_VIDEO " Camera Feed");
    // Force camera panel fully transparent so subsurface video is visible
    lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_opa(camera_panel_, LV_OPA_0, 0);
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(camera_panel_, 0, 0);
    lv_obj_set_style_text_color(video_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(video_label, &lv_font_montserrat_14, 0);
    lv_obj_align(video_label, LV_ALIGN_TOP_LEFT, 10, 10);
    
    // 保留camera_canvas_指针为nullptr（视频不使用Canvas）
    camera_canvas_ = nullptr;
    
    std::cout << "📺 摄像头区域已设置为透明，DeepStream视频将显示在 subsurface" << std::endl;
    
    // === 创建控制面板 === (右侧，占 25% 宽度，使用 Flex)
    control_panel_ = lv_obj_create(main_container);
    lv_obj_set_height(control_panel_, lv_pct(100));
    lv_obj_set_flex_grow(control_panel_, 1);  // 占 1/4 空间
    lv_obj_set_style_bg_color(control_panel_, color_surface, 0);
    lv_obj_set_style_radius(control_panel_, 12, 0);
    lv_obj_set_style_border_width(control_panel_, 2, 0);
    lv_obj_set_style_border_color(control_panel_, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_border_opa(control_panel_, LV_OPA_50, 0);
    lv_obj_set_style_pad_all(control_panel_, 15, 0);
    lv_obj_clear_flag(control_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "📐 [UI] 控制面板: flex_grow=1 (25% 宽度)" << std::endl;
    
    // 🔧 详细控制面板 - 使用 Flex 垂直布局
    lv_obj_set_flex_flow(control_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(control_panel_, 10, 0);
    
    // === 标题 ===
    lv_obj_t* control_title = lv_label_create(control_panel_);
    lv_label_set_text(control_title, LV_SYMBOL_SETTINGS " System Info");
    lv_obj_set_style_text_color(control_title, lv_color_hex(0x5B9BD5), 0);
    lv_obj_set_style_text_font(control_title, &lv_font_montserrat_16, 0);
    
    // === Jetson 监控区域 ===
    lv_obj_t* jetson_section = lv_obj_create(control_panel_);
    lv_obj_set_width(jetson_section, lv_pct(100));
    lv_obj_set_height(jetson_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(jetson_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(jetson_section, 8, 0);
    lv_obj_set_style_border_width(jetson_section, 1, 0);
    lv_obj_set_style_border_color(jetson_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(jetson_section, 10, 0);
    lv_obj_set_flex_flow(jetson_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(jetson_section, 6, 0);
    lv_obj_clear_flag(jetson_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* jetson_title = lv_label_create(jetson_section);
    lv_label_set_text(jetson_title, LV_SYMBOL_CHARGE " Jetson Orin Nano");
    lv_obj_set_style_text_color(jetson_title, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(jetson_title, &lv_font_montserrat_14, 0);
    
    // === CPU 信息（带进度条）===
    control_widgets_.cpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.cpu_label, "CPU: --% @ --MHz");
    lv_obj_set_style_text_color(control_widgets_.cpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.cpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.cpu_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(control_widgets_.cpu_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(control_widgets_.cpu_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(control_widgets_.cpu_bar, 0, LV_ANIM_OFF);
    
    // === GPU 信息（带进度条）===
    control_widgets_.gpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.gpu_label, "GPU: --% @ --MHz");
    lv_obj_set_style_text_color(control_widgets_.gpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.gpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.gpu_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(control_widgets_.gpu_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(control_widgets_.gpu_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(control_widgets_.gpu_bar, 0, LV_ANIM_OFF);
    
    // === 内存信息（带进度条）===
    control_widgets_.mem_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.mem_label, "RAM: --MB / --MB");
    lv_obj_set_style_text_color(control_widgets_.mem_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.mem_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.mem_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(control_widgets_.mem_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(control_widgets_.mem_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(control_widgets_.mem_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(control_widgets_.mem_bar, 0, LV_ANIM_OFF);
    
    // === SWAP 使用率 ===
    control_widgets_.swap_usage_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.swap_usage_label, "SWAP: --MB");
    lv_obj_set_style_text_color(control_widgets_.swap_usage_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.swap_usage_label, &lv_font_montserrat_12, 0);
    
    // === 温度信息 ===
    control_widgets_.cpu_temp_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.cpu_temp_label, "CPU: --°C");
    lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_temp_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.gpu_temp_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.gpu_temp_label, "GPU: --°C");
    lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_temp_label, &lv_font_montserrat_12, 0);
    
    // === 热区警告 ===
    control_widgets_.thermal_warning_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.thermal_warning_label, "");
    lv_obj_set_style_text_color(control_widgets_.thermal_warning_label, color_error, 0);
    lv_obj_set_style_text_font(control_widgets_.thermal_warning_label, &lv_font_montserrat_12, 0);
    lv_obj_add_flag(control_widgets_.thermal_warning_label, LV_OBJ_FLAG_HIDDEN);  // 默认隐藏
    
    // === 功率信息 ===
    control_widgets_.power_total_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.power_total_label, "Power: --W");
    lv_obj_set_style_text_color(control_widgets_.power_total_label, color_primary, 0);
    lv_obj_set_style_text_font(control_widgets_.power_total_label, &lv_font_montserrat_12, 0);
    
    // === AI 模型区域 ===
    lv_obj_t* ai_section = lv_obj_create(control_panel_);
    lv_obj_set_width(ai_section, lv_pct(100));
    lv_obj_set_height(ai_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(ai_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(ai_section, 8, 0);
    lv_obj_set_style_border_width(ai_section, 1, 0);
    lv_obj_set_style_border_color(ai_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(ai_section, 10, 0);
    lv_obj_set_flex_flow(ai_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(ai_section, 6, 0);
    lv_obj_clear_flag(ai_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* ai_title = lv_label_create(ai_section);
    lv_label_set_text(ai_title, LV_SYMBOL_IMAGE " AI Model");
    lv_obj_set_style_text_color(ai_title, lv_color_hex(0x7FB069), 0);
    lv_obj_set_style_text_font(ai_title, &lv_font_montserrat_14, 0);
    
    // === 模型名称 ===
    control_widgets_.ai_model_name_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_model_name_label, "Model: YOLOv8");
    lv_obj_set_style_text_color(control_widgets_.ai_model_name_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_model_name_label, &lv_font_montserrat_12, 0);
    
    // === FPS ===
    control_widgets_.ai_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_fps_label, "FPS: -- fps");
    lv_obj_set_style_text_color(control_widgets_.ai_fps_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_fps_label, &lv_font_montserrat_12, 0);
    
    // === 推理时间 ===
    control_widgets_.ai_inference_time_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_inference_time_label, "Inference: --ms");
    lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_primary, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_inference_time_label, &lv_font_montserrat_12, 0);
    
    // === 检测数量 ===
    control_widgets_.ai_total_detections_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_total_detections_label, "Detected: 0 objects");
    lv_obj_set_style_text_color(control_widgets_.ai_total_detections_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_total_detections_label, &lv_font_montserrat_12, 0);
    
    // === 置信度 ===
    control_widgets_.ai_confidence_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_confidence_label, "Confidence: --%");
    lv_obj_set_style_text_color(control_widgets_.ai_confidence_label, color_success, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_confidence_label, &lv_font_montserrat_12, 0);
    
    // === 系统版本信息区域 ===
    lv_obj_t* version_section = lv_obj_create(control_panel_);
    lv_obj_set_width(version_section, lv_pct(100));
    lv_obj_set_height(version_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(version_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(version_section, 8, 0);
    lv_obj_set_style_border_width(version_section, 1, 0);
    lv_obj_set_style_border_color(version_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(version_section, 10, 0);
    lv_obj_set_flex_flow(version_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(version_section, 4, 0);
    lv_obj_clear_flag(version_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* version_title = lv_label_create(version_section);
    lv_label_set_text(version_title, LV_SYMBOL_LIST " System Info");
    lv_obj_set_style_text_color(version_title, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(version_title, &lv_font_montserrat_12, 0);
    
    // JetPack版本
    control_widgets_.jetpack_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.jetpack_version_label, "JetPack: 5.1.2");
    lv_obj_set_style_text_color(control_widgets_.jetpack_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.jetpack_version_label, &lv_font_montserrat_12, 0);
    
    // CUDA版本
    control_widgets_.cuda_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.cuda_version_label, "CUDA: 11.4");
    lv_obj_set_style_text_color(control_widgets_.cuda_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.cuda_version_label, &lv_font_montserrat_12, 0);
    
    // TensorRT版本
    control_widgets_.tensorrt_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.tensorrt_version_label, "TensorRT: 8.5.2");
    lv_obj_set_style_text_color(control_widgets_.tensorrt_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.tensorrt_version_label, &lv_font_montserrat_12, 0);
    
    // LVGL版本
    control_widgets_.lvgl_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.lvgl_version_label, "LVGL: 9.0.0");
    lv_obj_set_style_text_color(control_widgets_.lvgl_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.lvgl_version_label, &lv_font_montserrat_12, 0);
    
    // OpenCV version
    control_widgets_.opencv_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.opencv_version_label, "OpenCV: 4.x");
    lv_obj_set_style_text_color(control_widgets_.opencv_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.opencv_version_label, &lv_font_montserrat_12, 0);
    
    // Build time
    control_widgets_.build_time_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.build_time_label, "Build: " __DATE__ " " __TIME__);
    lv_obj_set_style_text_color(control_widgets_.build_time_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.build_time_label, &lv_font_montserrat_12, 0);

    // === 系统控制（无X11场景下的基础操作） ===
    lv_obj_t* system_section = lv_obj_create(control_panel_);
    lv_obj_set_width(system_section, lv_pct(100));
    lv_obj_set_height(system_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(system_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(system_section, 8, 0);
    lv_obj_set_style_border_width(system_section, 1, 0);
    lv_obj_set_style_border_color(system_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(system_section, 10, 0);
    lv_obj_set_flex_flow(system_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(system_section, 6, 0);
    lv_obj_clear_flag(system_section, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* system_title = lv_label_create(system_section);
    lv_label_set_text(system_title, LV_SYMBOL_SETTINGS " System Control");
    lv_obj_set_style_text_color(system_title, lv_color_hex(0x7FB069), 0);
    lv_obj_set_style_text_font(system_title, &lv_font_montserrat_14, 0);

    auto create_action_btn = [&](const char* text, const char* action_cmd) {
        lv_obj_t* btn = lv_btn_create(system_section);
        lv_obj_set_width(btn, lv_pct(100));
        lv_obj_set_height(btn, 36);
        lv_obj_add_event_cb(btn, [](lv_event_t* e) {
            if (lv_event_get_code(e) != LV_EVENT_CLICKED) return;
            const char* cmd = static_cast<const char*>(lv_event_get_user_data(e));
            if (cmd) {
                std::cout << "[UI] 系统操作: " << cmd << std::endl;
                std::system(cmd);
            }
        }, LV_EVENT_CLICKED, (void*)action_cmd);

        lv_obj_t* label = lv_label_create(btn);
        lv_label_set_text(label, text);
        lv_obj_center(label);
        return btn;
    };

    create_action_btn(LV_SYMBOL_POWER " Shutdown", "/usr/bin/systemctl poweroff");
    create_action_btn(LV_SYMBOL_REFRESH " Reboot", "/usr/bin/systemctl reboot");
    create_action_btn(LV_SYMBOL_WIFI " Restart Network", "/usr/bin/systemctl restart NetworkManager");
    create_action_btn(LV_SYMBOL_HOME " Restart Service", "/usr/bin/systemctl restart bamboo-cpp-lvgl");
    
    // === 摄像头状态区域 ===
    lv_obj_t* camera_section = lv_obj_create(control_panel_);
    lv_obj_set_width(camera_section, lv_pct(100));
    lv_obj_set_height(camera_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(camera_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(camera_section, 8, 0);
    lv_obj_set_style_border_width(camera_section, 1, 0);
    lv_obj_set_style_border_color(camera_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(camera_section, 10, 0);
    lv_obj_set_flex_flow(camera_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(camera_section, 6, 0);
    lv_obj_clear_flag(camera_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* camera_section_title = lv_label_create(camera_section);
    lv_label_set_text(camera_section_title, LV_SYMBOL_VIDEO " Camera Status");
    lv_obj_set_style_text_color(camera_section_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(camera_section_title, &lv_font_montserrat_14, 0);
    
    // 摄像头状态
    control_widgets_.camera_status_label = lv_label_create(camera_section);
    lv_label_set_text(control_widgets_.camera_status_label, "Status: Offline");
    lv_obj_set_style_text_color(control_widgets_.camera_status_label, color_error, 0);
    lv_obj_set_style_text_font(control_widgets_.camera_status_label, &lv_font_montserrat_12, 0);
    
    // 摄像头 FPS
    control_widgets_.camera_fps_label = lv_label_create(camera_section);
    lv_label_set_text(control_widgets_.camera_fps_label, "FPS: -- fps");
    lv_obj_set_style_text_color(control_widgets_.camera_fps_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.camera_fps_label, &lv_font_montserrat_12, 0);
    
    // 分辨率
    control_widgets_.camera_resolution_label = lv_label_create(camera_section);
    lv_label_set_text(control_widgets_.camera_resolution_label, "Resolution: --");
    lv_obj_set_style_text_color(control_widgets_.camera_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera_resolution_label, &lv_font_montserrat_12, 0);
    
    // 格式
    control_widgets_.camera_format_label = lv_label_create(camera_section);
    lv_label_set_text(control_widgets_.camera_format_label, "Format: --");
    lv_obj_set_style_text_color(control_widgets_.camera_format_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera_format_label, &lv_font_montserrat_12, 0);
    
    // === Modbus 通信区域 ===
    lv_obj_t* modbus_section = lv_obj_create(control_panel_);
    lv_obj_set_width(modbus_section, lv_pct(100));
    lv_obj_set_height(modbus_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(modbus_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(modbus_section, 8, 0);
    lv_obj_set_style_border_width(modbus_section, 1, 0);
    lv_obj_set_style_border_color(modbus_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(modbus_section, 10, 0);
    lv_obj_set_flex_flow(modbus_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(modbus_section, 6, 0);
    lv_obj_clear_flag(modbus_section, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* modbus_title = lv_label_create(modbus_section);
    lv_label_set_text(modbus_title, LV_SYMBOL_SHUFFLE " Modbus TCP");
    lv_obj_set_style_text_color(modbus_title, lv_color_hex(0xD67B7B), 0);
    lv_obj_set_style_text_font(modbus_title, &lv_font_montserrat_14, 0);
    
    // PLC 连接状态
    control_widgets_.modbus_connection_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_connection_label, "PLC: Disconnected");
    lv_obj_set_style_text_color(control_widgets_.modbus_connection_label, color_error, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_connection_label, &lv_font_montserrat_12, 0);
    
    // 地址
    control_widgets_.modbus_address_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_address_label, "Addr: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_address_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_address_label, &lv_font_montserrat_12, 0);
    
    // 延迟
    control_widgets_.modbus_latency_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_latency_label, "Latency: --ms");
    lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_primary, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_latency_label, &lv_font_montserrat_12, 0);
    
    // 错误计数
    control_widgets_.modbus_error_count_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_error_count_label, "Errors: 0");
    lv_obj_set_style_text_color(control_widgets_.modbus_error_count_label, color_success, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_error_count_label, &lv_font_montserrat_12, 0);
    
    // 消息计数
    control_widgets_.modbus_message_count_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_message_count_label, "Messages: 0");
    lv_obj_set_style_text_color(control_widgets_.modbus_message_count_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_message_count_label, &lv_font_montserrat_12, 0);
    
    // 心跳状态
    control_widgets_.modbus_heartbeat_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_heartbeat_label, "Heartbeat: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_heartbeat_label, color_warning, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_heartbeat_label, &lv_font_montserrat_12, 0);
    
    // === 创建底部面板 === (按照原版结构：Start/Pause/Stop/Emergency/Power按钮)
    // Modbus extension (register summary)
    control_widgets_.modbus_system_status_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_system_status_label, "40001 System: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_system_status_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_plc_command_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_plc_command_label, "40002 PLC Cmd: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_plc_command_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_coord_ready_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_coord_ready_label, "40003 Coord Ready: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_coord_ready_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_coord_ready_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_x_coordinate_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_x_coordinate_label, "40004 X: -- mm");
    lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_x_coordinate_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_cut_quality_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_cut_quality_label, "40006 Quality: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_cut_quality_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_cut_quality_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_blade_number_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_blade_number_label, "40009 Blade: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_blade_number_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_health_status_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_health_status_label, "40010 Health: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_health_status_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_tail_status_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_tail_status_label, "40011 Tail: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_tail_status_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_tail_status_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_plc_alarm_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_plc_alarm_label, "40012 PLC Alarm: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_plc_alarm_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_plc_alarm_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_rail_direction_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_rail_direction_label, "40014 Rail Dir: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_rail_direction_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_rail_direction_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_remaining_length_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_remaining_length_label, "40015 Remain: -- mm");
    lv_obj_set_style_text_color(control_widgets_.modbus_remaining_length_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_remaining_length_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_coverage_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_coverage_label, "40017 Coverage: --%");
    lv_obj_set_style_text_color(control_widgets_.modbus_coverage_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_coverage_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_feed_speed_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_feed_speed_label, "40018 Speed Gear: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_feed_speed_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_feed_speed_label, &lv_font_montserrat_12, 0);

    control_widgets_.modbus_process_mode_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_process_mode_label, "40019 Mode: --");
    lv_obj_set_style_text_color(control_widgets_.modbus_process_mode_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_process_mode_label, &lv_font_montserrat_12, 0);

    footer_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_width(footer_panel_, lv_pct(100));
    lv_obj_set_height(footer_panel_, 80);  // 原版高度80px
    lv_obj_set_flex_grow(footer_panel_, 0);  // 不允许增长
    lv_obj_set_style_bg_color(footer_panel_, color_surface, 0);
    lv_obj_set_style_radius(footer_panel_, 20, 0);  // 原版圆角20
    lv_obj_set_style_border_width(footer_panel_, 1, 0);
    lv_obj_set_style_border_color(footer_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(footer_panel_, LV_OPA_40, 0);
    lv_obj_set_style_pad_all(footer_panel_, 16, 0);
    lv_obj_clear_flag(footer_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 🔧 设置底部为水平 Flex 布局
    lv_obj_set_flex_flow(footer_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(footer_panel_, LV_FLEX_ALIGN_SPACE_AROUND, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(footer_panel_, 12, 0);
    
    // === 主操作区域（Start/Pause/Stop按钮，占70%） ===
    lv_obj_t* main_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(main_controls, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_opa(main_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(main_controls, 0, 0);
    lv_obj_set_style_pad_all(main_controls, 0, 0);
    lv_obj_set_flex_flow(main_controls, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(main_controls, LV_FLEX_ALIGN_SPACE_EVENLY, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_clear_flag(main_controls, LV_OBJ_FLAG_SCROLLABLE);
    
    // Start 按钮
    lv_obj_t* start_btn = lv_btn_create(main_controls);
    lv_obj_set_size(start_btn, 110, 48);
    lv_obj_set_style_bg_color(start_btn, color_success, 0);  // 绿色
    lv_obj_set_style_radius(start_btn, 12, 0);
    lv_obj_t* start_label = lv_label_create(start_btn);
    lv_label_set_text(start_label, LV_SYMBOL_PLAY " START");
    lv_obj_set_style_text_font(start_label, &lv_font_montserrat_16, 0);
    lv_obj_center(start_label);
    
    // Pause 按钮
    lv_obj_t* pause_btn = lv_btn_create(main_controls);
    lv_obj_set_size(pause_btn, 110, 48);
    lv_obj_set_style_bg_color(pause_btn, color_warning, 0);  // 橙色
    lv_obj_set_style_radius(pause_btn, 12, 0);
    lv_obj_t* pause_label = lv_label_create(pause_btn);
    lv_label_set_text(pause_label, LV_SYMBOL_PAUSE " PAUSE");
    lv_obj_set_style_text_font(pause_label, &lv_font_montserrat_16, 0);
    lv_obj_center(pause_label);
    
    // Stop 按钮
    lv_obj_t* stop_btn = lv_btn_create(main_controls);
    lv_obj_set_size(stop_btn, 110, 48);
    lv_obj_set_style_bg_color(stop_btn, lv_color_hex(0x6B7280), 0);  // 灰色
    lv_obj_set_style_radius(stop_btn, 12, 0);
    lv_obj_t* stop_label = lv_label_create(stop_btn);
    lv_label_set_text(stop_label, LV_SYMBOL_STOP " STOP");
    lv_obj_set_style_text_font(stop_label, &lv_font_montserrat_16, 0);
    lv_obj_center(stop_label);
    
    // === 危险操作区域（Emergency按钮） ===
    lv_obj_t* danger_zone = lv_obj_create(footer_panel_);
    lv_obj_set_size(danger_zone, 70, lv_pct(100));
    lv_obj_set_style_bg_opa(danger_zone, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(danger_zone, 0, 0);
    lv_obj_set_style_pad_all(danger_zone, 0, 0);
    lv_obj_set_flex_flow(danger_zone, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(danger_zone, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_clear_flag(danger_zone, LV_OBJ_FLAG_SCROLLABLE);
    
    // Emergency 按钮（急停，红色大按钮）
    lv_obj_t* emergency_btn = lv_btn_create(danger_zone);
    lv_obj_set_size(emergency_btn, 60, 60);
    lv_obj_set_style_bg_color(emergency_btn, color_error, 0);  // 红色
    lv_obj_set_style_radius(emergency_btn, 30, 0);  // 圆形
    lv_obj_t* emergency_label = lv_label_create(emergency_btn);
    lv_label_set_text(emergency_label, LV_SYMBOL_WARNING);
    lv_obj_set_style_text_font(emergency_label, &lv_font_montserrat_24, 0);
    lv_obj_center(emergency_label);
    
    // === 辅助操作区域（Power按钮 + 状态标签，占20%） ===
    lv_obj_t* aux_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(aux_controls, lv_pct(20), lv_pct(100));
    lv_obj_set_style_bg_opa(aux_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(aux_controls, 0, 0);
    lv_obj_set_style_pad_all(aux_controls, 0, 0);
    lv_obj_set_flex_flow(aux_controls, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(aux_controls, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(aux_controls, 4, 0);
    lv_obj_clear_flag(aux_controls, LV_OBJ_FLAG_SCROLLABLE);
    
    // Power/Settings 按钮
    lv_obj_t* power_btn = lv_btn_create(aux_controls);
    lv_obj_set_size(power_btn, 48, 48);
    lv_obj_set_style_bg_color(power_btn, color_primary, 0);
    lv_obj_set_style_radius(power_btn, 12, 0);
    lv_obj_t* power_label = lv_label_create(power_btn);
    lv_label_set_text(power_label, LV_SYMBOL_SETTINGS);
    lv_obj_set_style_text_font(power_label, &lv_font_montserrat_16, 0);
    lv_obj_center(power_label);
    
    // Process 标签
    control_widgets_.process_label = lv_label_create(aux_controls);
    lv_label_set_text(control_widgets_.process_label, "Process: Ready");
    lv_obj_set_style_text_color(control_widgets_.process_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.process_label, &lv_font_montserrat_12, 0);
    
    // Stats 标签
    control_widgets_.stats_label = lv_label_create(aux_controls);
    lv_label_set_text(control_widgets_.stats_label, "Stats: 0/0");
    lv_obj_set_style_text_color(control_widgets_.stats_label, color_primary, 0);
    lv_obj_set_style_text_font(control_widgets_.stats_label, &lv_font_montserrat_12, 0);
    
    // UI FPS 标签
    control_widgets_.ui_fps_label = lv_label_create(aux_controls);
    lv_label_set_text(control_widgets_.ui_fps_label, "UI: -- fps");
    lv_obj_set_style_text_color(control_widgets_.ui_fps_label, color_primary, 0);
    lv_obj_set_style_text_font(control_widgets_.ui_fps_label, &lv_font_montserrat_12, 0);
    
    std::cout << "📐 [UI] 底部面板: 80px高度，Start/Pause/Stop/Emergency/Power按钮" << std::endl;
    
    // 加载主屏幕
    lv_screen_load(main_screen_);
    
    // 🔧 修复：递归标记所有子对象为脏，确保完整刷新
    lv_obj_invalidate(main_screen_);
    lv_obj_invalidate(header_panel_);
    lv_obj_invalidate(camera_panel_);
    lv_obj_invalidate(control_panel_);
    lv_obj_invalidate(footer_panel_);
    
    // 🔍 在布局完成后获取 camera_panel 的实际坐标
    lv_obj_update_layout(main_screen_);  // 强制更新布局
    lv_area_t camera_area;
    lv_obj_get_coords(camera_panel_, &camera_area);
    
    // 🆕 缓存 camera_panel 坐标
    camera_x1_ = camera_area.x1;
    camera_y1_ = camera_area.y1;
    camera_x2_ = camera_area.x2;
    camera_y2_ = camera_area.y2;
    
    std::cout << "\n🔍 [关键诊断] camera_panel 最终坐标: ("
              << camera_area.x1 << ", " << camera_area.y1 << ") → ("
              << camera_area.x2 << ", " << camera_area.y2 << ")" << std::endl;
    std::cout << "🔍 [关键诊断] camera_panel 尺寸: " 
              << (camera_area.x2 - camera_area.x1) << "x" << (camera_area.y2 - camera_area.y1) << std::endl;
    
    // 🔧 关键修复：将 camera_panel 区域的 buffer 设置为完全透明
    // 这样 subsurface 的视频就能显示出来
    int width = config_.screen_width;
    if (full_frame_buffer_) {
        std::cout << "🔧 [Wayland] 清除 camera_panel 区域的 buffer（设为完全透明）..." << std::endl;
        for (int y = camera_y1_; y <= camera_y2_; y++) {
            for (int x = camera_x1_; x <= camera_x2_; x++) {
                full_frame_buffer_[y * width + x] = 0x00000000;  // 完全透明
            }
        }
        std::cout << "✅ [Wayland] camera_panel 区域已清除，subsurface 视频将可见" << std::endl;
    }
    std::cout << "⚠️  [关键] DeepStream subsurface 当前位置: (0, 60) 尺寸: 960x640" << std::endl;
    std::cout << "⚠️  [关键] 如果两者不匹配，视频将显示在错误位置！\n" << std::endl;
    
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
    
    // 🔧 升级：绑定 wl_compositor version 5（服务器支持的最高版本）
    // Version 4: 添加 wl_surface_damage_buffer() 支持
    // Version 5: 协议稳定版本，最佳兼容性
    if (strcmp(interface, "wl_compositor") == 0) {
        // 使用服务器支持的最高版本（最多 v5）
        uint32_t use_version = (version >= 5) ? 5 : version;
        impl->wl_compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, use_version));
        std::cout << "✅ 绑定wl_compositor (v" << use_version << ", 服务器支持: v" << version << ")" << std::endl;
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
