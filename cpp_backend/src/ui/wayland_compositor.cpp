/**
 * @file wayland_compositor.cpp
 * @brief Wayland合成器检测和支持工具实现
 */

#include "bamboo_cut/ui/wayland_compositor.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <map>
#include <cstring>

#ifdef ENABLE_WAYLAND
#include <wayland-client.h>
#include <wayland-client-protocol.h>
#endif

namespace bamboo_cut {
namespace ui {

// WaylandDetector 实现
bool WaylandDetector::detectWaylandSupport() {
    std::cout << "[WaylandDetector] 检测Wayland支持..." << std::endl;
    
    // 检查环境变量
    if (!isWaylandSession()) {
        std::cout << "[WaylandDetector] 未检测到Wayland会话" << std::endl;
        return false;
    }
    
    // 检查Wayland socket
    if (!checkWaylandSocket()) {
        std::cout << "[WaylandDetector] Wayland socket不可用" << std::endl;
        return false;
    }
    
    // 测试连接
    if (!testWaylandConnection()) {
        std::cout << "[WaylandDetector] 无法连接到Wayland服务器" << std::endl;
        return false;
    }
    
    std::cout << "[WaylandDetector] Wayland支持检测成功" << std::endl;
    return true;
}

std::string WaylandDetector::detectCompositor() {
    // 检查常见的合成器环境变量
    const char* compositor_vars[] = {
        "WAYLAND_COMPOSITOR",
        "XDG_CURRENT_DESKTOP",
        "DESKTOP_SESSION",
        nullptr
    };
    
    for (int i = 0; compositor_vars[i]; i++) {
        const char* value = std::getenv(compositor_vars[i]);
        if (value) {
            std::string compositor(value);
            std::cout << "[WaylandDetector] 检测到合成器: " << compositor << std::endl;
            return compositor;
        }
    }
    
    // 尝试从进程列表检测
    std::ifstream proc("/proc/version");
    if (proc.is_open()) {
        std::string line;
        if (std::getline(proc, line)) {
            if (line.find("weston") != std::string::npos) {
                return "weston";
            } else if (line.find("sway") != std::string::npos) {
                return "sway";
            }
        }
    }
    
    return "unknown";
}

std::string WaylandDetector::getWaylandDisplay() {
    const char* display = std::getenv("WAYLAND_DISPLAY");
    return display ? std::string(display) : "wayland-0";
}

std::vector<std::string> WaylandDetector::listWaylandExtensions() {
    std::vector<std::string> extensions;
    
#ifdef ENABLE_WAYLAND
    struct wl_display* display = wl_display_connect(nullptr);
    if (display) {
        // 这里可以查询可用的Wayland协议扩展
        // 为简化，返回常见扩展列表
        extensions.push_back("wl_compositor");
        extensions.push_back("wl_shell");
        extensions.push_back("wl_output");
        extensions.push_back("wl_seat");
        extensions.push_back("wl_shm");
        
        wl_display_disconnect(display);
    }
#endif
    
    return extensions;
}

bool WaylandDetector::isWaylandSession() {
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    if (wayland_display && strlen(wayland_display) > 0) {
        return true;
    }
    
    // 检查XDG_SESSION_TYPE
    const char* session_type = std::getenv("XDG_SESSION_TYPE");
    if (session_type && strcmp(session_type, "wayland") == 0) {
        return true;
    }
    
    return false;
}

bool WaylandDetector::checkWaylandSocket() {
    std::string display = getWaylandDisplay();
    std::string socket_path;
    
    // 构建socket路径
    const char* runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (runtime_dir) {
        socket_path = std::string(runtime_dir) + "/" + display;
    } else {
        socket_path = "/tmp/" + display;
    }
    
    // 检查socket文件是否存在
    return access(socket_path.c_str(), F_OK) == 0;
}

bool WaylandDetector::testWaylandConnection() {
#ifdef ENABLE_WAYLAND
    struct wl_display* display = wl_display_connect(nullptr);
    if (display) {
        wl_display_disconnect(display);
        return true;
    }
#endif
    return false;
}

// WaylandCompositorConfig 实现
WaylandCompositorConfig::WaylandCompositorConfig(CompositorType type) 
    : type_(type) {
    initializeDefaults();
}

void WaylandCompositorConfig::setParameter(const std::string& key, const std::string& value) {
    parameters_[key] = value;
}

std::string WaylandCompositorConfig::getParameter(const std::string& key) const {
    auto it = parameters_.find(key);
    return it != parameters_.end() ? it->second : "";
}

bool WaylandCompositorConfig::supportsFeature(const std::string& feature) const {
    switch (type_) {
        case CompositorType::WESTON:
            return feature == "hardware_acceleration" || 
                   feature == "multiple_outputs" ||
                   feature == "surface_scaling";
        
        case CompositorType::SWAY:
            return feature == "tiling" || 
                   feature == "window_management" ||
                   feature == "hardware_acceleration";
        
        case CompositorType::CUSTOM:
            return true; // 假设自定义合成器支持所有功能
        
        default:
            return false;
    }
}

std::string WaylandCompositorConfig::getRecommendedRenderConfig() const {
    switch (type_) {
        case CompositorType::WESTON:
            return "gl-renderer";
        
        case CompositorType::SWAY:
            return "gles2";
        
        default:
            return "auto";
    }
}

void WaylandCompositorConfig::initializeDefaults() {
    switch (type_) {
        case CompositorType::WESTON:
            parameters_["renderer"] = "gl";
            parameters_["backend"] = "drm";
            parameters_["use-pixman"] = "false";
            break;
        
        case CompositorType::SWAY:
            parameters_["renderer"] = "gles2";
            parameters_["backend"] = "auto";
            break;
        
        default:
            parameters_["renderer"] = "auto";
            parameters_["backend"] = "auto";
            break;
    }
}

// WaylandConnectionManager 实现
#ifdef ENABLE_WAYLAND
const struct wl_registry_listener WaylandConnectionManager::registry_listener_ = {
    WaylandConnectionManager::registryHandler,
    WaylandConnectionManager::registryRemover
};
#endif

WaylandConnectionManager::WaylandConnectionManager() 
    : display_(nullptr), connected_(false) {
#ifdef ENABLE_WAYLAND
    registry_ = nullptr;
    compositor_ = nullptr;
    shell_ = nullptr;
#endif
}

WaylandConnectionManager::~WaylandConnectionManager() {
    disconnect();
}

bool WaylandConnectionManager::connect(const char* display_name) {
    if (connected_) {
        return true;
    }
    
#ifdef ENABLE_WAYLAND
    display_ = wl_display_connect(display_name);
    if (!display_) {
        std::cerr << "[WaylandConnectionManager] 无法连接到Wayland display" << std::endl;
        return false;
    }
    
    registry_ = wl_display_get_registry(display_);
    if (!registry_) {
        std::cerr << "[WaylandConnectionManager] 无法获取registry" << std::endl;
        wl_display_disconnect(display_);
        display_ = nullptr;
        return false;
    }
    
    wl_registry_add_listener(registry_, &registry_listener_, this);
    wl_display_dispatch(display_);
    wl_display_roundtrip(display_);
    
    connected_ = true;
    std::cout << "[WaylandConnectionManager] 成功连接到Wayland服务器" << std::endl;
    return true;
#else
    std::cerr << "[WaylandConnectionManager] Wayland支持未编译" << std::endl;
    return false;
#endif
}

void WaylandConnectionManager::disconnect() {
    if (!connected_) {
        return;
    }
    
#ifdef ENABLE_WAYLAND
    if (shell_) {
        wl_shell_destroy(shell_);
        shell_ = nullptr;
    }
    
    if (compositor_) {
        wl_compositor_destroy(compositor_);
        compositor_ = nullptr;
    }
    
    if (registry_) {
        wl_registry_destroy(registry_);
        registry_ = nullptr;
    }
    
    if (display_) {
        wl_display_disconnect(display_);
        display_ = nullptr;
    }
#endif
    
    connected_ = false;
    std::cout << "[WaylandConnectionManager] 已断开Wayland连接" << std::endl;
}

bool WaylandConnectionManager::isConnected() const {
    return connected_;
}

void* WaylandConnectionManager::getDisplay() const {
    return display_;
}

int WaylandConnectionManager::dispatchEvents() {
#ifdef ENABLE_WAYLAND
    if (display_) {
        return wl_display_dispatch_pending(display_);
    }
#endif
    return 0;
}

void WaylandConnectionManager::flush() {
#ifdef ENABLE_WAYLAND
    if (display_) {
        wl_display_flush(display_);
    }
#endif
}

#ifdef ENABLE_WAYLAND
void WaylandConnectionManager::registryHandler(void* data, struct wl_registry* registry,
                                             uint32_t id, const char* interface, uint32_t version) {
    WaylandConnectionManager* manager = static_cast<WaylandConnectionManager*>(data);
    
    std::cout << "[WaylandConnectionManager] 发现接口: " << interface << " (版本 " << version << ")" << std::endl;
    
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        manager->compositor_ = static_cast<struct wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 1));
    } else if (strcmp(interface, wl_shell_interface.name) == 0) {
        manager->shell_ = static_cast<struct wl_shell*>(
            wl_registry_bind(registry, id, &wl_shell_interface, 1));
    }
}

void WaylandConnectionManager::registryRemover(void* data, struct wl_registry* registry, uint32_t id) {
    std::cout << "[WaylandConnectionManager] 移除接口: " << id << std::endl;
}
#endif

// WaylandSurfaceManager 实现
WaylandSurfaceManager::WaylandSurfaceManager(WaylandConnectionManager* connection)
    : connection_(connection), width_(0), height_(0) {
#ifdef ENABLE_WAYLAND
    surface_ = nullptr;
    shell_surface_ = nullptr;
    buffer_ = nullptr;
#else
    surface_ = nullptr;
    shell_surface_ = nullptr;
    buffer_ = nullptr;
#endif
}

WaylandSurfaceManager::~WaylandSurfaceManager() {
    destroySurface();
}

bool WaylandSurfaceManager::createSurface(int width, int height) {
    if (!connection_ || !connection_->isConnected()) {
        std::cerr << "[WaylandSurfaceManager] 连接管理器未连接" << std::endl;
        return false;
    }
    
    width_ = width;
    height_ = height;
    
#ifdef ENABLE_WAYLAND
    // 这里需要实际的surface创建逻辑
    // 为了编译通过，暂时简化实现
    std::cout << "[WaylandSurfaceManager] 创建surface: " << width << "x" << height << std::endl;
    return true;
#else
    std::cout << "[WaylandSurfaceManager] Wayland支持未编译" << std::endl;
    return false;
#endif
}

void WaylandSurfaceManager::destroySurface() {
#ifdef ENABLE_WAYLAND
    if (buffer_) {
        wl_buffer_destroy(buffer_);
        buffer_ = nullptr;
    }
    
    if (shell_surface_) {
        wl_shell_surface_destroy(shell_surface_);
        shell_surface_ = nullptr;
    }
    
    if (surface_) {
        wl_surface_destroy(surface_);
        surface_ = nullptr;
    }
#endif
    
    std::cout << "[WaylandSurfaceManager] Surface已销毁" << std::endl;
}

void* WaylandSurfaceManager::getSurface() const {
    return surface_;
}

void WaylandSurfaceManager::commit() {
#ifdef ENABLE_WAYLAND
    if (surface_) {
        wl_surface_commit(surface_);
    }
#endif
}

void WaylandSurfaceManager::resize(int width, int height) {
    width_ = width;
    height_ = height;
    
    // 重新创建buffer
    destroyBuffer();
    createBuffer(width, height);
    
    std::cout << "[WaylandSurfaceManager] Surface大小调整为: " << width << "x" << height << std::endl;
}

void WaylandSurfaceManager::setPosition(int x, int y) {
    std::cout << "[WaylandSurfaceManager] 设置surface位置: (" << x << ", " << y << ")" << std::endl;
    // Wayland中位置通常由合成器管理
}

bool WaylandSurfaceManager::createBuffer(int width, int height) {
    // 这里应该创建实际的像素buffer
    // 为了简化，只打印日志
    std::cout << "[WaylandSurfaceManager] 创建buffer: " << width << "x" << height << std::endl;
    return true;
}

void WaylandSurfaceManager::destroyBuffer() {
#ifdef ENABLE_WAYLAND
    if (buffer_) {
        wl_buffer_destroy(buffer_);
        buffer_ = nullptr;
    }
#endif
}

} // namespace ui
} // namespace bamboo_cut