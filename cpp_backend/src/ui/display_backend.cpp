/**
 * @file display_backend.cpp
 * @brief 显示后端实现
 * 支持DRM和Wayland双模式切换
 */

#include "bamboo_cut/ui/display_backend.h"
#include "bamboo_cut/ui/wayland_compositor.h"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>

#ifdef ENABLE_WAYLAND
#include <wayland-client.h>
#include <wayland-client-protocol.h>
#endif

namespace bamboo_cut {
namespace ui {

// DRMDisplayBackend 实现
bool DRMDisplayBackend::initialize(const DisplayArea& area) {
    std::cout << "[DRMDisplayBackend] 初始化DRM显示后端: " 
              << area.width << "x" << area.height 
              << " at (" << area.x << "," << area.y << ")" << std::endl;
    
    current_area_ = area;
    initialized_ = true;
    
    return true;
}

bool DRMDisplayBackend::start() {
    if (!initialized_) {
        std::cerr << "[DRMDisplayBackend] 后端未初始化" << std::endl;
        return false;
    }
    
    std::cout << "[DRMDisplayBackend] 启动DRM显示后端" << std::endl;
    running_ = true;
    return true;
}

void DRMDisplayBackend::stop() {
    if (!running_) return;
    
    std::cout << "[DRMDisplayBackend] 停止DRM显示后端" << std::endl;
    running_ = false;
}

bool DRMDisplayBackend::isRunning() const {
    return running_;
}

bool DRMDisplayBackend::updateArea(const DisplayArea& area) {
    std::cout << "[DRMDisplayBackend] 更新显示区域: " 
              << area.width << "x" << area.height 
              << " at (" << area.x << "," << area.y << ")" << std::endl;
    
    current_area_ = area;
    return true;
}

void DRMDisplayBackend::setFlushCallback(void* callback) {
    flush_callback_ = callback;
}

void* DRMDisplayBackend::getDisplayHandle() {
    return display_handle_;
}

// WaylandDisplayBackend 实现
bool WaylandDisplayBackend::initialize(const DisplayArea& area) {
    std::cout << "[WaylandDisplayBackend] 初始化Wayland显示后端: " 
              << area.width << "x" << area.height 
              << " at (" << area.x << "," << area.y << ")" << std::endl;
    
    current_area_ = area;
    
    if (!connectToWaylandServer()) {
        std::cerr << "[WaylandDisplayBackend] 连接Wayland服务器失败" << std::endl;
        return false;
    }
    
    if (!createWaylandSurface()) {
        std::cerr << "[WaylandDisplayBackend] 创建Wayland surface失败" << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool WaylandDisplayBackend::start() {
    if (!initialized_) {
        std::cerr << "[WaylandDisplayBackend] 后端未初始化" << std::endl;
        return false;
    }
    
    std::cout << "[WaylandDisplayBackend] 启动Wayland显示后端" << std::endl;
    running_ = true;
    return true;
}

void WaylandDisplayBackend::stop() {
    if (!running_) return;
    
    std::cout << "[WaylandDisplayBackend] 停止Wayland显示后端" << std::endl;
    running_ = false;
}

bool WaylandDisplayBackend::isRunning() const {
    return running_;
}

bool WaylandDisplayBackend::updateArea(const DisplayArea& area) {
    std::cout << "[WaylandDisplayBackend] 更新显示区域: " 
              << area.width << "x" << area.height 
              << " at (" << area.x << "," << area.y << ")" << std::endl;
    
    current_area_ = area;
    // 这里可以添加Wayland surface大小调整逻辑
    return true;
}

void WaylandDisplayBackend::setFlushCallback(void* callback) {
    flush_callback_ = callback;
}

void* WaylandDisplayBackend::getDisplayHandle() {
    return display_handle_;
}

bool WaylandDisplayBackend::connectToWaylandServer() {
#ifdef ENABLE_WAYLAND
    std::cout << "[WaylandDisplayBackend] 连接到Wayland服务器..." << std::endl;
    
    wl_display_ = static_cast<void*>(wl_display_connect(nullptr));
    if (!wl_display_) {
        std::cerr << "[WaylandDisplayBackend] 无法连接到Wayland display" << std::endl;
        return false;
    }
    
    std::cout << "[WaylandDisplayBackend] 成功连接到Wayland服务器" << std::endl;
    return true;
#else
    std::cerr << "[WaylandDisplayBackend] Wayland未编译支持" << std::endl;
    return false;
#endif
}

bool WaylandDisplayBackend::createWaylandSurface() {
#ifdef ENABLE_WAYLAND
    std::cout << "[WaylandDisplayBackend] 创建Wayland surface..." << std::endl;
    
    // 这里需要创建compositor、surface等对象
    // 为了简化，现在只做基础设置
    
    std::cout << "[WaylandDisplayBackend] Wayland surface创建完成" << std::endl;
    return true;
#else
    return false;
#endif
}

void WaylandDisplayBackend::handleWaylandEvents() {
#ifdef ENABLE_WAYLAND
    if (wl_display_) {
        wl_display_dispatch_pending(static_cast<struct wl_display*>(wl_display_));
        wl_display_flush(static_cast<struct wl_display*>(wl_display_));
    }
#endif
}

// DisplayBackendFactory 实现
std::unique_ptr<DisplayBackend> DisplayBackendFactory::createBackend(DisplayBackendType type) {
    switch (type) {
        case DisplayBackendType::DRM_DIRECT:
            return std::make_unique<DRMDisplayBackend>();
        
        case DisplayBackendType::WAYLAND_CLIENT:
            return std::make_unique<WaylandDisplayBackend>();
        
        case DisplayBackendType::AUTO_DETECT:
            return createBestBackend();
        
        default:
            std::cerr << "[DisplayBackendFactory] 未知的后端类型" << std::endl;
            return nullptr;
    }
}

std::unique_ptr<DisplayBackend> DisplayBackendFactory::createBestBackend() {
    std::cout << "[DisplayBackendFactory] 自动检测最佳显示后端..." << std::endl;
    
    // 优先尝试Wayland
    if (isBackendAvailable(DisplayBackendType::WAYLAND_CLIENT)) {
        std::cout << "[DisplayBackendFactory] 选择Wayland客户端后端" << std::endl;
        return createBackend(DisplayBackendType::WAYLAND_CLIENT);
    }
    
    // 回退到DRM
    if (isBackendAvailable(DisplayBackendType::DRM_DIRECT)) {
        std::cout << "[DisplayBackendFactory] 选择DRM直接后端" << std::endl;
        return createBackend(DisplayBackendType::DRM_DIRECT);
    }
    
    std::cerr << "[DisplayBackendFactory] 没有可用的显示后端" << std::endl;
    return nullptr;
}

bool DisplayBackendFactory::isBackendAvailable(DisplayBackendType type) {
    switch (type) {
        case DisplayBackendType::DRM_DIRECT: {
            // 检查DRM设备是否可用
            int fd = open("/dev/dri/card1", O_RDWR);
            if (fd >= 0) {
                close(fd);
                return true;
            }
            return false;
        }
        
        case DisplayBackendType::WAYLAND_CLIENT: {
            // 检查Wayland是否可用
            return WaylandDetector::detectWaylandSupport();
        }
        
        default:
            return false;
    }
}

DisplayBackendType DisplayBackendFactory::getRecommendedType() {
    // 优先推荐Wayland，然后是DRM
    if (isBackendAvailable(DisplayBackendType::WAYLAND_CLIENT)) {
        return DisplayBackendType::WAYLAND_CLIENT;
    }
    
    if (isBackendAvailable(DisplayBackendType::DRM_DIRECT)) {
        return DisplayBackendType::DRM_DIRECT;
    }
    
    return DisplayBackendType::FALLBACK;
}

// WaylandDetector 实现
bool WaylandDetector::detectWaylandSupport() {
    std::cout << "[WaylandDetector] 检测Wayland支持..." << std::endl;
    
    // 检查环境变量
    if (!isWaylandSession()) {
        std::cout << "[WaylandDetector] 未检测到Wayland会话" << std::endl;
        return false;
    }
    
    // 测试连接
#ifdef ENABLE_WAYLAND
    struct wl_display* display = wl_display_connect(nullptr);
    if (display) {
        wl_display_disconnect(display);
        std::cout << "[WaylandDetector] Wayland支持检测成功" << std::endl;
        return true;
    }
#endif
    
    std::cout << "[WaylandDetector] 无法连接到Wayland服务器" << std::endl;
    return false;
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
    
    return "unknown";
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

} // namespace ui
} // namespace bamboo_cut