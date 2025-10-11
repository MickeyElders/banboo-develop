/**
 * @file gbm_display_backend.cpp
 * @brief GBM (Generic Buffer Management) 显示后端实现
 * 实现LVGL和GStreamer共享DRM资源的分层显示协调机制
 */

#include "bamboo_cut/ui/gbm_display_backend.h"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <sys/mman.h>
#include <drm_fourcc.h>

namespace bamboo_cut {
namespace ui {

// GBMDisplayBackend 实现
GBMDisplayBackend::GBMDisplayBackend()
    : drm_fd_(-1)
    , drm_resources_(nullptr)
    , connector_(nullptr)
    , gbm_device_(nullptr)
    , gbm_surface_(nullptr)
    , egl_display_(EGL_NO_DISPLAY)
    , egl_context_(EGL_NO_CONTEXT)
    , egl_surface_(EGL_NO_SURFACE)
    , initialized_(false) {
}

GBMDisplayBackend::~GBMDisplayBackend() {
    cleanup();
}

bool GBMDisplayBackend::initialize(const DRMSharedConfig& config) {
    std::lock_guard<std::mutex> lock(drm_mutex_);
    
    std::cout << "🔧 初始化GBM显示后端 (DRM资源共享协调器)..." << std::endl;
    
    config_ = config;
    
    try {
        // 1. 打开DRM设备
        if (!openDRMDevice()) {
            std::cerr << "❌ 打开DRM设备失败" << std::endl;
            return false;
        }
        
        // 2. 初始化GBM设备
        if (!initializeGBM()) {
            std::cerr << "❌ 初始化GBM设备失败" << std::endl;
            return false;
        }
        
        // 3. 设置DRM平面配置
        if (!setupDRMPlanes()) {
            std::cerr << "❌ 配置DRM平面失败" << std::endl;
            return false;
        }
        
        // 4. 初始化EGL (可选)
        if (!initializeEGL()) {
            std::cout << "⚠️ EGL初始化失败，继续使用CPU渲染" << std::endl;
        }
        
        // 5. 设置CRTC模式
        if (!setupCRTCMode()) {
            std::cerr << "❌ 设置CRTC模式失败" << std::endl;
            return false;
        }
        
        initialized_ = true;
        
        std::cout << "✅ GBM显示后端初始化完成" << std::endl;
        std::cout << "🎯 DRM资源协调配置：" << std::endl;
        std::cout << "  - DRM FD: " << drm_fd_ << std::endl;
        std::cout << "  - CRTC ID: " << config_.crtc_id << std::endl;
        std::cout << "  - Primary Plane: " << config_.primary_plane_id << " (LVGL)" << std::endl;
        std::cout << "  - Overlay Plane: " << config_.overlay_plane_id << " (GStreamer)" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ GBM后端初始化异常: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

bool GBMDisplayBackend::openDRMDevice() {
    std::cout << "🔧 打开nvidia-drm设备..." << std::endl;
    
    // 尝试打开nvidia-drm设备
    const char* drm_devices[] = {
        "/dev/dri/card1",  // 通常nvidia-drm在这个位置
        "/dev/dri/card0",
        nullptr
    };
    
    for (int i = 0; drm_devices[i]; i++) {
        drm_fd_ = open(drm_devices[i], O_RDWR | O_CLOEXEC);
        if (drm_fd_ >= 0) {
            // 验证是否是nvidia-drm设备
            drmVersionPtr version = drmGetVersion(drm_fd_);
            if (version) {
                std::cout << "🔍 检查DRM设备 " << drm_devices[i] << ": " << version->name << std::endl;
                if (strcmp(version->name, "nvidia-drm") == 0) {
                    std::cout << "✅ 找到nvidia-drm设备: " << drm_devices[i] << std::endl;
                    drmFreeVersion(version);
                    break;
                }
                drmFreeVersion(version);
            }
            close(drm_fd_);
            drm_fd_ = -1;
        }
    }
    
    if (drm_fd_ < 0) {
        std::cerr << "❌ 无法打开nvidia-drm设备" << std::endl;
        return false;
    }
    
    // 获取DRM资源
    drm_resources_ = drmModeGetResources(drm_fd_);
    if (!drm_resources_) {
        std::cerr << "❌ 无法获取DRM资源" << std::endl;
        return false;
    }
    
    std::cout << "📊 DRM资源统计:" << std::endl;
    std::cout << "  - CRTCs: " << drm_resources_->count_crtcs << std::endl;
    std::cout << "  - Connectors: " << drm_resources_->count_connectors << std::endl;
    std::cout << "  - Encoders: " << drm_resources_->count_encoders << std::endl;
    
    return true;
}

bool GBMDisplayBackend::initializeGBM() {
    std::cout << "🔧 初始化GBM设备..." << std::endl;
    
    gbm_device_ = gbm_create_device(drm_fd_);
    if (!gbm_device_) {
        std::cerr << "❌ 创建GBM设备失败" << std::endl;
        return false;
    }
    
    std::cout << "✅ GBM设备创建成功" << std::endl;
    return true;
}

bool GBMDisplayBackend::initializeEGL() {
    std::cout << "🔧 初始化EGL..." << std::endl;
    
    // 获取EGL显示
    egl_display_ = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm_device_, nullptr);
    if (egl_display_ == EGL_NO_DISPLAY) {
        std::cout << "⚠️ 无法获取EGL显示" << std::endl;
        return false;
    }
    
    // 初始化EGL
    if (!eglInitialize(egl_display_, nullptr, nullptr)) {
        std::cout << "⚠️ EGL初始化失败" << std::endl;
        egl_display_ = EGL_NO_DISPLAY;
        return false;
    }
    
    std::cout << "✅ EGL初始化成功" << std::endl;
    return true;
}

bool GBMDisplayBackend::setupDRMPlanes() {
    std::cout << "🔧 配置DRM平面分配..." << std::endl;
    
    // 如果配置中没有指定平面，自动检测
    if (config_.primary_plane_id == 0) {
        config_.primary_plane_id = findPrimaryPlane();
    }
    
    if (config_.overlay_plane_id == 0) {
        config_.overlay_plane_id = findOverlayPlane();
    }
    
    if (config_.connector_id == 0) {
        config_.connector_id = findConnector();
    }
    
    // 验证平面配置
    if (config_.primary_plane_id == 0) {
        std::cerr << "❌ 未找到可用的primary plane" << std::endl;
        return false;
    }
    
    if (config_.overlay_plane_id == 0) {
        std::cout << "⚠️ 未找到overlay plane，使用用户指定的plane-id=44" << std::endl;
        config_.overlay_plane_id = 44;  // 用户指定的overlay plane
    }
    
    if (config_.connector_id == 0) {
        std::cerr << "❌ 未找到可用的连接器" << std::endl;
        return false;
    }
    
    std::cout << "🎯 DRM平面分配完成:" << std::endl;
    std::cout << "  - Primary Plane (LVGL): " << config_.primary_plane_id << std::endl;
    std::cout << "  - Overlay Plane (GStreamer): " << config_.overlay_plane_id << std::endl;
    std::cout << "  - Connector: " << config_.connector_id << std::endl;
    
    return true;
}

uint32_t GBMDisplayBackend::findConnector() {
    std::cout << "🔍 查找活跃连接器..." << std::endl;
    
    for (int i = 0; i < drm_resources_->count_connectors; i++) {
        drmModeConnector* connector = drmModeGetConnector(drm_fd_, drm_resources_->connectors[i]);
        if (connector) {
            if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
                uint32_t connector_id = connector->connector_id;
                std::cout << "✅ 找到活跃连接器: " << connector_id << std::endl;
                
                // 保存连接器信息用于模式设置
                connector_ = connector;
                mode_ = connector->modes[0];  // 使用首选模式
                
                config_.width = mode_.hdisplay;
                config_.height = mode_.vdisplay;
                
                std::cout << "📺 显示模式: " << config_.width << "x" << config_.height 
                         << "@" << mode_.vrefresh << "Hz" << std::endl;
                
                return connector_id;
            }
            if (connector != connector_) {
                drmModeFreeConnector(connector);
            }
        }
    }
    
    return 0;
}

uint32_t GBMDisplayBackend::findPrimaryPlane() {
    std::cout << "🔍 查找primary plane..." << std::endl;
    
    drmModePlaneRes* plane_resources = drmModeGetPlaneResources(drm_fd_);
    if (!plane_resources) {
        std::cerr << "❌ 无法获取plane资源" << std::endl;
        return 0;
    }
    
    for (uint32_t i = 0; i < plane_resources->count_planes; i++) {
        uint32_t plane_id = plane_resources->planes[i];
        drmModePlane* plane = drmModeGetPlane(drm_fd_, plane_id);
        
        if (plane) {
            // 检查plane类型
            drmModeObjectProperties* props = drmModeObjectGetProperties(drm_fd_, plane_id, DRM_MODE_OBJECT_PLANE);
            if (props) {
                for (uint32_t j = 0; j < props->count_props; j++) {
                    drmModePropertyRes* prop = drmModeGetProperty(drm_fd_, props->props[j]);
                    if (prop && strcmp(prop->name, "type") == 0) {
                        uint64_t plane_type = props->prop_values[j];
                        if (plane_type == 1) {  // DRM_PLANE_TYPE_PRIMARY
                            std::cout << "✅ 找到primary plane: " << plane_id << std::endl;
                            drmModeFreeProperty(prop);
                            drmModeFreeObjectProperties(props);
                            drmModeFreePlane(plane);
                            drmModeFreePlaneResources(plane_resources);
                            return plane_id;
                        }
                        drmModeFreeProperty(prop);
                        break;
                    }
                    if (prop) drmModeFreeProperty(prop);
                }
                drmModeFreeObjectProperties(props);
            }
            drmModeFreePlane(plane);
        }
    }
    
    drmModeFreePlaneResources(plane_resources);
    return 0;
}

uint32_t GBMDisplayBackend::findOverlayPlane() {
    std::cout << "🔍 查找overlay plane..." << std::endl;
    
    drmModePlaneRes* plane_resources = drmModeGetPlaneResources(drm_fd_);
    if (!plane_resources) {
        std::cerr << "❌ 无法获取plane资源" << std::endl;
        return 0;
    }
    
    for (uint32_t i = 0; i < plane_resources->count_planes; i++) {
        uint32_t plane_id = plane_resources->planes[i];
        drmModePlane* plane = drmModeGetPlane(drm_fd_, plane_id);
        
        if (plane) {
            // 跳过已被占用的plane
            if (plane->crtc_id > 0 || plane->fb_id > 0) {
                drmModeFreePlane(plane);
                continue;
            }
            
            // 检查plane类型
            drmModeObjectProperties* props = drmModeObjectGetProperties(drm_fd_, plane_id, DRM_MODE_OBJECT_PLANE);
            if (props) {
                for (uint32_t j = 0; j < props->count_props; j++) {
                    drmModePropertyRes* prop = drmModeGetProperty(drm_fd_, props->props[j]);
                    if (prop && strcmp(prop->name, "type") == 0) {
                        uint64_t plane_type = props->prop_values[j];
                        if (plane_type == 0) {  // DRM_PLANE_TYPE_OVERLAY
                            std::cout << "✅ 找到overlay plane: " << plane_id << std::endl;
                            drmModeFreeProperty(prop);
                            drmModeFreeObjectProperties(props);
                            drmModeFreePlane(plane);
                            drmModeFreePlaneResources(plane_resources);
                            return plane_id;
                        }
                        drmModeFreeProperty(prop);
                        break;
                    }
                    if (prop) drmModeFreeProperty(prop);
                }
                drmModeFreeObjectProperties(props);
            }
            drmModeFreePlane(plane);
        }
    }
    
    drmModeFreePlaneResources(plane_resources);
    return 0;
}

bool GBMDisplayBackend::setupCRTCMode() {
    std::cout << "🔧 设置CRTC模式..." << std::endl;
    
    if (!connector_) {
        std::cerr << "❌ 连接器未初始化" << std::endl;
        return false;
    }
    
    // 查找CRTC
    if (config_.crtc_id == 0) {
        for (int i = 0; i < drm_resources_->count_crtcs; i++) {
            config_.crtc_id = drm_resources_->crtcs[i];
            break;  // 使用第一个可用的CRTC
        }
    }
    
    if (config_.crtc_id == 0) {
        std::cerr << "❌ 未找到可用的CRTC" << std::endl;
        return false;
    }
    
    std::cout << "🎯 CRTC配置: " << config_.crtc_id << std::endl;
    
    // 注意：在共享模式下，我们不立即设置CRTC模式
    // 而是让LVGL在初始化时设置，我们只是预留资源
    std::cout << "✅ CRTC模式配置完成（共享模式）" << std::endl;
    
    return true;
}

GBMFramebuffer* GBMDisplayBackend::createLVGLFramebuffer(uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(drm_mutex_);
    
    if (!initialized_) {
        std::cerr << "❌ GBM后端未初始化" << std::endl;
        return nullptr;
    }
    
    std::cout << "🔧 为LVGL创建framebuffer: " << width << "x" << height << std::endl;
    
    // 创建GBM buffer object
    gbm_bo* bo = gbm_bo_create(gbm_device_, width, height, GBM_FORMAT_XRGB8888, 
                               GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!bo) {
        std::cerr << "❌ 创建GBM buffer object失败" << std::endl;
        return nullptr;
    }
    
    // 获取DRM framebuffer信息
    uint32_t handle = gbm_bo_get_handle(bo).u32;
    uint32_t stride = gbm_bo_get_stride(bo);
    uint32_t size = height * stride;
    
    // 创建DRM framebuffer
    uint32_t fb_id;
    int ret = drmModeAddFB(drm_fd_, width, height, 24, 32, stride, handle, &fb_id);
    if (ret) {
        std::cerr << "❌ 创建DRM framebuffer失败: " << strerror(-ret) << std::endl;
        gbm_bo_destroy(bo);
        return nullptr;
    }
    
    // 创建framebuffer对象
    auto* fb = new GBMFramebuffer();
    fb->fb_id = fb_id;
    fb->handle = handle;
    fb->stride = stride;
    fb->size = size;
    fb->bo = bo;
    fb->map = nullptr;  // 按需映射
    
    std::cout << "✅ LVGL framebuffer创建成功: fb_id=" << fb_id << std::endl;
    
    return fb;
}

bool GBMDisplayBackend::commitLVGLFrame(GBMFramebuffer* fb) {
    std::lock_guard<std::mutex> lock(drm_mutex_);
    
    if (!fb || !initialized_) {
        return false;
    }
    
    // 将framebuffer设置到primary plane
    int ret = drmModeSetPlane(drm_fd_, config_.primary_plane_id, config_.crtc_id, 
                              fb->fb_id, 0, 0, 0, config_.width, config_.height,
                              0, 0, config_.width << 16, config_.height << 16);
    
    if (ret) {
        std::cerr << "❌ 设置primary plane失败: " << strerror(-ret) << std::endl;
        return false;
    }
    
    return true;
}

void GBMDisplayBackend::releaseFramebuffer(GBMFramebuffer* fb) {
    if (!fb) return;
    
    std::lock_guard<std::mutex> lock(drm_mutex_);
    
    if (fb->map) {
        munmap(fb->map, fb->size);
    }
    
    if (fb->fb_id > 0) {
        drmModeRmFB(drm_fd_, fb->fb_id);
    }
    
    if (fb->bo) {
        gbm_bo_destroy(fb->bo);
    }
    
    delete fb;
}

void GBMDisplayBackend::waitForVSync() {
    if (drm_fd_ < 0 || config_.crtc_id == 0) return;
    
    drmVBlank vbl;
    vbl.request.type = DRM_VBLANK_RELATIVE;
    vbl.request.sequence = 1;
    
    drmWaitVBlank(drm_fd_, &vbl);
}

bool GBMDisplayBackend::isDRMResourceAvailable() {
    std::lock_guard<std::mutex> lock(drm_mutex_);
    return initialized_ && drm_fd_ >= 0 && gbm_device_ != nullptr;
}

void GBMDisplayBackend::cleanup() {
    std::lock_guard<std::mutex> lock(drm_mutex_);
    
    std::cout << "🔧 清理GBM显示后端资源..." << std::endl;
    
    if (egl_surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(egl_display_, egl_surface_);
        egl_surface_ = EGL_NO_SURFACE;
    }
    
    if (egl_context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(egl_display_, egl_context_);
        egl_context_ = EGL_NO_CONTEXT;
    }
    
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglTerminate(egl_display_);
        egl_display_ = EGL_NO_DISPLAY;
    }
    
    if (gbm_surface_) {
        gbm_surface_destroy(gbm_surface_);
        gbm_surface_ = nullptr;
    }
    
    if (gbm_device_) {
        gbm_device_destroy(gbm_device_);
        gbm_device_ = nullptr;
    }
    
    if (connector_) {
        drmModeFreeConnector(connector_);
        connector_ = nullptr;
    }
    
    if (drm_resources_) {
        drmModeFreeResources(drm_resources_);
        drm_resources_ = nullptr;
    }
    
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }
    
    initialized_ = false;
    std::cout << "✅ GBM显示后端资源清理完成" << std::endl;
}

// GBMBackendManager 单例实现
GBMBackendManager& GBMBackendManager::getInstance() {
    static GBMBackendManager instance;
    return instance;
}

bool GBMBackendManager::initialize(const DRMSharedConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (backend_) {
        std::cout << "⚠️ GBM后端已初始化" << std::endl;
        return true;
    }
    
    std::cout << "🚀 初始化GBM后端管理器..." << std::endl;
    
    backend_ = std::make_unique<GBMDisplayBackend>();
    if (!backend_->initialize(config)) {
        backend_.reset();
        return false;
    }
    
    shared_config_ = config;
    
    std::cout << "✅ GBM后端管理器初始化完成" << std::endl;
    std::cout << "🎯 现在LVGL和GStreamer可以安全共享DRM资源" << std::endl;
    
    return true;
}

} // namespace ui
} // namespace bamboo_cut