/**
 * @file drm_resource_coordinator.cpp
 * @brief DRM资源协调器实现 - 解决LVGL和DeepStream单进程内DRM资源冲突
 */

#include "bamboo_cut/drm/drm_resource_coordinator.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <iomanip>

namespace bamboo_cut {
namespace drm {

// 静态成员定义
std::unique_ptr<DRMResourceCoordinator> DRMResourceCoordinator::instance_;
std::mutex DRMResourceCoordinator::instance_mutex_;

DRMResourceCoordinator::DRMResourceCoordinator()
    : shared_drm_fd_(-1)
    , lvgl_registered_(false)
    , ds_allocated_(false)
    , initialized_(false)
    , resources_scanned_(false) {
    std::cout << "🔧 [DRM协调器] 构造函数调用" << std::endl;
}

DRMResourceCoordinator::~DRMResourceCoordinator() {
    std::cout << "🔧 [DRM协调器] 析构函数调用" << std::endl;
    
    if (shared_drm_fd_ >= 0) {
        close(shared_drm_fd_);
        shared_drm_fd_ = -1;
    }
}

DRMResourceCoordinator* DRMResourceCoordinator::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (!instance_) {
        instance_ = std::unique_ptr<DRMResourceCoordinator>(new DRMResourceCoordinator());
    }
    
    return instance_.get();
}

bool DRMResourceCoordinator::initializeBeforeLVGL() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    std::cout << "🔧 [DRM协调器] 在LVGL之前初始化..." << std::endl;
    
    if (initialized_) {
        std::cout << "✅ [DRM协调器] 已经初始化，跳过" << std::endl;
        return true;
    }
    
    // 1. 查找并打开DRM设备
    shared_drm_fd_ = findDRMDevice();
    if (shared_drm_fd_ < 0) {
        std::cerr << "❌ [DRM协调器] 无法找到可用的DRM设备" << std::endl;
        return false;
    }
    
    std::cout << "✅ DRM设备已打开，FD=" << shared_drm_fd_ << std::endl;
    
    // 2. 检查DRM Master权限
    if (drmSetMaster(shared_drm_fd_) != 0) {
        std::cout << "⚠️  [DRM协调器] 暂时无法获取DRM Master权限（正常，LVGL会稍后获取）" << std::endl;
    } else {
        std::cout << "✅ DRM Master权限" << std::endl;
        // 立即释放，让LVGL获取
        drmDropMaster(shared_drm_fd_);
    }
    
    // 3. 扫描DRM资源
    if (!scanDRMResources()) {
        std::cerr << "❌ [DRM协调器] DRM资源扫描失败" << std::endl;
        return false;
    }
    
    // 4. 验证资源分配策略
    if (!validateResourcePlan()) {
        std::cerr << "❌ [DRM协调器] 资源分配策略验证失败" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "✅ [DRM协调器] 初始化完成" << std::endl;
    
    return true;
}

bool DRMResourceCoordinator::registerLVGLResources(int lvgl_drm_fd) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    std::cout << "🔧 [DRM协调器] 注册LVGL资源..." << std::endl;
    
    if (lvgl_registered_) {
        std::cout << "⚠️  LVGL资源已注册，跳过" << std::endl;
        return true;
    }
    
    // 如果LVGL使用了独立的DRM FD，更新共享FD
    if (lvgl_drm_fd >= 0 && lvgl_drm_fd != shared_drm_fd_) {
        std::cout << "📝 LVGL使用独立DRM FD: " << lvgl_drm_fd << std::endl;
        // 可以选择切换到LVGL的FD，或保持当前FD
    }
    
    // 重新扫描资源，识别LVGL占用的资源
    if (!scanDRMResources()) {
        std::cerr << "❌ 重新扫描DRM资源失败" << std::endl;
        return false;
    }
    
    // 查找被LVGL占用的Primary Plane
    int use_fd = (lvgl_drm_fd >= 0) ? lvgl_drm_fd : shared_drm_fd_;
    
    for (const auto& crtc_id : available_crtcs_) {
        uint32_t primary_plane = detectPrimaryPlane(use_fd, crtc_id);
        if (primary_plane > 0) {
            // 检查这个Primary Plane是否被占用
            ResourceStatus status = checkPlaneStatus(use_fd, primary_plane);
            if (status == ResourceStatus::OCCUPIED_LVGL || status == ResourceStatus::OCCUPIED_OTHER) {
                // 假设被占用的Primary Plane是LVGL使用的
                lvgl_allocation_.plane_id = primary_plane;
                lvgl_allocation_.crtc_id = crtc_id;
                lvgl_allocation_.is_primary = true;
                
                // 查找对应的connector
                for (const auto& conn_id : available_connectors_) {
                    lvgl_allocation_.connector_id = conn_id;
                    break; // 使用第一个可用的connector
                }
                
                std::cout << "📌 检测到LVGL资源分配:" << std::endl;
                lvgl_allocation_.print();
                break;
            }
        }
    }
    
    if (!lvgl_allocation_.isValid()) {
        std::cerr << "❌ 无法检测到有效的LVGL资源分配" << std::endl;
        return false;
    }
    
    lvgl_registered_ = true;
    std::cout << "✅ LVGL资源注册完成，无冲突" << std::endl;
    
    return true;
}

bool DRMResourceCoordinator::allocateOverlayForDeepStream(ResourceAllocation& allocation) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    std::cout << "🔧 [DRM协调器] 为DeepStream分配Overlay..." << std::endl;
    
    if (!initialized_ || !lvgl_registered_) {
        std::cerr << "❌ 协调器未正确初始化或LVGL未注册" << std::endl;
        return false;
    }
    
    if (ds_allocated_) {
        allocation = ds_allocation_;
        std::cout << "✅ DeepStream资源已分配，返回现有分配" << std::endl;
        return true;
    }
    
    // 查找可用的Overlay Plane，与LVGL共享相同的CRTC
    uint32_t target_crtc = lvgl_allocation_.crtc_id;
    std::vector<uint32_t> overlay_planes = detectOverlayPlanes(shared_drm_fd_, target_crtc);
    
    for (uint32_t overlay_id : overlay_planes) {
        // 确保这个Overlay Plane没有被占用
        ResourceStatus status = checkPlaneStatus(shared_drm_fd_, overlay_id);
        if (status == ResourceStatus::FREE) {
            // 找到可用的Overlay Plane
            ds_allocation_.plane_id = overlay_id;
            ds_allocation_.crtc_id = target_crtc;
            ds_allocation_.connector_id = lvgl_allocation_.connector_id; // 共享connector
            ds_allocation_.is_primary = false;
            
            allocation = ds_allocation_;
            ds_allocated_ = true;
            
            std::cout << "✅ 为DeepStream分配Overlay: Plane=" << overlay_id << std::endl;
            allocation.print();
            
            return true;
        }
    }
    
    std::cerr << "❌ 未找到可用的Overlay Plane" << std::endl;
    return false;
}

bool DRMResourceCoordinator::checkResourceConflict() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    if (!lvgl_registered_ || !ds_allocated_) {
        return true; // 如果任一方未分配，则无冲突
    }
    
    // 检查Plane冲突
    if (lvgl_allocation_.plane_id == ds_allocation_.plane_id) {
        std::cerr << "❌ 资源冲突: Plane ID重复!" << std::endl;
        return false;
    }
    
    // 允许共享CRTC和Connector（多层渲染模式）
    if (lvgl_allocation_.crtc_id == ds_allocation_.crtc_id) {
        std::cout << "✅ 共享CRTC: " << lvgl_allocation_.crtc_id << " (多层渲染模式)" << std::endl;
    }
    
    std::cout << "✅ 资源分配无冲突" << std::endl;
    return true;
}

void DRMResourceCoordinator::releaseDeepStreamResources() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    if (ds_allocated_) {
        std::cout << "🔧 [DRM协调器] 释放DeepStream资源" << std::endl;
        
        // 重置分配状态
        ds_allocation_ = ResourceAllocation();
        ds_allocated_ = false;
        
        std::cout << "✅ DeepStream资源已释放" << std::endl;
    }
}

void DRMResourceCoordinator::printSystemDRMState() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    std::cout << "\n========== DRM系统状态 ==========" << std::endl;
    
    if (shared_drm_fd_ < 0) {
        std::cout << "❌ DRM设备未打开" << std::endl;
        return;
    }
    
    // 打印所有CRTC
    drmModeRes* resources = drmModeGetResources(shared_drm_fd_);
    if (resources) {
        std::cout << "\n📺 CRTCs (" << resources->count_crtcs << "):" << std::endl;
        for (int i = 0; i < resources->count_crtcs; i++) {
            uint32_t crtc_id = resources->crtcs[i];
            drmModeCrtc* crtc = drmModeGetCrtc(shared_drm_fd_, crtc_id);
            if (crtc) {
                std::cout << "  CRTC " << crtc_id << ": " 
                          << (crtc->mode_valid ? "激活" : "未激活")
                          << " (FB: " << crtc->buffer_id << ")" << std::endl;
                drmModeFreeCrtc(crtc);
            }
        }
        drmModeFreeResources(resources);
    }
    
    // 打印所有Plane及其占用状态
    drmModePlaneRes* plane_res = drmModeGetPlaneResources(shared_drm_fd_);
    if (plane_res) {
        std::cout << "\n🎨 Planes (" << plane_res->count_planes << "):" << std::endl;
        for (uint32_t i = 0; i < plane_res->count_planes; i++) {
            uint32_t plane_id = plane_res->planes[i];
            drmModePlane* plane = drmModeGetPlane(shared_drm_fd_, plane_id);
            if (plane) {
                // 获取Plane类型
                uint32_t plane_type = getPlaneType(shared_drm_fd_, plane_id);
                std::string type_str = "Unknown";
                if (plane_type == 0) type_str = "Overlay";
                else if (plane_type == 1) type_str = "Primary";
                else if (plane_type == 2) type_str = "Cursor";
                
                std::cout << "  Plane " << std::setw(3) << plane_id << " ["
                          << std::setw(8) << type_str << "]: ";
                
                if (plane->crtc_id != 0 || plane->fb_id != 0) {
                    std::cout << "🔴 占用中 (CRTC: " << plane->crtc_id 
                              << ", FB: " << plane->fb_id << ")";
                } else {
                    std::cout << "🟢 空闲";
                }
                std::cout << std::endl;
                
                drmModeFreePlane(plane);
            }
        }
        drmModeFreePlaneResources(plane_res);
    }
    
    std::cout << "================================\n" << std::endl;
}

// ===== 私有方法实现 =====

bool DRMResourceCoordinator::scanDRMResources() {
    if (shared_drm_fd_ < 0) {
        return false;
    }
    
    std::cout << "🔍 [DRM协调器] 扫描DRM资源..." << std::endl;
    
    // 获取DRM驱动信息
    drmVersion* version = drmGetVersion(shared_drm_fd_);
    if (version) {
        drm_driver_name_ = std::string(version->name);
        std::cout << "📋 DRM驱动: " << drm_driver_name_ << " v" 
                  << version->version_major << "." << version->version_minor 
                  << "." << version->version_patchlevel << std::endl;
        drmFreeVersion(version);
    }
    
    // 获取基本资源
    drmModeRes* resources = drmModeGetResources(shared_drm_fd_);
    if (!resources) {
        std::cerr << "❌ 无法获取DRM资源" << std::endl;
        return false;
    }
    
    // 收集CRTC信息
    available_crtcs_.clear();
    for (int i = 0; i < resources->count_crtcs; i++) {
        available_crtcs_.push_back(resources->crtcs[i]);
    }
    
    // 收集连接器信息
    available_connectors_.clear();
    for (int i = 0; i < resources->count_connectors; i++) {
        drmModeConnector* connector = drmModeGetConnector(shared_drm_fd_, resources->connectors[i]);
        if (connector && connector->connection == DRM_MODE_CONNECTED) {
            available_connectors_.push_back(resources->connectors[i]);
        }
        if (connector) drmModeFreeConnector(connector);
    }
    
    drmModeFreeResources(resources);
    
    // 收集Plane信息
    plane_info_.clear();
    drmModePlaneRes* plane_res = drmModeGetPlaneResources(shared_drm_fd_);
    if (plane_res) {
        for (uint32_t i = 0; i < plane_res->count_planes; i++) {
            PlaneInfo info;
            info.plane_id = plane_res->planes[i];
            info.plane_type = getPlaneType(shared_drm_fd_, info.plane_id);
            info.status = checkPlaneStatus(shared_drm_fd_, info.plane_id);
            
            drmModePlane* plane = drmModeGetPlane(shared_drm_fd_, info.plane_id);
            if (plane) {
                info.possible_crtcs = plane->possible_crtcs;
                drmModeFreePlane(plane);
            }
            
            plane_info_.push_back(info);
        }
        drmModeFreePlaneResources(plane_res);
    }
    
    std::cout << "✅ DRM资源扫描完成：" << std::endl;
    std::cout << "  📌 LVGL资源: CRTC=" << (available_crtcs_.empty() ? 0 : available_crtcs_[0]) 
              << ", Plane=" << detectPrimaryPlane(shared_drm_fd_, available_crtcs_.empty() ? 0 : available_crtcs_[0]) << std::endl;
    
    auto overlay_planes = detectOverlayPlanes(shared_drm_fd_, available_crtcs_.empty() ? 0 : available_crtcs_[0]);
    if (!overlay_planes.empty()) {
        std::cout << "  📌 DeepStream资源: CRTC=" << (available_crtcs_.empty() ? 0 : available_crtcs_[0])
                  << ", Plane=" << overlay_planes[0] << std::endl;
    }
    
    resources_scanned_ = true;
    return true;
}

uint32_t DRMResourceCoordinator::detectPrimaryPlane(int drm_fd, uint32_t crtc_id) {
    for (const auto& info : plane_info_) {
        if (info.plane_type == 1) { // Primary plane type = 1
            // 检查这个plane是否支持目标CRTC
            int crtc_index = -1;
            for (size_t i = 0; i < available_crtcs_.size(); i++) {
                if (available_crtcs_[i] == crtc_id) {
                    crtc_index = static_cast<int>(i);
                    break;
                }
            }
            
            if (crtc_index >= 0 && (info.possible_crtcs & (1 << crtc_index))) {
                return info.plane_id;
            }
        }
    }
    return 0;
}

std::vector<uint32_t> DRMResourceCoordinator::detectOverlayPlanes(int drm_fd, uint32_t crtc_id) {
    std::vector<uint32_t> overlay_planes;
    
    for (const auto& info : plane_info_) {
        if (info.plane_type == 0 && info.status == ResourceStatus::FREE) { // Overlay plane type = 0
            // 检查这个plane是否支持目标CRTC
            int crtc_index = -1;
            for (size_t i = 0; i < available_crtcs_.size(); i++) {
                if (available_crtcs_[i] == crtc_id) {
                    crtc_index = static_cast<int>(i);
                    break;
                }
            }
            
            if (crtc_index >= 0 && (info.possible_crtcs & (1 << crtc_index))) {
                overlay_planes.push_back(info.plane_id);
            }
        }
    }
    
    return overlay_planes;
}

uint32_t DRMResourceCoordinator::getPlaneType(int drm_fd, uint32_t plane_id) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) {
        return 0; // 默认为Overlay
    }
    
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyRes* prop = drmModeGetProperty(drm_fd, props->props[i]);
        if (prop && strcmp(prop->name, "type") == 0) {
            uint64_t value = props->prop_values[i];
            drmModeFreeProperty(prop);
            drmModeFreeObjectProperties(props);
            return static_cast<uint32_t>(value);
        }
        if (prop) drmModeFreeProperty(prop);
    }
    
    drmModeFreeObjectProperties(props);
    return 0;
}

ResourceStatus DRMResourceCoordinator::checkPlaneStatus(int drm_fd, uint32_t plane_id) {
    drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
    if (!plane) {
        return ResourceStatus::OCCUPIED_OTHER;
    }
    
    bool is_occupied = (plane->crtc_id != 0 || plane->fb_id != 0);
    drmModeFreePlane(plane);
    
    if (!is_occupied) {
        return ResourceStatus::FREE;
    }
    
    // 简化判断：如果被占用且不是我们已知的DeepStream资源，假设是LVGL或其他
    if (ds_allocated_ && plane_id == ds_allocation_.plane_id) {
        return ResourceStatus::OCCUPIED_DS;
    }
    
    // 根据plane类型推断占用者
    uint32_t plane_type = getPlaneType(drm_fd, plane_id);
    if (plane_type == 1) { // Primary plane
        return ResourceStatus::OCCUPIED_LVGL;
    }
    
    return ResourceStatus::OCCUPIED_OTHER;
}

int DRMResourceCoordinator::findDRMDevice() {
    const char* drm_devices[] = {
        "/dev/dri/card0",
        "/dev/dri/card1",
    };
    
    for (const char* device : drm_devices) {
        int fd = open(device, O_RDWR);
        if (fd >= 0) {
            std::cout << "✅ 找到DRM设备: " << device << " (FD=" << fd << ")" << std::endl;
            return fd;
        }
    }
    
    std::cerr << "❌ 无法找到可用的DRM设备" << std::endl;
    return -1;
}

bool DRMResourceCoordinator::validateResourcePlan() {
    if (available_crtcs_.empty()) {
        std::cerr << "❌ 无可用CRTC" << std::endl;
        return false;
    }
    
    if (available_connectors_.empty()) {
        std::cerr << "❌ 无连接的显示器" << std::endl;
        return false;
    }
    
    // 检查是否有Primary Plane和Overlay Plane
    bool has_primary = false;
    bool has_overlay = false;
    
    for (const auto& info : plane_info_) {
        if (info.plane_type == 1) has_primary = true;
        if (info.plane_type == 0) has_overlay = true;
    }
    
    if (!has_primary) {
        std::cerr << "❌ 无可用Primary Plane" << std::endl;
        return false;
    }
    
    if (!has_overlay) {
        std::cout << "⚠️  无可用Overlay Plane，DeepStream将回退到AppSink模式" << std::endl;
    }
    
    std::cout << "✅ 资源分配策略验证通过" << std::endl;
    return true;
}

} // namespace drm
} // namespace bamboo_cut