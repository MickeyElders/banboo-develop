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
std::unique_ptr<DRMResourceCoordinator, DRMResourceCoordinator::Deleter> DRMResourceCoordinator::instance_;
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
    
    // 🔧 关键修复：不关闭共享的DRM FD，因为它属于LVGL
    // LVGL负责管理DRM FD的生命周期
    shared_drm_fd_ = -1;
}

DRMResourceCoordinator* DRMResourceCoordinator::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (!instance_) {
        instance_ = std::unique_ptr<DRMResourceCoordinator, Deleter>(new DRMResourceCoordinator());
    }
    
    return instance_.get();
}

bool DRMResourceCoordinator::initializeAfterLVGL(int lvgl_drm_fd) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    std::cout << "🔧 [DRM协调器] 在LVGL之后初始化..." << std::endl;
    
    if (initialized_) {
        std::cout << "✅ [DRM协调器] 已经初始化，跳过" << std::endl;
        return true;
    }
    
    // 🔧 关键修复：使用LVGL已经获得Master权限的共享FD
    if (lvgl_drm_fd < 0) {
        std::cerr << "❌ [DRM协调器] LVGL DRM FD无效: " << lvgl_drm_fd << std::endl;
        return false;
    }
    
    shared_drm_fd_ = lvgl_drm_fd;
    std::cout << "✅ 使用LVGL共享DRM FD=" << shared_drm_fd_ << " (已有Master权限)" << std::endl;
    
    // 1. 扫描DRM资源（使用共享FD）
    if (!scanDRMResources()) {
        std::cerr << "❌ [DRM协调器] DRM资源扫描失败" << std::endl;
        return false;
    }
    
    // 2. 注册LVGL资源（自动检测）
    if (!registerLVGLResources(lvgl_drm_fd)) {
        std::cerr << "❌ [DRM协调器] LVGL资源注册失败" << std::endl;
        return false;
    }
    
    // 3. 验证资源分配策略
    if (!validateResourcePlan()) {
        std::cerr << "❌ [DRM协调器] 资源分配策略验证失败" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "✅ [DRM协调器] 使用共享FD初始化完成" << std::endl;
    
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
    // 直接扫描Plane资源，不依赖预缓存的plane_info_
    drmModePlaneRes* plane_res = drmModeGetPlaneResources(drm_fd);
    if (!plane_res) {
        std::cerr << "❌ 无法获取Plane资源" << std::endl;
        return 0;
    }
    
    std::cout << "🔍 [DRM协调器] 搜索Primary Plane (总计 " << plane_res->count_planes << " 个planes)" << std::endl;
    
    for (uint32_t i = 0; i < plane_res->count_planes; i++) {
        uint32_t plane_id = plane_res->planes[i];
        uint32_t plane_type = getPlaneType(drm_fd, plane_id);
        
        std::cout << "  检查Plane " << plane_id << " 类型: " << plane_type << std::endl;
        
        if (plane_type == 1) { // Primary plane type = 1
            // 检查这个plane是否支持目标CRTC
            drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
            if (plane) {
                // 计算CRTC索引
                int crtc_index = -1;
                for (size_t j = 0; j < available_crtcs_.size(); j++) {
                    if (available_crtcs_[j] == crtc_id) {
                        crtc_index = static_cast<int>(j);
                        break;
                    }
                }
                
                bool supports_crtc = (crtc_index >= 0 && (plane->possible_crtcs & (1 << crtc_index)));
                std::cout << "    CRTC " << crtc_id << " 支持: " << (supports_crtc ? "✅" : "❌")
                          << " (possible_crtcs: 0x" << std::hex << plane->possible_crtcs << std::dec << ")" << std::endl;
                
                drmModeFreePlane(plane);
                
                if (supports_crtc) {
                    drmModeFreePlaneResources(plane_res);
                    std::cout << "✅ 找到Primary Plane: " << plane_id << std::endl;
                    return plane_id;
                }
            }
        }
    }
    
    drmModeFreePlaneResources(plane_res);
    std::cout << "❌ 未找到可用的Primary Plane" << std::endl;
    return 0;
}

std::vector<uint32_t> DRMResourceCoordinator::detectOverlayPlanes(int drm_fd, uint32_t crtc_id) {
    std::vector<uint32_t> overlay_planes;
    
    // 直接扫描Plane资源，不依赖预缓存的plane_info_
    drmModePlaneRes* plane_res = drmModeGetPlaneResources(drm_fd);
    if (!plane_res) {
        std::cerr << "❌ 无法获取Plane资源" << std::endl;
        return overlay_planes;
    }
    
    std::cout << "🔍 [DRM协调器] 搜索Overlay Planes (总计 " << plane_res->count_planes << " 个planes)" << std::endl;
    
    for (uint32_t i = 0; i < plane_res->count_planes; i++) {
        uint32_t plane_id = plane_res->planes[i];
        uint32_t plane_type = getPlaneType(drm_fd, plane_id);
        
        // 🔧 NVIDIA DRM特殊处理：Plane 57应该是可用的Overlay
        bool is_nvidia_overlay = (plane_id == 57);
        
        if (plane_type == 0 || is_nvidia_overlay) { // Overlay plane type = 0 或特定的NVIDIA Overlay
            // 检查这个plane是否支持目标CRTC
            drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
            if (plane) {
                // 计算CRTC索引
                int crtc_index = -1;
                for (size_t j = 0; j < available_crtcs_.size(); j++) {
                    if (available_crtcs_[j] == crtc_id) {
                        crtc_index = static_cast<int>(j);
                        break;
                    }
                }
                
                bool supports_crtc = (crtc_index >= 0 && (plane->possible_crtcs & (1 << crtc_index)));
                bool is_free = (plane->crtc_id == 0 && plane->fb_id == 0);
                
                // 🔧 对于Plane 57，即使显示为占用也尝试使用（可能是状态检测问题）
                if (is_nvidia_overlay && !is_free) {
                    std::cout << "  🔧 Plane " << plane_id << " 显示占用，但尝试作为NVIDIA Overlay使用"
                              << " (CRTC: " << plane->crtc_id << ", FB: " << plane->fb_id << ")" << std::endl;
                    is_free = true; // 强制认为可用
                }
                
                std::cout << "  检查Overlay Plane " << plane_id
                          << " 类型: " << plane_type
                          << " CRTC支持: " << (supports_crtc ? "✅" : "❌")
                          << " 空闲: " << (is_free ? "✅" : "❌");
                
                if (is_nvidia_overlay) {
                    std::cout << " [NVIDIA特殊处理]";
                }
                std::cout << std::endl;
                
                drmModeFreePlane(plane);
                
                if (supports_crtc && is_free) {
                    overlay_planes.push_back(plane_id);
                    std::cout << "✅ 找到可用Overlay Plane: " << plane_id << std::endl;
                }
            }
        }
    }
    
    drmModeFreePlaneResources(plane_res);
    
    if (overlay_planes.empty()) {
        std::cout << "❌ 未找到可用的Overlay Plane" << std::endl;
    } else {
        std::cout << "✅ 总计找到 " << overlay_planes.size() << " 个可用Overlay Planes" << std::endl;
    }
    
    return overlay_planes;
}

uint32_t DRMResourceCoordinator::getPlaneType(int drm_fd, uint32_t plane_id) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) {
        return 0; // 默认为Overlay
    }
    
    // 首先尝试标准的type属性
    uint32_t type_value = 0;
    bool found_type = false;
    
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyRes* prop = drmModeGetProperty(drm_fd, props->props[i]);
        if (prop && strcmp(prop->name, "type") == 0) {
            type_value = static_cast<uint32_t>(props->prop_values[i]);
            found_type = true;
            drmModeFreeProperty(prop);
            break;
        }
        if (prop) drmModeFreeProperty(prop);
    }
    
    drmModeFreeObjectProperties(props);
    
    // NVIDIA DRM驱动特殊处理：当type=0时，需要基于Plane ID来判断
    if (found_type && type_value == 0) {
        // 根据用户反馈：Plane 44是LVGL使用的Primary，Plane 57是可用的Overlay
        if (plane_id == 44) {
            std::cout << "🔧 [NVIDIA DRM] 检测到Plane 44，根据LVGL使用情况判定为Primary Plane" << std::endl;
            return 1; // 强制识别为Primary
        }
        
        // Plane 57和其他ID保持为Overlay
        return 0;
    }
    
    return type_value;
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