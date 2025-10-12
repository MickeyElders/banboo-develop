/**
 * @file drm_resource_coordinator.h
 * @brief DRM资源协调器 - 解决LVGL和DeepStream单进程内DRM资源冲突
 * 
 * 设计目标：
 * - LVGL使用Primary Plane渲染UI界面
 * - DeepStream使用Overlay Plane渲染视频流
 * - 两层硬件合成，无性能损失
 * - 避免"No space left on device"错误
 */

#pragma once

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>
#include <memory>

namespace bamboo_cut {
namespace drm {

/**
 * @brief DRM资源分配信息
 */
struct ResourceAllocation {
    uint32_t crtc_id;           ///< CRTC ID
    uint32_t connector_id;      ///< 连接器ID  
    uint32_t plane_id;          ///< Plane ID
    bool is_primary;            ///< 是否为Primary Plane
    
    ResourceAllocation() 
        : crtc_id(0), connector_id(0), plane_id(0), is_primary(false) {}
    
    /**
     * @brief 检查资源分配是否有效
     */
    bool isValid() const {
        return crtc_id > 0 && connector_id > 0 && plane_id > 0;
    }
    
    /**
     * @brief 打印资源分配信息
     */
    void print() const {
        std::cout << "  CRTC: " << crtc_id << std::endl;
        std::cout << "  Connector: " << connector_id << std::endl;
        std::cout << "  Plane: " << plane_id 
                  << " (" << (is_primary ? "Primary" : "Overlay") << ")" 
                  << std::endl;
    }
};

/**
 * @brief DRM资源使用状态
 */
enum class ResourceStatus {
    FREE,           ///< 空闲可用
    OCCUPIED_LVGL,  ///< 被LVGL占用
    OCCUPIED_DS,    ///< 被DeepStream占用
    OCCUPIED_OTHER  ///< 被其他进程占用
};

/**
 * @brief Plane类型信息
 */
struct PlaneInfo {
    uint32_t plane_id;
    uint32_t plane_type;        ///< 0=Overlay, 1=Primary, 2=Cursor
    uint32_t possible_crtcs;    ///< 支持的CRTC位掩码
    ResourceStatus status;
    
    PlaneInfo() : plane_id(0), plane_type(0), possible_crtcs(0), status(ResourceStatus::FREE) {}
};

/**
 * @brief DRM资源协调器
 * 
 * 单例模式，线程安全，负责协调LVGL和DeepStream的DRM资源使用
 */
class DRMResourceCoordinator {
public:
    /**
     * @brief 获取单例实例
     */
    static DRMResourceCoordinator* getInstance();
    
    /**
     * @brief 在LVGL初始化前进行预初始化
     * 
     * 扫描所有可用的DRM资源，但不立即获取Master权限
     * 提前规划资源分配策略
     * 
     * @return true 初始化成功，false 失败
     */
    bool initializeBeforeLVGL();
    
    /**
     * @brief 注册LVGL使用的DRM资源
     * 
     * 在LVGL成功获取DRM Master后调用，记录LVGL实际使用的资源
     * 
     * @param lvgl_drm_fd LVGL的DRM文件描述符（可选，如果LVGL使用独立FD）
     * @return true 注册成功，false 失败或有冲突
     */
    bool registerLVGLResources(int lvgl_drm_fd = -1);
    
    /**
     * @brief 为DeepStream分配Overlay Plane资源
     * 
     * 分配未被LVGL占用的Overlay Plane，确保与LVGL共享CRTC
     * 
     * @param allocation 输出参数，包含分配的资源信息
     * @return true 分配成功，false 无可用资源
     */
    bool allocateOverlayForDeepStream(ResourceAllocation& allocation);
    
    /**
     * @brief 检查资源冲突
     * 
     * 验证LVGL和DeepStream没有使用相同的Plane
     * 允许共享CRTC（用于硬件分层合成）
     * 
     * @return true 无冲突，false 有冲突
     */
    bool checkResourceConflict();
    
    /**
     * @brief 获取共享的DRM文件描述符
     * 
     * 如果LVGL和DeepStream可以共享同一个DRM FD，返回该FD
     * 
     * @return DRM文件描述符，-1表示不可用
     */
    int getSharedDrmFd() const { return shared_drm_fd_; }
    
    /**
     * @brief 释放DeepStream资源
     * 
     * 当DeepStream停止时释放其占用的Overlay Plane
     */
    void releaseDeepStreamResources();
    
    /**
     * @brief 获取系统DRM状态诊断信息
     */
    void printSystemDRMState();

private:
    // 单例相关
    DRMResourceCoordinator();
    ~DRMResourceCoordinator();
    DRMResourceCoordinator(const DRMResourceCoordinator&) = delete;
    DRMResourceCoordinator& operator=(const DRMResourceCoordinator&) = delete;
    
    /**
     * @brief 扫描DRM设备和资源
     */
    bool scanDRMResources();
    
    /**
     * @brief 检测Primary Plane
     */
    uint32_t detectPrimaryPlane(int drm_fd, uint32_t crtc_id);
    
    /**
     * @brief 检测可用的Overlay Plane
     */
    std::vector<uint32_t> detectOverlayPlanes(int drm_fd, uint32_t crtc_id);
    
    /**
     * @brief 获取Plane类型
     */
    uint32_t getPlaneType(int drm_fd, uint32_t plane_id);
    
    /**
     * @brief 检查Plane是否被占用
     */
    ResourceStatus checkPlaneStatus(int drm_fd, uint32_t plane_id);
    
    /**
     * @brief 查找可用的DRM设备
     */
    int findDRMDevice();
    
    /**
     * @brief 验证资源分配策略
     */
    bool validateResourcePlan();

private:
    static std::unique_ptr<DRMResourceCoordinator> instance_;
    static std::mutex instance_mutex_;
    
    // 线程安全保护
    mutable std::mutex resource_mutex_;
    
    // DRM设备信息
    int shared_drm_fd_;                    ///< 共享的DRM文件描述符
    std::string drm_driver_name_;          ///< DRM驱动名称
    
    // 系统资源信息
    std::vector<uint32_t> available_crtcs_;
    std::vector<uint32_t> available_connectors_;
    std::vector<PlaneInfo> plane_info_;
    
    // 资源分配状态
    ResourceAllocation lvgl_allocation_;   ///< LVGL资源分配
    ResourceAllocation ds_allocation_;     ///< DeepStream资源分配
    bool lvgl_registered_;                 ///< LVGL是否已注册
    bool ds_allocated_;                    ///< DeepStream是否已分配
    
    // 初始化状态
    bool initialized_;
    bool resources_scanned_;
};

} // namespace drm
} // namespace bamboo_cut