/**
 * DRM诊断工具头文件
 * 提供运行时DRM资源诊断和监控功能
 */

#ifndef BAMBOO_CUT_DRM_DIAGNOSTICS_H
#define BAMBOO_CUT_DRM_DIAGNOSTICS_H

#include <cstdint>

// 前向声明，避免循环依赖
namespace bamboo_cut {
namespace drm {

class DRMResourceCoordinator;

/**
 * DRM诊断工具类
 * 提供系统状态监控和资源验证功能
 */
class DRMDiagnostics {
public:
    /**
     * 打印完整的DRM系统状态
     * @param drm_fd DRM设备文件描述符
     */
    static void printSystemDRMState(int drm_fd);
    
    /**
     * 验证资源分配是否存在冲突
     * @param lvgl_res LVGL资源分配
     * @param ds_res DeepStream资源分配
     */
    static void verifyResourceAllocation(
        const class DRMResourceCoordinator::ResourceAllocation& lvgl_res,
        const class DRMResourceCoordinator::ResourceAllocation& ds_res);
    
    /**
     * 监控运行时资源使用情况
     * @param drm_fd DRM设备文件描述符
     */
    static void monitorResourceUsage(int drm_fd);
    
    /**
     * 打印Plane详细信息
     * @param drm_fd DRM设备文件描述符
     * @param plane_id Plane ID
     */
    static void printPlaneDetails(int drm_fd, uint32_t plane_id);
    
    /**
     * 检查DRM设备是否可用
     * @return true如果DRM设备可用
     */
    static bool isDrmDeviceAvailable();

private:
    // 辅助方法
    static const char* getPlaneTypeName(uint64_t type);
    static void printCrtcStatus(int drm_fd, uint32_t crtc_id);
    static void printConnectorStatus(int drm_fd, uint32_t connector_id);
};

} // namespace drm
} // namespace bamboo_cut

#endif // BAMBOO_CUT_DRM_DIAGNOSTICS_H