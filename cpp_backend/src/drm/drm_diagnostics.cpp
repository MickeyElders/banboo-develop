/**
 * @file drm_diagnostics.cpp
 * @brief DRM资源诊断工具实现 - 运行时DRM资源监控和验证
 */

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <iostream>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <fstream>

namespace bamboo_cut {
namespace drm {

/**
 * @brief DRM诊断工具类
 */
class DRMDiagnostics {
public:
    /**
     * @brief 简单的ResourceAllocation结构（避免依赖外部头文件）
     */
    struct ResourceAllocation {
        uint32_t crtc_id;
        uint32_t connector_id;
        uint32_t plane_id;
        bool is_primary;
        
        ResourceAllocation() : crtc_id(0), connector_id(0), plane_id(0), is_primary(false) {}
    };

    /**
     * @brief 打印系统DRM状态
     */
    static void printSystemDRMState(int drm_fd) {
        std::cout << "\n========== DRM系统状态 ==========" << std::endl;
        
        if (drm_fd < 0) {
            std::cout << "❌ DRM设备未打开" << std::endl;
            return;
        }
        
        // 1. 打印DRM驱动信息
        printDRMDriverInfo(drm_fd);
        
        // 2. 打印所有CRTC
        printCRTCInfo(drm_fd);
        
        // 3. 打印所有Plane及其占用状态
        printPlaneInfo(drm_fd);
        
        // 4. 打印连接器信息
        printConnectorInfo(drm_fd);
        
        std::cout << "================================\n" << std::endl;
    }
    
    /**
     * @brief 验证资源分配
     */
    static void verifyResourceAllocation(
        const ResourceAllocation& lvgl_res,
        const ResourceAllocation& ds_res) {
        
        std::cout << "\n========== 资源分配验证 ==========" << std::endl;
        
        std::cout << "LVGL资源:" << std::endl;
        printResourceAllocation(lvgl_res);
        
        std::cout << "\nDeepStream资源:" << std::endl;
        printResourceAllocation(ds_res);
        
        // 冲突检查
        bool has_conflict = false;
        if (lvgl_res.plane_id == ds_res.plane_id && 
            lvgl_res.plane_id != 0 && ds_res.plane_id != 0) {
            std::cout << "\n❌ 冲突检测: Plane ID重复!" << std::endl;
            has_conflict = true;
        }
        
        if (lvgl_res.crtc_id == ds_res.crtc_id && 
            lvgl_res.crtc_id != 0 && ds_res.crtc_id != 0) {
            std::cout << "\n✅ 共享CRTC: " << lvgl_res.crtc_id 
                      << " (多层渲染模式)" << std::endl;
        }
        
        if (!has_conflict) {
            std::cout << "\n✅ 资源分配无冲突" << std::endl;
        }
        
        std::cout << "================================\n" << std::endl;
    }
    
    /**
     * @brief 检查特定Plane的状态
     */
    static bool checkPlaneStatus(int drm_fd, uint32_t plane_id, const std::string& name) {
        drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
        if (!plane) {
            std::cout << "❌ " << name << " Plane " << plane_id << " 不存在" << std::endl;
            return false;
        }
        
        bool is_occupied = (plane->crtc_id != 0 || plane->fb_id != 0);
        
        std::cout << "📋 " << name << " Plane " << plane_id << ": "
                  << (is_occupied ? "🔴 占用中" : "🟢 空闲")
                  << " (CRTC: " << plane->crtc_id << ", FB: " << plane->fb_id << ")" << std::endl;
        
        drmModeFreePlane(plane);
        return true;
    }
    
    /**
     * @brief 监控DRM资源使用情况（定期调用）
     */
    static void monitorResourceUsage(int drm_fd) {
        static int monitor_count = 0;
        monitor_count++;
        
        std::cout << "\n🔍 DRM资源监控 #" << monitor_count << " :" << std::endl;
        
        // 统计各类型Plane的使用情况
        int total_planes = 0;
        int occupied_planes = 0;
        int primary_planes = 0;
        int overlay_planes = 0;
        int cursor_planes = 0;
        
        drmModePlaneRes* plane_res = drmModeGetPlaneResources(drm_fd);
        if (plane_res) {
            total_planes = plane_res->count_planes;
            
            for (uint32_t i = 0; i < plane_res->count_planes; i++) {
                uint32_t plane_id = plane_res->planes[i];
                drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
                
                if (plane) {
                    if (plane->crtc_id != 0 || plane->fb_id != 0) {
                        occupied_planes++;
                    }
                    
                    // 获取Plane类型
                    uint32_t plane_type = getPlaneType(drm_fd, plane_id);
                    if (plane_type == 0) overlay_planes++;
                    else if (plane_type == 1) primary_planes++;
                    else if (plane_type == 2) cursor_planes++;
                    
                    drmModeFreePlane(plane);
                }
            }
            drmModeFreePlaneResources(plane_res);
        }
        
        std::cout << "  📊 Plane统计: 总数=" << total_planes 
                  << ", 占用=" << occupied_planes 
                  << ", 空闲=" << (total_planes - occupied_planes) << std::endl;
        std::cout << "  📊 类型分布: Primary=" << primary_planes
                  << ", Overlay=" << overlay_planes
                  << ", Cursor=" << cursor_planes << std::endl;
        
        // 检查内存使用情况
        checkMemoryUsage();
    }

private:
    /**
     * @brief 打印DRM驱动信息
     */
    static void printDRMDriverInfo(int drm_fd) {
        drmVersion* version = drmGetVersion(drm_fd);
        if (version) {
            std::cout << "\n🔧 DRM驱动信息:" << std::endl;
            std::cout << "  名称: " << version->name << std::endl;
            std::cout << "  版本: " << version->version_major << "."
                      << version->version_minor << "." << version->version_patchlevel << std::endl;
            std::cout << "  描述: " << (version->desc ? version->desc : "N/A") << std::endl;
            drmFreeVersion(version);
        }
    }
    
    /**
     * @brief 打印CRTC信息
     */
    static void printCRTCInfo(int drm_fd) {
        drmModeRes* resources = drmModeGetResources(drm_fd);
        if (!resources) return;
        
        std::cout << "\n📺 CRTCs (" << resources->count_crtcs << "):" << std::endl;
        for (int i = 0; i < resources->count_crtcs; i++) {
            uint32_t crtc_id = resources->crtcs[i];
            drmModeCrtc* crtc = drmModeGetCrtc(drm_fd, crtc_id);
            if (crtc) {
                std::cout << "  CRTC " << std::setw(3) << crtc_id << ": "
                          << (crtc->mode_valid ? "✅ 激活" : "⚪ 未激活");
                
                if (crtc->mode_valid) {
                    std::cout << " " << crtc->mode.hdisplay << "x" << crtc->mode.vdisplay
                              << "@" << crtc->mode.vrefresh << "Hz";
                }
                
                std::cout << " (FB: " << crtc->buffer_id << ")" << std::endl;
                drmModeFreeCrtc(crtc);
            }
        }
        drmModeFreeResources(resources);
    }
    
    /**
     * @brief 打印Plane信息
     */
    static void printPlaneInfo(int drm_fd) {
        drmModePlaneRes* plane_res = drmModeGetPlaneResources(drm_fd);
        if (!plane_res) return;
        
        std::cout << "\n🎨 Planes (" << plane_res->count_planes << "):" << std::endl;
        for (uint32_t i = 0; i < plane_res->count_planes; i++) {
            uint32_t plane_id = plane_res->planes[i];
            drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
            if (plane) {
                // 获取Plane类型
                uint32_t plane_type = getPlaneType(drm_fd, plane_id);
                std::string type_str = "Unknown";
                if (plane_type == 0) type_str = "Overlay";
                else if (plane_type == 1) type_str = "Primary";
                else if (plane_type == 2) type_str = "Cursor";
                
                std::cout << "  Plane " << std::setw(3) << plane_id << " ["
                          << std::setw(8) << type_str << "]: ";
                
                if (plane->crtc_id != 0 || plane->fb_id != 0) {
                    std::cout << "🔴 占用中 (CRTC: " << plane->crtc_id 
                              << ", FB: " << plane->fb_id << ")";
                    
                    // 尝试识别占用者
                    if (plane_type == 1) {
                        std::cout << " <- 可能是LVGL";
                    } else if (plane_type == 0) {
                        std::cout << " <- 可能是DeepStream";
                    }
                } else {
                    std::cout << "🟢 空闲";
                }
                std::cout << std::endl;
                
                drmModeFreePlane(plane);
            }
        }
        drmModeFreePlaneResources(plane_res);
    }
    
    /**
     * @brief 打印连接器信息
     */
    static void printConnectorInfo(int drm_fd) {
        drmModeRes* resources = drmModeGetResources(drm_fd);
        if (!resources) return;
        
        std::cout << "\n🔌 连接器 (" << resources->count_connectors << "):" << std::endl;
        for (int i = 0; i < resources->count_connectors; i++) {
            uint32_t conn_id = resources->connectors[i];
            drmModeConnector* connector = drmModeGetConnector(drm_fd, conn_id);
            if (connector) {
                std::cout << "  Connector " << std::setw(3) << conn_id << ": ";
                
                if (connector->connection == DRM_MODE_CONNECTED) {
                    std::cout << "🟢 已连接 (" << connector->count_modes << " 模式)";
                    if (connector->count_modes > 0) {
                        auto& mode = connector->modes[0];
                        std::cout << " 主模式: " << mode.hdisplay << "x" << mode.vdisplay
                                  << "@" << mode.vrefresh << "Hz";
                    }
                } else if (connector->connection == DRM_MODE_DISCONNECTED) {
                    std::cout << "🔴 未连接";
                } else {
                    std::cout << "⚪ 状态未知";
                }
                std::cout << std::endl;
                
                drmModeFreeConnector(connector);
            }
        }
        drmModeFreeResources(resources);
    }
    
    /**
     * @brief 获取Plane类型
     */
    static uint32_t getPlaneType(int drm_fd, uint32_t plane_id) {
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
    
    /**
     * @brief 打印资源分配信息
     */
    static void printResourceAllocation(const ResourceAllocation& alloc) {
        if (alloc.crtc_id == 0 && alloc.connector_id == 0 && alloc.plane_id == 0) {
            std::cout << "  ⚪ 未分配" << std::endl;
            return;
        }
        
        std::cout << "  CRTC: " << alloc.crtc_id << std::endl;
        std::cout << "  Connector: " << alloc.connector_id << std::endl;
        std::cout << "  Plane: " << alloc.plane_id 
                  << " (" << (alloc.is_primary ? "Primary" : "Overlay") << ")" 
                  << std::endl;
    }
    
    /**
     * @brief 检查内存使用情况
     */
    static void checkMemoryUsage() {
        // 简单的内存检查
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            long total_mem = 0, available_mem = 0;
            
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") == 0) {
                    sscanf(line.c_str(), "MemTotal: %ld kB", &total_mem);
                } else if (line.find("MemAvailable:") == 0) {
                    sscanf(line.c_str(), "MemAvailable: %ld kB", &available_mem);
                }
            }
            meminfo.close();
            
            if (total_mem > 0 && available_mem > 0) {
                double memory_usage = 1.0 - (double)available_mem / total_mem;
                std::cout << "  💾 系统内存使用率: " << std::fixed << std::setprecision(1) 
                          << (memory_usage * 100) << "%" << std::endl;
                
                if (memory_usage > 0.9) {
                    std::cout << "  ⚠️  内存使用率过高，可能影响DRM性能" << std::endl;
                }
            }
        }
    }
};

} // namespace drm
} // namespace bamboo_cut

// 导出C接口用于外部调用
extern "C" {
    /**
     * @brief C接口：打印DRM系统状态
     */
    void drm_print_system_state(int drm_fd) {
        bamboo_cut::drm::DRMDiagnostics::printSystemDRMState(drm_fd);
    }
    
    /**
     * @brief C接口：监控资源使用情况
     */
    void drm_monitor_resource_usage(int drm_fd) {
        bamboo_cut::drm::DRMDiagnostics::monitorResourceUsage(drm_fd);
    }
    
    /**
     * @brief C接口：检查Plane状态
     */
    int drm_check_plane_status(int drm_fd, uint32_t plane_id, const char* name) {
        return bamboo_cut::drm::DRMDiagnostics::checkPlaneStatus(drm_fd, plane_id, std::string(name)) ? 1 : 0;
    }
}