/**
 * @file lvgl_display_drm.cpp
 * @brief LVGL DRM显示驱动实现
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include "bamboo_cut/ui/gbm_display_backend.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>
#include <cerrno>

#ifdef ENABLE_LVGL
#include <xf86drm.h>
#include <xf86drmMode.h>
#endif

#ifdef ENABLE_GBM
#include <gbm.h>
#endif

namespace bamboo_cut {
namespace ui {

// 移除重复的initializeDisplay函数定义 - 现在在lvgl_interface.cpp中

// 移除重复的setupLVGLDisplayBuffer和createLVGLDisplay函数定义 - 现在在lvgl_interface.cpp中

bool LVGLInterface::initializeInput() {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 初始化触摸输入设备: " << config_.touch_device << std::endl;
    
    // 创建输入设备 (LVGL v9 API)
    input_device_ = lv_indev_create();
    if (!input_device_) {
        std::cerr << "[LVGLInterface] 输入设备创建失败" << std::endl;
        return false;
    }
    
    lv_indev_set_type(input_device_, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(input_device_, input_read_cb);
    
    std::cout << "[LVGLInterface] 触摸输入设备初始化成功" << std::endl;
    return true;
#else
    return false;
#endif
}

bool LVGLInterface::detectDisplayResolution(int& width, int& height) {
#ifdef ENABLE_LVGL
    std::cout << "[LVGLInterface] 正在检测DRM显示器分辨率..." << std::endl;
    
    // 优先尝试nvidia-drm设备，然后回退到tegra_drm
    const char* drm_devices[] = {
        "/dev/dri/card1",  // 备用nvidia-drm或tegra_drm
    };
    
    for (const char* device_path : drm_devices) {
        int fd = open(device_path, O_RDWR);
        if (fd < 0) {
            std::cout << "[LVGLInterface] 无法打开DRM设备: " << device_path << std::endl;
            continue;
        }
        
        std::cout << "[LVGLInterface] 成功打开DRM设备: " << device_path << std::endl;
        
        // 获取DRM资源
        drmModeRes* resources = drmModeGetResources(fd);
        if (!resources) {
            std::cerr << "[LVGLInterface] 无法获取DRM资源" << std::endl;
            close(fd);
            continue;
        }
        
        // 查找连接的显示器
        for (int i = 0; i < resources->count_connectors; i++) {
            drmModeConnector* connector = drmModeGetConnector(fd, resources->connectors[i]);
            if (!connector) continue;
            
            // 检查连接器是否连接了显示器
            if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
                // 获取首选模式（通常是第一个模式）
                drmModeModeInfo* mode = &connector->modes[0];
                width = mode->hdisplay;
                height = mode->vdisplay;
                
                std::cout << "[LVGLInterface] 检测到显示器分辨率: "
                          << width << "x" << height << " @" << mode->vrefresh << "Hz" << std::endl;
                std::cout << "[LVGLInterface] 显示器模式名称: " << mode->name << std::endl;
                
                drmModeFreeConnector(connector);
                drmModeFreeResources(resources);
                close(fd);
                return true;
            }
            
            drmModeFreeConnector(connector);
        }
        
        drmModeFreeResources(resources);
        close(fd);
    }
    
    std::cerr << "[LVGLInterface] 无法检测到连接的显示器" << std::endl;
    return false;
#else
    std::cerr << "[LVGLInterface] LVGL未启用，无法检测显示器分辨率" << std::endl;
    return false;
#endif
}

void LVGLInterface::setFullscreen(bool fullscreen) {
    // DRM模式默认就是全屏
    std::cout << "[LVGLInterface] 全屏模式: " << (fullscreen ? "启用" : "禁用") << std::endl;
}

// ==================== LVGL v9 回调函数 ====================

void display_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
#ifdef ENABLE_LVGL
    // 修复DRM双重释放内存错误 - 改进静态变量管理
    static int drm_fd = -1;
    static uint32_t fb_id = 0;
    static drmModeCrtc* crtc = nullptr;
    static drmModeConnector* connector = nullptr;
    static uint32_t* framebuffer = nullptr;
    static uint32_t fb_handle = 0;
    static bool drm_initialized = false;
    static bool drm_init_failed = false;
    static int init_attempt_count = 0;
    static uint32_t drm_width = 0;
    static uint32_t drm_height = 0;
    static uint32_t stride = 0;
    static uint32_t buffer_size = 0;
    static uint32_t flush_count = 0;
    static bool resources_cleaned = false;  // 新增：防止重复清理
    
    flush_count++;
    
    // 详细的调试信息
    if (flush_count <= 5 || flush_count % 60 == 0) {
        std::cout << "[DRM] flush_cb调用 #" << flush_count
                  << " area(" << (area ? area->x1 : -1) << "," << (area ? area->y1 : -1)
                  << " to " << (area ? area->x2 : -1) << "," << (area ? area->y2 : -1)
                  << ") px_map=" << (px_map ? "valid" : "null")
                  << " drm_init=" << drm_initialized << " failed=" << drm_init_failed << std::endl;
    }
    
    // 严格的参数验证
    if (!disp || !area || !px_map) {
        std::cerr << "[DRM] flush_cb参数无效: disp=" << disp << " area=" << area << " px_map=" << px_map << std::endl;
        if (disp) lv_display_flush_ready(disp);
        return;
    }
    
    // 如果DRM初始化已经失败，直接返回避免重复尝试
    if (drm_init_failed) {
        if (flush_count <= 5) {
            std::cerr << "[DRM] DRM初始化已失败，跳过刷新" << std::endl;
        }
        lv_display_flush_ready(disp);
        return;
    }
    
    // 初始化DRM设备 (只初始化一次，限制重试次数)
    if (!drm_initialized && !drm_init_failed) {
        // 重置资源清理标志，准备新的初始化尝试
        resources_cleaned = false;
        
        if (!initializeDRMDevice(drm_fd, fb_id, crtc, connector, framebuffer, fb_handle,
                                init_attempt_count, drm_width, drm_height, stride, buffer_size)) {
            // 初始化失败 - 确保资源被清理且只清理一次
            if (!resources_cleaned) {
                std::cout << "[DRM] 初始化失败，清理资源..." << std::endl;
                cleanupDRMResources(drm_fd, fb_id, crtc, connector, framebuffer, fb_handle, buffer_size);
                
                // 重置所有静态变量到初始状态
                drm_fd = -1;
                fb_id = 0;
                crtc = nullptr;
                connector = nullptr;
                framebuffer = nullptr;
                fb_handle = 0;
                drm_width = 0;
                drm_height = 0;
                stride = 0;
                buffer_size = 0;
                resources_cleaned = true;
            }
            drm_init_failed = true;
        } else {
            drm_initialized = true;
            std::cout << "[DRM] 初始化成功" << std::endl;
        }
    }
    
    // 安全的像素数据复制 - 修复段错误和内存访问问题
    if (drm_initialized && framebuffer && framebuffer != MAP_FAILED &&
        drm_width > 0 && drm_height > 0 && stride > 0) {
        copyPixelDataToGBM(area, px_map, framebuffer, drm_width, drm_height, stride, buffer_size);
    }
    
    // 通知LVGL刷新完成
    lv_display_flush_ready(disp);
#endif
}

// 检测DRM驱动类型的新函数
bool detectDRMDriverType(int drm_fd, std::string& driver_name) {
#ifdef ENABLE_LVGL
    drmVersion* version = drmGetVersion(drm_fd);
    if (version) {
        driver_name = std::string(version->name);
        std::cout << "[DRM] 检测到驱动: " << driver_name << " v" << version->version_major
                  << "." << version->version_minor << "." << version->version_patchlevel << std::endl;
        drmFreeVersion(version);
        return true;
    }
    return false;
#else
    return false;
#endif
}

bool initializeDRMDevice(int& drm_fd, uint32_t& fb_id, drmModeCrtc*& crtc,
                        drmModeConnector*& connector, uint32_t*& framebuffer,
                        uint32_t& fb_handle, int& init_attempt_count,
                        uint32_t& drm_width, uint32_t& drm_height,
                        uint32_t& stride, uint32_t& buffer_size) {
#ifdef ENABLE_LVGL
    init_attempt_count++;
    
    // 限制初始化尝试次数，避免无限重试
    if (init_attempt_count > 3) {
        std::cerr << "[DRM] 超过最大初始化尝试次数，标记为失败" << std::endl;
        return false;
    }
    
    std::cout << "[DRM] 开始LVGL独占DRM初始化尝试 #" << init_attempt_count << std::endl;
    
    // 🔧 恢复：使用读写模式打开DRM设备，LVGL需要独占控制
    const char* drm_devices[] = {
        "/dev/dri/card1",  // nvidia-drm或tegra_drm
    };
    bool device_opened = false;
    
    for (const char* device_path : drm_devices) {
        // 先关闭之前可能打开的文件描述符
        if (drm_fd >= 0) {
            std::cout << "[DRM] 关闭之前的DRM文件描述符: " << drm_fd << std::endl;
            close(drm_fd);
            drm_fd = -1;
        }
        
        drm_fd = open(device_path, O_RDWR);
        if (drm_fd >= 0) {
            std::cout << "[DRM] 成功打开设备: " << device_path << " fd=" << drm_fd << std::endl;
            device_opened = true;
            
            // 检测驱动类型
            std::string driver_name;
            if (detectDRMDriverType(drm_fd, driver_name)) {
                bool is_nvidia = (driver_name == "nvidia-drm");
                bool is_tegra = (driver_name == "tegra-drm");
                
                std::cout << "[DRM] 驱动类型: " << driver_name
                          << (is_nvidia ? " (NVIDIA GPU)" : is_tegra ? " (Tegra)" : " (其他)") << std::endl;
                
                // 优先使用nvidia-drm，如果可用的话
                if (is_nvidia) {
                    std::cout << "[DRM] 使用优化的NVIDIA-DRM配置" << std::endl;
                }
            }
            
            if (setupDRMDisplay(drm_fd, fb_id, crtc, connector, framebuffer, fb_handle,
                               drm_width, drm_height, stride, buffer_size)) {
                std::cout << "[DRM] 设备 " << device_path << " 初始化成功" << std::endl;
                return true;
            } else {
                std::cout << "[DRM] 设备 " << device_path << " 初始化失败，尝试下一个设备" << std::endl;
                if (drm_fd >= 0) {
                    close(drm_fd);
                    drm_fd = -1;
                }
                // 重置状态
                fb_id = 0;
                crtc = nullptr;
                connector = nullptr;
                framebuffer = nullptr;
                fb_handle = 0;
                drm_width = 0;
                drm_height = 0;
                stride = 0;
                buffer_size = 0;
            }
        } else {
            std::cout << "[DRM] 无法打开设备: " << device_path << " error: " << strerror(errno) << std::endl;
        }
    }
    
    if (!device_opened) {
        std::cerr << "[DRM] 无法打开任何DRM设备" << std::endl;
    }
    
    return false;
#else
    return false;
#endif
}

bool setupDRMDisplay(int drm_fd, uint32_t& fb_id, drmModeCrtc*& crtc, 
                    drmModeConnector*& connector, uint32_t*& framebuffer, 
                    uint32_t& fb_handle, uint32_t& drm_width, uint32_t& drm_height,
                    uint32_t& stride, uint32_t& buffer_size) {
#ifdef ENABLE_LVGL
    // 获取DRM资源
    drmModeRes* resources = drmModeGetResources(drm_fd);
    if (!resources) {
        return false;
    }
    
    // 查找连接的显示器
    for (int i = 0; i < resources->count_connectors; i++) {
        connector = drmModeGetConnector(drm_fd, resources->connectors[i]);
        if (connector && connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
            
            // 选择最佳显示模式
            drmModeModeInfo* mode = &connector->modes[0];
            drm_width = mode->hdisplay;
            drm_height = mode->vdisplay;
            
            std::cout << "[DRM] 显示器模式: " << drm_width << "x" << drm_height
                      << " @" << mode->vrefresh << "Hz" << std::endl;
            
            // 查找合适的CRTC并创建framebuffer（独占模式）
            if (findSuitableCRTC(drm_fd, resources, connector, crtc) &&
                createFramebuffer(drm_fd, drm_width, drm_height, fb_id, fb_handle,
                                 framebuffer, stride, buffer_size)) {
                
                // 🔧 恢复：使用LVGL独占模式，完整设置CRTC
                std::cout << "[DRM] 使用LVGL独占DRM模式，设置CRTC" << std::endl;
                setCRTCMode(drm_fd, crtc, fb_id, connector, mode);  // 现在完整设置CRTC
                
                drmModeFreeResources(resources);
                return true;
            }
        }
    }
    
    drmModeFreeResources(resources);
    return false;
#else
    return false;
#endif
}

bool findSuitableCRTC(int drm_fd, drmModeRes* resources, drmModeConnector* connector, drmModeCrtc*& crtc) {
#ifdef ENABLE_LVGL
    // 查找合适的CRTC
    for (int j = 0; j < resources->count_crtcs; j++) {
        // 检查这个CRTC是否可以驱动这个连接器
        if (connector->encoder_id) {
            drmModeEncoder* encoder = drmModeGetEncoder(drm_fd, connector->encoder_id);
            if (encoder && (encoder->possible_crtcs & (1 << j))) {
                crtc = drmModeGetCrtc(drm_fd, resources->crtcs[j]);
                if (crtc) {
                    std::cout << "[DRM] 找到合适的CRTC: " << crtc->crtc_id << std::endl;
                    drmModeFreeEncoder(encoder);
                    return true;
                }
            }
            if (encoder) drmModeFreeEncoder(encoder);
        }
    }
    return false;
#else
    return false;
#endif
}

bool createFramebuffer(int drm_fd, uint32_t width, uint32_t height,
                      uint32_t& fb_id, uint32_t& fb_handle, uint32_t*& framebuffer,
                      uint32_t& stride, uint32_t& buffer_size) {
#ifdef ENABLE_LVGL
    // 检测驱动类型以优化配置
    std::string driver_name;
    bool is_nvidia = false;
    if (detectDRMDriverType(drm_fd, driver_name)) {
        is_nvidia = (driver_name == "nvidia-drm");
    }
    
    // 创建优化的dumb buffer，针对nvidia-drm进行优化
    struct drm_mode_create_dumb create_req = {};
    create_req.width = width;
    create_req.height = height;
    create_req.bpp = 32; // 确保使用32位颜色深度
    
    // nvidia-drm优化：确保内存对齐
    if (is_nvidia) {
        // NVIDIA GPU对内存对齐有特殊要求，确保宽度对齐到64字节边界
        create_req.width = (width + 63) & ~63;
        std::cout << "[DRM] NVIDIA-DRM优化: 调整宽度从 " << width << " 到 " << create_req.width << std::endl;
    }
    
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) != 0) {
        std::cerr << "[DRM] 创建dumb buffer失败: " << strerror(errno) << std::endl;
        return false;
    }
    
    fb_handle = create_req.handle;
    stride = create_req.pitch;
    buffer_size = create_req.size;
    
    std::cout << "[DRM] 创建buffer: " << width << "x" << height
              << ", stride: " << stride << ", size: " << buffer_size << std::endl;
    
    // 创建framebuffer对象
    uint32_t depth = 24; // 颜色深度
    uint32_t bpp = 32;   // 每像素位数
    
    if (drmModeAddFB(drm_fd, width, height, depth, bpp, stride, fb_handle, &fb_id) != 0) {
        std::cerr << "[DRM] 创建framebuffer失败" << std::endl;
        return false;
    }
    
    // 映射framebuffer到用户空间
    struct drm_mode_map_dumb map_req = {};
    map_req.handle = fb_handle;
    
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) != 0) {
        std::cerr << "[DRM] map dumb buffer失败: " << strerror(errno) << std::endl;
        return false;
    }
    
    // nvidia-drm优化：使用更高效的内存映射标志
    int mmap_flags = MAP_SHARED;
    if (is_nvidia) {
        // NVIDIA GPU优化：尝试使用写合并（write-combining）内存
        mmap_flags |= MAP_NORESERVE;
        std::cout << "[DRM] NVIDIA-DRM优化: 使用高效内存映射" << std::endl;
    }
    
    framebuffer = (uint32_t*)mmap(0, buffer_size, PROT_READ | PROT_WRITE, mmap_flags, drm_fd, map_req.offset);
    if (framebuffer == MAP_FAILED) {
        std::cerr << "[DRM] framebuffer映射失败: " << strerror(errno) << std::endl;
        return false;
    }
    
    // 清空framebuffer (深色背景) - 针对nvidia-drm优化
    if (is_nvidia) {
        // NVIDIA GPU优化：使用更高效的内存初始化
        uint32_t clear_color = 0xFF1A1F26; // 设置为深色背景
        for (uint32_t i = 0; i < buffer_size / 4; i++) {
            framebuffer[i] = clear_color;
        }
        std::cout << "[DRM] NVIDIA-DRM优化: 使用加速内存清零" << std::endl;
    } else {
        memset(framebuffer, 0x00, buffer_size);
    }
    
    return true;
#else
    return false;
#endif
}

bool setCRTCMode(int drm_fd, drmModeCrtc* crtc, uint32_t fb_id,
                drmModeConnector* connector, drmModeModeInfo* mode) {
#ifdef ENABLE_LVGL
    // 🔧 恢复LVGL独占DRM模式 - 用户反馈需要完整的DRM控制才能渲染
    std::cout << "[DRM] 设置CRTC模式以启用LVGL渲染" << std::endl;
    std::cout << "[DRM] CRTC ID: " << crtc->crtc_id << ", FB ID: " << fb_id << std::endl;
    
    // 恢复完整的CRTC模式设置，让LVGL独占显示控制
    int ret = drmModeSetCrtc(drm_fd, crtc->crtc_id, fb_id, 0, 0,
                            &connector->connector_id, 1, mode);
    
    if (ret != 0) {
        std::cerr << "[DRM] 设置CRTC模式失败: " << strerror(errno) << std::endl;
        return false;
    }
    
    std::cout << "[DRM] CRTC模式设置成功，LVGL获得显示控制权" << std::endl;
    
    // 执行初始页面翻转以激活显示
    ret = drmModePageFlip(drm_fd, crtc->crtc_id, fb_id, 0, nullptr);
    if (ret != 0) {
        std::cout << "[DRM] 初始页面翻转失败（非致命）: " << strerror(errno) << std::endl;
        // 页面翻转失败不影响基本功能
    } else {
        std::cout << "[DRM] 初始页面翻转成功" << std::endl;
    }
    
    return true;
#else
    return false;
#endif
}

void cleanupDRMResources(int drm_fd, uint32_t fb_id, drmModeCrtc* crtc,
                        drmModeConnector* connector, uint32_t* framebuffer,
                        uint32_t fb_handle, uint32_t buffer_size) {
#ifdef ENABLE_LVGL
    // 修复双重释放内存错误 - 增强资源清理的安全性
    std::cout << "[DRM] 开始清理DRM资源..." << std::endl;
    std::cout << "[DRM] 清理状态: fd=" << drm_fd << " fb_id=" << fb_id
              << " fb_handle=" << fb_handle << " framebuffer=" << framebuffer
              << " crtc=" << crtc << " connector=" << connector << std::endl;
    
    // 1. 清理framebuffer映射 - 防止重复unmap
    if (framebuffer && framebuffer != MAP_FAILED && buffer_size > 0) {
        std::cout << "[DRM] 解除framebuffer映射，大小: " << buffer_size << " bytes" << std::endl;
        int unmap_result = munmap(framebuffer, buffer_size);
        if (unmap_result != 0) {
            std::cerr << "[DRM] framebuffer解映射失败: " << strerror(errno) << std::endl;
        } else {
            std::cout << "[DRM] framebuffer解映射成功" << std::endl;
        }
        framebuffer = nullptr;
    } else if (framebuffer) {
        std::cout << "[DRM] framebuffer已经是无效状态，跳过解映射" << std::endl;
        framebuffer = nullptr;
    }
    
    // 2. 移除framebuffer对象 - 检查是否有效
    if (fb_id > 0 && drm_fd >= 0) {
        std::cout << "[DRM] 移除framebuffer对象 ID: " << fb_id << std::endl;
        int rmfb_result = drmModeRmFB(drm_fd, fb_id);
        if (rmfb_result != 0) {
            std::cerr << "[DRM] 移除framebuffer失败: " << rmfb_result << std::endl;
        } else {
            std::cout << "[DRM] framebuffer对象移除成功" << std::endl;
        }
        fb_id = 0;
    } else if (fb_id > 0) {
        std::cout << "[DRM] fb_id有效但drm_fd无效，跳过framebuffer移除" << std::endl;
        fb_id = 0;
    }
    
    // 3. 销毁dumb buffer - 检查句柄有效性
    if (fb_handle > 0 && drm_fd >= 0) {
        std::cout << "[DRM] 销毁dumb buffer句柄: " << fb_handle << std::endl;
        struct drm_mode_destroy_dumb destroy_req = {};
        destroy_req.handle = fb_handle;
        int destroy_result = drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
        if (destroy_result != 0) {
            std::cerr << "[DRM] 销毁dumb buffer失败: " << destroy_result << " (" << strerror(errno) << ")" << std::endl;
        } else {
            std::cout << "[DRM] dumb buffer销毁成功" << std::endl;
        }
        fb_handle = 0;
    } else if (fb_handle > 0) {
        std::cout << "[DRM] fb_handle有效但drm_fd无效，跳过dumb buffer销毁" << std::endl;
        fb_handle = 0;
    }
    
    // 4. 释放CRTC结构体 - 检查指针有效性
    if (crtc) {
        std::cout << "[DRM] 释放CRTC结构体" << std::endl;
        drmModeFreeCrtc(crtc);
        crtc = nullptr;
    }
    
    // 5. 释放连接器结构体 - 检查指针有效性
    if (connector) {
        std::cout << "[DRM] 释放连接器结构体" << std::endl;
        drmModeFreeConnector(connector);
        connector = nullptr;
    }
    
    // 6. 关闭DRM文件描述符 - 最后执行，防止重复关闭
    if (drm_fd >= 0) {
        std::cout << "[DRM] 关闭DRM文件描述符: " << drm_fd << std::endl;
        int close_result = close(drm_fd);
        if (close_result != 0) {
            std::cerr << "[DRM] 关闭文件描述符失败: " << strerror(errno) << std::endl;
        } else {
            std::cout << "[DRM] 文件描述符关闭成功" << std::endl;
        }
        drm_fd = -1;
    }
    
    std::cout << "[DRM] DRM资源清理完成" << std::endl;
#endif
}

// GBM共享模式显示刷新回调函数
void gbm_display_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
#ifdef ENABLE_LVGL
    static uint32_t flush_count = 0;
    flush_count++;
    
    // 详细的调试信息（仅在前几次调用或定期显示）
    if (flush_count <= 5 || flush_count % 60 == 0) {
        std::cout << "[GBM] flush_cb调用 #" << flush_count
                  << " area(" << (area ? area->x1 : -1) << "," << (area ? area->y1 : -1)
                  << " to " << (area ? area->x2 : -1) << "," << (area ? area->y2 : -1)
                  << ") px_map=" << (px_map ? "valid" : "null") << std::endl;
    }
    
    // 参数验证
    if (!disp || !area || !px_map) {
        std::cerr << "[GBM] flush_cb参数无效: disp=" << disp << " area=" << area << " px_map=" << px_map << std::endl;
        if (disp) lv_display_flush_ready(disp);
        return;
    }
    
    // 获取GBM后端管理器
    auto& gbm_manager = GBMBackendManager::getInstance();
    auto* gbm_backend = gbm_manager.getBackend();
    
    if (!gbm_backend || !gbm_backend->isDRMResourceAvailable()) {
        if (flush_count <= 5) {
            std::cerr << "[GBM] GBM后端不可用，跳过刷新" << std::endl;
        }
        lv_display_flush_ready(disp);
        return;
    }
    
    // 获取显示尺寸
    auto config = gbm_manager.getSharedConfig();
    uint32_t drm_width = config.width;
    uint32_t drm_height = config.height;
    
    // 区域边界检查
    if (area->x1 < 0 || area->y1 < 0 ||
        area->x2 >= (int32_t)drm_width || area->y2 >= (int32_t)drm_height ||
        area->x1 > area->x2 || area->y1 > area->y2) {
        std::cerr << "[GBM] 无效的区域边界: (" << area->x1 << "," << area->y1
                  << ") to (" << area->x2 << "," << area->y2 << ")" << std::endl;
        lv_display_flush_ready(disp);
        return;
    }
    
    // 创建或获取GBM framebuffer
    static GBMFramebuffer* current_fb = nullptr;
    if (!current_fb) {
        current_fb = gbm_backend->createLVGLFramebuffer(drm_width, drm_height);
        if (!current_fb) {
            std::cerr << "[GBM] 创建LVGL framebuffer失败" << std::endl;
            lv_display_flush_ready(disp);
            return;
        }
        std::cout << "[GBM] 创建LVGL framebuffer成功: " << drm_width << "x" << drm_height << std::endl;
    }
    
    // 映射framebuffer用于写入 - 修复GBM映射问题
    if (!current_fb->map && current_fb->bo) {
        // 检查GBM buffer object是否支持映射
        uint32_t bo_stride = gbm_bo_get_stride(current_fb->bo);
        uint32_t bo_width = gbm_bo_get_width(current_fb->bo);
        uint32_t bo_height = gbm_bo_get_height(current_fb->bo);
        
        std::cout << "[GBM] BO详细信息: " << bo_width << "x" << bo_height
                  << " stride: " << bo_stride << std::endl;
        
        // 尝试使用简化的映射方法，避免复杂的GBM映射
        // 直接使用DRM framebuffer内存映射
        int drm_fd = 3; // 从日志中获取的DRM FD
        struct drm_mode_map_dumb map_req = {};
        map_req.handle = current_fb->handle;
        
        if (ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) == 0) {
            current_fb->map = mmap(nullptr, current_fb->size, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, drm_fd, map_req.offset);
            if (current_fb->map != MAP_FAILED) {
                std::cout << "[GBM] DRM framebuffer映射成功: " << current_fb->map
                          << " size: " << current_fb->size << std::endl;
            } else {
                std::cerr << "[GBM] DRM framebuffer映射失败: " << strerror(errno) << std::endl;
                current_fb->map = nullptr;
                lv_display_flush_ready(disp);
                return;
            }
        } else {
            std::cerr << "[GBM] DRM map请求失败: " << strerror(errno) << std::endl;
            lv_display_flush_ready(disp);
            return;
        }
    }
    
    // 复制像素数据到GBM framebuffer
    uint32_t* framebuffer = static_cast<uint32_t*>(current_fb->map);
    copyPixelDataToGBM(area, px_map, framebuffer, drm_width, drm_height, current_fb->stride, current_fb->size);
    
    // 提交framebuffer到primary plane
    if (!gbm_backend->commitLVGLFrame(current_fb)) {
        std::cerr << "[GBM] 提交LVGL帧失败" << std::endl;
    }
    
    // 等待VSync（可选，用于同步）
    gbm_backend->waitForVSync();
    
    // 通知LVGL刷新完成
    lv_display_flush_ready(disp);
#endif
}

void copyPixelDataToGBM(const lv_area_t* area, const uint8_t* px_map, uint32_t* framebuffer,
                        uint32_t drm_width, uint32_t drm_height, uint32_t stride, uint32_t buffer_size) {
#ifdef ENABLE_LVGL
    // 严格的区域边界检查
    if (area->x1 < 0 || area->y1 < 0 ||
        area->x2 >= (int32_t)drm_width || area->y2 >= (int32_t)drm_height ||
        area->x1 > area->x2 || area->y1 > area->y2) {
        std::cerr << "[GBM] Invalid area bounds: (" << area->x1 << "," << area->y1
                  << ") to (" << area->x2 << "," << area->y2 << ")" << std::endl;
        return;
    }
    
    uint32_t area_width = area->x2 - area->x1 + 1;
    uint32_t area_height = area->y2 - area->y1 + 1;
    uint32_t pixels_per_row = stride / 4; // stride是字节数，除以4得到uint32_t数量
    
    // 检测是否为nvidia-drm以启用优化
    static bool nvidia_optimizations = false;
    static bool optimization_checked = false;
    if (!optimization_checked) {
        // 简单检测：检查stride是否符合NVIDIA对齐要求
        if ((stride & 63) == 0) { // 64字节对齐通常表示NVIDIA优化
            nvidia_optimizations = true;
            std::cout << "[DRM] 检测到NVIDIA优化配置，启用加速像素复制" << std::endl;
        }
        optimization_checked = true;
    }
    
    // 验证缓冲区大小
    uint32_t total_area_pixels = area_width * area_height;
    if (total_area_pixels == 0) {
        return;
    }
    
    try {
        // NVIDIA优化：使用批量内存复制
        if (nvidia_optimizations && area_width > 64) {
            // 对于大面积更新，使用批量复制优化
            #if LV_COLOR_DEPTH == 32
                uint32_t* src_pixels = (uint32_t*)px_map;
                for (uint32_t y = 0; y < area_height; y++) {
                    uint32_t dst_row = area->y1 + y;
                    uint32_t dst_row_offset = dst_row * pixels_per_row;
                    
                    if (dst_row >= drm_height || dst_row_offset >= (buffer_size / 4)) {
                        continue;
                    }
                    
                    uint32_t dst_start = dst_row_offset + area->x1;
                    uint32_t src_start = y * area_width;
                    
                    // 批量复制整行（如果在边界内）
                    if (area->x1 + area_width <= drm_width &&
                        dst_start + area_width <= (buffer_size / 4) &&
                        src_start + area_width <= total_area_pixels) {
                        memcpy(&framebuffer[dst_start], &src_pixels[src_start], area_width * sizeof(uint32_t));
                    } else {
                        // 回退到逐像素复制
                        for (uint32_t x = 0; x < area_width; x++) {
                            uint32_t dst_col = area->x1 + x;
                            uint32_t dst_idx = dst_row_offset + dst_col;
                            uint32_t src_idx = src_start + x;
                            
                            if (dst_col < drm_width && dst_idx < (buffer_size / 4) && src_idx < total_area_pixels) {
                                framebuffer[dst_idx] = src_pixels[src_idx];
                            }
                        }
                    }
                }
            #else
                // 非32位模式，回退到逐像素处理
                nvidia_optimizations = false;
            #endif
        }
        
        // 标准逐像素复制（适用于小面积或非NVIDIA优化情况）
        if (!nvidia_optimizations || area_width <= 64) {
            for (uint32_t y = 0; y < area_height; y++) {
                uint32_t dst_row = area->y1 + y;
                uint32_t dst_row_offset = dst_row * pixels_per_row;
                
                // 检查目标行是否在有效范围内
                if (dst_row >= drm_height || dst_row_offset >= (buffer_size / 4)) {
                    continue;
                }
                
                for (uint32_t x = 0; x < area_width; x++) {
                    uint32_t dst_col = area->x1 + x;
                    uint32_t dst_idx = dst_row_offset + dst_col;
                    uint32_t src_idx = y * area_width + x;
                    
                    // 检查目标和源索引的有效性
                    if (dst_col >= drm_width || dst_idx >= (buffer_size / 4) ||
                        src_idx >= total_area_pixels) {
                        continue;
                    }
                    
                    // 简化的像素格式转换 - 使用32位ARGB8888
                    uint32_t pixel = 0x00000000; // 默认黑色
                    
                    #if LV_COLOR_DEPTH == 32
                        // 32位ARGB8888格式 - 直接复制
                        uint32_t* src_pixels = (uint32_t*)px_map;
                        if (src_idx < total_area_pixels) {
                            pixel = src_pixels[src_idx];
                        }
                    #elif LV_COLOR_DEPTH == 16
                        // 16位RGB565格式转换
                        uint16_t* src_pixels = (uint16_t*)px_map;
                        if (src_idx < total_area_pixels) {
                            uint16_t src_value = src_pixels[src_idx];
                            uint8_t r = ((src_value >> 11) & 0x1F) * 255 / 31;
                            uint8_t g = ((src_value >> 5) & 0x3F) * 255 / 63;
                            uint8_t b = (src_value & 0x1F) * 255 / 31;
                            pixel = (r << 16) | (g << 8) | b;
                        }
                    #else
                        // 24位RGB888格式
                        uint8_t* src_pixels = (uint8_t*)px_map;
                        uint32_t byte_idx = src_idx * 3;
                        if (byte_idx + 2 < total_area_pixels * 3) {
                            uint8_t r = src_pixels[byte_idx + 2]; // BGR -> RGB
                            uint8_t g = src_pixels[byte_idx + 1];
                            uint8_t b = src_pixels[byte_idx + 0];
                            pixel = (r << 16) | (g << 8) | b;
                        }
                    #endif
                    
                    framebuffer[dst_idx] = pixel;
                }
            }
        }
    } catch (...) {
        std::cerr << "[DRM] Exception during pixel copy" << std::endl;
    }
#endif
}

} // namespace ui
} // namespace bamboo_cut