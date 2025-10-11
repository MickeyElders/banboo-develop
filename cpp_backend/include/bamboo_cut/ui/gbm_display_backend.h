/**
 * @file gbm_display_backend.h
 * @brief GBM (Generic Buffer Management) 显示后端
 * 支持LVGL和GStreamer共享DRM资源的分层显示
 */

#pragma once

#include <gbm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <memory>
#include <mutex>

namespace bamboo_cut {
namespace ui {

/**
 * @brief DRM资源共享配置
 */
struct DRMSharedConfig {
    int drm_fd;                    // DRM设备文件描述符
    uint32_t crtc_id;             // CRTC ID
    uint32_t connector_id;        // 连接器ID
    uint32_t primary_plane_id;    // Primary plane ID (LVGL使用)
    uint32_t overlay_plane_id;    // Overlay plane ID (GStreamer使用)
    uint32_t width;               // 屏幕宽度
    uint32_t height;              // 屏幕高度
    bool sharing_enabled;         // 是否启用资源共享
};

/**
 * @brief GBM帧缓冲区对象
 */
struct GBMFramebuffer {
    uint32_t fb_id;               // DRM framebuffer ID
    uint32_t handle;              // GEM handle
    uint32_t stride;              // 行步长
    uint32_t size;                // 缓冲区大小
    void* map;                    // 内存映射地址
    gbm_bo* bo;                   // GBM buffer object
};

/**
 * @brief GBM显示后端类
 * 实现LVGL和GStreamer的DRM资源共享
 */
class GBMDisplayBackend {
public:
    GBMDisplayBackend();
    ~GBMDisplayBackend();

    /**
     * @brief 初始化GBM显示后端
     * @param config DRM共享配置
     * @return 是否成功
     */
    bool initialize(const DRMSharedConfig& config);

    /**
     * @brief 创建LVGL专用的framebuffer
     * @param width 宽度
     * @param height 高度
     * @return framebuffer指针，失败时返回nullptr
     */
    GBMFramebuffer* createLVGLFramebuffer(uint32_t width, uint32_t height);

    /**
     * @brief 获取DRM设备文件描述符（供GStreamer使用）
     * @return DRM fd
     */
    int getDRMFd() const { return drm_fd_; }

    /**
     * @brief 获取overlay plane ID（供GStreamer使用）
     * @return overlay plane ID
     */
    uint32_t getOverlayPlaneId() const { return config_.overlay_plane_id; }

    /**
     * @brief 获取primary plane ID（供LVGL使用）
     */
    uint32_t getPrimaryPlaneId() const { return config_.primary_plane_id; }

    /**
     * @brief 获取CRTC ID
     */
    uint32_t getCRTCId() const { return config_.crtc_id; }

    /**
     * @brief 获取连接器ID
     */
    uint32_t getConnectorId() const { return config_.connector_id; }

    /**
     * @brief 提交LVGL帧缓冲区到primary plane
     * @param fb framebuffer对象
     * @return 是否成功
     */
    bool commitLVGLFrame(GBMFramebuffer* fb);

    /**
     * @brief 释放framebuffer
     * @param fb framebuffer对象
     */
    void releaseFramebuffer(GBMFramebuffer* fb);

    /**
     * @brief 等待VSync垂直同步
     */
    void waitForVSync();

    /**
     * @brief 检查DRM资源是否可用
     * @return 是否可用
     */
    bool isDRMResourceAvailable();

    /**
     * @brief 获取EGL显示对象（如果需要）
     */
    EGLDisplay getEGLDisplay() const { return egl_display_; }

    /**
     * @brief 获取GBM设备对象
     */
    gbm_device* getGBMDevice() const { return gbm_device_; }

private:
    /**
     * @brief 打开DRM设备
     * @return 是否成功
     */
    bool openDRMDevice();

    /**
     * @brief 初始化GBM设备
     * @return 是否成功
     */
    bool initializeGBM();

    /**
     * @brief 初始化EGL
     * @return 是否成功
     */
    bool initializeEGL();

    /**
     * @brief 检测和配置DRM平面
     * @return 是否成功
     */
    bool setupDRMPlanes();

    /**
     * @brief 查找可用的连接器
     * @return 连接器ID，失败时返回0
     */
    uint32_t findConnector();

    /**
     * @brief 查找primary plane
     * @return plane ID，失败时返回0
     */
    uint32_t findPrimaryPlane();

    /**
     * @brief 查找overlay plane
     * @return plane ID，失败时返回0
     */
    uint32_t findOverlayPlane();

    /**
     * @brief 设置CRTC模式
     * @return 是否成功
     */
    bool setupCRTCMode();

    /**
     * @brief 清理资源
     */
    void cleanup();

private:
    DRMSharedConfig config_;       // 配置
    
    // DRM相关
    int drm_fd_;                   // DRM设备文件描述符
    drmModeRes* drm_resources_;    // DRM资源
    drmModeConnector* connector_;  // 连接器
    drmModeModeInfo mode_;         // 显示模式
    
    // GBM相关
    gbm_device* gbm_device_;       // GBM设备
    gbm_surface* gbm_surface_;     // GBM表面
    
    // EGL相关
    EGLDisplay egl_display_;       // EGL显示
    EGLContext egl_context_;       // EGL上下文
    EGLConfig egl_config_;         // EGL配置
    EGLSurface egl_surface_;       // EGL表面
    
    // 同步
    std::mutex drm_mutex_;         // DRM操作互斥锁
    
    bool initialized_;             // 是否已初始化
};

/**
 * @brief GBM显示后端单例管理器
 */
class GBMBackendManager {
public:
    static GBMBackendManager& getInstance();
    
    /**
     * @brief 初始化GBM后端
     * @param config 配置
     * @return 是否成功
     */
    bool initialize(const DRMSharedConfig& config);
    
    /**
     * @brief 获取GBM后端实例
     */
    GBMDisplayBackend* getBackend() { return backend_.get(); }
    
    /**
     * @brief 获取共享的DRM配置
     */
    const DRMSharedConfig& getSharedConfig() const { return shared_config_; }

private:
    GBMBackendManager() = default;
    ~GBMBackendManager() = default;
    
    std::unique_ptr<GBMDisplayBackend> backend_;
    DRMSharedConfig shared_config_;
    std::mutex init_mutex_;
};

} // namespace ui
} // namespace bamboo_cut