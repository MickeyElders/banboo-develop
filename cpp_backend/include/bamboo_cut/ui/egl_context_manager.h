/**
 * @file egl_context_manager.h
 * @brief EGL上下文管理器 - 实现LVGL与nvarguscamerasrc的EGL环境共享
 */

#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <memory>
#include <mutex>

namespace bamboo_cut {
namespace ui {

/**
 * @brief EGL上下文配置结构
 */
struct EGLContextConfig {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    EGLConfig config;
    bool is_initialized;
    
    EGLContextConfig() 
        : display(EGL_NO_DISPLAY)
        , context(EGL_NO_CONTEXT)
        , surface(EGL_NO_SURFACE)
        , config(nullptr)
        , is_initialized(false) {}
};

/**
 * @brief EGL上下文共享管理器
 * 负责创建和管理LVGL与nvarguscamerasrc之间的共享EGL环境
 */
class EGLContextManager {
public:
    static EGLContextManager& getInstance();
    
    /**
     * @brief 初始化主EGL上下文（由LVGL创建）
     * @param drm_fd DRM文件描述符
     * @param gbm_device GBM设备指针
     * @return 是否成功
     */
    bool initializePrimaryContext(int drm_fd, void* gbm_device);
    
    /**
     * @brief 创建共享EGL上下文（供nvarguscamerasrc使用）
     * @return 共享的EGL上下文配置
     */
    EGLContextConfig createSharedContext();
    
    /**
     * @brief 获取主EGL display（供nvarguscamerasrc环境变量设置）
     * @return EGL display指针字符串
     */
    std::string getEGLDisplayPointer() const;
    
    /**
     * @brief 设置nvarguscamerasrc所需的环境变量
     */
    void setupArgusEnvironment();
    
    /**
     * @brief 检查EGL上下文是否已初始化
     */
    bool isInitialized() const { return primary_context_.is_initialized; }
    
    /**
     * @brief 清理EGL资源
     */
    void cleanup();
    
    /**
     * @brief 获取主EGL上下文（用于调试）
     */
    const EGLContextConfig& getPrimaryContext() const { return primary_context_; }

private:
    EGLContextManager() = default;
    ~EGLContextManager();
    
    // 禁用拷贝和赋值
    EGLContextManager(const EGLContextManager&) = delete;
    EGLContextManager& operator=(const EGLContextManager&) = delete;
    
    /**
     * @brief 创建EGL display
     */
    bool createEGLDisplay(int drm_fd, void* gbm_device);
    
    /**
     * @brief 选择EGL配置
     */
    bool chooseEGLConfig();
    
    /**
     * @brief 创建EGL上下文
     */
    bool createEGLContext();
    
    /**
     * @brief 创建EGL surface
     */
    bool createEGLSurface(void* gbm_device);

private:
    mutable std::mutex context_mutex_;
    EGLContextConfig primary_context_;
    int drm_fd_;
    void* gbm_device_;
    
    // EGL扩展函数指针
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT_;
    PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC eglCreatePlatformWindowSurfaceEXT_;
};

} // namespace ui
} // namespace bamboo_cut