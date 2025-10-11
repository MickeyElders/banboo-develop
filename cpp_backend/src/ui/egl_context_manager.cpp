/**
 * @file egl_context_manager.cpp
 * @brief EGL上下文管理器实现 - LVGL与nvarguscamerasrc的EGL环境共享
 */

#include "bamboo_cut/ui/egl_context_manager.h"
#include "bamboo_cut/utils/logger.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <gbm.h>
#include <xf86drm.h>

namespace bamboo_cut {
namespace ui {

EGLContextManager& EGLContextManager::getInstance() {
    static EGLContextManager instance;
    return instance;
}

EGLContextManager::~EGLContextManager() {
    cleanup();
}

bool EGLContextManager::initializePrimaryContext(int drm_fd, void* gbm_device) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    if (primary_context_.is_initialized) {
        Logger::log("EGL上下文已经初始化", Logger::WARNING);
        return true;
    }
    
    drm_fd_ = drm_fd;
    gbm_device_ = gbm_device;
    
    // 获取EGL扩展函数指针
    eglGetPlatformDisplayEXT_ = 
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    eglCreatePlatformWindowSurfaceEXT_ = 
        (PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC)eglGetProcAddress("eglCreatePlatformWindowSurfaceEXT");
    
    if (!eglGetPlatformDisplayEXT_ || !eglCreatePlatformWindowSurfaceEXT_) {
        Logger::log("无法获取EGL平台扩展函数", Logger::ERROR);
        return false;
    }
    
    // 创建EGL display
    if (!createEGLDisplay(drm_fd, gbm_device)) {
        Logger::log("创建EGL display失败", Logger::ERROR);
        return false;
    }
    
    // 初始化EGL
    EGLint major, minor;
    if (!eglInitialize(primary_context_.display, &major, &minor)) {
        Logger::log("初始化EGL失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    Logger::log("EGL版本: " + std::to_string(major) + "." + std::to_string(minor), Logger::INFO);
    
    // 选择EGL配置
    if (!chooseEGLConfig()) {
        Logger::log("选择EGL配置失败", Logger::ERROR);
        return false;
    }
    
    // 绑定OpenGL ES API
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        Logger::log("绑定OpenGL ES API失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    // 创建EGL上下文
    if (!createEGLContext()) {
        Logger::log("创建EGL上下文失败", Logger::ERROR);
        return false;
    }
    
    // 创建EGL surface
    if (!createEGLSurface(gbm_device)) {
        Logger::log("创建EGL surface失败", Logger::ERROR);
        return false;
    }
    
    // 激活上下文
    if (!eglMakeCurrent(primary_context_.display, 
                       primary_context_.surface, 
                       primary_context_.surface, 
                       primary_context_.context)) {
        Logger::log("激活EGL上下文失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    primary_context_.is_initialized = true;
    Logger::log("EGL主上下文初始化成功", Logger::INFO);
    
    // 设置nvarguscamerasrc环境变量
    setupArgusEnvironment();
    
    return true;
}

bool EGLContextManager::createEGLDisplay(int drm_fd, void* gbm_device) {
    // 使用GBM平台创建EGL display
    primary_context_.display = eglGetPlatformDisplayEXT_(
        EGL_PLATFORM_GBM_KHR, 
        gbm_device, 
        NULL
    );
    
    if (primary_context_.display == EGL_NO_DISPLAY) {
        Logger::log("创建GBM EGL display失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    return true;
}

bool EGLContextManager::chooseEGLConfig() {
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLint num_configs;
    if (!eglChooseConfig(primary_context_.display, config_attribs, 
                        &primary_context_.config, 1, &num_configs)) {
        Logger::log("选择EGL配置失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    if (num_configs == 0) {
        Logger::log("没有找到合适的EGL配置", Logger::ERROR);
        return false;
    }
    
    return true;
}

bool EGLContextManager::createEGLContext() {
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    primary_context_.context = eglCreateContext(
        primary_context_.display,
        primary_context_.config,
        EGL_NO_CONTEXT,
        context_attribs
    );
    
    if (primary_context_.context == EGL_NO_CONTEXT) {
        Logger::log("创建EGL上下文失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return false;
    }
    
    return true;
}

bool EGLContextManager::createEGLSurface(void* gbm_device) {
    // 创建GBM surface用于EGL
    struct gbm_device* gbm_dev = static_cast<struct gbm_device*>(gbm_device);
    struct gbm_surface* gbm_surf = gbm_surface_create(
        gbm_dev,
        1920, 1080,  // 默认分辨率
        GBM_FORMAT_ARGB8888,
        GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING
    );
    
    if (!gbm_surf) {
        Logger::log("创建GBM surface失败", Logger::ERROR);
        return false;
    }
    
    // 创建EGL window surface
    primary_context_.surface = eglCreatePlatformWindowSurfaceEXT_(
        primary_context_.display,
        primary_context_.config,
        gbm_surf,
        NULL
    );
    
    if (primary_context_.surface == EGL_NO_SURFACE) {
        Logger::log("创建EGL surface失败: " + std::to_string(eglGetError()), Logger::ERROR);
        gbm_surface_destroy(gbm_surf);
        return false;
    }
    
    return true;
}

EGLContextConfig EGLContextManager::createSharedContext() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    EGLContextConfig shared_config;
    
    if (!primary_context_.is_initialized) {
        Logger::log("主EGL上下文未初始化，无法创建共享上下文", Logger::ERROR);
        return shared_config;
    }
    
    // 复制display和config
    shared_config.display = primary_context_.display;
    shared_config.config = primary_context_.config;
    
    // 创建共享的EGL上下文
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    shared_config.context = eglCreateContext(
        shared_config.display,
        shared_config.config,
        primary_context_.context,  // 共享主上下文
        context_attribs
    );
    
    if (shared_config.context == EGL_NO_CONTEXT) {
        Logger::log("创建共享EGL上下文失败: " + std::to_string(eglGetError()), Logger::ERROR);
        return shared_config;
    }
    
    shared_config.is_initialized = true;
    Logger::log("创建共享EGL上下文成功", Logger::INFO);
    
    return shared_config;
}

std::string EGLContextManager::getEGLDisplayPointer() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    if (!primary_context_.is_initialized) {
        return "";
    }
    
    // 将EGL display指针转换为字符串
    std::ostringstream oss;
    oss << std::hex << reinterpret_cast<uintptr_t>(primary_context_.display);
    return oss.str();
}

void EGLContextManager::setupArgusEnvironment() {
    if (!primary_context_.is_initialized) {
        Logger::log("EGL上下文未初始化，无法设置Argus环境", Logger::WARNING);
        return;
    }
    
    // 获取EGL display指针字符串
    std::string display_ptr = getEGLDisplayPointer();
    
    // 设置nvarguscamerasrc需要的环境变量
    std::string egl_display_env = "EGL_DISPLAY=" + display_ptr;
    
    // 设置环境变量
    if (setenv("EGL_DISPLAY", display_ptr.c_str(), 1) != 0) {
        Logger::log("设置EGL_DISPLAY环境变量失败", Logger::WARNING);
    } else {
        Logger::log("设置EGL_DISPLAY=" + display_ptr, Logger::INFO);
    }
    
    // 设置其他Argus相关环境变量
    setenv("DISPLAY", ":0", 1);
    setenv("GST_DEBUG", "2", 1);
    setenv("GST_GL_PLATFORM", "egl", 1);
    setenv("GST_GL_API", "gles2", 1);
    
    Logger::log("Argus环境变量设置完成", Logger::INFO);
}

void EGLContextManager::cleanup() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    if (primary_context_.is_initialized) {
        if (primary_context_.context != EGL_NO_CONTEXT) {
            eglDestroyContext(primary_context_.display, primary_context_.context);
            primary_context_.context = EGL_NO_CONTEXT;
        }
        
        if (primary_context_.surface != EGL_NO_SURFACE) {
            eglDestroySurface(primary_context_.display, primary_context_.surface);
            primary_context_.surface = EGL_NO_SURFACE;
        }
        
        if (primary_context_.display != EGL_NO_DISPLAY) {
            eglTerminate(primary_context_.display);
            primary_context_.display = EGL_NO_DISPLAY;
        }
        
        primary_context_.is_initialized = false;
        Logger::log("EGL上下文资源清理完成", Logger::INFO);
    }
}

} // namespace ui
} // namespace bamboo_cut