/**
 * @file egl_context_manager.h
 * @brief Shared EGL display/context/stream manager for LVGL + DeepStream (Wayland/DRM)
 */

#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <mutex>

struct wl_display;
struct wl_surface;
struct wl_egl_window;
struct gbm_device;
struct gbm_surface;

namespace bamboo_cut {
namespace ui {

struct EGLContextConfig {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    EGLConfig config;
    EGLStreamKHR stream;
    wl_display* wl_display;
    wl_surface* wl_surface;
    wl_egl_window* wl_window;
    int drm_fd;
    gbm_device* gbm_dev;
    gbm_surface* gbm_surf;
    int width;
    int height;
    bool is_initialized;

    EGLContextConfig()
        : display(EGL_NO_DISPLAY)
        , context(EGL_NO_CONTEXT)
        , surface(EGL_NO_SURFACE)
        , config(nullptr)
        , stream(EGL_NO_STREAM_KHR)
        , wl_display(nullptr)
        , wl_surface(nullptr)
        , wl_window(nullptr)
        , drm_fd(-1)
        , gbm_dev(nullptr)
        , gbm_surf(nullptr)
        , width(0)
        , height(0)
        , is_initialized(false) {}
};

/**
 * @brief Singleton that owns a shared EGLDisplay/EGLContext/EGLStream.
 *        The manager is initialized once from LVGL (Wayland) and reused
 *        by DeepStream's nveglstreamsink.
 */
class EGLContextManager {
public:
    static EGLContextManager& getInstance();

    bool ensureInitialized(struct wl_display* wl_display, struct wl_surface* wl_surface, int width, int height);
    bool initializeSharedContext(struct wl_display* wl_display, struct wl_surface* wl_surface, int width, int height);

    EGLDisplay getDisplay() const;
    EGLContext getContext() const;
    EGLSurface getSurface() const;
    EGLConfig getConfig() const;
    EGLStreamKHR getStream() const;
    wl_egl_window* getWlEglWindow() const;

    bool makeCurrent();
    bool isInitialized() const { return primary_context_.is_initialized; }
    void cleanup();

private:
    EGLContextManager() = default;
    ~EGLContextManager();

    EGLContextManager(const EGLContextManager&) = delete;
    EGLContextManager& operator=(const EGLContextManager&) = delete;

    bool createDisplay(struct wl_display* wl_display);
    bool chooseConfig();
    bool createContext();
    bool createSurface(struct wl_surface* wl_surface, int width, int height);
    bool createStream();
    void cleanupLocked();

private:
    mutable std::mutex context_mutex_;
    EGLContextConfig primary_context_;

    PFNEGLCREATESTREAMKHRPROC eglCreateStreamKHR_ = nullptr;
    PFNEGLDESTROYSTREAMKHRPROC eglDestroyStreamKHR_ = nullptr;
};

} // namespace ui
} // namespace bamboo_cut
