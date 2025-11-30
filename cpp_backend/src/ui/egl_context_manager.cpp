#include "bamboo_cut/ui/egl_context_manager.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#ifdef ENABLE_WAYLAND
#include <wayland-client.h>
#include <wayland-egl.h>
#endif
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <cstdlib>

#ifndef EGL_NO_STREAM_KHR
#define EGL_NO_STREAM_KHR reinterpret_cast<EGLStreamKHR>(0)
#endif

namespace bamboo_cut {
namespace ui {

namespace {
constexpr EGLint kEglConfigAttrs[] = {
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT | EGL_STREAM_BIT_KHR,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_NONE};

constexpr EGLint kEglContextAttrs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 2,
    EGL_NONE};
}  // namespace

EGLContextManager& EGLContextManager::getInstance() {
    static EGLContextManager instance;
    return instance;
}

bool EGLContextManager::ensureInitialized(wl_display* wl_display, wl_surface* wl_surface, int width, int height) {
    
    // 强制走 DRM 平台，避免残留的 Wayland 环境影响
    setenv("EGL_PLATFORM", "drm", 1);
    unsetenv("WAYLAND_DISPLAY");
    if (!getenv("XDG_RUNTIME_DIR")) {
        setenv("XDG_RUNTIME_DIR", "/tmp", 1);
    }

    std::cout << "[EGL] ensureInitialized: width=" << width << " height=" << height
              << " wl_display=" << wl_display << " wl_surface=" << wl_surface << std::endl;
    return initializeSharedContext(wl_display, wl_surface, width, height);
}

bool EGLContextManager::initializeSharedContext(wl_display* wl_display, wl_surface* wl_surface, int width, int height) {
    

    std::cout << "[EGL] initializeSharedContext begin: w=" << width << " h=" << height << std::endl;

    if (!createDisplay(wl_display)) {
        std::cerr << "[EGL] Failed to create display" << std::endl;
        return false;
    }

    if (!chooseConfig()) {
        std::cerr << "[EGL] Failed to choose config" << std::endl;
        cleanupLocked();
        return false;
    }

    if (!createContext()) {
        std::cerr << "[EGL] Failed to create context" << std::endl;
        cleanupLocked();
        return false;
    }

    if (!createSurface(wl_surface, width, height)) {
        std::cerr << "[EGL] Failed to create Wayland EGL surface" << std::endl;
        cleanupLocked();
        return false;
    }

    std::cout << "[EGL] Shared display/context/stream initialized" << std::endl;
    std::cout << "[EGL] egl_display=" << primary_context_.display
              << " surface=" << primary_context_.surface
              << " context=" << primary_context_.context
              << " stream=" << primary_context_.stream << std::endl;

    if (!createStream()) {
        std::cerr << "[EGL] Failed to create EGLStream" << std::endl;
        cleanupLocked();
        return false;
    }

    if (eglMakeCurrent(primary_context_.display, primary_context_.surface, primary_context_.surface,
                       primary_context_.context) != EGL_TRUE) {
        std::cerr << "[EGL] eglMakeCurrent failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        cleanupLocked();
        return false;
    }

    std::cout << "[EGL] Shared display/context/stream initialized" << std::endl;
    std::cout << "[EGL] egl_display=" << primary_context_.display
              << " surface=" << primary_context_.surface
              << " context=" << primary_context_.context
              << " stream=" << primary_context_.stream << std::endl;

    primary_context_.width = width;
    primary_context_.height = height;
    primary_context_.wl_display_handle = wl_display;
    primary_context_.wl_surface_handle = wl_surface;
    primary_context_.is_initialized = true;

    std::cout << "[EGL] Shared display/context/stream initialized" << std::endl;
    return true;
}

bool EGLContextManager::createDisplay(wl_display* wl_display) {
    std::cout << "[EGL] eglGetDisplay(EGL_DEFAULT_DISPLAY) ..." << std::endl;
    primary_context_.display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (primary_context_.display == EGL_NO_DISPLAY) {
        std::cerr << "[EGL] eglGetDisplay returned NO_DISPLAY" << std::endl;
        return false;
    }

    EGLint major = 0;
    EGLint minor = 0;
    std::cout << "[EGL] eglInitialize..." << std::endl;
    if (eglInitialize(primary_context_.display, &major, &minor) != EGL_TRUE) {
        std::cerr << "[EGL] eglInitialize failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }

    if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) {
        std::cerr << "[EGL] eglBindAPI failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }

    std::cout << "[EGL] Initialized display (ES) version " << major << "." << minor
              << " display=" << primary_context_.display << std::endl;
    (void)wl_display;  // kept for signature symmetry; EGL_DEFAULT_DISPLAY already uses Wayland connection.
    return true;
}

bool EGLContextManager::chooseConfig() {
    EGLint num_configs = 0;
    if (eglChooseConfig(primary_context_.display, kEglConfigAttrs, &primary_context_.config, 1, &num_configs) != EGL_TRUE ||
        num_configs == 0) {
        std::cerr << "[EGL] eglChooseConfig failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }
    return true;
}

bool EGLContextManager::createContext() {
    primary_context_.context =
        eglCreateContext(primary_context_.display, primary_context_.config, EGL_NO_CONTEXT, kEglContextAttrs);
    if (primary_context_.context == EGL_NO_CONTEXT) {
        std::cerr << "[EGL] eglCreateContext failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }
    return true;
}

bool EGLContextManager::createSurface(wl_surface* wl_surface, int width, int height) {
    // 优先使用 Wayland surface；若为空则回退到 DRM/GBM surface
#if defined(ENABLE_WAYLAND)
    if (wl_surface) {
        primary_context_.wl_window_handle = wl_egl_window_create(wl_surface, width, height);
        if (!primary_context_.wl_window_handle) {
            std::cerr << "[EGL] wl_egl_window_create failed" << std::endl;
            return false;
        }

        primary_context_.surface =
            eglCreateWindowSurface(primary_context_.display, primary_context_.config, primary_context_.wl_window_handle, nullptr);
        if (primary_context_.surface == EGL_NO_SURFACE) {
            std::cerr << "[EGL] eglCreateWindowSurface failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
            return false;
        }
        return true;
    }
#endif

    // DRM/GBM 回退路径
    std::cout << "[EGL] Creating GBM surface for DRM" << std::endl;

    std::vector<std::string> candidates;
    if (const char* env = getenv("EGL_DRM_DEVICE_FILE")) {
        candidates.emplace_back(env);
    }
    candidates.emplace_back("/dev/dri/card1");
    candidates.emplace_back("/dev/dri/card0");

    for (const auto& drm_path : candidates) {
        int fd = open(drm_path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            std::cerr << "[EGL] Failed to open " << drm_path << ", errno=" << errno << std::endl;
            continue;
        }
        std::cout << "[EGL] Using DRM device: " << drm_path << " fd=" << fd << std::endl;

        gbm_device* gbm_dev = gbm_create_device(fd);
        if (!gbm_dev) {
            std::cerr << "[EGL] gbm_create_device failed on " << drm_path << std::endl;
            close(fd);
            continue;
        }

        auto try_surface = [&](uint32_t fmt) -> gbm_surface* {
            return gbm_surface_create(gbm_dev, width, height, fmt,
                                      GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
        };

        gbm_surface* gbm_surf = try_surface(GBM_FORMAT_XRGB8888);
        if (!gbm_surf) {
            gbm_surf = try_surface(GBM_FORMAT_ARGB8888);
        }

        if (!gbm_surf) {
            std::cerr << "[EGL] gbm_surface_create failed on " << drm_path << std::endl;
            gbm_device_destroy(gbm_dev);
            close(fd);
            continue;
        }

        EGLSurface surface = eglCreateWindowSurface(
            primary_context_.display,
            primary_context_.config,
            reinterpret_cast<EGLNativeWindowType>(gbm_surf),
            nullptr);

        if (surface == EGL_NO_SURFACE) {
            std::cerr << "[EGL] eglCreateWindowSurface failed (GBM) on " << drm_path
                      << ": 0x" << std::hex << eglGetError() << std::dec << std::endl;
            gbm_surface_destroy(gbm_surf);
            gbm_device_destroy(gbm_dev);
            close(fd);
            continue;
        }

        primary_context_.drm_fd = fd;
        primary_context_.gbm_dev = gbm_dev;
        primary_context_.gbm_surf = gbm_surf;
        primary_context_.surface = surface;

        std::cout << "[EGL] GBM surface created: device=" << drm_path
                  << " fd=" << primary_context_.drm_fd
                  << " gbm_dev=" << primary_context_.gbm_dev
                  << " gbm_surf=" << primary_context_.gbm_surf << std::endl;
        return true;
    }

    std::cerr << "[EGL] Failed to create GBM surface on all DRM devices" << std::endl;
    return false;
}

bool EGLContextManager::createStream() {
    eglCreateStreamKHR_ =
        reinterpret_cast<PFNEGLCREATESTREAMKHRPROC>(eglGetProcAddress("eglCreateStreamKHR"));
    eglDestroyStreamKHR_ =
        reinterpret_cast<PFNEGLDESTROYSTREAMKHRPROC>(eglGetProcAddress("eglDestroyStreamKHR"));

    if (!eglCreateStreamKHR_) {
        std::cerr << "[EGL] eglCreateStreamKHR not available" << std::endl;
        return false;
    }

    const EGLint stream_attribs[] = {
        EGL_STREAM_FIFO_LENGTH_KHR, 2,
        EGL_CONSUMER_LATENCY_USEC_KHR, 16000,
        EGL_NONE};

    primary_context_.stream = eglCreateStreamKHR_(primary_context_.display, stream_attribs);
    if (primary_context_.stream == EGL_NO_STREAM_KHR) {
        std::cerr << "[EGL] eglCreateStreamKHR failed: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }

    return true;
}

bool EGLContextManager::makeCurrent() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    if (!primary_context_.is_initialized) {
        return false;
    }
    return eglMakeCurrent(primary_context_.display, primary_context_.surface, primary_context_.surface,
                          primary_context_.context) == EGL_TRUE;
}

EGLDisplay EGLContextManager::getDisplay() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.display;
}

EGLContext EGLContextManager::getContext() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.context;
}

EGLSurface EGLContextManager::getSurface() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.surface;
}

EGLConfig EGLContextManager::getConfig() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.config;
}

EGLStreamKHR EGLContextManager::getStream() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.stream;
}

wl_egl_window* EGLContextManager::getWlEglWindow() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return primary_context_.wl_window_handle;
}

void EGLContextManager::cleanupLocked() {
    if (primary_context_.display != EGL_NO_DISPLAY) {
        eglMakeCurrent(primary_context_.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    if (primary_context_.surface != EGL_NO_SURFACE) {
        eglDestroySurface(primary_context_.display, primary_context_.surface);
    }

    if (primary_context_.context != EGL_NO_CONTEXT) {
        eglDestroyContext(primary_context_.display, primary_context_.context);
    }

    if (primary_context_.stream != EGL_NO_STREAM_KHR && eglDestroyStreamKHR_) {
        eglDestroyStreamKHR_(primary_context_.display, primary_context_.stream);
    }

    if (primary_context_.display != EGL_NO_DISPLAY) {
        eglTerminate(primary_context_.display);
    }

#if defined(ENABLE_WAYLAND)
    if (primary_context_.wl_window_handle) {
        wl_egl_window_destroy(primary_context_.wl_window_handle);
    }
#endif

    if (primary_context_.gbm_surf) {
        gbm_surface_destroy(primary_context_.gbm_surf);
    }
    if (primary_context_.gbm_dev) {
        gbm_device_destroy(primary_context_.gbm_dev);
    }
    if (primary_context_.drm_fd >= 0) {
        close(primary_context_.drm_fd);
    }

    primary_context_ = EGLContextConfig();
}

void EGLContextManager::cleanup() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    cleanupLocked();
}

EGLContextManager::~EGLContextManager() {
    cleanup();
}

}  // namespace ui
}  // namespace bamboo_cut

