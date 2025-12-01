#pragma once

// 轻量占位 EGL 管理器，供 DeepStream 传递 EGL display/stream 时使用。
// 若需要真实 EGLStream 功能，请替换为实际实现。

#if __has_include(<EGL/egl.h>)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#else
using EGLDisplay = void*;
using EGLStreamKHR = void*;
#define EGL_NO_DISPLAY nullptr
#define EGL_NO_STREAM_KHR nullptr
#endif

namespace bamboo_cut {
namespace ui {

class EGLContextManager {
public:
    static EGLContextManager& getInstance() {
        static EGLContextManager instance;
        return instance;
    }

    bool initialize() {
        initialized_ = true; // 占位，不创建实际 EGL 资源
        return true;
    }

    bool isInitialized() const { return initialized_; }
    EGLDisplay getDisplay() const { return EGL_NO_DISPLAY; }
    EGLStreamKHR getStream() const { return EGL_NO_STREAM_KHR; }

private:
    EGLContextManager() = default;
    bool initialized_{false};
};

} // namespace ui
} // namespace bamboo_cut
