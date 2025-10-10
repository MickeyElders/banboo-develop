/**
 * @file wayland_compat.h
 * @brief Wayland兼容性头文件
 * 解决编译时的依赖和兼容性问题
 */

#ifndef BAMBOO_CUT_UI_WAYLAND_COMPAT_H
#define BAMBOO_CUT_UI_WAYLAND_COMPAT_H

// 前向声明和类型定义
#ifdef ENABLE_WAYLAND
    // Wayland可用时的实际包含
    #include <wayland-client.h>
    #ifdef __cplusplus
        extern "C" {
    #endif
    // 确保Wayland协议定义可用
    #ifdef __cplusplus
        }
    #endif
#else
    // Wayland不可用时的占位符定义
    struct wl_display;
    struct wl_surface;
    struct wl_compositor;
    struct wl_registry;
    struct wl_shell;
    struct wl_shell_surface;
    struct wl_egl_window;
    
    typedef uint32_t wl_fixed_t;
    typedef void (*wl_notify_func_t)(struct wl_listener *listener, void *data);
    
    struct wl_listener {
        struct wl_list link;
        wl_notify_func_t notify;
    };
    
    struct wl_list {
        struct wl_list *prev;
        struct wl_list *next;
    };
    
    // 空实现的内联函数
    static inline struct wl_display* wl_display_connect(const char* name) { return nullptr; }
    static inline void wl_display_disconnect(struct wl_display* display) {}
    static inline int wl_display_dispatch(struct wl_display* display) { return 0; }
    static inline int wl_display_dispatch_pending(struct wl_display* display) { return 0; }
    static inline int wl_display_flush(struct wl_display* display) { return 0; }
    static inline int wl_display_roundtrip(struct wl_display* display) { return 0; }
#endif

// LVGL兼容性
#ifdef ENABLE_LVGL
    #include <lvgl/lvgl.h>
#else
    // LVGL占位符定义
    typedef void* lv_display_t;
    typedef void* lv_area_t;
    typedef uint8_t* lv_color_t;
    typedef void* lv_draw_buf_t;
    typedef void* lv_indev_t;
    typedef void* lv_indev_data_t;
    
    static inline void lv_display_flush_ready(lv_display_t* disp) {}
#endif

// EGL兼容性
#ifdef ENABLE_WAYLAND
    #include <EGL/egl.h>
    #include <GLES2/gl2.h>
#else
    typedef void* EGLDisplay;
    typedef void* EGLConfig;
    typedef void* EGLContext;
    typedef void* EGLSurface;
    typedef void* EGLNativeDisplayType;
    typedef void* EGLNativeWindowType;
    typedef int32_t EGLint;
    typedef unsigned int EGLBoolean;
    
    #define EGL_NO_DISPLAY ((EGLDisplay)0)
    #define EGL_NO_CONTEXT ((EGLContext)0)
    #define EGL_NO_SURFACE ((EGLSurface)0)
    #define EGL_FALSE 0
    #define EGL_TRUE 1
    
    // EGL占位符函数
    static inline EGLDisplay eglGetDisplay(EGLNativeDisplayType display_id) { return nullptr; }
    static inline EGLBoolean eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor) { return EGL_FALSE; }
    static inline EGLBoolean eglTerminate(EGLDisplay dpy) { return EGL_TRUE; }
    static inline EGLBoolean eglChooseConfig(EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs, EGLint config_size, EGLint *num_config) { return EGL_FALSE; }
    static inline EGLContext eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint *attrib_list) { return nullptr; }
    static inline EGLSurface eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win, const EGLint *attrib_list) { return nullptr; }
    static inline EGLBoolean eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx) { return EGL_FALSE; }
    static inline EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) { return EGL_FALSE; }
    static inline EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface) { return EGL_TRUE; }
    static inline EGLBoolean eglDestroyContext(EGLDisplay dpy, EGLContext ctx) { return EGL_TRUE; }
    
    // OpenGL ES占位符
    static inline void glClearColor(float red, float green, float blue, float alpha) {}
    static inline void glClear(unsigned int mask) {}
    #define GL_COLOR_BUFFER_BIT 0x00004000
#endif

#endif // BAMBOO_CUT_UI_WAYLAND_COMPAT_H