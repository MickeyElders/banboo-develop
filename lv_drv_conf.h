/**
 * @file lv_drv_conf.h
 * Configuration for lv_drivers - Wayland架构专用配置
 * Bamboo Recognition System - Wayland迁移版本
 */

#ifndef LV_DRV_CONF_H
#define LV_DRV_CONF_H

#include "lv_conf.h"

/*==================
 * WAYLAND DISPLAY
 *==================*/

/* Wayland display driver */
#define USE_WAYLAND       1

#if USE_WAYLAND
# define WAYLAND_HOR_RES    1920
# define WAYLAND_VER_RES    1200
#endif

/*==================
 * INPUT DEVICES
 *==================*/

/* Wayland mouse/pointer input */
#define USE_WAYLAND_POINTER     1

/* Wayland keyboard input */  
#define USE_WAYLAND_KEYBOARD    1

/* Wayland touch input */
#define USE_WAYLAND_TOUCH       1

/*==================
 * DISPLAY DRIVERS
 *==================*/

/* Linux frame buffer device (/dev/fbX) - 已禁用，使用Wayland */
#define USE_FBDEV           0

/* DRM/KMS 直接访问 - 已禁用，使用Wayland合成器 */
#define USE_DRM             0

/* SDL display driver - 可选，主要用于开发测试 */
#define USE_SDL             0

/* Monitor of a computer - 桌面开发用，在嵌入式环境禁用 */
#define USE_MONITOR         0

/* SSD1306 OLED display - 不适用 */
#define USE_SSD1306         0

/* Windows display driver - 不适用 */
#define USE_WINDRV          0

/*==================
 * INPUT DEVICE DRIVERS
 *==================*/

/* libinput-based drivers for Wayland */
#define USE_LIBINPUT_GENERIC    1

/* XKB keyboard (Linux) - Wayland需要 */
#define USE_XKB                 1

/* Touchpad as mouse (libinput) */
#define USE_LIBINPUT_TOUCHPAD   0

/* Mouse cursor using libinput */
#define USE_LIBINPUT_MOUSE      1

/* Keyboard using libinput */
#define USE_LIBINPUT_KEYBOARD   1

/* FreeBSD touchscreen (/dev/input/eventX) - 已禁用，使用Wayland输入 */
#define USE_BSD_TOUCHPAD        0

/* EVDEV based drivers (libinput/evdev) - 已禁用，使用Wayland输入 */
#define USE_EVDEV               0

/* Mouse wheel encoder */
#define USE_MOUSEWHEEL          1

/*==================
 * WAYLAND SPECIFIC
 *==================*/

#if USE_WAYLAND

/* Wayland display name (通常是 "wayland-0") */
#define WAYLAND_DISPLAY_NAME    "wayland-0"

/* Window title */
#define WAYLAND_WINDOW_TITLE    "Bamboo Recognition System"

/* Enable window decorations */
#define WAYLAND_WINDOW_DECORATIONS  0

/* Window is fullscreen */
#define WAYLAND_FULLSCREEN      0

/* Window position (if not fullscreen) */
#define WAYLAND_WINDOW_X        0
#define WAYLAND_WINDOW_Y        0

/* Window size (if not fullscreen) */
#define WAYLAND_WINDOW_WIDTH    1920
#define WAYLAND_WINDOW_HEIGHT   1200

/* Use shared memory for buffer (recommended)
 * 设置为0以便优先使用EGL/DMABUF路径，减少与DeepStream的冲突 */
#define WAYLAND_USE_SHM         0

/* DPI scaling factor */
#define WAYLAND_DPI_SCALE       1

#endif

/*==================
 * OPTIMIZATION
 *==================*/

/* Enable GPU acceleration if available */
#define LV_USE_GPU_WAYLAND      1

/* Buffer swap method */
#define WAYLAND_SWAP_METHOD     0  /* 0: copy, 1: exchange */

/* V-sync enable */
#define WAYLAND_VSYNC           1

/* Multi-threading support */
#define WAYLAND_THREAD_SAFE     1

/*==================
 * DEBUG & LOGGING
 *==================*/

/* Enable debug output */
#define WAYLAND_DEBUG           0

/* Log level (0: silent, 1: error, 2: warn, 3: info, 4: debug) */
#define WAYLAND_LOG_LEVEL       2

/*==================
 * COMPATIBILITY
 *==================*/

/* 为了向后兼容，保留一些老的定义但设为禁用 */
#define USE_FBDEV               0
#define USE_DRM                 0  
#define USE_GBM                 0
#define USE_EVDEV               0

/* Jetson专用优化 */
#ifdef JETSON_ORIN_NX
# define WAYLAND_BUFFER_COUNT   3  /* Triple buffering for smooth playback */
# define WAYLAND_USE_DMABUF     1  /* DMA buffer for zero-copy */
#else
# define WAYLAND_BUFFER_COUNT   2  /* Double buffering */
# define WAYLAND_USE_DMABUF     0
#endif

/*==================
 * PERFORMANCE TUNING
 *==================*/

/* Rendering performance */
#define WAYLAND_RENDER_CACHE    1
#define WAYLAND_PARTIAL_UPDATE  1  /* Only update changed areas */

/* Memory optimization */
#define WAYLAND_MEMORY_POOL     1  /* Use memory pools */

/* Input performance */
#define WAYLAND_INPUT_SMOOTH    1  /* Smooth input processing */

#endif /*LV_DRV_CONF_H*/
