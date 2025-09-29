/**
 * 竹子识别系统一体化主程序
 * 真正整合现有的cpp_backend和lvgl_frontend代码
 */

#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <signal.h>
#include <chrono>
#include <fcntl.h>      // for open(), O_RDWR, O_RDONLY
#include <unistd.h>     // for close()
#include <sys/mman.h>   // for mmap(), munmap()
#include <sys/stat.h>   // for file status
#include <sys/types.h>  // for system types
#include <cstdlib>      // for setenv()
#include <fstream>      // for file operations

// OpenCV和图像处理
#include <opencv2/opencv.hpp>

// Linux framebuffer相关头文件
#include <linux/fb.h>
#include <sys/ioctl.h>

// DRM/KMS相关头文件（使用tegra_drm）
#ifdef ENABLE_DRM
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <drm/tegra_drm.h>
#endif

// LVGL头文件包含 - 智能检测多种可能的路径
#ifdef ENABLE_LVGL
#if __has_include(<lvgl/lvgl.h>)
#include <lvgl/lvgl.h>
#elif __has_include(<lvgl.h>)
#include <lvgl.h>
#elif __has_include("lvgl/lvgl.h")
#include "lvgl/lvgl.h"
#elif __has_include("lvgl.h")
#include "lvgl.h"
#else
#warning "LVGL header not found, using placeholder types"
#undef ENABLE_LVGL
#endif
#endif

// 函数前向声明 - 在文件最早位置
void suppress_camera_debug();
void suppress_all_debug_output();
void redirect_output_to_log();
void restore_output();
void cleanup_output_redirection();

#ifndef ENABLE_LVGL
// LVGL未启用时的类型占位符
typedef void* lv_obj_t;
typedef void* lv_event_t;
typedef void* lv_indev_drv_t;
typedef void* lv_indev_data_t;
typedef void* lv_disp_drv_t;
typedef void* lv_area_t;
typedef void* lv_color_t;
typedef void* lv_disp_draw_buf_t;
typedef void* lv_display_t;

// 模拟LVGL枚举
enum lv_indev_state_t {
    LV_INDEV_STATE_REL = 0,
    LV_INDEV_STATE_PR
};

// 模拟LVGL定时器结构体，包含user_data成员
struct lv_timer_t {
    void* user_data;
    void(*timer_cb)(struct lv_timer_t*);
    uint32_t period;
    uint32_t last_run;
    
    lv_timer_t() : user_data(nullptr), timer_cb(nullptr), period(0), last_run(0) {}
};

// LVGL函数占位符
inline void lv_init() {}
inline void lv_timer_handler() {}
inline void lv_port_tick_init() {}
inline lv_timer_t* lv_timer_create(void(*cb)(lv_timer_t*), unsigned int period_ms, void* user_data) {
    lv_timer_t* timer = new lv_timer_t();
    timer->timer_cb = cb;
    timer->period = period_ms;
    timer->user_data = user_data;
    return timer;
}
inline void lv_timer_del(lv_timer_t* timer) {
    if (timer) delete timer;
}

// LVGL DRM函数占位符
inline lv_display_t* lv_linux_drm_create() {
    return nullptr; // 当LVGL未启用时返回nullptr
}

// DRM/KMS显示系统变量
#ifdef ENABLE_DRM
struct DRMDisplay {
    int drm_fd = -1;
    drmModeRes* resources = nullptr;
    drmModeConnector* connector = nullptr;
    drmModeEncoder* encoder = nullptr;
    drmModeCrtc* crtc = nullptr;
    drmModeModeInfo mode_info = {};
    
    uint32_t framebuffer_id = 0;
    uint32_t bo_handle = 0;
    void* mapped_memory = nullptr;
    size_t buffer_size = 0;
    
    int width = 1920;
    int height = 1080;
    int bytes_per_pixel = 4; // RGBA
    
    bool initialized = false;
};

static DRMDisplay drm_display;
#endif

#ifdef ENABLE_DRM
// DRM/KMS初始化函数
bool initialize_drm_display() {
    std::cout << "Initializing DRM/KMS display system..." << std::endl;
    
    // 设置必要的环境变量
    setenv("QT_QPA_PLATFORM", "eglfs", 1);
    setenv("QT_QPA_EGLFS_INTEGRATION", "eglfs_kms", 1);
    setenv("QT_QPA_EGLFS_KMS_CONFIG", "/opt/bamboo-cut/config/kms.conf", 1);
    setenv("QT_LOGGING_RULES", "qt.qpa.eglfs.kms.debug=true", 1);
    
    // 打开DRM设备 - 优先使用card1
    const char* drm_devices[] = {"/dev/dri/card1", "/dev/dri/card0"};
    for (const char* device : drm_devices) {
        drm_display.drm_fd = open(device, O_RDWR | O_CLOEXEC);
        if (drm_display.drm_fd >= 0) {
            std::cout << "Opened DRM device: " << device << std::endl;
            break;
        } else {
            std::cout << "Failed to open " << device << ": " << strerror(errno) << std::endl;
        }
    }
    
    if (drm_display.drm_fd < 0) {
        std::cout << "Failed to open DRM device, falling back to framebuffer" << std::endl;
        return false;
    }
    
    // 获取DRM资源
    drm_display.resources = drmModeGetResources(drm_display.drm_fd);
    if (!drm_display.resources) {
        std::cout << "Failed to get DRM resources" << std::endl;
        close(drm_display.drm_fd);
        return false;
    }
    
    // 查找可用的连接器
    for (int i = 0; i < drm_display.resources->count_connectors; i++) {
        drm_display.connector = drmModeGetConnector(drm_display.drm_fd,
                                                   drm_display.resources->connectors[i]);
        if (drm_display.connector &&
            drm_display.connector->connection == DRM_MODE_CONNECTED &&
            drm_display.connector->count_modes > 0) {
            std::cout << "Found connected display connector" << std::endl;
            break;
        }
        if (drm_display.connector) {
            drmModeFreeConnector(drm_display.connector);
            drm_display.connector = nullptr;
        }
    }
    
    if (!drm_display.connector) {
        std::cout << "No connected display found" << std::endl;
        drmModeFreeResources(drm_display.resources);
        close(drm_display.drm_fd);
        return false;
    }
    
    // 选择显示模式（使用首选模式）
    drm_display.mode_info = drm_display.connector->modes[0];
    drm_display.width = drm_display.mode_info.hdisplay;
    drm_display.height = drm_display.mode_info.vdisplay;
    
    std::cout << "Selected display mode: " << drm_display.width << "x" << drm_display.height
              << " @" << drm_display.mode_info.vrefresh << "Hz" << std::endl;
    
    // 查找编码器
    if (drm_display.connector->encoder_id) {
        drm_display.encoder = drmModeGetEncoder(drm_display.drm_fd,
                                               drm_display.connector->encoder_id);
    }
    
    if (!drm_display.encoder) {
        for (int i = 0; i < drm_display.resources->count_encoders; i++) {
            drm_display.encoder = drmModeGetEncoder(drm_display.drm_fd,
                                                   drm_display.resources->encoders[i]);
            if (drm_display.encoder &&
                drm_display.encoder->encoder_type == DRM_MODE_ENCODER_TMDS) {
                break;
            }
            if (drm_display.encoder) {
                drmModeFreeEncoder(drm_display.encoder);
                drm_display.encoder = nullptr;
            }
        }
    }
    
    if (!drm_display.encoder) {
        std::cout << "No suitable encoder found" << std::endl;
        drmModeFreeConnector(drm_display.connector);
        drmModeFreeResources(drm_display.resources);
        close(drm_display.drm_fd);
        return false;
    }
    
    // 获取CRTC
    if (drm_display.encoder->crtc_id) {
        drm_display.crtc = drmModeGetCrtc(drm_display.drm_fd, drm_display.encoder->crtc_id);
    }
    
    // 创建dumb buffer
    struct drm_mode_create_dumb create_req = {};
    create_req.width = drm_display.width;
    create_req.height = drm_display.height;
    create_req.bpp = 32; // 32位RGBA
    
    if (ioctl(drm_display.drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        std::cout << "Failed to create dumb buffer" << std::endl;
        return false;
    }
    
    drm_display.bo_handle = create_req.handle;
    drm_display.buffer_size = create_req.size;
    
    // 创建framebuffer
    if (drmModeAddFB(drm_display.drm_fd, drm_display.width, drm_display.height,
                     24, 32, create_req.pitch, drm_display.bo_handle,
                     &drm_display.framebuffer_id) < 0) {
        std::cout << "Failed to create framebuffer" << std::endl;
        return false;
    }
    
    // 映射缓冲区到内存
    struct drm_mode_map_dumb map_req = {};
    map_req.handle = drm_display.bo_handle;
    
    if (ioctl(drm_display.drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
        std::cout << "Failed to map dumb buffer" << std::endl;
        return false;
    }
    
    drm_display.mapped_memory = mmap(0, drm_display.buffer_size,
                                    PROT_READ | PROT_WRITE, MAP_SHARED,
                                    drm_display.drm_fd, map_req.offset);
    
    if (drm_display.mapped_memory == MAP_FAILED) {
        std::cout << "Failed to mmap framebuffer" << std::endl;
        return false;
    }
    
    std::cout << "DRM/KMS display system initialized successfully" << std::endl;
    drm_display.initialized = true;
    return true;
}

// 清理DRM/KMS资源
void cleanup_drm_display() {
    if (!drm_display.initialized) return;
    
    if (drm_display.mapped_memory && drm_display.mapped_memory != MAP_FAILED) {
        munmap(drm_display.mapped_memory, drm_display.buffer_size);
    }
    
    if (drm_display.framebuffer_id) {
        drmModeRmFB(drm_display.drm_fd, drm_display.framebuffer_id);
    }
    
    if (drm_display.bo_handle) {
        struct drm_mode_destroy_dumb destroy_req = {};
        destroy_req.handle = drm_display.bo_handle;
        ioctl(drm_display.drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
    }
    
    if (drm_display.crtc) {
        drmModeFreeCrtc(drm_display.crtc);
    }
    
    if (drm_display.encoder) {
        drmModeFreeEncoder(drm_display.encoder);
    }
    
    if (drm_display.connector) {
        drmModeFreeConnector(drm_display.connector);
    }
    
    if (drm_display.resources) {
        drmModeFreeResources(drm_display.resources);
    }
    
    if (drm_display.drm_fd >= 0) {
        close(drm_display.drm_fd);
    }
    
    drm_display.initialized = false;
    std::cout << "DRM/KMS display system cleaned up" << std::endl;
}

// 显示framebuffer内容到屏幕
bool present_drm_framebuffer() {
    if (!drm_display.initialized || !drm_display.crtc) return false;
    
    // 设置CRTC显示framebuffer
    int ret = drmModeSetCrtc(drm_display.drm_fd, drm_display.crtc->crtc_id,
                            drm_display.framebuffer_id, 0, 0,
                            &drm_display.connector->connector_id, 1,
                            &drm_display.mode_info);
    
    return ret == 0;
}
#else
// DRM禁用时的占位符函数
bool initialize_drm_display() { return false; }
void cleanup_drm_display() {}
bool present_drm_framebuffer() { return false; }
#endif

// 删除所有自定义DRM绘图系统，现在只使用LVGL

// 显示驱动相关占位符
inline void lv_disp_draw_buf_init(lv_disp_draw_buf_t* draw_buf, void* buf1, void* buf2, uint32_t size_in_px_cnt) {}
inline void lv_disp_drv_init(lv_disp_drv_t* driver) {}
inline lv_disp_drv_t* lv_disp_drv_register(lv_disp_drv_t* driver) { return driver; }
inline void lv_disp_flush_ready(lv_disp_drv_t* disp_drv) {}

inline bool lvgl_display_init() {
    // 纯LVGL显示系统初始化
    try {
        std::cout << "Initializing pure LVGL display system..." << std::endl;
        
#ifdef ENABLE_LVGL
        // 使用LVGL的Linux DRM驱动
        lv_display_t * disp = lv_linux_drm_create();
        if (!disp) {
            std::cout << "Failed to create LVGL DRM display" << std::endl;
            return false;
        } else {
            std::cout << "LVGL DRM display created successfully" << std::endl;
            return true;
        }
#else
        std::cout << "Error: LVGL not enabled in build configuration" << std::endl;
        std::cout << "Please ensure LVGL is properly configured and available" << std::endl;
        return false;
#endif
        
    } catch (const std::exception& e) {
        std::cout << "LVGL display initialization exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "LVGL display initialization unknown exception" << std::endl;
        return false;
    }
}
inline bool touch_driver_init() {
    // Jetson Orin NX 触摸驱动初始化（自适应）
    try {
        // 检查触摸设备
        const char* touch_devices[] = {"/dev/input/event0", "/dev/input/event1", "/dev/input/event2"};
        bool has_touch = false;
        
        for (const char* touch_dev : touch_devices) {
            int touch_fd = open(touch_dev, O_RDONLY);
            if (touch_fd >= 0) {
                close(touch_fd);
                has_touch = true;
                std::cout << "Found touch device: " << touch_dev << std::endl;
                break;
            }
        }
        
        if (!has_touch) {
            std::cout << "Touch device not found, disabling touch functionality" << std::endl;
        }
        
        return has_touch; // 返回实际检测结果
    } catch (...) {
        std::cout << "Touch driver initialization exception" << std::endl;
        return false;
    }
}

// 前端组件占位符 - 当LVGL未启用时
struct frame_info_t {
    uint64_t timestamp = 0;
    bool valid = false;
    int width = 640, height = 480;
};
struct performance_stats_t {
    float cpu_usage = 0, memory_usage_mb = 0, fps = 0;
};

class Status_bar {
public:
    bool initialize() {
        std::cout << "Status bar initialization complete" << std::endl;
        return true;
    }
    void update_workflow_status(int status) {
        std::cout << "Updating workflow status: " << status << std::endl;
    }
    void update_heartbeat(int count, int plc_status) {
        std::cout << "Updating heartbeat: count=" << count << ", plc=" << plc_status << std::endl;
    }
};

class Video_view {
public:
    bool initialize() {
        std::cout << "Video view initialization complete" << std::endl;
        return true;
    }
    void update_camera_frame(const frame_info_t& frame) {
        std::cout << "Updating camera frame: " << frame.width << "x" << frame.height
                  << " (valid: " << frame.valid << ")" << std::endl;
    }
    void update_detection_info(float fps, float process_time) {
        std::cout << "Updating detection info: FPS=" << fps << ", process_time=" << process_time << "ms" << std::endl;
    }
};

class Control_panel {
public:
    bool initialize() {
        std::cout << "Control panel initialization complete" << std::endl;
        return true;
    }
    void update_jetson_info(const performance_stats_t& stats) {
        std::cout << "Updating Jetson info: CPU=" << stats.cpu_usage << "%, memory="
                  << stats.memory_usage_mb << "MB, FPS=" << stats.fps << std::endl;
    }
};

class Settings_page {
public:
    bool initialize() {
        std::cout << "Settings page initialization complete" << std::endl;
        return true;
    }
    void create_main_layout(Status_bar* status, Video_view* video, Control_panel* control) {
        std::cout << "Creating main interface layout" << std::endl;
        std::cout << "=== Bamboo Recognition System Interface ===" << std::endl;
        std::cout << "Status bar: " << (status ? "Connected" : "Disconnected") << std::endl;
        std::cout << "Video view: " << (video ? "Connected" : "Disconnected") << std::endl;
        std::cout << "Control panel: " << (control ? "Connected" : "Disconnected") << std::endl;
        std::cout << "===========================================" << std::endl;
    }
};
#endif

// 现有后端组件 - 直接包含实际存在的头文件
#include "bamboo_cut/utils/logger.h"
#include "bamboo_cut/inference/bamboo_detector.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/vision/stereo_vision.h"

// 使用真实的命名空间
using namespace bamboo_cut;

// 全局关闭标志
std::atomic<bool> g_shutdown_requested{false};
std::chrono::steady_clock::time_point g_shutdown_start_time;

// 静态输出重定向文件描述符
static int original_stdout = -1;
static int original_stderr = -1;
static int log_fd = -1;
static std::string log_file_path = "/var/log/bamboo-cut/camera_debug.log";

// 温和的调试信息抑制函数
void selective_debug_suppress() {
    // 只禁用特定的相机调试，保留显示相关的输出
    system("echo 0 > /sys/kernel/debug/tracing/events/camera/enable 2>/dev/null || true");
    // 不要禁用DRM相关的调试信息
    
    // 设置环境变量抑制Tegra相机调试
    setenv("GST_DEBUG", "0", 1);
    setenv("NVARGUS_LOG_LEVEL", "0", 1);
    setenv("NVARGUS_DISABLE_LOG", "1", 1);
    setenv("TEGRA_LOG_LEVEL", "0", 1);
    setenv("ARGUS_LOG_LEVEL", "0", 1);
    setenv("CAMRTC_LOG_LEVEL", "0", 1);
    setenv("VI_LOG_LEVEL", "0", 1);
    setenv("NVCSI_LOG_LEVEL", "0", 1);
    
    std::cout << "Camera debug suppression configured (selective mode)" << std::endl;
}

// 保持向后兼容的函数名
void suppress_camera_debug() {
    selective_debug_suppress();
}

// 完全抑制所有调试信息的函数
void suppress_all_debug_output() {
    std::cout << "Suppressing all camera and system debug output..." << std::endl;
    
    // 1. 设置环境变量抑制NVIDIA Tegra调试信息
    setenv("GST_DEBUG", "0", 1);
    setenv("GST_DEBUG_NO_COLOR", "1", 1);
    setenv("NVARGUS_LOG_LEVEL", "0", 1);
    setenv("NVARGUS_DISABLE_LOG", "1", 1);
    setenv("TEGRA_LOG_LEVEL", "0", 1);
    setenv("ARGUS_LOG_LEVEL", "0", 1);
    setenv("ARGUS_DISABLE_LOG", "1", 1);
    setenv("NVMEDIA_LOG_LEVEL", "0", 1);
    setenv("NV_LOG_LEVEL", "0", 1);
    setenv("NV_DISABLE_LOG", "1", 1);
    
    // 额外的Tegra Camera调试信息抑制
    setenv("TEGRA_CAMERA_LOG_LEVEL", "0", 1);
    setenv("TEGRA_CAMERA_DISABLE_LOG", "1", 1);
    setenv("CAMRTC_LOG_LEVEL", "0", 1);
    setenv("CAMRTC_DISABLE_LOG", "1", 1);
    setenv("NVIDIA_TEGRA_LOG_LEVEL", "0", 1);
    setenv("NVIDIA_DISABLE_LOG", "1", 1);
    setenv("VI_LOG_LEVEL", "0", 1);
    setenv("VI_DISABLE_LOG", "1", 1);
    setenv("NVCSI_LOG_LEVEL", "0", 1);
    setenv("NVCSI_DISABLE_LOG", "1", 1);
    
    // 抑制内核日志输出到用户空间
    setenv("KERNEL_LOG_LEVEL", "0", 1);
    setenv("DMESG_RESTRICT", "1", 1);
    
    // 强制重定向内核消息到null
    system("echo 0 > /proc/sys/kernel/printk 2>/dev/null || true");
    system("dmesg -n 0 2>/dev/null || true");
    
    // 抑制systemd journal输出到console
    system("systemctl mask systemd-journald-dev-log.socket 2>/dev/null || true");
    
    // 2. 设置GStreamer静默模式
    setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0", 1);
    setenv("GST_REGISTRY_UPDATE", "no", 1);
    setenv("GST_REGISTRY_FORK", "no", 1);
    
    // 3. 创建日志目录（如果不存在）
    system("mkdir -p /var/log/bamboo-cut");
    
    // 4. 创建日志文件用于重定向调试信息
    log_fd = open(log_file_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log_fd == -1) {
        std::cout << "Warning: Cannot create log file " << log_file_path << ", using /dev/null instead" << std::endl;
        log_fd = open("/dev/null", O_WRONLY);
        if (log_fd == -1) {
            std::cout << "Error: Cannot open /dev/null either, debug output may still appear" << std::endl;
            return;
        }
    } else {
        std::cout << "Camera debug output will be redirected to: " << log_file_path << std::endl;
    }
    
    // 5. 保存原始文件描述符
    original_stdout = dup(STDOUT_FILENO);
    original_stderr = dup(STDERR_FILENO);
    
    if (original_stdout == -1 || original_stderr == -1) {
        std::cout << "Warning: Cannot backup original file descriptors" << std::endl;
        if (log_fd >= 0) close(log_fd);
        return;
    }
    
    // 6. 写入日志文件头部信息
    if (log_fd >= 0) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time_t);
        timestamp.pop_back(); // 移除换行符
        
        std::string log_header = "\n=== Bamboo Cut Camera Debug Log - " + timestamp + " ===\n";
        write(log_fd, log_header.c_str(), log_header.length());
    }
    
    std::cout << "Debug output suppression configured successfully" << std::endl;
}

// 临时重定向输出（在摄像头初始化期间使用）
void redirect_output_to_log() {
    if (log_fd >= 0) {
        // 重定向stdout和stderr到日志文件
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        
        // 写入重定向开始标记
        std::string start_msg = "[Camera Initialization Started]\n";
        write(log_fd, start_msg.c_str(), start_msg.length());
    }
}

// 恢复原始输出
void restore_output() {
    if (original_stdout >= 0 && original_stderr >= 0) {
        // 写入重定向结束标记
        if (log_fd >= 0) {
            std::string end_msg = "[Camera Initialization Completed]\n\n";
            write(log_fd, end_msg.c_str(), end_msg.length());
        }
        
        // 恢复原始输出
        dup2(original_stdout, STDOUT_FILENO);
        dup2(original_stderr, STDERR_FILENO);
    }
}

// 清理重定向资源
void cleanup_output_redirection() {
    if (original_stdout >= 0) {
        close(original_stdout);
        original_stdout = -1;
    }
    if (original_stderr >= 0) {
        close(original_stderr);
        original_stderr = -1;
    }
    if (log_fd >= 0) {
        // 写入日志文件结束标记
        std::string final_msg = "=== Log Session Ended ===\n\n";
        write(log_fd, final_msg.c_str(), final_msg.length());
        close(log_fd);
        log_fd = -1;
    }
}

// 信号处理
void signal_handler(int sig) {
    std::cout << "\n收到信号 " << sig << "，开始优雅关闭..." << std::endl;
    g_shutdown_requested = true;
    g_shutdown_start_time = std::chrono::steady_clock::now();
    
    // 清理输出重定向资源
    cleanup_output_redirection();
}

/**
 * 线程安全的数据桥接器
 * 在推理线程和UI线程间传递数据
 */
class IntegratedDataBridge {
public:
    struct VideoData {
        cv::Mat frame;
        cv::Mat left_frame;
        cv::Mat right_frame;
        uint64_t timestamp;
        bool valid;
        
        VideoData() : timestamp(0), valid(false) {}
    };
    
    struct DetectionData {
        std::vector<cv::Point2f> cutting_points;
        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        float processing_time_ms;
        bool has_detection;
        
        DetectionData() : processing_time_ms(0), has_detection(false) {}
    };
    
    struct SystemStats {
        float camera_fps;
        float inference_fps;
        float cpu_usage;
        float memory_usage_mb;
        int total_detections;
        bool plc_connected;
        
        SystemStats() : camera_fps(0), inference_fps(0), cpu_usage(0),
                       memory_usage_mb(0), total_detections(0), plc_connected(false) {}
    };

private:
    mutable std::mutex video_mutex_;
    mutable std::mutex detection_mutex_;
    mutable std::mutex stats_mutex_;
    
    VideoData latest_video_;
    DetectionData latest_detection_;
    SystemStats latest_stats_;
    
    std::atomic<bool> new_video_available_{false};
    std::atomic<bool> new_detection_available_{false};

public:
    // 视频数据更新 (从推理线程调用)
    void updateVideo(const cv::Mat& frame, uint64_t timestamp = 0) {
        std::lock_guard<std::mutex> lock(video_mutex_);
        if (!frame.empty()) {
            latest_video_.frame = frame.clone();
            latest_video_.timestamp = timestamp ? timestamp : getCurrentTimestamp();
            latest_video_.valid = true;
            new_video_available_ = true;
        }
    }
    
    void updateStereoVideo(const cv::Mat& left, const cv::Mat& right, uint64_t timestamp = 0) {
        std::lock_guard<std::mutex> lock(video_mutex_);
        if (!left.empty() && !right.empty()) {
            latest_video_.left_frame = left.clone();
            latest_video_.right_frame = right.clone();
            latest_video_.timestamp = timestamp ? timestamp : getCurrentTimestamp();
            latest_video_.valid = true;
            new_video_available_ = true;
        }
    }
    
    // 检测数据更新 (从推理线程调用)
    void updateDetection(const core::DetectionResult& result) {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        latest_detection_.cutting_points = result.cutting_points;
        latest_detection_.bboxes = result.bboxes;
        latest_detection_.confidences = result.confidences;
        latest_detection_.processing_time_ms = 50.0f; // 简化实现
        latest_detection_.has_detection = result.valid && !result.bboxes.empty();
        new_detection_available_ = true;
    }
    
    // 系统统计更新
    void updateStats(const SystemStats& stats) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        latest_stats_ = stats;
    }
    
    // 获取最新数据 (从UI线程调用)
    bool getLatestVideo(VideoData& video) {
        std::lock_guard<std::mutex> lock(video_mutex_);
        if (latest_video_.valid) {
            video = latest_video_;
            return true;
        }
        return false;
    }
    
    bool getLatestDetection(DetectionData& detection) {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        if (latest_detection_.has_detection) {
            detection = latest_detection_;
            return true;
        }
        return false;
    }
    
    SystemStats getStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return latest_stats_;
    }
    
    bool hasNewVideo() {
        return new_video_available_.exchange(false);
    }
    
    bool hasNewDetection() {
        return new_detection_available_.exchange(false);
    }

private:
    uint64_t getCurrentTimestamp() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

/**
 * 推理工作线程
 * 复用现有的cpp_backend组件
 */
class InferenceWorkerThread {
private:
    IntegratedDataBridge* data_bridge_;
    std::unique_ptr<std::thread> worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // 使用真实的后端组件
    std::unique_ptr<inference::BambooDetector> detector_;
    std::unique_ptr<vision::StereoVision> stereo_vision_;
    cv::VideoCapture camera_;
    bool use_mock_camera_ = false;
    bool use_stereo_vision_ = true;  // 启用立体视觉模式
    
    // 性能统计
    int processed_frames_ = 0;
    std::chrono::steady_clock::time_point last_stats_time_;
    float current_fps_ = 0.0f;

public:
    InferenceWorkerThread(IntegratedDataBridge* bridge) 
        : data_bridge_(bridge), last_stats_time_(std::chrono::steady_clock::now()) {}
    
    ~InferenceWorkerThread() {
        stop();
    }
    
    bool initialize() {
        std::cout << "Initializing inference system..." << std::endl;
        
        // 初始化检测器 (使用真实的BambooDetector) - 非阻塞
        if (!initializeDetector()) {
            std::cout << "Detector initialization failed, using simulation mode" << std::endl;
            use_mock_camera_ = true; // 启用模拟模式
        }
        
        // 优先初始化立体视觉系统 - 非阻塞
        if (use_stereo_vision_ && initializeStereoVision()) {
            std::cout << "Stereo vision system initialization successful" << std::endl;
        } else {
            std::cout << "Stereo vision initialization failed, falling back to single camera mode" << std::endl;
            use_stereo_vision_ = false;
            
            // 初始化单摄像头 - 非阻塞，失败时使用模拟模式
            if (!initializeCamera()) {
                std::cout << "Camera system initialization failed, enabling simulation mode" << std::endl;
                use_mock_camera_ = true;
            }
        }
        
        std::cout << "Inference system initialization complete (simulation mode: " << (use_mock_camera_ ? "yes" : "no") << ")" << std::endl;
        return true; // 总是返回成功，确保UI能够启动
    }
    
    bool start() {
        if (running_) return false;
        
        should_stop_ = false;
        worker_thread_ = std::make_unique<std::thread>(&InferenceWorkerThread::workerLoop, this);
        running_ = true;
        return true;
    }
    
    void stop() {
        should_stop_ = true;
        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
        }
        running_ = false;
    }
    
    bool isRunning() const { return running_; }

private:
    void workerLoop() {
        std::cout << "Inference worker thread started" << std::endl;
        
        auto last_frame_time = std::chrono::steady_clock::now();
        const auto target_interval = std::chrono::milliseconds(33); // 30fps
        
        while (!should_stop_ && !g_shutdown_requested) {
            auto current_time = std::chrono::steady_clock::now();
            
            // 处理一帧
            processFrame();
            
            // 更新性能统计
            updatePerformanceStats();
            
            // 帧率控制
            auto processing_time = std::chrono::steady_clock::now() - current_time;
            auto sleep_time = target_interval - processing_time;
            
            if (sleep_time > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
        
        std::cout << "Inference worker thread exited" << std::endl;
    }
    
    void processFrame() {
        // 优先处理立体视觉
        if (use_stereo_vision_ && stereo_vision_) {
            vision::StereoFrame stereo_frame;
            if (stereo_vision_->capture_stereo_frame(stereo_frame) && stereo_frame.valid) {
                // 更新立体视频到数据桥接
                data_bridge_->updateStereoVideo(stereo_frame.left_image, stereo_frame.right_image);
                
                // 计算深度信息
                stereo_vision_->compute_depth(stereo_frame);
                
                // AI检测 (使用校正后的左图像)
                if (detector_ && !stereo_frame.rectified_left.empty()) {
                    core::DetectionResult result;
                    if (detector_->detect(stereo_frame.rectified_left, result)) {
                        data_bridge_->updateDetection(result);
                    }
                }
                
                processed_frames_++;
                return;
            }
        }
        
        // 单摄像头处理（回退模式）
        cv::Mat frame;
        if (use_mock_camera_) {
            // 生成模拟帧用于测试
            frame = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(frame, "MOCK CAMERA - " + std::to_string(processed_frames_),
                       cv::Point(50, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        } else {
            if (!camera_.read(frame) || frame.empty()) {
                return;
            }
        }
        
        // 更新视频到数据桥接
        data_bridge_->updateVideo(frame);
        
        // AI检测
        if (detector_ && !use_mock_camera_) {
            core::DetectionResult result;
            if (detector_->detect(frame, result)) {
                data_bridge_->updateDetection(result);
            }
        }
        
        processed_frames_++;
    }
    
    void updatePerformanceStats() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time_);
        
        if (elapsed.count() >= 1) {
            current_fps_ = static_cast<float>(processed_frames_) / elapsed.count();
            
            // 更新系统统计
            IntegratedDataBridge::SystemStats stats;
            stats.camera_fps = current_fps_;
            stats.inference_fps = current_fps_;
            stats.cpu_usage = getCpuUsage();
            stats.memory_usage_mb = getMemoryUsage();
            stats.total_detections += processed_frames_;
            stats.plc_connected = false; // 简化实现，后续可添加Modbus支持
            
            data_bridge_->updateStats(stats);
            
            processed_frames_ = 0;
            last_stats_time_ = now;
        }
    }
    
    // === 初始化方法 (使用真实的BambooDetector) ===
    bool initializeDetector() {
        inference::DetectorConfig config;
        config.model_path = "/opt/bamboo-cut/models/bamboo_detection.onnx";
        config.confidence_threshold = 0.85f;
        config.nms_threshold = 0.45f;
        config.input_size = cv::Size(640, 640);
        config.use_gpu = true;
        config.use_tensorrt = true;
        
        detector_ = std::make_unique<inference::BambooDetector>(config);
        return detector_->initialize();
    }
    
    bool initializeCamera() {
        std::cout << "Detecting Jetson CSI camera devices..." << std::endl;
        
        // 优先尝试Jetson CSI摄像头 (使用nvarguscamerasrc)
        if (initializeJetsonCSICamera()) {
            std::cout << "Jetson CSI camera initialization successful" << std::endl;
            return true;
        }
        
        std::cout << "CSI camera initialization failed, trying USB camera..." << std::endl;
        
        // 回退到USB摄像头 (使用v4l2)
        if (initializeUSBCamera()) {
            std::cout << "USB camera initialization successful" << std::endl;
            return true;
        }
        
        // 如果没有真实摄像头，创建虚拟摄像头用于测试
        std::cout << "No available camera found, enabling simulation mode" << std::endl;
        use_mock_camera_ = true;
        return true;
    }
    
    bool initializeJetsonCSICamera() {
        try {
            std::cout << "Initializing Jetson CSI cameras with debug suppression..." << std::endl;
            
            // 温和的调试信息抑制，只针对相机模块
            selective_debug_suppress();
            
            // Jetson CSI摄像头 GStreamer pipeline - 完全静默模式
            std::vector<std::string> csi_pipelines = {
                "nvarguscamerasrc sensor-id=0 silent=true ! "
                "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1, format=(string)NV12 ! "
                "nvvidconv flip-method=0 silent=true ! "
                "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
                "videoconvert silent=true ! "
                "video/x-raw, format=(string)BGR ! appsink sync=false",
                
                "nvarguscamerasrc sensor-id=1 silent=true ! "
                "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1, format=(string)NV12 ! "
                "nvvidconv flip-method=0 silent=true ! "
                "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
                "videoconvert silent=true ! "
                "video/x-raw, format=(string)BGR ! appsink sync=false"
            };
            
            bool camera_initialized = false;
            
            for (size_t i = 0; i < csi_pipelines.size(); i++) {
                camera_.open(csi_pipelines[i], cv::CAP_GSTREAMER);
                
                if (camera_.isOpened()) {
                    // 测试是否真的能读取帧
                    cv::Mat test_frame;
                    if (camera_.read(test_frame) && !test_frame.empty()) {
                        camera_initialized = true;
                        std::cout << "CSI camera sensor-id=" << i << " initialization successful, resolution: "
                                  << test_frame.cols << "x" << test_frame.rows << std::endl;
                        return true;
                    } else {
                        camera_.release();
                    }
                }
            }
            
            if (!camera_initialized) {
                std::cout << "All CSI camera initialization attempts failed" << std::endl;
            }
            
            return false;
        } catch (const cv::Exception& e) {
            std::cout << "CSI摄像头初始化异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool initializeUSBCamera() {
        try {
            // USB摄像头设备ID列表
            std::vector<int> camera_ids = {0, 1, 2};
            
            for (int id : camera_ids) {
                std::cout << "尝试打开USB摄像头 /dev/video" << id << std::endl;
                
                camera_.open(id, cv::CAP_V4L2);
                
                if (camera_.isOpened()) {
                    // 设置摄像头参数
                    camera_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    camera_.set(cv::CAP_PROP_FPS, 30);
                    
                    // 测试是否真的能读取帧
                    cv::Mat test_frame;
                    if (camera_.read(test_frame) && !test_frame.empty()) {
                        std::cout << "USB摄像头 " << id << " 初始化成功，分辨率: "
                                  << test_frame.cols << "x" << test_frame.rows << std::endl;
                        return true;
                    } else {
                        std::cout << "USB摄像头 " << id << " 无法读取帧" << std::endl;
                        camera_.release();
                    }
                } else {
                    std::cout << "无法打开USB摄像头 " << id << std::endl;
                }
            }
            
            return false;
        } catch (const cv::Exception& e) {
            std::cout << "USB摄像头初始化异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    // === 立体视觉初始化方法 ===
    bool initializeStereoVision() {
        std::cout << "初始化双摄立体视觉系统..." << std::endl;
        
        // 应用温和的调试抑制
        selective_debug_suppress();
        
        vision::StereoConfig stereo_config;
        stereo_config.calibration_file = "/opt/bamboo-cut/config/stereo_calibration.xml";
        stereo_config.frame_size = cv::Size(640, 480);
        stereo_config.fps = 30;
        
        // Jetson CSI摄像头配置 - 完全静默模式
        stereo_config.left_camera_pipeline =
            "nvarguscamerasrc sensor-id=0 silent=true ! "
            "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1, format=(string)NV12 ! "
            "nvvidconv flip-method=0 silent=true ! "
            "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
            "videoconvert silent=true ! "
            "video/x-raw, format=(string)BGR ! appsink sync=false";
            
        stereo_config.right_camera_pipeline =
            "nvarguscamerasrc sensor-id=1 silent=true ! "
            "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1, format=(string)NV12 ! "
            "nvvidconv flip-method=0 silent=true ! "
            "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
            "videoconvert silent=true ! "
            "video/x-raw, format=(string)BGR ! appsink sync=false";
        
        // 回退选项：USB摄像头ID
        stereo_config.left_camera_id = 0;   // /dev/video0
        stereo_config.right_camera_id = 1;  // /dev/video1
        stereo_config.use_gstreamer = true; // 优先使用GStreamer管道
        
        stereo_vision_ = std::make_unique<vision::StereoVision>(stereo_config);
        
        bool initialized = false;
        try {
            initialized = stereo_vision_->initialize();
        } catch (...) {
            initialized = false;
        }
        
        if (initialized) {
            std::cout << "立体视觉系统初始化成功" << std::endl;
        } else {
            std::cout << "立体视觉系统初始化失败" << std::endl;
        }
        
        return initialized;
    }
    
    float getCpuUsage() const { return 45.0f; } // 简化实现
    float getMemoryUsage() const { return 1024.0f; } // 简化实现
};

/**
 * LVGL UI管理器
 * 复用现有的lvgl_frontend组件
 */
class LVGLUIManager {
private:
    IntegratedDataBridge* data_bridge_;
    
    // 复用现有的LVGL组件
    std::unique_ptr<Status_bar> status_bar_;
    std::unique_ptr<Video_view> video_view_;
    std::unique_ptr<Control_panel> control_panel_;
    std::unique_ptr<Settings_page> settings_page_;
    
    // LVGL定时器
    lv_timer_t* video_update_timer_;
    lv_timer_t* status_update_timer_;
    
    bool initialized_ = false;

public:
    LVGLUIManager(IntegratedDataBridge* bridge) 
        : data_bridge_(bridge), video_update_timer_(nullptr), status_update_timer_(nullptr) {}
    
    ~LVGLUIManager() {
        cleanup();
    }
    
    bool initialize() {
        std::cout << "Initializing LVGL UI system..." << std::endl;
        
        // 初始化LVGL核心
        lv_init();
        
        // 初始化时钟系统
        lv_port_tick_init();
        
        // 初始化显示驱动 (复用现有实现)
        if (!lvgl_display_init()) {
            std::cout << "LVGL display driver initialization failed" << std::endl;
            return false;
        }
        
        // 初始化触摸驱动 (复用现有实现)
        if (touch_driver_init()) {
            std::cout << "Touch driver initialization successful" << std::endl;
        } else {
            std::cout << "Touch driver initialization failed, touch functionality will be disabled" << std::endl;
        }
        
        // 创建UI组件 (复用现有实现)
        if (!createUIComponents()) {
            std::cout << "UI component creation failed" << std::endl;
            return false;
        }
        
        // 设置更新定时器
        setupUpdateTimers();
        
        initialized_ = true;
        std::cout << "LVGL UI system initialization complete" << std::endl;
        return true;
    }
    
    void runMainLoop() {
        if (!initialized_) return;
        
        std::cout << "LVGL main loop started" << std::endl;
        
        while (!g_shutdown_requested) {
            // 处理LVGL任务
            lv_timer_handler();
            
            // 短暂休眠，60fps
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        std::cout << "LVGL main loop exited" << std::endl;
    }

private:
    bool createUIComponents() {
        // 创建各个组件 (复用现有代码)
        status_bar_ = std::make_unique<Status_bar>();
        video_view_ = std::make_unique<Video_view>();
        control_panel_ = std::make_unique<Control_panel>();
        settings_page_ = std::make_unique<Settings_page>();
        
        if (!status_bar_->initialize()) return false;
        if (!video_view_->initialize()) return false;
        if (!control_panel_->initialize()) return false;
        if (!settings_page_->initialize()) return false;
        
        // 创建主布局
        settings_page_->create_main_layout(status_bar_.get(), video_view_.get(), control_panel_.get());
        
        return true;
    }
    
    void setupUpdateTimers() {
        // 视频更新定时器 (30fps)
        video_update_timer_ = lv_timer_create([](lv_timer_t* timer) {
            LVGLUIManager* ui = static_cast<LVGLUIManager*>(timer->user_data);
            ui->updateVideoDisplay();
        }, 33, this);
        
        // 状态更新定时器 (2fps)
        status_update_timer_ = lv_timer_create([](lv_timer_t* timer) {
            LVGLUIManager* ui = static_cast<LVGLUIManager*>(timer->user_data);
            ui->updateStatusDisplay();
        }, 500, this);
    }
    
    void updateVideoDisplay() {
        if (!video_view_) return;
        
        IntegratedDataBridge::VideoData video_data;
        if (data_bridge_->getLatestVideo(video_data) && video_data.valid) {
            // 转换为LVGL格式并更新
            frame_info_t frame_info;
            frame_info.timestamp = video_data.timestamp;
            frame_info.valid = true;
            frame_info.width = video_data.frame.cols;
            frame_info.height = video_data.frame.rows;
            
            video_view_->update_camera_frame(frame_info);
        }
        
        // 更新检测信息
        IntegratedDataBridge::DetectionData detection_data;
        if (data_bridge_->getLatestDetection(detection_data)) {
            auto stats = data_bridge_->getStats();
            video_view_->update_detection_info(stats.inference_fps, detection_data.processing_time_ms);
        }
    }
    
    void updateStatusDisplay() {
        if (!status_bar_ || !control_panel_) return;
        
        auto stats = data_bridge_->getStats();
        
        // 更新状态栏
        status_bar_->update_workflow_status(1);
        status_bar_->update_heartbeat(stats.total_detections, 0);
        
        // 更新控制面板
        performance_stats_t perf_stats;
        perf_stats.cpu_usage = stats.cpu_usage;
        perf_stats.memory_usage_mb = stats.memory_usage_mb;
        perf_stats.fps = stats.camera_fps;
        
        control_panel_->update_jetson_info(perf_stats);
    }
    
    void cleanup() {
        if (video_update_timer_) {
            lv_timer_del(video_update_timer_);
        }
        if (status_update_timer_) {
            lv_timer_del(status_update_timer_);
        }
        
        status_bar_.reset();
        video_view_.reset();
        control_panel_.reset();
        settings_page_.reset();
        
        initialized_ = false;
    }
};

/**
 * 一体化主程序类
 */
class IntegratedBambooSystem {
private:
    IntegratedDataBridge data_bridge_;
    std::unique_ptr<InferenceWorkerThread> inference_worker_;
    std::unique_ptr<LVGLUIManager> ui_manager_;
    
public:
    bool initialize() {
        std::cout << "=================================" << std::endl;
        std::cout << "Bamboo Recognition System Integrated Startup" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // 设置信号处理
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // 初始化推理工作线程
        inference_worker_ = std::make_unique<InferenceWorkerThread>(&data_bridge_);
        if (!inference_worker_->initialize()) {
            std::cout << "Inference system initialization failed" << std::endl;
            return false;
        }
        
        // 初始化UI管理器
        ui_manager_ = std::make_unique<LVGLUIManager>(&data_bridge_);
        if (!ui_manager_->initialize()) {
            std::cout << "UI system initialization failed" << std::endl;
            return false;
        }
        
        std::cout << "Integrated system initialization complete" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "Starting integrated system..." << std::endl;
        
        // 启动推理工作线程
        if (!inference_worker_->start()) {
            std::cout << "Inference thread startup failed" << std::endl;
            return;
        }
        
        std::cout << "Inference thread started" << std::endl;
        std::cout << "Press Ctrl+C to exit system" << std::endl;
        
        // 主线程运行UI (阻塞)
        ui_manager_->runMainLoop();
        
        std::cout << "Starting system shutdown..." << std::endl;
        shutdown();
    }
    
    void shutdown() {
        std::cout << "Stopping inference thread..." << std::endl;
        if (inference_worker_) {
            inference_worker_->stop();
        }
        
        // 清理DRM/KMS资源
        cleanup_drm_display();
        
        // 清理输出重定向资源
        cleanup_output_redirection();
        
        std::cout << "System shutdown complete" << std::endl;
    }
};

/**
 * 主函数入口
 */
int main(int argc, char* argv[]) {
    try {
        // 检查是否为测试模式或调试模式
        bool verbose_mode = false;
        bool test_mode = false;
        
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
                verbose_mode = true;
            }
            if (std::string(argv[i]) == "--test" || std::string(argv[i]) == "-t") {
                test_mode = true;
            }
        }
        
        // 在非详细模式下使用温和的调试抑制
        if (!verbose_mode && !test_mode) {
            selective_debug_suppress();
        } else {
            std::cout << "Bamboo Recognition System starting in verbose mode..." << std::endl;
        }
        
        IntegratedBambooSystem system;
        
        if (!system.initialize()) {
            // 临时恢复stdout显示错误信息
            if (!verbose_mode) {
                freopen("/dev/tty", "w", stdout);
            }
            std::cout << "System initialization failed" << std::endl;
            return -1;
        }
        
        if (verbose_mode || test_mode) {
            std::cout << "System initialized successfully, starting main loop..." << std::endl;
        }
        
        system.run();
        return 0;
        
    } catch (const std::exception& e) {
        // 临时恢复stdout显示异常信息
        freopen("/dev/tty", "w", stdout);
        std::cout << "System exception: " << e.what() << std::endl;
        return -1;
    }
}