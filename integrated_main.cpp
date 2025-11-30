/**
 * 绔瑰瓙璇嗗埆绯荤粺涓€浣撳寲涓荤▼搴?
 * 鐪熸鏁村悎鐜版湁鐨刢pp_backend鍜宭vgl_frontend浠ｇ爜
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
#include <algorithm>    // for std::max
// OpenCV鍜屽浘鍍忓鐞?
#include <opencv2/opencv.hpp>

#ifdef ENABLE_WAYLAND
#include <wayland-client.h>
#endif


// LVGL澶存枃浠跺寘鍚?- 鏅鸿兘妫€娴嬪绉嶅彲鑳界殑璺緞
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

// 鍑芥暟鍓嶅悜澹版槑 - 鍦ㄦ枃浠舵渶鏃╀綅缃?
void suppress_camera_debug();
void suppress_all_debug_output();
void redirect_output_to_log();
void restore_output();
void cleanup_output_redirection();

#ifndef ENABLE_LVGL
// LVGL鏈惎鐢ㄦ椂鐨勭被鍨嬪崰浣嶇
typedef void* lv_obj_t;
typedef void* lv_event_t;
typedef void* lv_indev_drv_t;
typedef void* lv_indev_data_t;
typedef void* lv_disp_drv_t;
typedef void* lv_area_t;
typedef void* lv_color_t;
typedef void* lv_disp_draw_buf_t;
typedef void* lv_display_t;

// 妯℃嫙LVGL鏋氫妇
enum lv_indev_state_t {
    LV_INDEV_STATE_REL = 0,
    LV_INDEV_STATE_PR
};

// 妯℃嫙LVGL瀹氭椂鍣ㄧ粨鏋勪綋锛屽寘鍚玼ser_data鎴愬憳
struct lv_timer_t {
    void* user_data;
    void(*timer_cb)(struct lv_timer_t*);
    uint32_t period;
    uint32_t last_run;
    
    lv_timer_t() : user_data(nullptr), timer_cb(nullptr), period(0), last_run(0) {}
};

// LVGL鍑芥暟鍗犱綅绗?
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

// LVGL DRM鍑芥暟鍗犱綅绗?
inline lv_display_t* lv_linux_drm_create() {
    return nullptr; // 褰揕VGL鏈惎鐢ㄦ椂杩斿洖nullptr
}

// 鏄剧ず椹卞姩鐩稿叧鍗犱綅绗?
inline void lv_disp_draw_buf_init(lv_disp_draw_buf_t* draw_buf, void* buf1, void* buf2, uint32_t size_in_px_cnt) {}
inline void lv_disp_drv_init(lv_disp_drv_t* driver) {}
inline lv_disp_drv_t* lv_disp_drv_register(lv_disp_drv_t* driver) { return driver; }
inline void lv_disp_flush_ready(lv_disp_drv_t* disp_drv) {}

inline bool lvgl_display_init() {
    // 绾疞VGL鏄剧ず绯荤粺鍒濆鍖?
    try {
        std::cout << "Initializing pure LVGL display system..." << std::endl;
        
#ifdef ENABLE_LVGL
        // 浣跨敤LVGL鐨凩inux DRM椹卞姩
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
    // Jetson Orin NX 瑙︽懜椹卞姩鍒濆鍖栵紙鑷€傚簲锛?
    try {
        // 妫€鏌ヨЕ鎽歌澶?
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
        
        return has_touch; // 杩斿洖瀹為檯妫€娴嬬粨鏋?
    } catch (...) {
        std::cout << "Touch driver initialization exception" << std::endl;
        return false;
    }
}

// 鍓嶇缁勪欢鍗犱綅绗?- 褰揕VGL鏈惎鐢ㄦ椂
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

// 鐜版湁鍚庣缁勪欢 - 鐩存帴鍖呭惈瀹為檯瀛樺湪鐨勫ご鏂囦欢
#include "bamboo_cut/utils/logger.h"
#include "bamboo_cut/inference/bamboo_detector.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h"

// 浣跨敤鐪熷疄鐨勫懡鍚嶇┖闂?
using namespace bamboo_cut;

// 鍏ㄥ眬鍏抽棴鏍囧織
std::atomic<bool> g_shutdown_requested{false};
std::chrono::steady_clock::time_point g_shutdown_start_time;

// 闈欐€佽緭鍑洪噸瀹氬悜鏂囦欢鎻忚堪绗?
static int original_stdout = -1;
static int original_stderr = -1;
static int log_fd = -1;
static std::string log_file_path = "/var/log/bamboo-cut/camera_debug.log";

// 娓╁拰鐨勮皟璇曚俊鎭姂鍒跺嚱鏁?
void selective_debug_suppress() {
    // 鍙鐢ㄧ壒瀹氱殑鐩告満璋冭瘯锛屼繚鐣欐樉绀虹浉鍏崇殑杈撳嚭
    if (system("echo 0 > /sys/kernel/debug/tracing/events/camera/enable 2>/dev/null || true") != 0) {
        // 蹇界暐绯荤粺璋冪敤澶辫触锛岀户缁墽琛?
    }
    // 涓嶈绂佺敤DRM鐩稿叧鐨勮皟璇曚俊鎭?
    
    // 璁剧疆鐜鍙橀噺鎶戝埗Tegra鐩告満璋冭瘯
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

// 淇濇寔鍚戝悗鍏煎鐨勫嚱鏁板悕
void suppress_camera_debug() {
    selective_debug_suppress();
}

// 瀹屽叏鎶戝埗鎵€鏈夎皟璇曚俊鎭殑鍑芥暟
void suppress_all_debug_output() {
    std::cout << "Suppressing all camera and system debug output..." << std::endl;
    
    // 1. 璁剧疆鐜鍙橀噺鎶戝埗NVIDIA Tegra璋冭瘯淇℃伅
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
    
    // 棰濆鐨凾egra Camera璋冭瘯淇℃伅鎶戝埗
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
    
    // 鎶戝埗鍐呮牳鏃ュ織杈撳嚭鍒扮敤鎴风┖闂?
    setenv("KERNEL_LOG_LEVEL", "0", 1);
    setenv("DMESG_RESTRICT", "1", 1);
    
    // 寮哄埗閲嶅畾鍚戝唴鏍告秷鎭埌null
    if (system("echo 0 > /proc/sys/kernel/printk 2>/dev/null || true") != 0) {
        // 蹇界暐绯荤粺璋冪敤澶辫触锛岀户缁墽琛?
    }
    if (system("dmesg -n 0 2>/dev/null || true") != 0) {
        // 蹇界暐绯荤粺璋冪敤澶辫触锛岀户缁墽琛?
    }
    
    // 鎶戝埗systemd journal杈撳嚭鍒癱onsole
    if (system("systemctl mask systemd-journald-dev-log.socket 2>/dev/null || true") != 0) {
        // 蹇界暐绯荤粺璋冪敤澶辫触锛岀户缁墽琛?
    }
    
    // 2. 璁剧疆GStreamer闈欓粯妯″紡
    setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0", 1);
    setenv("GST_REGISTRY_UPDATE", "no", 1);
    setenv("GST_REGISTRY_FORK", "no", 1);
    
    // 3. 鍒涘缓鏃ュ織鐩綍锛堝鏋滀笉瀛樺湪锛?
    if (system("mkdir -p /var/log/bamboo-cut") != 0) {
        std::cout << "Warning: Failed to create log directory" << std::endl;
    }
    
    // 4. 鍒涘缓鏃ュ織鏂囦欢鐢ㄤ簬閲嶅畾鍚戣皟璇曚俊鎭?
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
    
    // 5. 淇濆瓨鍘熷鏂囦欢鎻忚堪绗?
    original_stdout = dup(STDOUT_FILENO);
    original_stderr = dup(STDERR_FILENO);
    
    if (original_stdout == -1 || original_stderr == -1) {
        std::cout << "Warning: Cannot backup original file descriptors" << std::endl;
        if (log_fd >= 0) close(log_fd);
        return;
    }
    
    // 6. 鍐欏叆鏃ュ織鏂囦欢澶撮儴淇℃伅
    if (log_fd >= 0) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time_t);
        timestamp.pop_back(); // 绉婚櫎鎹㈣绗?
        
        std::string log_header = "\n=== Bamboo Cut Camera Debug Log - " + timestamp + " ===\n";
        if (write(log_fd, log_header.c_str(), log_header.length()) == -1) {
            // 蹇界暐鍐欏叆澶辫触锛岀户缁墽琛?
        }
    }
    
    std::cout << "Debug output suppression configured successfully" << std::endl;
}

// 涓存椂閲嶅畾鍚戣緭鍑猴紙鍦ㄦ憚鍍忓ご鍒濆鍖栨湡闂翠娇鐢級
void redirect_output_to_log() {
    if (log_fd >= 0) {
        // 閲嶅畾鍚憇tdout鍜宻tderr鍒版棩蹇楁枃浠?
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        
        // 鍐欏叆閲嶅畾鍚戝紑濮嬫爣璁?
        std::string start_msg = "[Camera Initialization Started]\n";
        if (write(log_fd, start_msg.c_str(), start_msg.length()) == -1) {
            // 蹇界暐鍐欏叆澶辫触锛岀户缁墽琛?
        }
    }
}

// 鎭㈠鍘熷杈撳嚭
void restore_output() {
    if (original_stdout >= 0 && original_stderr >= 0) {
        // 鍐欏叆閲嶅畾鍚戠粨鏉熸爣璁?
        if (log_fd >= 0) {
            std::string end_msg = "[Camera Initialization Completed]\n\n";
            if (write(log_fd, end_msg.c_str(), end_msg.length()) == -1) {
                // 蹇界暐鍐欏叆澶辫触锛岀户缁墽琛?
            }
        }
        
        // 鎭㈠鍘熷杈撳嚭
        dup2(original_stdout, STDOUT_FILENO);
        dup2(original_stderr, STDERR_FILENO);
    }
}

// 娓呯悊閲嶅畾鍚戣祫婧?
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
        // 鍐欏叆鏃ュ織鏂囦欢缁撴潫鏍囪
        std::string final_msg = "=== Log Session Ended ===\n\n";
        if (write(log_fd, final_msg.c_str(), final_msg.length()) == -1) {
            // 蹇界暐鍐欏叆澶辫触锛岀户缁墽琛?
        }
        close(log_fd);
        log_fd = -1;
    }
}

// 淇″彿澶勭悊
void signal_handler(int sig) {
    std::cout << "\n鏀跺埌淇″彿 " << sig << "锛屽紑濮嬩紭闆呭叧闂?.." << std::endl;
    g_shutdown_requested = true;
    g_shutdown_start_time = std::chrono::steady_clock::now();
    
    // 娓呯悊杈撳嚭閲嶅畾鍚戣祫婧?
    cleanup_output_redirection();
}

/**
 * 绾跨▼瀹夊叏鐨勬暟鎹ˉ鎺ュ櫒
 * 鍦ㄦ帹鐞嗙嚎绋嬪拰UI绾跨▼闂翠紶閫掓暟鎹?
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
    // 瑙嗛鏁版嵁鏇存柊 (浠庢帹鐞嗙嚎绋嬭皟鐢?
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
    
    // 妫€娴嬫暟鎹洿鏂?(浠庢帹鐞嗙嚎绋嬭皟鐢?
    void updateDetection(const core::DetectionResult& result) {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        latest_detection_.cutting_points = result.cutting_points;
        latest_detection_.bboxes = result.bboxes;
        latest_detection_.confidences = result.confidences;
        latest_detection_.processing_time_ms = 50.0f; // 绠€鍖栧疄鐜?
        latest_detection_.has_detection = result.valid && !result.bboxes.empty();
        new_detection_available_ = true;
    }
    
    // 绯荤粺缁熻鏇存柊
    void updateStats(const SystemStats& stats) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        latest_stats_ = stats;
    }
    
    // 鑾峰彇鏈€鏂版暟鎹?(浠嶶I绾跨▼璋冪敤)
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
 * 鎺ㄧ悊宸ヤ綔绾跨▼
 * 澶嶇敤鐜版湁鐨刢pp_backend缁勪欢
 */
class InferenceWorkerThread {
private:
    IntegratedDataBridge* data_bridge_;
    std::unique_ptr<std::thread> worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // 浣跨敤鐪熷疄鐨勫悗绔粍浠?
    std::unique_ptr<inference::BambooDetector> detector_;
    std::unique_ptr<deepstream::DeepStreamManager> deepstream_manager_;
    bool use_mock_data_ = false;
    void* lvgl_interface_ptr_ = nullptr;  // 鐢ㄤ簬瀛樺偍LVGL鐣岄潰鎸囬拡
    
    // Wayland閰嶇疆 - 鏇夸唬DRM Overlay
    bool wayland_available_ = false;
    
    // 鎬ц兘缁熻
    int processed_frames_ = 0;
    std::chrono::steady_clock::time_point last_stats_time_;
    float current_fps_ = 0.0f;

public:
    InferenceWorkerThread(IntegratedDataBridge* bridge)
        : data_bridge_(bridge), last_stats_time_(std::chrono::steady_clock::now()) {}
    
    // 璁剧疆LVGL鐣岄潰鎸囬拡
    void setLVGLInterface(void* lvgl_interface) {
        lvgl_interface_ptr_ = lvgl_interface;
    }
    
    // 妫€鏌ayland鐜
    bool checkWaylandEnvironment() {
        // 强制走 DRM/GBM 路径，不再依赖合成器
        wayland_available_ = false;
        return false;
    }
    
    ~InferenceWorkerThread() {
        stop();
    }
    
    bool initialize() {
        std::cout << "[推理系统] 初始化 DRM/GBM 流水线..." << std::endl;

        deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
        deepstream::DeepStreamConfig config;
        return deepstream_manager_->initialize(config);

#ifdef ENABLE_WAYLAND
        // 鑾峰彇LVGL鐨刉ayland瀵硅薄
        if (!lvgl_interface_ptr_) {
            std::cerr << "❌ [推理系统] LVGL接口未设置" << std::endl;
            return false;
        }
        
        auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLWaylandInterface*>(lvgl_interface_ptr_);
        
        // 馃敡 鍏抽敭锛氱瓑寰匧VGL鐨刉ayland瀵硅薄瀹屽叏鍒濆鍖?
        int retry_count = 0;
        const int MAX_RETRIES = 20;
        
        void* parent_display = nullptr;
        void* parent_compositor = nullptr;
        void* parent_subcompositor = nullptr;
        void* parent_surface = nullptr;
        
        while (retry_count < MAX_RETRIES) {
            parent_display = lvgl_if->getWaylandDisplay();
            parent_compositor = lvgl_if->getWaylandCompositor();
            parent_subcompositor = lvgl_if->getWaylandSubcompositor();
            parent_surface = lvgl_if->getWaylandSurface();
            
            if (parent_display && parent_compositor && parent_subcompositor && parent_surface) {
                std::cout << "鉁?宸茶幏鍙朙VGL Wayland鐖剁獥鍙ｅ璞★紙閲嶈瘯" << retry_count << "娆★級" << std::endl;
                break;
            }
            
            std::cout << "鈴?绛夊緟LVGL Wayland瀵硅薄鍒濆鍖?..锛堢" << (retry_count + 1) << "娆″皾璇曪級" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            retry_count++;
        }
        
        if (!parent_display || !parent_compositor || !parent_subcompositor || !parent_surface) {
            std::cerr << "鉂?鏃犳硶鑾峰彇LVGL Wayland瀵硅薄锛堝凡閲嶈瘯" << MAX_RETRIES << "娆★級" << std::endl;
            std::cerr << "馃攧 DeepStream灏嗕娇鐢ˋppSink杞欢鍚堟垚妯″紡" << std::endl;
            
            // 闄嶇骇鍒癆ppSink妯″紡
            deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
            
            // 馃敡 淇锛氬垱寤洪粯璁ら厤缃?
            deepstream::DeepStreamConfig config;
            config.sink_mode = deepstream::VideoSinkMode::APPSINK;
            config.camera_width = 1280;
            config.camera_height = 720;
            config.camera_fps = 30;
            config.camera_id = 0;
            
            return deepstream_manager_->initialize(config);
        }
        
        std::cout << "✅ 已获取LVGL Wayland父窗口对象" << std::endl;
        
        // 馃敡 鍏抽敭淇锛氳幏鍙?camera_panel 鐨勫疄闄呭潗鏍囷紙鍦‵lex甯冨眬瀹屾垚鍚庯級
                // ?? camera_panel ???Flex ??????
        int camera_x = 0, camera_y = 60, camera_width = 960, camera_height = 640;
        bool got_cam_panel = lvgl_if->getCameraPanelCoords(camera_x, camera_y, camera_width, camera_height);
        std::cout << "[DeepStream] camera panel coords "
                  << (got_cam_panel ? "" : "(fallback) ")
                  << "(" << camera_x << ", " << camera_y << ") "
                  << camera_width << "x" << camera_height << std::endl;

        // 内边距，防止视频顶边贴合
        const int padding = 10;
        if (camera_width > padding * 2 && camera_height > padding * 2) {
            camera_x += padding;
            camera_y += padding;
            camera_width -= padding * 2;
            camera_height -= padding * 2;
        }

        // 对齐到偶数并向下32对齐，兼容 NVMM/硬件拷贝
        auto align32 = [](int v) {
            v = (v / 2) * 2;
            return (v / 32) * 32;
        };
        int aligned_w = align32(camera_width);
        int aligned_h = align32(camera_height);
        if (aligned_w >= 64) camera_width = aligned_w;
        if (aligned_h >= 64) camera_height = aligned_h;

        std::cout << "[DeepStream] camera region with padding: ("
                  << camera_x << ", " << camera_y << ") "
                  << camera_width << "x" << camera_height << std::endl;

deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
        
        // 閰嶇疆Subsurface锛堜娇鐢ㄥ疄闄呭潗鏍囷級
        deepstream::SubsurfaceConfig subsurface_config;
        subsurface_config.offset_x = camera_x;
        subsurface_config.offset_y = camera_y;
        subsurface_config.width = camera_width;
        subsurface_config.height = camera_height;
        subsurface_config.use_sync_mode = false;  // 寮傛妯″紡锛岃棰戠嫭绔嬪埛鏂?
        
        // 馃敡 鍏抽敭锛氫娇鐢⊿ubsurface妯″紡鍒濆鍖?
        if (!deepstream_manager_->initializeWithSubsurface(
                parent_display,
                parent_compositor,
                parent_subcompositor,
                parent_surface,
                subsurface_config)) {
            std::cerr << "❌ DeepStream Subsurface初始化失败" << std::endl;
            return false;
        }
        
        std::cout << "✅ [推理系统] Wayland Subsurface架构初始化完成" << std::endl;
        std::cout << "馃摵 瑙嗛灏嗙敱 Wayland 鍚堟垚鍣ㄨ嚜鍔ㄥ悎鎴愬埌 LVGL 绐楀彛" << std::endl;
        
        return true;
#endif
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
        std::cout << "Inference worker thread started (寤惰繜鍚姩妯″紡)" << std::endl;
        
        // 寤惰繜鍚姩DeepStream锛岀‘淇滾VGL瀹屽叏鍒濆鍖?
        if (!use_mock_data_ && deepstream_manager_) {
            std::cout << "宸ヤ綔绾跨▼涓欢杩熷惎鍔―eepStream..." << std::endl;
            if (!startDeepStreamManagerDelayed()) {
                std::cout << "DeepStream寤惰繜鍚姩澶辫触锛屽垏鎹㈠埌妯℃嫙妯″紡" << std::endl;
                use_mock_data_ = true;
            }
        }
        
        auto last_frame_time = std::chrono::steady_clock::now();
        const auto target_interval = std::chrono::milliseconds(33); // 30fps
        
        while (!should_stop_ && !g_shutdown_requested) {
            auto current_time = std::chrono::steady_clock::now();
            
            // 澶勭悊涓€甯?
            processFrame();
            
            // 鏇存柊鎬ц兘缁熻
            updatePerformanceStats();
            
            // 甯х巼鎺у埗
            auto processing_time = std::chrono::steady_clock::now() - current_time;
            auto sleep_time = target_interval - processing_time;
            
            if (sleep_time > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
        
        std::cout << "Inference worker thread exited" << std::endl;
    }
    
    void processFrame() {
        // DeepStream 绠＄悊鍣ㄥ鐞嗗疄闄呯殑瑙嗛鏄剧ず鍜?AI 鎺ㄧ悊
        // integrated_main 鍙鐞嗘ā鎷熸暟鎹敤浜庢祴璇?
        if (use_mock_data_) {
            cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
            cv::putText(frame, "DEEPSTREAM MODE - Frame " + std::to_string(processed_frames_),
                       cv::Point(50, 360), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Current Mode: " + std::to_string(static_cast<int>(deepstream_manager_ ?
                       static_cast<int>(deepstream_manager_->getCurrentMode()) : 0)),
                       cv::Point(50, 420), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2);
            
            // 鏇存柊妯℃嫙瑙嗛鍒版暟鎹ˉ鎺?
            data_bridge_->updateVideo(frame);
            
            processed_frames_++;
        }
    }
    
    void updatePerformanceStats() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time_);
        
        if (elapsed.count() >= 1) {
            current_fps_ = static_cast<float>(processed_frames_) / elapsed.count();
            
            // 鏇存柊绯荤粺缁熻
            IntegratedDataBridge::SystemStats stats;
            stats.camera_fps = current_fps_;
            stats.inference_fps = current_fps_;
            stats.cpu_usage = getCpuUsage();
            stats.memory_usage_mb = getMemoryUsage();
            stats.total_detections += processed_frames_;
            stats.plc_connected = false; // 绠€鍖栧疄鐜帮紝鍚庣画鍙坊鍔燤odbus鏀寔
            
            data_bridge_->updateStats(stats);
            
            processed_frames_ = 0;
            last_stats_time_ = now;
        }
    }
    
    // === 鍒濆鍖栨柟娉?(绂佺敤TensorRT锛岄伩鍏嶄笌DeepStream nvinfer鍐茬獊) ===
    bool initializeDetector() {
        std::cout << "馃敡 [BambooDetector] 绂佺敤TensorRT鍒濆鍖栵紝閬垮厤涓嶥eepStream nvinfer鍐茬獊" << std::endl;
        
        inference::DetectorConfig config;
        config.model_path = "/opt/bamboo-cut/models/bamboo_detection.onnx";
        config.confidence_threshold = 0.85f;
        config.nms_threshold = 0.45f;
        config.input_size = cv::Size(640, 640);
        config.use_gpu = true;
        config.use_tensorrt = false;  // 馃敡 鍏抽敭淇锛氱鐢═ensorRT锛岃DeepStream nvinfer鐙崰
        
        detector_ = std::make_unique<inference::BambooDetector>(config);
        bool result = detector_->initialize();
        
        if (result) {
            std::cout << "✅ [BambooDetector] 检测器初始化成功（OpenCV DNN模式，避免TensorRT冲突）" << std::endl;
        } else {
            std::cout << "⚠️ [BambooDetector] 检测器初始化失败" << std::endl;
        }
        
        return result;
    }
    
    // === DeepStream 绠＄悊鍣ㄥ垵濮嬪寲鏂规硶 ===
    bool initializeDeepStreamManager() {
        std::cout << "馃幀 [DeepStream] 鍒濆鍖?DeepStream 绠＄悊鍣?.." << std::endl;
        
        try {
            // 鍒涘缓 DeepStream 绠＄悊鍣ㄥ疄渚嬶紙濡傛灉鏈塋VGL鐣岄潰鎸囬拡鍒欎紶鍏ワ級
            if (lvgl_interface_ptr_) {
                deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>(lvgl_interface_ptr_);
                std::cout << "馃敆 [DeepStream] 绠＄悊鍣ㄥ凡杩炴帴LVGL鐣岄潰" << std::endl;
            } else {
                deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
                std::cout << "⚠️  [DeepStream] 管理器创建（无LVGL界面连接）" << std::endl;
            }
            
            // 閰嶇疆 DeepStream 鍙傛暟
            deepstream::DeepStreamConfig config;
            config.screen_width = 1280;
            config.screen_height = 800;
            config.header_height = 80;
            config.footer_height = 80;
            config.video_width_ratio = 0.75f;
            config.video_height_ratio = 1.0f;
            config.camera_id = 0;
            config.camera_id_2 = 1;
            config.dual_mode = deepstream::DualCameraMode::SINGLE_CAMERA;
            
            // 馃敡 淇锛氫娇鐢ㄦ憚鍍忓ご鏀寔鐨勫垎杈ㄧ巼
            config.camera_width = 1280;   // 浣跨敤鎽勫儚澶村師鐢熸敮鎸佺殑1280x720
            config.camera_height = 720;   // 60fps楂樻€ц兘妯″紡
            config.camera_fps = 60;       // 浣跨敤60fps鑾峰緱鏈€浣虫€ц兘
            config.test_pattern = 0;      // 浣跨敤smpte鏍囧噯褰╂潯鍥炬
            
            std::cout << "馃帴 [鎽勫儚澶碷 浣跨敤鏀寔鐨勫垎杈ㄧ巼: " << config.camera_width << "x" << config.camera_height << "@" << config.camera_fps << "fps" << std::endl;
            
            // 妫€鏌ayland鐜骞堕厤缃畐aylandsink
            if (checkWaylandEnvironment()) {
                std::cout << "馃幆 [DeepStream] 妫€娴嬪埌Wayland鐜锛岄厤缃畐aylandsink娓叉煋..." << std::endl;
                std::cout << "✅[DeepStream] Wayland环境已设置" << std::endl;
            } else {
                std::cout << "馃摫 [DeepStream] 鏃燱ayland鐜锛屽皢浣跨敤AppSink杞欢鍚堟垚" << std::endl;
            }
            
            // 鍒濆鍖?DeepStream 绠＄悊鍣?(浣嗘殏涓嶅惎鍔?
            if (!deepstream_manager_->initialize(config)) {
                std::cout << "鉂?[DeepStream] 绠＄悊鍣ㄥ垵濮嬪寲澶辫触" << std::endl;
                return false;
            }
            
            std::cout << "鉁?[DeepStream] 绠＄悊鍣ㄥ垵濮嬪寲瀹屾垚 (寤惰繜鍚姩妯″紡)" << std::endl;
            
            // 鏄剧ず褰撳墠sink妯″紡
            auto current_mode = deepstream_manager_->getCurrentSinkMode();
            const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
            std::cout << "馃摵 [DeepStream] 褰撳墠sink妯″紡: " << mode_names[static_cast<int>(current_mode)];
            
            if (wayland_available_ && current_mode == deepstream::VideoSinkMode::WAYLANDSINK) {
                std::cout << " (Wayland纭欢娓叉煋)" << std::endl;
            } else {
                std::cout << " (杞欢鍚堟垚妯″紡)" << std::endl;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "鉂?[DeepStream] 绠＄悊鍣ㄥ垵濮嬪寲寮傚父: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool startDeepStreamManagerDelayed() {
        if (!deepstream_manager_) {
            std::cout << "错误：DeepStream管理器尚未初始化" << std::endl;
            return false;
        }
        // DRM/GBM 路径：直接启动 DeepStream 管理器
        std::cout << "启动DeepStream管理器（DRM/GBM）..." << std::endl;
        if (!deepstream_manager_->start()) {
            std::cout << "❌ DeepStream 管理器启动失败" << std::endl;
            return false;
        }
        // 启动canvas更新线程 (如果支持LVGL界面集成)
        if (deepstream_manager_) {
            std::cout << "启动Canvas更新线程...（DRM/GBM）" << std::endl;
            deepstream_manager_->startCanvasUpdateThread();
        }
        std::cout << "✅ DeepStream 管理器延迟启动成功" << std::endl;
        return true;
    }
    
    float getCpuUsage() const { return 45.0f; } // 绠€鍖栧疄鐜?
    float getMemoryUsage() const { return 1024.0f; } // 绠€鍖栧疄鐜?
};

/**
 * LVGL UI绠＄悊鍣?
 * 浣跨敤浼樺寲鐨凩VGL鐣岄潰瀹炵幇
 */
class LVGLUIManager {
private:
    IntegratedDataBridge* data_bridge_;
    
    // 浣跨敤Wayland浼樺寲鐨凩VGL鐣岄潰瀹炵幇
    std::unique_ptr<bamboo_cut::ui::LVGLWaylandInterface> lvgl_wayland_interface_;
    
    // 鍏煎鎬ф柟娉曟槧灏?
    bool initialized_ = false;

public:
    LVGLUIManager(IntegratedDataBridge* bridge)
        : data_bridge_(bridge) {}
    
    ~LVGLUIManager() {
        cleanup();
    }

    // 鍏煎鎬ф柟娉曪細鍒涘缓涓荤晫闈?
    bool create_main_screen() {
        return initialize();
    }

    // 鍏煎鎬ф柟娉曪細鏇存柊绯荤粺鐘舵€?
    void update_system_status(const char* status, lv_color_t color) {
        std::cout << "System Status Updated: " << status << std::endl;
    }

    // 鍏煎鎬ф柟娉曪細鏇存柊妫€娴嬫暟閲?
    void update_detection_count(int count) {
        std::cout << "Detection Count Updated: " << count << std::endl;
    }

    // 鍏煎鎬ф柟娉曪細鏇存柊FPS
    void update_fps(float fps) {
        std::cout << "FPS Updated: " << fps << std::endl;
    }
    
    // 鑾峰彇LVGL Wayland鐣岄潰鎸囬拡锛堢敤浜庝紶閫掔粰DeepStreamManager锛?
    void* getLVGLInterface() {
        #ifdef ENABLE_LVGL
        return lvgl_wayland_interface_.get();
        #else
        return nullptr;
        #endif
    }
    
    bool initialize() {
        std::cout << "Initializing LVGL UI system with optimized interface..." << std::endl;
        
        #ifdef ENABLE_LVGL
        try {
            // DRM/GBM 路径，不依赖 Wayland
            std::cout << "[Display] Using DRM/GBM LVGL interface (Wayland disabled)" << std::endl;
            lvgl_wayland_interface_ = std::make_unique<bamboo_cut::ui::LVGLWaylandInterface>();
            
            // 配置 LVGL（DRM/GBM 参数沿用屏幕尺寸）
            bamboo_cut::ui::LVGLWaylandConfig config;
            config.screen_width = 1280;
            config.screen_height = 800;
            config.refresh_rate = 60;
            config.enable_touch = true;
            config.touch_device = "/dev/input/event0";
            config.wayland_display = "wayland-0";
            
            std::cout << "初始化 LVGL DRM/GBM 界面..." << std::endl;
            if (!lvgl_wayland_interface_->initialize(config)) {
                std::cout << "LVGL DRM/GBM interface initialization failed" << std::endl;
                return false;
            }
            
            std::cout << "LVGL DRM/GBM 界面初始化完成，启动界面线程..." << std::endl;
            // 鍚姩鐣岄潰绾跨▼
            if (!lvgl_wayland_interface_->start()) {
                std::cout << "LVGL Wayland interface start failed" << std::endl;
                return false;
            }
            
            // 缁欑晫闈㈢嚎绋嬩竴浜涙椂闂存潵绋冲畾鍚姩
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << "LVGL Wayland鐣岄潰绾跨▼鍚姩瀹屾垚" << std::endl;
            
            std::cout << "Wayland浼樺寲鐨凩VGL鐣岄潰鍒涘缓鎴愬姛" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "LVGL Wayland interface creation exception: " << e.what() << std::endl;
            return false;
        }
        #else
        std::cout << "LVGL not enabled, using placeholder UI" << std::endl;
        
        // 鍗犱綅绗﹀疄鐜帮細妯℃嫙鍒濆鍖栨垚鍔?
        std::cout << "Simulated LVGL UI initialization (LVGL disabled)" << std::endl;
        #endif
        
        initialized_ = true;
        std::cout << "LVGL UI system initialization complete" << std::endl;
        return true;
    }
    
    void runMainLoop() {
        if (!initialized_) return;
        
        std::cout << "LVGL main loop started with optimized interface" << std::endl;
        
        #ifdef ENABLE_LVGL
        if (lvgl_wayland_interface_ && lvgl_wayland_interface_->isRunning()) {
            std::cout << "Using Wayland浼樺寲鐨凩VGL interface main loop" << std::endl;
            // LVGL Wayland鐣岄潰宸茬粡鍦ㄨ嚜宸辩殑绾跨▼涓繍琛岋紝杩欓噷鍙渶瑕佺瓑寰?
            while (!g_shutdown_requested && lvgl_wayland_interface_->isRunning()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        #else
        // 鍗犱綅绗︿富寰幆
        while (!g_shutdown_requested) {
            // 妯℃嫙鐣岄潰鏇存柊
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 60fps
        }
        #endif
        
        std::cout << "LVGL main loop exited" << std::endl;
    }

private:
    void cleanup() {
        #ifdef ENABLE_LVGL
        if (lvgl_wayland_interface_) {
            lvgl_wayland_interface_->stop();
            lvgl_wayland_interface_.reset();
        }
        #endif
        
        initialized_ = false;
    }

private:
    // 妫€鏌ayland鍚堟垚鍣ㄧ姸鎬?
    bool checkWaylandCompositor() {
        // 鍚敤 Wayland 璋冭瘯
        setenv("WAYLAND_DEBUG", "1", 1);

        // 纭繚姝ｇ‘鐨?runtime 鐩綍锛堜紭鍏堜娇鐢ㄧ幇鏈?XDG_RUNTIME_DIR锛?
        const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
        if (!runtime_dir || access(runtime_dir, W_OK) != 0) {
            // Jetson + nvweston 鍦烘櫙锛氫娇鐢?/run/nvidia-wayland
            std::cout << "鈿狅笍 XDG_RUNTIME_DIR 涓嶅彲鍐欐垨鏈厤缃紝璁剧疆涓?/run/nvidia-wayland" << std::endl;
            setenv("XDG_RUNTIME_DIR", "/run/nvidia-wayland", 1);
            runtime_dir = getenv("XDG_RUNTIME_DIR");
        }
        // 妫€鏌?WAYLAND_DISPLAY 鐜鍙橀噺
        const char* wayland_display = getenv("WAYLAND_DISPLAY");
        if (!wayland_display) {
            wayland_display = "wayland-0";
            setenv("WAYLAND_DISPLAY", wayland_display, 1);
        }
        
        // 鏋勫缓 socket 璺緞
        std::string socket_path = std::string(runtime_dir) + "/" + wayland_display;
        
        // 馃敡 淇锛氫娇鐢╝ccess()妫€鏌ocket鏂囦欢锛堟纭殑鏂规硶锛?
        if (access(socket_path.c_str(), F_OK) != 0) {
            std::cout << "Wayland socket涓嶅瓨鍦? " << socket_path << std::endl;
            std::cout << "閿欒浠ｇ爜: " << strerror(errno) << std::endl;
            return false;
        }
        
        std::cout << "鉁?Wayland鍚堟垚鍣ㄦ娴嬫垚鍔? " << wayland_display << std::endl;
        std::cout << "   Socket璺緞: " << socket_path << std::endl;
        return true;
    }
};

/**
 * 涓€浣撳寲涓荤▼搴忕被
 */
class IntegratedBambooSystem {
private:
    IntegratedDataBridge data_bridge_;
    std::unique_ptr<InferenceWorkerThread> inference_worker_;
    std::unique_ptr<LVGLUIManager> ui_manager_;
    
public:
    bool initialize() {
        std::cout << "=================================" << std::endl;
        std::cout << "Bamboo Recognition System" << std::endl;
        std::cout << "DRM/GBM 模式" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // 璁剧疆淇″彿澶勭悊
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // === Step1: DRM/GBM 路径（禁用 Wayland） ===
        std::cout << "\n[Display] Step1: DRM/GBM mode (Wayland disabled)" << std::endl;
        std::cout << "[Display] Wayland skipped; DRM/GBM active" << std::endl;
        
        // === Step2: 初始化 LVGL DRM/GBM 界面 ===
        std::cout << "\n[LVGL] Step2: 初始化 DRM/GBM LVGL 界面..." << std::endl;
        ui_manager_ = std::make_unique<LVGLUIManager>(&data_bridge_);
        if (!ui_manager_->initialize()) {
            std::cout << "❌[LVGL] LVGL DRM/GBM 界面初始化失败" << std::endl;
            return false;
        }
        std::cout << "✅[LVGL] LVGL DRM/GBM 界面初始化成功" << std::endl;
        
        // === Step3: 初始化 DeepStream DRM 流水线 ===
        std::cout << "\n[DeepStream] Step3: 初始化 DeepStream DRM/GBM 模式..." << std::endl;
        inference_worker_ = std::make_unique<InferenceWorkerThread>(&data_bridge_);
        
        // 浼犻€扡VGL Wayland鐣岄潰鎸囬拡缁欐帹鐞嗗伐浣滅嚎绋?
        #ifdef ENABLE_LVGL
        if (ui_manager_ && ui_manager_->getLVGLInterface()) {
            inference_worker_->setLVGLInterface(ui_manager_->getLVGLInterface());
            std::cout << "馃敆 [闆嗘垚] LVGL Wayland鐣岄潰鎸囬拡宸蹭紶閫掔粰鎺ㄧ悊宸ヤ綔绾跨▼" << std::endl;
        }
        #endif
        
        // 妫€鏌ayland鐜閰嶇疆
        if (inference_worker_->checkWaylandEnvironment()) {
            std::cout << "馃幆 [闆嗘垚] DeepStream灏嗕娇鐢╳aylandsink纭欢娓叉煋" << std::endl;
        } else {
            std::cout << "馃摫 [闆嗘垚] DeepStream闄嶇骇鍒癆ppSink杞欢鍚堟垚妯″紡" << std::endl;
        }
        
        if (!inference_worker_->initialize()) {
            std::cout << "鉂?[鎺ㄧ悊绯荤粺] Inference system initialization failed" << std::endl;
            return false;
        }
        
        std::cout << "\n=================================" << std::endl;
        std::cout << "✅Wayland架构系统初始化完成" << std::endl;
        std::cout << "=================================" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "Starting Wayland integrated system..." << std::endl;
        
        // 馃敡 鍏抽敭淇锛氫紭鍖朩ayland瀹㈡埛绔惎鍔ㄩ『搴忥紝閬垮厤xdg_positioner鍐茬獊
        std::cout << "馃敡 绛夊緟LVGL Wayland鐣岄潰瀹屽叏鍚姩鍜岃繛鎺ョǔ瀹?.." << std::endl;
        
        #ifdef ENABLE_LVGL
        if (ui_manager_ && ui_manager_->getLVGLInterface()) {
            auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLWaylandInterface*>(ui_manager_->getLVGLInterface());
            int wait_count = 0;
            const int MAX_WAIT_SECONDS = 15;
            
            while (!lvgl_if->isFullyInitialized() && wait_count < MAX_WAIT_SECONDS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                wait_count++;
                std::cout << "鈴?绛夊緟LVGL Wayland瀹屽叏鍒濆鍖?.. (" << (wait_count * 0.5) << "绉?" << std::endl;
            }
            
            if (lvgl_if->isFullyInitialized()) {
                std::cout << "鉁?LVGL Wayland宸插畬鍏ㄥ垵濮嬪寲" << std::endl;
            } else {
                std::cout << "⚠️ 警告：LVGL初始化超时，但继续启动系统" << std::endl;
            }
        } else
        #endif
        {
            std::cout << "馃摑 LVGL涓嶅彲鐢紝浣跨敤鍥哄畾寤惰繜" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
        
        // 馃幆 鍏抽敭锛氶澶栫瓑寰呯‘淇漌ayland杩炴帴瀹屽叏绋冲畾
        std::cout << "馃攧 棰濆绛夊緟Wayland display杩炴帴绋冲畾..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        // 鍚姩鎺ㄧ悊宸ヤ綔绾跨▼锛孏Streamer灏嗕娇鐢ㄧ嫭绔嬬殑waylandsink杩炴帴
        std::cout << "鐜板湪鍚姩鎺ㄧ悊绾跨▼锛堢嫭绔媤aylandsink杩炴帴妯″紡锛?.." << std::endl;
        if (!inference_worker_->start()) {
            std::cout << "Inference thread startup failed" << std::endl;
            return;
        }
        
        std::cout << "鎺ㄧ悊绾跨▼宸插惎鍔紝Wayland绯荤粺瀹屽叏灏辩华" << std::endl;
        std::cout << "Press Ctrl+C to exit system" << std::endl;
        
        // 涓荤嚎绋嬭繍琛孶I (闃诲)
        ui_manager_->runMainLoop();
        
        std::cout << "Starting system shutdown..." << std::endl;
        shutdown();
    }
    
    void shutdown() {
        std::cout << "Stopping inference thread..." << std::endl;
        if (inference_worker_) {
            inference_worker_->stop();
        }
        
        // 娓呯悊杈撳嚭閲嶅畾鍚戣祫婧?
        cleanup_output_redirection();
        
        std::cout << "System shutdown complete" << std::endl;
    }

private:
    // Wayland compositor check is disabled; always use DRM/GBM path
    bool checkWaylandCompositor() { return false; }
};

/**
 * 涓诲嚱鏁板叆鍙?
 */
int main(int argc, char* argv[]) {
    try {
        // 妫€鏌ユ槸鍚︿负娴嬭瘯妯″紡鎴栬皟璇曟ā寮?
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
        
        // 鍦ㄩ潪璇︾粏妯″紡涓嬩娇鐢ㄦ俯鍜岀殑璋冭瘯鎶戝埗
        if (!verbose_mode && !test_mode) {
            selective_debug_suppress();
        } else {
            std::cout << "Bamboo Recognition System starting in verbose mode..." << std::endl;
        }
        
        IntegratedBambooSystem system;
        
        if (!system.initialize()) {
            // 涓存椂鎭㈠stdout鏄剧ず閿欒淇℃伅
            if (!verbose_mode) {
                if (freopen("/dev/tty", "w", stdout) == nullptr) {
                    // 蹇界暐鎭㈠澶辫触锛岀户缁墽琛?
                }
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
        // 涓存椂鎭㈠stdout鏄剧ず寮傚父淇℃伅
        if (freopen("/dev/tty", "w", stdout) == nullptr) {
            // 蹇界暐鎭㈠澶辫触锛岀户缁墽琛?
        }
        std::cout << "System exception: " << e.what() << std::endl;
        return -1;
    }
}
