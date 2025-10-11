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
#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_interface.h"

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
    if (system("echo 0 > /sys/kernel/debug/tracing/events/camera/enable 2>/dev/null || true") != 0) {
        // 忽略系统调用失败，继续执行
    }
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
    if (system("echo 0 > /proc/sys/kernel/printk 2>/dev/null || true") != 0) {
        // 忽略系统调用失败，继续执行
    }
    if (system("dmesg -n 0 2>/dev/null || true") != 0) {
        // 忽略系统调用失败，继续执行
    }
    
    // 抑制systemd journal输出到console
    if (system("systemctl mask systemd-journald-dev-log.socket 2>/dev/null || true") != 0) {
        // 忽略系统调用失败，继续执行
    }
    
    // 2. 设置GStreamer静默模式
    setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0", 1);
    setenv("GST_REGISTRY_UPDATE", "no", 1);
    setenv("GST_REGISTRY_FORK", "no", 1);
    
    // 3. 创建日志目录（如果不存在）
    if (system("mkdir -p /var/log/bamboo-cut") != 0) {
        std::cout << "Warning: Failed to create log directory" << std::endl;
    }
    
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
        if (write(log_fd, log_header.c_str(), log_header.length()) == -1) {
            // 忽略写入失败，继续执行
        }
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
        if (write(log_fd, start_msg.c_str(), start_msg.length()) == -1) {
            // 忽略写入失败，继续执行
        }
    }
}

// 恢复原始输出
void restore_output() {
    if (original_stdout >= 0 && original_stderr >= 0) {
        // 写入重定向结束标记
        if (log_fd >= 0) {
            std::string end_msg = "[Camera Initialization Completed]\n\n";
            if (write(log_fd, end_msg.c_str(), end_msg.length()) == -1) {
                // 忽略写入失败，继续执行
            }
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
        if (write(log_fd, final_msg.c_str(), final_msg.length()) == -1) {
            // 忽略写入失败，继续执行
        }
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
    std::unique_ptr<deepstream::DeepStreamManager> deepstream_manager_;
    bool use_mock_data_ = false;
    void* lvgl_interface_ptr_ = nullptr;  // 用于存储LVGL界面指针
    
    // 性能统计
    int processed_frames_ = 0;
    std::chrono::steady_clock::time_point last_stats_time_;
    float current_fps_ = 0.0f;

public:
    InferenceWorkerThread(IntegratedDataBridge* bridge)
        : data_bridge_(bridge), last_stats_time_(std::chrono::steady_clock::now()) {}
    
    // 设置LVGL界面指针
    void setLVGLInterface(void* lvgl_interface) {
        lvgl_interface_ptr_ = lvgl_interface;
    }
    
    ~InferenceWorkerThread() {
        stop();
    }
    
    bool initialize() {
        std::cout << "Initializing inference system..." << std::endl;
        
        // 初始化检测器 (使用真实的BambooDetector)
        if (!initializeDetector()) {
            std::cout << "Detector initialization failed, using simulation mode" << std::endl;
            use_mock_data_ = true;
        }
        
        // 初始化 DeepStream 管理器
        if (!initializeDeepStreamManager()) {
            std::cout << "DeepStream manager initialization failed, using simulation mode" << std::endl;
            use_mock_data_ = true;
        }
        
        std::cout << "Inference system initialization complete (simulation mode: " << (use_mock_data_ ? "yes" : "no") << ")" << std::endl;
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
        std::cout << "Inference worker thread started (延迟启动模式)" << std::endl;
        
        // 延迟启动DeepStream，确保LVGL完全初始化
        if (!use_mock_data_ && deepstream_manager_) {
            std::cout << "工作线程中延迟启动DeepStream..." << std::endl;
            if (!startDeepStreamManagerDelayed()) {
                std::cout << "DeepStream延迟启动失败，切换到模拟模式" << std::endl;
                use_mock_data_ = true;
            }
        }
        
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
        // DeepStream 管理器处理实际的视频显示和 AI 推理
        // integrated_main 只处理模拟数据用于测试
        if (use_mock_data_) {
            cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
            cv::putText(frame, "DEEPSTREAM MODE - Frame " + std::to_string(processed_frames_),
                       cv::Point(50, 360), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Current Mode: " + std::to_string(static_cast<int>(deepstream_manager_ ?
                       static_cast<int>(deepstream_manager_->getCurrentMode()) : 0)),
                       cv::Point(50, 420), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2);
            
            // 更新模拟视频到数据桥接
            data_bridge_->updateVideo(frame);
            
            processed_frames_++;
        }
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
    
    // === DeepStream 管理器初始化方法 ===
    bool initializeDeepStreamManager() {
        std::cout << "初始化 DeepStream 管理器..." << std::endl;
        
        try {
            // 创建 DeepStream 管理器实例（如果有LVGL界面指针则传入）
            if (lvgl_interface_ptr_) {
                deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>(lvgl_interface_ptr_);
                std::cout << "DeepStream管理器已连接LVGL界面" << std::endl;
            } else {
                deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
                std::cout << "DeepStream管理器创建（无LVGL界面连接）" << std::endl;
            }
            
            // 配置 DeepStream 参数
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
            config.camera_width = 960;   // 匹配Canvas尺寸
            config.camera_height = 640;  // 匹配Canvas尺寸
            config.camera_fps = 30;      // 确保30fps提高稳定性
            config.test_pattern = 0;     // 使用smpte标准彩条图案
            
            // 初始化 DeepStream 管理器 (但暂不启动)
            if (!deepstream_manager_->initialize(config)) {
                std::cout << "DeepStream 管理器初始化失败" << std::endl;
                return false;
            }
            
            // 设置视频输出模式为nvdrmvideosink (叠加平面模式)
            std::cout << "配置nvdrmvideosink叠加平面模式..." << std::endl;
            
            // 设置为kmssink KMS多层渲染模式（默认模式）
            if (!deepstream_manager_->switchSinkMode(deepstream::VideoSinkMode::KMSSINK)) {
                std::cout << "警告：kmssink模式设置失败，尝试回退到appsink模式" << std::endl;
                if (!deepstream_manager_->switchSinkMode(deepstream::VideoSinkMode::APPSINK)) {
                    std::cout << "错误：所有视频输出模式设置都失败" << std::endl;
                    return false;
                }
            }
            
            std::cout << "DeepStream 管理器初始化完成 (延迟启动模式)" << std::endl;
            
            // 显示当前sink模式
            auto current_mode = deepstream_manager_->getCurrentSinkMode();
            const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
            std::cout << "当前sink模式: " << mode_names[static_cast<int>(current_mode)] << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "DeepStream 管理器初始化异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    // === 延迟启动DeepStream管理器 ===
    bool startDeepStreamManagerDelayed() {
        if (!deepstream_manager_) {
            std::cout << "错误：DeepStream管理器尚未初始化" << std::endl;
            return false;
        }
        
        // 使用新的LVGL初始化检查机制
        std::cout << "等待LVGL完全初始化..." << std::endl;
        
        if (lvgl_interface_ptr_) {
            auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLInterface*>(lvgl_interface_ptr_);
            int wait_count = 0;
            const int MAX_WAIT_SECONDS = 20;  // 最大等待10秒
            
            while (!lvgl_if->isFullyInitialized() && wait_count < MAX_WAIT_SECONDS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                wait_count++;
                std::cout << "等待LVGL初始化完成... (InferenceWorker: " << (wait_count * 0.5) << "秒)" << std::endl;
            }
            
            if (lvgl_if->isFullyInitialized()) {
                std::cout << "✅ LVGL已完全初始化，继续启动DeepStream" << std::endl;
            } else {
                std::cout << "⚠️ 警告：LVGL初始化超时，继续启动DeepStream" << std::endl;
            }
        } else {
            std::cout << "警告：LVGL接口不可用，使用固定延迟" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        std::cout << "启动DeepStream管理器..." << std::endl;
        if (!deepstream_manager_->start()) {
            std::cout << "DeepStream 管理器启动失败" << std::endl;
            return false;
        }
        
        // 启动canvas更新线程 (如果支持LVGL界面集成)
        if (deepstream_manager_) {
            std::cout << "启动Canvas更新线程...（从integrated_main）" << std::endl;
            deepstream_manager_->startCanvasUpdateThread();
        }
        
        std::cout << "DeepStream 管理器延迟启动成功" << std::endl;
        return true;
    }
    
    float getCpuUsage() const { return 45.0f; } // 简化实现
    float getMemoryUsage() const { return 1024.0f; } // 简化实现
};

/**
 * LVGL UI管理器
 * 使用优化的LVGL界面实现
 */
class LVGLUIManager {
private:
    IntegratedDataBridge* data_bridge_;
    
    // 使用优化的LVGL界面实现
    std::unique_ptr<bamboo_cut::ui::LVGLInterface> lvgl_interface_;
    
    // 兼容性方法映射
    bool initialized_ = false;

public:
    LVGLUIManager(IntegratedDataBridge* bridge)
        : data_bridge_(bridge) {}
    
    ~LVGLUIManager() {
        cleanup();
    }

    // 兼容性方法：创建主界面
    bool create_main_screen() {
        return initialize();
    }

    // 兼容性方法：更新系统状态
    void update_system_status(const char* status, lv_color_t color) {
        std::cout << "System Status Updated: " << status << std::endl;
    }

    // 兼容性方法：更新检测数量
    void update_detection_count(int count) {
        std::cout << "Detection Count Updated: " << count << std::endl;
    }

    // 兼容性方法：更新FPS
    void update_fps(float fps) {
        std::cout << "FPS Updated: " << fps << std::endl;
    }
    
    // 获取LVGL界面指针（用于传递给DeepStreamManager）
    void* getLVGLInterface() {
        #ifdef ENABLE_LVGL
        return lvgl_interface_.get();
        #else
        return nullptr;
        #endif
    }
    
    bool initialize() {
        std::cout << "Initializing LVGL UI system with optimized interface..." << std::endl;
        
        #ifdef ENABLE_LVGL
        try {
            // 创建共享的DataBridge (创建一个简单的包装器)
            auto shared_bridge = std::make_shared<core::DataBridge>();
            
            // 创建优化的LVGL界面实例
            lvgl_interface_ = std::make_unique<bamboo_cut::ui::LVGLInterface>(shared_bridge);
            
            // 配置LVGL
            bamboo_cut::ui::LVGLConfig config;
            config.screen_width = 1280;
            config.screen_height = 800;
            config.refresh_rate = 60;
            config.enable_touch = true;
            config.touch_device = "/dev/input/event0";
            
            std::cout << "正在初始化LVGL界面..." << std::endl;
            if (!lvgl_interface_->initialize(config)) {
                std::cout << "LVGL interface initialization failed" << std::endl;
                return false;
            }
            
            std::cout << "LVGL界面初始化成功，正在启动界面线程..." << std::endl;
            // 启动界面线程
            if (!lvgl_interface_->start()) {
                std::cout << "LVGL interface start failed" << std::endl;
                return false;
            }
            
            // 给界面线程一些时间来稳定启动
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << "LVGL界面线程启动完成" << std::endl;
            
            std::cout << "Optimized LVGL interface created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "LVGL interface creation exception: " << e.what() << std::endl;
            return false;
        }
        #else
        std::cout << "LVGL not enabled, using placeholder UI" << std::endl;
        
        // 占位符实现：模拟初始化成功
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
        if (lvgl_interface_ && lvgl_interface_->isRunning()) {
            std::cout << "Using optimized LVGL interface main loop" << std::endl;
            // LVGL界面已经在自己的线程中运行，这里只需要等待
            while (!g_shutdown_requested && lvgl_interface_->isRunning()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        #else
        // 占位符主循环
        while (!g_shutdown_requested) {
            // 模拟界面更新
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 60fps
        }
        #endif
        
        std::cout << "LVGL main loop exited" << std::endl;
    }

private:
    void cleanup() {
        #ifdef ENABLE_LVGL
        if (lvgl_interface_) {
            lvgl_interface_->stop();
            lvgl_interface_.reset();
        }
        #endif
        
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
        
        // 初始化UI管理器
        ui_manager_ = std::make_unique<LVGLUIManager>(&data_bridge_);
        if (!ui_manager_->initialize()) {
            std::cout << "UI system initialization failed" << std::endl;
            return false;
        }
        
        // 初始化推理工作线程
        inference_worker_ = std::make_unique<InferenceWorkerThread>(&data_bridge_);
        
        // 传递LVGL界面指针给推理工作线程
        #ifdef ENABLE_LVGL
        if (ui_manager_ && ui_manager_->getLVGLInterface()) {
            inference_worker_->setLVGLInterface(ui_manager_->getLVGLInterface());
            std::cout << "LVGL界面指针已传递给推理工作线程" << std::endl;
        }
        #endif
        
        if (!inference_worker_->initialize()) {
            std::cout << "Inference system initialization failed" << std::endl;
            return false;
        }
        
        std::cout << "Integrated system initialization complete" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "Starting integrated system..." << std::endl;
        
        // 🔧 恢复启动顺序：LVGL优先获得DRM独占控制权
        std::cout << "🔧 LVGL优先启动，获得DRM独占控制权..." << std::endl;
        
        std::cout << "等待LVGL完全初始化并获得DRM控制..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2)); // 给LVGL充足时间获得DRM控制
        
        // 然后启动推理工作线程，GStreamer将在appsink模式下工作
        std::cout << "现在启动推理线程（appsink模式）..." << std::endl;
        if (!inference_worker_->start()) {
            std::cout << "Inference thread startup failed" << std::endl;
            return;
        }
        
        std::cout << "推理线程已启动，系统完全就绪" << std::endl;
        std::cout << "Press Ctrl+C to exit system" << std::endl;
        
        // 主线程运行UI (阻塞) - LVGL现在是次要的
        ui_manager_->runMainLoop();
        
        std::cout << "Starting system shutdown..." << std::endl;
        shutdown();
    }
    
    void shutdown() {
        std::cout << "Stopping inference thread..." << std::endl;
        if (inference_worker_) {
            inference_worker_->stop();
        }
        
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
                if (freopen("/dev/tty", "w", stdout) == nullptr) {
                    // 忽略恢复失败，继续执行
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
        // 临时恢复stdout显示异常信息
        if (freopen("/dev/tty", "w", stdout) == nullptr) {
            // 忽略恢复失败，继续执行
        }
        std::cout << "System exception: " << e.what() << std::endl;
        return -1;
    }
}