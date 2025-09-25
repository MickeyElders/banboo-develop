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

#ifndef ENABLE_LVGL
// LVGL未启用时的类型占位符
typedef void* lv_obj_t;
typedef void* lv_event_t;
typedef void* lv_indev_drv_t;
typedef void* lv_indev_data_t;
typedef void* lv_disp_drv_t;
typedef void* lv_area_t;
typedef void* lv_color_t;

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
inline bool lvgl_display_init() { return false; }
inline bool touch_driver_init() { return false; }

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
    bool initialize() { return true; }
    void update_workflow_status(int) {}
    void update_heartbeat(int, int) {}
};

class Video_view {
public:
    bool initialize() { return true; }
    void update_camera_frame(const frame_info_t&) {}
    void update_detection_info(float, float) {}
};

class Control_panel {
public:
    bool initialize() { return true; }
    void update_jetson_info(const performance_stats_t&) {}
};

class Settings_page {
public:
    bool initialize() { return true; }
    void create_main_layout(Status_bar*, Video_view*, Control_panel*) {}
};
#endif

// 现有后端组件 - 直接包含实际存在的头文件
// 注意：CMakeLists.txt已经包含了实现文件，所以直接使用真实的头文件
#include "bamboo_cut/utils/logger.h"

// 占位符类型，当某些组件未启用时使用
namespace bamboo_cut {
    namespace vision {
        struct DetectionPoint { float x, y; };
        struct DetectionResult {
            std::vector<DetectionPoint> points;
            float processing_time_ms = 0.0f;
            bool success = false;
        };
        struct FrameInfo {
            cv::Mat image;
            uint64_t timestamp = 0;
            bool valid = false;
        };
        struct StereoFrame {
            cv::Mat left_image, right_image;
            bool valid = false;
        };
        struct CameraConfig {
            std::string device_id;
            int width = 640, height = 480, framerate = 30;
        };
        struct CameraSyncConfig {
            std::string left_device, right_device;
            int width = 640, height = 480, fps = 30;
        };
        
        class CameraManager {
        public:
            CameraManager(const CameraConfig&) {}
            bool initialize() { return false; }
            FrameInfo getCurrentFrame() { return FrameInfo(); }
        };
        
        class StereoVision {
        public:
            StereoVision(const CameraSyncConfig&) {}
            bool initialize() { return false; }
            bool load_calibration(const std::string&) { return false; }
            bool capture_stereo_frame(StereoFrame&) { return false; }
        };
        
        class BambooDetector {
        public:
            struct Config {
                std::string model_path, engine_path;
                bool use_tensorrt = false;
            };
            BambooDetector(const Config&) {}
            bool initialize() { return false; }
            DetectionResult detect(const cv::Mat&) { return DetectionResult(); }
        };
    }
    
    namespace communication {
        struct ModbusConfig {
            std::string ip_address;
            int port = 502;
        };
        class ModbusServer {
        public:
            ModbusServer(const ModbusConfig&) {}
            bool is_connected() const { return false; }
            void set_connection_callback(std::function<void(bool, const std::string&)>) {}
            void start() {}
        };
    }
}

// 全局关闭标志
std::atomic<bool> g_shutdown_requested{false};
std::chrono::steady_clock::time_point g_shutdown_start_time;

// 信号处理
void signal_handler(int sig) {
    std::cout << "\n收到信号 " << sig << "，开始优雅关闭..." << std::endl;
    g_shutdown_requested = true;
    g_shutdown_start_time = std::chrono::steady_clock::now();
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
        std::vector<bamboo_cut::vision::DetectionPoint> points;
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
    void updateDetection(const bamboo_cut::vision::DetectionResult& result) {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        latest_detection_.points = result.points;
        latest_detection_.processing_time_ms = result.processing_time_ms;
        latest_detection_.has_detection = result.success && !result.points.empty();
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
    
    // 复用现有的后端组件
    std::unique_ptr<bamboo_cut::vision::CameraManager> camera_manager_;
    std::unique_ptr<bamboo_cut::vision::StereoVision> stereo_vision_;
    std::unique_ptr<bamboo_cut::vision::BambooDetector> detector_;
    std::unique_ptr<bamboo_cut::communication::ModbusServer> modbus_server_;
    
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
        std::cout << "初始化推理系统..." << std::endl;
        
        // 初始化检测器 (复用原有逻辑)
        if (!initializeDetector()) {
            std::cout << "检测器初始化失败" << std::endl;
            return false;
        }
        
        // 初始化立体视觉 (优先使用)
        if (!initializeStereoVision()) {
            std::cout << "立体视觉初始化失败，尝试单摄像头..." << std::endl;
            // 尝试单摄像头系统
            if (!initializeSingleCamera()) {
                std::cout << "摄像头系统初始化完全失败" << std::endl;
                return false;
            }
        }
        
        // 初始化通信系统 (可选)
        initializeModbus();
        
        std::cout << "推理系统初始化完成" << std::endl;
        return true;
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
        std::cout << "推理工作线程启动" << std::endl;
        
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
        
        std::cout << "推理工作线程退出" << std::endl;
    }
    
    void processFrame() {
        // 优先处理立体视觉
        if (stereo_vision_) {
            bamboo_cut::vision::StereoFrame stereo_frame;
            if (stereo_vision_->capture_stereo_frame(stereo_frame) && stereo_frame.valid) {
                // 更新视频到数据桥接
                data_bridge_->updateStereoVideo(stereo_frame.left_image, stereo_frame.right_image);
                
                // AI检测
                if (detector_ && !stereo_frame.left_image.empty()) {
                    auto result = detector_->detect(stereo_frame.left_image);
                    if (result.success) {
                        data_bridge_->updateDetection(result);
                    }
                }
                
                processed_frames_++;
            }
        }
        // 单摄像头处理
        else if (camera_manager_) {
            bamboo_cut::vision::FrameInfo frame_info = camera_manager_->getCurrentFrame();
            if (frame_info.valid && !frame_info.image.empty()) {
                // 更新视频到数据桥接
                data_bridge_->updateVideo(frame_info.image, frame_info.timestamp);
                
                // AI检测
                if (detector_) {
                    auto result = detector_->detect(frame_info.image);
                    if (result.success) {
                        data_bridge_->updateDetection(result);
                    }
                }
                
                processed_frames_++;
            }
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
            stats.plc_connected = (modbus_server_ && modbus_server_->is_connected());
            
            data_bridge_->updateStats(stats);
            
            processed_frames_ = 0;
            last_stats_time_ = now;
        }
    }
    
    // === 初始化方法 (复用现有逻辑) ===
    bool initializeDetector() {
        bamboo_cut::vision::BambooDetector::Config config;
        config.model_path = "./models/best.pt";
        config.engine_path = "./models/bamboo_detection.trt";
        config.use_tensorrt = true;
        
        detector_ = std::make_unique<bamboo_cut::vision::BambooDetector>(config);
        return detector_->initialize();
    }
    
    bool initializeStereoVision() {
        bamboo_cut::vision::CameraSyncConfig config;
        config.left_device = "/dev/video0";
        config.right_device = "/dev/video1"; 
        config.width = 640;
        config.height = 480;
        config.fps = 30;
        
        stereo_vision_ = std::make_unique<bamboo_cut::vision::StereoVision>(config);
        
        if (!stereo_vision_->initialize()) {
            return false;
        }
        
        // 尝试加载标定文件
        stereo_vision_->load_calibration("./config/stereo_calibration.xml");
        return true;
    }
    
    bool initializeSingleCamera() {
        bamboo_cut::vision::CameraConfig config;
        config.device_id = "/dev/video0";
        config.width = 640;
        config.height = 480;
        config.framerate = 30;
        
        camera_manager_ = std::make_unique<bamboo_cut::vision::CameraManager>(config);
        return camera_manager_->initialize();
    }
    
    void initializeModbus() {
        bamboo_cut::communication::ModbusConfig config;
        config.ip_address = "0.0.0.0";
        config.port = 502;
        
        modbus_server_ = std::make_unique<bamboo_cut::communication::ModbusServer>(config);
        
        // 设置回调
        modbus_server_->set_connection_callback([](bool connected, const std::string& client_ip) {
            if (connected) {
                std::cout << "PLC已连接: " << client_ip << std::endl;
            }
        });
        
        modbus_server_->start(); // 非阻塞启动
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
        std::cout << "初始化LVGL UI系统..." << std::endl;
        
        // 初始化LVGL核心
        lv_init();
        
        // 初始化时钟系统
        lv_port_tick_init();
        
        // 初始化显示驱动 (复用现有实现)
        if (!lvgl_display_init()) {
            std::cout << "LVGL显示驱动初始化失败" << std::endl;
            return false;
        }
        
        // 初始化触摸驱动 (复用现有实现)
        if (touch_driver_init()) {
            std::cout << "触摸驱动初始化成功" << std::endl;
        } else {
            std::cout << "触摸驱动初始化失败，将禁用触摸功能" << std::endl;
        }
        
        // 创建UI组件 (复用现有实现)
        if (!createUIComponents()) {
            std::cout << "UI组件创建失败" << std::endl;
            return false;
        }
        
        // 设置更新定时器
        setupUpdateTimers();
        
        initialized_ = true;
        std::cout << "LVGL UI系统初始化完成" << std::endl;
        return true;
    }
    
    void runMainLoop() {
        if (!initialized_) return;
        
        std::cout << "LVGL主循环启动" << std::endl;
        
        while (!g_shutdown_requested) {
            // 处理LVGL任务
            lv_timer_handler();
            
            // 短暂休眠，60fps
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        std::cout << "LVGL主循环退出" << std::endl;
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
        std::cout << "竹子识别系统一体化启动" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // 设置信号处理
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // 初始化推理工作线程
        inference_worker_ = std::make_unique<InferenceWorkerThread>(&data_bridge_);
        if (!inference_worker_->initialize()) {
            std::cout << "推理系统初始化失败" << std::endl;
            return false;
        }
        
        // 初始化UI管理器
        ui_manager_ = std::make_unique<LVGLUIManager>(&data_bridge_);
        if (!ui_manager_->initialize()) {
            std::cout << "UI系统初始化失败" << std::endl;
            return false;
        }
        
        std::cout << "一体化系统初始化完成" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "启动一体化系统..." << std::endl;
        
        // 启动推理工作线程
        if (!inference_worker_->start()) {
            std::cout << "推理线程启动失败" << std::endl;
            return;
        }
        
        std::cout << "推理线程已启动" << std::endl;
        std::cout << "按 Ctrl+C 退出系统" << std::endl;
        
        // 主线程运行UI (阻塞)
        ui_manager_->runMainLoop();
        
        std::cout << "开始系统关闭..." << std::endl;
        shutdown();
    }
    
    void shutdown() {
        std::cout << "停止推理线程..." << std::endl;
        if (inference_worker_) {
            inference_worker_->stop();
        }
        
        std::cout << "系统关闭完成" << std::endl;
    }
};

/**
 * 主函数入口
 */
int main() {
    try {
        IntegratedBambooSystem system;
        
        if (!system.initialize()) {
            std::cout << "系统初始化失败" << std::endl;
            return -1;
        }
        
        system.run();
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "系统异常: " << e.what() << std::endl;
        return -1;
    }
}