#ifndef BAMBOO_CUT_CAMERA_MANAGER_H
#define BAMBOO_CUT_CAMERA_MANAGER_H

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

namespace bamboo_cut {
namespace vision {

struct CameraConfig {
    std::string device_id = "/dev/video0";  // 设备ID
    int width = 1920;
    int height = 1080;
    int framerate = 30;
    std::string pipeline;                   // 自定义GStreamer pipeline
    bool use_hardware_acceleration = true;
    int buffer_size = 10;                   // 帧缓冲区大小
    bool auto_exposure = true;
    int exposure_value = -1;                // 手动曝光值 (-1为自动)
    int gain_value = -1;                    // 手动增益值 (-1为自动)
};

struct FrameInfo {
    cv::Mat image;
    uint64_t timestamp;
    int frame_id;
    bool valid;
    
    FrameInfo() : timestamp(0), frame_id(0), valid(false) {}
};

class CameraManager {
public:
    using FrameCallback = std::function<void(const FrameInfo&)>;
    
    explicit CameraManager(const CameraConfig& config);
    virtual ~CameraManager();
    
    // 初始化摄像头
    bool initialize();
    
    // 启动/停止捕获
    bool startCapture();
    bool stopCapture();
    
    // 获取当前帧
    FrameInfo getCurrentFrame();
    
    // 设置帧回调函数
    void setFrameCallback(FrameCallback callback);
    
    // 摄像头控制
    bool setExposure(int exposure_value);
    bool setGain(int gain_value);
    bool setAutoExposure(bool enable);
    
    // 状态查询
    bool isInitialized() const { return initialized_; }
    bool isCapturing() const { return capturing_; }
    bool isRunning() const;
    int getCameraCount() const;
    cv::Mat getLatestFrame(int camera_id) const;
    std::vector<cv::Mat> getLatestFrames() const;
    
    // 性能统计
    struct PerformanceStats {
        float fps;
        int dropped_frames;
        float avg_processing_time_ms;
        uint64_t total_frames;
    };
    
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();
    
    // 摄像头信息
    struct CameraInfo {
        std::string device_path;
        std::string driver;
        std::string card_name;
        std::vector<std::string> supported_formats;
        int current_width;
        int current_height;
        float current_fps;
    };
    
    CameraInfo getCameraInfo() const;
    
    // 多摄像头支持
    static std::vector<std::string> listAvailableCameras();

private:
    CameraConfig config_;
    bool initialized_;
    std::atomic<bool> capturing_;
    
    // GStreamer相关
    GstElement* pipeline_;
    GstElement* appsink_;
    
    // 线程管理
    std::unique_ptr<std::thread> capture_thread_;
    std::atomic<bool> stop_capture_;
    
    // 帧缓冲
    std::queue<FrameInfo> frame_buffer_;
    mutable std::mutex buffer_mutex_;
    std::condition_variable buffer_condition_;
    
    // 回调函数
    FrameCallback frame_callback_;
    std::mutex callback_mutex_;
    
    // 性能统计
    mutable PerformanceStats stats_;
    mutable std::mutex stats_mutex_;
    uint64_t frame_counter_;
    std::chrono::steady_clock::time_point last_stats_time_;
    
    // 内部方法
    bool initializeGStreamer();
    std::string buildGStreamerPipeline(int camera_id);
    void captureLoop();
    
    // GStreamer回调
    static GstFlowReturn onNewSample(GstAppSink* appsink, gpointer user_data);
    GstFlowReturn handleNewSample(GstAppSink* appsink);
    
    // 帧处理
    void processFrame(const cv::Mat& frame);
    void updatePerformanceStats();
    
    // 错误处理
    static gboolean onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);
    void handleBusMessage(GstMessage* message);
};

// 双摄像头管理器
class DualCameraManager {
public:
    struct DualCameraConfig {
        CameraConfig camera1_config;
        CameraConfig camera2_config;
        bool enable_synchronization = true;
        float sync_tolerance_ms = 10.0f;    // 同步容差
    };
    
    explicit DualCameraManager(const DualCameraConfig& config);
    virtual ~DualCameraManager();
    
    bool initialize();
    bool startCapture();
    bool stopCapture();
    
    // 同步帧对
    struct SynchronizedFrames {
        FrameInfo frame1;
        FrameInfo frame2;
        bool synchronized;
        uint64_t timestamp_diff;
    };
    
    using SyncFrameCallback = std::function<void(const SynchronizedFrames&)>;
    void setSyncFrameCallback(SyncFrameCallback callback);
    
    SynchronizedFrames getSynchronizedFrames();
    
    // 单独访问摄像头
    std::shared_ptr<CameraManager> getCamera1() { return camera1_; }
    std::shared_ptr<CameraManager> getCamera2() { return camera2_; }

private:
    DualCameraConfig config_;
    std::shared_ptr<CameraManager> camera1_;
    std::shared_ptr<CameraManager> camera2_;
    
    SyncFrameCallback sync_callback_;
    std::mutex sync_mutex_;
    
    // 同步逻辑
    std::queue<FrameInfo> sync_buffer1_;
    std::queue<FrameInfo> sync_buffer2_;
    std::mutex sync_buffer_mutex_;
    
    void onCamera1Frame(const FrameInfo& frame);
    void onCamera2Frame(const FrameInfo& frame);
    void trySync();
    
    bool areSynchronized(const FrameInfo& frame1, const FrameInfo& frame2);
};

} // namespace vision
} // namespace bamboo_cut

#endif // BAMBOO_CUT_CAMERA_MANAGER_H 