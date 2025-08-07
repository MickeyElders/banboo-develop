#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

// GStreamer 头文件
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

// V4L2 头文件
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

// CUDA 头文件 (如果可用)
// 注意：ENABLE_CUDA应该只在CMake检测到CUDA时定义
#ifdef ENABLE_CUDA
    // 只在真正有CUDA时才包含这些头文件
    // 如果编译失败说明CMake配置有问题
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif

namespace bamboo_cut {
namespace vision {

/**
 * @brief 硬件加速类型枚举
 */
enum class HardwareAccelerationType {
    NONE,           // 无硬件加速
    V4L2_HW,        // V4L2 硬件加速
    GSTREAMER_HW,   // GStreamer 硬件加速
    CUDA_HW,        // CUDA 硬件加速
    MIXED_HW        // 混合硬件加速
};

/**
 * @brief 摄像头接口类型
 */
enum class CameraInterfaceType {
    USB,            // USB 摄像头
    MIPI_CSI,       // MIPI CSI 接口
    GMSL,           // GMSL 接口
    ETHERNET        // 网络摄像头
};

/**
 * @brief 硬件加速配置
 */
struct HardwareAccelerationConfig {
    // 硬件加速类型
    HardwareAccelerationType acceleration_type{HardwareAccelerationType::MIXED_HW};
    CameraInterfaceType interface_type{CameraInterfaceType::MIPI_CSI};
    
    // 图像格式和分辨率
    int width{1920};
    int height{1080};
    int fps{30};
    std::string pixel_format{"NV12"};  // NV12, YUYV, MJPEG, etc.
    
    // 硬件加速参数
    bool enable_gpu_memory_mapping{true};    // GPU 内存映射
    bool enable_zero_copy{true};             // 零拷贝传输
    bool enable_hardware_isp{true};          // 硬件 ISP
    bool enable_hardware_encoding{true};     // 硬件编码
    bool enable_hardware_decoding{true};     // 硬件解码
    
    // V4L2 硬件参数
    int v4l2_buffer_count{4};                // V4L2 缓冲区数量
    bool enable_v4l2_dma{true};              // V4L2 DMA
    bool enable_v4l2_hw_format{true};        // V4L2 硬件格式
    
    // GStreamer 硬件参数
    std::string gst_pipeline;                // 自定义 GStreamer 流水线
    bool enable_gst_hw_convert{true};        // GStreamer 硬件转换
    bool enable_gst_hw_scale{true};          // GStreamer 硬件缩放
    bool enable_gst_hw_filter{true};         // GStreamer 硬件滤波
    
    // CUDA 硬件参数
    int cuda_device_id{0};                   // CUDA 设备 ID
    bool enable_cuda_memory_pool{true};      // CUDA 内存池
    bool enable_cuda_streams{true};          // CUDA 流
    
    // 性能优化
    int num_capture_threads{2};              // 捕获线程数
    bool enable_async_capture{true};         // 异步捕获
    bool enable_frame_skip{false};           // 跳帧处理
    int max_queue_size{10};                  // 最大队列大小
    
    // ISP 硬件参数
    bool enable_auto_exposure{true};         // 自动曝光
    bool enable_auto_white_balance{true};    // 自动白平衡
    bool enable_auto_focus{true};            // 自动对焦
    bool enable_noise_reduction{true};       // 降噪
    bool enable_edge_enhancement{true};      // 边缘增强
    
    HardwareAccelerationConfig() = default;
    bool validate() const;
};

/**
 * @brief 硬件加速性能统计
 */
struct HardwareAccelerationStats {
    // 捕获性能
    uint64_t total_frames_captured{0};
    uint64_t total_frames_processed{0};
    uint64_t dropped_frames{0};
    double avg_capture_time_ms{0.0};
    double avg_processing_time_ms{0.0};
    double current_fps{0.0};
    double target_fps{0.0};
    
    // 硬件加速性能
    double gpu_memory_usage_mb{0.0};
    double cpu_usage_percent{0.0};
    double memory_bandwidth_gbps{0.0};
    double gpu_utilization_percent{0.0};
    
    // 延迟统计
    double avg_latency_ms{0.0};
    double min_latency_ms{0.0};
    double max_latency_ms{0.0};
    double p95_latency_ms{0.0};
    double p99_latency_ms{0.0};
    
    // 硬件加速状态
    bool gpu_memory_mapping_active{false};
    bool zero_copy_active{false};
    bool hardware_isp_active{false};
    bool hardware_encoding_active{false};
    bool hardware_decoding_active{false};
    
    // 错误统计
    uint64_t hardware_errors{0};
    uint64_t memory_errors{0};
    uint64_t timeout_errors{0};
    
    HardwareAccelerationStats() = default;
};

/**
 * @brief 硬件加速帧结构
 */
struct HardwareAcceleratedFrame {
    cv::Mat cpu_frame;                       // CPU 内存中的帧
    cv::Mat gpu_frame;                       // GPU 内存中的帧 (如果启用)
    
    // 硬件加速信息
    HardwareAccelerationType acceleration_type{HardwareAccelerationType::NONE};
    std::string pixel_format;
    int width{0};
    int height{0};
    int channels{0};
    
    // 时间戳和性能信息
    uint64_t capture_timestamp{0};
    uint64_t processing_timestamp{0};
    double capture_time_ms{0.0};
    double processing_time_ms{0.0};
    
    // 硬件加速状态
    bool gpu_memory_mapped{false};
    bool zero_copy_used{false};
    bool hardware_processed{false};
    
    HardwareAcceleratedFrame() = default;
    bool is_valid() const;
};

/**
 * @brief 硬件加速摄像头管理器类
 * 
 * 提供完整的硬件加速摄像头功能，包括 GPU 加速、硬件 ISP、零拷贝传输等
 */
class HardwareAcceleratedCamera {
public:
    HardwareAcceleratedCamera();
    explicit HardwareAcceleratedCamera(const HardwareAccelerationConfig& config);
    ~HardwareAcceleratedCamera();

    // 禁用拷贝
    HardwareAcceleratedCamera(const HardwareAcceleratedCamera&) = delete;
    HardwareAcceleratedCamera& operator=(const HardwareAcceleratedCamera&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_capturing() const { return capturing_; }

    // 摄像头控制
    bool start_capture();
    void stop_capture();
    bool capture_frame(HardwareAcceleratedFrame& frame);
    bool capture_frame_async(std::function<void(const HardwareAcceleratedFrame&)> callback);

    // 硬件加速功能
    bool enable_gpu_memory_mapping(bool enable);
    bool enable_zero_copy(bool enable);
    bool enable_hardware_isp(bool enable);
    bool enable_hardware_encoding(bool enable);
    bool enable_hardware_decoding(bool enable);

    // 摄像头参数控制
    bool set_resolution(int width, int height);
    bool set_fps(int fps);
    bool set_pixel_format(const std::string& format);
    bool set_exposure(int exposure);
    bool set_gain(int gain);
    bool set_white_balance(int temperature);
    bool set_focus(int focus);

    // 配置管理
    void set_config(const HardwareAccelerationConfig& config);
    HardwareAccelerationConfig get_config() const { return config_; }

    // 性能监控
    HardwareAccelerationStats get_performance_stats() const;
    void reset_performance_stats();

    // 回调函数类型
    using FrameCallback = std::function<void(const HardwareAcceleratedFrame&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    // 设置回调
    void set_frame_callback(FrameCallback callback);
    void set_error_callback(ErrorCallback callback);

private:
    // V4L2 硬件加速
    bool initialize_v4l2_hardware();
    bool setup_v4l2_buffers();
    bool capture_v4l2_frame(HardwareAcceleratedFrame& frame);
    void cleanup_v4l2_hardware();

    // GStreamer 硬件加速
    bool initialize_gstreamer_hardware();
    bool setup_gstreamer_pipeline();
    bool capture_gstreamer_frame(HardwareAcceleratedFrame& frame);
    void cleanup_gstreamer_hardware();

    // CUDA 硬件加速
    bool initialize_cuda_hardware();
    bool setup_cuda_memory_pool();
    bool map_gpu_memory(cv::Mat& cpu_frame, cv::Mat& gpu_frame);
    void unmap_gpu_memory(cv::Mat& gpu_frame);
    void cleanup_cuda_hardware();

    // 硬件 ISP 处理
    bool apply_hardware_isp(HardwareAcceleratedFrame& frame);
    bool apply_auto_exposure(HardwareAcceleratedFrame& frame);
    bool apply_auto_white_balance(HardwareAcceleratedFrame& frame);
    bool apply_noise_reduction(HardwareAcceleratedFrame& frame);
    bool apply_edge_enhancement(HardwareAcceleratedFrame& frame);

    // 异步捕获
    void capture_thread_function();
    void process_captured_frame(const HardwareAcceleratedFrame& frame);

    // 性能监控
    void update_performance_stats(const HardwareAcceleratedFrame& frame);
    void measure_gpu_utilization();
    void measure_memory_usage();

    // 错误处理
    void set_error(const std::string& error);
    bool handle_hardware_error(const std::string& error);

    // 配置和状态
    HardwareAccelerationConfig config_;
    bool initialized_{false};
    bool capturing_{false};

    // 硬件资源
    int v4l2_fd_{-1};                        // V4L2 文件描述符
    std::vector<void*> v4l2_buffers_;        // V4L2 缓冲区
    GstElement* gst_pipeline_{nullptr};      // GStreamer 流水线
    GstElement* gst_appsrc_{nullptr};        // GStreamer 源
    GstElement* gst_appsink_{nullptr};       // GStreamer 接收器

#ifdef ENABLE_CUDA
    cudaStream_t cuda_stream_{nullptr};      // CUDA 流
    void* cuda_memory_pool_{nullptr};        // CUDA 内存池
    size_t cuda_memory_pool_size_{0};        // CUDA 内存池大小
#endif

    // 异步捕获
    std::thread capture_thread_;
    std::atomic<bool> stop_capture_{false};
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_queue_cv_;
    std::queue<HardwareAcceleratedFrame> frame_queue_;

    // 回调函数
    FrameCallback frame_callback_;
    ErrorCallback error_callback_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    HardwareAccelerationStats performance_stats_;

    // 错误处理
    std::string last_error_;
};

/**
 * @brief 多摄像头硬件加速管理器
 * 
 * 管理多个硬件加速摄像头，支持同步和异步捕获
 */
class MultiCameraHardwareManager {
public:
    struct MultiCameraConfig {
        std::vector<std::string> camera_devices;     // 摄像头设备列表
        std::vector<HardwareAccelerationConfig> camera_configs; // 每个摄像头的配置
        bool enable_synchronization{true};           // 启用同步
        bool enable_stereo_mode{false};              // 立体模式
        int sync_tolerance_ms{10};                   // 同步容差 (毫秒)
        int max_cameras{4};                          // 最大摄像头数量
        
        MultiCameraConfig() = default;
        bool validate() const;
    };

    MultiCameraHardwareManager();
    explicit MultiCameraHardwareManager(const MultiCameraConfig& config);
    ~MultiCameraHardwareManager();

    // 禁用拷贝
    MultiCameraHardwareManager(const MultiCameraHardwareManager&) = delete;
    MultiCameraHardwareManager& operator=(const MultiCameraHardwareManager&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 多摄像头控制
    bool start_all_cameras();
    void stop_all_cameras();
    bool capture_synchronized_frames(std::vector<HardwareAcceleratedFrame>& frames);
    bool capture_stereo_frames(HardwareAcceleratedFrame& left_frame, HardwareAcceleratedFrame& right_frame);

    // 摄像头管理
    bool add_camera(const std::string& device, const HardwareAccelerationConfig& config);
    bool remove_camera(const std::string& device);
    bool enable_camera(const std::string& device, bool enable);
    std::vector<std::string> get_active_cameras() const;

    // 同步控制
    bool enable_synchronization(bool enable);
    bool set_sync_tolerance(int tolerance_ms);
    bool calibrate_synchronization();

    // 配置管理
    void set_config(const MultiCameraConfig& config);
    MultiCameraConfig get_config() const { return config_; }

    // 性能监控
    std::vector<HardwareAccelerationStats> get_all_camera_stats() const;
    HardwareAccelerationStats get_combined_stats() const;

    // 回调函数
    using MultiFrameCallback = std::function<void(const std::vector<HardwareAcceleratedFrame>&)>;
    using StereoFrameCallback = std::function<void(const HardwareAcceleratedFrame&, const HardwareAcceleratedFrame&)>;

    void set_multi_frame_callback(MultiFrameCallback callback);
    void set_stereo_frame_callback(StereoFrameCallback callback);

private:
    // 内部方法
    bool initialize_cameras();
    bool setup_synchronization();
    bool capture_individual_frames(std::vector<HardwareAcceleratedFrame>& frames);
    bool synchronize_frames(std::vector<HardwareAcceleratedFrame>& frames);
    void cleanup_cameras();

    // 配置和状态
    MultiCameraConfig config_;
    bool initialized_{false};

    // 摄像头管理
    std::vector<std::unique_ptr<HardwareAcceleratedCamera>> cameras_;
    std::vector<std::string> camera_devices_;
    std::vector<bool> camera_enabled_;

    // 同步控制
    std::mutex sync_mutex_;
    std::condition_variable sync_cv_;
    std::vector<uint64_t> last_frame_timestamps_;

    // 回调函数
    MultiFrameCallback multi_frame_callback_;
    StereoFrameCallback stereo_frame_callback_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    std::vector<HardwareAccelerationStats> camera_stats_;
};

} // namespace vision
} // namespace bamboo_cut 