#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_DEEPSTREAM
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <nvds_meta.h>
#include <nvds_parse.h>
#endif

#include "bamboo_cut/core/types.h"

namespace bamboo_cut {
namespace vision {

#ifdef ENABLE_DEEPSTREAM
// DeepStream 元数据结构
struct DeepStreamMeta {
    NvDsObjectMeta* object_meta;
    NvDsFrameMeta* frame_meta;
    NvDsBatchMeta* batch_meta;
    uint32_t frame_number;
    uint64_t timestamp;
    
    DeepStreamMeta() : object_meta(nullptr), frame_meta(nullptr), 
                      batch_meta(nullptr), frame_number(0), timestamp(0) {}
};

// DeepStream 检测结果
struct DeepStreamDetection {
    core::Rectangle bounding_box;
    float confidence;
    uint32_t class_id;
    std::string label;
    uint32_t track_id;
    
    DeepStreamDetection() : confidence(0.0f), class_id(0), track_id(0) {}
};
#endif

/**
 * @brief DeepStream 流水线配置
 */
struct DeepStreamConfig {
    std::string model_path;              // TensorRT 模型路径
    std::string label_file;              // 标签文件路径
    int input_width{1280};               // 输入宽度
    int input_height{720};               // 输入高度
    int batch_size{1};                   // 批处理大小
    float confidence_threshold{0.5f};    // 置信度阈值
    float nms_threshold{0.4f};           // NMS 阈值
    bool enable_tracking{true};          // 启用目标跟踪
    bool enable_fp16{true};              // 启用 FP16
    int gpu_id{0};                       // GPU 设备 ID
    std::string pipeline_config;         // 自定义流水线配置
    
    // GStreamer 配置
    std::string source_type{"v4l2"};     // 视频源类型
    std::string source_device{"/dev/video0"}; // 视频源设备
    int framerate{30};                   // 帧率
    std::string sink_type{"appsink"};    // 输出类型
    
    DeepStreamConfig() = default;
    bool validate() const;
};

/**
 * @brief DeepStream GStreamer 流水线类
 * 
 * 基于 DeepStream SDK 的高性能 GStreamer 流水线
 */
class DeepStreamPipeline {
public:
    explicit DeepStreamPipeline(const DeepStreamConfig& config = DeepStreamConfig{});
    ~DeepStreamPipeline();

    // 禁用拷贝
    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 流水线控制
    bool start();
    void stop();
    bool is_running() const { return running_; }
    void pause();
    void resume();

    // 数据处理
    bool push_frame(const cv::Mat& frame);
    std::vector<core::DetectionResult> pull_detections();
    
    // 批处理
    bool push_frames(const std::vector<cv::Mat>& frames);
    std::vector<std::vector<core::DetectionResult>> pull_detections_batch();

    // 回调函数类型
    using DetectionCallback = std::function<void(const std::vector<core::DetectionResult>&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    using StatusCallback = std::function<void(const std::string&)>;

    // 设置回调
    void set_detection_callback(DetectionCallback callback);
    void set_error_callback(ErrorCallback callback);
    void set_status_callback(StatusCallback callback);

    // 配置管理
    void set_config(const DeepStreamConfig& config);
    DeepStreamConfig get_config() const { return config_; }

    // 性能统计
    struct PerformanceStats {
        uint64_t total_frames_processed{0};
        uint64_t total_detections{0};
        double avg_processing_time_ms{0.0};
        double min_processing_time_ms{0.0};
        double max_processing_time_ms{0.0};
        double fps{0.0};
        double gpu_utilization{0.0};
        double memory_utilization{0.0};
        core::Timestamp last_update;
    };
    PerformanceStats get_performance_stats() const;

private:
#ifdef ENABLE_DEEPSTREAM
    // GStreamer 组件
    GstElement* pipeline_{nullptr};
    GstElement* source_{nullptr};
    GstElement* sink_{nullptr};
    GstElement* nvinfer_{nullptr};
    GstElement* nvtracker_{nullptr};
    GstElement* nvmultistreamtiler_{nullptr};
    GstElement* nvvideoconvert_{nullptr};
    GstElement* nvosd_{nullptr};
    
    // 应用源和接收器
    GstAppSrc* appsrc_{nullptr};
    GstAppSink* appsink_{nullptr};
    
    // 内部方法
    bool create_pipeline();
    bool create_elements();
    bool link_elements();
    bool set_element_properties();
    bool setup_callbacks();
    
    // GStreamer 回调
    static void on_new_sample(GstElement* sink, gpointer user_data);
    static void on_error(GstBus* bus, GstMessage* msg, gpointer user_data);
    static void on_warning(GstBus* bus, GstMessage* msg, gpointer user_data);
    static void on_eos(GstBus* bus, GstMessage* msg, gpointer user_data);
    
    // 元数据处理
    std::vector<core::DetectionResult> parse_detections(GstBuffer* buffer);
    bool extract_metadata(GstBuffer* buffer, std::vector<DeepStreamMeta>& metadata);
    core::DetectionResult convert_detection(const DeepStreamDetection& ds_detection);
    
    // 内存管理
    GstBuffer* create_gst_buffer(const cv::Mat& frame);
    void release_gst_buffer(GstBuffer* buffer);
#endif

    // 配置和状态
    DeepStreamConfig config_;
    bool initialized_{false};
    bool running_{false};

    // 回调函数
    DetectionCallback detection_callback_;
    ErrorCallback error_callback_;
    StatusCallback status_callback_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

/**
 * @brief DeepStream 多流处理器
 * 
 * 处理多个视频流的 DeepStream 流水线
 */
class DeepStreamMultiStream {
public:
    struct MultiStreamConfig {
        std::vector<std::string> source_devices;  // 多个视频源设备
        DeepStreamConfig base_config;             // 基础配置
        int num_streams{2};                       // 流数量
        bool enable_synchronization{true};        // 启用同步
        int sync_tolerance_ms{10};                // 同步容差
        
        MultiStreamConfig() = default;
        bool validate() const;
    };

    explicit DeepStreamMultiStream(const MultiStreamConfig& config = MultiStreamConfig{});
    ~DeepStreamMultiStream();

    // 禁用拷贝
    DeepStreamMultiStream(const DeepStreamMultiStream&) = delete;
    DeepStreamMultiStream& operator=(const DeepStreamMultiStream&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 流控制
    bool start_all_streams();
    void stop_all_streams();
    bool is_running() const { return running_; }

    // 数据处理
    bool push_frames(const std::vector<cv::Mat>& frames);
    std::vector<std::vector<core::DetectionResult>> pull_detections();

    // 配置管理
    void set_config(const MultiStreamConfig& config);
    MultiStreamConfig get_config() const { return config_; }

    // 性能统计
    struct MultiStreamStats {
        std::vector<DeepStreamPipeline::PerformanceStats> stream_stats;
        double avg_fps{0.0};
        double total_gpu_utilization{0.0};
        double total_memory_utilization{0.0};
    };
    MultiStreamStats get_performance_stats() const;

private:
    // 内部方法
    bool create_streams();
    bool synchronize_streams();

    // 配置和状态
    MultiStreamConfig config_;
    bool initialized_{false};
    bool running_{false};

    // 流水线组件
    std::vector<std::unique_ptr<DeepStreamPipeline>> pipelines_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    MultiStreamStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

} // namespace vision
} // namespace bamboo_cut 