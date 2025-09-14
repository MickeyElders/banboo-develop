#include "bamboo_cut/vision/deepstream_pipeline.h"
#include <iostream>
#include <chrono>
#include <mutex>

namespace bamboo_cut {
namespace vision {

// DeepStreamConfig validation implementation
bool DeepStreamConfig::validate() const {
    if (model_path.empty()) {
        return false;
    }
    if (input_width <= 0 || input_height <= 0) {
        return false;
    }
    if (batch_size <= 0) {
        return false;
    }
    if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
        return false;
    }
    if (nms_threshold < 0.0f || nms_threshold > 1.0f) {
        return false;
    }
    return true;
}

DeepStreamPipeline::DeepStreamPipeline() : config_(), initialized_(false), running_(false) {
    std::cout << "创建DeepStreamPipeline实例" << std::endl;
}

DeepStreamPipeline::DeepStreamPipeline(const DeepStreamConfig& config) 
    : config_(config), initialized_(false), running_(false) {
    std::cout << "创建DeepStreamPipeline实例，使用自定义配置" << std::endl;
}

DeepStreamPipeline::~DeepStreamPipeline() {
    shutdown();
    std::cout << "销毁DeepStreamPipeline实例" << std::endl;
}

bool DeepStreamPipeline::initialize() {
    if (initialized_) {
        std::cout << "DeepStreamPipeline已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化DeepStreamPipeline..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "DeepStreamPipeline配置无效" << std::endl;
            return false;
        }
        
#ifdef ENABLE_DEEPSTREAM
        // 初始化GStreamer
        gst_init(nullptr, nullptr);
        
        // 创建流水线
        if (!create_pipeline()) {
            std::cerr << "创建DeepStream流水线失败" << std::endl;
            return false;
        }
        
        // 创建元素
        if (!create_elements()) {
            std::cerr << "创建DeepStream元素失败" << std::endl;
            return false;
        }
        
        // 链接元素
        if (!link_elements()) {
            std::cerr << "链接DeepStream元素失败" << std::endl;
            return false;
        }
        
        // 设置属性
        if (!set_element_properties()) {
            std::cerr << "设置元素属性失败" << std::endl;
            return false;
        }
        
        // 设置回调
        if (!setup_callbacks()) {
            std::cerr << "设置回调失败" << std::endl;
            return false;
        }
#else
        std::cout << "DeepStream未启用，使用简化模式" << std::endl;
#endif
        
        initialized_ = true;
        std::cout << "DeepStreamPipeline初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "DeepStreamPipeline初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void DeepStreamPipeline::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭DeepStreamPipeline..." << std::endl;
    
    stop();
    
#ifdef ENABLE_DEEPSTREAM
    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
#endif
    
    initialized_ = false;
    std::cout << "DeepStreamPipeline已关闭" << std::endl;
}

bool DeepStreamPipeline::start() {
    if (!initialized_) {
        std::cerr << "流水线未初始化" << std::endl;
        return false;
    }
    
    if (running_) {
        std::cout << "流水线已在运行中" << std::endl;
        return true;
    }
    
    std::cout << "启动DeepStream流水线..." << std::endl;
    
#ifdef ENABLE_DEEPSTREAM
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "启动流水线失败" << std::endl;
        return false;
    }
#endif
    
    running_ = true;
    std::cout << "DeepStream流水线已启动" << std::endl;
    return true;
}

void DeepStreamPipeline::stop() {
    if (!running_) {
        return;
    }
    
    std::cout << "停止DeepStream流水线..." << std::endl;
    
#ifdef ENABLE_DEEPSTREAM
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }
#endif
    
    running_ = false;
    std::cout << "DeepStream流水线已停止" << std::endl;
}

void DeepStreamPipeline::pause() {
    if (!running_) {
        return;
    }
    
#ifdef ENABLE_DEEPSTREAM
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_PAUSED);
    }
#endif
    
    std::cout << "DeepStream流水线已暂停" << std::endl;
}

void DeepStreamPipeline::resume() {
    if (!running_) {
        return;
    }
    
#ifdef ENABLE_DEEPSTREAM
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    }
#endif
    
    std::cout << "DeepStream流水线已恢复" << std::endl;
}

bool DeepStreamPipeline::push_frame(const cv::Mat& frame) {
    if (!running_) {
        std::cerr << "流水线未运行" << std::endl;
        return false;
    }
    
    if (frame.empty()) {
        std::cerr << "输入帧为空" << std::endl;
        return false;
    }
    
    try {
#ifdef ENABLE_DEEPSTREAM
        // 创建GStreamer缓冲区
        GstBuffer* buffer = create_gst_buffer(frame);
        if (!buffer) {
            std::cerr << "创建GStreamer缓冲区失败" << std::endl;
            return false;
        }
        
        // 推送到应用源
        GstFlowReturn ret = gst_app_src_push_buffer(appsrc_, buffer);
        if (ret != GST_FLOW_OK) {
            std::cerr << "推送缓冲区失败" << std::endl;
            return false;
        }
#else
        // 简化模式：直接处理帧
        std::cout << "处理帧，尺寸: " << frame.cols << "x" << frame.rows << std::endl;
#endif
        
        // 更新性能统计
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.total_frames_processed++;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "推送帧异常: " << e.what() << std::endl;
        return false;
    }
}

std::vector<core::DetectionResult> DeepStreamPipeline::pull_detections() {
    std::vector<core::DetectionResult> detections;
    
    if (!running_) {
        return detections;
    }
    
    try {
#ifdef ENABLE_DEEPSTREAM
        // 从应用接收器拉取样本
        GstSample* sample = gst_app_sink_try_pull_sample(appsink_, 0);
        if (sample) {
            GstBuffer* buffer = gst_sample_get_buffer(sample);
            if (buffer) {
                detections = parse_detections(buffer);
            }
            gst_sample_unref(sample);
        }
#else
        // 简化模式：创建虚拟检测结果
        core::DetectionResult detection;
        detection.bounding_box = {150, 150, 250, 250};
        detection.confidence = 0.75f;
        detection.class_id = 1;
        detection.label = "deepstream_object";
        detections.push_back(detection);
#endif
        
        // 更新性能统计
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.total_detections += detections.size();
        }
        
        return detections;
        
    } catch (const std::exception& e) {
        std::cerr << "拉取检测结果异常: " << e.what() << std::endl;
        return detections;
    }
}

bool DeepStreamPipeline::push_frames(const std::vector<cv::Mat>& frames) {
    bool success = true;
    for (const auto& frame : frames) {
        if (!push_frame(frame)) {
            success = false;
        }
    }
    return success;
}

std::vector<std::vector<core::DetectionResult>> DeepStreamPipeline::pull_detections_batch() {
    std::vector<std::vector<core::DetectionResult>> batch_results;
    
    // 简化实现：返回单个检测结果
    auto detections = pull_detections();
    if (!detections.empty()) {
        batch_results.push_back(detections);
    }
    
    return batch_results;
}

void DeepStreamPipeline::set_detection_callback(DetectionCallback callback) {
    detection_callback_ = callback;
}

void DeepStreamPipeline::set_error_callback(ErrorCallback callback) {
    error_callback_ = callback;
}

void DeepStreamPipeline::set_status_callback(StatusCallback callback) {
    status_callback_ = callback;
}

void DeepStreamPipeline::set_config(const DeepStreamConfig& config) {
    config_ = config;
}

DeepStreamPipeline::PerformanceStats DeepStreamPipeline::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

#ifdef ENABLE_DEEPSTREAM
bool DeepStreamPipeline::create_pipeline() {
    pipeline_ = gst_pipeline_new("deepstream-pipeline");
    return pipeline_ != nullptr;
}

bool DeepStreamPipeline::create_elements() {
    // 创建应用源和接收器
    source_ = gst_element_factory_make("appsrc", "source");
    sink_ = gst_element_factory_make("appsink", "sink");
    
    if (!source_ || !sink_) {
        std::cerr << "创建应用源或接收器失败" << std::endl;
        return false;
    }
    
    appsrc_ = GST_APP_SRC(source_);
    appsink_ = GST_APP_SINK(sink_);
    
    // 创建DeepStream元素（简化）
    nvinfer_ = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    nvvideoconvert_ = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    
    if (!nvinfer_ || !nvvideoconvert_) {
        std::cerr << "创建DeepStream元素失败" << std::endl;
        return false;
    }
    
    return true;
}

bool DeepStreamPipeline::link_elements() {
    // 将所有元素添加到流水线
    gst_bin_add_many(GST_BIN(pipeline_), source_, nvvideoconvert_, nvinfer_, sink_, nullptr);
    
    // 链接元素
    if (!gst_element_link_many(source_, nvvideoconvert_, nvinfer_, sink_, nullptr)) {
        std::cerr << "链接元素失败" << std::endl;
        return false;
    }
    
    return true;
}

bool DeepStreamPipeline::set_element_properties() {
    // 设置应用源属性
    g_object_set(G_OBJECT(source_),
                "caps", gst_caps_new_simple("video/x-raw",
                                          "format", G_TYPE_STRING, "BGR",
                                          "width", G_TYPE_INT, config_.input_width,
                                          "height", G_TYPE_INT, config_.input_height,
                                          "framerate", GST_TYPE_FRACTION, config_.framerate, 1,
                                          nullptr),
                "format", GST_FORMAT_TIME,
                nullptr);
    
    // 设置推理引擎属性
    g_object_set(G_OBJECT(nvinfer_),
                "config-file-path", config_.model_path.c_str(),
                "batch-size", config_.batch_size,
                nullptr);
    
    return true;
}

bool DeepStreamPipeline::setup_callbacks() {
    // 设置应用接收器回调
    g_signal_connect(sink_, "new-sample", G_CALLBACK(on_new_sample), this);
    
    // 设置总线回调
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    g_signal_connect(G_OBJECT(bus), "message::error", G_CALLBACK(on_error), this);
    g_signal_connect(G_OBJECT(bus), "message::warning", G_CALLBACK(on_warning), this);
    g_signal_connect(G_OBJECT(bus), "message::eos", G_CALLBACK(on_eos), this);
    gst_object_unref(bus);
    
    return true;
}

void DeepStreamPipeline::on_new_sample(GstElement* sink, gpointer user_data) {
    auto* pipeline = static_cast<DeepStreamPipeline*>(user_data);
    
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (sample) {
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (buffer) {
            auto detections = pipeline->parse_detections(buffer);
            if (pipeline->detection_callback_) {
                pipeline->detection_callback_(detections);
            }
        }
        gst_sample_unref(sample);
    }
}

void DeepStreamPipeline::on_error(GstBus* bus, GstMessage* msg, gpointer user_data) {
    auto* pipeline = static_cast<DeepStreamPipeline*>(user_data);
    
    GError* err;
    gchar* debug_info;
    
    gst_message_parse_error(msg, &err, &debug_info);
    std::string error_msg = "DeepStream错误: " + std::string(err->message);
    
    if (pipeline->error_callback_) {
        pipeline->error_callback_(error_msg);
    }
    
    g_clear_error(&err);
    g_free(debug_info);
}

void DeepStreamPipeline::on_warning(GstBus* bus, GstMessage* msg, gpointer user_data) {
    // 处理警告消息
}

void DeepStreamPipeline::on_eos(GstBus* bus, GstMessage* msg, gpointer user_data) {
    // 处理流结束消息
}

std::vector<core::DetectionResult> DeepStreamPipeline::parse_detections(GstBuffer* buffer) {
    std::vector<core::DetectionResult> detections;
    
    // DeepStream元数据解析的简化实现
    core::DetectionResult detection;
    detection.bounding_box = {200, 200, 150, 150};
    detection.confidence = 0.80f;
    detection.class_id = 0;
    detection.label = "detected_object";
    detections.push_back(detection);
    
    return detections;
}

bool DeepStreamPipeline::extract_metadata(GstBuffer* buffer, std::vector<DeepStreamMeta>& metadata) {
    // 元数据提取的简化实现
    return true;
}

core::DetectionResult DeepStreamPipeline::convert_detection(const DeepStreamDetection& ds_detection) {
    core::DetectionResult result;
    result.bounding_box = ds_detection.bounding_box;
    result.confidence = ds_detection.confidence;
    result.class_id = ds_detection.class_id;
    result.label = ds_detection.label;
    return result;
}

GstBuffer* DeepStreamPipeline::create_gst_buffer(const cv::Mat& frame) {
    // GStreamer缓冲区创建的简化实现
    size_t size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        memcpy(map.data, frame.data, size);
        gst_buffer_unmap(buffer, &map);
    }
    
    return buffer;
}

void DeepStreamPipeline::release_gst_buffer(GstBuffer* buffer) {
    if (buffer) {
        gst_buffer_unref(buffer);
    }
}
#endif

void DeepStreamPipeline::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "DeepStreamPipeline错误: " << error << std::endl;
    
    if (error_callback_) {
        error_callback_(error);
    }
}

// DeepStreamMultiStream implementation
bool DeepStreamMultiStream::MultiStreamConfig::validate() const {
    if (source_devices.empty()) {
        return false;
    }
    if (num_streams <= 0) {
        return false;
    }
    if (!base_config.validate()) {
        return false;
    }
    return true;
}

DeepStreamMultiStream::DeepStreamMultiStream() : config_(), initialized_(false), running_(false) {
    std::cout << "创建DeepStreamMultiStream实例" << std::endl;
}

DeepStreamMultiStream::DeepStreamMultiStream(const MultiStreamConfig& config) 
    : config_(config), initialized_(false), running_(false) {
    std::cout << "创建DeepStreamMultiStream实例，使用自定义配置" << std::endl;
}

DeepStreamMultiStream::~DeepStreamMultiStream() {
    shutdown();
    std::cout << "销毁DeepStreamMultiStream实例" << std::endl;
}

bool DeepStreamMultiStream::initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        if (!config_.validate()) {
            return false;
        }
        
        if (!create_streams()) {
            return false;
        }
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "DeepStreamMultiStream初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void DeepStreamMultiStream::shutdown() {
    if (!initialized_) {
        return;
    }
    
    stop_all_streams();
    pipelines_.clear();
    
    initialized_ = false;
}

bool DeepStreamMultiStream::start_all_streams() {
    if (!initialized_) {
        return false;
    }
    
    bool success = true;
    for (auto& pipeline : pipelines_) {
        if (!pipeline->start()) {
            success = false;
        }
    }
    
    running_ = success;
    return success;
}

void DeepStreamMultiStream::stop_all_streams() {
    for (auto& pipeline : pipelines_) {
        if (pipeline) {
            pipeline->stop();
        }
    }
    running_ = false;
}

bool DeepStreamMultiStream::push_frames(const std::vector<cv::Mat>& frames) {
    if (frames.size() != pipelines_.size()) {
        return false;
    }
    
    bool success = true;
    for (size_t i = 0; i < frames.size() && i < pipelines_.size(); ++i) {
        if (!pipelines_[i]->push_frame(frames[i])) {
            success = false;
        }
    }
    
    return success;
}

std::vector<std::vector<core::DetectionResult>> DeepStreamMultiStream::pull_detections() {
    std::vector<std::vector<core::DetectionResult>> all_detections;
    
    for (auto& pipeline : pipelines_) {
        if (pipeline) {
            auto detections = pipeline->pull_detections();
            all_detections.push_back(detections);
        }
    }
    
    return all_detections;
}

void DeepStreamMultiStream::set_config(const MultiStreamConfig& config) {
    config_ = config;
}

DeepStreamMultiStream::MultiStreamStats DeepStreamMultiStream::get_performance_stats() const {
    MultiStreamStats stats;
    
    for (const auto& pipeline : pipelines_) {
        if (pipeline) {
            stats.stream_stats.push_back(pipeline->get_performance_stats());
        }
    }
    
    // 计算总体统计
    if (!stats.stream_stats.empty()) {
        double total_fps = 0.0;
        for (const auto& stream_stat : stats.stream_stats) {
            total_fps += stream_stat.fps;
        }
        stats.avg_fps = total_fps / stats.stream_stats.size();
    }
    
    return stats;
}

bool DeepStreamMultiStream::create_streams() {
    pipelines_.clear();
    
    for (int i = 0; i < config_.num_streams; ++i) {
        auto pipeline = std::make_unique<DeepStreamPipeline>(config_.base_config);
        if (!pipeline->initialize()) {
            return false;
        }
        pipelines_.push_back(std::move(pipeline));
    }
    
    return true;
}

bool DeepStreamMultiStream::synchronize_streams() {
    // 流同步的简化实现
    return true;
}

void DeepStreamMultiStream::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "DeepStreamMultiStream错误: " << error << std::endl;
}

} // namespace vision
} // namespace bamboo_cut