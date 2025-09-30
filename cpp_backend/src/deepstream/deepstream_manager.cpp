/**
 * @file deepstream_manager.cpp
 * @brief DeepStream AI推理和视频显示管理器实现
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>

namespace bamboo_cut {
namespace deepstream {

DeepStreamManager::DeepStreamManager()
    : pipeline_(nullptr)
    , pipeline2_(nullptr)
    , bus_(nullptr)
    , bus2_(nullptr)
    , bus_watch_id_(0)
    , bus_watch_id2_(0)
    , running_(false)
    , initialized_(false) {
}

DeepStreamManager::~DeepStreamManager() {
    stop();
    cleanup();
}

bool DeepStreamManager::initialize(const DeepStreamConfig& config) {
    std::cout << "初始化 DeepStream 系统..." << std::endl;
    
    config_ = config;
    
    // 初始化 GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        std::cout << "GStreamer 初始化完成" << std::endl;
    }
    
    // 计算视频布局
    video_layout_ = calculateVideoLayout(config);
    
    std::cout << "视频布局计算完成:" << std::endl;
    std::cout << "  偏移: (" << video_layout_.offset_x << ", " << video_layout_.offset_y << ")" << std::endl;
    std::cout << "  尺寸: " << video_layout_.width << "x" << video_layout_.height << std::endl;
    std::cout << "  可用区域: " << video_layout_.available_width << "x" << video_layout_.available_height << std::endl;
    
    initialized_ = true;
    return true;
}

bool DeepStreamManager::start() {
    if (!initialized_) {
        std::cerr << "DeepStream 未初始化" << std::endl;
        return false;
    }
    
    if (running_) {
        std::cout << "DeepStream 已在运行" << std::endl;
        return true;
    }
    
    std::cout << "启动 DeepStream 管道..." << std::endl;
    std::cout << "双摄模式: " << static_cast<int>(config_.dual_mode) << std::endl;
    
    if (config_.dual_mode == DualCameraMode::SPLIT_SCREEN) {
        // 并排显示模式：创建两个独立管道
        return startSplitScreenMode();
    } else {
        // 单摄像头或立体视觉模式：单管道
        return startSinglePipelineMode();
    }
}

bool DeepStreamManager::startSinglePipelineMode() {
    // 构建管道
    std::string pipeline_str = buildPipeline(config_, video_layout_);
    std::cout << "管道字符串: " << pipeline_str << std::endl;
    
    // 创建管道
    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (!pipeline_ || error) {
        std::cerr << "创建管道失败: " << (error ? error->message : "未知错误") << std::endl;
        if (error) g_error_free(error);
        return false;
    }
    
    // 设置消息总线
    bus_ = gst_element_get_bus(pipeline_);
    bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
    
    // 启动管道
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "启动管道失败" << std::endl;
        cleanup();
        return false;
    }
    
    running_ = true;
    std::cout << "DeepStream 管道启动成功" << std::endl;
    return true;
}

bool DeepStreamManager::startSplitScreenMode() {
    // 构建左侧摄像头管道
    std::string pipeline1_str = buildSplitScreenPipeline(config_, video_layout_);
    std::cout << "左侧管道: " << pipeline1_str << std::endl;
    
    // 构建右侧摄像头管道
    DeepStreamConfig config2 = config_;
    config2.camera_id = config_.camera_id_2;  // 使用副摄像头
    std::string pipeline2_str = buildSplitScreenPipeline(config2, video_layout_);
    
    // 修改右侧管道的偏移
    size_t offset_pos = pipeline2_str.find("offset-x=");
    if (offset_pos != std::string::npos) {
        size_t end_pos = pipeline2_str.find(" ", offset_pos);
        int right_offset = video_layout_.width / 2 + 10;  // 右半边偏移
        pipeline2_str.replace(offset_pos, end_pos - offset_pos, 
                             "offset-x=" + std::to_string(right_offset));
    }
    // 修改plane-id
    size_t plane_pos = pipeline2_str.find("plane-id=0");
    if (plane_pos != std::string::npos) {
        pipeline2_str.replace(plane_pos, 10, "plane-id=1");
    }
    
    std::cout << "右侧管道: " << pipeline2_str << std::endl;
    
    // 创建两个管道
    GError *error1 = nullptr, *error2 = nullptr;
    pipeline_ = gst_parse_launch(pipeline1_str.c_str(), &error1);
    pipeline2_ = gst_parse_launch(pipeline2_str.c_str(), &error2);
    
    if (!pipeline_ || error1) {
        std::cerr << "创建左侧管道失败: " << (error1 ? error1->message : "未知错误") << std::endl;
        if (error1) g_error_free(error1);
        return false;
    }
    
    if (!pipeline2_ || error2) {
        std::cerr << "创建右侧管道失败: " << (error2 ? error2->message : "未知错误") << std::endl;
        if (error2) g_error_free(error2);
        return false;
    }
    
    // 设置消息总线
    bus_ = gst_element_get_bus(pipeline_);
    bus2_ = gst_element_get_bus(pipeline2_);
    bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
    bus_watch_id2_ = gst_bus_add_watch(bus2_, busCallback, this);
    
    // 启动两个管道
    GstStateChangeReturn ret1 = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    GstStateChangeReturn ret2 = gst_element_set_state(pipeline2_, GST_STATE_PLAYING);
    
    if (ret1 == GST_STATE_CHANGE_FAILURE || ret2 == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "启动管道失败" << std::endl;
        cleanup();
        return false;
    }
    
    running_ = true;
    std::cout << "双摄像头并排显示管道启动成功" << std::endl;
    return true;
}

void DeepStreamManager::stop() {
    if (!running_) return;
    
    std::cout << "停止 DeepStream 管道..." << std::endl;
    
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }
    
    if (pipeline2_) {
        gst_element_set_state(pipeline2_, GST_STATE_NULL);
    }
    
    running_ = false;
    std::cout << "DeepStream 管道已停止" << std::endl;
}

bool DeepStreamManager::switchDualMode(DualCameraMode mode) {
    if (config_.dual_mode == mode) {
        std::cout << "已是当前模式，无需切换" << std::endl;
        return true;
    }
    
    std::cout << "切换双摄模式: " << static_cast<int>(config_.dual_mode) 
              << " -> " << static_cast<int>(mode) << std::endl;
    
    // 停止当前管道
    bool was_running = running_;
    if (running_) {
        stop();
        cleanup();
    }
    
    // 更新配置
    config_.dual_mode = mode;
    
    // 重新计算布局
    video_layout_ = calculateVideoLayout(config_);
    
    // 如果之前在运行，重新启动
    if (was_running) {
        return start();
    }
    
    return true;
}

bool DeepStreamManager::updateLayout(int screen_width, int screen_height) {
    config_.screen_width = screen_width;
    config_.screen_height = screen_height;
    
    // 重新计算布局
    video_layout_ = calculateVideoLayout(config_);
    
    std::cout << "布局已更新: " << video_layout_.width << "x" << video_layout_.height 
              << " at (" << video_layout_.offset_x << ", " << video_layout_.offset_y << ")" << std::endl;
    
    // 如果正在运行，需要重启管道以应用新布局
    if (running_) {
        stop();
        return start();
    }
    
    return true;
}

VideoLayout DeepStreamManager::calculateVideoLayout(const DeepStreamConfig& config) {
    VideoLayout layout;
    
    // 计算可用区域（减去顶部和底部栏）
    layout.available_width = config.screen_width;
    layout.available_height = config.screen_height - config.header_height - config.footer_height;
    
    // 计算视频区域尺寸（按比例）
    layout.width = static_cast<int>(layout.available_width * config.video_width_ratio);
    layout.height = static_cast<int>(layout.available_height * config.video_height_ratio);
    
    // 计算偏移量（居左上，但考虑顶部栏）
    layout.offset_x = 0;  // 左对齐
    layout.offset_y = config.header_height;  // 顶部栏下方
    
    return layout;
}

std::string DeepStreamManager::buildPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    switch (config.dual_mode) {
        case DualCameraMode::SINGLE_CAMERA:
            // 单摄像头使用简化的并排显示管道
            return buildSplitScreenPipeline(config, layout);
        case DualCameraMode::SPLIT_SCREEN:
            return buildSplitScreenPipeline(config, layout);
        case DualCameraMode::STEREO_VISION:
            return buildStereoVisionPipeline(config, layout);
        default:
            return buildSplitScreenPipeline(config, layout);
    }
}

std::string DeepStreamManager::buildSplitScreenPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    std::ostringstream pipeline;
    
    // 并排显示 - 使用nvoverlaysink避免DRM冲突
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=30/1,format=NV12 ! "
             << "nvvideoconvert ! "
             << "nvoverlaysink "
             << "overlay-x=" << layout.offset_x << " "
             << "overlay-y=" << layout.offset_y << " "
             << "overlay-w=" << layout.width << " "
             << "overlay-h=" << layout.height;
    
    return pipeline.str();
}

std::string DeepStreamManager::buildStereoVisionPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    std::ostringstream pipeline;
    
    // 双摄立体视觉 - 使用 nvstreammux 合并两路流，nvoverlaysink避免DRM冲突
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=30/1,format=NV12 ! "
             << "m.sink_0 "  // 连接到 mux 的第一个输入
             
             << "nvarguscamerasrc sensor-id=" << config.camera_id_2 << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=30/1,format=NV12 ! "
             << "m.sink_1 "  // 连接到 mux 的第二个输入
             
             << "nvstreammux name=m batch-size=1 width=" << config.camera_width
             << " height=" << config.camera_height << " ! "
             << "nvvideoconvert ! "
             << "nvoverlaysink "
             << "overlay-x=" << layout.offset_x << " "
             << "overlay-y=" << layout.offset_y << " "
             << "overlay-w=" << layout.width << " "
             << "overlay-h=" << layout.height;
    
    return pipeline.str();
}

gboolean DeepStreamManager::busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
    DeepStreamManager* manager = static_cast<DeepStreamManager*>(data);
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "DeepStream 错误: " << err->message << std::endl;
            if (debug) {
                std::cerr << "调试信息: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(err);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError *err;
            gchar *debug;
            gst_message_parse_warning(msg, &err, &debug);
            std::cout << "DeepStream 警告: " << err->message << std::endl;
            if (debug) {
                std::cout << "调试信息: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(err);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cout << "DeepStream 流结束" << std::endl;
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->pipeline_)) {
                GstState old_state, new_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, nullptr);
                std::cout << "DeepStream 状态变更: " 
                          << gst_element_state_get_name(old_state) << " -> " 
                          << gst_element_state_get_name(new_state) << std::endl;
            }
            break;
        }
        default:
            break;
    }
    
    return TRUE;
}

void DeepStreamManager::cleanup() {
    if (bus_watch_id_ > 0) {
        g_source_remove(bus_watch_id_);
        bus_watch_id_ = 0;
    }
    
    if (bus_watch_id2_ > 0) {
        g_source_remove(bus_watch_id2_);
        bus_watch_id2_ = 0;
    }
    
    if (bus_) {
        gst_object_unref(bus_);
        bus_ = nullptr;
    }
    
    if (bus2_) {
        gst_object_unref(bus2_);
        bus2_ = nullptr;
    }
    
    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    if (pipeline2_) {
        gst_object_unref(pipeline2_);
        pipeline2_ = nullptr;
    }
}

} // namespace deepstream
} // namespace bamboo_cut