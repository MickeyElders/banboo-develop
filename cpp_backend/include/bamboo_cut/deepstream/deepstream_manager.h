/**
 * @file deepstream_manager.h
 * @brief DeepStream AI推理和视频显示管理器
 * 实现动态布局计算、AI推理和硬件加速显示
 */

#pragma once

#include <gst/gst.h>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace bamboo_cut {
namespace deepstream {

/**
 * @brief DeepStream配置参数
 */
struct DeepStreamConfig {
    int screen_width;           // 屏幕宽度
    int screen_height;          // 屏幕高度
    int header_height;          // 顶部栏高度
    int footer_height;          // 底部栏高度
    float video_width_ratio;    // 视频区域宽度比例 (0.0-1.0)
    float video_height_ratio;   // 视频区域高度比例 (0.0-1.0)
    int camera_id;              // 摄像头ID
    std::string nvinfer_config; // nvinfer配置文件路径
    
    DeepStreamConfig()
        : screen_width(1280)
        , screen_height(800)
        , header_height(80)
        , footer_height(80)
        , video_width_ratio(0.75f)
        , video_height_ratio(1.0f)
        , camera_id(0)
        , nvinfer_config("/opt/bamboo-cut/config/nvinfer_config.txt") {}
};

/**
 * @brief 视频显示区域信息
 */
struct VideoLayout {
    int offset_x;       // X偏移量
    int offset_y;       // Y偏移量
    int width;          // 视频宽度
    int height;         // 视频高度
    int available_width; // 可用区域宽度
    int available_height; // 可用区域高度
};

/**
 * @brief DeepStream管理器类
 * 负责AI推理、视频显示和布局计算
 */
class DeepStreamManager {
public:
    DeepStreamManager();
    ~DeepStreamManager();

    /**
     * @brief 初始化DeepStream系统
     */
    bool initialize(const DeepStreamConfig& config);

    /**
     * @brief 启动视频流和AI推理
     */
    bool start();

    /**
     * @brief 停止视频流和AI推理
     */
    void stop();

    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return running_; }

    /**
     * @brief 获取视频布局信息
     */
    VideoLayout getVideoLayout() const { return video_layout_; }

    /**
     * @brief 动态更新布局（屏幕尺寸变化时）
     */
    bool updateLayout(int screen_width, int screen_height);

private:
    /**
     * @brief 计算视频显示区域布局
     */
    VideoLayout calculateVideoLayout(const DeepStreamConfig& config);

    /**
     * @brief 构建GStreamer管道
     */
    std::string buildPipeline(const DeepStreamConfig& config, const VideoLayout& layout);

    /**
     * @brief 初始化DeepStream
     */
    bool initializeDeepStream();

    /**
     * @brief GStreamer消息处理回调
     */
    static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer data);

    /**
     * @brief 清理资源
     */
    void cleanup();

private:
    DeepStreamConfig config_;
    VideoLayout video_layout_;
    
    GstElement* pipeline_;
    GstBus* bus_;
    guint bus_watch_id_;
    
    bool running_;
    bool initialized_;
};

} // namespace deepstream
} // namespace bamboo_cut