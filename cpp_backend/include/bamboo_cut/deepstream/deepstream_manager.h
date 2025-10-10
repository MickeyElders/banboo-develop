/**
 * @file deepstream_manager.h
 * @brief DeepStream AI推理和视频显示管理器
 * 实现动态布局计算、AI推理和硬件加速显示，支持双摄像头
 */

#pragma once

#include <gst/gst.h>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace bamboo_cut {
namespace deepstream {

/**
 * @brief 双摄显示模式
 */
enum class DualCameraMode {
    SINGLE_CAMERA,      ///< 单摄像头
    SPLIT_SCREEN,       ///< 并排显示
    STEREO_VISION       ///< 立体视觉合成
};

/**
 * @brief DeepStream配置参数
 */
struct DeepStreamConfig {
    // 基础配置
    int screen_width;           // 屏幕宽度
    int screen_height;          // 屏幕高度
    int header_height;          // 顶部栏高度
    int footer_height;          // 底部栏高度
    float video_width_ratio;    // 视频区域宽度比例 (0.0-1.0)
    float video_height_ratio;   // 视频区域高度比例 (0.0-1.0)
    int camera_id;              // 主摄像头ID
    int camera_id_2;            // 副摄像头ID（双摄模式）
    std::string nvinfer_config; // nvinfer配置文件路径
    
    // 双摄配置
    DualCameraMode dual_mode;   // 双摄模式
    int camera_width;           // 摄像头分辨率宽度
    int camera_height;          // 摄像头分辨率高度
    int camera_fps;             // 摄像头帧率
    
    DeepStreamConfig()
        : screen_width(1280)
        , screen_height(800)
        , header_height(80)
        , footer_height(80)
        , video_width_ratio(0.75f)
        , video_height_ratio(1.0f)
        , camera_id(0)
        , camera_id_2(1)
        , nvinfer_config("/opt/bamboo-cut/config/nvinfer_config.txt")
        , dual_mode(DualCameraMode::SINGLE_CAMERA)
        , camera_width(1280)
        , camera_height(720)
        , camera_fps(30) {}  // 确保30fps提高稳定性
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
 * 负责AI推理、视频显示和布局计算，支持双摄像头
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
     * @brief 切换双摄显示模式
     * @param mode 新的显示模式
     * @return 是否成功
     */
    bool switchDualMode(DualCameraMode mode);

    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return running_; }

    /**
     * @brief 获取视频布局信息
     */
    VideoLayout getVideoLayout() const { return video_layout_; }

    /**
     * @brief 获取当前双摄模式
     */
    DualCameraMode getCurrentMode() const { return config_.dual_mode; }

    /**
     * @brief 动态更新布局（屏幕尺寸变化时）
     */
    bool updateLayout(int screen_width, int screen_height);

    /**
     * @brief 切换视频sink模式
     * @param use_wayland 是否使用waylandsink
     * @return 是否成功
     */
    bool switchSinkMode(bool use_wayland);

    /**
     * @brief 检查当前是否使用Wayland sink
     */
    bool isUsingWaylandSink() const { return use_wayland_sink_; }

private:
    /**
     * @brief 计算视频显示区域布局
     */
    VideoLayout calculateVideoLayout(const DeepStreamConfig& config);

    /**
     * @brief 构建并排显示管道
     */
    std::string buildSplitScreenPipeline(const DeepStreamConfig& config, int offset_x, int offset_y, int width, int height);

    /**
     * @brief 构建立体视觉管道
     */
    std::string buildStereoVisionPipeline(const DeepStreamConfig& config, const VideoLayout& layout);

    /**
     * @brief 构建GStreamer管道
     */
    std::string buildPipeline(const DeepStreamConfig& config, const VideoLayout& layout);

    /**
     * @brief 启动单管道模式
     */
    bool startSinglePipelineMode();

    /**
     * @brief 启动并排显示模式
     */
    bool startSplitScreenMode();

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
    
    GstElement* pipeline_;      // 主管道
    GstElement* pipeline2_;     // 副管道（双摄模式）
    GstBus* bus_;              // 主消息总线
    GstBus* bus2_;             // 副消息总线
    guint bus_watch_id_;       // 主总线监听ID
    guint bus_watch_id2_;      // 副总线监听ID
    
    bool running_;
    bool initialized_;
};

} // namespace deepstream
} // namespace bamboo_cut