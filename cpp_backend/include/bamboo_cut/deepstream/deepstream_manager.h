/**
 * @file deepstream_manager.h
 * @brief DeepStream AI推理和视频显示管理器
 * 实现动态布局计算、AI推理和硬件加速显示，支持双摄像头
 */

#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>

// DRM资源协调器已移除，使用Wayland架构
// #include "bamboo_cut/drm/drm_resource_coordinator.h"  // 已删除

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
 * @brief 视频输出模式
 */
enum class VideoSinkMode {
    NVDRMVIDEOSINK,     ///< nvdrmvideosink DRM叠加平面模式
    WAYLANDSINK,        ///< waylandsink模式
    KMSSINK,            ///< kmssink KMS多层渲染模式
    APPSINK             ///< appsink 软件合成到LVGL画布模式 (推荐)
};

/**
 * @brief 摄像头源模式
 */
enum class CameraSourceMode {
    NVARGUSCAMERA,      ///< v4l2src (V4L2摄像头，绕过Argus) 或 nvarguscamerasrc
    VIDEOTESTSRC,       ///< videotestsrc (虚拟测试源)
    FILESRC,            ///< filesrc (文件源)
    V4L2SRC             ///< v4l2src (直接V4L2接口，避免Argus超时)
};

/**
 * @brief DRM叠加平面配置
 */
struct DRMOverlayConfig {
    int plane_id;           // DRM叠加平面ID
    int connector_id;       // 连接器ID
    int crtc_id;           // CRTC ID
    int z_order;           // Z轴排序 (数值越大越靠前)
    bool enable_scaling;   // 是否启用硬件缩放
    
    DRMOverlayConfig()
        : plane_id(-1)      // -1表示自动检测可用平面
        , connector_id(-1)  // -1表示自动检测
        , crtc_id(-1)       // -1表示自动检测
        , z_order(1)        // 默认在LVGL层之上
        , enable_scaling(true) {}
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
    
    // 视频输出配置
    VideoSinkMode sink_mode;    // 视频输出模式
    DRMOverlayConfig overlay;   // DRM叠加平面配置
    
    // 摄像头源配置
    CameraSourceMode camera_source;  // 摄像头源模式
    int test_pattern;                // 测试图案模式（videotestsrc使用）
    std::string video_file_path;     // 视频文件路径（filesrc使用）
    
    // APPSINK软件合成配置
    bool enable_software_composition;   // 启用软件合成到LVGL画布
    int composition_fps;               // 合成帧率
    std::string target_pixel_format;   // 目标像素格式 (ARGB32, RGB565等)
    
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
        , camera_fps(30)
        , sink_mode(VideoSinkMode::APPSINK)  // 默认使用appsink软件合成模式（稳定可靠）
        , overlay()  // 使用默认叠加平面配置
        , camera_source(CameraSourceMode::NVARGUSCAMERA)  // 默认使用真实摄像头
        , test_pattern(0)  // 默认测试图案
        , video_file_path("/opt/bamboo-cut/test_video.mp4")  // 默认测试视频文件
        , enable_software_composition(true)  // 启用软件合成
        , composition_fps(30)  // 30fps合成
        , target_pixel_format("BGRA") {}  // 使用更兼容的BGRA格式
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
    
    /**
     * @brief 构造函数（支持LVGL界面集成）
     * @param lvgl_interface LVGL界面实例指针，用于appsink软件合成
     */
    DeepStreamManager(void* lvgl_interface);
    
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
     * @param sink_mode 视频输出模式
     * @return 是否成功
     */
    bool switchSinkMode(VideoSinkMode sink_mode);

    /**
     * @brief 获取当前视频sink模式
     */
    VideoSinkMode getCurrentSinkMode() const { return config_.sink_mode; }

    /**
     * @brief 配置DRM叠加平面
     * @param overlay_config 叠加平面配置
     * @return 是否成功
     */
    bool configureDRMOverlay(const DRMOverlayConfig& overlay_config);

    /**
     * @brief 自动检测可用的DRM叠加平面
     * @return 检测到的叠加平面配置，失败时返回空配置
     */
    DRMOverlayConfig detectAvailableOverlayPlane();
    
    /**
     * @brief 验证多层显示设置
     * @return 是否验证通过
     */
    bool verifyMultiLayerDisplaySetup();
    
    /**
     * @brief 检查Wayland环境
     * @return 是否环境正常
     */
    bool checkWaylandEnvironment();
    
    /**
     * @brief 计算Wayland视频布局
     * @param config DeepStream配置
     * @return 计算出的视频布局
     */
    VideoLayout calculateWaylandVideoLayout(const DeepStreamConfig& config);
    
    /**
     * @brief 启动Canvas更新线程
     */
    void startCanvasUpdateThread();
    
    /**
     * @brief 停止Canvas更新线程
     */
    void stopCanvasUpdateThread();

private:
    /**
     * @brief 设置DRM叠加平面
     * @return 是否成功
     */
    bool setupDRMOverlayPlane();
    
    /**
     * @brief 检查NVMM缓冲区可用性
     * @return 是否有足够的NVMM缓冲区可用
     */
    bool checkNVMMBufferAvailability();

    /**
     * @brief 构建nvdrmvideosink管道
     */
    std::string buildNVDRMVideoSinkPipeline(const DeepStreamConfig& config, int offset_x, int offset_y, int width, int height);

    /**
     * @brief 构建waylandsink管道
     */
    std::string buildWaylandSinkPipeline(const DeepStreamConfig& config, int offset_x, int offset_y, int width, int height);
    
    /**
     * @brief 构建kmssink管道
     */
    std::string buildKMSSinkPipeline(const DeepStreamConfig& config, int offset_x, int offset_y, int width, int height);
    
    
    /**
     * @brief 构建摄像头源字符串
     */
    std::string buildCameraSource(const DeepStreamConfig& config);
    
    /**
     * @brief 构建appsink软件合成管道
     */
    std::string buildAppSinkPipeline(const DeepStreamConfig& config, int offset_x, int offset_y, int width, int height);
    
    /**
     * @brief appsink新样本回调
     */
    static GstFlowReturn newSampleCallback(GstAppSink* appsink, gpointer user_data);
    
    /**
     * @brief 软件合成帧到LVGL画布
     */
    void compositeFrameToLVGL(GstMapInfo* map_info, int width, int height);
    
    /**
     * @brief 设置AppSink回调函数
     */
    void setupAppSinkCallbacks();
    
    /**
     * @brief 获取最新合成帧
     * @param frame 输出的帧数据
     * @return 是否有新帧可用
     */
    bool getLatestCompositeFrame(cv::Mat& frame);
    
    /**
     * @brief Canvas更新线程主循环
     */
    void canvasUpdateLoop();
    
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

    /**
     * @brief 配置JetPack 6的NVMM环境
     */
    void configureNVMMEnvironment();

private:
    DeepStreamConfig config_;
    VideoLayout video_layout_;
    
    GstElement* pipeline_;      // 主管道
    GstElement* pipeline2_;     // 副管道（双摄模式）
    GstBus* bus_;              // 主消息总线
    GstBus* bus2_;             // 副消息总线
    guint bus_watch_id_;       // 主总线监听ID
    guint bus_watch_id2_;      // 副总线监听ID
    
    // APPSINK软件合成相关
    GstElement* appsink_;       // appsink元素
    std::mutex frame_mutex_;    // 帧数据同步互斥锁
    cv::Mat latest_frame_;      // 最新帧数据
    std::atomic<bool> new_frame_available_{false};  // 新帧可用标志
    void* lvgl_interface_;      // LVGL界面实例指针
    std::thread canvas_update_thread_;          // Canvas更新线程
    std::atomic<bool> canvas_update_running_;   // Canvas更新线程运行标志
    
    // 🔧 线程安全保护
    mutable std::mutex drm_mutex_;      // DRM资源访问互斥锁
    mutable std::mutex pipeline_mutex_; // GStreamer管道操作互斥锁
    
    // Wayland环境配置
    bool wayland_available_;
    
    bool running_;
    bool initialized_;
};

} // namespace deepstream
} // namespace bamboo_cut