/**
 * GStreamer视频流接收器
 * 替换共享内存方案，通过GStreamer接收后端发送的视频流
 */

#ifndef CAMERA_GSTREAMER_RECEIVER_H
#define CAMERA_GSTREAMER_RECEIVER_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GStreamer接收器结构 */
typedef struct {
    GstElement* pipeline;           // GStreamer管道
    GstElement* appsink;           // 应用接收器
    
    // 配置参数
    char stream_url[256];          // 流URL
    char stream_format[32];        // 流格式 (H264/JPEG)
    int width;                     // 图像宽度
    int height;                    // 图像高度
    int fps;                       // 帧率
    
    // 状态
    bool connected;                // 连接状态
    bool receiving;                // 接收状态
    
    // 统计信息
    uint64_t frames_received;      // 接收帧数
    uint64_t frames_dropped;       // 丢弃帧数
    double avg_fps;                // 平均帧率
    uint64_t last_frame_time;      // 上次帧时间
} gstreamer_receiver_t;

/* 统计信息结构 */
typedef struct {
    uint64_t frames_received;      // 总接收帧数
    uint64_t frames_dropped;       // 总丢帧数
    double current_fps;            // 当前帧率
    double avg_fps;                // 平均帧率
    bool is_connected;             // 连接状态
    uint64_t connection_time;      // 连接时间
    uint64_t last_frame_time;      // 最后帧时间
} gstreamer_stats_t;

/**
 * 创建GStreamer接收器
 * @param stream_url 流URL，如"udp://127.0.0.1:5000"或"http://127.0.0.1:5000/"
 * @param stream_format 流格式，"H264"或"JPEG"
 * @param width 期望的图像宽度
 * @param height 期望的图像高度
 * @param fps 期望的帧率
 * @return GStreamer接收器指针，失败返回NULL
 */
gstreamer_receiver_t* gstreamer_receiver_create(const char* stream_url, const char* stream_format, 
                                               int width, int height, int fps);

/**
 * 销毁GStreamer接收器
 * @param receiver GStreamer接收器指针
 */
void gstreamer_receiver_destroy(gstreamer_receiver_t* receiver);

/**
 * 连接到视频流
 * @param receiver GStreamer接收器
 * @return 成功返回true，失败返回false
 */
bool gstreamer_receiver_connect(gstreamer_receiver_t* receiver);

/**
 * 断开视频流连接
 * @param receiver GStreamer接收器
 */
void gstreamer_receiver_disconnect(gstreamer_receiver_t* receiver);

/**
 * 开始接收视频流
 * @param receiver GStreamer接收器
 * @return 成功返回true，失败返回false
 */
bool gstreamer_receiver_start(gstreamer_receiver_t* receiver);

/**
 * 停止接收视频流
 * @param receiver GStreamer接收器
 */
void gstreamer_receiver_stop(gstreamer_receiver_t* receiver);

/**
 * 读取一帧图像
 * @param receiver GStreamer接收器
 * @param frame 输出的OpenCV Mat对象
 * @param timeout_ms 超时时间（毫秒）
 * @return 成功读取返回true，失败返回false
 */
bool gstreamer_receiver_read_frame(gstreamer_receiver_t* receiver, cv::Mat& frame, int timeout_ms);

/**
 * 检查连接状态
 * @param receiver GStreamer接收器
 * @return 已连接返回true，未连接返回false
 */
bool gstreamer_receiver_is_connected(gstreamer_receiver_t* receiver);

/**
 * 检查接收状态
 * @param receiver GStreamer接收器
 * @return 正在接收返回true，否则返回false
 */
bool gstreamer_receiver_is_receiving(gstreamer_receiver_t* receiver);

/**
 * 获取统计信息
 * @param receiver GStreamer接收器
 * @param stats 输出统计信息
 */
void gstreamer_receiver_get_stats(gstreamer_receiver_t* receiver, gstreamer_stats_t* stats);

/**
 * 重置统计信息
 * @param receiver GStreamer接收器
 */
void gstreamer_receiver_reset_stats(gstreamer_receiver_t* receiver);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_GSTREAMER_RECEIVER_H