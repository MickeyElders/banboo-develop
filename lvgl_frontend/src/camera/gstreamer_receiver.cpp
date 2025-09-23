#include "camera/gstreamer_receiver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

// 获取当前时间戳（毫秒）
static uint64_t get_timestamp_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// GStreamer总线消息回调
static gboolean bus_message_handler(GstBus* bus, GstMessage* message, gpointer user_data) {
    gstreamer_receiver_t* receiver = (gstreamer_receiver_t*)user_data;
    
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError* err;
            gchar* debug_info;
            gst_message_parse_error(message, &err, &debug_info);
            printf("GStreamer错误: %s\n", err->message);
            if (debug_info) {
                printf("调试信息: %s\n", debug_info);
                g_free(debug_info);
            }
            g_error_free(err);
            receiver->connected = false;
            break;
        }
        case GST_MESSAGE_EOS:
            printf("GStreamer流结束\n");
            receiver->receiving = false;
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed(message, &old_state, &new_state, &pending_state);
            if (GST_MESSAGE_SRC(message) == GST_OBJECT(receiver->pipeline)) {
                printf("GStreamer状态变化: %s -> %s\n", 
                       gst_element_state_get_name(old_state),
                       gst_element_state_get_name(new_state));
                if (new_state == GST_STATE_PLAYING) {
                    receiver->receiving = true;
                }
            }
            break;
        }
        default:
            break;
    }
    
    return TRUE;
}

// 构建接收管道
static std::string build_receive_pipeline(const char* stream_url, const char* stream_format) {
    std::string pipeline;
    
    if (strcmp(stream_format, "H264") == 0) {
        // H.264 UDP流接收
        pipeline = "udpsrc uri=" + std::string(stream_url) + 
                  " ! application/x-rtp,encoding-name=H264,payload=96"
                  " ! rtph264depay"
                  " ! avdec_h264"
                  " ! videoconvert"
                  " ! video/x-raw,format=BGR"
                  " ! appsink name=appsink max-buffers=2 drop=true sync=false";
    } else if (strcmp(stream_format, "JPEG") == 0) {
        // MJPEG HTTP流接收
        pipeline = "souphttpsrc location=" + std::string(stream_url) +
                  " ! multipartdemux"
                  " ! jpegdec"
                  " ! videoconvert"
                  " ! video/x-raw,format=BGR"
                  " ! appsink name=appsink max-buffers=2 drop=true sync=false";
    } else {
        printf("错误：不支持的流格式: %s\n", stream_format);
        return "";
    }
    
    return pipeline;
}

gstreamer_receiver_t* gstreamer_receiver_create(const char* stream_url, const char* stream_format,
                                               int width, int height, int fps) {
    if (!stream_url || !stream_format || width <= 0 || height <= 0 || fps <= 0) {
        printf("错误：GStreamer接收器参数无效\n");
        return nullptr;
    }
    
    // 初始化GStreamer（如果尚未初始化）
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        printf("GStreamer已初始化\n");
    }
    
    gstreamer_receiver_t* receiver = (gstreamer_receiver_t*)calloc(1, sizeof(gstreamer_receiver_t));
    if (!receiver) {
        printf("错误：分配GStreamer接收器内存失败\n");
        return nullptr;
    }
    
    // 初始化基本参数
    strncpy(receiver->stream_url, stream_url, sizeof(receiver->stream_url) - 1);
    strncpy(receiver->stream_format, stream_format, sizeof(receiver->stream_format) - 1);
    receiver->width = width;
    receiver->height = height;
    receiver->fps = fps;
    
    // 初始化状态
    receiver->connected = false;
    receiver->receiving = false;
    receiver->frames_received = 0;
    receiver->frames_dropped = 0;
    receiver->avg_fps = 0.0;
    receiver->last_frame_time = 0;
    
    printf("创建GStreamer接收器: %s (%s, %dx%d@%dfps)\n", 
           stream_url, stream_format, width, height, fps);
    
    return receiver;
}

void gstreamer_receiver_destroy(gstreamer_receiver_t* receiver) {
    if (!receiver) return;
    
    printf("销毁GStreamer接收器\n");
    
    // 停止接收
    gstreamer_receiver_stop(receiver);
    
    // 断开连接
    gstreamer_receiver_disconnect(receiver);
    
    free(receiver);
}

bool gstreamer_receiver_connect(gstreamer_receiver_t* receiver) {
    if (!receiver) return false;
    
    if (receiver->connected) {
        printf("警告：GStreamer接收器已连接\n");
        return true;
    }
    
    printf("连接GStreamer流: %s\n", receiver->stream_url);
    
    try {
        // 构建接收管道
        std::string pipeline_str = build_receive_pipeline(receiver->stream_url, receiver->stream_format);
        if (pipeline_str.empty()) {
            printf("错误：构建接收管道失败\n");
            return false;
        }
        
        printf("GStreamer接收管道: %s\n", pipeline_str.c_str());
        
        // 创建管道
        GError* error = nullptr;
        receiver->pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (error) {
            printf("错误：创建GStreamer管道失败: %s\n", error->message);
            g_error_free(error);
            return false;
        }
        
        if (!receiver->pipeline) {
            printf("错误：GStreamer管道创建失败\n");
            return false;
        }
        
        // 获取appsink元素
        receiver->appsink = gst_bin_get_by_name(GST_BIN(receiver->pipeline), "appsink");
        if (!receiver->appsink) {
            printf("错误：无法获取appsink元素\n");
            gst_object_unref(receiver->pipeline);
            receiver->pipeline = nullptr;
            return false;
        }
        
        // 设置总线消息处理
        GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(receiver->pipeline));
        gst_bus_add_watch(bus, bus_message_handler, receiver);
        gst_object_unref(bus);
        
        receiver->connected = true;
        printf("GStreamer接收器连接成功\n");
        return true;
        
    } catch (const std::exception& e) {
        printf("GStreamer连接异常: %s\n", e.what());
        return false;
    }
}

void gstreamer_receiver_disconnect(gstreamer_receiver_t* receiver) {
    if (!receiver || !receiver->connected) return;
    
    printf("断开GStreamer连接\n");
    
    // 停止接收
    gstreamer_receiver_stop(receiver);
    
    // 清理管道
    if (receiver->pipeline) {
        gst_element_set_state(receiver->pipeline, GST_STATE_NULL);
        gst_object_unref(receiver->pipeline);
        receiver->pipeline = nullptr;
    }
    
    if (receiver->appsink) {
        gst_object_unref(receiver->appsink);
        receiver->appsink = nullptr;
    }
    
    receiver->connected = false;
    receiver->receiving = false;
    
    printf("GStreamer连接已断开\n");
}

bool gstreamer_receiver_start(gstreamer_receiver_t* receiver) {
    if (!receiver || !receiver->connected) {
        printf("错误：GStreamer接收器未连接\n");
        return false;
    }
    
    if (receiver->receiving) {
        printf("警告：GStreamer接收器已在接收\n");
        return true;
    }
    
    printf("启动GStreamer接收\n");
    
    // 启动管道
    GstStateChangeReturn ret = gst_element_set_state(receiver->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        printf("错误：启动GStreamer管道失败\n");
        return false;
    }
    
    printf("GStreamer接收启动成功\n");
    return true;
}

void gstreamer_receiver_stop(gstreamer_receiver_t* receiver) {
    if (!receiver || !receiver->receiving) return;
    
    printf("停止GStreamer接收\n");
    
    if (receiver->pipeline) {
        gst_element_set_state(receiver->pipeline, GST_STATE_PAUSED);
    }
    
    receiver->receiving = false;
    
    printf("GStreamer接收已停止\n");
}

bool gstreamer_receiver_read_frame(gstreamer_receiver_t* receiver, cv::Mat& frame, int timeout_ms) {
    if (!receiver || !receiver->receiving || !receiver->appsink) {
        return false;
    }
    
    try {
        // 从appsink拉取样本
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(receiver->appsink), 
                                                        timeout_ms * GST_MSECOND);
        if (!sample) {
            return false;
        }
        
        // 获取缓冲区
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            gst_sample_unref(sample);
            return false;
        }
        
        // 获取caps信息
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        
        int width, height;
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);
        
        // 映射缓冲区数据
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }
        
        // 创建OpenCV Mat
        frame = cv::Mat(height, width, CV_8UC3, map.data).clone();
        
        // 清理资源
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        
        // 更新统计信息
        receiver->frames_received++;
        uint64_t current_time = get_timestamp_ms();
        if (receiver->last_frame_time > 0) {
            uint64_t interval = current_time - receiver->last_frame_time;
            if (interval > 0) {
                double current_fps = 1000.0 / interval;
                // 简单的滑动平均
                if (receiver->avg_fps == 0.0) {
                    receiver->avg_fps = current_fps;
                } else {
                    receiver->avg_fps = receiver->avg_fps * 0.9 + current_fps * 0.1;
                }
            }
        }
        receiver->last_frame_time = current_time;
        
        return true;
        
    } catch (const std::exception& e) {
        printf("GStreamer读取帧异常: %s\n", e.what());
        return false;
    }
}

bool gstreamer_receiver_is_connected(gstreamer_receiver_t* receiver) {
    return receiver && receiver->connected;
}

bool gstreamer_receiver_is_receiving(gstreamer_receiver_t* receiver) {
    return receiver && receiver->receiving;
}

void gstreamer_receiver_get_stats(gstreamer_receiver_t* receiver, gstreamer_stats_t* stats) {
    if (!receiver || !stats) return;
    
    stats->frames_received = receiver->frames_received;
    stats->frames_dropped = receiver->frames_dropped;
    stats->current_fps = receiver->avg_fps;
    stats->avg_fps = receiver->avg_fps;
    stats->is_connected = receiver->connected;
    stats->connection_time = 0; // TODO: 实现连接时间跟踪
    stats->last_frame_time = receiver->last_frame_time;
}

void gstreamer_receiver_reset_stats(gstreamer_receiver_t* receiver) {
    if (!receiver) return;
    
    receiver->frames_received = 0;
    receiver->frames_dropped = 0;
    receiver->avg_fps = 0.0;
    receiver->last_frame_time = 0;
}