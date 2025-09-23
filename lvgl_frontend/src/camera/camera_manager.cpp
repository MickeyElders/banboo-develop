#include "camera/camera_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <opencv2/opencv.hpp>

// 获取当前时间戳（毫秒）
static uint64_t get_timestamp_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// GStreamer流读取线程函数
static void* gstreamer_read_thread(void* arg) {
    camera_manager_t* manager = (camera_manager_t*)arg;
    cv::Mat frame;
    
    printf("GStreamer流读取线程启动\n");
    
    while (manager->thread_running) {
        // 如果未连接，尝试连接
        if (!manager->connected) {
            if (gstreamer_receiver_connect(manager->gst_receiver)) {
                manager->connected = true;
                if (gstreamer_receiver_start(manager->gst_receiver)) {
                    printf("GStreamer流连接并启动成功\n");
                } else {
                    printf("GStreamer流启动失败\n");
                    gstreamer_receiver_disconnect(manager->gst_receiver);
                    manager->connected = false;
                }
            } else {
                // 连接失败，等待重试
                uint64_t current_time = get_timestamp_ms();
                if (current_time - manager->connection_retry_time > 2000) { // 2秒重试一次
                    printf("尝试重新连接GStreamer流...\n");
                    manager->connection_retry_time = current_time;
                }
                usleep(200000); // 200ms，给后端更多时间
                continue;
            }
        }
        
        // 从GStreamer流读取一帧图像
        if (gstreamer_receiver_read_frame(manager->gst_receiver, frame, 100)) {
            pthread_mutex_lock(&manager->frame_mutex);
            
            // 调整图像尺寸到显示区域
            cv::Mat resized_frame;
            if (frame.cols != manager->width || frame.rows != manager->height) {
                cv::resize(frame, resized_frame, cv::Size(manager->width, manager->height));
            } else {
                resized_frame = frame;
            }
            
            // 转换BGR到RGB格式（LVGL使用RGB）
            cv::Mat rgb_frame;
            cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
            
            // 复制图像数据到缓冲区
            if (manager->img_buffer) {
                memcpy(manager->img_buffer, rgb_frame.data,
                       manager->width * manager->height * 3);
            }
            
            // 更新统计信息
            manager->frame_count++;
            uint64_t current_time = get_timestamp_ms();
            if (manager->last_update_time > 0) {
                uint64_t interval = current_time - manager->last_update_time;
                if (interval > 1000) { // 每秒更新一次平均帧率
                    manager->avg_fps = 1000.0 / interval;
                    manager->last_update_time = current_time;
                }
            } else {
                manager->last_update_time = current_time;
            }
            
            pthread_mutex_unlock(&manager->frame_mutex);
            
        } else {
            // 读取失败，检查连接状态
            if (!gstreamer_receiver_is_connected(manager->gst_receiver)) {
                manager->connected = false;
                printf("GStreamer流连接丢失\n");
            }
            usleep(10000); // 10ms
        }
    }
    
    printf("GStreamer流读取线程退出\n");
    return nullptr;
}

camera_manager_t* camera_manager_create(const char* stream_url, const char* stream_format, 
                                       int width, int height, int fps) {
    if (!stream_url || !stream_format || width <= 0 || height <= 0 || fps <= 0) {
        printf("错误：摄像头管理器参数无效\n");
        return nullptr;
    }
    
    camera_manager_t* manager = (camera_manager_t*)calloc(1, sizeof(camera_manager_t));
    if (!manager) {
        printf("错误：分配摄像头管理器内存失败\n");
        return nullptr;
    }
    
    // 初始化基本参数
    strncpy(manager->stream_url, stream_url, sizeof(manager->stream_url) - 1);
    strncpy(manager->stream_format, stream_format, sizeof(manager->stream_format) - 1);
    manager->width = width;
    manager->height = height;
    manager->fps = fps;
    
    // 创建GStreamer接收器
    manager->gst_receiver = gstreamer_receiver_create(stream_url, stream_format, width, height, fps);
    if (!manager->gst_receiver) {
        printf("错误：创建GStreamer接收器失败\n");
        free(manager);
        return nullptr;
    }
    
    // 分配图像缓冲区（RGB格式）
    manager->img_buffer = (uint8_t*)malloc(width * height * 3);
    if (!manager->img_buffer) {
        printf("错误：分配图像缓冲区失败\n");
        gstreamer_receiver_destroy(manager->gst_receiver);
        free(manager);
        return nullptr;
    }
    
    // 初始化图像描述符
    manager->img_dsc.header.always_zero = 0;
    manager->img_dsc.header.w = width;
    manager->img_dsc.header.h = height;
    manager->img_dsc.data_size = width * height * 3;
    manager->img_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;
    manager->img_dsc.data = manager->img_buffer;
    
    // 初始化线程同步
    pthread_mutex_init(&manager->frame_mutex, nullptr);
    manager->thread_running = false;
    
    // 初始化统计信息
    manager->frame_count = 0;
    manager->avg_fps = 0.0;
    manager->last_update_time = 0;
    
    // 初始化连接状态
    manager->connected = false;
    manager->connection_retry_time = 0;
    
    printf("创建摄像头管理器（GStreamer流）: %s (%s, %dx%d@%dfps)\n", 
           stream_url, stream_format, width, height, fps);
    return manager;
}

void camera_manager_destroy(camera_manager_t* manager) {
    if (!manager) return;
    
    printf("销毁摄像头管理器\n");
    
    // 停止捕获
    camera_manager_stop_capture(manager);
    
    // 清理摄像头资源
    camera_manager_deinit(manager);
    
    // 释放图像缓冲区
    if (manager->img_buffer) {
        free(manager->img_buffer);
    }
    
    // 销毁GStreamer接收器
    if (manager->gst_receiver) {
        gstreamer_receiver_destroy(manager->gst_receiver);
    }
    
    // 销毁互斥锁
    pthread_mutex_destroy(&manager->frame_mutex);
    
    free(manager);
}

bool camera_manager_init(camera_manager_t* manager) {
    if (!manager) return false;
    
    printf("异步初始化摄像头管理器（GStreamer流模式）...\n");
    
    // 完全异步模式：不等待GStreamer流，立即返回成功
    // GStreamer流连接和数据获取将在后台线程中处理
    manager->connected = false;
    manager->connection_retry_time = 0;
    
    printf("摄像头管理器异步初始化完成（无阻塞模式）\n");
    printf("GStreamer流连接将在后台线程中自动重试\n");
    
    return true; // 总是返回成功，不阻塞UI启动
}

void camera_manager_deinit(camera_manager_t* manager) {
    if (!manager) return;
    
    printf("清理摄像头管理器资源\n");
    
    // 停止捕获线程
    camera_manager_stop_capture(manager);
    
    // 断开GStreamer连接
    if (manager->gst_receiver) {
        gstreamer_receiver_stop(manager->gst_receiver);
        gstreamer_receiver_disconnect(manager->gst_receiver);
        manager->connected = false;
    }
}

bool camera_manager_start_capture(camera_manager_t* manager) {
    if (!manager || !manager->gst_receiver) return false;
    
    if (manager->thread_running) {
        printf("警告：摄像头捕获已经在运行\n");
        return true;
    }
    
    printf("启动GStreamer流读取...\n");
    
    // 启动读取线程
    manager->thread_running = true;
    if (pthread_create(&manager->capture_thread, nullptr, gstreamer_read_thread, manager) != 0) {
        printf("错误：创建GStreamer流读取线程失败: %s\n", strerror(errno));
        manager->thread_running = false;
        return false;
    }
    
    printf("GStreamer流读取线程启动成功\n");
    return true;
}

void camera_manager_stop_capture(camera_manager_t* manager) {
    if (!manager || !manager->thread_running) return;
    
    printf("停止GStreamer流读取...\n");
    
    // 停止读取线程
    manager->thread_running = false;
    
    // 等待线程退出
    pthread_join(manager->capture_thread, nullptr);
    
    printf("GStreamer流读取已停止\n");
}

bool camera_manager_create_video_object(camera_manager_t* manager, lv_obj_t* parent) {
    if (!manager || !parent) return false;
    
    printf("创建LVGL视频显示对象...\n");
    
    // 创建图像对象
    manager->video_img = lv_img_create(parent);
    if (!manager->video_img) {
        printf("错误：创建LVGL图像对象失败\n");
        return false;
    }
    
    // 设置图像源
    lv_img_set_src(manager->video_img, &manager->img_dsc);
    
    // 设置图像大小和位置
    lv_obj_set_size(manager->video_img, manager->width, manager->height);
    lv_obj_center(manager->video_img);
    
    // 设置图像样式
    lv_obj_set_style_border_width(manager->video_img, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(manager->video_img, lv_color_hex(0x666666), LV_PART_MAIN);
    
    printf("LVGL视频显示对象创建成功\n");
    return true;
}

void camera_manager_update_display(camera_manager_t* manager) {
    if (!manager || !manager->video_img) return;
    
    pthread_mutex_lock(&manager->frame_mutex);
    
    // 通知LVGL图像数据已更新
    lv_img_cache_invalidate_src(&manager->img_dsc);
    lv_obj_invalidate(manager->video_img);
    
    pthread_mutex_unlock(&manager->frame_mutex);
}

double camera_manager_get_fps(camera_manager_t* manager) {
    if (!manager) return 0.0;
    
    return manager->avg_fps;
}

bool camera_manager_is_running(camera_manager_t* manager) {
    return manager && manager->thread_running;
}

bool camera_manager_is_connected(camera_manager_t* manager) {
    return manager && manager->connected;
}

void camera_manager_get_stats(camera_manager_t* manager, uint64_t* frame_count, double* avg_fps) {
    if (!manager) return;
    
    pthread_mutex_lock(&manager->frame_mutex);
    
    if (frame_count) *frame_count = manager->frame_count;
    if (avg_fps) *avg_fps = manager->avg_fps;
    
    pthread_mutex_unlock(&manager->frame_mutex);
}

void camera_manager_get_gstreamer_stats(camera_manager_t* manager, gstreamer_stats_t* stats) {
    if (!manager || !manager->gst_receiver || !stats) return;
    
    gstreamer_receiver_get_stats(manager->gst_receiver, stats);
}