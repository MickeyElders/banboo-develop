/**
 * V4L2摄像头驱动
 * 支持CSI摄像头和USB摄像头
 */

#ifndef CAMERA_V4L2_CAMERA_H
#define CAMERA_V4L2_CAMERA_H

#include <linux/videodev2.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <opencv2/opencv.hpp>
#include "common/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* V4L2缓冲区结构 */
typedef struct {
    void *start;
    size_t length;
    struct v4l2_buffer buffer;
} v4l2_buffer_t;

/* V4L2摄像头状态 */
typedef enum {
    V4L2_STATE_CLOSED = 0,
    V4L2_STATE_OPENED,
    V4L2_STATE_CONFIGURED,
    V4L2_STATE_STREAMING,
    V4L2_STATE_ERROR
} v4l2_state_t;

/* V4L2摄像头设备结构 */
typedef struct {
    int fd;                         // 文件描述符
    char device_path[64];           // 设备路径
    v4l2_state_t state;             // 设备状态
    
    // 格式信息
    struct v4l2_format format;      // 视频格式
    struct v4l2_capability cap;     // 设备能力
    
    // 缓冲区管理
    v4l2_buffer_t *buffers;         // 缓冲区数组
    int buffer_count;               // 缓冲区数量
    
    // 控制参数
    int brightness;                 // 亮度
    int contrast;                   // 对比度
    int saturation;                 // 饱和度
    int hue;                        // 色调
    int exposure;                   // 曝光
    int gain;                       // 增益
    bool auto_exposure;             // 自动曝光
    bool auto_white_balance;        // 自动白平衡
    
    // 统计信息
    uint64_t frame_count;           // 帧计数
    uint64_t dropped_frames;        // 丢帧数
    double fps;                     // 当前帧率
    uint64_t last_frame_time;       // 上一帧时间
    
    // 线程同步
    pthread_mutex_t mutex;          // 互斥锁
    bool thread_running;            // 线程运行标志
} v4l2_camera_t;

/* 摄像头信息结构 */
typedef struct {
    char device_path[64];           // 设备路径
    char driver[32];                // 驱动名称
    char card[32];                  // 设备名称
    uint32_t version;               // 版本号
    uint32_t capabilities;          // 能力标志
    
    // 支持的格式
    struct {
        uint32_t pixelformat;       // 像素格式
        char description[32];       // 格式描述
        uint32_t width_min;         // 最小宽度
        uint32_t width_max;         // 最大宽度
        uint32_t height_min;        // 最小高度
        uint32_t height_max;        // 最大高度
    } formats[16];
    int format_count;               // 格式数量
} v4l2_camera_info_t;

/* 帧回调函数类型 */
typedef void (*v4l2_frame_callback_t)(const cv::Mat& frame, void* user_data);

/* API函数声明 */

/**
 * 创建V4L2摄像头对象
 * @param device_path 设备路径，如"/dev/video0"
 * @return 摄像头对象指针，失败返回NULL
 */
v4l2_camera_t* v4l2_camera_create(const char* device_path);

/**
 * 销毁V4L2摄像头对象
 * @param camera 摄像头对象指针
 */
void v4l2_camera_destroy(v4l2_camera_t* camera);

/**
 * 打开摄像头设备
 * @param camera 摄像头对象
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_open(v4l2_camera_t* camera);

/**
 * 关闭摄像头设备
 * @param camera 摄像头对象
 */
void v4l2_camera_close(v4l2_camera_t* camera);

/**
 * 配置摄像头格式
 * @param camera 摄像头对象
 * @param width 宽度
 * @param height 高度
 * @param pixelformat 像素格式，如V4L2_PIX_FMT_YUYV
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_format(v4l2_camera_t* camera, int width, int height, uint32_t pixelformat);

/**
 * 设置帧率
 * @param camera 摄像头对象
 * @param fps 目标帧率
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_fps(v4l2_camera_t* camera, int fps);

/**
 * 分配缓冲区
 * @param camera 摄像头对象
 * @param buffer_count 缓冲区数量
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_allocate_buffers(v4l2_camera_t* camera, int buffer_count);

/**
 * 释放缓冲区
 * @param camera 摄像头对象
 */
void v4l2_camera_free_buffers(v4l2_camera_t* camera);

/**
 * 开始流式传输
 * @param camera 摄像头对象
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_start_streaming(v4l2_camera_t* camera);

/**
 * 停止流式传输
 * @param camera 摄像头对象
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_stop_streaming(v4l2_camera_t* camera);

/**
 * 获取一帧图像
 * @param camera 摄像头对象
 * @param frame 输出的OpenCV Mat对象
 * @param timeout_ms 超时时间（毫秒），0表示非阻塞，-1表示无限等待
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_get_frame(v4l2_camera_t* camera, cv::Mat& frame, int timeout_ms);

/**
 * 设置摄像头控制参数
 * @param camera 摄像头对象
 * @param control_id 控制ID，如V4L2_CID_BRIGHTNESS
 * @param value 控制值
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_control(v4l2_camera_t* camera, uint32_t control_id, int value);

/**
 * 获取摄像头控制参数
 * @param camera 摄像头对象
 * @param control_id 控制ID
 * @param value 输出的控制值
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_get_control(v4l2_camera_t* camera, uint32_t control_id, int* value);

/**
 * 设置亮度
 * @param camera 摄像头对象
 * @param brightness 亮度值（0-255）
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_brightness(v4l2_camera_t* camera, int brightness);

/**
 * 设置对比度
 * @param camera 摄像头对象
 * @param contrast 对比度值（0-255）
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_contrast(v4l2_camera_t* camera, int contrast);

/**
 * 设置曝光
 * @param camera 摄像头对象
 * @param exposure 曝光值，-1为自动曝光
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_exposure(v4l2_camera_t* camera, int exposure);

/**
 * 设置增益
 * @param camera 摄像头对象
 * @param gain 增益值，-1为自动增益
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_gain(v4l2_camera_t* camera, int gain);

/**
 * 启用/禁用自动曝光
 * @param camera 摄像头对象
 * @param enable 是否启用
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_auto_exposure(v4l2_camera_t* camera, bool enable);

/**
 * 启用/禁用自动白平衡
 * @param camera 摄像头对象
 * @param enable 是否启用
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_set_auto_white_balance(v4l2_camera_t* camera, bool enable);

/**
 * 获取摄像头信息
 * @param camera 摄像头对象
 * @param info 输出的摄像头信息
 * @return 成功返回true，失败返回false
 */
bool v4l2_camera_get_info(v4l2_camera_t* camera, v4l2_camera_info_t* info);

/**
 * 获取性能统计信息
 * @param camera 摄像头对象
 * @param stats 输出的统计信息
 */
void v4l2_camera_get_stats(v4l2_camera_t* camera, performance_stats_t* stats);

/**
 * 重置性能统计
 * @param camera 摄像头对象
 */
void v4l2_camera_reset_stats(v4l2_camera_t* camera);

/**
 * 检查摄像头是否正在流式传输
 * @param camera 摄像头对象
 * @return 正在流式传输返回true，否则返回false
 */
bool v4l2_camera_is_streaming(v4l2_camera_t* camera);

/**
 * 获取当前帧率
 * @param camera 摄像头对象
 * @return 当前帧率
 */
double v4l2_camera_get_current_fps(v4l2_camera_t* camera);

/**
 * 枚举所有可用的V4L2设备
 * @param devices 输出的设备列表
 * @param max_devices 最大设备数量
 * @return 实际找到的设备数量
 */
int v4l2_enumerate_devices(char devices[][64], int max_devices);

/**
 * 检查设备是否支持指定格式
 * @param device_path 设备路径
 * @param width 宽度
 * @param height 高度
 * @param pixelformat 像素格式
 * @return 支持返回true，不支持返回false
 */
bool v4l2_check_format_support(const char* device_path, int width, int height, uint32_t pixelformat);

/**
 * 获取设备支持的所有格式
 * @param device_path 设备路径
 * @param formats 输出的格式列表
 * @param max_formats 最大格式数量
 * @return 实际格式数量
 */
int v4l2_get_supported_formats(const char* device_path, struct v4l2_fmtdesc* formats, int max_formats);

/* 工具函数 */

/**
 * 像素格式转换为字符串
 * @param pixelformat 像素格式
 * @return 格式字符串
 */
const char* v4l2_pixelformat_to_string(uint32_t pixelformat);

/**
 * 打印摄像头信息
 * @param camera 摄像头对象
 */
void v4l2_camera_print_info(v4l2_camera_t* camera);

/**
 * 打印摄像头控制参数
 * @param camera 摄像头对象
 */
void v4l2_camera_print_controls(v4l2_camera_t* camera);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_V4L2_CAMERA_H