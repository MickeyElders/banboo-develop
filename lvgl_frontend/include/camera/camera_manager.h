/**
 * 摄像头管理器
 * 集成V4L2摄像头驱动到LVGL界面
 */

#ifndef CAMERA_CAMERA_MANAGER_H
#define CAMERA_CAMERA_MANAGER_H

#include "camera/shared_memory_reader.h"
#include "lvgl.h"
#include <pthread.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 摄像头管理器结构 */
typedef struct {
    shared_memory_reader_t *shared_reader;  // 共享内存读取器
    lv_obj_t *video_img;            // LVGL图像对象
    lv_img_dsc_t img_dsc;          // 图像描述符
    uint8_t *img_buffer;           // 图像缓冲区
    
    // 线程控制
    pthread_t capture_thread;       // 捕获线程
    bool thread_running;           // 线程运行标志
    pthread_mutex_t frame_mutex;   // 帧数据互斥锁
    
    // 配置参数
    char shared_memory_key[256];   // 共享内存key路径
    int width;                     // 图像宽度
    int height;                    // 图像高度
    int fps;                       // 帧率
    
    // 性能统计
    uint64_t frame_count;          // 处理的帧数
    double avg_fps;                // 平均帧率
    uint64_t last_update_time;     // 上次更新时间
    
    // 连接状态
    bool connected;                // 共享内存连接状态
    uint64_t connection_retry_time; // 连接重试时间
} camera_manager_t;

/**
 * 创建摄像头管理器
 * @param shared_memory_key 共享内存key路径，如"/tmp/bamboo_camera_shm"
 * @param width 图像宽度
 * @param height 图像高度
 * @param fps 目标帧率
 * @return 摄像头管理器指针，失败返回NULL
 */
camera_manager_t* camera_manager_create(const char* shared_memory_key, int width, int height, int fps);

/**
 * 销毁摄像头管理器
 * @param manager 摄像头管理器指针
 */
void camera_manager_destroy(camera_manager_t* manager);

/**
 * 初始化摄像头
 * @param manager 摄像头管理器
 * @return 成功返回true，失败返回false
 */
bool camera_manager_init(camera_manager_t* manager);

/**
 * 清理摄像头资源
 * @param manager 摄像头管理器
 */
void camera_manager_deinit(camera_manager_t* manager);

/**
 * 开始摄像头捕获
 * @param manager 摄像头管理器
 * @return 成功返回true，失败返回false
 */
bool camera_manager_start_capture(camera_manager_t* manager);

/**
 * 停止摄像头捕获
 * @param manager 摄像头管理器
 */
void camera_manager_stop_capture(camera_manager_t* manager);

/**
 * 创建LVGL视频显示对象
 * @param manager 摄像头管理器
 * @param parent 父对象
 * @return 成功返回true，失败返回false
 */
bool camera_manager_create_video_object(camera_manager_t* manager, lv_obj_t* parent);

/**
 * 更新视频显示
 * @param manager 摄像头管理器
 */
void camera_manager_update_display(camera_manager_t* manager);

/**
 * 获取当前帧率
 * @param manager 摄像头管理器
 * @return 当前帧率
 */
double camera_manager_get_fps(camera_manager_t* manager);

/**
 * 获取摄像头状态
 * @param manager 摄像头管理器
 * @return 摄像头是否正在运行
 */
bool camera_manager_is_running(camera_manager_t* manager);

/**
 * 获取共享内存连接状态
 * @param manager 摄像头管理器
 * @return 共享内存是否已连接
 */
bool camera_manager_is_connected(camera_manager_t* manager);

/**
 * 获取性能统计
 * @param manager 摄像头管理器
 * @param frame_count 输出帧数
 * @param avg_fps 输出平均帧率
 */
void camera_manager_get_stats(camera_manager_t* manager, uint64_t* frame_count, double* avg_fps);

/**
 * 获取共享内存统计
 * @param manager 摄像头管理器
 * @param stats 输出共享内存统计信息
 */
void camera_manager_get_shared_memory_stats(camera_manager_t* manager, shared_memory_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_CAMERA_MANAGER_H