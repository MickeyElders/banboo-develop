/**
 * 共享内存读取器 - 前端专用
 * 用于从后端共享内存读取摄像头帧数据
 */

#ifndef CAMERA_SHARED_MEMORY_READER_H
#define CAMERA_SHARED_MEMORY_READER_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/* 共享内存帧头结构（与后端保持一致） */
typedef struct {
    uint32_t frame_id;          // 帧ID
    uint64_t timestamp;         // 时间戳（毫秒）
    uint32_t width;             // 图像宽度
    uint32_t height;            // 图像高度
    uint32_t channels;          // 通道数
    uint32_t data_size;         // 数据大小
    uint8_t  format;            // 图像格式 (0=BGR, 1=RGB, 2=YUV)
    uint8_t  ready;             // 数据就绪标志 (0=写入中, 1=就绪)
    uint8_t  reserved[2];       // 预留字节对齐
} shared_frame_header_t;

/* 共享内存统计信息 */
typedef struct {
    uint64_t total_frames_read;
    uint64_t frames_dropped;
    double   fps;
    uint32_t current_frame_id;
    uint32_t last_read_frame_id;
    bool     connected;
} shared_memory_stats_t;

/* 共享内存读取器结构 */
typedef struct {
    char key_path[256];             // 共享内存key路径
    key_t shm_key;                  // IPC密钥
    int shm_id;                     // 共享内存ID
    void* shm_ptr;                  // 共享内存指针
    size_t shm_size;                // 共享内存大小
    
    // 帧信息
    size_t frame_size;              // 单帧数据大小
    size_t buffer_count;            // 缓冲区数量
    uint32_t last_read_frame_id;    // 上次读取的帧ID
    
    // 状态信息
    bool connected;                 // 连接状态
    shared_memory_stats_t stats;    // 统计信息
    
    // OpenCV Mat缓存
    cv::Mat cached_frame;           // 缓存的帧数据
    bool frame_cache_valid;         // 帧缓存有效性
} shared_memory_reader_t;

/**
 * 创建共享内存读取器
 * @param key_path 共享内存key路径
 * @return 读取器指针，失败返回NULL
 */
shared_memory_reader_t* shared_memory_reader_create(const char* key_path);

/**
 * 销毁共享内存读取器
 * @param reader 读取器指针
 */
void shared_memory_reader_destroy(shared_memory_reader_t* reader);

/**
 * 连接到共享内存
 * @param reader 读取器
 * @return 成功返回true，失败返回false
 */
bool shared_memory_reader_connect(shared_memory_reader_t* reader);

/**
 * 断开共享内存连接
 * @param reader 读取器
 */
void shared_memory_reader_disconnect(shared_memory_reader_t* reader);

/**
 * 检查是否有新帧可读
 * @param reader 读取器
 * @return 有新帧返回true，否则返回false
 */
bool shared_memory_reader_has_new_frame(shared_memory_reader_t* reader);

/**
 * 读取最新帧数据到OpenCV Mat
 * @param reader 读取器
 * @param frame 输出的OpenCV Mat对象
 * @param timeout_ms 超时时间（毫秒），0表示非阻塞，-1表示无限等待
 * @return 成功返回true，失败返回false
 */
bool shared_memory_reader_read_frame(shared_memory_reader_t* reader, cv::Mat& frame, int timeout_ms);

/**
 * 读取最新帧数据到缓冲区
 * @param reader 读取器
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @param width 输出图像宽度
 * @param height 输出图像高度
 * @param channels 输出图像通道数
 * @param timeout_ms 超时时间（毫秒）
 * @return 成功返回true，失败返回false
 */
bool shared_memory_reader_read_frame_to_buffer(
    shared_memory_reader_t* reader, 
    uint8_t* buffer, 
    size_t buffer_size,
    uint32_t* width,
    uint32_t* height, 
    uint32_t* channels,
    int timeout_ms
);

/**
 * 获取当前帧ID
 * @param reader 读取器
 * @return 当前帧ID
 */
uint32_t shared_memory_reader_get_current_frame_id(shared_memory_reader_t* reader);

/**
 * 获取帧数据尺寸信息
 * @param reader 读取器
 * @param width 输出宽度
 * @param height 输出高度
 * @param channels 输出通道数
 * @return 成功返回true，失败返回false
 */
bool shared_memory_reader_get_frame_info(
    shared_memory_reader_t* reader,
    uint32_t* width,
    uint32_t* height,
    uint32_t* channels
);

/**
 * 获取统计信息
 * @param reader 读取器
 * @param stats 输出统计信息
 */
void shared_memory_reader_get_stats(shared_memory_reader_t* reader, shared_memory_stats_t* stats);

/**
 * 重置统计信息
 * @param reader 读取器
 */
void shared_memory_reader_reset_stats(shared_memory_reader_t* reader);

/**
 * 检查共享内存连接状态
 * @param reader 读取器
 * @return 已连接返回true，否则返回false
 */
bool shared_memory_reader_is_connected(shared_memory_reader_t* reader);

/**
 * 等待共享内存可用
 * @param key_path 共享内存key路径
 * @param timeout_ms 超时时间（毫秒）
 * @return 可用返回true，超时返回false
 */
bool shared_memory_reader_wait_for_availability(const char* key_path, int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_SHARED_MEMORY_READER_H