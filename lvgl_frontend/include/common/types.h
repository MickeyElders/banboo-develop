/**
 * 通用类型定义
 */

#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <cstddef>

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#endif

/* 系统状态 */
typedef enum {
    SYSTEM_STATUS_IDLE = 0,         // 待机
    SYSTEM_STATUS_INITIALIZING,     // 初始化中
    SYSTEM_STATUS_READY,            // 就绪
    SYSTEM_STATUS_RUNNING,          // 运行中
    SYSTEM_STATUS_PAUSED,           // 暂停
    SYSTEM_STATUS_ERROR,            // 错误
    SYSTEM_STATUS_SHUTDOWN          // 关闭
} system_status_t;

/* 错误码定义 */
typedef enum {
    ERROR_CODE_SUCCESS = 0,         // 成功
    ERROR_CODE_INIT_FAILED,         // 初始化失败
    ERROR_CODE_CONFIG_ERROR,        // 配置错误
    ERROR_CODE_CAMERA_ERROR,        // 摄像头错误
    ERROR_CODE_AI_ERROR,            // AI推理错误
    ERROR_CODE_DISPLAY_ERROR,       // 显示错误
    ERROR_CODE_TOUCH_ERROR,         // 触摸错误
    ERROR_CODE_MEMORY_ERROR,        // 内存错误
    ERROR_CODE_TIMEOUT,             // 超时
    ERROR_CODE_UNKNOWN              // 未知错误
} error_code_t;

/* 日志级别 */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} log_level_t;

/* 摄像头配置 */
typedef struct {
    char device_path[64];           // 设备路径
    int width;                      // 宽度
    int height;                     // 高度
    int fps;                        // 帧率
    int buffer_count;               // 缓冲区数量
    bool use_hardware_acceleration; // 硬件加速
    int exposure;                   // 曝光值
    int gain;                       // 增益
    char shared_memory_key[256];    // 共享内存key路径
    bool use_shared_memory;         // 使用共享内存
} camera_config_t;

/* AI模型配置 */
typedef struct {
    char model_path[256];           // 模型文件路径
    char engine_path[256];          // TensorRT引擎路径
    int input_width;                // 输入宽度
    int input_height;               // 输入高度
    float confidence_threshold;     // 置信度阈值
    float nms_threshold;            // NMS阈值
    bool use_tensorrt;              // 使用TensorRT
    bool use_int8;                  // 使用INT8量化
    bool use_dla;                   // 使用DLA
} ai_config_t;

/* 显示配置 */
typedef struct {
    char framebuffer_device[64];    // 帧缓冲设备
    int width;                      // 显示宽度
    int height;                     // 显示高度
    int bpp;                        // 每像素位数
    bool vsync;                     // 垂直同步
    int brightness;                 // 亮度
} display_config_t;

/* 触摸配置 */
typedef struct {
    char device_path[64];           // 触摸设备路径
    bool calibration_enabled;       // 校准启用
    int sensitivity;                // 灵敏度
    bool gesture_enabled;           // 手势识别
} touch_config_t;

/* 系统配置 */
typedef struct {
    bool debug_mode;                // 调试模式
    bool fullscreen;                // 全屏模式
    log_level_t log_level;          // 日志级别
    char log_file[256];             // 日志文件路径
    
    camera_config_t camera;         // 摄像头配置
    ai_config_t ai;                 // AI配置
    display_config_t display;       // 显示配置
    touch_config_t touch;           // 触摸配置
} system_config_t;

/* 检测点结构 */
typedef struct {
    float x;                        // X坐标 (像素)
    float y;                        // Y坐标 (像素)
    float x_mm;                     // X坐标 (毫米)
    float y_mm;                     // Y坐标 (毫米)
    float confidence;               // 置信度
    int class_id;                   // 类别ID (0=切点, 1=节点)
    char class_name[32];            // 类别名称
} detection_point_t;

/* 检测结果 */
typedef struct {
    detection_point_t points[16];   // 检测点数组 (最多16个)
    int point_count;                // 检测点数量
    float processing_time_ms;       // 处理时间(毫秒)
    uint64_t timestamp;             // 时间戳
    bool success;                   // 是否成功
    char error_message[256];        // 错误信息
} detection_result_t;

/* 帧信息 */
typedef struct {
#ifdef __cplusplus
    cv::Mat image;                  // 图像数据
#else
    void* image_data;               // 图像数据指针(C语言)
#endif
    uint64_t timestamp;             // 时间戳
    int frame_id;                   // 帧ID
    int width;                      // 宽度
    int height;                     // 高度
    bool valid;                     // 是否有效
} frame_info_t;

/* 性能统计 */
typedef struct {
    float fps;                      // 帧率
    float avg_processing_time_ms;   // 平均处理时间
    int dropped_frames;             // 丢帧数
    uint64_t total_frames;          // 总帧数
    float cpu_usage;                // CPU使用率
    float memory_usage_mb;          // 内存使用量(MB)
    float gpu_usage;                // GPU使用率
} performance_stats_t;

/* 事件类型 */
typedef enum {
    EVENT_TYPE_CAMERA_FRAME_READY = 0,  // 摄像头帧就绪
    EVENT_TYPE_DETECTION_COMPLETE,      // 检测完成
    EVENT_TYPE_TOUCH_INPUT,             // 触摸输入
    EVENT_TYPE_BUTTON_CLICK,            // 按钮点击
    EVENT_TYPE_SYSTEM_ERROR,            // 系统错误
    EVENT_TYPE_CONFIG_CHANGED,          // 配置改变
    EVENT_TYPE_SHUTDOWN_REQUEST         // 关闭请求
} event_type_t;

/* 事件数据 */
typedef struct {
    event_type_t type;              // 事件类型
    void* data;                     // 事件数据
    size_t data_size;               // 数据大小
    uint64_t timestamp;             // 时间戳
} event_data_t;

/* 触摸事件数据 */
typedef struct {
    int x;                          // X坐标
    int y;                          // Y坐标
    bool pressed;                   // 是否按下
    int pressure;                   // 压力值
} touch_event_t;

/* 按钮点击事件数据 */
typedef struct {
    int button_id;                  // 按钮ID
    char button_name[32];           // 按钮名称
} button_click_event_t;

/* 常用常量定义 */
#define MAX_PATH_LENGTH         256
#define MAX_NAME_LENGTH         64
#define MAX_ERROR_MESSAGE       256
#define MAX_DETECTION_POINTS    16
#define MAX_FRAME_BUFFER_SIZE   10

/* 像素到毫米转换比例 (需要根据实际相机标定) */
#define PIXEL_TO_MM_RATIO       0.1f

/* 默认配置值 */
#define DEFAULT_CAMERA_WIDTH    1920
#define DEFAULT_CAMERA_HEIGHT   1080
#define DEFAULT_CAMERA_FPS      30
#define DEFAULT_DISPLAY_WIDTH   1920
#define DEFAULT_DISPLAY_HEIGHT  1080
#define DEFAULT_TOUCH_SENSITIVITY  10

/* 颜色定义 (ARGB8888格式) */
#define COLOR_BLACK             0xFF000000
#define COLOR_WHITE             0xFFFFFFFF
#define COLOR_RED               0xFFFF0000
#define COLOR_GREEN             0xFF00FF00
#define COLOR_BLUE              0xFF0000FF
#define COLOR_YELLOW            0xFFFFFF00
#define COLOR_CYAN              0xFF00FFFF
#define COLOR_MAGENTA           0xFFFF00FF
#define COLOR_GRAY              0xFF808080
#define COLOR_DARK_GRAY         0xFF404040
#define COLOR_LIGHT_GRAY        0xFFC0C0C0

/* 透明度定义 */
#define OPACITY_TRANSPARENT     0
#define OPACITY_SEMI_TRANSPARENT 128
#define OPACITY_OPAQUE          255

#endif // COMMON_TYPES_H