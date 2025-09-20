/**
 * 后端通信客户端
 * 负责与cpp_backend后端进程通信
 */

#ifndef BACKEND_BACKEND_CLIENT_H
#define BACKEND_BACKEND_CLIENT_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include "common/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 后端通信状态 */
typedef enum {
    BACKEND_DISCONNECTED = 0,
    BACKEND_CONNECTING,
    BACKEND_CONNECTED,
    BACKEND_ERROR
} backend_status_t;

/* PLC通信状态 */
typedef enum {
    PLC_DISCONNECTED = 0,
    PLC_CONNECTED,
    PLC_ERROR
} plc_status_t;

/* 系统状态 */
typedef enum {
    SYSTEM_STOPPED = 0,
    SYSTEM_RUNNING = 1,
    SYSTEM_ERROR = 2,
    SYSTEM_PAUSED = 3,
    SYSTEM_EMERGENCY = 4,
    SYSTEM_MAINTENANCE = 5
} system_status_t;

/* 切割坐标数据 */
typedef struct {
    bool coordinate_ready;      // 坐标就绪标志
    int32_t x_coordinate;       // X坐标 (0.1mm精度)
    int16_t blade_number;       // 刀片编号 (1=刀片1, 2=刀片2, 3=双刀片)
    int16_t cutting_quality;    // 切割质量 (0=正常, 1=异常)
    uint64_t timestamp;         // 时间戳
} cutting_coordinate_t;

/* 系统健康信息 */
typedef struct {
    float cpu_usage;            // CPU使用率 (%)
    float memory_usage;         // 内存使用率 (%)
    float gpu_usage;            // GPU使用率 (%)
    float gpu_memory_usage;     // GPU内存使用率 (%)
    float temperature;          // 系统温度 (°C)
    uint32_t ai_fps;           // AI推理帧率
    uint32_t camera_fps;       // 摄像头帧率
    uint32_t heartbeat_count;  // 心跳计数
} system_health_t;

/* 后端客户端结构 */
typedef struct {
    // 连接信息
    char backend_host[64];      // 后端主机地址
    int backend_port;           // 后端端口
    int socket_fd;              // Socket文件描述符
    
    // 状态信息
    backend_status_t backend_status;
    plc_status_t plc_status;
    system_status_t system_status;
    
    // 数据缓存
    cutting_coordinate_t current_coordinate;
    system_health_t system_health;
    
    // 线程控制
    pthread_t communication_thread;
    bool thread_running;
    pthread_mutex_t data_mutex;
    
    // 统计信息
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t last_heartbeat_time;
} backend_client_t;

/**
 * 创建后端客户端
 * @param host 后端主机地址
 * @param port 后端端口
 * @return 客户端指针，失败返回NULL
 */
backend_client_t* backend_client_create(const char* host, int port);

/**
 * 销毁后端客户端
 * @param client 客户端指针
 */
void backend_client_destroy(backend_client_t* client);

/**
 * 连接到后端
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool backend_client_connect(backend_client_t* client);

/**
 * 断开后端连接
 * @param client 客户端指针
 */
void backend_client_disconnect(backend_client_t* client);

/**
 * 启动通信线程
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool backend_client_start_communication(backend_client_t* client);

/**
 * 停止通信线程
 * @param client 客户端指针
 */
void backend_client_stop_communication(backend_client_t* client);

/**
 * 发送PLC命令
 * @param client 客户端指针
 * @param command PLC命令代码
 * @return 成功返回true，失败返回false
 */
bool backend_client_send_plc_command(backend_client_t* client, int16_t command);

/**
 * 获取切割坐标
 * @param client 客户端指针
 * @param coordinate 输出的坐标数据
 * @return 成功返回true，失败返回false
 */
bool backend_client_get_coordinate(backend_client_t* client, cutting_coordinate_t* coordinate);

/**
 * 获取系统健康信息
 * @param client 客户端指针
 * @param health 输出的健康信息
 * @return 成功返回true，失败返回false
 */
bool backend_client_get_system_health(backend_client_t* client, system_health_t* health);

/**
 * 获取后端连接状态
 * @param client 客户端指针
 * @return 后端状态
 */
backend_status_t backend_client_get_backend_status(backend_client_t* client);

/**
 * 获取PLC连接状态
 * @param client 客户端指针
 * @return PLC状态
 */
plc_status_t backend_client_get_plc_status(backend_client_t* client);

/**
 * 获取系统状态
 * @param client 客户端指针
 * @return 系统状态
 */
system_status_t backend_client_get_system_status(backend_client_t* client);

/**
 * 检查后端进程是否运行
 * @return 运行返回true，否则返回false
 */
bool backend_client_is_backend_running(void);

/**
 * 启动后端进程
 * @return 成功返回true，失败返回false
 */
bool backend_client_start_backend_process(void);

/**
 * 停止后端进程
 * @return 成功返回true，失败返回false
 */
bool backend_client_stop_backend_process(void);

#ifdef __cplusplus
}
#endif

#endif // BACKEND_BACKEND_CLIENT_H