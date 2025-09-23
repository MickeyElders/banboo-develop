/**
 * UNIX Domain Socket客户端
 * 负责与cpp_backend后端进程通信
 */

#ifndef BACKEND_UNIX_SOCKET_CLIENT_H
#define BACKEND_UNIX_SOCKET_CLIENT_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "common/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 消息类型枚举 */
typedef enum {
    MSG_STATUS_REQUEST = 1,     // 前端请求状态
    MSG_STATUS_RESPONSE = 2,    // 后端状态响应  
    MSG_PLC_COMMAND = 3,        // 前端发送PLC命令
    MSG_PLC_COMMAND_ACK = 4,    // 后端命令确认
    MSG_HEARTBEAT = 5,          // 心跳包
    MSG_MODBUS_DATA = 6,        // Modbus数据更新
    MSG_SYSTEM_ERROR = 7        // 系统错误通知
} message_type_t;

/* 通信消息结构 */
typedef struct {
    message_type_t type;
    uint32_t sequence;
    uint64_t timestamp;
    uint32_t data_length;
    char data[512];         // JSON格式数据
} communication_message_t;

/* PLC连接状态 */
typedef enum {
    PLC_DISCONNECTED = 0,
    PLC_CONNECTING = 1,
    PLC_CONNECTED = 2,
    PLC_ERROR = 3,
    PLC_TIMEOUT = 4
} plc_connection_status_t;

/* 后端连接状态 */
typedef enum {
    BACKEND_DISCONNECTED = 0,
    BACKEND_CONNECTING,
    BACKEND_CONNECTED,
    BACKEND_ERROR
} backend_status_t;

/* Modbus寄存器数据 */
typedef struct {
    uint16_t system_status;      // 40001
    uint16_t plc_command;        // 40002
    uint16_t coord_ready;        // 40003
    uint32_t x_coordinate;       // 40004-40005
    uint16_t cut_quality;        // 40006
    uint32_t heartbeat;          // 40007-40008
    uint16_t blade_number;       // 40009
    uint16_t system_health;      // 40010
} modbus_registers_t;

/* 系统状态数据 */
typedef struct {
    plc_connection_status_t plc_status;
    uint32_t heartbeat_counter;
    uint32_t plc_response_time_ms;
    bool emergency_stop;
    char last_error[256];
    uint64_t uptime_seconds;
    modbus_registers_t modbus_data;
} system_status_data_t;

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

/* Unix Socket客户端结构 */
typedef struct {
    // 连接信息
    char socket_path[256];      // Socket文件路径
    int socket_fd;              // Socket文件描述符
    
    // 状态信息
    backend_status_t backend_status;
    system_status_data_t system_status;
    
    // 数据缓存
    cutting_coordinate_t current_coordinate;
    system_health_t system_health;
    
    // 线程控制
    pthread_t communication_thread;
    pthread_t heartbeat_thread;
    bool thread_running;
    pthread_mutex_t data_mutex;
    
    // 消息序列号
    uint32_t message_sequence;
    
    // 统计信息
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t last_heartbeat_time;
    uint64_t connection_time;
    
    // 回调函数
    void (*status_update_callback)(const system_status_data_t* status);
    void (*modbus_data_callback)(const modbus_registers_t* data);
    void (*error_callback)(const char* error_msg);
} unix_socket_client_t;

/**
 * 创建Unix Socket客户端
 * @param socket_path Socket文件路径
 * @return 客户端指针，失败返回NULL
 */
unix_socket_client_t* unix_socket_client_create(const char* socket_path);

/**
 * 销毁Unix Socket客户端
 * @param client 客户端指针
 */
void unix_socket_client_destroy(unix_socket_client_t* client);

/**
 * 连接到后端
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_connect(unix_socket_client_t* client);

/**
 * 断开后端连接
 * @param client 客户端指针
 */
void unix_socket_client_disconnect(unix_socket_client_t* client);

/**
 * 启动通信线程
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_start_communication(unix_socket_client_t* client);

/**
 * 停止通信线程
 * @param client 客户端指针
 */
void unix_socket_client_stop_communication(unix_socket_client_t* client);

/**
 * 发送PLC命令
 * @param client 客户端指针
 * @param command PLC命令代码
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_send_plc_command(unix_socket_client_t* client, int16_t command);

/**
 * 请求系统状态更新
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_request_status(unix_socket_client_t* client);

/**
 * 发送心跳包
 * @param client 客户端指针
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_send_heartbeat(unix_socket_client_t* client);

/**
 * 获取切割坐标
 * @param client 客户端指针
 * @param coordinate 输出的坐标数据
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_get_coordinate(unix_socket_client_t* client, cutting_coordinate_t* coordinate);

/**
 * 获取系统健康信息
 * @param client 客户端指针
 * @param health 输出的健康信息
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_get_system_health(unix_socket_client_t* client, system_health_t* health);

/**
 * 获取后端连接状态
 * @param client 客户端指针
 * @return 后端状态
 */
backend_status_t unix_socket_client_get_backend_status(unix_socket_client_t* client);

/**
 * 获取PLC连接状态
 * @param client 客户端指针
 * @return PLC状态
 */
plc_connection_status_t unix_socket_client_get_plc_status(unix_socket_client_t* client);

/**
 * 获取系统状态
 * @param client 客户端指针
 * @param status 输出的系统状态
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_get_system_status(unix_socket_client_t* client, system_status_data_t* status);

/**
 * 获取Modbus寄存器数据
 * @param client 客户端指针
 * @param registers 输出的寄存器数据
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_get_modbus_registers(unix_socket_client_t* client, modbus_registers_t* registers);

/**
 * 设置状态更新回调
 * @param client 客户端指针
 * @param callback 回调函数
 */
void unix_socket_client_set_status_callback(unix_socket_client_t* client, 
                                          void (*callback)(const system_status_data_t* status));

/**
 * 设置Modbus数据回调
 * @param client 客户端指针
 * @param callback 回调函数
 */
void unix_socket_client_set_modbus_callback(unix_socket_client_t* client,
                                          void (*callback)(const modbus_registers_t* data));

/**
 * 设置错误回调
 * @param client 客户端指针
 * @param callback 回调函数
 */
void unix_socket_client_set_error_callback(unix_socket_client_t* client,
                                         void (*callback)(const char* error_msg));

/**
 * 检查后端进程是否运行
 * @return 运行返回true，否则返回false
 */
bool unix_socket_client_is_backend_running(void);

/**
 * 启动后端进程
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_start_backend_process(void);

/**
 * 停止后端进程
 * @return 成功返回true，失败返回false
 */
bool unix_socket_client_stop_backend_process(void);

#ifdef __cplusplus
}
#endif

#endif // BACKEND_UNIX_SOCKET_CLIENT_H