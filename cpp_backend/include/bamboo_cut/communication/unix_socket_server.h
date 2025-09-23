#pragma once

#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>

#ifdef ENABLE_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace bamboo_cut {
namespace communication {

// 消息类型枚举
enum class MessageType : uint32_t {
    STATUS_REQUEST = 1,     // 前端请求状态
    STATUS_RESPONSE = 2,    // 后端状态响应  
    PLC_COMMAND = 3,        // 前端发送PLC命令
    PLC_COMMAND_ACK = 4,    // 后端命令确认
    HEARTBEAT = 5,          // 心跳包
    MODBUS_DATA = 6,        // Modbus数据更新
    SYSTEM_ERROR = 7        // 系统错误通知
};

// 通信消息结构
struct CommunicationMessage {
    MessageType type;
    uint32_t sequence;
    uint64_t timestamp;
    uint32_t data_length;
    char data[512];         // JSON格式数据
    
    CommunicationMessage() : type(MessageType::HEARTBEAT), 
                           sequence(0), timestamp(0), data_length(0) {
        memset(data, 0, sizeof(data));
    }
};

// PLC连接状态
enum class PLCConnectionStatus {
    DISCONNECTED = 0,
    CONNECTING = 1,
    CONNECTED = 2,
    ERROR = 3,
    TIMEOUT = 4
};

// 系统状态数据
struct SystemStatusData {
    PLCConnectionStatus plc_status;
    uint32_t heartbeat_counter;
    uint32_t plc_response_time_ms;
    bool emergency_stop;
    std::string last_error;
    uint64_t uptime_seconds;
    
    // Modbus寄存器数据 (基于PLC.md扩展)
    struct ModbusData {
        uint16_t system_status;      // 40001 - 系统状态
        uint16_t plc_command;        // 40002 - PLC命令
        uint16_t coord_ready;        // 40003 - 坐标就绪标志
        uint32_t x_coordinate;       // 40004-40005 - X坐标
        uint16_t cut_quality;        // 40006 - 切割质量
        uint32_t heartbeat;          // 40007-40008 - 心跳计数器
        uint16_t blade_number;       // 40009 - 刀片编号
        uint16_t system_health;      // 40010 - 系统健康状态
        
        // 新增寄存器 (基于PLC.md)
        uint16_t waste_detection;    // 40011 - 废料检测状态
        uint16_t rail_direction;     // 40014 - 导轨方向
        uint32_t remaining_length;   // 40015-40016 - 剩余长度
        uint16_t coverage_rate;      // 40017 - 覆盖率
        uint16_t detection_status;   // 40018 - 检测状态
        uint16_t process_mode;       // 40019 - 处理模式
    } modbus_data;
};

// 客户端连接信息
struct ClientConnection {
    int socket_fd;
    std::string client_name;
    uint64_t connect_time;
    uint64_t last_heartbeat;
    uint32_t message_count;
    bool is_active;
};

// 事件回调函数类型
using MessageCallback = std::function<void(const CommunicationMessage& msg, int client_fd)>;
using ClientConnectedCallback = std::function<void(int client_fd, const std::string& client_name)>;
using ClientDisconnectedCallback = std::function<void(int client_fd)>;

/**
 * @brief UNIX Domain Socket服务器类
 * 
 * 负责前后端通信，提供：
 * - 多客户端连接支持
 * - 消息广播和单播
 * - 心跳监控
 * - 连接状态管理
 * - JSON格式数据传输
 */
class UnixSocketServer {
public:
    UnixSocketServer(const std::string& socket_path);
    ~UnixSocketServer();

    // 禁用拷贝构造和赋值
    UnixSocketServer(const UnixSocketServer&) = delete;
    UnixSocketServer& operator=(const UnixSocketServer&) = delete;

    // 服务器控制
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // 消息发送
    bool send_message(int client_fd, const CommunicationMessage& msg);
    bool broadcast_message(const CommunicationMessage& msg);
    bool send_status_update(const SystemStatusData& status);
    bool send_modbus_data_update(const SystemStatusData::ModbusData& data);
    
    // 系统状态管理
    void update_system_status(const SystemStatusData& status);
    void update_plc_status(PLCConnectionStatus status, uint32_t response_time_ms = 0);
    void update_modbus_registers(const SystemStatusData::ModbusData& data);
    void set_emergency_stop(bool emergency);
    void set_last_error(const std::string& error);
    
    // 客户端管理
    std::vector<ClientConnection> get_connected_clients() const;
    int get_client_count() const;
    bool is_client_connected(int client_fd) const;
    void disconnect_client(int client_fd);
    
    // 事件回调设置
    void set_message_callback(MessageCallback callback);
    void set_client_connected_callback(ClientConnectedCallback callback);
    void set_client_disconnected_callback(ClientDisconnectedCallback callback);
    
    // 获取状态
    SystemStatusData get_system_status() const;
    std::string get_last_error() const;
    
    // 统计信息
    struct Statistics {
        uint64_t total_connections;
        uint64_t total_messages_sent;
        uint64_t total_messages_received;
        uint64_t active_clients;
        uint64_t uptime_seconds;
        std::chrono::steady_clock::time_point start_time;
    };
    Statistics get_statistics() const;

private:
    // 核心服务器功能
    void server_thread();
    void heartbeat_monitor_thread();
    bool handle_client_connection(int client_fd);
    bool process_message(const CommunicationMessage& msg, int client_fd);
    
    // 消息处理
    void handle_status_request(int client_fd, uint32_t sequence);
    void handle_plc_command(const json& command_data, int client_fd, uint32_t sequence);
    void handle_heartbeat(int client_fd);
    
    // 客户端管理
    void add_client(int client_fd);
    void remove_client(int client_fd);
    void cleanup_inactive_clients();
    
    // 辅助函数
    bool create_socket();
    bool bind_socket();
    void cleanup_socket();
    std::string get_current_timestamp() const;
    uint64_t get_unix_timestamp() const;
    
    // JSON序列化/反序列化
    json system_status_to_json(const SystemStatusData& status) const;
    json modbus_data_to_json(const SystemStatusData::ModbusData& data) const;
    
    // 配置
    std::string socket_path_;
    int server_socket_;
    
    // 线程管理
    std::atomic<bool> running_;
    std::thread server_thread_;
    std::thread heartbeat_thread_;
    
    // 客户端连接管理
    mutable std::mutex clients_mutex_;
    std::vector<ClientConnection> clients_;
    
    // 系统状态
    mutable std::mutex status_mutex_;
    SystemStatusData system_status_;
    
    // 事件回调
    MessageCallback message_callback_;
    ClientConnectedCallback client_connected_callback_;
    ClientDisconnectedCallback client_disconnected_callback_;
    
    // 统计信息
    mutable std::mutex stats_mutex_;
    Statistics statistics_;
    
    // 配置参数
    static constexpr int MAX_CLIENTS = 10;
    static constexpr int HEARTBEAT_INTERVAL_MS = 1000;
    static constexpr int CLIENT_TIMEOUT_MS = 5000;
    static constexpr int POLL_TIMEOUT_MS = 100;
};

} // namespace communication
} // namespace bamboo_cut