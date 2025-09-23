/**
 * TCP Socket客户端
 * 用于前端与后端通信，替代UNIX Domain Socket
 */

#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// 前向声明 - 避免包含整个types.h
struct CommunicationMessage;

// 连接状态枚举
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    CONNECTION_ERROR,
    RECONNECTING
};

// 统计信息结构
struct TcpClientStats {
    uint64_t total_connections = 0;
    uint64_t total_reconnections = 0;
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t connection_errors = 0;
    std::chrono::steady_clock::time_point last_connect_time;
    std::chrono::steady_clock::time_point last_message_time;
};

class TcpSocketClient {
public:
    // 回调函数类型定义
    using MessageCallback = std::function<void(const CommunicationMessage& message)>;
    using ConnectionCallback = std::function<void(ConnectionStatus status)>;
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    explicit TcpSocketClient(const std::string& server_address = "127.0.0.1", uint16_t port = 8888);
    ~TcpSocketClient();

    // 禁用复制构造和赋值
    TcpSocketClient(const TcpSocketClient&) = delete;
    TcpSocketClient& operator=(const TcpSocketClient&) = delete;

    // 连接控制
    bool connect();
    void disconnect();
    bool is_connected() const { return status_ == ConnectionStatus::CONNECTED; }
    ConnectionStatus get_status() const { return status_; }

    // 自动重连控制
    void enable_auto_reconnect(bool enable = true) { auto_reconnect_enabled_ = enable; }
    void set_reconnect_interval(int seconds) { reconnect_interval_seconds_ = seconds; }

    // 消息发送
    bool send_message(const CommunicationMessage& message);
    bool send_json_message(const std::string& json_str);

    // 回调函数设置
    void set_message_callback(MessageCallback callback) { message_callback_ = std::move(callback); }
    void set_connection_callback(ConnectionCallback callback) { connection_callback_ = std::move(callback); }
    void set_error_callback(ErrorCallback callback) { error_callback_ = std::move(callback); }

    // 统计信息
    TcpClientStats get_statistics() const;
    std::string get_server_info() const;

private:
    // 网络配置
    std::string server_address_;
    uint16_t server_port_;
    int client_socket_;

    // 连接状态
    std::atomic<ConnectionStatus> status_{ConnectionStatus::DISCONNECTED};
    std::atomic<bool> running_{false};

    // 重连配置
    std::atomic<bool> auto_reconnect_enabled_{true};
    std::atomic<int> reconnect_interval_seconds_{5};

    // 线程管理
    std::thread receive_thread_;
    std::thread reconnect_thread_;

    // 回调函数
    MessageCallback message_callback_;
    ConnectionCallback connection_callback_;
    ErrorCallback error_callback_;

    // 统计信息
    mutable std::mutex stats_mutex_;
    TcpClientStats stats_;

    // 内部方法
    bool create_socket();
    void close_socket();
    bool connect_to_server();
    void receive_loop();
    void reconnect_loop();
    
    // 消息处理
    bool receive_message(CommunicationMessage& message);
    bool send_raw_data(const void* data, size_t size);
    bool receive_raw_data(void* data, size_t size);

    // JSON消息处理（兼容性支持）
    bool handle_json_message(const std::string& json_str);

    // 状态更新
    void update_status(ConnectionStatus new_status);
    void trigger_error(const std::string& error_msg);

    // 统计更新
    void update_stats_connection();
    void update_stats_reconnection();
    void update_stats_message_sent(size_t bytes);
    void update_stats_message_received(size_t bytes);
    void update_stats_error();
};