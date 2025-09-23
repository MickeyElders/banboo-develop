/**
 * TCP Socket服务器
 * 用于前后端通信，替代UNIX Domain Socket
 */

#pragma once

#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <functional>
#include <string>
#include <memory>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace bamboo_cut {
namespace communication {

// 前向声明
struct CommunicationMessage;

// 统计信息结构
struct TcpServerStats {
    uint64_t total_connections = 0;
    uint64_t active_clients = 0;
    uint64_t total_messages_sent = 0;
    uint64_t total_messages_received = 0;
    uint64_t total_bytes_sent = 0;
    uint64_t total_bytes_received = 0;
    uint64_t connection_errors = 0;
};

// 客户端连接信息
struct ClientConnection {
    int socket_fd;
    std::string client_ip;
    uint16_t client_port;
    std::chrono::steady_clock::time_point connect_time;
    std::chrono::steady_clock::time_point last_activity;
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    bool is_active = true;
};

class TcpSocketServer {
public:
    // 回调函数类型定义
    using MessageCallback = std::function<void(const CommunicationMessage& message, int client_fd)>;
    using ClientConnectedCallback = std::function<void(int client_fd, const std::string& client_info)>;
    using ClientDisconnectedCallback = std::function<void(int client_fd)>;
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    explicit TcpSocketServer(const std::string& bind_address = "127.0.0.1", uint16_t port = 8888);
    ~TcpSocketServer();

    // 禁用复制构造和赋值
    TcpSocketServer(const TcpSocketServer&) = delete;
    TcpSocketServer& operator=(const TcpSocketServer&) = delete;

    // 服务器控制
    bool start();
    void stop();
    bool is_running() const { return running_; }

    // 消息发送
    bool send_message(int client_fd, const CommunicationMessage& message);
    bool broadcast_message(const CommunicationMessage& message);

    // 回调函数设置
    void set_message_callback(MessageCallback callback) { message_callback_ = std::move(callback); }
    void set_client_connected_callback(ClientConnectedCallback callback) { client_connected_callback_ = std::move(callback); }
    void set_client_disconnected_callback(ClientDisconnectedCallback callback) { client_disconnected_callback_ = std::move(callback); }
    void set_error_callback(ErrorCallback callback) { error_callback_ = std::move(callback); }

    // 统计信息
    TcpServerStats get_statistics() const;
    std::vector<std::string> get_connected_clients() const;

    // 客户端管理
    void disconnect_client(int client_fd);
    void disconnect_all_clients();

private:
    // 网络配置
    std::string bind_address_;
    uint16_t port_;
    int server_socket_;

    // 线程管理
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> client_threads_;

    // 客户端管理
    std::mutex clients_mutex_;
    std::vector<std::shared_ptr<ClientConnection>> clients_;

    // 回调函数
    MessageCallback message_callback_;
    ClientConnectedCallback client_connected_callback_;
    ClientDisconnectedCallback client_disconnected_callback_;
    ErrorCallback error_callback_;

    // 统计信息
    mutable std::mutex stats_mutex_;
    TcpServerStats stats_;

    // 内部方法
    bool create_server_socket();
    void accept_loop();
    void handle_client(std::shared_ptr<ClientConnection> client);
    void cleanup_client(int client_fd);
    void cleanup_disconnected_clients();
    
    // 消息处理
    bool receive_message(int client_fd, CommunicationMessage& message);
    bool send_raw_data(int client_fd, const void* data, size_t size);
    bool receive_raw_data(int client_fd, void* data, size_t size);

    // JSON消息处理（兼容字符串格式）
    bool handle_json_message(const std::string& json_str, int client_fd);

    // 统计更新
    void update_stats_connection();
    void update_stats_disconnection();
    void update_stats_message_sent(size_t bytes);
    void update_stats_message_received(size_t bytes);
    void update_stats_error();
};

} // namespace communication
} // namespace bamboo_cut