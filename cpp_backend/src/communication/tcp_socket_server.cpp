/**
 * TCP Socket服务器实现
 */

#include <bamboo_cut/communication/tcp_socket_server.h>
#include <bamboo_cut/core/types.h>
#include <bamboo_cut/core/logger.h>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace bamboo_cut {
namespace communication {

TcpSocketServer::TcpSocketServer(const std::string& bind_address, uint16_t port)
    : bind_address_(bind_address), port_(port), server_socket_(-1) {
    LOG_INFO("创建TCP Socket服务器: {}:{}", bind_address_, port_);
}

TcpSocketServer::~TcpSocketServer() {
    stop();
}

bool TcpSocketServer::start() {
    if (running_) {
        LOG_WARN("TCP Socket服务器已在运行");
        return true;
    }

    if (!create_server_socket()) {
        LOG_ERROR("创建服务器socket失败");
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread(&TcpSocketServer::accept_loop, this);

    LOG_INFO("TCP Socket服务器启动成功，监听 {}:{}", bind_address_, port_);
    return true;
}

void TcpSocketServer::stop() {
    if (!running_) {
        return;
    }

    LOG_INFO("开始停止TCP Socket服务器...");
    running_ = false;

    // 断开所有客户端连接
    disconnect_all_clients();

    // 关闭服务器socket，这将导致accept()调用返回
    if (server_socket_ != -1) {
        shutdown(server_socket_, SHUT_RDWR);  // 优雅关闭
        close(server_socket_);
        server_socket_ = -1;
    }

    // 等待accept线程结束（带超时）
    if (accept_thread_.joinable()) {
        // 使用条件变量或超时机制避免无限等待
        auto start_time = std::chrono::steady_clock::now();
        while (accept_thread_.joinable()) {
            accept_thread_.join();
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(2)) {
                LOG_WARN("Accept线程等待超时，强制结束");
                break;
            }
        }
    }

    // 等待所有客户端处理线程结束（带超时）
    auto thread_start_time = std::chrono::steady_clock::now();
    for (auto& thread : client_threads_) {
        if (thread.joinable()) {
            thread.join();
            auto elapsed = std::chrono::steady_clock::now() - thread_start_time;
            if (elapsed > std::chrono::seconds(3)) {
                LOG_WARN("客户端线程等待超时，继续下一个");
                break;
            }
        }
    }
    client_threads_.clear();

    LOG_INFO("TCP Socket服务器已完全停止");
}

bool TcpSocketServer::create_server_socket() {
    // 创建socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
        LOG_ERROR("创建socket失败: {}", strerror(errno));
        return false;
    }

    // 设置SO_REUSEADDR选项
    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        LOG_WARN("设置SO_REUSEADDR失败: {}", strerror(errno));
    }

    // 绑定地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);
    
    if (bind_address_ == "0.0.0.0" || bind_address_ == "*") {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, bind_address_.c_str(), &server_addr.sin_addr) <= 0) {
            LOG_ERROR("无效的绑定地址: {}", bind_address_);
            close(server_socket_);
            return false;
        }
    }

    if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        LOG_ERROR("绑定地址失败: {}", strerror(errno));
        close(server_socket_);
        return false;
    }

    // 开始监听
    if (listen(server_socket_, 10) == -1) {
        LOG_ERROR("监听失败: {}", strerror(errno));
        close(server_socket_);
        return false;
    }

    return true;
}

void TcpSocketServer::accept_loop() {
    LOG_INFO("开始接受客户端连接");

    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_addr, &client_addr_len);
        
        if (client_socket == -1) {
            if (running_) {  // 只有在服务器运行时才记录错误
                LOG_ERROR("接受连接失败: {}", strerror(errno));
                update_stats_error();
            }
            continue;
        }

        // 设置非阻塞模式
        int flags = fcntl(client_socket, F_GETFL, 0);
        fcntl(client_socket, F_SETFL, flags | O_NONBLOCK);

        // 创建客户端连接对象
        auto client = std::make_shared<ClientConnection>();
        client->socket_fd = client_socket;
        client->client_ip = inet_ntoa(client_addr.sin_addr);
        client->client_port = ntohs(client_addr.sin_port);
        client->connect_time = std::chrono::steady_clock::now();
        client->last_activity = client->connect_time;

        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients_.push_back(client);
        }

        update_stats_connection();

        LOG_INFO("客户端连接: {}:{} (fd={})", client->client_ip, client->client_port, client_socket);

        if (client_connected_callback_) {
            std::string client_info = client->client_ip + ":" + std::to_string(client->client_port);
            client_connected_callback_(client_socket, client_info);
        }

        // 启动客户端处理线程
        client_threads_.emplace_back(&TcpSocketServer::handle_client, this, client);
    }
}

void TcpSocketServer::handle_client(std::shared_ptr<ClientConnection> client) {
    LOG_DEBUG("开始处理客户端: fd={}", client->socket_fd);

    while (running_ && client->is_active) {
        CommunicationMessage message;
        
        if (receive_message(client->socket_fd, message)) {
            client->last_activity = std::chrono::steady_clock::now();
            client->messages_received++;
            update_stats_message_received(sizeof(message));

            if (message_callback_) {
                try {
                    message_callback_(message, client->socket_fd);
                } catch (const std::exception& e) {
                    LOG_ERROR("消息回调异常: {}", e.what());
                }
            }
        } else {
            // 接收失败，可能是连接断开
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 检查连接超时
        auto now = std::chrono::steady_clock::now();
        auto inactive_duration = std::chrono::duration_cast<std::chrono::seconds>(now - client->last_activity);
        if (inactive_duration.count() > 300) {  // 5分钟超时
            LOG_WARN("客户端超时断开: fd={}", client->socket_fd);
            break;
        }
    }

    cleanup_client(client->socket_fd);
}

bool TcpSocketServer::receive_message(int client_fd, CommunicationMessage& message) {
    // 清零消息结构体，避免垃圾数据
    memset(&message, 0, sizeof(message));
    
    // 首先尝试接收消息头（type, timestamp, data_length）
    struct MessageHeader {
        MessageType type;
        uint64_t timestamp;
        uint32_t data_length;
    } header;
    
    if (!receive_raw_data(client_fd, &header, sizeof(header))) {
        return false;
    }
    
    // 验证消息类型是否有效
    uint16_t type_value = static_cast<uint16_t>(header.type);
    if (type_value < 1 || type_value > 8) {
        LOG_WARN("收到无效消息类型: {} (原始值: {})", type_value, type_value);
        return false;
    }
    
    // 验证数据长度是否合理
    if (header.data_length > sizeof(message.data)) {
        LOG_WARN("数据长度过大: {} > {}", header.data_length, sizeof(message.data));
        return false;
    }
    
    // 复制验证过的头部信息
    message.type = header.type;
    message.timestamp = header.timestamp;
    message.data_length = header.data_length;
    
    // 如果有数据，接收数据部分
    if (header.data_length > 0) {
        if (!receive_raw_data(client_fd, message.data, header.data_length)) {
            LOG_WARN("接收消息数据失败");
            return false;
        }
        // 确保数据以null结尾
        message.data[header.data_length] = '\0';
    }

    // 检查是否是JSON字符串格式（兼容性处理）
    if (message.type == MessageType::JSON_STRING && header.data_length > 0) {
        std::string json_str(message.data, header.data_length);
        return handle_json_message(json_str, client_fd);
    }

    return true;
}

bool TcpSocketServer::handle_json_message(const std::string& json_str, int client_fd) {
    try {
        auto json_data = nlohmann::json::parse(json_str);
        
        CommunicationMessage msg;
        msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (json_data.contains("type")) {
            std::string type_str = json_data["type"];
            if (type_str == "status_request") {
                msg.type = MessageType::STATUS_REQUEST;
            } else if (type_str == "plc_command") {
                msg.type = MessageType::PLC_COMMAND;
            } else {
                msg.type = MessageType::STATUS_REQUEST; // 默认类型
            }
        }

        // 将JSON数据复制到消息的data字段
        strncpy(msg.data, json_str.c_str(), sizeof(msg.data) - 1);
        msg.data[sizeof(msg.data) - 1] = '\0';
        msg.data_length = json_str.length();

        if (message_callback_) {
            message_callback_(msg, client_fd);
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("解析JSON消息失败: {}", e.what());
        return false;
    }
}

bool TcpSocketServer::send_message(int client_fd, const CommunicationMessage& message) {
    if (send_raw_data(client_fd, &message, sizeof(message))) {
        update_stats_message_sent(sizeof(message));
        
        // 更新客户端统计
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& client : clients_) {
            if (client->socket_fd == client_fd) {
                client->messages_sent++;
                client->last_activity = std::chrono::steady_clock::now();
                break;
            }
        }
        return true;
    }
    return false;
}

bool TcpSocketServer::broadcast_message(const CommunicationMessage& message) {
    bool success = true;
    std::vector<int> active_fds;
    
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (const auto& client : clients_) {
            if (client->is_active) {
                active_fds.push_back(client->socket_fd);
            }
        }
    }

    for (int fd : active_fds) {
        if (!send_message(fd, message)) {
            success = false;
        }
    }

    return success;
}

bool TcpSocketServer::send_raw_data(int client_fd, const void* data, size_t size) {
    size_t bytes_sent = 0;
    const char* buffer = static_cast<const char*>(data);

    while (bytes_sent < size) {
        ssize_t result = send(client_fd, buffer + bytes_sent, size - bytes_sent, MSG_NOSIGNAL);
        
        if (result == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            LOG_ERROR("发送数据失败: {}", strerror(errno));
            return false;
        }
        
        bytes_sent += result;
    }

    return true;
}

bool TcpSocketServer::receive_raw_data(int client_fd, void* data, size_t size) {
    size_t bytes_received = 0;
    char* buffer = static_cast<char*>(data);

    while (bytes_received < size) {
        ssize_t result = recv(client_fd, buffer + bytes_received, size - bytes_received, 0);
        
        if (result == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            LOG_DEBUG("接收数据失败: {}", strerror(errno));
            return false;
        }
        
        if (result == 0) {
            // 连接已关闭
            return false;
        }
        
        bytes_received += result;
    }

    return true;
}

void TcpSocketServer::cleanup_client(int client_fd) {
    LOG_INFO("清理客户端连接: fd={}", client_fd);

    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients_.erase(
            std::remove_if(clients_.begin(), clients_.end(),
                [client_fd](const std::shared_ptr<ClientConnection>& client) {
                    if (client->socket_fd == client_fd) {
                        client->is_active = false;
                        return true;
                    }
                    return false;
                }), 
            clients_.end());
    }

    close(client_fd);
    update_stats_disconnection();

    if (client_disconnected_callback_) {
        client_disconnected_callback_(client_fd);
    }
}

void TcpSocketServer::disconnect_client(int client_fd) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& client : clients_) {
        if (client->socket_fd == client_fd) {
            client->is_active = false;
            break;
        }
    }
}

void TcpSocketServer::disconnect_all_clients() {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& client : clients_) {
        client->is_active = false;
        close(client->socket_fd);
    }
    clients_.clear();
}

TcpServerStats TcpSocketServer::get_statistics() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(stats_mutex_));
    TcpServerStats stats = stats_;
    
    // 更新活跃客户端数量
    {
        std::lock_guard<std::mutex> clients_lock(const_cast<std::mutex&>(clients_mutex_));
        stats.active_clients = clients_.size();
    }
    
    return stats;
}

std::vector<std::string> TcpSocketServer::get_connected_clients() const {
    std::vector<std::string> clients;
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(clients_mutex_));
    
    for (const auto& client : clients_) {
        if (client->is_active) {
            clients.push_back(client->client_ip + ":" + std::to_string(client->client_port));
        }
    }
    
    return clients;
}

void TcpSocketServer::update_stats_connection() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_connections++;
}

void TcpSocketServer::update_stats_disconnection() {
    // active_clients在get_statistics中动态计算
}

void TcpSocketServer::update_stats_message_sent(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_messages_sent++;
    stats_.total_bytes_sent += bytes;
}

void TcpSocketServer::update_stats_message_received(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_messages_received++;
    stats_.total_bytes_received += bytes;
}

void TcpSocketServer::update_stats_error() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.connection_errors++;
}

} // namespace communication
} // namespace bamboo_cut