/**
 * TCP Socket客户端实现
 */

#include "backend/tcp_socket_client.h"
#include "common/types.h"

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <iostream>

TcpSocketClient::TcpSocketClient(const std::string& server_address, uint16_t port)
    : server_address_(server_address), server_port_(port), client_socket_(-1) {
    printf("创建TCP Socket客户端: %s:%d\n", server_address_.c_str(), server_port_);
}

TcpSocketClient::~TcpSocketClient() {
    disconnect();
}

bool TcpSocketClient::connect() {
    if (status_ == ConnectionStatus::CONNECTED) {
        printf("TCP客户端已连接\n");
        return true;
    }

    if (status_ == ConnectionStatus::CONNECTING) {
        printf("TCP客户端正在连接中\n");
        return false;
    }

    update_status(ConnectionStatus::CONNECTING);

    if (!create_socket()) {
        update_status(ConnectionStatus::CONNECTION_ERROR);
        return false;
    }

    if (!connect_to_server()) {
        close_socket();
        update_status(ConnectionStatus::CONNECTION_ERROR);
        return false;
    }

    update_status(ConnectionStatus::CONNECTED);
    running_ = true;

    // 启动接收线程
    receive_thread_ = std::thread(&TcpSocketClient::receive_loop, this);

    // 启动重连线程（如果启用自动重连）
    if (auto_reconnect_enabled_) {
        reconnect_thread_ = std::thread(&TcpSocketClient::reconnect_loop, this);
    }

    update_stats_connection();
    printf("TCP客户端连接成功: %s:%d\n", server_address_.c_str(), server_port_);
    return true;
}

void TcpSocketClient::disconnect() {
    if (status_ == ConnectionStatus::DISCONNECTED) {
        return;
    }

    running_ = false;
    update_status(ConnectionStatus::DISCONNECTED);

    close_socket();

    // 等待线程结束
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
    if (reconnect_thread_.joinable()) {
        reconnect_thread_.join();
    }

    printf("TCP客户端已断开连接\n");
}

bool TcpSocketClient::create_socket() {
    client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket_ == -1) {
        trigger_error("创建socket失败: " + std::string(strerror(errno)));
        return false;
    }

    // 设置非阻塞模式
    int flags = fcntl(client_socket_, F_GETFL, 0);
    fcntl(client_socket_, F_SETFL, flags | O_NONBLOCK);

    return true;
}

void TcpSocketClient::close_socket() {
    if (client_socket_ != -1) {
        close(client_socket_);
        client_socket_ = -1;
    }
}

bool TcpSocketClient::connect_to_server() {
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port_);

    if (inet_pton(AF_INET, server_address_.c_str(), &server_addr.sin_addr) <= 0) {
        trigger_error("无效的服务器地址: " + server_address_);
        return false;
    }

    int result = ::connect(client_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    if (result == -1) {
        if (errno == EINPROGRESS) {
            // 非阻塞连接，需要等待完成
            fd_set write_fds;
            FD_ZERO(&write_fds);
            FD_SET(client_socket_, &write_fds);

            struct timeval timeout;
            timeout.tv_sec = 5;  // 5秒超时
            timeout.tv_usec = 0;

            int select_result = select(client_socket_ + 1, nullptr, &write_fds, nullptr, &timeout);
            if (select_result <= 0) {
                trigger_error("连接超时或失败");
                return false;
            }

            // 检查连接是否成功
            int error = 0;
            socklen_t len = sizeof(error);
            if (getsockopt(client_socket_, SOL_SOCKET, SO_ERROR, &error, &len) == -1 || error != 0) {
                trigger_error("连接失败: " + std::string(strerror(error)));
                return false;
            }
        } else {
            trigger_error("连接失败: " + std::string(strerror(errno)));
            return false;
        }
    }

    return true;
}

void TcpSocketClient::receive_loop() {
    printf("开始TCP客户端接收循环\n");

    while (running_ && status_ == ConnectionStatus::CONNECTED) {
        CommunicationMessage message;
        
        if (receive_message(message)) {
            update_stats_message_received(sizeof(message));
            
            if (message_callback_) {
                try {
                    message_callback_(message);
                } catch (const std::exception& e) {
                    printf("消息回调异常: %s\n", e.what());
                }
            }
        } else {
            // 接收失败，可能是连接断开
            if (running_) {
                update_status(ConnectionStatus::CONNECTION_ERROR);
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    printf("TCP客户端接收循环结束\n");
}

void TcpSocketClient::reconnect_loop() {
    while (running_) {
        if (status_ == ConnectionStatus::CONNECTION_ERROR && auto_reconnect_enabled_) {
            printf("尝试重新连接...\n");
            update_status(ConnectionStatus::RECONNECTING);
            
            std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval_seconds_));
            
            if (running_) {
                close_socket();
                if (create_socket() && connect_to_server()) {
                    update_status(ConnectionStatus::CONNECTED);
                    update_stats_reconnection();
                    printf("重新连接成功\n");
                } else {
                    update_status(ConnectionStatus::CONNECTION_ERROR);
                    update_stats_error();
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

bool TcpSocketClient::send_message(const CommunicationMessage& message) {
    if (status_ != ConnectionStatus::CONNECTED) {
        return false;
    }

    if (send_raw_data(&message, sizeof(message))) {
        update_stats_message_sent(sizeof(message));
        return true;
    }
    
    // 发送失败，可能连接断开
    update_status(ConnectionStatus::CONNECTION_ERROR);
    return false;
}

bool TcpSocketClient::send_json_message(const std::string& json_str) {
    CommunicationMessage message;
    message.type = MessageType::JSON_STRING;
    message.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    strncpy(message.data, json_str.c_str(), sizeof(message.data) - 1);
    message.data[sizeof(message.data) - 1] = '\0';
    message.data_length = json_str.length();

    return send_message(message);
}

bool TcpSocketClient::receive_message(CommunicationMessage& message) {
    return receive_raw_data(&message, sizeof(message));
}

bool TcpSocketClient::send_raw_data(const void* data, size_t size) {
    if (client_socket_ == -1) {
        return false;
    }

    size_t bytes_sent = 0;
    const char* buffer = static_cast<const char*>(data);

    while (bytes_sent < size) {
        ssize_t result = send(client_socket_, buffer + bytes_sent, size - bytes_sent, MSG_NOSIGNAL);
        
        if (result == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            printf("发送数据失败: %s\n", strerror(errno));
            return false;
        }
        
        bytes_sent += result;
    }

    return true;
}

bool TcpSocketClient::receive_raw_data(void* data, size_t size) {
    if (client_socket_ == -1) {
        return false;
    }

    size_t bytes_received = 0;
    char* buffer = static_cast<char*>(data);

    while (bytes_received < size) {
        ssize_t result = recv(client_socket_, buffer + bytes_received, size - bytes_received, 0);
        
        if (result == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
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

void TcpSocketClient::update_status(ConnectionStatus new_status) {
    ConnectionStatus old_status = status_.exchange(new_status);
    
    if (old_status != new_status && connection_callback_) {
        connection_callback_(new_status);
    }
}

void TcpSocketClient::trigger_error(const std::string& error_msg) {
    printf("TCP客户端错误: %s\n", error_msg.c_str());
    update_stats_error();
    
    if (error_callback_) {
        error_callback_(error_msg);
    }
}

TcpClientStats TcpSocketClient::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::string TcpSocketClient::get_server_info() const {
    return server_address_ + ":" + std::to_string(server_port_);
}

void TcpSocketClient::update_stats_connection() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_connections++;
    stats_.last_connect_time = std::chrono::steady_clock::now();
}

void TcpSocketClient::update_stats_reconnection() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_reconnections++;
    stats_.last_connect_time = std::chrono::steady_clock::now();
}

void TcpSocketClient::update_stats_message_sent(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_sent++;
    stats_.bytes_sent += bytes;
    stats_.last_message_time = std::chrono::steady_clock::now();
}

void TcpSocketClient::update_stats_message_received(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_received++;
    stats_.bytes_received += bytes;
    stats_.last_message_time = std::chrono::steady_clock::now();
}

void TcpSocketClient::update_stats_error() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.connection_errors++;
}