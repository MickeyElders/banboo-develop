#include "bamboo_cut/communication/unix_socket_server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>

namespace bamboo_cut {
namespace communication {

UnixSocketServer::UnixSocketServer(const std::string& socket_path)
    : socket_path_(socket_path)
    , server_socket_(-1)
    , running_(false)
{
    // 初始化统计信息
    statistics_.total_connections = 0;
    statistics_.total_messages_sent = 0;
    statistics_.total_messages_received = 0;
    statistics_.active_clients = 0;
    statistics_.uptime_seconds = 0;
    statistics_.start_time = std::chrono::steady_clock::now();
    
    // 初始化系统状态
    system_status_.plc_status = PLCConnectionStatus::DISCONNECTED;
    system_status_.heartbeat_counter = 0;
    system_status_.plc_response_time_ms = 0;
    system_status_.emergency_stop = false;
    system_status_.uptime_seconds = 0;
    
    // 初始化Modbus数据
    memset(&system_status_.modbus_data, 0, sizeof(system_status_.modbus_data));
}

UnixSocketServer::~UnixSocketServer() {
    stop();
}

bool UnixSocketServer::start() {
    if (running_.load()) {
        std::cerr << "服务器已在运行中" << std::endl;
        return false;
    }

    // 创建和绑定Socket
    if (!create_socket() || !bind_socket()) {
        cleanup_socket();
        return false;
    }

    // 设置为监听模式
    if (listen(server_socket_, MAX_CLIENTS) == -1) {
        std::cerr << "监听失败: " << strerror(errno) << std::endl;
        cleanup_socket();
        return false;
    }

    running_.store(true);
    
    // 启动线程
    server_thread_ = std::thread(&UnixSocketServer::server_thread, this);
    heartbeat_thread_ = std::thread(&UnixSocketServer::heartbeat_monitor_thread, this);
    
    std::cout << "UNIX Domain Socket服务器启动成功: " << socket_path_ << std::endl;
    return true;
}

void UnixSocketServer::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    // 关闭所有客户端连接
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& client : clients_) {
            if (client.socket_fd >= 0) {
                close(client.socket_fd);
            }
        }
        clients_.clear();
    }

    // 等待线程结束
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }

    cleanup_socket();
    std::cout << "UNIX Domain Socket服务器已停止" << std::endl;
}

bool UnixSocketServer::create_socket() {
    server_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
        std::cerr << "创建Socket失败: " << strerror(errno) << std::endl;
        return false;
    }

    // 设置为非阻塞模式
    int flags = fcntl(server_socket_, F_GETFL, 0);
    if (flags == -1) {
        std::cerr << "获取Socket标志失败: " << strerror(errno) << std::endl;
        return false;
    }
    
    if (fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK) == -1) {
        std::cerr << "设置非阻塞模式失败: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

bool UnixSocketServer::bind_socket() {
    // 删除现有的socket文件
    unlink(socket_path_.c_str());

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_socket_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cerr << "绑定Socket失败: " << strerror(errno) << std::endl;
        return false;
    }

    // 设置Socket文件权限
    chmod(socket_path_.c_str(), 0666);
    return true;
}

void UnixSocketServer::cleanup_socket() {
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    // 删除socket文件
    unlink(socket_path_.c_str());
}

void UnixSocketServer::server_thread() {
    std::cout << "Socket服务器线程启动" << std::endl;
    
    while (running_.load()) {
        // 使用poll检查新连接
        struct pollfd pfd;
        pfd.fd = server_socket_;
        pfd.events = POLLIN;
        
        int poll_result = poll(&pfd, 1, POLL_TIMEOUT_MS);
        if (poll_result == -1) {
            if (errno != EINTR && running_.load()) {
                std::cerr << "Poll失败: " << strerror(errno) << std::endl;
            }
            continue;
        }
        
        if (poll_result > 0 && (pfd.revents & POLLIN)) {
            // 接受新连接
            int client_fd = accept(server_socket_, nullptr, nullptr);
            if (client_fd >= 0) {
                // 设置客户端Socket为非阻塞
                int flags = fcntl(client_fd, F_GETFL, 0);
                fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
                
                add_client(client_fd);
                
                std::cout << "新客户端连接: fd=" << client_fd << std::endl;
                
                if (client_connected_callback_) {
                    client_connected_callback_(client_fd, "frontend_client");
                }
                
                // 更新统计信息
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    statistics_.total_connections++;
                }
            }
        }
        
        // 处理已连接客户端的消息
        std::vector<int> clients_to_remove;
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            for (auto& client : clients_) {
                if (!client.is_active) continue;
                
                if (!handle_client_connection(client.socket_fd)) {
                    clients_to_remove.push_back(client.socket_fd);
                }
            }
        }
        
        // 移除断开的客户端
        for (int client_fd : clients_to_remove) {
            remove_client(client_fd);
        }
    }
    
    std::cout << "Socket服务器线程退出" << std::endl;
}

bool UnixSocketServer::handle_client_connection(int client_fd) {
    CommunicationMessage msg;
    
    // 使用poll检查是否有数据可读
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    
    int poll_result = poll(&pfd, 1, 0);
    if (poll_result <= 0) {
        return true; // 没有数据，但连接正常
    }
    
    ssize_t bytes_read = recv(client_fd, &msg, sizeof(msg), 0);
    if (bytes_read == sizeof(msg)) {
        // 处理消息
        if (process_message(msg, client_fd)) {
            // 更新统计信息
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                statistics_.total_messages_received++;
            }
            
            // 更新客户端最后心跳时间
            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                for (auto& client : clients_) {
                    if (client.socket_fd == client_fd) {
                        client.last_heartbeat = get_unix_timestamp();
                        client.message_count++;
                        break;
                    }
                }
            }
            
            return true;
        }
    } else if (bytes_read == 0) {
        // 客户端断开连接
        std::cout << "客户端断开连接: fd=" << client_fd << std::endl;
        return false;
    } else if (bytes_read == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return true; // 非阻塞模式，没有数据
        } else {
            std::cerr << "接收消息失败: " << strerror(errno) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool UnixSocketServer::process_message(const CommunicationMessage& msg, int client_fd) {
    try {
        switch (msg.type) {
            case MessageType::STATUS_REQUEST:
                handle_status_request(client_fd, msg.sequence);
                break;
                
            case MessageType::PLC_COMMAND:
#ifdef ENABLE_JSON
                {
                    json command_data = json::parse(std::string(msg.data, msg.data_length));
                    handle_plc_command(command_data, client_fd, msg.sequence);
                }
#endif
                break;
                
            case MessageType::HEARTBEAT:
                handle_heartbeat(client_fd);
                break;
                
            default:
                std::cerr << "未知消息类型: " << static_cast<int>(msg.type) << std::endl;
                return false;
        }
        
        // 调用用户回调
        if (message_callback_) {
            message_callback_(msg, client_fd);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "处理消息异常: " << e.what() << std::endl;
        return false;
    }
}

void UnixSocketServer::handle_status_request(int client_fd, uint32_t sequence) {
    UnixSocketStatusData status;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status = system_status_;
        
        // 更新运行时间
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - statistics_.start_time);
        status.uptime_seconds = elapsed.count();
    }
    
    send_status_update(status);
}

void UnixSocketServer::handle_plc_command(const json& command_data, int client_fd, uint32_t sequence) {
#ifdef ENABLE_JSON
    // 处理PLC命令并发送确认
    CommunicationMessage ack_msg;
    ack_msg.type = MessageType::PLC_COMMAND_ACK;
    ack_msg.sequence = sequence;
    ack_msg.timestamp = get_unix_timestamp();
    
    json ack_data;
    ack_data["status"] = "received";
    ack_data["command"] = command_data;
    
    std::string ack_str = ack_data.dump();
    ack_msg.data_length = std::min(ack_str.length(), sizeof(ack_msg.data) - 1);
    strncpy(ack_msg.data, ack_str.c_str(), ack_msg.data_length);
    
    send_message(client_fd, ack_msg);
#endif
}

void UnixSocketServer::handle_heartbeat(int client_fd) {
    // 发送心跳响应
    CommunicationMessage heartbeat_msg;
    heartbeat_msg.type = MessageType::HEARTBEAT;
    heartbeat_msg.sequence = 0;
    heartbeat_msg.timestamp = get_unix_timestamp();
    heartbeat_msg.data_length = 0;
    
    send_message(client_fd, heartbeat_msg);
}

void UnixSocketServer::add_client(int client_fd) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    ClientConnection client;
    client.socket_fd = client_fd;
    client.client_name = "frontend_client_" + std::to_string(client_fd);
    client.connect_time = get_unix_timestamp();
    client.last_heartbeat = client.connect_time;
    client.message_count = 0;
    client.is_active = true;
    
    clients_.push_back(client);
}

void UnixSocketServer::remove_client(int client_fd) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    auto it = std::find_if(clients_.begin(), clients_.end(),
                          [client_fd](const ClientConnection& client) {
                              return client.socket_fd == client_fd;
                          });
    
    if (it != clients_.end()) {
        close(it->socket_fd);
        clients_.erase(it);
        
        std::cout << "客户端已移除: fd=" << client_fd << std::endl;
        
        if (client_disconnected_callback_) {
            client_disconnected_callback_(client_fd);
        }
    }
}

void UnixSocketServer::heartbeat_monitor_thread() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS));
        
        if (!running_.load()) break;
        
        cleanup_inactive_clients();
        
        // 更新心跳计数器
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            system_status_.heartbeat_counter++;
        }
    }
}

void UnixSocketServer::cleanup_inactive_clients() {
    uint64_t current_time = get_unix_timestamp();
    std::vector<int> inactive_clients;
    
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (const auto& client : clients_) {
            if (client.is_active && 
                (current_time - client.last_heartbeat) > CLIENT_TIMEOUT_MS / 1000) {
                inactive_clients.push_back(client.socket_fd);
            }
        }
    }
    
    for (int client_fd : inactive_clients) {
        std::cout << "清理不活跃客户端: fd=" << client_fd << std::endl;
        remove_client(client_fd);
    }
}

bool UnixSocketServer::send_message(int client_fd, const CommunicationMessage& msg) {
    ssize_t bytes_sent = send(client_fd, &msg, sizeof(msg), MSG_NOSIGNAL);
    if (bytes_sent == sizeof(msg)) {
        // 更新统计信息
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            statistics_.total_messages_sent++;
        }
        return true;
    } else {
        if (errno != EPIPE && errno != ECONNRESET) {
            std::cerr << "发送消息失败: " << strerror(errno) << std::endl;
        }
        return false;
    }
}

bool UnixSocketServer::broadcast_message(const CommunicationMessage& msg) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    bool success = true;
    
    for (const auto& client : clients_) {
        if (client.is_active) {
            if (!send_message(client.socket_fd, msg)) {
                success = false;
            }
        }
    }
    
    return success;
}

bool UnixSocketServer::send_status_update(const SystemStatusData& status) {
#ifdef ENABLE_JSON
    CommunicationMessage msg;
    msg.type = MessageType::STATUS_RESPONSE;
    msg.sequence = 0;
    msg.timestamp = get_unix_timestamp();
    
    json status_json = system_status_to_json(status);
    std::string status_str = status_json.dump();
    
    msg.data_length = std::min(status_str.length(), sizeof(msg.data) - 1);
    strncpy(msg.data, status_str.c_str(), msg.data_length);
    
    return broadcast_message(msg);
#else
    return false;
#endif
}

bool UnixSocketServer::send_modbus_data_update(const UnixSocketStatusData::ModbusData& data) {
#ifdef ENABLE_JSON
    CommunicationMessage msg;
    msg.type = MessageType::MODBUS_DATA;
    msg.sequence = 0;
    msg.timestamp = get_unix_timestamp();
    
    json modbus_json = modbus_data_to_json(data);
    std::string modbus_str = modbus_json.dump();
    
    msg.data_length = std::min(modbus_str.length(), sizeof(msg.data) - 1);
    strncpy(msg.data, modbus_str.c_str(), msg.data_length);
    
    return broadcast_message(msg);
#else
    return false;
#endif
}

void UnixSocketServer::update_system_status(const UnixSocketStatusData& status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    system_status_ = status;
}

void UnixSocketServer::update_plc_status(PLCConnectionStatus status, uint32_t response_time_ms) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    system_status_.plc_status = status;
    system_status_.plc_response_time_ms = response_time_ms;
}

void UnixSocketServer::update_modbus_registers(const UnixSocketStatusData::ModbusData& data) {
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        system_status_.modbus_data = data;
    }
    
    // 立即广播更新
    send_modbus_data_update(data);
}

void UnixSocketServer::set_emergency_stop(bool emergency) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    system_status_.emergency_stop = emergency;
}

void UnixSocketServer::set_last_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    system_status_.last_error = error;
}

UnixSocketStatusData UnixSocketServer::get_system_status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return system_status_;
}

std::string UnixSocketServer::get_last_error() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return system_status_.last_error;
}

std::vector<ClientConnection> UnixSocketServer::get_connected_clients() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return clients_;
}

int UnixSocketServer::get_client_count() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return clients_.size();
}

UnixSocketServer::Statistics UnixSocketServer::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Statistics stats = statistics_;
    
    // 更新活跃客户端数量
    {
        std::lock_guard<std::mutex> client_lock(clients_mutex_);
        stats.active_clients = clients_.size();
    }
    
    // 更新运行时间
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stats.start_time);
    stats.uptime_seconds = elapsed.count();
    
    return stats;
}

uint64_t UnixSocketServer::get_unix_timestamp() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string UnixSocketServer::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

#ifdef ENABLE_JSON
json UnixSocketServer::system_status_to_json(const UnixSocketStatusData& status) const {
    json j;
    j["plc_status"] = static_cast<int>(status.plc_status);
    j["heartbeat_counter"] = status.heartbeat_counter;
    j["plc_response_time_ms"] = status.plc_response_time_ms;
    j["emergency_stop"] = status.emergency_stop;
    j["last_error"] = status.last_error;
    j["uptime_seconds"] = status.uptime_seconds;
    j["modbus_data"] = modbus_data_to_json(status.modbus_data);
    j["timestamp"] = get_current_timestamp();
    
    return j;
}

json UnixSocketServer::modbus_data_to_json(const UnixSocketStatusData::ModbusData& data) const {
    json j;
    j["system_status"] = data.system_status;
    j["plc_command"] = data.plc_command;
    j["coord_ready"] = data.coord_ready;
    j["x_coordinate"] = data.x_coordinate;
    j["cut_quality"] = data.cut_quality;
    j["heartbeat"] = data.heartbeat;
    j["blade_number"] = data.blade_number;
    j["system_health"] = data.system_health;
    
    return j;
}
#endif

} // namespace communication
} // namespace bamboo_cut