#include "bamboo_cut/communication/modbus_server.h"
#ifdef ENABLE_MODBUS
#include <modbus/modbus.h>
#endif
#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace bamboo_cut {
namespace communication {

ModbusServer::ModbusServer(const ModbusConfig& config)
    : config_(config)
    , last_heartbeat_time_(std::chrono::steady_clock::now())
{
    // 初始化统计信息
    statistics_.start_time = std::chrono::steady_clock::now();
    statistics_.last_request_time = statistics_.start_time;
    
    // 初始化寄存器为0
    std::memset(registers_, 0, sizeof(registers_));
    
    // 设置初始系统状态
    set_system_status(SystemStatus::STOPPED);
    set_system_health(SystemHealth::NORMAL);
}

ModbusServer::~ModbusServer() {
    stop();
}

bool ModbusServer::start() {
    if (running_.load()) {
        log_error("服务器已在运行中");
        return false;
    }

#ifdef ENABLE_MODBUS
    try {
        // 创建Modbus TCP上下文
        modbus_ctx_ = modbus_new_tcp(config_.ip_address.c_str(), config_.port);
        if (!modbus_ctx_) {
            log_error("无法创建Modbus上下文");
            return false;
        }

        // 设置响应超时
        modbus_set_response_timeout(modbus_ctx_, 
            config_.response_timeout_ms / 1000, 
            (config_.response_timeout_ms % 1000) * 1000);

        // 创建并绑定服务器套接字
        server_socket_ = modbus_tcp_listen(modbus_ctx_, config_.max_connections);
        if (server_socket_ == -1) {
            handle_modbus_error(modbus_ctx_, "监听失败");
            modbus_free(modbus_ctx_);
            modbus_ctx_ = nullptr;
            return false;
        }

        // 设置运行标志
        running_.store(true);
        
        // 启动线程
        server_thread_ = std::thread(&ModbusServer::server_thread, this);
        heartbeat_thread_ = std::thread(&ModbusServer::heartbeat_thread, this);
        timeout_monitor_thread_ = std::thread(&ModbusServer::timeout_monitor_thread, this);

        std::cout << "Modbus TCP服务器启动成功 - " 
                  << config_.ip_address << ":" << config_.port << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        log_error("启动服务器异常: " + std::string(e.what()));
        return false;
    }
#else
    log_error("Modbus功能未启用，请重新编译时启用ENABLE_MODBUS");
    return false;
#endif
}

void ModbusServer::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    client_connected_.store(false);

#ifdef ENABLE_MODBUS
    // 关闭服务器套接字
    if (server_socket_ != -1) {
        close(server_socket_);
        server_socket_ = -1;
    }

    // 等待线程结束
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
    if (timeout_monitor_thread_.joinable()) {
        timeout_monitor_thread_.join();
    }

    // 清理Modbus上下文
    if (modbus_ctx_) {
        modbus_free(modbus_ctx_);
        modbus_ctx_ = nullptr;
    }
#endif

    std::cout << "Modbus TCP服务器已停止" << std::endl;
}

void ModbusServer::server_thread() {
#ifdef ENABLE_MODBUS
    while (running_.load()) {
        try {
            // 等待客户端连接
            int client_socket = modbus_tcp_accept(modbus_ctx_, &server_socket_);
            if (client_socket == -1) {
                if (running_.load()) {
                    handle_modbus_error(modbus_ctx_, "接受连接失败");
                }
                continue;
            }

            // 创建客户端上下文
            modbus_t* client_ctx = modbus_new_tcp_pi("0.0.0.0", "502");
            if (client_ctx == nullptr) {
                close(client_socket);
                continue;
            }

            // 设置客户端套接字
            modbus_set_socket(client_ctx, client_socket);

            // 获取客户端IP
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            getpeername(client_socket, 
                       (struct sockaddr*)&client_addr, &addr_len);
            std::string client_ip = inet_ntoa(client_addr.sin_addr);

            std::cout << "客户端连接: " << client_ip << std::endl;
            
            // 更新统计信息
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                statistics_.total_connections++;
            }

            // 设置连接状态并触发回调
            client_connected_.store(true);
            if (connection_callback_) {
                connection_callback_(true, client_ip);
            }

            // 处理客户端连接
            bool connection_active = handle_client_connection(client_ctx);
            
            // 连接断开处理
            client_connected_.store(false);
            if (connection_callback_) {
                connection_callback_(false, client_ip);
            }

            modbus_close(client_ctx);
            modbus_free(client_ctx);
            
            std::cout << "客户端断开: " << client_ip << std::endl;
        }
        catch (const std::exception& e) {
            log_error("服务器线程异常: " + std::string(e.what()));
        }
    }
#else
    // 如果Modbus未启用，线程直接退出
    return;
#endif
}

bool ModbusServer::handle_client_connection(modbus_t* ctx) {
#ifdef ENABLE_MODBUS
    uint8_t query[MODBUS_TCP_MAX_ADU_LENGTH];
    
    while (running_.load() && client_connected_.load()) {
        try {
            // 接收Modbus请求
            int rc = modbus_receive(ctx, query);
            if (rc == -1) {
                // 检查是否为超时或连接断开
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue; // 超时，继续等待
                } else {
                    handle_modbus_error(ctx, "接收请求失败");
                    return false;
                }
            }

            // 更新统计信息
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                statistics_.total_requests++;
                statistics_.last_request_time = std::chrono::steady_clock::now();
            }

            // 更新寄存器数据
            update_registers();

            // 创建modbus映射结构
            modbus_mapping_t mb_mapping;
            mb_mapping.nb_registers = REG_COUNT;
            mb_mapping.tab_registers = registers_;

            // 处理Modbus请求并发送响应
            int reply_rc = modbus_reply(ctx, query, rc, &mb_mapping);
            if (reply_rc == -1) {
                handle_modbus_error(ctx, "发送响应失败");
                std::lock_guard<std::mutex> lock(stats_mutex_);
                statistics_.total_errors++;
            }

            // 检查PLC命令
            PLCCommand cmd = static_cast<PLCCommand>(read_int16_from_registers(REG_PLC_COMMAND - 40001));
            if (cmd != PLCCommand::NONE) {
                process_plc_command(cmd);
                // 清除命令寄存器
                write_int16_to_registers(REG_PLC_COMMAND - 40001, static_cast<uint16_t>(PLCCommand::NONE));
            }
        }
        catch (const std::exception& e) {
            log_error("处理客户端连接异常: " + std::string(e.what()));
            std::lock_guard<std::mutex> lock(stats_mutex_);
            statistics_.total_errors++;
        }
    }
    
    return true;
#else
    // 如果Modbus未启用，直接返回失败
    return false;
#endif
}

void ModbusServer::heartbeat_thread() {
    while (running_.load()) {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.heartbeat_interval_ms));
            
            if (client_connected_.load()) {
                // 更新心跳计数器
                uint32_t counter = heartbeat_counter_.fetch_add(1) + 1;
                
                // 更新心跳时间
                last_heartbeat_time_ = std::chrono::steady_clock::now();
                
                // 写入心跳寄存器
                std::lock_guard<std::mutex> lock(registers_mutex_);
                write_int32_to_registers(REG_HEARTBEAT - 40001, counter);
            }
        }
        catch (const std::exception& e) {
            log_error("心跳线程异常: " + std::string(e.what()));
        }
    }
}

void ModbusServer::timeout_monitor_thread() {
    while (running_.load()) {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(timeout_mutex_);
            
            // 检查进料检测超时
            if (feed_detection_active_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - feed_detection_start_).count();
                if (elapsed >= config_.feed_detection_timeout_s) {
                    feed_detection_active_ = false;
                    if (timeout_callback_) {
                        timeout_callback_("feed_detection_timeout");
                    }
                }
            }
            
            // 检查夹持超时
            if (clamp_active_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - clamp_start_).count();
                if (elapsed >= config_.clamp_timeout_s) {
                    clamp_active_ = false;
                    if (timeout_callback_) {
                        timeout_callback_("clamp_timeout");
                    }
                }
            }
            
            // 检查切割执行超时
            if (cut_execution_active_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - cut_execution_start_).count();
                if (elapsed >= config_.cut_execution_timeout_s) {
                    cut_execution_active_ = false;
                    if (timeout_callback_) {
                        timeout_callback_("cut_execution_timeout");
                    }
                }
            }
            
            // 检查心跳超时
            if (client_connected_.load()) {
                auto heartbeat_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_heartbeat_time_).count();
                if (heartbeat_elapsed > config_.heartbeat_interval_ms * 3) { // 3倍心跳间隔为超时
                    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                    statistics_.heartbeat_timeouts++;
                }
            }
        }
        catch (const std::exception& e) {
            log_error("超时监控线程异常: " + std::string(e.what()));
        }
    }
}

void ModbusServer::update_registers() {
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    std::lock_guard<std::mutex> reg_lock(registers_mutex_);
    
    // 更新系统状态寄存器
    write_int16_to_registers(REG_SYSTEM_STATUS - 40001, 
                            static_cast<uint16_t>(system_status_.status));
    
    // 更新坐标就绪标志
    write_int16_to_registers(REG_COORD_READY - 40001,
                            system_status_.coordinate_ready ? 1 : 0);
    
    // 更新X坐标（如果有坐标数据）
    if (system_status_.coordinate_ready) {
        write_int32_to_registers(REG_X_COORDINATE - 40001, 
                                static_cast<uint32_t>(current_coordinate_.x_coordinate));
        
        // 更新切割质量
        write_int16_to_registers(REG_CUT_QUALITY - 40001,
                                static_cast<uint16_t>(current_coordinate_.quality));
        
        // 更新刀片编号
        write_int16_to_registers(REG_BLADE_NUMBER - 40001,
                                static_cast<uint16_t>(current_coordinate_.blade_number));
    }
    
    // 更新系统健康状态
    write_int16_to_registers(REG_SYSTEM_HEALTH - 40001,
                            static_cast<uint16_t>(system_status_.health));
}

void ModbusServer::process_plc_command(PLCCommand command) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    system_status_.last_command = command;
    
    // 记录命令处理时间
    auto now = std::chrono::steady_clock::now();
    
    switch (command) {
        case PLCCommand::FEED_DETECTION:
            {
                std::lock_guard<std::mutex> timeout_lock(timeout_mutex_);
                feed_detection_start_ = now;
                feed_detection_active_ = true;
            }
            break;
            
        case PLCCommand::CUT_PREPARE:
            {
                std::lock_guard<std::mutex> timeout_lock(timeout_mutex_);
                clamp_start_ = now;
                clamp_active_ = true;
            }
            break;
            
        case PLCCommand::CUT_COMPLETE:
            {
                std::lock_guard<std::mutex> timeout_lock(timeout_mutex_);
                cut_execution_start_ = now;
                cut_execution_active_ = true;
                // 清除坐标数据
                clear_coordinate_data();
            }
            break;
            
        case PLCCommand::START_FEEDING:
            // 启动送料，重置相关超时计时器
            reset_feed_detection_timer();
            break;
            
        case PLCCommand::PAUSE:
            set_system_status(SystemStatus::PAUSED);
            break;
            
        case PLCCommand::EMERGENCY_STOP:
            trigger_emergency_stop();
            break;
            
        case PLCCommand::RESUME:
            if (system_status_.status == SystemStatus::PAUSED) {
                set_system_status(SystemStatus::RUNNING);
            }
            break;
            
        case PLCCommand::NONE:
        default:
            break;
    }
    
    // 触发命令回调
    if (command_callback_) {
        command_callback_(command);
    }
}

// 坐标数据管理
void ModbusServer::set_coordinate_data(const CoordinateData& data) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_coordinate_ = data;
    system_status_.coordinate_ready = true;
}

void ModbusServer::clear_coordinate_data() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_coordinate_ = CoordinateData{};
    system_status_.coordinate_ready = false;
}

CoordinateData ModbusServer::get_coordinate_data() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return current_coordinate_;
}

// 系统状态管理
void ModbusServer::set_system_status(SystemStatus status) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    system_status_.status = status;
}

void ModbusServer::set_system_health(SystemHealth health) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    system_status_.health = health;
}

SystemStatus ModbusServer::get_system_status() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return system_status_.status;
}

PLCCommand ModbusServer::get_last_plc_command() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return system_status_.last_command;
}

// 心跳和健康监控
uint32_t ModbusServer::get_heartbeat_counter() const {
    return heartbeat_counter_.load();
}

bool ModbusServer::is_heartbeat_active() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_heartbeat_time_).count();
    return elapsed <= config_.heartbeat_interval_ms * 2;
}

void ModbusServer::reset_heartbeat() {
    heartbeat_counter_.store(0);
    last_heartbeat_time_ = std::chrono::steady_clock::now();
}

// 安全机制
void ModbusServer::trigger_emergency_stop() {
    emergency_stopped_.store(true);
    set_system_status(SystemStatus::EMERGENCY_STOP);
    set_system_health(SystemHealth::CRITICAL_ERROR);
    
    if (emergency_stop_callback_) {
        emergency_stop_callback_();
    }
}

void ModbusServer::acknowledge_emergency_stop() {
    std::lock_guard<std::mutex> lock(emergency_mutex_);
    emergency_stopped_.store(false);
    set_system_status(SystemStatus::STOPPED);
    set_system_health(SystemHealth::NORMAL);
}

bool ModbusServer::is_emergency_stopped() const {
    return emergency_stopped_.load();
}

// 超时管理
void ModbusServer::reset_feed_detection_timer() {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    feed_detection_start_ = std::chrono::steady_clock::now();
    feed_detection_active_ = true;
}

void ModbusServer::reset_clamp_timer() {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    clamp_start_ = std::chrono::steady_clock::now();
    clamp_active_ = true;
}

void ModbusServer::reset_cut_execution_timer() {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    cut_execution_start_ = std::chrono::steady_clock::now();
    cut_execution_active_ = true;
}

bool ModbusServer::is_feed_detection_timeout() const {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    if (!feed_detection_active_) return false;
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - feed_detection_start_).count();
    return elapsed >= config_.feed_detection_timeout_s;
}

bool ModbusServer::is_clamp_timeout() const {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    if (!clamp_active_) return false;
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - clamp_start_).count();
    return elapsed >= config_.clamp_timeout_s;
}

bool ModbusServer::is_cut_execution_timeout() const {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    if (!cut_execution_active_) return false;
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - cut_execution_start_).count();
    return elapsed >= config_.cut_execution_timeout_s;
}

// 事件回调设置
void ModbusServer::set_connection_callback(ConnectionCallback callback) {
    connection_callback_ = callback;
}

void ModbusServer::set_command_callback(CommandCallback callback) {
    command_callback_ = callback;
}

void ModbusServer::set_emergency_stop_callback(EmergencyStopCallback callback) {
    emergency_stop_callback_ = callback;
}

void ModbusServer::set_timeout_callback(TimeoutCallback callback) {
    timeout_callback_ = callback;
}

// 错误处理
std::string ModbusServer::get_last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

void ModbusServer::clear_error() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_.clear();
}

ModbusServer::Statistics ModbusServer::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

// 私有方法实现
void ModbusServer::write_int16_to_registers(int start_reg, uint16_t value) {
    if (start_reg >= 0 && start_reg < REG_COUNT) {
        registers_[start_reg] = value;
    }
}

void ModbusServer::write_int32_to_registers(int start_reg, uint32_t value) {
    if (start_reg >= 0 && start_reg + 1 < REG_COUNT) {
        // Little-Endian格式存储32位值
        registers_[start_reg] = static_cast<uint16_t>(value & 0xFFFF);
        registers_[start_reg + 1] = static_cast<uint16_t>((value >> 16) & 0xFFFF);
    }
}

uint16_t ModbusServer::read_int16_from_registers(int start_reg) const {
    if (start_reg >= 0 && start_reg < REG_COUNT) {
        return registers_[start_reg];
    }
    return 0;
}

uint32_t ModbusServer::read_int32_from_registers(int start_reg) const {
    if (start_reg >= 0 && start_reg + 1 < REG_COUNT) {
        // Little-Endian格式读取32位值
        uint32_t low = registers_[start_reg];
        uint32_t high = registers_[start_reg + 1];
        return low | (high << 16);
    }
    return 0;
}

void ModbusServer::log_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = error;
    std::cerr << "[ModbusServer错误] " << error << std::endl;
}

void ModbusServer::handle_modbus_error(modbus_t* ctx, const std::string& operation) {
#ifdef ENABLE_MODBUS
    std::string error_msg = operation + ": " + modbus_strerror(errno);
    log_error(error_msg);
#else
    std::string error_msg = operation + ": Modbus功能未启用";
    log_error(error_msg);
#endif
}

} // namespace communication
} // namespace bamboo_cut