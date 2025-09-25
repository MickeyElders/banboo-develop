/**
 * @file modbus_interface.cpp
 * @brief C++ LVGL一体化系统Modbus通信接口实现
 * @version 5.0.0
 * @date 2024
 * 
 * 与西门子PLC的Modbus TCP通信实现
 */

#include "bamboo_cut/communication/modbus_interface.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

namespace bamboo_cut {
namespace communication {

ModbusInterface::ModbusInterface(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(data_bridge)
    , modbus_ctx_(nullptr)
    , communication_errors_(0)
    , reconnect_attempts_(0) {
    
    // 初始化统计数据
    stats_.total_requests = 0;
    stats_.successful_requests = 0;
    stats_.failed_requests = 0;
    stats_.reconnect_count = 0;
    stats_.success_rate = 0.0f;
    stats_.last_success_time = 0;
    stats_.last_error_time = 0;
    
    // 初始化寄存器缓存
    register_cache_.fill(0);
    register_dirty_.fill(false);
    
    last_communication_time_ = std::chrono::high_resolution_clock::now();
    last_heartbeat_time_ = std::chrono::high_resolution_clock::now();
    last_reconnect_attempt_ = std::chrono::high_resolution_clock::now();
}

ModbusInterface::~ModbusInterface() {
    stop();
    if (modbus_ctx_) {
        modbus_close(modbus_ctx_);
        modbus_free(modbus_ctx_);
        modbus_ctx_ = nullptr;
    }
}

bool ModbusInterface::initialize(const ModbusConfig& config) {
    config_ = config;
    
    std::cout << "[ModbusInterface] 初始化Modbus连接..." << std::endl;
    std::cout << "[ModbusInterface] PLC地址: " << config_.server_ip 
              << ":" << config_.server_port << std::endl;
    
    // 创建Modbus TCP上下文
    modbus_ctx_ = modbus_new_tcp(config_.server_ip.c_str(), config_.server_port);
    if (!modbus_ctx_) {
        handleError("无法创建Modbus上下文");
        return false;
    }
    
    // 设置从站ID
    modbus_set_slave(modbus_ctx_, config_.slave_id);
    
    // 设置超时
    modbus_set_response_timeout(modbus_ctx_, config_.timeout_ms / 1000, 
                               (config_.timeout_ms % 1000) * 1000);
    
    std::cout << "[ModbusInterface] Modbus初始化完成" << std::endl;
    return true;
}

bool ModbusInterface::start() {
    if (running_.load()) {
        return false;
    }
    
    if (!modbus_ctx_) {
        std::cerr << "[ModbusInterface] Modbus上下文未初始化" << std::endl;
        return false;
    }
    
    should_stop_ = false;
    running_ = true;
    communication_thread_ = std::thread(&ModbusInterface::communicationLoop, this);
    
    std::cout << "[ModbusInterface] 通信线程已启动" << std::endl;
    return true;
}

void ModbusInterface::stop() {
    if (!running_.load()) {
        return;
    }
    
    should_stop_ = true;
    running_ = false;
    
    if (communication_thread_.joinable()) {
        communication_thread_.join();
    }
    
    disconnect();
    
    std::cout << "[ModbusInterface] 通信线程已停止" << std::endl;
}

bool ModbusInterface::reconnect() {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    disconnect();
    
    auto now = std::chrono::high_resolution_clock::now();
    last_reconnect_attempt_ = now;
    reconnect_attempts_++;
    
    return connectToPLC();
}

bool ModbusInterface::readRegister(ModbusRegister reg, uint16_t& value) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (!connected_.load()) {
        return false;
    }
    
    int address = toModbusAddress(reg);
    uint16_t read_value;
    
    stats_.total_requests++;
    
    int result = modbus_read_holding_registers(modbus_ctx_, address, 1, &read_value);
    if (result == 1) {
        value = read_value;
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        logModbusOperation("读寄存器", true, std::to_string(address) + "=" + std::to_string(value));
        return true;
    } else {
        stats_.failed_requests++;
        stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        stats_.last_error = "读寄存器失败: " + std::string(modbus_strerror(errno));
        
        handleError("读寄存器失败: " + std::to_string(address));
        return false;
    }
}

bool ModbusInterface::writeRegister(ModbusRegister reg, uint16_t value) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (!connected_.load()) {
        return false;
    }
    
    int address = toModbusAddress(reg);
    
    stats_.total_requests++;
    
    int result = modbus_write_register(modbus_ctx_, address, value);
    if (result == 1) {
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        logModbusOperation("写寄存器", true, std::to_string(address) + "=" + std::to_string(value));
        return true;
    } else {
        stats_.failed_requests++;
        stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        stats_.last_error = "写寄存器失败: " + std::string(modbus_strerror(errno));
        
        handleError("写寄存器失败: " + std::to_string(address));
        return false;
    }
}

bool ModbusInterface::readRegisters(ModbusRegister start_reg, int count, uint16_t* values) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (!connected_.load()) {
        return false;
    }
    
    int address = toModbusAddress(start_reg);
    
    stats_.total_requests++;
    
    int result = modbus_read_holding_registers(modbus_ctx_, address, count, values);
    if (result == count) {
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        logModbusOperation("读多寄存器", true, 
                          std::to_string(address) + "-" + std::to_string(address + count - 1));
        return true;
    } else {
        stats_.failed_requests++;
        stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        stats_.last_error = "读多寄存器失败: " + std::string(modbus_strerror(errno));
        
        handleError("读多寄存器失败: " + std::to_string(address));
        return false;
    }
}

bool ModbusInterface::writeRegisters(ModbusRegister start_reg, int count, const uint16_t* values) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (!connected_.load()) {
        return false;
    }
    
    int address = toModbusAddress(start_reg);
    
    stats_.total_requests++;
    
    int result = modbus_write_registers(modbus_ctx_, address, count, values);
    if (result == count) {
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        logModbusOperation("写多寄存器", true, 
                          std::to_string(address) + "-" + std::to_string(address + count - 1));
        return true;
    } else {
        stats_.failed_requests++;
        stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        stats_.last_error = "写多寄存器失败: " + std::string(modbus_strerror(errno));
        
        handleError("写多寄存器失败: " + std::to_string(address));
        return false;
    }
}

ModbusInterface::ConnectionStats ModbusInterface::getConnectionStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    ConnectionStats stats = stats_;
    if (stats.total_requests > 0) {
        stats.success_rate = (float)stats.successful_requests / stats.total_requests * 100.0f;
    }
    
    return stats;
}

void ModbusInterface::communicationLoop() {
    std::cout << "[ModbusInterface] 通信循环开始" << std::endl;
    
    // 首次连接尝试
    if (!connectToPLC()) {
        std::cerr << "[ModbusInterface] 初始连接失败" << std::endl;
    }
    
    while (!should_stop_.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        
        // 检查连接状态
        if (!connected_.load()) {
            if (config_.auto_reconnect && reconnect_attempts_ < MAX_RECONNECT_ATTEMPTS) {
                auto now = std::chrono::high_resolution_clock::now();
                auto time_since_last_attempt = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_reconnect_attempt_);
                
                if (time_since_last_attempt.count() >= config_.reconnect_interval) {
                    std::cout << "[ModbusInterface] 尝试重新连接..." << std::endl;
                    reconnect();
                }
            }
        } else {
            // 执行通信任务
            try {
                updateSystemDataToPLC();
                readCommandsFromPLC();
                updateHeartbeat();
                
                last_communication_time_ = std::chrono::high_resolution_clock::now();
                communication_errors_ = 0;
                
            } catch (const std::exception& e) {
                handleError("通信循环异常: " + std::string(e.what()));
                communication_errors_++;
                
                if (communication_errors_ > 5) {
                    connected_ = false;
                    std::cerr << "[ModbusInterface] 连续通信错误，断开连接" << std::endl;
                }
            }
        }
        
        // 控制循环频率
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            loop_end - loop_start);
        
        if (loop_duration.count() < COMMUNICATION_INTERVAL_MS) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(COMMUNICATION_INTERVAL_MS - loop_duration.count()));
        }
    }
    
    std::cout << "[ModbusInterface] 通信循环结束" << std::endl;
}

bool ModbusInterface::connectToPLC() {
    if (!modbus_ctx_) {
        return false;
    }
    
    int result = modbus_connect(modbus_ctx_);
    if (result == 0) {
        connected_ = true;
        reconnect_attempts_ = 0;
        communication_errors_ = 0;
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.reconnect_count++;
        
        std::cout << "[ModbusInterface] 连接到PLC成功" << std::endl;
        return true;
    } else {
        connected_ = false;
        handleError("连接PLC失败: " + std::string(modbus_strerror(errno)));
        return false;
    }
}

void ModbusInterface::disconnect() {
    if (modbus_ctx_ && connected_.load()) {
        modbus_close(modbus_ctx_);
        connected_ = false;
        std::cout << "[ModbusInterface] 已断开PLC连接" << std::endl;
    }
}

void ModbusInterface::updateSystemDataToPLC() {
    if (!data_bridge_) return;
    
    // 获取系统状态
    auto stats = data_bridge_->getStats();
    auto modbus_regs = data_bridge_->getModbusRegisters();
    
    // 更新系统状态寄存器
    writeRegister(ModbusRegister::SYSTEM_STATUS, modbus_regs.system_status);
    writeRegister(ModbusRegister::COORD_READY, modbus_regs.coord_ready);
    
    // 更新X坐标
    uint16_t x_high, x_low;
    splitUint32(modbus_regs.x_coordinate, x_high, x_low);
    writeRegister(ModbusRegister::X_COORDINATE_H, x_high);
    writeRegister(ModbusRegister::X_COORDINATE_L, x_low);
    
    // 更新切割质量和刀具信息
    writeRegister(ModbusRegister::CUT_QUALITY, modbus_regs.cut_quality);
    writeRegister(ModbusRegister::BLADE_NUMBER, modbus_regs.blade_number);
    writeRegister(ModbusRegister::HEALTH_STATUS, modbus_regs.health_status);
}

void ModbusInterface::readCommandsFromPLC() {
    uint16_t command;
    if (readRegister(ModbusRegister::PLC_COMMAND, command)) {
        if (command != 0) {
            processPLCCommand(command);
            // 清除命令寄存器
            writeRegister(ModbusRegister::PLC_COMMAND, 0);
        }
    }
}

void ModbusInterface::processPLCCommand(uint16_t command) {
    std::cout << "[ModbusInterface] 收到PLC命令: " << command << std::endl;
    
    switch (command) {
        case 1: // 进料检测
            std::cout << "[ModbusInterface] 执行进料检测" << std::endl;
            data_bridge_->setCurrentStep(1);
            break;
            
        case 2: // 切割准备
            std::cout << "[ModbusInterface] 执行切割准备" << std::endl;
            data_bridge_->setCurrentStep(2);
            break;
            
        case 3: // 切割完成
            std::cout << "[ModbusInterface] 切割完成" << std::endl;
            data_bridge_->setCurrentStep(3);
            break;
            
        case 4: // 开始进料
            std::cout << "[ModbusInterface] 开始进料" << std::endl;
            data_bridge_->setCurrentStep(4);
            break;
            
        case 5: // 暂停
            std::cout << "[ModbusInterface] 系统暂停" << std::endl;
            data_bridge_->setSystemRunning(false);
            break;
            
        case 6: // 紧急停止
            std::cout << "[ModbusInterface] 紧急停止" << std::endl;
            data_bridge_->setEmergencyStop(true);
            break;
            
        case 7: // 恢复
            std::cout << "[ModbusInterface] 系统恢复" << std::endl;
            data_bridge_->setSystemRunning(true);
            data_bridge_->setEmergencyStop(false);
            break;
            
        default:
            std::cout << "[ModbusInterface] 未知PLC命令: " << command << std::endl;
            break;
    }
}

void ModbusInterface::updateHeartbeat() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_heartbeat_time_);
    
    if (duration.count() >= HEARTBEAT_INTERVAL_MS) {
        data_bridge_->updateHeartbeat();
        uint32_t heartbeat = data_bridge_->getHeartbeat();
        
        uint16_t heartbeat_high, heartbeat_low;
        splitUint32(heartbeat, heartbeat_high, heartbeat_low);
        
        writeRegister(ModbusRegister::HEARTBEAT_H, heartbeat_high);
        writeRegister(ModbusRegister::HEARTBEAT_L, heartbeat_low);
        
        last_heartbeat_time_ = now;
    }
}

void ModbusInterface::handleError(const std::string& error_msg) {
    std::cerr << "[ModbusInterface] 错误: " << error_msg << std::endl;
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.last_error = error_msg;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void ModbusInterface::logModbusOperation(const std::string& operation, 
                                       bool success, 
                                       const std::string& details) {
    if (success) {
        // std::cout << "[Modbus] " << operation << " 成功: " << details << std::endl;
    } else {
        std::cerr << "[Modbus] " << operation << " 失败: " << details << std::endl;
    }
}

} // namespace communication
} // namespace bamboo_cut