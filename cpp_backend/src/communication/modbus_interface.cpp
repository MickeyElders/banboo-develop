/**
 * @file modbus_interface.cpp
 * @brief Modbus TCP communication implementation for the vision client.
 */

#include "bamboo_cut/communication/modbus_interface.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <cerrno>
#include <utility>

namespace bamboo_cut {
namespace communication {

ModbusInterface::ModbusInterface(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(std::move(data_bridge))
    , modbus_ctx_(nullptr)
    , communication_errors_(0)
    , reconnect_attempts_(0) {
    
    stats_.total_requests = 0;
    stats_.successful_requests = 0;
    stats_.failed_requests = 0;
    stats_.reconnect_count = 0;
    stats_.success_rate = 0.0f;
    stats_.last_success_time = 0;
    stats_.last_error_time = 0;
    
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
    
    std::cout << "[ModbusInterface] init Modbus TCP (" << config_.server_ip
              << ":" << config_.server_port << ")" << std::endl;
    
    modbus_ctx_ = modbus_new_tcp(config_.server_ip.c_str(), config_.server_port);
    if (!modbus_ctx_) {
        handleError("failed to create Modbus context");
        return false;
    }
    
    modbus_set_slave(modbus_ctx_, config_.slave_id);
    modbus_set_response_timeout(modbus_ctx_, config_.timeout_ms / 1000,
                                (config_.timeout_ms % 1000) * 1000);
    
    std::cout << "[ModbusInterface] Modbus initialized" << std::endl;
    return true;
}

bool ModbusInterface::start() {
    if (running_.load()) {
        return false;
    }
    
    if (!modbus_ctx_) {
        std::cerr << "[ModbusInterface] context not initialized" << std::endl;
        return false;
    }
    
    should_stop_ = false;
    running_ = true;
    communication_thread_ = std::thread(&ModbusInterface::communicationLoop, this);
    
    std::cout << "[ModbusInterface] communication thread started" << std::endl;
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
    
    std::cout << "[ModbusInterface] communication thread stopped" << std::endl;
}

bool ModbusInterface::reconnect() {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    disconnect();
    
    last_reconnect_attempt_ = std::chrono::high_resolution_clock::now();
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
    int result = modbus_read_registers(modbus_ctx_, address, 1, &read_value);
    if (result == 1) {
        value = read_value;
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        logModbusOperation("read reg", true, std::to_string(address) + "=" + std::to_string(value));
        return true;
    }
    
    stats_.failed_requests++;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats_.last_error = "read register failed: " + std::string(modbus_strerror(errno));
    handleError("read register failed: " + std::to_string(address));
    return false;
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
        logModbusOperation("write reg", true, std::to_string(address) + "=" + std::to_string(value));
        return true;
    }
    
    stats_.failed_requests++;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats_.last_error = "write register failed: " + std::string(modbus_strerror(errno));
    handleError("write register failed: " + std::to_string(address));
    return false;
}

bool ModbusInterface::readRegisters(ModbusRegister start_reg, int count, uint16_t* values) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (!connected_.load()) {
        return false;
    }
    
    int address = toModbusAddress(start_reg);
    
    stats_.total_requests++;
    int result = modbus_read_registers(modbus_ctx_, address, count, values);
    if (result == count) {
        stats_.successful_requests++;
        stats_.last_success_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        logModbusOperation("read regs", true,
                          std::to_string(address) + "-" + std::to_string(address + count - 1));
        return true;
    }
    
    stats_.failed_requests++;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats_.last_error = "read registers failed: " + std::string(modbus_strerror(errno));
    handleError("read registers failed: " + std::to_string(address));
    return false;
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
        logModbusOperation("write regs", true,
                          std::to_string(address) + "-" + std::to_string(address + count - 1));
        return true;
    }
    
    stats_.failed_requests++;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats_.last_error = "write registers failed: " + std::string(modbus_strerror(errno));
    handleError("write registers failed: " + std::to_string(address));
    return false;
}

ModbusInterface::ConnectionStats ModbusInterface::getConnectionStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    ConnectionStats stats = stats_;
    if (stats.total_requests > 0) {
        stats.success_rate = static_cast<float>(stats.successful_requests) / stats.total_requests * 100.0f;
    }
    
    return stats;
}

void ModbusInterface::communicationLoop() {
    std::cout << "[ModbusInterface] communication loop started" << std::endl;
    
    if (!connectToPLC()) {
        std::cerr << "[ModbusInterface] initial connection failed" << std::endl;
    }
    
    while (!should_stop_.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        
        if (!connected_.load()) {
            if (config_.auto_reconnect && reconnect_attempts_ < MAX_RECONNECT_ATTEMPTS) {
                auto now = std::chrono::high_resolution_clock::now();
                auto time_since_last_attempt = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_reconnect_attempt_);
                
                if (time_since_last_attempt.count() >= config_.reconnect_interval) {
                    std::cout << "[ModbusInterface] try reconnect..." << std::endl;
                    reconnect();
                }
            }
        } else {
            try {
                updateSystemDataToPLC();
                readCommandsFromPLC();
                updateHeartbeat();
                
                last_communication_time_ = std::chrono::high_resolution_clock::now();
                communication_errors_ = 0;
            } catch (const std::exception& e) {
                handleError(std::string("communication exception: ") + e.what());
                communication_errors_++;
                
                if (communication_errors_ > 5) {
                    connected_ = false;
                    std::cerr << "[ModbusInterface] too many communication errors, disconnect" << std::endl;
                }
            }
        }
        
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            loop_end - loop_start);
        
        if (loop_duration.count() < COMMUNICATION_INTERVAL_MS) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(COMMUNICATION_INTERVAL_MS - loop_duration.count()));
        }
    }
    
    std::cout << "[ModbusInterface] communication loop stopped" << std::endl;
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
        
        std::cout << "[ModbusInterface] connected to PLC" << std::endl;
        return true;
    }
    
    connected_ = false;
    handleError("connect PLC failed: " + std::string(modbus_strerror(errno)));
    return false;
}

void ModbusInterface::disconnect() {
    if (modbus_ctx_ && connected_.load()) {
        modbus_close(modbus_ctx_);
        connected_ = false;
        std::cout << "[ModbusInterface] disconnected from PLC" << std::endl;
    }
}

void ModbusInterface::updateSystemDataToPLC() {
    if (!data_bridge_) return;
    
    const auto modbus_regs = data_bridge_->getModbusRegisters();
    
    // Data first, flags last to avoid stale reads on PLC side.
    writeRegister(ModbusRegister::SYSTEM_STATUS, modbus_regs.system_status);
    
    uint16_t x_high = 0, x_low = 0;
    splitUint32(modbus_regs.x_coordinate, x_high, x_low);
    writeRegister(ModbusRegister::X_COORDINATE_H, x_high);
    writeRegister(ModbusRegister::X_COORDINATE_L, x_low);
    
    writeRegister(ModbusRegister::CUT_QUALITY, modbus_regs.cut_quality);
    writeRegister(ModbusRegister::BLADE_NUMBER, modbus_regs.blade_number);
    writeRegister(ModbusRegister::HEALTH_STATUS, modbus_regs.health_status);
    writeRegister(ModbusRegister::TAIL_STATUS, modbus_regs.tail_status);
    writeRegister(ModbusRegister::RAIL_DIRECTION, modbus_regs.rail_direction);
    
    uint16_t length_high = 0, length_low = 0;
    splitUint32(modbus_regs.remaining_length, length_high, length_low);
    writeRegister(ModbusRegister::REMAIN_LENGTH_H, length_high);
    writeRegister(ModbusRegister::REMAIN_LENGTH_L, length_low);
    
    writeRegister(ModbusRegister::COVERAGE, modbus_regs.coverage);
    writeRegister(ModbusRegister::COORD_READY, modbus_regs.coord_ready);
}

void ModbusInterface::readCommandsFromPLC() {
    uint16_t command = 0;
    if (readRegister(ModbusRegister::PLC_COMMAND, command)) {
        data_bridge_->setPLCCommandValue(command);
        if (command != 0) {
            processPLCCommand(command);
        }
    }
    
    uint16_t process_mode = 0;
    if (readRegister(ModbusRegister::PROCESS_MODE, process_mode)) {
        data_bridge_->setProcessMode(process_mode);
    }
    
    uint16_t feed_speed = 0;
    if (readRegister(ModbusRegister::FEED_SPEED_GEAR, feed_speed)) {
        data_bridge_->setFeedSpeedGear(feed_speed);
    }
    
    uint16_t plc_alarm = 0;
    if (readRegister(ModbusRegister::PLC_EXT_ALARM, plc_alarm)) {
        data_bridge_->setPLCAlarmCode(plc_alarm);
    }
}

void ModbusInterface::processPLCCommand(uint16_t command) {
    std::cout << "[ModbusInterface] PLC command: " << command << std::endl;
    
    switch (command) {
        case 1: // feed detection
            data_bridge_->setCurrentStep(1);
            break;
        case 2: // cut prepare
            data_bridge_->setCurrentStep(2);
            break;
        case 3: // cut complete
            data_bridge_->setCurrentStep(3);
            data_bridge_->clearCoordinateReady();
            break;
        case 4: // start feeding
            data_bridge_->setCurrentStep(4);
            break;
        case 5: // pause
            data_bridge_->setSystemRunning(false);
            break;
        case 6: // emergency stop
            data_bridge_->setEmergencyStop(true);
            break;
        case 7: // resume
            data_bridge_->setEmergencyStop(false);
            data_bridge_->setSystemRunning(true);
            break;
        case 8: // repeat detection
            data_bridge_->setCurrentStep(1);
            break;
        case 9: // tail handling
            data_bridge_->setCurrentStep(5);
            data_bridge_->clearCoordinateReady();
            break;
        default:
            std::cout << "[ModbusInterface] unknown PLC command: " << command << std::endl;
            break;
    }
}

void ModbusInterface::updateHeartbeat() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat_time_);
    
    if (duration.count() >= HEARTBEAT_INTERVAL_MS) {
        data_bridge_->updateHeartbeat();
        uint32_t heartbeat = data_bridge_->getHeartbeat();
        
        uint16_t heartbeat_high = 0, heartbeat_low = 0;
        splitUint32(heartbeat, heartbeat_high, heartbeat_low);
        
        writeRegister(ModbusRegister::HEARTBEAT_H, heartbeat_high);
        writeRegister(ModbusRegister::HEARTBEAT_L, heartbeat_low);
        
        last_heartbeat_time_ = now;
    }
}

void ModbusInterface::handleError(const std::string& error_msg) {
    std::cerr << "[ModbusInterface] error: " << error_msg << std::endl;
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.last_error = error_msg;
    stats_.last_error_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void ModbusInterface::logModbusOperation(const std::string& operation,
                                       bool success,
                                       const std::string& details) {
    (void)operation;
    (void)success;
    (void)details;
    // To avoid flooding logs, keep silent by default.
}

} // namespace communication
} // namespace bamboo_cut
