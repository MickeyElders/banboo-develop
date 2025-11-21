/**
 * @file modbus_interface.h
 * @brief Modbus TCP interface between vision client and PLC server.
 */

#pragma once

#include <modbus/modbus.h>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <array>
#include <cstdint>
#include "bamboo_cut/core/data_bridge.h"

namespace bamboo_cut {
namespace communication {

/**
 * @brief Modbus configuration.
 */
struct ModbusConfig {
    std::string server_ip;      // PLC IP
    int server_port;            // PLC port
    int slave_id;               // Slave ID
    int timeout_ms;             // Response timeout (ms)
    int reconnect_interval;     // Reconnect interval (s)
    bool auto_reconnect;        // Auto reconnect flag
    
    ModbusConfig()
        : server_ip("192.168.1.100")
        , server_port(502)
        , slave_id(1)
        , timeout_ms(1000)
        , reconnect_interval(5)
        , auto_reconnect(true) {}
};

/**
 * @brief Modbus register map (holding registers).
 * Ownership follows the new PLC/vision contract:
 * - Vision writes: system status, coordinates, quality, heartbeat, blade, health, tail, direction, remaining length, coverage.
 * - PLC writes: command, extended alarm, speed gear, process mode.
 */
enum class ModbusRegister {
    SYSTEM_STATUS      = 40001, // Vision -> PLC
    PLC_COMMAND        = 40002, // PLC -> Vision
    COORD_READY        = 40003, // Vision -> PLC
    X_COORDINATE_H     = 40004, // Vision -> PLC
    X_COORDINATE_L     = 40005, // Vision -> PLC
    CUT_QUALITY        = 40006, // Vision -> PLC
    HEARTBEAT_H        = 40007, // Vision -> PLC
    HEARTBEAT_L        = 40008, // Vision -> PLC
    BLADE_NUMBER       = 40009, // Vision -> PLC
    HEALTH_STATUS      = 40010, // Vision -> PLC
    TAIL_STATUS        = 40011, // Vision -> PLC
    PLC_EXT_ALARM      = 40012, // PLC -> Vision
    RAIL_DIRECTION     = 40014, // Vision -> PLC
    REMAIN_LENGTH_H    = 40015, // Vision -> PLC
    REMAIN_LENGTH_L    = 40016, // Vision -> PLC
    COVERAGE           = 40017, // Vision -> PLC
    FEED_SPEED_GEAR    = 40018, // PLC -> Vision
    PROCESS_MODE       = 40019  // PLC -> Vision
};

/**
 * @brief Modbus communication interface.
 */
class ModbusInterface {
public:
    explicit ModbusInterface(std::shared_ptr<core::DataBridge> data_bridge);
    ~ModbusInterface();

    bool initialize(const ModbusConfig& config);
    bool start();
    void stop();
    bool isConnected() const { return connected_.load(); }
    bool isRunning() const { return running_.load(); }
    bool reconnect();

    bool readRegister(ModbusRegister reg, uint16_t& value);
    bool writeRegister(ModbusRegister reg, uint16_t value);
    bool readRegisters(ModbusRegister start_reg, int count, uint16_t* values);
    bool writeRegisters(ModbusRegister start_reg, int count, const uint16_t* values);

    struct ConnectionStats {
        uint32_t total_requests;
        uint32_t successful_requests;
        uint32_t failed_requests;
        uint32_t reconnect_count;
        float success_rate;
        int64_t last_success_time;
        int64_t last_error_time;
        std::string last_error;
    };
    
    ConnectionStats getConnectionStats() const;

private:
    void communicationLoop();
    bool connectToPLC();
    void disconnect();

    void updateSystemDataToPLC();
    void readCommandsFromPLC();
    void processPLCCommand(uint16_t command);
    void updateHeartbeat();
    void handleError(const std::string& error_msg);
    void logModbusOperation(const std::string& operation, bool success, const std::string& details = "");

private:
    std::shared_ptr<core::DataBridge> data_bridge_;
    ModbusConfig config_;
    
    std::thread communication_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> connected_{false};
    
    modbus_t* modbus_ctx_;
    std::mutex modbus_mutex_;
    
    mutable std::mutex stats_mutex_;
    ConnectionStats stats_;
    
    std::chrono::high_resolution_clock::time_point last_communication_time_;
    std::chrono::high_resolution_clock::time_point last_heartbeat_time_;
    int communication_errors_;
    
    std::array<uint16_t, 32> register_cache_;
    std::array<bool, 32> register_dirty_;
    
    std::chrono::high_resolution_clock::time_point last_reconnect_attempt_;
    int reconnect_attempts_;
    
    static const int MAX_RECONNECT_ATTEMPTS = 10;
    static const int COMMUNICATION_INTERVAL_MS = 20;   // 20ms loop
    static const int HEARTBEAT_INTERVAL_MS = 20;       // 20ms heartbeat
};

inline int toModbusAddress(ModbusRegister reg) {
    return static_cast<int>(reg) - 40001;
}

inline void splitUint32(uint32_t value, uint16_t& high, uint16_t& low) {
    high = static_cast<uint16_t>((value >> 16) & 0xFFFF);
    low = static_cast<uint16_t>(value & 0xFFFF);
}

inline uint32_t combineUint16(uint16_t high, uint16_t low) {
    return (static_cast<uint32_t>(high) << 16) | static_cast<uint32_t>(low);
}

} // namespace communication
} // namespace bamboo_cut
