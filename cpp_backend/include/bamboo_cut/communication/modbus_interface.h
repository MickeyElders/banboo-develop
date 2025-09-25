/**
 * @file modbus_interface.h
 * @brief C++ LVGL一体化系统Modbus通信接口
 * 与西门子PLC的Modbus TCP通信实现
 */

#pragma once

#include <modbus/modbus.h>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include "bamboo_cut/core/data_bridge.h"

namespace bamboo_cut {
namespace communication {

/**
 * @brief Modbus配置参数
 */
struct ModbusConfig {
    std::string server_ip;      // PLC IP地址
    int server_port;            // 服务端口
    int slave_id;               // 从站ID
    int timeout_ms;             // 超时时间(ms)
    int reconnect_interval;     // 重连间隔(s)
    bool auto_reconnect;        // 自动重连
    
    ModbusConfig()
        : server_ip("192.168.1.100")
        , server_port(502)
        , slave_id(1)
        , timeout_ms(1000)
        , reconnect_interval(5)
        , auto_reconnect(true) {}
};

/**
 * @brief Modbus寄存器映射定义
 */
enum class ModbusRegister {
    SYSTEM_STATUS = 40001,      // 系统状态 (0=停止, 1=运行, 2=错误, 3=暂停, 4=紧急, 5=维护)
    PLC_COMMAND = 40002,        // PLC命令 (0=无, 1=进料检测, 2=切割准备, 3=切割完成, 4=开始进料, 5=暂停, 6=紧急, 7=恢复)
    COORD_READY = 40003,        // 坐标就绪标志 (0=未就绪, 1=就绪)
    X_COORDINATE_H = 40004,     // X坐标高位 (坐标*10的高16位)
    X_COORDINATE_L = 40005,     // X坐标低位 (坐标*10的低16位)
    CUT_QUALITY = 40006,        // 切割质量 (0=正常, 1=异常)
    HEARTBEAT_H = 40007,        // 心跳计数器高位
    HEARTBEAT_L = 40008,        // 心跳计数器低位
    BLADE_NUMBER = 40009,       // 刀具编号 (0=无, 1=刀1, 2=刀2, 3=双刀)
    HEALTH_STATUS = 40010       // 健康状态 (0=正常, 1=警告, 2=错误, 3=严重)
};

/**
 * @brief Modbus通信接口类
 * 负责与PLC的Modbus TCP通信
 */
class ModbusInterface {
public:
    explicit ModbusInterface(std::shared_ptr<core::DataBridge> data_bridge);
    ~ModbusInterface();

    /**
     * @brief 初始化Modbus连接
     */
    bool initialize(const ModbusConfig& config);

    /**
     * @brief 启动通信线程
     */
    bool start();

    /**
     * @brief 停止通信线程
     */
    void stop();

    /**
     * @brief 检查连接状态
     */
    bool isConnected() const { return connected_.load(); }

    /**
     * @brief 检查通信线程是否在运行
     */
    bool isRunning() const { return running_.load(); }

    /**
     * @brief 手动重连
     */
    bool reconnect();

    /**
     * @brief 读取单个寄存器
     */
    bool readRegister(ModbusRegister reg, uint16_t& value);

    /**
     * @brief 写入单个寄存器
     */
    bool writeRegister(ModbusRegister reg, uint16_t value);

    /**
     * @brief 读取多个寄存器
     */
    bool readRegisters(ModbusRegister start_reg, int count, uint16_t* values);

    /**
     * @brief 写入多个寄存器
     */
    bool writeRegisters(ModbusRegister start_reg, int count, const uint16_t* values);

    /**
     * @brief 获取连接统计信息
     */
    struct ConnectionStats {
        uint32_t total_requests;    // 总请求数
        uint32_t successful_requests; // 成功请求数
        uint32_t failed_requests;   // 失败请求数
        uint32_t reconnect_count;   // 重连次数
        float success_rate;         // 成功率
        int64_t last_success_time;  // 最后成功时间
        int64_t last_error_time;    // 最后错误时间
        std::string last_error;     // 最后错误信息
    };
    
    ConnectionStats getConnectionStats() const;

private:
    /**
     * @brief 通信线程主循环
     */
    void communicationLoop();

    /**
     * @brief 连接到PLC
     */
    bool connectToPLC();

    /**
     * @brief 断开连接
     */
    void disconnect();

    /**
     * @brief 更新系统数据到PLC
     */
    void updateSystemDataToPLC();

    /**
     * @brief 从PLC读取命令
     */
    void readCommandsFromPLC();

    /**
     * @brief 处理PLC命令
     */
    void processPLCCommand(uint16_t command);

    /**
     * @brief 更新心跳
     */
    void updateHeartbeat();

    /**
     * @brief 错误处理
     */
    void handleError(const std::string& error_msg);

    /**
     * @brief 日志记录
     */
    void logModbusOperation(const std::string& operation, bool success, const std::string& details = "");

private:
    std::shared_ptr<core::DataBridge> data_bridge_;
    ModbusConfig config_;
    
    std::thread communication_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> connected_{false};
    
    // Modbus上下文
    modbus_t* modbus_ctx_;
    std::mutex modbus_mutex_;
    
    // 连接统计
    mutable std::mutex stats_mutex_;
    ConnectionStats stats_;
    
    // 通信状态
    std::chrono::high_resolution_clock::time_point last_communication_time_;
    std::chrono::high_resolution_clock::time_point last_heartbeat_time_;
    int communication_errors_;
    
    // 寄存器缓存
    std::array<uint16_t, 16> register_cache_;
    std::array<bool, 16> register_dirty_;
    
    // 重连控制
    std::chrono::high_resolution_clock::time_point last_reconnect_attempt_;
    int reconnect_attempts_;
    
    static const int MAX_RECONNECT_ATTEMPTS = 10;
    static const int COMMUNICATION_INTERVAL_MS = 100;  // 100ms通信周期
    static const int HEARTBEAT_INTERVAL_MS = 1000;     // 1s心跳周期
};

/**
 * @brief Modbus寄存器地址转换
 */
inline int toModbusAddress(ModbusRegister reg) {
    return static_cast<int>(reg) - 40001;  // 转换为0基址
}

/**
 * @brief 32位数据分解为两个16位寄存器
 */
inline void splitUint32(uint32_t value, uint16_t& high, uint16_t& low) {
    high = static_cast<uint16_t>((value >> 16) & 0xFFFF);
    low = static_cast<uint16_t>(value & 0xFFFF);
}

/**
 * @brief 两个16位寄存器合并为32位数据
 */
inline uint32_t combineUint16(uint16_t high, uint16_t low) {
    return (static_cast<uint32_t>(high) << 16) | static_cast<uint32_t>(low);
}

} // namespace communication
} // namespace bamboo_cut