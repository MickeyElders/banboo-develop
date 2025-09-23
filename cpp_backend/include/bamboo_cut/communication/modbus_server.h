#pragma once

#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>

// 条件包含 libmodbus 头文件
#ifdef ENABLE_MODBUS
extern "C" {
    #include <modbus/modbus.h>
}
#endif

// 通信消息类型包含
#include <bamboo_cut/core/types.h>

// 使用来自 types.h 的定义
using bamboo_cut::communication::CommunicationMessage;

namespace bamboo_cut {
namespace communication {

// 寄存器地址定义 (基于PLC.md文档扩展)
constexpr int REG_SYSTEM_STATUS = 40001;      // 系统状态
constexpr int REG_PLC_COMMAND = 40002;        // PLC命令
constexpr int REG_COORD_READY = 40003;        // 坐标就绪标志
constexpr int REG_X_COORDINATE = 40004;       // X坐标 (INT32, 占用40004-40005)
constexpr int REG_CUT_QUALITY = 40006;        // 切割质量
constexpr int REG_HEARTBEAT = 40007;          // 心跳计数器 (INT32, 占用40007-40008)
constexpr int REG_BLADE_NUMBER = 40009;       // 刀片编号
constexpr int REG_SYSTEM_HEALTH = 40010;      // 系统健康状态

// 新增寄存器 (基于PLC.md)
constexpr int REG_WASTE_DETECTION = 40011;    // 废料检测状态 (0=不是废料, 1=是废料, 9=尾料)
constexpr int REG_RAIL_DIRECTION = 40014;     // 导轨方向 (0=正向, 1=反向)
constexpr int REG_REMAINING_LENGTH = 40015;   // 剩余长度 (INT32, 占用40015-40016)
constexpr int REG_COVERAGE_RATE = 40017;      // 覆盖率
constexpr int REG_DETECTION_STATUS = 40018;   // 检测状态 (0=待机, 1=检测中, 2=完成, 3=重复检测)
constexpr int REG_PROCESS_MODE = 40019;       // 处理模式 (0=自动, 1=手动)

constexpr int REG_COUNT = 19;                 // 总寄存器数量 (40001-40019)

// 系统状态枚举
enum class SystemStatus : uint16_t {
    STOPPED = 0,         // 停止
    RUNNING = 1,         // 运行
    ERROR = 2,           // 错误
    PAUSED = 3,          // 暂停
    EMERGENCY_STOP = 4,  // 紧急停止
    MAINTENANCE = 5      // 维护模式
};

// PLC命令枚举 (基于PLC.md文档)
enum class PLCCommand : uint16_t {
    NONE = 0,             // 无命令
    FEED_DETECTION = 1,   // 进料检测 - 物料进入识别区时发送
    CUT_PREPARE = 2,      // 切割准备 - 夹持完成，准备切割
    CUT_COMPLETE = 3,     // 切割完成 - 切割操作完成
    START_FEEDING = 4,    // 启动送料 - 启动滚筒正向进料
    PAUSE = 5,            // 暂停
    EMERGENCY_STOP = 6,   // 紧急停止
    RESUME = 7,           // 恢复运行
    WASTE_REJECT = 8,     // 废料推出 - 启动反向推出废料
    DETECTION_START = 9,  // 开始检测 - 视觉系统开始检测
    POSITION_READY = 10,  // 位置就绪 - 导轨移动到位
    CLAMP_ENGAGED = 11,   // 夹持完成 - 夹持气缸动作完成
    SAFETY_CHECK = 12     // 安全检查 - 系统安全状态检查
};

// 坐标就绪标志
enum class CoordinateReady : uint16_t {
    NO_COORDINATE = 0,    // 无坐标
    HAS_COORDINATE = 1    // 有坐标
};

// 切割质量状态
enum class CutQuality : uint16_t {
    NORMAL = 0,           // 正常
    ABNORMAL = 1          // 异常
};

// 刀片编号
enum class BladeNumber : uint16_t {
    NONE = 0,             // 无刀片
    BLADE_1 = 1,          // 刀片1
    BLADE_2 = 2,          // 刀片2
    DUAL_BLADES = 3       // 双刀片同时
};

// 系统健康状态
enum class SystemHealth : uint16_t {
    NORMAL = 0,           // 正常
    WARNING = 1,          // 警告
    ERROR = 2,            // 错误
    CRITICAL_ERROR = 3    // 严重错误
};

// 坐标数据结构
struct CoordinateData {
    int32_t x_coordinate{0};           // X坐标 (0.1mm精度)
    BladeNumber blade_number{BladeNumber::NONE};  // 刀片编号
    CutQuality quality{CutQuality::NORMAL};       // 切割质量
    
    CoordinateData() = default;
    CoordinateData(int32_t x, BladeNumber blade, CutQuality qual = CutQuality::NORMAL)
        : x_coordinate(x), blade_number(blade), quality(qual) {}
};

// 系统状态数据结构
struct SystemStatusData {
    SystemStatus status{SystemStatus::STOPPED};
    SystemHealth health{SystemHealth::NORMAL};
    PLCCommand last_command{PLCCommand::NONE};
    bool coordinate_ready{false};
    uint32_t heartbeat_counter{0};
    
    SystemStatusData() = default;
};

// 通信配置
struct ModbusConfig {
    std::string ip_address{"0.0.0.0"};    // 监听IP地址
    int port{502};                         // Modbus TCP端口
    int max_connections{10};               // 最大连接数
    int response_timeout_ms{1000};         // 响应超时 (毫秒)
            int heartbeat_interval_ms{100};        // 心跳间隔 (毫秒, 按协议要求100ms)
    int communication_update_ms{20};       // 通信更新间隔 (毫秒)
    
    // 超时设置 (基于协议文档)
    int feed_detection_timeout_s{15};      // 进料检测超时 (秒)
    int clamp_timeout_s{60};               // 夹持固定超时 (秒)
    int cut_execution_timeout_s{120};      // 切割执行超时 (秒)
    int emergency_response_timeout_ms{100}; // 紧急停止响应时间 (毫秒)
};

// 事件回调函数类型定义
using ConnectionCallback = std::function<void(bool connected, const std::string& client_ip)>;
using CommandCallback = std::function<void(PLCCommand command)>;
using EmergencyStopCallback = std::function<void()>;
using TimeoutCallback = std::function<void(const std::string& timeout_type)>;

/**
 * @brief Modbus TCP服务器类
 * 
 * 基于协议文档实现的完整Modbus TCP服务器，支持：
 * - 主动推送坐标数据
 * - 心跳监控机制
 * - 安全机制（紧急停止、系统健康监控）
 * - 完善的错误处理和超时管理
 */
class ModbusServer {
public:
    ModbusServer();
    explicit ModbusServer(const ModbusConfig& config);
    ~ModbusServer();

    // 禁用拷贝构造和赋值
    ModbusServer(const ModbusServer&) = delete;
    ModbusServer& operator=(const ModbusServer&) = delete;

    // 服务器控制
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    bool is_connected() const { return client_connected_.load(); }
    bool is_plc_connected() const;
    

    // 坐标数据管理 (主动推送)
    void set_coordinate_data(const CoordinateData& data);
    void clear_coordinate_data();
    CoordinateData get_coordinate_data() const;

    // 系统状态管理
    void set_system_status(SystemStatus status);
    void set_system_health(SystemHealth health);
    SystemStatus get_system_status() const;
    PLCCommand get_last_plc_command() const;

    // 心跳和健康监控
    uint32_t get_heartbeat_counter() const;
    bool is_heartbeat_active() const;
    void reset_heartbeat();

    // 安全机制
    void trigger_emergency_stop();
    void acknowledge_emergency_stop();
    bool is_emergency_stopped() const;

    // 超时管理
    void reset_feed_detection_timer();
    void reset_clamp_timer();
    void reset_cut_execution_timer();
    bool is_feed_detection_timeout() const;
    bool is_clamp_timeout() const;
    bool is_cut_execution_timeout() const;

    // 事件回调设置
    void set_connection_callback(ConnectionCallback callback);
    void set_command_callback(CommandCallback callback);
    void set_emergency_stop_callback(EmergencyStopCallback callback);
    void set_timeout_callback(TimeoutCallback callback);

    // 错误处理
    std::string get_last_error() const;
    void clear_error();

    // 统计信息
    struct Statistics {
        uint64_t total_connections{0};
        uint64_t total_requests{0};
        uint64_t total_errors{0};
        uint64_t heartbeat_timeouts{0};
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point last_request_time;
    };
    Statistics get_statistics() const;

private:
    // 核心通信功能
    void server_thread();
    void heartbeat_thread();
    void timeout_monitor_thread();
    bool handle_client_connection(modbus_t* ctx);
    void update_registers();
    void process_plc_command(PLCCommand command);

    // 寄存器操作
    void write_int16_to_registers(int start_reg, uint16_t value);
    void write_int32_to_registers(int start_reg, uint32_t value);
    uint16_t read_int16_from_registers(int start_reg) const;
    uint32_t read_int32_from_registers(int start_reg) const;

    // 错误处理
    void log_error(const std::string& error);
    void handle_modbus_error(modbus_t* ctx, const std::string& operation);

    // 配置
    ModbusConfig config_;

    // Modbus上下文
    modbus_t* modbus_ctx_{nullptr};
    int server_socket_{-1};

    // 寄存器映射 (Modbus协议使用基于0的寄存器地址)
    mutable std::mutex registers_mutex_;
    uint16_t registers_[REG_COUNT]{0};

    // 线程管理
    std::atomic<bool> running_{false};
    std::atomic<bool> client_connected_{false};
    std::thread server_thread_;
    std::thread heartbeat_thread_;
    std::thread timeout_monitor_thread_;

    // 数据状态
    mutable std::mutex data_mutex_;
    CoordinateData current_coordinate_;
    SystemStatusData system_status_;
    
    // 心跳管理
    std::atomic<uint32_t> heartbeat_counter_{0};
    std::chrono::steady_clock::time_point last_heartbeat_time_;

    // 安全状态
    std::atomic<bool> emergency_stopped_{false};
    mutable std::mutex emergency_mutex_;

    // 超时管理
    mutable std::mutex timeout_mutex_;
    std::chrono::steady_clock::time_point feed_detection_start_;
    std::chrono::steady_clock::time_point clamp_start_;
    std::chrono::steady_clock::time_point cut_execution_start_;
    bool feed_detection_active_{false};
    bool clamp_active_{false};
    bool cut_execution_active_{false};

    // 事件回调
    ConnectionCallback connection_callback_;
    CommandCallback command_callback_;
    EmergencyStopCallback emergency_stop_callback_;
    TimeoutCallback timeout_callback_;

    // 错误信息
    mutable std::mutex error_mutex_;
    std::string last_error_;

    // 统计信息
    mutable std::mutex stats_mutex_;
    Statistics statistics_;
    
    // PLC连接状态监控
    std::atomic<bool> plc_connection_active_;
    std::chrono::steady_clock::time_point last_plc_communication_;
    std::thread plc_monitor_thread_;
    void plc_connection_monitor_thread();
};

} // namespace communication
} // namespace bamboo_cut 