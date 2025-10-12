/**
 * @file bamboo_system.h
 * @brief C++ LVGL一体化竹子识别系统主控制器
 * 整合AI推理、LVGL界面和Modbus通信的核心系统类
 */

#pragma once

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/inference/bamboo_detector.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include "bamboo_cut/communication/modbus_interface.h"

namespace bamboo_cut {
namespace core {

/**
 * @brief 系统配置参数
 */
struct SystemConfig {
    // AI推理配置
    inference::DetectorConfig detector_config;
    
    // LVGL Wayland界面配置
    ui::LVGLWaylandConfig ui_config;
    
    // Modbus通信配置
    communication::ModbusConfig modbus_config;
    
    // 系统运行参数
    struct {
        bool enable_ai_inference;       // 启用AI推理
        bool enable_ui_interface;       // 启用界面显示
        bool enable_modbus_communication; // 启用Modbus通信
        bool enable_auto_start;         // 自动启动
        int main_loop_interval_ms;      // 主循环间隔(ms)
        int stats_update_interval_ms;   // 统计更新间隔(ms)
        int workflow_step_timeout_ms;   // 工作流程步骤超时(ms)
    } system_params;
    
    SystemConfig() {
        system_params.enable_ai_inference = true;
        system_params.enable_ui_interface = true;
        system_params.enable_modbus_communication = true;
        system_params.enable_auto_start = false;
        system_params.main_loop_interval_ms = 10;
        system_params.stats_update_interval_ms = 1000;
        system_params.workflow_step_timeout_ms = 30000;
    }
    
    /**
     * @brief 从配置文件加载
     */
    bool loadFromFile(const std::string& config_file);
    
    /**
     * @brief 保存到配置文件
     */
    bool saveToFile(const std::string& config_file) const;
};

/**
 * @brief 系统状态枚举
 */
enum class SystemState {
    UNINITIALIZED = 0,  // 未初始化
    INITIALIZING = 1,   // 初始化中
    READY = 2,          // 就绪
    RUNNING = 3,        // 运行中
    PAUSED = 4,         // 暂停
    ERROR = 5,          // 错误
    EMERGENCY = 6,      // 紧急停止
    SHUTDOWN = 7        // 关闭中
};

/**
 * @brief 工作流程管理器
 */
class WorkflowManager {
public:
    WorkflowManager(std::shared_ptr<DataBridge> data_bridge);
    ~WorkflowManager();

    /**
     * @brief 启动工作流程
     */
    void startWorkflow();

    /**
     * @brief 停止工作流程
     */
    void stopWorkflow();

    /**
     * @brief 暂停工作流程
     */
    void pauseWorkflow();

    /**
     * @brief 恢复工作流程
     */
    void resumeWorkflow();

    /**
     * @brief 紧急停止
     */
    void emergencyStop();

    /**
     * @brief 检查工作流程是否运行
     */
    bool isRunning() const { return workflow_running_.load(); }

    /**
     * @brief 获取当前步骤
     */
    int getCurrentStep() const;

private:
    /**
     * @brief 工作流程线程主循环
     */
    void workflowLoop();

    /**
     * @brief 执行工作流程步骤
     */
    void executeWorkflowStep(int step);

    /**
     * @brief 检查步骤完成条件
     */
    bool isStepCompleted(int step);

    /**
     * @brief 处理步骤超时
     */
    void handleStepTimeout(int step);

private:
    std::shared_ptr<DataBridge> data_bridge_;
    std::thread workflow_thread_;
    std::atomic<bool> workflow_running_{false};
    std::atomic<bool> workflow_paused_{false};
    std::atomic<bool> should_stop_{false};
    
    std::chrono::high_resolution_clock::time_point step_start_time_;
    int current_step_;
    int step_timeout_ms_;
};

/**
 * @brief C++ LVGL一体化竹子识别系统主类
 * 系统的核心控制器，协调所有子系统的运行
 */
class BambooSystem {
public:
    BambooSystem();
    ~BambooSystem();

    /**
     * @brief 初始化系统
     */
    bool initialize(const SystemConfig& config);

    /**
     * @brief 启动系统
     */
    bool start();

    /**
     * @brief 停止系统
     */
    void stop();

    /**
     * @brief 暂停系统
     */
    void pause();

    /**
     * @brief 恢复系统
     */
    void resume();

    /**
     * @brief 紧急停止
     */
    void emergencyStop();

    /**
     * @brief 运行系统主循环（阻塞）
     */
    int run();

    /**
     * @brief 检查系统是否在运行
     */
    bool isRunning() const { return current_state_ == SystemState::RUNNING; }

    /**
     * @brief 获取当前系统状态
     */
    SystemState getCurrentState() const { return current_state_; }

    /**
     * @brief 获取系统错误信息
     */
    std::string getLastError() const { return last_error_; }

    /**
     * @brief 获取系统统计信息
     */
    struct SystemInfo {
        SystemState state;
        std::string state_name;
        std::chrono::seconds uptime;
        float cpu_usage;
        float memory_usage;
        int current_workflow_step;
        bool ai_inference_active;
        bool ui_interface_active;
        bool modbus_communication_active;
        std::string last_error;
        std::chrono::system_clock::time_point start_time;
        
        // 性能统计
        struct {
            float inference_fps;
            float ui_fps;
            int total_detections;
            int total_cuts;
            float system_efficiency;
        } performance;
    };
    
    SystemInfo getSystemInfo() const;

    /**
     * @brief 重新加载配置
     */
    bool reloadConfig(const SystemConfig& config);

    /**
     * @brief 保存当前状态
     */
    bool saveState(const std::string& state_file) const;

    /**
     * @brief 加载保存的状态
     */
    bool loadState(const std::string& state_file);

    /**
     * @brief 获取数据桥接对象（用于外部访问）
     */
    std::shared_ptr<DataBridge> getDataBridge() const { return data_bridge_; }

private:
    /**
     * @brief 系统主循环
     */
    void mainLoop();

    /**
     * @brief 初始化所有子系统
     */
    bool initializeSubsystems();

    /**
     * @brief 启动所有子系统
     */
    bool startSubsystems();

    /**
     * @brief 停止所有子系统
     */
    void stopSubsystems();

    /**
     * @brief 更新系统统计信息
     */
    void updateSystemStats();

    /**
     * @brief 监控系统健康状态
     */
    void monitorSystemHealth();

    /**
     * @brief 处理系统错误
     */
    void handleSystemError(const std::string& error_msg);

    /**
     * @brief 状态转换
     */
    void changeState(SystemState new_state);

    /**
     * @brief 设置信号处理器
     */
    void setupSignalHandlers();

    /**
     * @brief 信号处理回调
     */
    static void signalHandler(int signal);

private:
    // 系统配置
    SystemConfig config_;
    
    // 系统状态
    std::atomic<SystemState> current_state_{SystemState::UNINITIALIZED};
    mutable std::string last_error_;
    std::chrono::system_clock::time_point start_time_;
    
    // 核心数据桥接
    std::shared_ptr<DataBridge> data_bridge_;
    
    // 子系统组件
    std::unique_ptr<inference::InferenceThread> inference_thread_;
    std::unique_ptr<ui::LVGLWaylandInterface> ui_wayland_interface_;
    std::unique_ptr<communication::ModbusInterface> modbus_interface_;
    std::unique_ptr<WorkflowManager> workflow_manager_;
    
    // 主循环控制
    std::thread main_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // 统计和监控
    mutable std::mutex stats_mutex_;
    SystemInfo system_info_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    
    // 静态实例（用于信号处理）
    static BambooSystem* instance_;
    
    // 性能监控
    struct {
        std::chrono::high_resolution_clock::time_point last_performance_check;
        int frame_count;
        int detection_count;
        float average_inference_time;
    } performance_monitor_;
};

/**
 * @brief 系统状态转字符串
 */
std::string systemStateToString(SystemState state);

/**
 * @brief 获取系统版本信息
 */
struct VersionInfo {
    int major;
    int minor;
    int patch;
    std::string build_date;
    std::string git_commit;
    
    std::string toString() const;
};

VersionInfo getVersionInfo();

} // namespace core
} // namespace bamboo_cut