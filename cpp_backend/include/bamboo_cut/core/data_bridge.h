/**
 * @file data_bridge.h
 * @brief C++ LVGL一体化系统数据桥接层
 * 线程安全的数据交换机制，连接AI推理和LVGL界面
 */

#pragma once

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace bamboo_cut {
namespace core {

/**
 * @brief 检测结果数据结构
 */
struct DetectionResult {
    std::vector<cv::Rect> bboxes;               // 检测框
    std::vector<cv::Point2f> cutting_points;   // 切割点
    std::vector<float> confidences;             // 置信度
    int64_t timestamp;                          // 时间戳
    bool valid;                                 // 数据有效性
    
    DetectionResult() : timestamp(0), valid(false) {}
    
    void clear() {
        bboxes.clear();
        cutting_points.clear();
        confidences.clear();
        timestamp = 0;
        valid = false;
    }
};

/**
 * @brief 系统性能统计
 */
struct SystemStats {
    float camera_fps;           // 摄像头帧率
    float inference_fps;        // 推理帧率
    int total_detections;       // 总检测数
    std::string system_status;  // 系统状态
    
    // Jetson性能数据
    struct {
        float cpu_usage;        // CPU使用率
        float gpu_usage;        // GPU使用率
        float cpu_temp;         // CPU温度
        float gpu_temp;         // GPU温度
        float memory_usage;     // 内存使用率
        float power_draw;       // 功耗
    } jetson;
    
    SystemStats() : camera_fps(0), inference_fps(0), total_detections(0),
                   system_status("初始化中") {
        jetson.cpu_usage = 0;
        jetson.gpu_usage = 0;
        jetson.cpu_temp = 0;
        jetson.gpu_temp = 0;
        jetson.memory_usage = 0;
        jetson.power_draw = 0;
    }
};

/**
 * @brief Modbus寄存器数据
 */
struct ModbusRegisters {
    uint16_t system_status;     // 40001 - 系统状态
    uint16_t plc_command;       // 40002 - PLC命令
    uint16_t coord_ready;       // 40003 - 坐标就绪
    uint32_t x_coordinate;      // 40004 - X坐标(高低位)
    uint16_t cut_quality;       // 40006 - 切割质量
    uint32_t heartbeat;         // 40007 - 心跳计数器
    uint16_t blade_number;      // 40009 - 刀具编号
    uint16_t health_status;     // 40010 - 健康状态
    
    ModbusRegisters() : system_status(0), plc_command(0), coord_ready(0),
                       x_coordinate(0), cut_quality(0), heartbeat(0),
                       blade_number(3), health_status(0) {}
};

/**
 * @brief 线程安全的数据桥接类
 * 负责在AI推理线程、LVGL界面线程和Modbus通信线程之间安全传递数据
 */
class DataBridge {
public:
    DataBridge();
    ~DataBridge();

    // 视频帧数据更新和获取
    void updateFrame(const cv::Mat& frame);
    bool getLatestFrame(cv::Mat& frame);
    bool getDisplayFrame(cv::Mat& frame);
    
    // 检测结果数据更新和获取
    void updateDetection(const DetectionResult& result);
    bool getDetectionResult(DetectionResult& result);
    
    // 系统统计信息更新和获取
    void updateStats(const SystemStats& stats);
    SystemStats getStats() const;
    
    // Modbus寄存器数据更新和获取
    void updateModbusRegisters(const ModbusRegisters& registers);
    ModbusRegisters getModbusRegisters() const;
    
    // 系统控制信号
    void setSystemRunning(bool running);
    bool isSystemRunning() const;
    
    void setEmergencyStop(bool stop);
    bool isEmergencyStop() const;
    
    // 工作流程步骤控制
    void setCurrentStep(int step);
    int getCurrentStep() const;
    
    // 心跳更新
    void updateHeartbeat();
    uint32_t getHeartbeat() const;

private:
    // 视频数据
    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    cv::Mat display_frame_;
    std::atomic<bool> new_frame_available_{false};
    
    // 检测结果数据
    mutable std::mutex detection_mutex_;
    DetectionResult detection_data_;
    std::atomic<bool> new_detection_available_{false};
    
    // 系统统计数据
    mutable std::mutex stats_mutex_;
    SystemStats stats_;
    
    // Modbus寄存器数据
    mutable std::mutex modbus_mutex_;
    ModbusRegisters modbus_registers_;
    
    // 系统控制状态
    std::atomic<bool> system_running_{false};
    std::atomic<bool> emergency_stop_{false};
    std::atomic<int> current_step_{1};
    std::atomic<uint32_t> heartbeat_counter_{0};
    
    // 条件变量用于线程同步
    std::condition_variable frame_cv_;
    std::condition_variable detection_cv_;
};

} // namespace core
} // namespace bamboo_cut