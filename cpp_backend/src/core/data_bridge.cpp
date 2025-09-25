/**
 * @file data_bridge.cpp
 * @brief C++ LVGL一体化系统数据桥接层实现
 */

#include "bamboo_cut/core/data_bridge.h"
#include <chrono>

namespace bamboo_cut {
namespace core {

DataBridge::DataBridge() {
    // 初始化显示帧缓冲区
    display_frame_ = cv::Mat::zeros(480, 640, CV_8UC3);
}

DataBridge::~DataBridge() = default;

void DataBridge::updateFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!frame.empty()) {
        frame.copyTo(latest_frame_);
        // 为显示准备缩放后的帧
        cv::resize(frame, display_frame_, cv::Size(640, 480));
        new_frame_available_.store(true);
        frame_cv_.notify_one();
    }
}

bool DataBridge::getLatestFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!latest_frame_.empty()) {
        latest_frame_.copyTo(frame);
        return true;
    }
    return false;
}

bool DataBridge::getDisplayFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!display_frame_.empty()) {
        display_frame_.copyTo(frame);
        return true;
    }
    return false;
}

void DataBridge::updateDetection(const DetectionResult& result) {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    detection_data_ = result;
    detection_data_.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    new_detection_available_.store(true);
    detection_cv_.notify_one();
}

bool DataBridge::getDetectionResult(DetectionResult& result) {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    if (detection_data_.valid) {
        result = detection_data_;
        return true;
    }
    return false;
}

void DataBridge::updateStats(const SystemStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = stats;
}

SystemStats DataBridge::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void DataBridge::updateModbusRegisters(const ModbusRegisters& registers) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_ = registers;
}

ModbusRegisters DataBridge::getModbusRegisters() const {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    return modbus_registers_;
}

void DataBridge::setSystemRunning(bool running) {
    system_running_.store(running);
    
    // 更新Modbus寄存器
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.system_status = running ? 1 : 0;
}

bool DataBridge::isSystemRunning() const {
    return system_running_.load();
}

void DataBridge::setEmergencyStop(bool stop) {
    emergency_stop_.store(stop);
    
    if (stop) {
        setSystemRunning(false);
        std::lock_guard<std::mutex> lock(modbus_mutex_);
        modbus_registers_.plc_command = 6; // 紧急停止命令
    }
}

bool DataBridge::isEmergencyStop() const {
    return emergency_stop_.load();
}

void DataBridge::setCurrentStep(int step) {
    current_step_.store(step);
    
    // 更新PLC命令基于当前步骤
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    switch (step) {
        case 1: // Feed Detection
            modbus_registers_.plc_command = 1;
            break;
        case 2: // Vision Recognition
            modbus_registers_.plc_command = 0;
            break;
        case 3: // Coordinate Transfer
            modbus_registers_.plc_command = 0;
            break;
        case 4: // Cut Prepare
            modbus_registers_.plc_command = 2;
            break;
        case 5: // Execute Cut
            modbus_registers_.plc_command = 3;
            break;
        default:
            modbus_registers_.plc_command = 0;
            break;
    }
}

int DataBridge::getCurrentStep() const {
    return current_step_.load();
}

void DataBridge::updateHeartbeat() {
    uint32_t current = heartbeat_counter_.fetch_add(1);
    
    // 防止溢出，重置为0
    if (current > 4294967290U) {
        heartbeat_counter_.store(0);
    }
    
    // 更新Modbus寄存器
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.heartbeat = heartbeat_counter_.load();
}

uint32_t DataBridge::getHeartbeat() const {
    return heartbeat_counter_.load();
}

} // namespace core
} // namespace bamboo_cut